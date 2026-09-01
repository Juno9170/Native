// Package session implements the WebSocket side of the wire protocol
// (see PROTOCOL.md): one connection = one pronunciation-scoring session.
package session

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"

	"native/backend/internal/inference"
)

const (
	// Audio format: pcm16 s16le, 16 kHz, mono => 32000 bytes per second.
	bytesPerSecond = 16000 * 2

	// maxAudioBytes caps one recording at ~3 minutes.
	maxAudioBytes = bytesPerSecond * 180

	// chunkBytes is the unit streamed to /transcribe while recording: ~2.5 s.
	chunkBytes = bytesPerSecond * 5 / 2

	// Reader "peek" jobs: every peekEvery bytes of new audio (~0.5 s), the
	// trailing peekWindow (~3 s) is transcribed solely to update the
	// scrolling reader position. Peeks are not part of the scored transcript.
	// wav2vec2 needs a multi-second window for accurate phonemes; 0.5 s
	// cadence is near the physical floor (a word can't be recognized before
	// it has been said).
	peekEvery  = bytesPerSecond / 2
	peekWindow = bytesPerSecond * 3

	maxWords = 200

	// readLimit caps any single WebSocket message (audio frames are small;
	// text frames are tiny). 1 MiB is far above any legitimate frame.
	readLimit = 1 << 20

	// jobQueueDepth buffers pending transcribe jobs per connection.
	jobQueueDepth = 8
)

var nextID atomic.Uint64

type clientMessage struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type partialMessage struct {
	Type         string  `json:"type"`
	IPA          string  `json:"ipa"`
	ProcessedSec float64 `json:"processedSec"`
	// WordIndex is the last expected word matched by the speech so far
	// (drives the scrolling reader); omitted when alignment is unavailable.
	WordIndex *int `json:"wordIndex,omitempty"`
}

type progressMessage struct {
	Type      string `json:"type"`
	WordIndex int    `json:"wordIndex"`
}

// job is one unit of inference work: a scored chunk (appended to the
// cumulative transcript) or a peek (trailing window, reader position only).
type job struct {
	pcm  []byte
	peek bool
}

type finalMessage struct {
	Type        string                 `json:"type"`
	IPA         string                 `json:"ipa"`
	ExpectedIPA string                 `json:"expectedIpa"`
	Score       float64                `json:"score"`
	Words       []inference.WordResult `json:"words"`
}

type errorMessage struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// Handler upgrades /ws requests and runs one session per connection.
type Handler struct {
	infer    *inference.Client
	upgrader websocket.Upgrader
}

// NewHandler builds the /ws handler. CheckOrigin allows all origins (v1).
func NewHandler(infer *inference.Client) *Handler {
	return &Handler{
		infer: infer,
		upgrader: websocket.Upgrader{
			ReadBufferSize:  64 * 1024,
			WriteBufferSize: 16 * 1024,
			CheckOrigin:     func(r *http.Request) bool { return true },
		},
	}
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	conn, err := h.upgrader.Upgrade(w, r, nil)
	if err != nil {
		// Upgrade already wrote the error response.
		return
	}
	ctx, cancel := context.WithCancel(r.Context())
	id := nextID.Add(1)
	s := &session{
		conn:   conn,
		infer:  h.infer,
		ctx:    ctx,
		cancel: cancel,
		log:    slog.With("session", id, "remote", r.RemoteAddr),
	}
	s.run()
}

type session struct {
	conn   *websocket.Conn
	infer  *inference.Client
	ctx    context.Context
	cancel context.CancelFunc
	log    *slog.Logger

	writeMu sync.Mutex
}

func (s *session) run() {
	defer s.cancel()
	defer s.conn.Close()
	s.conn.SetReadLimit(readLimit)
	s.log.Info("session opened")

	text, ok := s.awaitStart()
	if !ok {
		return
	}
	s.log.Info("session started", "words", len(strings.Fields(text)))
	if err := s.writeJSON(map[string]string{"type": "ready"}); err != nil {
		return
	}

	// Warm the inference text cache (espeak phonemization of the passage) so
	// the first /align during recording doesn't stall for seconds on long
	// passages. An empty IPA populates the cache and returns immediately.
	go func() {
		if _, err := s.infer.Align(s.ctx, text, ""); err != nil {
			s.log.Warn("cache warm-up failed", "err", err)
		}
	}()

	// Inference jobs are serialized through one worker goroutine so chunks
	// transcribe in order, while audio intake never blocks on inference.
	jobs := make(chan job, jobQueueDepth)
	var wg sync.WaitGroup
	wg.Add(1)
	go s.worker(jobs, &wg, text)

	buf := make([]byte, 0, chunkBytes*2)
	chunkStart := 0
	peekStart := 0

readLoop:
	for {
		mt, data, err := s.conn.ReadMessage()
		if err != nil {
			if s.ctx.Err() == nil {
				s.log.Info("client disconnected", "err", err)
			}
			s.cancel()
			break readLoop
		}
		switch mt {
		case websocket.BinaryMessage:
			if len(buf)+len(data) > maxAudioBytes {
				s.fail("maximum recording length exceeded (~3 minutes)")
				break readLoop
			}
			buf = append(buf, data...)
			if len(buf)-chunkStart >= chunkBytes {
				if !s.enqueue(jobs, job{pcm: buf[chunkStart:]}) {
					break readLoop
				}
				chunkStart = len(buf)
				peekStart = len(buf)
			} else if len(buf)-peekStart >= peekEvery {
				// Reader peek: trailing window only, position update only.
				start := len(buf) - peekWindow
				if start < 0 {
					start = 0
				}
				if !s.enqueue(jobs, job{pcm: buf[start:], peek: true}) {
					break readLoop
				}
				peekStart = len(buf)
			}
		case websocket.TextMessage:
			var m clientMessage
			if err := json.Unmarshal(data, &m); err != nil || m.Type != "stop" {
				s.fail("unexpected message during recording")
				break readLoop
			}
			s.log.Info("recording stopped", "audioSec", float64(len(buf))/bytesPerSecond)
			break readLoop
		}
	}

	if s.ctx.Err() != nil {
		// Session was cancelled (disconnect or fatal error); worker exits
		// on its own once its in-flight request observes the cancelled ctx.
		close(jobs)
		wg.Wait()
		return
	}

	// Enqueue the remaining tail, then wait for all partial jobs to finish
	// before the final full-utterance pass.
	if len(buf) > chunkStart {
		if !s.enqueue(jobs, job{pcm: buf[chunkStart:]}) {
			close(jobs)
			wg.Wait()
			return
		}
	}
	close(jobs)
	wg.Wait()
	if s.ctx.Err() != nil {
		return
	}

	fullIPA, err := s.infer.Transcribe(s.ctx, buf)
	if err != nil {
		s.log.Error("full transcription failed", "err", err)
		s.fail("transcription failed")
		return
	}
	score, err := s.infer.Score(s.ctx, text, fullIPA)
	if err != nil {
		s.log.Error("scoring failed", "err", err)
		s.fail("scoring failed")
		return
	}
	buf = nil // discard the audio buffer

	if err := s.writeJSON(finalMessage{
		Type:        "final",
		IPA:         fullIPA,
		ExpectedIPA: score.ExpectedIPA,
		Score:       score.Score,
		Words:       score.Words,
	}); err != nil {
		return
	}
	s.log.Info("session complete", "score", score.Score)
	s.closePolitely(websocket.CloseNormalClosure, "done")
}

// awaitStart reads until a valid {"type":"start","text":...} arrives.
func (s *session) awaitStart() (string, bool) {
	for {
		mt, data, err := s.conn.ReadMessage()
		if err != nil {
			return "", false
		}
		if mt != websocket.TextMessage {
			s.fail("expected start message")
			return "", false
		}
		var m clientMessage
		if err := json.Unmarshal(data, &m); err != nil || m.Type != "start" {
			s.fail("expected start message")
			return "", false
		}
		text := strings.TrimSpace(m.Text)
		if text == "" {
			s.fail("text must not be empty")
			return "", false
		}
		if n := len(strings.Fields(text)); n > maxWords {
			s.fail(fmt.Sprintf("text too long: %d words (max %d)", n, maxWords))
			return "", false
		}
		return text, true
	}
}

// worker processes inference jobs in order. Chunks append to the cumulative
// transcript and emit partial frames; peeks (trailing windows) only update
// the reader position via progress frames. Peek failures are non-fatal —
// the reader just updates on the next one.
func (s *session) worker(jobs <-chan job, wg *sync.WaitGroup, text string) {
	defer wg.Done()
	processed := 0
	var cumulative strings.Builder
	lastIdx := -1 // reader position; never moves backward
	for j := range jobs {
		ipa, err := s.infer.Transcribe(s.ctx, j.pcm)
		if err != nil {
			if s.ctx.Err() != nil {
				return // session already torn down
			}
			if j.peek {
				s.log.Warn("peek transcription failed", "err", err)
				continue
			}
			s.log.Error("chunk transcription failed", "err", err)
			s.fail("transcription failed") // protocol: error is fatal
			return
		}
		if j.peek {
			if ipa == "" {
				continue
			}
			// The window contains only the last few words spoken, so align it
			// LOCALLY against a band around the current position — appending
			// it to the cumulative transcript would duplicate the overlap and
			// the lenient substitution costs would walk the reader forward
			// into unsaid words.
			from := max(0, lastIdx-3)
			idx, err := s.infer.AlignWindow(s.ctx, text, ipa, from)
			if err != nil {
				s.log.Warn("window align failed", "err", err)
				continue
			}
			if idx > lastIdx {
				lastIdx = idx
				if err := s.writeJSON(progressMessage{Type: "progress", WordIndex: idx}); err != nil {
					return
				}
			}
			continue
		}
		processed += len(j.pcm)
		msg := partialMessage{
			Type:         "partial",
			IPA:          ipa,
			ProcessedSec: float64(processed) / bytesPerSecond,
		}
		// Reader progress: align the cumulative transcription against the
		// passage. Non-fatal — a failed align just omits wordIndex.
		if ipa != "" {
			if cumulative.Len() > 0 {
				cumulative.WriteString(" ")
			}
			cumulative.WriteString(ipa)
			if idx, err := s.infer.Align(s.ctx, text, cumulative.String()); err != nil {
				s.log.Warn("align failed", "err", err)
			} else if idx > lastIdx {
				// Monotonic everywhere: letting chunks pull the position back
				// made the strip visibly ping-pong against peeks.
				lastIdx = idx
				msg.WordIndex = &idx
			}
		}
		err = s.writeJSON(msg)
		if err != nil {
			return
		}
	}
}

// enqueue hands a copy of the job's audio to the worker; returns false if
// the session was cancelled while the queue was full.
func (s *session) enqueue(jobs chan<- job, j job) bool {
	cp := make([]byte, len(j.pcm))
	copy(cp, j.pcm)
	j.pcm = cp
	select {
	case jobs <- j:
		return true
	case <-s.ctx.Done():
		return false
	}
}

// fail sends a fatal error frame, cancels the session, and closes the conn.
// Closing the conn is what unblocks a ReadMessage blocked in the other
// goroutine; the deferred Close in run() afterwards is a harmless no-op.
func (s *session) fail(msg string) {
	s.log.Warn("session failed", "reason", msg)
	_ = s.writeJSON(errorMessage{Type: "error", Message: msg})
	s.closePolitely(websocket.ClosePolicyViolation, msg)
	s.cancel()
	_ = s.conn.Close()
}

func (s *session) writeJSON(v any) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.WriteJSON(v)
}

func (s *session) closePolitely(code int, msg string) {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	frame := websocket.FormatCloseMessage(code, msg)
	_ = s.conn.WriteControl(websocket.CloseMessage, frame, time.Now().Add(2*time.Second))
}
