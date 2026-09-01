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

	// Inference jobs are serialized through one worker goroutine so chunks
	// transcribe in order, while audio intake never blocks on inference.
	jobs := make(chan []byte, jobQueueDepth)
	var wg sync.WaitGroup
	wg.Add(1)
	go s.worker(jobs, &wg)

	buf := make([]byte, 0, chunkBytes*2)
	chunkStart := 0

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
				if !s.enqueue(jobs, buf[chunkStart:]) {
					break readLoop
				}
				chunkStart = len(buf)
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
		if !s.enqueue(jobs, buf[chunkStart:]) {
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

// worker transcribes audio chunks in order and emits partial frames.
func (s *session) worker(jobs <-chan []byte, wg *sync.WaitGroup) {
	defer wg.Done()
	processed := 0
	for chunk := range jobs {
		ipa, err := s.infer.Transcribe(s.ctx, chunk)
		if err != nil {
			if s.ctx.Err() != nil {
				return // session already torn down
			}
			s.log.Error("chunk transcription failed", "err", err)
			s.fail("transcription failed") // protocol: error is fatal
			return
		}
		processed += len(chunk)
		err = s.writeJSON(partialMessage{
			Type:         "partial",
			IPA:          ipa,
			ProcessedSec: float64(processed) / bytesPerSecond,
		})
		if err != nil {
			return
		}
	}
}

// enqueue hands a copy of the chunk to the worker; returns false if the
// session was cancelled while the queue was full.
func (s *session) enqueue(jobs chan<- []byte, chunk []byte) bool {
	cp := make([]byte, len(chunk))
	copy(cp, chunk)
	select {
	case jobs <- cp:
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
