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

	// Reader "peek" jobs: every peekEvery bytes of new audio (~0.33 s), the
	// trailing peekWindow (~2.5 s) is transcribed (denoise off, low latency)
	// solely to update the scrolling reader position. Peeks run in their own
	// worker so they never queue behind scored chunks, and stale peeks are
	// dropped. wav2vec2 needs a multi-second window for accurate phonemes;
	// this cadence is near the physical floor (a word can't be recognized
	// before it has been said).
	peekEvery  = bytesPerSecond / 3
	peekWindow = bytesPerSecond * 5 / 2

	maxWords = 200

	// readLimit caps any single WebSocket message (audio frames are small;
	// text frames are tiny). 1 MiB is far above any legitimate frame.
	readLimit = 1 << 20

	// jobQueueDepth buffers pending transcribe jobs per connection.
	jobQueueDepth = 8

	// Idle handling: the read deadline ticks every idleTick of silence. Each
	// tick fires a catch-up peek (the reader may lag at fast speech; once
	// audio stops, no new peeks would fire otherwise). If the reader is
	// within nearEndWords of the passage end, finishSilence of silence
	// auto-finishes even without a last-word detection.
	idleTick      = time.Second
	finishSilence = 2500 * time.Millisecond
	nearEndWords  = 3
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

// job is one unit of scored inference work: a chunk appended to the
// cumulative transcript. Peeks travel on a separate channel (see peekJob).
type job struct {
	pcm []byte
}

// peekJob is a reader-position update: a trailing audio window. whole marks
// peeks whose window covers the entire recording so far (recording shorter
// than peekWindow): those use prefix alignment, since the window IS the full
// speech — local window alignment would let the first word's few phonemes
// float to any cheap match in the band.
type peekJob struct {
	pcm   []byte
	whole bool
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
	s.lastIdx.Store(-1)
	s.run()
}

type session struct {
	conn   *websocket.Conn
	infer  *inference.Client
	ctx    context.Context
	cancel context.CancelFunc
	log    *slog.Logger

	writeMu sync.Mutex

	// lastIdx is the reader position (last matched word), shared by the
	// chunk and peek workers; only ever advances.
	lastIdx atomic.Int32
	// lastAudioAt is the last time an audio frame arrived (unixnano),
	// read by the idle watcher.
	lastAudioAt atomic.Int64

	wordCount  int
	finishOnce sync.Once
	autoStop   atomic.Bool
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
	s.wordCount = len(strings.Fields(text))
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

	// Chunk jobs are serialized through one worker goroutine so chunks
	// transcribe in order; peeks run in a second worker so reader updates
	// never queue behind them. Audio intake never blocks on inference.
	jobs := make(chan job, jobQueueDepth)
	peeks := make(chan peekJob, 2) // small: stale peeks are dropped, not queued
	var wg sync.WaitGroup
	wg.Add(2)
	go s.chunkWorker(jobs, &wg, text)
	go s.peekWorker(peeks, &wg, text)

	buf := make([]byte, 0, chunkBytes*2)
	chunkStart := 0
	peekStart := 0
	var bufMu sync.Mutex // read loop appends; idle watcher copies windows

	// lastAudioAt drives the idle watcher (gorilla poisons a conn after ANY
	// read error — even a deadline timeout — so idle detection can't live on
	// the read deadline; it's a separate timer goroutine instead).
	s.lastAudioAt.Store(time.Now().UnixNano())
	idleDone := make(chan struct{})
	idleStopped := make(chan struct{})
	go s.idleWatcher(idleDone, idleStopped, &bufMu, &buf, peeks)

readLoop:
	for {
		mt, data, err := s.conn.ReadMessage()
		if err != nil {
			if s.autoStop.Load() {
				// autoFinish tripped the read deadline: finish exactly as if
				// the client had sent "stop".
				s.log.Info("auto-finished", "lastIdx", s.lastIdx.Load())
				break readLoop
			}
			if s.ctx.Err() == nil {
				s.log.Info("client disconnected", "err", err)
			}
			s.cancel()
			break readLoop
		}
		switch mt {
		case websocket.BinaryMessage:
			s.lastAudioAt.Store(time.Now().UnixNano())
			bufMu.Lock()
			if len(buf)+len(data) > maxAudioBytes {
				bufMu.Unlock()
				s.fail("maximum recording length exceeded (~3 minutes)")
				break readLoop
			}
			buf = append(buf, data...)
			if len(buf)-chunkStart >= chunkBytes {
				tail := append([]byte(nil), buf[chunkStart:]...)
				chunkStart = len(buf)
				peekStart = len(buf)
				bufMu.Unlock()
				if !s.enqueue(jobs, job{pcm: tail}) {
					break readLoop
				}
			} else if len(buf)-peekStart >= peekEvery {
				// Reader peek: trailing window only, position update only.
				// Dropped (not queued) if the peek worker is behind — a stale
				// window is worthless, the next one lands in ~0.33 s.
				start := len(buf) - peekWindow
				whole := start <= 0
				if start < 0 {
					start = 0
				}
				cp := make([]byte, len(buf)-start)
				copy(cp, buf[start:])
				peekStart = len(buf)
				bufMu.Unlock()
				select {
				case peeks <- peekJob{pcm: cp, whole: whole}:
				default:
				}
			} else {
				bufMu.Unlock()
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

	// Stop the idle watcher before closing peeks (it may be about to send).
	close(idleDone)
	<-idleStopped

	if s.ctx.Err() != nil {
		// Session was cancelled (disconnect or fatal error); workers exit
		// on their own once in-flight requests observe the cancelled ctx.
		close(jobs)
		close(peeks)
		wg.Wait()
		return
	}

	// Enqueue the remaining tail, then wait for all partial jobs to finish
	// before the final full-utterance pass.
	if len(buf) > chunkStart {
		if !s.enqueue(jobs, job{pcm: buf[chunkStart:]}) {
			close(jobs)
			close(peeks)
			wg.Wait()
			return
		}
	}
	close(jobs)
	close(peeks)
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

// idleWatcher fires catch-up peeks when audio stops arriving, so the reader
// can catch up to fast speech after it ends (peeks normally only fire on new
// audio). Near the passage end, finishSilence of silence auto-finishes even
// without a confident last-word detection. (Gorilla poisons a connection
// after any read error — even a deadline timeout — so idle detection is this
// timer goroutine, not a read deadline.)
func (s *session) idleWatcher(done <-chan struct{}, stopped chan<- struct{}, bufMu *sync.Mutex, buf *[]byte, peeks chan<- peekJob) {
	defer close(stopped)
	tick := time.NewTicker(250 * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case <-done:
			return
		case <-s.ctx.Done():
			return
		case <-tick.C:
		}
		idle := time.Since(time.Unix(0, s.lastAudioAt.Load()))
		if idle < idleTick {
			continue
		}
		if s.lastIdx.Load() < int32(s.wordCount-1) {
			bufMu.Lock()
			if len(*buf) > 0 {
				start := len(*buf) - peekWindow
				whole := start <= 0
				if start < 0 {
					start = 0
				}
				cp := make([]byte, len(*buf)-start)
				copy(cp, (*buf)[start:])
				select {
				case peeks <- peekJob{pcm: cp, whole: whole}:
				default:
				}
			}
			bufMu.Unlock()
		}
		if idle >= finishSilence &&
			s.lastIdx.Load() >= int32(s.wordCount-nearEndWords) {
			// Near the end and speech stopped: finish even if the last word
			// was never confidently detected.
			s.finishOnce.Do(func() { go s.autoFinish(0) })
		}
	}
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

// advanceIdx moves the reader position forward (never backward); reports
// whether it moved. Reaching the last word arms auto-finish.
func (s *session) advanceIdx(idx int) bool {
	for {
		cur := s.lastIdx.Load()
		if int32(idx) <= cur {
			return false
		}
		if s.lastIdx.CompareAndSwap(cur, int32(idx)) {
			if idx == s.wordCount-1 {
				s.finishOnce.Do(func() { go s.autoFinish(700 * time.Millisecond) })
			}
			return true
		}
	}
}

// autoFinish ends the session after the last word is detected (or, with
// grace 0, when speech stops near the passage end). The final word is already
// in the buffer (that's how it was detected); the grace period captures its
// tail. We notify the client, then interrupt the read loop via the read
// deadline (a blocked ReadMessage can't otherwise be interrupted without
// closing the conn, which we still need for "final").
func (s *session) autoFinish(grace time.Duration) {
	if grace > 0 {
		select {
		case <-time.After(grace):
		case <-s.ctx.Done():
			return
		}
	}
	s.log.Info("finishing session", "grace", grace)
	if err := s.writeJSON(map[string]string{"type": "finishing"}); err != nil {
		return
	}
	s.autoStop.Store(true)
	_ = s.conn.SetReadDeadline(time.Now())
}

// chunkWorker transcribes scored chunks in order, appends to the cumulative
// transcript, and emits partial frames. Transcription failure is fatal.
func (s *session) chunkWorker(jobs <-chan job, wg *sync.WaitGroup, text string) {
	defer wg.Done()
	processed := 0
	var cumulative strings.Builder
	for j := range jobs {
		ipa, err := s.infer.Transcribe(s.ctx, j.pcm)
		if err != nil {
			if s.ctx.Err() != nil {
				return // session already torn down
			}
			s.log.Error("chunk transcription failed", "err", err)
			s.fail("transcription failed") // protocol: error is fatal
			return
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
			} else if s.advanceIdx(idx) {
				msg.WordIndex = &idx
			}
		}
		err = s.writeJSON(msg)
		if err != nil {
			return
		}
	}
}

// peekWorker transcribes trailing-window peeks (denoise off, low latency)
// and emits progress frames when the reader position advances. Peek failures
// are non-fatal — the reader just updates on the next one.
func (s *session) peekWorker(peeks <-chan peekJob, wg *sync.WaitGroup, text string) {
	defer wg.Done()
	for p := range peeks {
		ipa, err := s.infer.TranscribeFast(s.ctx, p.pcm)
		if err != nil {
			if s.ctx.Err() != nil {
				return // session already torn down
			}
			s.log.Warn("peek transcription failed", "err", err)
			continue
		}
		if ipa == "" {
			continue
		}
		// The window contains only the last few words spoken, so align it
		// LOCALLY against a band around the current position — except at the
		// very start, where the window is the whole recording so far and
		// prefix fitting is exactly right (and can't float forward).
		var idx int
		if p.whole {
			idx, err = s.infer.Align(s.ctx, text, ipa)
		} else {
			// A reader realistically skips at most 1-2 words between updates
			// (peeks fire every ~0.33 s); cap at +3 so a duplicate word later
			// in the passage can never teleport the strip.
			cur := s.lastIdx.Load()
			from := int(max(0, cur-3))
			idx, err = s.infer.AlignWindow(s.ctx, text, ipa, from, int(cur)+3)
		}
		if err != nil {
			s.log.Warn("peek align failed", "err", err)
			continue
		}
		if s.advanceIdx(idx) {
			if err := s.writeJSON(progressMessage{Type: "progress", WordIndex: idx}); err != nil {
				return
			}
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
