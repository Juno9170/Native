// Package inference is a thin HTTP client for the Python inference service.
package inference

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Health mirrors GET /health on the inference service.
type Health struct {
	Status string `json:"status"`
	Device string `json:"device"`
	Model  string `json:"model"`
}

// WordResult is one aligned word in a /score response.
type WordResult struct {
	Word     string  `json:"word"`
	Expected string  `json:"expected"`
	IPA      string  `json:"ipa"`
	Score    float64 `json:"score"`
}

// ScoreResponse mirrors POST /score on the inference service.
type ScoreResponse struct {
	ExpectedIPA string       `json:"expectedIpa"`
	Score       float64      `json:"score"`
	Words       []WordResult `json:"words"`
}

// Client calls the inference service over HTTP.
type Client struct {
	base string
	http *http.Client
}

// New builds a Client for the given base URL (e.g. http://localhost:9000).
// The timeout is generous: the first /transcribe after boot may hit a cold model.
func New(base string) *Client {
	return &Client{
		base: strings.TrimSuffix(base, "/"),
		http: &http.Client{Timeout: 60 * time.Second},
	}
}

// Health calls GET /health. The caller controls the deadline via ctx.
func (c *Client) Health(ctx context.Context) (*Health, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base+"/health", nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("inference /health returned %s", resp.Status)
	}
	var h Health
	if err := json.NewDecoder(resp.Body).Decode(&h); err != nil {
		return nil, fmt.Errorf("decoding /health response: %w", err)
	}
	return &h, nil
}

// Transcribe posts raw pcm16 bytes to /transcribe and returns the IPA string.
func (c *Client) Transcribe(ctx context.Context, pcm []byte) (string, error) {
	return c.transcribe(ctx, c.base+"/transcribe", pcm)
}

// TranscribeFast is Transcribe with denoising disabled — for reader peeks,
// where latency matters more than noise robustness.
func (c *Client) TranscribeFast(ctx context.Context, pcm []byte) (string, error) {
	return c.transcribe(ctx, c.base+"/transcribe?denoise=0", pcm)
}

func (c *Client) transcribe(ctx context.Context, url string, pcm []byte) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(pcm))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/octet-stream")

	resp, err := c.http.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return "", fmt.Errorf("inference /transcribe returned %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}
	var out struct {
		IPA string `json:"ipa"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", fmt.Errorf("decoding /transcribe response: %w", err)
	}
	return out.IPA, nil
}

// Score posts the reference text and the actual IPA to /score.
func (c *Client) Score(ctx context.Context, text, actualIPA string) (*ScoreResponse, error) {
	payload, err := json.Marshal(map[string]string{
		"text":      text,
		"actualIpa": actualIPA,
	})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.base+"/score", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("inference /score returned %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}
	var out ScoreResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("decoding /score response: %w", err)
	}
	return &out, nil
}

// Align posts the reference text and IPA to /align (mode "prefix": actualIPA
// is the cumulative transcript) and returns the last matched word index
// (-1 = nothing matched yet).
func (c *Client) Align(ctx context.Context, text, actualIPA string) (int, error) {
	return c.align(ctx, map[string]any{
		"text":      text,
		"actualIpa": actualIPA,
		"mode":      "prefix",
	})
}

// AlignWindow is mode "window": actualIPA is a short trailing window, aligned
// locally against the band of words starting at fromWord, never answering
// beyond maxWord (a further match is always a duplicate-word coincidence).
func (c *Client) AlignWindow(ctx context.Context, text, actualIPA string, fromWord, maxWord int) (int, error) {
	return c.align(ctx, map[string]any{
		"text":      text,
		"actualIpa": actualIPA,
		"mode":      "window",
		"fromWord":  fromWord,
		"maxWord":   maxWord,
	})
}

func (c *Client) align(ctx context.Context, payload map[string]any) (int, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return -1, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.base+"/align", bytes.NewReader(body))
	if err != nil {
		return -1, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return -1, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return -1, fmt.Errorf("inference /align returned %s: %s", resp.Status, strings.TrimSpace(string(b)))
	}
	var out struct {
		WordIndex int `json:"wordIndex"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return -1, fmt.Errorf("decoding /align response: %w", err)
	}
	return out.WordIndex, nil
}
