// AudioWorklet processor: forwards mono Float32 frames to the main thread.
// Loaded via audioWorklet.addModule('/pcm-worklet.js').
class PcmForwarder extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input[0] && input[0].length > 0) {
      // Copy the channel data — the underlying buffer is reused by the engine.
      this.port.postMessage(input[0].slice(0));
    }
    return true;
  }
}

registerProcessor('pcm-forwarder', PcmForwarder);
