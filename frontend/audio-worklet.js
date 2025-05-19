class AudioQueueProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = [];
    this.port.onmessage = (event) => {
      const chunk = event.data;
      if (chunk && chunk.length) {
        // Convert typed array to normal array for easier manipulation
        this.buffer.push(...chunk);
      }
    };
  }

  process(inputs, outputs) {
    const output = outputs[0];
    const channel = output[0];
    for (let i = 0; i < channel.length; i++) {
      channel[i] = this.buffer.length ? this.buffer.shift() : 0;
    }
    return true;
  }
}

registerProcessor('audio-queue-processor', AudioQueueProcessor);
