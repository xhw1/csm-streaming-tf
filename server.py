import asyncio
import base64
import io
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np
import soundfile as sf

from generator import load_csm_1b

app = FastAPI()

generator = None

@app.on_event("startup")
def startup_event():
    global generator
    generator = load_csm_1b()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            audio_b64 = data.get("audio")
            transcript = data.get("transcript")
            response_text = data.get("response_text")

            if not all([audio_b64, transcript, response_text]):
                await websocket.send_json({"error": "Missing fields"})
                continue

            audio_bytes = base64.b64decode(audio_b64)
            audio_array, sr = sf.read(io.BytesIO(audio_bytes))
            if sr != 24000:
                import librosa
                audio_array = librosa.resample(audio_array, sr, 24000)
            if audio_array.ndim > 1:
                audio_array = np.mean(audio_array, axis=1)
            conversation = [
                {
                    "role": "0",
                    "content": [
                        {"type": "text", "text": transcript},
                        {"type": "audio", "audio": audio_array.astype(np.float32)},
                    ],
                },
                {
                    "role": "0",
                    "content": [
                        {"type": "text", "text": response_text},
                    ],
                },
            ]
            inputs = generator.processor.apply_chat_template(
                conversation, tokenize=True, return_dict=True
            ).to(generator.device)

            gen_iter = generator.generate_stream(inputs, chunk_token_size=20)
            while True:
                chunk = await asyncio.to_thread(next, gen_iter, None)
                if chunk is None:
                    break
                np_chunk = chunk.cpu().numpy().astype(np.float32)
                buf = io.BytesIO()
                sf.write(buf, np_chunk, 24000, format="WAV")
                chunk_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                await websocket.send_json({"audio_chunk": chunk_b64})
            await websocket.send_json({"status": "complete"})
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        await websocket.send_json({"error": str(e)})
