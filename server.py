import asyncio
import base64
import io
import os
import tempfile

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np
import soundfile as sf
import openai

from generator import load_csm_1b

app = FastAPI()

generator = None
openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chat_history = []

@app.on_event("startup")
def startup_event():
    global generator
    generator = load_csm_1b()


async def transcribe_audio(audio_bytes: bytes) -> str:
    """Send audio to gpt-4o-mini-transcribe and return the transcript."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        response = await openai_client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=open(tmp.name, "rb"),
            response_format="text",
        )
    return response


async def get_chat_response(history) -> str:
    """Get assistant response from gpt-4o-mini using the chat history."""
    completion = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history,
    )
    return completion.choices[0].message.content

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            audio_b64 = data.get("audio")

            if not audio_b64:
                await websocket.send_json({"error": "Missing audio"})
                continue

            audio_bytes = base64.b64decode(audio_b64)

            # Get transcript using gpt-4o-mini-transcribe
            transcript = await transcribe_audio(audio_bytes)

            # Update history and get assistant response
            chat_history.append({"role": "user", "content": transcript})
            if len(chat_history) > 20:
                chat_history.pop(0)
            response_text = await get_chat_response(chat_history)
            chat_history.append({"role": "assistant", "content": response_text})
            if len(chat_history) > 20:
                chat_history.pop(0)
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
