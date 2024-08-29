from io import BytesIO
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import asyncio
from pydub import AudioSegment
import subprocess
import os 
from pydub import AudioSegment
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
from peft import PeftConfig, PeftModel
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))



app = FastAPI()
model_id = "kawther1/whisperlargeev2"
language = "ar" 
task = "transcribe"

class Transcriber:
    def __init__(self, model_id, language, task):
     
        self.peft_config = PeftConfig.from_pretrained(model_id)
        self.base_model_name_or_path = self.peft_config.base_model_name_or_path
        self.model = WhisperForConditionalGeneration.from_pretrained(self.base_model_name_or_path)
        self.model = PeftModel.from_pretrained(self.model, model_id, language=language, task=task)
        self.processor = WhisperProcessor.from_pretrained(self.base_model_name_or_path, language=language, task=task)
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def transcribe(self, audio_data):
        # Prétraiter l'audio
        audio = AudioSegment.from_file(BytesIO(audio_data))
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio_array = audio.get_array_of_samples()
        
        # Préparer les entrées du modèle
        inputs = self.processor(audio_array, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        input_features = inputs.input_features.to(torch.float32)
        attention_mask = inputs.attention_mask

        # Générer les transcriptions
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                max_new_tokens= 448 - len(input_features[0]),
                do_sample=True
          )
        
        
        # Décoder les prédictions en texte
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription
transcriber = Transcriber(model_id, language, task)
audio_chunks = []

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="pic5.jpg")


@app.get("/", response_class=HTMLResponse)
def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_chunks.append(data)
    except WebSocketDisconnect:
        await websocket.close()
    finally:
       transcription = save_audio()
       print("Transcription:", transcription)

def save_audio():
    if audio_chunks:
        combined_audio = b''.join(audio_chunks)
        audio = AudioSegment.from_file(BytesIO(combined_audio), format="webm")
        audio.export("recorded_audio.wav", format="wav")
        audio_chunks.clear()
        with open("recorded_audio.wav", "rb") as f:
            audio_data = f.read()

        transcription = transcriber.transcribe(audio_data)
        return transcription
