import os
import json
import io
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from openai import OpenAI

router = APIRouter()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@router.post("/transcribe_audio")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    response_format: str = Form("verbose_json"),
    timestamp_granularities: str = Form("word")
):
    try:
        # Read file content
        audio_content = await audio_file.read()
        
        # Create a file-like object with the filename
        audio_io = io.BytesIO(audio_content)
        audio_io.name = audio_file.filename

        transcription = openai_client.audio.transcriptions.create(
            file=audio_io,
            model=model,
            response_format=response_format,
            timestamp_granularities=timestamp_granularities.split(',')
        )

        return transcription.model_dump()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))