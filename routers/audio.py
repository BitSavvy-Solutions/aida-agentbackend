import os
import json
import io
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from openai import OpenAI
from typing import Optional
from sarvamai import SarvamAI


router = APIRouter()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@router.post("/transcribe_audio")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    response_format: str = Form("verbose_json"),
    timestamp_granularities: str = Form("word"),
    language_code: Optional[str] = Form(None),
    mode: str = Form("transcribe"),  # For Sarvam models: transcribe, translate, verbatim, translit, codemix
):
    try:
        # Read file content
        audio_content = await audio_file.read()
        
        # Create a file-like object with the filename
        audio_io = io.BytesIO(audio_content)
        audio_io.name = audio_file.filename

        # Determine which service to use based on the model
        if model.startswith("saaras:") or model.startswith("saarika:"):
                
            api_key = os.getenv("SARVAM_API_KEY")
            if not api_key:
                raise HTTPException(status_code=500, detail="SARVAM_API_KEY not configured")
                
            client = SarvamAI(api_subscription_key=api_key)
            
            # Prepare parameters
            sarvam_params = {
                "file": audio_io,
                "model": model,
                "mode": mode,
            }
            
            # Add language code if provided
            if language_code:
                sarvam_params["language_code"] = language_code
            
            # Use translate or transcribe based on mode
            if mode == "translate":
                response = client.speech_to_text.translate(**sarvam_params)
            else:
                response = client.speech_to_text.transcribe(**sarvam_params)
            
            # Convert to dictionary if it's not already
            if not isinstance(response, dict):
                response = response.__dict__
                
            return response
        else:
            # Use OpenAI
            transcription = openai_client.audio.transcriptions.create(
                file=audio_io,
                model=model,
                response_format=response_format,
                timestamp_granularities=timestamp_granularities.split(',')
            )

            return transcription.model_dump()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))