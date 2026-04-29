# apis/transcription.py
import os
import io
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from openai import OpenAI
from sarvamai import SarvamAI

# Initialize OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --- 1. Define the Standardized Response Schema ---
class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float

class UnifiedTranscriptionResponse(BaseModel):
    text: str
    language: Optional[str] = None
    words: Optional[List[WordTimestamp]] = None
    duration: Optional[float] = None
    provider: str

# --- 2. Provider-Specific Handlers ---
def _handle_openai(audio_io: io.BytesIO, model: str, response_format: str, timestamp_granularities: str) -> UnifiedTranscriptionResponse:
    transcription = openai_client.audio.transcriptions.create(
        file=audio_io,
        model=model,
        response_format=response_format,
        timestamp_granularities=timestamp_granularities.split(',') if timestamp_granularities else []
    )
    
    raw_data = transcription.model_dump()
    
    # Map OpenAI response to Unified Schema
    return UnifiedTranscriptionResponse(
        text=raw_data.get("text", ""),
        language=raw_data.get("language"),
        words=raw_data.get("words"), # OpenAI already returns a compatible list of dicts
        duration=raw_data.get("duration"),
        provider="openai"
    )

def _handle_sarvam(audio_io: io.BytesIO, model: str, mode: str, language_code: Optional[str]) -> UnifiedTranscriptionResponse:
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise ValueError("SARVAM_API_KEY not configured")
        
    client = SarvamAI(api_subscription_key=api_key)
    
    sarvam_params = {
        "file": audio_io,
        "model": model,
        "mode": mode,
    }
    if language_code:
        sarvam_params["language_code"] = language_code
    
    if mode == "translate":
        response = client.speech_to_text.translate(**sarvam_params)
    else:
        response = client.speech_to_text.transcribe(**sarvam_params)
    
    raw_data = response if isinstance(response, dict) else response.__dict__
    
    # Map Sarvam response to Unified Schema
    return UnifiedTranscriptionResponse(
        text=raw_data.get("transcript", ""),
        language=raw_data.get("language_code"),
        words=raw_data.get("timestamps"), # Maps to None based on Sarvam's current payload
        duration=None, # Sarvam doesn't provide duration in the root payload
        provider="sarvam"
    )

# --- 3. Main Entry Point ---
def process_audio_transcription(
    audio_content: bytes, 
    filename: str, 
    model: str, 
    mode: str, 
    language_code: Optional[str],
    response_format: str,
    timestamp_granularities: str
) -> UnifiedTranscriptionResponse:
    
    audio_io = io.BytesIO(audio_content)
    audio_io.name = filename

    # Route to the correct provider based on model name
    if model.startswith("saaras:") or model.startswith("saarika:"):
        return _handle_sarvam(audio_io, model, mode, language_code)
    else:
        return _handle_openai(audio_io, model, response_format, timestamp_granularities)