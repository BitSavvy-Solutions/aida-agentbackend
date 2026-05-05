# apis/transcription.py
import os
import io
import json
import tempfile
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
def _handle_openai(audio_io: io.BytesIO, model: str, mode: str, response_format: str, timestamp_granularities: str) -> UnifiedTranscriptionResponse:
    if mode == "translate":
        # Use OpenAI's Translation endpoint
        response = openai_client.audio.translations.create(
            file=audio_io,
            model=model,
            response_format=response_format
            # Note: OpenAI's translation endpoint does NOT support timestamp_granularities
        )
    else:
        # Use OpenAI's Transcription endpoint
        response = openai_client.audio.transcriptions.create(
            file=audio_io,
            model=model,
            response_format=response_format,
            timestamp_granularities=timestamp_granularities.split(',') if timestamp_granularities else []
        )
    
    raw_data = response.model_dump()
    
    # Map OpenAI response to Unified Schema
    return UnifiedTranscriptionResponse(
        text=raw_data.get("text", ""),
        language=raw_data.get("language", "english" if mode == "translate" else None),
        words=raw_data.get("words"), 
        duration=raw_data.get("duration"),
        provider="openai"
    )

def _handle_sarvam(audio_content: bytes, filename: str, model: str, mode: str, language_code: Optional[str]) -> UnifiedTranscriptionResponse:
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise ValueError("SARVAM_API_KEY not configured")
        
    client = SarvamAI(api_subscription_key=api_key)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Write the audio bytes to a temporary file for the SDK to upload
        input_path = os.path.join(temp_dir, filename)
        with open(input_path, "wb") as f:
            f.write(audio_content)
            
        sarvam_params = {
            "model": model,
            "mode": mode,
        }
        if language_code:
            sarvam_params["language_code"] = language_code
            
        # 2. Create and run the Batch Job
        job = client.speech_to_text_job.create_job(**sarvam_params)
        job.upload_files(file_paths=[input_path])
        job.start()
        
        # Wait for the job to finish (this handles >30s audio)
        job.wait_until_complete()
        
        file_results = job.get_file_results()
        if not file_results.get('successful'):
            failed = file_results.get('failed', [])
            err_msg = failed[0].get('error_message', 'Unknown error') if failed else "Unknown error"
            raise Exception(f"Sarvam Batch API failed: {err_msg}")
            
        # 3. Download and read the outputs
        output_dir = os.path.join(temp_dir, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        job.download_outputs(output_dir=output_dir)
        
        output_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
        if not output_files:
            raise Exception("No output files downloaded from Sarvam.")
            
        out_file_path = os.path.join(output_dir, output_files[0])
        with open(out_file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
        # 4. Map Sarvam's Batch timestamp format to our Unified Schema
        words_list = []
        timestamps = raw_data.get("timestamps")
        if timestamps and isinstance(timestamps, dict) and "words" in timestamps:
            w = timestamps.get("words", [])
            s = timestamps.get("start_time_seconds", [])
            e = timestamps.get("end_time_seconds", [])
            
            for i in range(len(w)):
                words_list.append(WordTimestamp(
                    word=w[i],
                    start=s[i] if i < len(s) else 0.0,
                    end=e[i] if i < len(e) else 0.0
                ))

        return UnifiedTranscriptionResponse(
            text=raw_data.get("transcript", ""),
            language=raw_data.get("language_code"),
            words=words_list if words_list else None, 
            duration=None, 
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
    
    # Route to the correct provider based on model name
    if model.startswith("saaras:") or model.startswith("saarika:"):
        # Pass raw bytes and filename to Sarvam handler for temp file creation
        return _handle_sarvam(audio_content, filename, model, mode, language_code)
    else:
        audio_io = io.BytesIO(audio_content)
        audio_io.name = filename
        return _handle_openai(audio_io, model, mode, response_format, timestamp_granularities)