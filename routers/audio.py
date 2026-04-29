# routers/audio.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
from apis.transcription import process_audio_transcription, UnifiedTranscriptionResponse

router = APIRouter()

@router.post("/transcribe_audio", response_model=UnifiedTranscriptionResponse)
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    response_format: str = Form("verbose_json"),
    timestamp_granularities: str = Form("word"),
    language_code: Optional[str] = Form(None),
    mode: str = Form("transcribe"),
):
    try:
        audio_content = await audio_file.read()
        
        # The transcription service handles the routing and schema normalization
        result = process_audio_transcription(
            audio_content=audio_content,
            filename=audio_file.filename,
            model=model,
            mode=mode,
            language_code=language_code,
            response_format=response_format,
            timestamp_granularities=timestamp_granularities
        )

        return result

    except ValueError as ve:
        # Catch specific validation/configuration errors from our service
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))