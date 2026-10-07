import os
import json
import uuid
import re
import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from apis.chunk_enhancer import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from apis.credit_manager import queue_credit_deduction
from dependencies.auth import validate_api_token

router = APIRouter()


# Pydantic Model for Request Body
class ChatRequest(BaseModel):
    user_input: Optional[str] = None
    image_data_urls: List[str] = []
    pdf_attachments: List[str] = []  # base64 data URLs, e.g. "data:application/pdf;base64,..."
    model: str = 'google/gemini-flash-1.5'
    user_id: Optional[str] = None
    message_history: List[Dict[str, Any]] = []
    thread_id: Optional[str] = None


def build_human_content(text: Optional[str], image_urls: List[str], pdf_urls: List[str], pdf_offset: int = 0):
    # Plain string when there are no attachments, so text-only history stays unchanged
    if not image_urls and not pdf_urls:
        return text or ""

    content = []
    if text:
        content.append({"type": "text", "text": text})
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    for idx, pdf_data_url in enumerate(pdf_urls):
        content.append({
            "type": "file",
            "file": {
                "file_data": pdf_data_url,
                "filename": f"attachment_{pdf_offset + idx + 1}.pdf"
            }
        })
    return content


openrouter_key = os.getenv("OPENROUTER_API_KEY")
ALLOWED_ANONYMOUS_MODELS = [r"google/gemini-3.1-flash-lite-preview", r"^.*deepseek.*"]
COMPILED_ANONYMOUS_PATTERNS = [re.compile(p, re.IGNORECASE) for p in ALLOWED_ANONYMOUS_MODELS]


@router.post("/iverse_agent")
async def iverse_agent(req: Request, body: ChatRequest):
    thread_id = body.thread_id or str(uuid.uuid4())

    resolved_user_id: Optional[str] = None
    auth_method: str = "anonymous"

    # Determine if the requested model is free early on
    is_free_model = any(p.match(body.model) for p in COMPILED_ANONYMOUS_PATTERNS)

    # 1. Token Auth Resolution
    auth_header = req.headers.get("Authorization", "").strip()
    token_string = ""
    
    if auth_header.startswith("Bearer "):
        token_string = auth_header[7:].strip()

    if token_string:
        token_doc = await validate_api_token(token_string)

        if token_doc:
            resolved_user_id = token_doc.get("userId")
            auth_method = "token"
            
            # Extract scopes and check if blocked
            scopes = token_doc.get("scopes", [])
            if "aida:blocked" in scopes:
                raise HTTPException(
                    status_code=403,
                    detail="Your account has been suspended. Please contact support."
                )
        else:
            # Only raise an error for invalid tokens if the model requires sign in
            if not is_free_model:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid or expired token. Please sign in again."
                )

    # 2. Legacy Fallback
    # Only runs if no Bearer token was provided
    elif body.user_id:
        resolved_user_id = body.user_id
        auth_method = "legacy"

    # 3. Access Control
    # Logged in users bypass this check entirely.
    # Anonymous users are restricted to free models.
    if auth_method == "anonymous" and not is_free_model:
        raise HTTPException(
            status_code=403,
            detail=f"Model '{body.model}' requires sign in."
        )

    # Input Validation
    if not body.user_input and not body.image_data_urls and not body.pdf_attachments and not body.message_history:
        raise HTTPException(status_code=400, detail="Input required")

    # The current turn is the top-level fields when a client sends them, otherwise the last human history item
    has_top_level_turn = bool(body.user_input or body.image_data_urls or body.pdf_attachments)
    last_human_idx = max((i for i, m in enumerate(body.message_history) if m.get('type') == 'human'), default=-1)

    # Format Messages
    formatted_messages = []
    pdf_count = 0
    for idx, msg in enumerate(body.message_history):
        if msg.get('type') == 'ai':
            formatted_messages.append(AIMessage(content=msg.get('content')))
        elif msg.get('type') == 'human':
            history_images = msg.get('image_data_urls') or []
            history_pdfs = msg.get('pdf_attachments') or []
            formatted_messages.append(HumanMessage(
                content=build_human_content(msg.get('content'), history_images, history_pdfs, pdf_count)
            ))
            pdf_count += len(history_pdfs)

    # Top-level fields are kept for clients that still send the current turn separately
    if has_top_level_turn:
        formatted_messages.append(HumanMessage(
            content=build_human_content(body.user_input, body.image_data_urls, body.pdf_attachments, pdf_count)
        ))

    # Stream Logic

    llm = ChatOpenAI(
        model=body.model,
        api_key=openrouter_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
        stream_usage=True,
        extra_body={
            "reasoning": {
                "enabled": True
            }
        }
    )

    async def chat_stream_processor():
        total_tokens = 0
        yield f'data: {json.dumps({"thread_id": thread_id, "delta_content": ""})}\n\n'

        try:
            async for chunk in llm.astream(formatted_messages):
                payload = {"thread_id": thread_id}

                if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                    usage = chunk.usage_metadata
                    total_tokens = usage.get('total_tokens', 0)
                    payload["token_usage"] = usage

                if hasattr(chunk, 'response_metadata') and chunk.response_metadata:
                    cost = chunk.response_metadata.get('cost', 0)

                    if cost > 0:
                        payload["cost"] = cost

                        if resolved_user_id:
                            charge_id = str(uuid.uuid4())
                            payload["charge_id"] = charge_id
                            await queue_credit_deduction(
                                resolved_user_id,
                                cost,
                                charge_id,
                                thread_id,
                                body.model
                            )

                if chunk.content:
                    payload["delta_content"] = chunk.content

                if hasattr(chunk, "additional_kwargs") and "images" in chunk.additional_kwargs:
                    payload["images"] = chunk.additional_kwargs["images"]
                
                if hasattr(chunk, "additional_kwargs"):
                    reasoning = chunk.additional_kwargs.get("reasoning_content")
                    if reasoning:
                        payload["reasoning_content"] = reasoning

                if len(payload) > 1:
                    yield f'data: {json.dumps(payload)}\n\n'

            yield f'data: {json.dumps({"thread_id": thread_id, "complete": True, "final_token_usage": {"total_tokens": total_tokens}})}\n\n'

        except Exception as e:
            logging.error(f"Stream Error: {e}")
            yield f'data: {json.dumps({"error": str(e), "thread_id": thread_id})}\n\n'

    return StreamingResponse(chat_stream_processor(), media_type="text/event-stream")