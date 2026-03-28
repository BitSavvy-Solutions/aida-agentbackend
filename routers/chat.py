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

router = APIRouter()

# Pydantic Model for Request Body
class ChatRequest(BaseModel):
    user_input: Optional[str] = None
    image_data_urls: List[str] = []
    model: str = 'google/gemini-flash-1.5'
    user_id: Optional[str] = None
    message_history: List[Dict[str, Any]] = []
    thread_id: Optional[str] = None

# Config
openrouter_key = os.getenv("OPENROUTER_API_KEY")
ALLOWED_ANONYMOUS_MODELS = [r"google/gemini-3-flash-preview", r"^.*deepseek.*"]
COMPILED_ANONYMOUS_PATTERNS = [re.compile(p, re.IGNORECASE) for p in ALLOWED_ANONYMOUS_MODELS]

@router.post("/iverse_agent")
async def iverse_agent(body: ChatRequest):
    thread_id = body.thread_id or str(uuid.uuid4())
    # --- Security Check ---
    if not body.user_id:
        is_allowed = any(p.match(body.model) for p in COMPILED_ANONYMOUS_PATTERNS)
        if not is_allowed:
            raise HTTPException(status_code=403, detail=f"Model '{body.model}' requires sign-in")

    if not body.user_input and not body.image_data_urls and not body.message_history:
        raise HTTPException(status_code=400, detail="Input required")

    # --- Format Messages ---
    formatted_messages = []
    for msg in body.message_history:
        if msg.get('type') == 'ai':
            formatted_messages.append(AIMessage(content=msg.get('content')))
        elif msg.get('type') == 'human':
            formatted_messages.append(HumanMessage(content=msg.get('content')))

    new_content = []
    if body.user_input:
        new_content.append({"type": "text", "text": body.user_input})
    for url in body.image_data_urls:
        new_content.append({"type": "image_url", "image_url": {"url": url}})
    
    if new_content:
        formatted_messages.append(HumanMessage(content=new_content))

    # --- Stream Logic ---
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
                        if body.user_id:
                            charge_id = str(uuid.uuid4())
                            payload["charge_id"] = charge_id
                            await queue_credit_deduction(body.user_id, cost, charge_id, thread_id, body.model)

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