import email
import os
import json
import uuid
import re
import logging
import asyncio
import httpx
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from apis.chunk_enhancer import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from apis.credit_manager import queue_credit_deduction
from dependencies.auth import validate_api_token

router = APIRouter()

# ADDED: In-memory cache to track user pings (prevents spamming the DB)
user_ping_cache = {}

GLOBAL_USER_API_URL = os.getenv("GLOBAL_USER_API_URL", "http://host.docker.internal:7071/api/users")

async def ping_global_user_api(user_id: str):
    """Pings the global user API asynchronously, max once per 24 hours per user."""
    now = datetime.now()
    last_ping = user_ping_cache.get(user_id)
    
    # Check if we already pinged this user in the last 24 hours
    if last_ping and (now - last_ping) < timedelta(hours=24):
        logging.debug(f">>>CACHE HIT<<< User {user_id} already pinged today. Skipping API call.")
        return
        
    # Update cache immediately to prevent race conditions
    user_ping_cache[user_id] = now
    logging.info(f">>>PINGING<<< Sending activity update for User {user_id}...")
    
    # Optional: Prevent memory leaks by clearing old cache entries if it gets too big
    if len(user_ping_cache) > 5000:
        stale_keys = [k for k, v in user_ping_cache.items() if (now - v) > timedelta(hours=24)]
        for k in stale_keys:
            user_ping_cache.pop(k, None)

    try:
        # Determine if the identifier is an email or a user_id
        is_email = "@" in user_id
        
        # If it's an email, send it in the body. If it's an ID, append it to the URL.
        url = GLOBAL_USER_API_URL if is_email else f"{GLOBAL_USER_API_URL}/{user_id}"
        payload = {"action": "ping", "email": user_id} if is_email else {"action": "ping"}

        # Fire the request asynchronously
        async with httpx.AsyncClient() as client:
            response = await client.put(
                url, 
                json=payload, 
                timeout=3.0
            )
            if response.status_code == 200:
                logging.info(f">>>SUCCESS<<< Pinged {user_id}. Status: 200")
            else:
                logging.error(f">>>ERROR<<< Failed to ping. Status: {response.status_code} - {response.text}")
   
    except Exception as e:
        logging.error(f"Exception while pinging user activity for {user_id}: {e}")

# Pydantic Model for Request Body
class ChatRequest(BaseModel):
    user_input: Optional[str] = None
    image_data_urls: List[str] = []
    model: str = 'google/gemini-flash-1.5'
    email: Optional[str] = None
    user_id: Optional[str] = None
    message_history: List[Dict[str, Any]] = []
    thread_id: Optional[str] = None


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
    if not body.user_input and not body.image_data_urls and not body.message_history:
        raise HTTPException(status_code=400, detail="Input required")
    
    # ADDED: Fire the background ping (It will instantly exit if 24h haven't passed)
    if body.user_id:
        asyncio.create_task(ping_global_user_api(body.user_id))
    elif body.email:
        asyncio.create_task(ping_global_user_api(body.email))

    # Format Messages
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