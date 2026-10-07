# routers/models.py
import os
import asyncio
import logging
from typing import List, Optional, Dict, Tuple
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import httpx

router = APIRouter()
logger = logging.getLogger(__name__)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
CACHE_TTL_SECONDS = int(os.getenv("MODELS_CACHE_TTL_SECONDS", "300"))
SEARCH_LIMIT = int(os.getenv("MODELS_SEARCH_LIMIT", "50"))

# ------------------------------------------------------------------
# Curated defaults
# ------------------------------------------------------------------

DEFAULT_TEXT_MODEL = "deepseek/deepseek-v4-pro"

DEFAULT_TEXT_MODEL_IDS = [
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-chat-v3-0324",
    "openai/gpt-5.1",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3.1-flash-lite-preview",
    "~anthropic/claude-sonnet-latest",
    "~anthropic/claude-haiku-latest",
    "google/gemini-2.5-flash-image",
    "perplexity/sonar",
    "z-ai/glm-5.2",
    "qwen/qwen3.7-max",
    "minimax/minimax-m3",
    "moonshotai/kimi-k2.7-code",
    "xiaomi/mimo-v2.5-pro",
]

FREE_MODELS = {
    "google/gemini-3.1-flash-lite-preview",
    "deepseek/deepseek-chat-v3-0324",
}

RECOMMENDED_FREE = [
    "google/gemini-3.1-flash-lite-preview",
    "deepseek/deepseek-chat-v3-0324",
]

RECOMMENDED_PAID = [
    "openai/gpt-5.1",
    "~anthropic/claude-sonnet-latest",
]

CATEGORY_OVERRIDES = {
    "deepseek/deepseek-v4-pro": "reasoning",
    "deepseek/deepseek-v3.2": "reasoning",
    "deepseek/deepseek-chat-v3-0324": "reasoning",
    "openai/gpt-5.1": "reasoning",
    "google/gemini-3.1-pro-preview": "reasoning",
    "google/gemini-3.1-flash-lite-preview": "chat",
    "~anthropic/claude-sonnet-latest": "reasoning",
    "~anthropic/claude-haiku-latest": "reasoning",
    "google/gemini-2.5-flash-image": "vision",
    "perplexity/sonar": "chat",
    "z-ai/glm-5.2": "reasoning",
    "qwen/qwen3.7-max": "reasoning",
    "minimax/minimax-m3": "reasoning",
    "moonshotai/kimi-k2.7-code": "reasoning",
    "xiaomi/mimo-v2.5-pro": "reasoning",
}


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------

class ModelPricing(BaseModel):
    prompt: Optional[float] = None
    completion: Optional[float] = None
    image: Optional[float] = None
    request: Optional[float] = None


class ModelInfo(BaseModel):
    id: str
    name: str
    category: str
    modality: Optional[str] = None
    context_length: Optional[int] = None
    pricing: Optional[ModelPricing] = None
    description: Optional[str] = None
    available: bool = True
    reason: Optional[str] = None


AUDIO_MODELS = [
    ModelInfo(id="whisper-1", name="OpenAI Whisper", category="audio", modality="audio->text"),
    ModelInfo(id="whisper-1|translate", name="OpenAI Whisper (Translate to EN)", category="audio", modality="audio->text"),
    ModelInfo(id="saaras:v3", name="Sarvam Saaras v3", category="audio", modality="audio->text"),
    ModelInfo(id="saaras:v3|translate", name="Sarvam Saaras v3 (Translate to EN)", category="audio", modality="audio->text"),
]

class Defaults(BaseModel):
    text: str
    audio: str


class ModelsResponse(BaseModel):
    text: List[ModelInfo]
    audio: List[ModelInfo]
    defaults: Defaults
    recommended: List[str]


# ------------------------------------------------------------------
# Caches
# ------------------------------------------------------------------

_detail_cache: Dict[str, dict] = {}
_background_task: Optional[asyncio.Task] = None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _headers():
    return {
        "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://iverse.space"),
        "X-Title": "AIDA",
    }


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _categorize(model_id: str, modality: Optional[str], instruct_type: Optional[str]) -> str:
    mid = model_id.lower()
    mod = (modality or "").lower()

    if "image" in mod:
        if any(r in mid for r in ["o1", "o3", "o4", "r1", "reasoning", "deepseek-r1", "claude-3-7-sonnet"]):
            return "reasoning"
        return "vision"

    if any(r in mid for r in ["o1", "o3", "o4", "r1", "reasoning", "deepseek-r1", "claude-3-7-sonnet", "kimi-k2", "qwen3"]):
        return "reasoning"

    return "chat"


def _transform(raw: dict, available: bool = True, reason: Optional[str] = None) -> Optional[ModelInfo]:
    model_id = raw.get("id")
    if not model_id:
        return None

    arch = raw.get("architecture") or {}
    pricing_raw = raw.get("pricing") or {}

    return ModelInfo(
        id=model_id,
        name=raw.get("name") or model_id.split("/")[-1].replace("-", " ").title(),
        category=CATEGORY_OVERRIDES.get(model_id)
                 or _categorize(model_id, arch.get("modality"), arch.get("instruct_type")),
        modality=arch.get("modality"),
        context_length=raw.get("context_length"),
        pricing=ModelPricing(
            prompt=_safe_float(pricing_raw.get("prompt")),
            completion=_safe_float(pricing_raw.get("completion")),
            image=_safe_float(pricing_raw.get("image")),
            request=_safe_float(pricing_raw.get("request")),
        ),
        description=raw.get("description"),
        available=available,
        reason=reason,
    )


# ------------------------------------------------------------------
# Price cap: only return models that cost less than $16
# ------------------------------------------------------------------

MAX_MODEL_PRICE = 16.0


def _is_price_allowed(info: ModelInfo) -> bool:
    """
    Return False if any known price component is >= $16.
    Models with no pricing data are allowed through.
    """
    if info.pricing is None:
        return True

    known_prices = [
        p for p in (
            info.pricing.prompt,
            info.pricing.completion,
            info.pricing.image,
            info.pricing.request,
        )
        if p is not None
    ]

    if not known_prices:
        return True

    return max(known_prices) < MAX_MODEL_PRICE


def _fallback_model(model_id: str) -> Optional[ModelInfo]:
    """
    Generate a minimal fallback ModelInfo when OpenRouter detail fetch fails.
    """
    return ModelInfo(
        id=model_id,
        name=model_id.split("/")[-1].replace("-", " ").title(),
        category=CATEGORY_OVERRIDES.get(model_id, "chat"),
        pricing=ModelPricing(),
    )


# ------------------------------------------------------------------
# Availability logic
# ------------------------------------------------------------------

async def _resolve_user_entitlement(token_string: Optional[str]) -> dict:
    """
    Lightweight entitlement resolver. Returns free/anonymous defaults if
    no token is supplied or if lookup fails.
    """
    result = {
        "is_anonymous": True,
    }

    if not token_string:
        return result

    # If you wire in auth later, set is_anonymous=False here.
    # result["is_anonymous"] = False

    return result


def _availability(model_id: str, entitlement: dict) -> Tuple[bool, Optional[str]]:
    if entitlement.get("is_anonymous", True) and model_id not in FREE_MODELS:
        return False, "Requires sign in"
    return True, None


# ------------------------------------------------------------------
# OpenRouter fetching
# ------------------------------------------------------------------

async def _fetch_model_detail(slug: str) -> Optional[dict]:
    if slug in _detail_cache:
        return _detail_cache[slug]

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(f"{OPENROUTER_MODELS_URL}/{slug}", headers=_headers())
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            payload = resp.json()
            raw = payload.get("data") if isinstance(payload.get("data"), dict) else payload
            if isinstance(raw, dict) and raw.get("id"):
                _detail_cache[raw["id"]] = raw
                return raw
    except Exception as exc:
        logger.warning("Could not fetch model detail for %s: %s", slug, exc)

    return None


async def _ensure_default_details(ids: List[str]):
    missing = [i for i in ids if i not in _detail_cache]
    if missing:
        await asyncio.gather(*[_fetch_model_detail(i) for i in missing])


async def _fetch_openrouter_search(q: str, limit: int) -> List[dict]:
    params = {
        "q": q,
        "output_modalities": "text",
        "sort": "most-popular",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(OPENROUTER_MODELS_URL, headers=_headers(), params=params)
        resp.raise_for_status()
        items = resp.json().get("data", [])
        for r in items:
            if r.get("id"):
                _detail_cache[r["id"]] = r
        return items[:limit]


async def _fetch_openrouter_full() -> List[dict]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(OPENROUTER_MODELS_URL, headers=_headers())
        resp.raise_for_status()
        return resp.json().get("data", [])


async def refresh_full_cache():
    try:
        raw_list = await _fetch_openrouter_full()
        for r in raw_list:
            if r.get("id"):
                _detail_cache[r["id"]] = r
        logger.info("OpenRouter background cache warmed with %s models", len(raw_list))
    except Exception as exc:
        logger.error("Background cache refresh failed: %s", exc)


async def periodic_refresh_cache():
    while True:
        await refresh_full_cache()
        await asyncio.sleep(CACHE_TTL_SECONDS)


# ------------------------------------------------------------------
# Build responses
# ------------------------------------------------------------------

def _sort_default_models(models: List[ModelInfo]) -> List[ModelInfo]:
    recommended = RECOMMENDED_FREE + RECOMMENDED_PAID
    order = {m: i for i, m in enumerate(recommended)}

    def sort_key(m: ModelInfo):
        available_rank = 0 if m.available else 1
        return (available_rank, order.get(m.id, 999), m.name)

    return sorted(models, key=sort_key)


async def _build_default_models(entitlement: dict) -> List[ModelInfo]:
    await _ensure_default_details(DEFAULT_TEXT_MODEL_IDS)

    results = []
    for mid in DEFAULT_TEXT_MODEL_IDS:
        available, reason = _availability(mid, entitlement)
        raw = _detail_cache.get(mid)
        info = _transform(raw, available=available, reason=reason) if raw else _fallback_model(mid)

        if info:
            info.available = available
            info.reason = reason
            if _is_price_allowed(info):
                results.append(info)

    return _sort_default_models(results)


# ------------------------------------------------------------------
# Endpoint
# ------------------------------------------------------------------

@router.get("/models", response_model=ModelsResponse)
async def get_models(request: Request, q: Optional[str] = None, limit: int = 50):
    auth_header = request.headers.get("Authorization", "")
    token_string = ""
    if auth_header.startswith("Bearer "):
        token_string = auth_header[7:].strip()

    entitlement = await _resolve_user_entitlement(token_string or None)

    if q and q.strip():
        raw_results = await _fetch_openrouter_search(q.strip(), min(limit, SEARCH_LIMIT))
        text_models = []
        for raw in raw_results:
            mid = raw.get("id")
            available, reason = _availability(mid, entitlement)
            info = _transform(raw, available=available, reason=reason)
            if info and _is_price_allowed(info):
                text_models.append(info)
    else:
        text_models = await _build_default_models(entitlement)

    # Default selection: pick the first available model, falling back to the global default
    default_text = DEFAULT_TEXT_MODEL
    available_ids = {m.id for m in text_models if m.available}
    if default_text not in available_ids:
        default_text = next((m.id for m in text_models if m.available), DEFAULT_TEXT_MODEL)

    recommended = list(RECOMMENDED_PAID if not entitlement.get("is_anonymous", True) else RECOMMENDED_FREE)
    recommended = [m for m in recommended if m in {x.id for x in text_models} or not q]

    return ModelsResponse(
        text=text_models,
        audio=AUDIO_MODELS,
        defaults=Defaults(text=default_text, audio="whisper-1"),
        recommended=recommended,
    )