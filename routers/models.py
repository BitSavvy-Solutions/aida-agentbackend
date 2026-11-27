# routers/models.py
import os
import re
import time
import asyncio
import logging
from typing import List, Optional, Dict, Tuple, Any
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import httpx

router = APIRouter()
logger = logging.getLogger(__name__)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_MODEL_URL = "https://openrouter.ai/api/v1/model"
CACHE_TTL_SECONDS = int(os.getenv("MODELS_CACHE_TTL_SECONDS", "300"))
SEARCH_LIMIT = int(os.getenv("MODELS_SEARCH_LIMIT", "50"))
ENDPOINTS_CACHE_TTL_SECONDS = int(os.getenv("MODEL_ENDPOINTS_CACHE_TTL_SECONDS", "300"))

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


class ModelReasoning(BaseModel):
    supported_efforts: Optional[List[str]] = None
    default_effort: Optional[str] = None
    default_enabled: Optional[bool] = None
    mandatory: bool = False


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
    reasoning: Optional[ModelReasoning] = None
    intelligence_index: Optional[float] = None


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
# Detail schemas
# ------------------------------------------------------------------

PRICING_DETAIL_KEYS = [
    "prompt",
    "completion",
    "input_cache_read",
    "input_cache_write",
    "request",
    "image",
    "web_search",
    "internal_reasoning",
]


class ModelPricingDetail(BaseModel):
    prompt: Optional[float] = None
    completion: Optional[float] = None
    input_cache_read: Optional[float] = None
    input_cache_write: Optional[float] = None
    request: Optional[float] = None
    image: Optional[float] = None
    web_search: Optional[float] = None
    internal_reasoning: Optional[float] = None


class ProviderEndpoint(BaseModel):
    provider_name: str
    tag: Optional[str] = None
    context_length: Optional[int] = None
    quantization: Optional[str] = None
    status: Optional[int] = None
    uptime_last_30m: Optional[float] = None
    pricing: Optional[ModelPricingDetail] = None
    supported_parameters: List[str] = []


class ModelDetailResponse(BaseModel):
    id: str
    canonical_slug: Optional[str] = None
    name: str
    description: Optional[str] = None
    created: Optional[int] = None
    hugging_face_id: Optional[str] = None
    knowledge_cutoff: Optional[str] = None
    context_length: Optional[int] = None
    architecture: Optional[Dict[str, Any]] = None
    pricing: Optional[ModelPricingDetail] = None
    top_provider: Optional[Dict[str, Any]] = None
    supported_parameters: List[str] = []
    api_endpoints: List[str] = []
    benchmarks: Optional[Dict[str, Any]] = None
    category: str = "chat"
    is_free: bool = False
    available: bool = True
    reason: Optional[str] = None
    reasoning: Optional[ModelReasoning] = None
    providers: List[ProviderEndpoint] = []


class RelatedModelsResponse(BaseModel):
    same_author: List[ModelInfo] = []
    similar: List[ModelInfo] = []


# ------------------------------------------------------------------
# Caches
# ------------------------------------------------------------------

_detail_cache: Dict[str, dict] = {}
_background_task: Optional[asyncio.Task] = None

# Derived catalog cache, invalidated whenever _detail_cache changes
_catalog_cache: Dict[str, List[ModelInfo]] = {}
_detail_cache_version = 0

# Per-model provider endpoints cache: model_id -> (monotonic_ts, endpoints)
_endpoints_cache: Dict[str, Tuple[float, List[dict]]] = {}


def _bump_detail_version() -> None:
    """Invalidate the derived catalog cache whenever the raw cache changes."""
    global _detail_cache_version
    _detail_cache_version += 1
    _catalog_cache.clear()


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


# Display names with bracketed variant markers (e.g. "(batch)") are not
# accessible from the UX and must be filtered out of all model lists.
BLOCKED_NAME_PATTERNS = [
    re.compile(r"\(\s*batch\s*\)", re.IGNORECASE),
]


def _is_name_allowed(name: Optional[str]) -> bool:
    if not name:
        return True
    return not any(p.search(name) for p in BLOCKED_NAME_PATTERNS)


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


def _reasoning_info(raw: dict) -> Optional[ModelReasoning]:
    """None when the model does not support reasoning."""
    if "reasoning" not in (raw.get("supported_parameters") or []):
        return None
    details = raw.get("reasoning") or {}
    return ModelReasoning(
        supported_efforts=details.get("supported_efforts"),
        default_effort=details.get("default_effort"),
        default_enabled=details.get("default_enabled"),
        mandatory=bool(details.get("mandatory")),
    )


def _intelligence_index(raw: dict) -> Optional[float]:
    aa = (raw.get("benchmarks") or {}).get("artificial_analysis") or {}
    return _safe_float(aa.get("intelligence_index"))


def _transform(raw: dict, available: bool = True, reason: Optional[str] = None, include_description: bool = True) -> Optional[ModelInfo]:
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
        description=raw.get("description") if include_description else None,
        available=available,
        reason=reason,
        reasoning=_reasoning_info(raw),
        intelligence_index=_intelligence_index(raw),
    )


def _pricing_detail(pricing_raw: dict) -> ModelPricingDetail:
    return ModelPricingDetail(
        **{k: _safe_float(pricing_raw.get(k)) for k in PRICING_DETAIL_KEYS}
    )


# ------------------------------------------------------------------
# Price cap: applies to the OUTPUT (completion) price only
# ------------------------------------------------------------------

MAX_MODEL_PRICE = float(os.getenv("MAX_MODEL_OUTPUT_PRICE", "16"))


def _is_price_allowed(info: ModelInfo) -> bool:
    """
    The cap applies to the OUTPUT (completion) price only.
    Models with no completion price data are allowed through.
    """
    if info.pricing is None or info.pricing.completion is None:
        return True
    return info.pricing.completion < MAX_MODEL_PRICE


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
# Free model detection (single source of truth)
# ------------------------------------------------------------------

def _raw_pricing_is_free(raw: dict) -> bool:
    """Free means every usage-based price component is zero or negative."""
    pricing = raw.get("pricing") or {}
    prompt = _safe_float(pricing.get("prompt"))
    completion = _safe_float(pricing.get("completion"))
    request = _safe_float(pricing.get("request"))
    if prompt is None or completion is None:
        return False  # can't verify, fail closed
    if prompt > 0 or completion > 0:
        return False
    if request is not None and request > 0:
        return False
    return True


def _is_free_pricing(pricing: Optional[ModelPricing]) -> bool:
    """A model is free when every usage-based price component is zero."""
    if pricing is None:
        return False
    return (
        pricing.prompt == 0
        and pricing.completion == 0
        and not pricing.request
    )


async def is_model_free(model_id: str) -> bool:
    # Strip trailing variant suffixes like ":online" (web search)
    base_id = model_id
    while base_id.endswith(":online"):
        base_id = base_id[: -len(":online")]

    raw = _detail_cache.get(base_id)
    if raw is None:
        raw = await _fetch_model_detail(base_id)

    if raw is not None:
        return _raw_pricing_is_free(raw)

    return base_id.endswith(":free")


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
    if entitlement.get("is_anonymous", True):
        raw = _detail_cache.get(model_id)
        if raw is None or not _raw_pricing_is_free(raw):
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
            resp = await client.get(f"{OPENROUTER_MODEL_URL}/{slug}", headers=_headers())
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            payload = resp.json()
            raw = payload.get("data") if isinstance(payload.get("data"), dict) else payload
            if isinstance(raw, dict) and raw.get("id"):
                _detail_cache[raw["id"]] = raw
                if slug != raw["id"]:
                    _detail_cache[slug] = raw  # aliases resolve to a concrete id
                _bump_detail_version()
                return raw
    except Exception as exc:
        logger.warning("Could not fetch model detail for %s: %s", slug, exc)

    return None


async def _fetch_model_endpoints(model_id: str) -> List[dict]:
    now = time.monotonic()
    cached = _endpoints_cache.get(model_id)
    if cached and now - cached[0] < ENDPOINTS_CACHE_TTL_SECONDS:
        return cached[1]

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                f"{OPENROUTER_MODELS_URL}/{model_id}/endpoints", headers=_headers()
            )
            resp.raise_for_status()
            data = resp.json().get("data") or {}
            endpoints = data.get("endpoints") or []
            _endpoints_cache[model_id] = (time.monotonic(), endpoints)
            return endpoints
    except Exception as exc:
        logger.warning("Could not fetch endpoints for %s: %s", model_id, exc)
        return []


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
        _bump_detail_version()
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
        _bump_detail_version()
        logger.info("OpenRouter background cache warmed with %s models", len(raw_list))
    except Exception as exc:
        logger.error("Background cache refresh failed: %s", exc)


async def periodic_refresh_cache():
    while True:
        await refresh_full_cache()
        await asyncio.sleep(CACHE_TTL_SECONDS)


# ------------------------------------------------------------------
# Query-time enforcement
# ------------------------------------------------------------------

async def enforce_model_price_cap(model_id: str) -> None:
    """
    Raises 403 if the requested model is a blocked variant or its
    output price exceeds the account cap. Called before any LLM call.
    """
    base_id = model_id
    while base_id.endswith(":online"):
        base_id = base_id[: -len(":online")]

    raw = _detail_cache.get(base_id)
    if raw is None:
        raw = await _fetch_model_detail(base_id)
    if raw is None:
        # Unknown model: let OpenRouter surface its own error
        return

    if not _is_name_allowed(raw.get("name")):
        raise HTTPException(
            status_code=403,
            detail=f"Model '{model_id}' is a restricted variant and is not available.",
        )

    completion = _safe_float((raw.get("pricing") or {}).get("completion"))
    if completion is not None and completion >= MAX_MODEL_PRICE:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Model '{model_id}' costs ${completion:.2f} per million output tokens. "
                f"Models higher than ${MAX_MODEL_PRICE:.2f} output cost are not allowed "
                f"on your account."
            ),
        )


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
        raw = _detail_cache.get(mid)
        if raw and not _is_name_allowed(raw.get("name")):
            continue
        available, reason = _availability(mid, entitlement)
        info = _transform(raw, available=available, reason=reason) if raw else _fallback_model(mid)

        if info:
            info.available = available
            info.reason = reason
            if _is_price_allowed(info):
                results.append(info)

    return _sort_default_models(results)


def _build_full_catalog(entitlement: dict, free_only: bool, cap: bool = True) -> List[ModelInfo]:
    """Build the full catalog from the warmed cache. Cached per cache-version,
    anonymity, free-filter, and cap setting, so repeat requests are pure serialization."""
    key = f"{_detail_cache_version}|anon={entitlement.get('is_anonymous', True)}|free={free_only}|cap={cap}"
    cached = _catalog_cache.get(key)
    if cached is not None:
        return cached

    results: List[ModelInfo] = []
    for raw in _detail_cache.values():
        model_id = raw.get("id")
        if not model_id:
            continue
        if not _is_name_allowed(raw.get("name")):
            continue
        available, reason = _availability(model_id, entitlement)
        info = _transform(raw, available=available, reason=reason, include_description=False)
        if not info:
            continue
        if cap and not _is_price_allowed(info):
            continue
        if free_only and not _is_free_pricing(info.pricing):
            continue
        results.append(info)

    # Signed-in-capable models first, then free models, then alphabetical
    results.sort(key=lambda m: (not m.available, not _is_free_pricing(m.pricing), m.name.lower()))
    _catalog_cache[key] = results
    return results


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@router.get("/models", response_model=ModelsResponse)
async def get_models(request: Request, q: Optional[str] = None, limit: int = 50, free: bool = False, full: bool = False, cap: bool = True):
    auth_header = request.headers.get("Authorization", "")
    token_string = ""
    if auth_header.startswith("Bearer "):
        token_string = auth_header[7:].strip()

    entitlement = await _resolve_user_entitlement(token_string or None)

    if full:
        # Cold-start safety: warm the cache synchronously if the background
        # refresh has not populated it yet.
        if not _detail_cache:
            await refresh_full_cache()
        text_models = _build_full_catalog(entitlement, free_only=free, cap=cap)
    elif q and q.strip():
        raw_results = await _fetch_openrouter_search(q.strip(), min(limit, SEARCH_LIMIT))
        text_models = []
        for raw in raw_results:
            mid = raw.get("id")
            if not _is_name_allowed(raw.get("name")):
                continue
            available, reason = _availability(mid, entitlement)
            info = _transform(raw, available=available, reason=reason)
            if info and _is_price_allowed(info):
                text_models.append(info)
    else:
        text_models = await _build_default_models(entitlement)

    # Apply the free filter to search/default responses too
    if free and not full:
        text_models = [m for m in text_models if _is_free_pricing(m.pricing)]

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


# NOTE: this route must be registered BEFORE the catch-all /model/{model_id:path}
# route below, otherwise "related" would be swallowed into the model id.
@router.get("/model/{model_id:path}/related", response_model=RelatedModelsResponse)
async def get_model_related(model_id: str):
    if not _detail_cache:
        await refresh_full_cache()

    raw = _detail_cache.get(model_id)
    if raw is None:
        raw = await _fetch_model_detail(model_id)
    if raw is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    target_id = raw.get("id", model_id)
    author = target_id.split("/")[0]
    arch = raw.get("architecture") or {}
    target_category = CATEGORY_OVERRIDES.get(target_id) or _categorize(
        target_id, arch.get("modality"), arch.get("instruct_type")
    )
    target_intel = _intelligence_index(raw)

    same_author: List[ModelInfo] = []
    similar: List[ModelInfo] = []
    seen = {target_id}

    for oid, other in _detail_cache.items():
        mid = other.get("id")
        if not mid or mid in seen:
            continue
        seen.add(mid)
        if not _is_name_allowed(other.get("name")):
            continue
        info = _transform(other, include_description=False)
        if info is None or not _is_price_allowed(info):
            continue
        if mid.split("/")[0] == author:
            if len(same_author) < 8:
                same_author.append(info)
            continue
        if info.category == target_category:
            similar.append(info)

    def similarity(info: ModelInfo):
        oi = info.intelligence_index
        if target_intel is not None:
            if oi is not None:
                return (0, abs(oi - target_intel), info.name.lower())
            return (1, 0, info.name.lower())
        if oi is not None:
            return (0, -oi, info.name.lower())
        return (2, 0, info.name.lower())

    similar.sort(key=similarity)

    return RelatedModelsResponse(same_author=same_author, similar=similar[:8])


@router.get("/model/{model_id:path}", response_model=ModelDetailResponse)
async def get_model_detail(request: Request, model_id: str):
    model_id = model_id.strip().strip("/")
    if not model_id:
        raise HTTPException(status_code=404, detail="Model id is required")

    raw = _detail_cache.get(model_id)
    if raw is None:
        raw = await _fetch_model_detail(model_id)
    if raw is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    # Availability follows the same rules as the picker
    auth_header = request.headers.get("Authorization", "")
    token_string = auth_header[7:].strip() if auth_header.startswith("Bearer ") else ""
    entitlement = await _resolve_user_entitlement(token_string or None)

    resolved_id = raw.get("id", model_id)
    available, reason = _availability(resolved_id, entitlement)
    if not _is_name_allowed(raw.get("name")):
        available, reason = False, "This model variant is not accessible"

    endpoints_raw = await _fetch_model_endpoints(resolved_id)
    providers = [
        ProviderEndpoint(
            provider_name=e.get("provider_name") or "Unknown",
            tag=e.get("tag"),
            context_length=e.get("context_length"),
            quantization=e.get("quantization"),
            status=e.get("status"),
            uptime_last_30m=e.get("uptime_last_30m"),
            pricing=_pricing_detail(e.get("pricing") or {}),
            supported_parameters=e.get("supported_parameters") or [],
        )
        for e in endpoints_raw
    ]

    arch = raw.get("architecture") or {}

    return ModelDetailResponse(
        id=resolved_id,
        canonical_slug=raw.get("canonical_slug"),
        name=raw.get("name") or resolved_id.split("/")[-1].replace("-", " ").title(),
        description=raw.get("description"),
        created=raw.get("created"),
        hugging_face_id=raw.get("hugging_face_id"),
        knowledge_cutoff=raw.get("knowledge_cutoff"),
        context_length=raw.get("context_length"),
        architecture=arch,
        pricing=_pricing_detail(raw.get("pricing") or {}),
        top_provider=raw.get("top_provider"),
        supported_parameters=raw.get("supported_parameters") or [],
        api_endpoints=raw.get("endpoints") or [],
        benchmarks=raw.get("benchmarks"),
        category=CATEGORY_OVERRIDES.get(resolved_id)
        or _categorize(resolved_id, arch.get("modality"), arch.get("instruct_type")),
        is_free=_raw_pricing_is_free(raw),
        available=available,
        reason=reason,
        reasoning=_reasoning_info(raw),
        providers=providers,
    )