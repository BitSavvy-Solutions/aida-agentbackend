import logging
from datetime import datetime, timezone
from typing import Optional
from db.mongo import get_tokens_collection

logger = logging.getLogger(__name__)

async def validate_api_token(token_string: str) -> Optional[dict]:
    """
    Looks up the token in the database. No prefix checks, just a direct lookup.
    """
    if not token_string:
        return None

    try:
        collection = get_tokens_collection()

        token_doc = await collection.find_one(
            {"tokenId": token_string},
            {"_id": 0}
        )

        if not token_doc:
            return None

        if not token_doc.get("isActive", False):
            return None

        # Check expiry
        expires_at_str = token_doc.get("expiresAt")
        if expires_at_str:
            expires_at = datetime.fromisoformat(
                expires_at_str.replace("Z", "+00:00")
            )
            if datetime.now(timezone.utc) > expires_at:
                # Deactivate expired token
                await collection.update_one(
                    {"tokenId": token_string},
                    {"$set": {"isActive": False}}
                )
                return None

        return token_doc

    except Exception as e:
        logger.error(f"Token validation error: {e}", exc_info=True)
        return None