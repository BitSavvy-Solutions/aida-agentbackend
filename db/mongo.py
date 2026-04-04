# db/mongo.py

import os
import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

logger = logging.getLogger(__name__)

_client: Optional[AsyncIOMotorClient] = None

def get_mongo_client() -> AsyncIOMotorClient:
    """
    Lazy singleton Motor client. Initialized once on first call and reused
    for the lifetime of the process. Motor handles its own internal connection
    pooling so this single client is safe to share across all async requests.
    """
    global _client
    if _client is None:
        conn_str = os.getenv("MONGODB_CONNECTION_STRING")
        if not conn_str:
            raise EnvironmentError(
                "MONGODB_CONNECTION_STRING is not set in environment variables."
            )
        _client = AsyncIOMotorClient(conn_str)
        logger.info("Motor MongoDB client initialized.")
    return _client


def get_tokens_collection() -> AsyncIOMotorCollection:
    """
    Returns the api_tokens collection reference.
    The collection name is read from env so it matches the Azure Functions backend exactly.
    """
    client = get_mongo_client()
    db_name = os.getenv("MONGODB_DB_NAME", "userdb")
    collection_name = os.getenv("API_TOKENS_COLLECTION_NAME", "api_tokens")
    return client[db_name][collection_name]