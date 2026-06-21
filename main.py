from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Import Routers
from routers import chat, audio, scraper, github, user, models

import asyncio
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start OpenRouter background cache refresh
    cache_task = asyncio.create_task(models.periodic_refresh_cache())

    try:
        from db.mongo import get_mongo_client
        client = get_mongo_client()
        await client.admin.command("ping")
        logging.info("MongoDB connection verified on startup.")
    except Exception as e:
        logging.error(
            f"MongoDB startup check failed: {e}. "
            "Token-authenticated requests will fail until the connection is restored."
        )

    yield

    # Shutdown: cancel background task
    cache_task.cancel()
    try:
        await cache_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Iverse Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api")
app.include_router(audio.router, prefix="/api")
app.include_router(scraper.router, prefix="/api")
app.include_router(github.router, prefix="/api/github")
app.include_router(user.router, prefix="/api/user")
app.include_router(models.router, prefix="/api")

@app.get("/")
async def health_check():
    return {"status": "running", "service": "aida-agentbackend"}