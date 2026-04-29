from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Import Routers
from routers import chat, audio, scraper, github, user

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Replaces the deprecated @app.on_event("startup") pattern.
    Runs startup logic before the app starts accepting requests,
    and shutdown logic after the last request is handled.
    """
    # Startup: verify MongoDB can be reached before accepting requests
    try:
        from db.mongo import get_mongo_client
        client = get_mongo_client()
        await client.admin.command("ping")
        logging.info("MongoDB connection verified on startup.")
    except Exception as e:
        # Log but do not crash - the app can still serve requests that
        # don't need auth (anonymous free model requests will still work)
        logging.error(
            f"MongoDB startup check failed: {e}. "
            "Token-authenticated requests will fail until the connection is restored."
        )
    yield
    # Shutdown: nothing to clean up for now


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

@app.get("/")
async def health_check():
    return {"status": "running", "service": "aida-agentbackend"}