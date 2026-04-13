from dotenv import load_dotenv
import os
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Import Routers
from routers import chat, audio, scraper, github, user

# Setup Logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Iverse Backend")

# CORS (Allow frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(chat.router, prefix="/api")
app.include_router(audio.router, prefix="/api")
app.include_router(scraper.router, prefix="/api")
app.include_router(github.router, prefix="/api/github")
app.include_router(user.router, prefix="/api/user")

@app.get("/")
async def health_check():
    return {"status": "running", "service": "aida-agentbackend"}