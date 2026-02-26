import os
import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class GithubTokenRequest(BaseModel):
    code: str

@router.post("/token")
async def exchange_github_token(body: GithubTokenRequest):
    client_id = os.getenv("GITHUB_CLIENT_ID")
    client_secret = os.getenv("GITHUB_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise HTTPException(status_code=500, detail="Server misconfiguration: Missing GitHub credentials")

    try:
        response = requests.post(
            "https://github.com/login/oauth/access_token",
            json={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": body.code
            },
            headers={"Accept": "application/json"}
        )
        data = response.json()
        
        if "error" in data:
            raise HTTPException(status_code=400, detail=data.get("error_description"))
            
        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))