import os
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from motor.motor_asyncio import AsyncIOMotorClient
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime, timezone

router = APIRouter()

# --- Database Connection ---
CONNECTION_STRING = os.getenv("MONGODB_CONNECTION_STRING")
DB_NAME = "userdb"

# We initialize the client lazily
client = AsyncIOMotorClient(CONNECTION_STRING)
db = client[DB_NAME]

# --- Security / Auth ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
TOKEN_PREFIX = "ethikey_"

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    Validates the 'ethikey_' token against the database.
    Returns the full user document so endpoints don't have to fetch it again.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not token.startswith(TOKEN_PREFIX):
        raise credentials_exception

    # 1. Find the token in the database
    token_doc = await db.api_tokens.find_one({"tokenId": token})
    
    if not token_doc or not token_doc.get("isActive", False):
        raise credentials_exception

    # 2. Check if the token is expired
    expires_at_str = token_doc.get("expiresAt")
    if expires_at_str:
        # Handle the 'Z' timezone indicator from JS/Python isoformat
        expires_at = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
        if datetime.now(timezone.utc) > expires_at:
            raise credentials_exception

    # 3. Get the user associated with this token
    user_id = token_doc.get("userId")
    if not user_id:
        raise credentials_exception

    user_doc = await db.users.find_one({"userId": user_id})
    if not user_doc:
        raise credentials_exception

    return user_doc

# --- Response Models ---

class UserProfileResponse(BaseModel):
    userId: str
    email: str
    name: Optional[str] = None
    profilePictureUrl: Optional[str] = None
    balance: float

class TransactionLog(BaseModel):
    transactionId: str
    amount: float
    date: datetime
    type: str # "CREDIT" or "USAGE"
    description: str
    details: Optional[Dict[str, Any]] = None

# --- Endpoints ---

@router.get("/profile", response_model=UserProfileResponse)
async def get_user_profile(user: str = Depends(get_current_user)):
    """
    Fetches user profile info + current balance using token.
    """
    user_id = user.get("userId")

    # Get Credit Balance
    credit_doc = await db.user_credits.find_one({"userId": user_id})
    current_balance = credit_doc.get("balance", 0.0) if credit_doc else 0.0

    return {
        "userId": user_id,
        "email": user.get("email"),
        "name": user.get("profile", {}).get("name"),
        "profilePictureUrl": user.get("profilePictureUrl"),
        "balance": current_balance
    }


@router.get("/credits/history", response_model=List[TransactionLog])
async def get_credit_history(user: str = Depends(get_current_user)):
    """
    Fetches ONLY payment history (Stripe Top-ups) using token.
    """
    user_id = user.get("userId")

    # Query Transactions
    # Filter: userId matches AND type is 'CREDIT'
    cursor = db.credit_transactions.find({
        "userId": user_id,
        "type": "CREDIT" 
    }).sort("transactionDate", -1).limit(50)

    history = []
    async for doc in cursor:
        public_id = doc.get("chargeId") or doc.get("transactionId") or str(doc.get("_id"))
        desc = doc.get("details", {}).get("description") or doc.get("reason") or "Top-up"

        history.append({
            "transactionId": public_id,
            "amount": doc.get("creditChange"),
            "date": doc.get("transactionDate"),
            "type": "CREDIT",
            "description": desc,
            "details": doc.get("details")
        })
    
    return history


@router.get("/usage/logs", response_model=List[TransactionLog])
async def get_usage_logs(user: str = Depends(get_current_user)):
    """
    Fetches ONLY AI usage logs using token.
    """
    user_id = user.get("userId")

    # Query Transactions
    # Filter: userId matches AND creditChange is negative (spending)
    cursor = db.credit_transactions.find({
        "userId": user_id,
        "creditChange": {"$lt": 0} 
    }).sort("transactionDate", -1).limit(100)

    logs = []
    async for doc in cursor:
        details = doc.get("details", {})
        public_id = doc.get("chargeId") or doc.get("transactionId") or str(doc.get("_id"))
        model_name = details.get("modelUsed") or doc.get("reason") or "Unknown Model"

        logs.append({
            "transactionId": public_id,
            "amount": doc.get("creditChange"),
            "date": doc.get("transactionDate"),
            "type": "USAGE",
            "description": model_name, 
            "details": details
        })
    
    return logs