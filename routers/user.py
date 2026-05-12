import os
from fastapi import APIRouter, HTTPException, Depends, status, Query
from fastapi.security import OAuth2PasswordBearer
from motor.motor_asyncio import AsyncIOMotorClient
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime, timezone, timedelta

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
    date: str
    type: str # "CREDIT" or "USAGE"
    description: str
    details: Optional[Dict[str, Any]] = None

class PaginatedResponse(BaseModel):
    data: List[TransactionLog]
    has_more: bool
    total_cost: float = 0.0

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


@router.get("/credits/history", response_model=PaginatedResponse)
async def get_credit_history(
    cursor: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100),
    user: str = Depends(get_current_user)
):
    """
    Fetches ONLY payment history (Stripe Top-ups) using token.
    """
    user_id = user.get("userId")

    # Base query for credits
    query = {
        "userId": user_id,
        "type": "CREDIT"
    }

    # Apply cursor if loading more
    if cursor:
        query["transactionDate"] = {"$lt": cursor}

    # Fetch limit + 1 to check if there are more records
    db_cursor = db.credit_transactions.find(query).sort("transactionDate", -1).limit(limit + 1)

    history = []
    async for doc in db_cursor:
        public_id = doc.get("chargeId") or doc.get("transactionId") or str(doc.get("_id"))
        desc = doc.get("details", {}).get("description") or doc.get("reason") or "Top-up"

        tx_date = doc.get("transactionDate")
        if isinstance(tx_date, datetime):
            tx_date = tx_date.isoformat()

        history.append({
            "transactionId": public_id,
            "amount": doc.get("creditChange"),
            "date": tx_date,
            "type": "CREDIT",
            "description": desc,
            "details": doc.get("details")
        })

    has_more = len(history) > limit
    if has_more:
        history.pop()

    return {"data": history, "has_more": has_more, "total_cost": 0.0}


@router.get("/usage/logs", response_model=PaginatedResponse)
async def get_usage_logs(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    cursor: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100),
    user: str = Depends(get_current_user)
):
    """
    Fetches ONLY AI usage logs using token.
    """
    user_id = user.get("userId")
    # 1. Base Query for Usage
    query = {"userId": user_id, "creditChange": {"$lt": 0}}
    date_filter = {}

    # Handle Date Range (String comparison works if dates are stored as ISO strings)
    if start_date:
        # Start of day
        date_filter["$gte"] = f"{start_date}T00:00:00"
    if end_date:
        # End of day
        date_filter["$lte"] = f"{end_date}T23:59:59.999999"

    # 2. Apply Cursor (Overrides upper bound)
    if cursor:
        date_filter["$lt"] = cursor

    if date_filter:
        query["transactionDate"] = date_filter

    # 3. Calculate Total Cost (Only for the selected date range, ignoring cursor)
    cost_query = {"userId": user_id, "creditChange": {"$lt": 0}}
    cost_date_filter = {}
    if start_date: cost_date_filter["$gte"] = f"{start_date}T00:00:00"
    if end_date: cost_date_filter["$lte"] = f"{end_date}T23:59:59.999999"
    if cost_date_filter: cost_query["transactionDate"] = cost_date_filter

    pipeline = [
        {"$match": cost_query},
        {"$group": {
                "_id": None, 
                "total_cost": {"$sum": "$creditChange"}
            }
        }
    ]
    agg_result = await db.credit_transactions.aggregate(pipeline).to_list(1)
    # Convert negative usage to positive cost
    total_cost = abs(agg_result[0]["total_cost"]) if agg_result else 0.0

    # 4. Fetch Paginated Data
    db_cursor = db.credit_transactions.find(query).sort("transactionDate", -1).limit(limit + 1)

    logs = []
    async for doc in db_cursor:
        details = doc.get("details", {})
        public_id = doc.get("chargeId") or doc.get("transactionId") or str(doc.get("_id"))
        model_name = details.get("modelUsed") or doc.get("reason") or "Unknown Model"

        tx_date = doc.get("transactionDate")
        if isinstance(tx_date, datetime):
            tx_date = tx_date.isoformat()

        logs.append({
            "transactionId": public_id,
            "amount": doc.get("creditChange"),
            "date": tx_date,
            "type": "USAGE",
            "description": model_name, 
            "details": details
        })
    
    has_more = len(logs) > limit
    if has_more:
        logs.pop()
    
    return {"data": logs, "has_more": has_more, "total_cost": total_cost}