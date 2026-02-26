# apis/credit_manager.py
import os
import json
import logging
import base64
from azure.storage.queue.aio import QueueClient

logger = logging.getLogger("uvicorn.error")

# In Docker, this will point to Azurite via .env
CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
QUEUE_NAME = "credit-deductions"

async def queue_credit_deduction(
    user_id: str,
    cost: float,
    charge_id: str,
    thread_id: str,
    model_name: str
):
    """
    Asynchronously sends a BASE64-ENCODED message to the credit deduction queue.
    """
    if not all([user_id, cost > 0, charge_id]):
        return

    if not CONNECTION_STRING:
        logger.error("CRITICAL: AZURE_BLOB_CONNECTION_STRING is not set.")
        return

    try:
        queue_message = {
            "userId": user_id,
            "cost": cost,
            "chargeId": charge_id,
            "thread_id": thread_id,
            "model": model_name
        }

        # Base64 encoding required for Azure Queue Storage
        message_string = json.dumps(queue_message)
        message_bytes = message_string.encode('utf-8')
        base64_bytes = base64.b64encode(message_bytes)
        base64_message = base64_bytes.decode('utf-8')

        queue_client = QueueClient.from_connection_string(
            conn_str=CONNECTION_STRING,
            queue_name=QUEUE_NAME
        )

        async with queue_client:
            # Create queue if it doesn't exist (safety check for local dev)
            try:
                await queue_client.create_queue()
            except Exception:
                pass # Queue likely exists
            
            await queue_client.send_message(base64_message)

        logger.info(f"Queued deduction {charge_id} for user {user_id}: ${cost}")

    except Exception as e:
        logger.error(f"Failed to queue credit deduction: {e}", exc_info=True)