import uuid

from logging import getLogger, Logger
from fastapi import APIRouter, WebSocket
import time

from starlette.websockets import WebSocketDisconnect

from core import processing

logger: Logger = getLogger(__name__)
router: APIRouter = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Connects a websocket to start processing tunneled data from the websocket

    Arguments:
        websocket (Websocket): The websocket that is connected
    """
    generated_uuid: str = uuid.uuid4()

    await websocket.accept()
    logger.info(
        f"Established connection with Client {generated_uuid} at %s", time.monotonic()
    )

    try:
        while True:
            raw_bytes: bytes = await websocket.receive_bytes()
            await processing.process_websocket_bytes(
                raw_bytes=raw_bytes, websocket=websocket
            )
    except WebSocketDisconnect as e:
        logger.info(
            f"Disconnected with Client {generated_uuid} at %s", time.monotonic()
        )
    except Exception as e:
        logger.error(f"Error: {type(e).__name__}: {e}")
