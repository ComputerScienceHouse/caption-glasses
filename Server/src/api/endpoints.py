from fastapi import APIRouter, WebSocket
from core import processing

router: APIRouter = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Connects a websocket to start processing tunneled data from the websocket

    Arguments:
        websocket (Websocket): The websocket that is connected
    """
    await websocket.accept()


    await processing.create_connection(websocket)
