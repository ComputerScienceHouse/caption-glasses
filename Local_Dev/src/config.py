import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger: logging.Logger = logging.getLogger(__name__)

def _get_env_variable(name: str, default: str | None = None) -> str | None:

    """
    Retrieves an environment variable, with an optional default value.

    Args:
            name (str): The name of the environment variable to retrieve.
            default (str | None): An optional default value to return if the environment variable is not set.

    Returns:
            str | None: The value of the environment variable, or the default value if it is not set.
    """

    try:
        value: str = os.getenv(name, default)

        if value in (None, ""):
            logger.warning(
                f"Environment variable '{name}' is not set, using default value: '{default if default is not None else 'None'}'"
            )
            return default

        return value
    except Exception as e:
        logger.error(f"Error retrieving environment variable '{name}': {e}")
        return default
    
DEVICE_CAPTURE_RATE:int = int(_get_env_variable("DEVICE_CAPTURE_RATE","44100")) # Change this to your microphones rates
WEBSOCKET_URI:str = _get_env_variable("WEBSOCKET_URI","ws://localhost:2001/ws") # Change this to your server's WebSocket URI