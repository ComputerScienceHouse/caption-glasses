import os
import json
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


BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

HF_TOKEN: str = _get_env_variable("HF_TOKEN", "default")

PHRASE_TIMEOUT: float = float(
    _get_env_variable("PHRASE_TIMEOUT", 0.6)
)  # Silence gap (sec) to trigger final transcription.
MAX_DURATION: float = float(
    _get_env_variable("MAX_DURATION", 3.0)
)  # Max duration before forcing a final result
VAD_THRESHOLD: float = float(
    _get_env_variable("VAD_THRESHOLD", 0.4)
)  # VAD sensitivity (lower = more sensitive)

SAMPLE_RATE = 16000
CHUNK_SIZE = 2048
