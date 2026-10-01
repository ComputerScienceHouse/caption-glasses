import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger: logging.Logger = logging.getLogger(__name__)

def get_env_variable(name: str, default: str | None = None) -> str | None:
    try:
        value: str = os.getenv(name, default)

        if value in (None, ""):
            logger.warning(
                f"Environment variable '{name}' not set, using default value: '{default if default is not None else 'None'}'"
            )
            return default

        return value
    except Exception as e:
        logger.error(f"Error retrieving environment variable '{name}': {e}")
        return default

def env_float(name: str, default: str) -> float:
    return float(get_env_variable(name, default))

def env_int(name: str, default: str) -> int:
    return int(get_env_variable(name, default))

def env_bool(name: str, default: str) -> bool:
    v = get_env_variable(name, default)
    if v is None:
        return default.lower() in ("1", "true", "yes")
    return str(v).lower() in ("1", "true", "yes")