from __future__ import annotations
from logging import Logger, getLogger
logger: Logger = getLogger(__name__)

try:
    from better_profanity import profanity as _bp
    _HAS_BETTER_PROFANITY = True
except ImportError:
    _HAS_BETTER_PROFANITY = False
    _bp = None
    logger.warning(
        "better_profanity not installed; profanity filter will pass text through unchanged"
    )

def mask_caption_text(text: str, enabled: bool) -> str:
    if not enabled or not text:
        return text
    if not _HAS_BETTER_PROFANITY or _bp is None:
        return text
    return _bp.censor(text)