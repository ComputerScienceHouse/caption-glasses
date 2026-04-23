"""
Patches for both huggingface and pyannote to ensure they still work
"""

from logging import Logger, getLogger

import huggingface_hub
import pyannote.audio.core.model as _pyannote_model

import torch

logger: Logger = getLogger(__name__)


def _patch_auth_token(fn):
    """
    Takes in a function to replace auth tokens

    Arguments:
        fn (function): The function to be patched

    Returns:
        function: The function, now patched from use_auth_token to token
    """

    def patched(*args, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token", None)  # or remap to "token"
        return fn(*args, **kwargs)

    return patched


logger.info("Starting Patches for Auth Token")
huggingface_hub.hf_hub_download = _patch_auth_token(huggingface_hub.hf_hub_download)
_pyannote_model.hf_hub_download = _patch_auth_token(_pyannote_model.hf_hub_download)
if hasattr(huggingface_hub, "cached_download"):  # check and see if it was cached
    huggingface_hub.cached_download = _patch_auth_token(huggingface_hub.cached_download)

# How the thing above worked before this? God only knows


logger.info("Starting Patches for Torch Loading")
_old_torch_load: function = torch.load


def _patched_torch_load(*args, **kwargs):
    """
    Forces torch to not only use weights
    """
    kwargs["weights_only"] = False
    return _old_torch_load(*args, **kwargs)


torch.load = _patched_torch_load
