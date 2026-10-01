from __future__ import annotations
import inspect
import torch
import numpy as np
from numpy import ndarray
from faster_whisper import WhisperModel
from logging import Logger, getLogger

from config import (
    SAMPLE_RATE,
    WHISPER_COMPRESSION_RATIO_THRESHOLD,
    WHISPER_FINAL_CONDITION_PREVIOUS,
    WHISPER_LOG_PROB_THRESHOLD,
    WHISPER_MODEL,
    WHISPER_NO_SPEECH_NOISE_SHIFT,
    WHISPER_NO_SPEECH_THRESHOLD,
    WHISPER_PARTIAL_CONDITION_PREVIOUS,
    WHISPER_REPETITION_PENALTY,
    WHISPER_VAD_MIN_SILENCE_MS,
    WHISPER_VAD_SPEECH_PAD_MS,
    WHISPER_VAD_THRESHOLD,
    WHISPER_WITHOUT_TIMESTAMPS,
)
from core.session_settings import SessionSettings

logger: Logger = getLogger(__name__)

device: str = "cuda" if torch.cuda.is_available() else "cpu"
compute_type: str = "float16" if device == "cuda" else "int8"

logger.info(
    "Loading Whisper model %s on %s (%s).",
    WHISPER_MODEL,
    device.upper(),
    compute_type,
)
speech_model: WhisperModel = WhisperModel(
    WHISPER_MODEL,
    device=device,
    compute_type=compute_type,
)

logger.info("Loading VAD via torch hub")
vad_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
)
vad_model = vad_model.to("cpu")

_TRANSCRIBE_SUPPORTS_MULTILINGUAL = "multilingual" in inspect.signature(
    speech_model.transcribe
).parameters
_TRANSCRIBE_SUPPORTS_LID_THRESHOLD = "language_detection_threshold" in inspect.signature(
    speech_model.transcribe
).parameters

_TRANSLATE_LID_NON_EN_MIN_PROB = 0.08
_TRANSLATE_LID_NON_EN_STRONG_PROB = 0.12
_TRANSLATE_LID_SOFT_EN_MAX = 0.82


def _rms(audio: ndarray) -> float:
    if audio is None or len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))

def _filter_segments(segments, settings: SessionSettings, is_final: bool) -> str:
    min_lp = (
        settings.segment_min_logprob_final
        if is_final
        else settings.segment_min_logprob_partial
    )
    max_nsp = (
        settings.segment_max_no_speech_final
        if is_final
        else settings.segment_max_no_speech_partial
    )
    kept: list[str] = []
    for seg in segments:
        if seg.avg_logprob < min_lp:
            continue
        if seg.no_speech_prob > max_nsp:
            continue
        t = seg.text.strip()
        if t:
            kept.append(t)
    return " ".join(kept)

def _log_language_probs(info, *, task: str, is_final: bool) -> None:
    if info is None or task != "translate":
        return
    detected = getattr(info, "language", None)
    probability = getattr(info, "language_probability", None)
    all_probs = getattr(info, "all_language_probs", None)
    phase = "final" if is_final else "partial"
    if all_probs:
        ranked = ", ".join(f"{lang}={prob:.4f}" for lang, prob in all_probs)
        logger.info(
            "Whisper LID (%s): detected=%s (%.4f); all=[%s]",
            phase,
            detected,
            float(probability or 0.0),
            ranked,
        )
    elif detected:
        logger.info(
            "Whisper LID (%s): detected=%s (%.4f)",
            phase,
            detected,
            float(probability or 0.0),
        )

def _resolve_translate_source_language(
    info,
    *,
    source_lang_hint: str | None = None,
) -> str:
    hint = (source_lang_hint or "").strip().lower()[:2]
    if hint and hint != "en":
        return hint

    fallback = ""
    if info is not None and getattr(info, "language", None):
        fallback = str(info.language).lower()[:2]

    all_probs = getattr(info, "all_language_probs", None) if info is not None else None
    if not all_probs:
        return fallback

    top_lang, top_prob = all_probs[0]
    if top_lang != "en":
        return top_lang[:2]

    best_non_en_lang = ""
    best_non_en_prob = 0.0
    for lang, prob in all_probs[1:]:
        if lang == "en":
            continue
        if prob > best_non_en_prob:
            best_non_en_lang = lang
            best_non_en_prob = prob

    if not best_non_en_lang:
        return fallback or "en"

    if best_non_en_prob >= _TRANSLATE_LID_NON_EN_STRONG_PROB:
        return best_non_en_lang[:2]
    if (
        best_non_en_prob >= _TRANSLATE_LID_NON_EN_MIN_PROB
        and top_prob < _TRANSLATE_LID_SOFT_EN_MAX
    ):
        return best_non_en_lang[:2]
    return fallback or "en"

def get_speech(
    audio: ndarray,
    is_final: bool = True,
    task: str = "transcribe",
    noise_score: float = 0.0,
    settings: SessionSettings | None = None,
    fresh_context: bool = False,
    source_lang_hint: str | None = None,
) -> dict[str, str]:
    """
    Process audio and return transcribed text.
    """
    cfg = settings or SessionSettings()

    if len(audio) < 1600:
        return {"text": ""}

    min_rms = cfg.transcribe_min_rms
    if _rms(audio) < min_rms:
        return {"text": ""}

    if task == "translate":
        beam = 1
        use_previous = False
    elif is_final:
        beam = cfg.final_beam
        use_previous = (
            WHISPER_FINAL_CONDITION_PREVIOUS
            if task == "transcribe" and not fresh_context
            else False
        )
    else:
        beam = cfg.partial_beam
        use_previous = WHISPER_PARTIAL_CONDITION_PREVIOUS if task == "transcribe" else False

    ns = WHISPER_NO_SPEECH_THRESHOLD + WHISPER_NO_SPEECH_NOISE_SHIFT * max(
        0.0, min(1.0, noise_score)
    )
    log_prob = WHISPER_LOG_PROB_THRESHOLD
    compression = WHISPER_COMPRESSION_RATIO_THRESHOLD
    rep_penalty = WHISPER_REPETITION_PENALTY
    if cfg.content_mode == "lyrics":
        ns = float(min(0.92, max(0.55, ns - 0.04)))
        log_prob = -0.85
        compression = 1.35
    ns = float(min(0.99, max(0.5, ns)))

    use_whisper_vad = cfg.whisper_vad_filter and task != "translate"
    transcribe_kw: dict = dict(
        beam_size=int(beam),
        language=(
            "en"
            if task == "transcribe"
            else ((source_lang_hint or "").strip().lower()[:2] or None)
        ),
        task=task,
        condition_on_previous_text=use_previous,
        temperature=0.0,
        vad_filter=use_whisper_vad,
        no_speech_threshold=ns,
        log_prob_threshold=log_prob,
        compression_ratio_threshold=compression,
        repetition_penalty=rep_penalty,
    )
    if use_whisper_vad:
        transcribe_kw["vad_parameters"] = dict(
            min_silence_duration_ms=WHISPER_VAD_MIN_SILENCE_MS,
            speech_pad_ms=WHISPER_VAD_SPEECH_PAD_MS,
            onset=WHISPER_VAD_THRESHOLD,
            offset=max(0.0, WHISPER_VAD_THRESHOLD - 0.15),
        )
    if WHISPER_WITHOUT_TIMESTAMPS:
        transcribe_kw["without_timestamps"] = True
    if task == "translate":
        if _TRANSCRIBE_SUPPORTS_LID_THRESHOLD:
            transcribe_kw["language_detection_threshold"] = 0.35
        if _TRANSCRIBE_SUPPORTS_MULTILINGUAL and is_final:
            transcribe_kw["multilingual"] = True
        elif not _TRANSCRIBE_SUPPORTS_MULTILINGUAL and is_final:
            logger.warning(
                "faster-whisper does not support multilingual=True; upgrade to >=1.1.0 "
                "for per-segment language detection."
            )

    try:
        segments, info = speech_model.transcribe(audio, **transcribe_kw)
        text = _filter_segments(list(segments), cfg, is_final)
        _log_language_probs(info, task=task, is_final=is_final)
        lang = ""
        if task == "translate":
            lang = _resolve_translate_source_language(
                info,
                source_lang_hint=source_lang_hint,
            )
            raw = getattr(info, "language", None) if info is not None else None
            if lang and raw and str(raw).lower()[:2] != lang:
                logger.info(
                    "Whisper LID resolved source language: %s -> %s",
                    raw,
                    lang,
                )
        elif info is not None and getattr(info, "language", None):
            lang = str(info.language).lower()[:2]
        return {"text": text, "language": lang}
    except (ValueError, RuntimeError):
        return {"text": "", "language": ""}

def check_vad(audio: ndarray) -> float:
    with torch.no_grad():
        sub_chunks = torch.from_numpy(audio).to("cpu").split(512)
        max_prob: float = 0.0
        for sub in sub_chunks:
            if sub.shape[0] < 512:
                sub = torch.nn.functional.pad(sub, (0, 512 - sub.shape[0]))
            prob = vad_model(sub, SAMPLE_RATE).item()
            max_prob = max(max_prob, prob)
        return max_prob