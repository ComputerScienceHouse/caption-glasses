from client.state import state
from client.tuning_specs import ALL_SLIDER_SPECS

def slider_stored_value(spec, value):
    if spec["fmt"] == "d":
        return int(round(float(value)))
    return float(value)

def apply_settings_from_server(settings):
    if not isinstance(settings, dict):
        return
    for spec in ALL_SLIDER_SPECS:
        k = spec["key"]
        if k in settings and settings[k] is not None:
            state["slider_values"][k] = slider_stored_value(spec, settings[k])
    state["singing_mode"] = settings.get("content_mode") == "lyrics"

def settings_payload():
    payload = {
        s["key"]: slider_stored_value(s, state["slider_values"][s["key"]])
        for s in ALL_SLIDER_SPECS
    }
    if state["singing_mode"]:
        payload.update(content_mode="lyrics", skip_denoise=True, whisper_vad_filter=False)
    else:
        payload.update(content_mode="speech", skip_denoise=False, whisper_vad_filter=True)
    return payload

def reset_speech_settings():
    from client.ws_client import ws_send

    state["singing_mode"] = False
    ws_send({"type": "set_mode", "value": "balanced"})

def reset_lyrics_settings():
    from client.ws_client import ws_send

    state["singing_mode"] = True
    ws_send({"type": "set_mode", "value": "lyrics"})

def update_telemetry(msg):
    tel = state["telemetry"]
    if msg.get("env") is not None:
        tel["env"] = msg["env"]
    if msg.get("noise_score") is not None:
        tel["noise_score"] = float(msg["noise_score"])
    if msg.get("speech_prob") is not None:
        tel["speech_prob"] = float(msg["speech_prob"])
    if msg.get("input_rms") is not None:
        tel["input_rms"] = float(msg["input_rms"])
    if msg.get("capturing") is not None:
        tel["capturing"] = bool(msg["capturing"])