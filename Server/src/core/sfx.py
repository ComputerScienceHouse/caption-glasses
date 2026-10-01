import time

from config import SFX_HYSTERESIS_WINDOWS, SFX_KEEPALIVE_SEC, SFX_LABEL_TTL_SEC


class SfxTracker:
    __slots__ = ("active", "pending_counts", "last_sent_key", "last_sent_monotonic")

    def __init__(self) -> None:
        self.active: dict[str, dict] = {}
        self.pending_counts: dict[str, int] = {}
        self.last_sent_key: tuple = ()
        self.last_sent_monotonic: float = 0.0

    def update(
        self,
        detections: list[dict] | None,
        *,
        max_labels: int = 4,
        now: float | None = None,
    ) -> list[dict] | None:
        if now is None:
            now = time.monotonic()

        seen_cats: set[str] = set()
        for det in detections or []:
            cat = str(det.get("category") or "ambient")
            seen_cats.add(cat)
            transient = bool(det.get("transient"))
            count = self.pending_counts.get(cat, 0) + 1
            if transient or cat in self.active or count >= SFX_HYSTERESIS_WINDOWS:
                self.active[cat] = {
                    "label": str(det.get("label", "")),
                    "category": cat,
                    "score": float(det.get("score", 0.0)),
                    "expires_at": now + SFX_LABEL_TTL_SEC,
                }
                self.pending_counts.pop(cat, None)
            else:
                self.pending_counts[cat] = count

        for cat in list(self.pending_counts):
            if cat not in seen_cats:
                del self.pending_counts[cat]

        for cat in list(self.active):
            if self.active[cat]["expires_at"] <= now:
                del self.active[cat]

        display = sorted(
            self.active.values(), key=lambda d: d["score"], reverse=True
        )[: max(1, int(max_labels))]
        if not display:
            self.last_sent_key = ()
            return None

        key = tuple((d["label"], d["category"]) for d in display)
        if key == self.last_sent_key and (
            now - self.last_sent_monotonic
        ) < SFX_KEEPALIVE_SEC:
            return None

        self.last_sent_key = key
        self.last_sent_monotonic = now
        return [
            {
                "label": d["label"],
                "category": d["category"],
                "score": round(float(d["score"]), 3),
            }
            for d in display
        ]
