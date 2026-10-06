"""Append-only storage for frozen PRISM predictions."""

from __future__ import annotations

import json
from pathlib import Path

from .models import FrozenPrediction


class FrozenPredictionStore:
    """Write-once JSON records keyed by prediction_id."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def freeze(self, prediction: FrozenPrediction) -> Path:
        path = self.root / f"{prediction.prediction_id}.json"
        if path.exists():
            raise FileExistsError(
                f"Frozen prediction already exists and cannot be overwritten: {prediction.prediction_id}"
            )
        payload = prediction.as_dict()
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return path
