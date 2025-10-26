import json
import os
from typing import Any, Dict, List

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFECTS_FILE = os.path.join(DATA_DIR, "defects_metadata.json")


def _read_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_dummy_defects(image_filename: str) -> List[Dict]:
    """
    Return list of defect dicts for an image filename from JSON; fallback to a default set.
    Expected defect format: {"type": str, "boundingBox": [x1,y1,x2,y2], "confidence": float}
    """
    data = _read_json(DEFECTS_FILE)
    if isinstance(data, dict):
        # try filename key first, then 'default'
        defects = data.get(image_filename) or data.get("default")
        if isinstance(defects, list) and defects:
            return defects
    # fallback default
    return [
        {"type": "Crack", "boundingBox": [120, 80, 165, 110], "confidence": 0.94},
        {"type": "Scratch", "boundingBox": [280, 150, 340, 165], "confidence": 0.87},
    ]

