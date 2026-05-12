"""OCR text extraction and text-visual score fusion for logo recognition."""
import re

import numpy as np
from PIL import Image
from Levenshtein import distance as levenshtein

_reader = None


def _get_reader(gpu: bool = True):
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
    return _reader


def run_ocr(crop: Image.Image, gpu: bool = True) -> str:
    """Return concatenated OCR text from crop, lowercased. Empty string if no text found."""
    reader = _get_reader(gpu)
    results = reader.readtext(np.array(crop), detail=0)
    return " ".join(r.strip() for r in results).lower().strip()


def _normalize_label(label: str) -> str:
    label = re.sub(r"_text$", "", label)
    return re.sub(r"[_\-\s]+", "", label).lower()


def text_similarity(ocr_text: str, class_label: str) -> float:
    """Normalized Levenshtein similarity between OCR text and class label.

    Strips spaces/underscores from both sides so 'coca cola' matches 'coca_cola'.
    Returns 0.0 if OCR text is empty.
    """
    if not ocr_text:
        return 0.0
    a = re.sub(r"\s+", "", ocr_text.lower())
    b = _normalize_label(class_label)
    if not a or not b:
        return 0.0
    lev = levenshtein(a, b)
    return max(0.0, 1.0 - lev / max(len(a), len(b)))
