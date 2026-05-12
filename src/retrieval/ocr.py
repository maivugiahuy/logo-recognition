"""OCR text extraction and text-visual score fusion for logo recognition."""
import re

import numpy as np
from PIL import Image
from Levenshtein import distance as levenshtein

_easyocr_reader = None
_paddle_reader = None


def _get_easyocr(gpu: bool = True):
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
    return _easyocr_reader


def _get_paddle(gpu: bool = True):
    global _paddle_reader
    if _paddle_reader is None:
        from paddleocr import PaddleOCR
        _paddle_reader = PaddleOCR(use_angle_cls=False, lang="en",
                                   use_gpu=gpu, show_log=False)
    return _paddle_reader


def run_ocr(crop: Image.Image, gpu: bool = True, backend: str = "easyocr") -> str:
    """Return concatenated OCR text from crop, lowercased. Empty string if no text found."""
    arr = np.array(crop)
    if backend == "paddle":
        reader = _get_paddle(gpu)
        result = reader.ocr(arr, cls=False)
        if not result or not result[0]:
            return ""
        texts = [line[1][0] for line in result[0] if line and line[1]]
        return " ".join(texts).lower().strip()
    else:  # easyocr
        reader = _get_easyocr(gpu)
        results = reader.readtext(arr, detail=0)
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
