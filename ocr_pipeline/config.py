from enum import Enum
from typing import Callable, Literal

class OCRModel(Enum):
    """Supported OCR models"""
    QWEN_VLM = "qwen_vlm"
    GOT_OCR = "got_ocr"


# Default configuration
DEFAULT_OCR_MODEL = OCRModel.QWEN_VLM
DEFAULT_OCR_TYPE = "format"
DEFAULT_LOAD_IN_4BIT = True


def get_ocr_images_function(model: Literal["qwen_vlm", "got_ocr"] = "qwen_vlm", load_in_4bit: bool = DEFAULT_LOAD_IN_4BIT) -> Callable:
    """
    Get the appropriate OCR function based on the model selection
    """
    if model == "qwen_vlm":
        from .qwen_vlm_ocr import ocr_images as qwen_ocr
        return lambda images, ocr_type="format": qwen_ocr(images, ocr_type=ocr_type, load_in_4bit=load_in_4bit)
    elif model == "got_ocr":
        from .ocr_inference import ocr_images as got_ocr
        return got_ocr
    else:
        raise ValueError(f"Unsupported OCR model: {model}")