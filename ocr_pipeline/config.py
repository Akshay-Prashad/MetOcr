from enum import Enum
from typing import Callable, Literal, Optional

class OCRModel(Enum):
    QWEN_VLM = "qwen_vlm"

DEFAULT_OCR_MODEL = OCRModel.QWEN_VLM
DEFAULT_LOAD_IN_4BIT = True


def get_ocr_images_function(
    model: Literal["qwen_vlm"] = "qwen_vlm",
    load_in_4bit: bool = DEFAULT_LOAD_IN_4BIT,
    adapter_path: Optional[str] = None,
    model_id: Optional[str] = None,
    force_cpu: bool = False,
) -> Callable:
    if model == "qwen_vlm":
        from .qwen_vlm_ocr import ocr_images as qwen_ocr, DEFAULT_MODEL_ID
        mid = model_id or DEFAULT_MODEL_ID
        return lambda images, ocr_type="format": qwen_ocr(
            images, load_in_4bit=load_in_4bit, adapter_path=adapter_path,
            model_id=mid, force_cpu=force_cpu,
        )
    else:
        raise ValueError(f"Unsupported OCR model: {model}")
