from pathlib import Path

import cv2
import numpy as np
from deskew import determine_skew
from pdf2image import convert_from_path
from PIL import Image


def pdf_to_images(pdf_path: str, dpi: int = 300) -> list[Image.Image]:
    pages = convert_from_path(pdf_path, dpi=dpi, fmt="png")
    return pages


def tiff_to_images(tiff_path: str) -> list[Image.Image]:
    import tifffile
    with tifffile.TiffFile(tiff_path) as tif:
        pages = [Image.fromarray(page.asarray()) for page in tif.pages]
    return pages


def preprocess_image(pil_img: Image.Image) -> Image.Image:
    img = np.array(pil_img.convert("RGB"))

    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    angle = determine_skew(grey)
    if abs(angle) > 0.3:
        (h, w) = grey.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        grey = cv2.warpAffine(
            grey, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    grey = cv2.fastNlMeansDenoising(grey, h=10)

    binary = cv2.adaptiveThreshold(
        grey, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10,
    )

    return Image.fromarray(binary)


def load_input(path: str) -> list[Image.Image]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return pdf_to_images(str(p), dpi=300)
    elif suffix in (".tif", ".tiff"):
        return tiff_to_images(str(p))
    else:
        raise ValueError("Input must be PDF or TIFF")


def preprocess_images(images: list[Image.Image]) -> list[Image.Image]:
    return [preprocess_image(img) for img in images]
