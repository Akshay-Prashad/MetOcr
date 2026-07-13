"""
Preprocessing: split the two-page TIFF scan into the left observation table,
then slice that into row-bands (a handful of rows per crop) so a small VLM
on an 8GB card can actually resolve individual handwritten digits instead
of losing them in a full 31-row table.

Preprocessing chain: deskew (page curvature/binding skew), CLAHE contrast
enhancement (local contrast lift for faint pencil/ink on aged paper),
sharpen, and upscale before slicing. Row slicing uses uniform spacing
(h / n_rows) -- no gridline detection here, kept deliberately simple so
we can isolate how much preprocessing alone changes model performance
before adding more moving parts.

Tune CROP_* constants by eye once for your scanner/form layout -- they don't
need to be perfect, just consistent across a batch of sheets from the same
register (same printed form layout repeats across the whole volume).

Calibration note: bottom_frac must end right after the last data row (row 31)
and before the "Sums" row -- including Sums/Means/footer rows in the crop
height throws off row_h = h / n_rows and causes progressive row misalignment
that gets worse toward the bottom of the page. Verified bottom_frac=0.70 for
MO-9_1_029_c.tif's layout; recalibrate per scan batch if it differs.
"""
import os
import numpy as np
import cv2
from PIL import Image, ImageEnhance

Image.MAX_IMAGE_PIXELS = None  # large scans are expected


def load_page(tif_path: str) -> Image.Image:
    return Image.open(tif_path).convert("RGB")


def split_spread(im: Image.Image, left_frac: float = 0.5):
    """Split a two-page spread into left and right page images."""
    w, h = im.size
    split_x = int(w * left_frac)
    left = im.crop((0, 0, split_x, h))
    right = im.crop((split_x, 0, w, h))
    return left, right


def deskew(im: Image.Image, max_angle: float = 5.0) -> Image.Image:
    """
    Correct small rotational skew from scanning/binding curvature.
    Estimates skew from the dominant horizontal ruled lines on the form
    (these registers have strong printed rule lines, which give a much
    more reliable skew signal than the handwriting itself).
    """
    gray = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=200,
                             minLineLength=gray.shape[1] // 3, maxLineGap=20)
    if lines is None:
        return im  # no confident line signal, skip rather than risk a bad rotation

    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < max_angle:  # ignore near-vertical lines (column rules)
            angles.append(angle)

    if not angles:
        return im

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.1:
        return im  # not worth the resample

    return im.rotate(median_angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))


def enhance_contrast(im: Image.Image, clip_limit: float = 2.0) -> Image.Image:
    """
    CLAHE (local contrast enhancement) to lift faint pencil/ink strokes off
    an aged/uneven paper background, without blowing out already-dark ink.
    Applied on the L channel only so colors (if any annotations use colored
    ink) aren't distorted.
    """
    arr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(arr)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    arr = cv2.merge((l, a, b))
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_LAB2RGB))


def sharpen(im: Image.Image, factor: float = 1.5) -> Image.Image:
    """Mild unsharp-mask style sharpening to crisp up thin pencil strokes."""
    return ImageEnhance.Sharpness(im).enhance(factor)


def upscale(im: Image.Image, min_row_px: int = 60, row_h_px: float = None) -> Image.Image:
    """
    Upscale so each printed row is at least `min_row_px` tall in the final
    crop. VLM vision encoders downsample internally regardless of input
    size, but a too-small source image means the digit is already lost
    before the model sees it -- upscaling low-res scans gives the encoder
    more real signal to work with (this doesn't invent detail, but it
    avoids extra loss from aggressive downstream resizing).
    """
    if row_h_px is None or row_h_px >= min_row_px:
        return im
    scale = min_row_px / row_h_px
    w, h = im.size
    return im.resize((int(w * scale), int(h * scale)), resample=Image.LANCZOS)


def crop_table_region(page_im: Image.Image, top_frac: float, bottom_frac: float,
                       left_frac: float = 0.0, right_frac: float = 1.0) -> Image.Image:
    """Crop out the data-table region, dropping headers/margins/notes."""
    w, h = page_im.size
    box = (
        int(w * left_frac),
        int(h * top_frac),
        int(w * right_frac),
        int(h * bottom_frac),
    )
    return page_im.crop(box)


def slice_row_bands(table_im: Image.Image, n_rows: int, rows_per_band: int = 6,
                     overlap_rows: int = 1):
    """
    Slice a cropped table image into horizontal bands of `rows_per_band` data
    rows each, with a 1-row overlap so digits split across a band boundary
    aren't cut in half. Assumes rows are evenly spaced top-to-bottom in the
    crop -- this holds as long as top_frac/bottom_frac tightly bound the
    actual data rows (see calibration note at top of file).

    Returns list of (band_image, first_row_idx, last_row_idx) -- 1-indexed
    row numbers matching "day of month".
    """
    w, h = table_im.size
    row_h = h / n_rows
    bands = []
    row = 1
    while row <= n_rows:
        last_row = min(row + rows_per_band - 1, n_rows)
        top = max(0, int((row - 1 - overlap_rows) * row_h)) if row > 1 else 0
        bottom = min(h, int((last_row + overlap_rows) * row_h))
        band_im = table_im.crop((0, top, w, bottom))
        bands.append((band_im, row, last_row))
        row = last_row + 1
    return bands


def preprocess_page(page_im: Image.Image, do_deskew: bool = True,
                     do_contrast: bool = True, do_sharpen: bool = True) -> Image.Image:
    """Apply the preprocessing chain in the order that matters: deskew
    first (geometry), then contrast/sharpen (pixel values), so contrast
    enhancement isn't wasted on a misaligned image."""
    im = page_im
    if do_deskew:
        im = deskew(im)
    if do_contrast:
        im = enhance_contrast(im)
    if do_sharpen:
        im = sharpen(im)
    return im


def prepare_crops(tif_path: str, out_dir: str, rows_per_band: int = 6,
                   preprocess: bool = True, min_row_px: int = 60):
    """
    Full pipeline: load -> split spread -> preprocess (deskew/contrast/sharpen)
    -> crop left-page table region -> upscale if needed -> slice into
    row-bands -> save to disk.
    """
    os.makedirs(out_dir, exist_ok=True)
    im = load_page(tif_path)
    left_page, right_page = split_spread(im)

    if preprocess:
        left_page = preprocess_page(left_page)

    # Calibrated against MO-9_1_029_c.tif's layout: table data starts below
    # the column-header block and ends right after row 31, before "Sums".
    table = crop_table_region(left_page, top_frac=0.17, bottom_frac=0.70)

    n_rows = 31
    row_h_px = table.size[1] / n_rows
    table = upscale(table, min_row_px=min_row_px, row_h_px=row_h_px)

    bands = slice_row_bands(table, n_rows=n_rows, rows_per_band=rows_per_band)
    paths = []
    base = os.path.splitext(os.path.basename(tif_path))[0]
    for band_im, first_row, last_row in bands:
        fname = f"{base}_rows{first_row:02d}-{last_row:02d}.png"
        fpath = os.path.join(out_dir, fname)
        band_im.save(fpath)
        paths.append((fpath, first_row, last_row))
    return paths


if __name__ == "__main__":
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tif = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, "MO-9_1_029_c.tif")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(script_dir, "crops")
    crops = prepare_crops(tif, out)
    for p, f, l in crops:
        print(f"rows {f:02d}-{l:02d} -> {p}")
