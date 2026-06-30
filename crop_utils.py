"""
Preprocessing: split the two-page TIFF scan into the left observation table,
then slice that into row-bands (a handful of rows per crop) so a small VLM
on an 8GB card can actually resolve individual handwritten digits instead
of losing them in a full 31-row table.

Tune CROP_* constants by eye once for your scanner/form layout -- they don't
need to be perfect, just consistent across a batch of sheets from the same
register (same printed form layout repeats across the whole volume).
"""

import os
from PIL import Image

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
    crop, which holds for these printed forms.

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


def prepare_crops(tif_path: str, out_dir: str, rows_per_band: int = 6):
    """
    Full pipeline: load -> split spread -> crop left-page table region ->
    slice into row-bands -> save to disk. Adjust top_frac/bottom_frac once
    by inspecting a sample page (see inspect_crop.py).
    """
    os.makedirs(out_dir, exist_ok=True)
    im = load_page(tif_path)
    left_page, right_page = split_spread(im)

    # These fractions are calibrated against MO-9_1_029_c.tif's layout:
    # table data starts below the column-header block and ends above
    # the "Sums" row. Recalibrate if a sheet's scan crop differs.
    table = crop_table_region(left_page, top_frac=0.17, bottom_frac=0.865)

    bands = slice_row_bands(table, n_rows=31, rows_per_band=rows_per_band)

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
