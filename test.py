from crop_utils import load_page, split_spread, crop_table_region

im = load_page("MO-9_1_029_c.tif")
left_page, _ = split_spread(im)
w, h = left_page.size
print("page size:", w, h)

# try progressively tighter bottom_frac values and save crops to inspect
for bf in [0.865, 0.80, 0.75, 0.70]:
    table = crop_table_region(left_page, top_frac=0.17, bottom_frac=bf)
    table.save(f"table_bottom_{bf}.png")
    print(bf, "-> size", table.size)
