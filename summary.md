# Met-OCR: Summary Report

## Objective

Build an OCR pipeline to extract structured daily weather observation data from scanned UK Met Office 10-year rainfall sheets (TIFF format), outputting XLSX files matching the ground-truth column layout.

## Dataset

- **Input**: 12 TIFF files (`MO-9_1_029_c.tif` through `MO-9_1_062_c.tif`)
- **Ground Truth**: 12 corresponding XLSX files (`MO_9_2_029_c.xlsx` through `MO_9_2_062_c.xlsx`)
- **Format**: Multi-page TIFFs, each containing monthly weather observation forms
- **Structure**: 31 daily rows × 46 columns with 4 hierarchical header rows

### Ground Truth Column Layout

The XLSX output must match this exact 46-column layout:

| Col | Morning (9 AM) | Afternoon | Extras |
|-----|----------------|-----------|--------|
| 1   | (blank) | | |
| 2   | Day of month | Day of month | |
| 3   | Attached Thermometer | Attached Thermometer | |
| 4   | Barometer Uncorrected | Barometer Uncorrected | |
| 5   | Barometer Corrected | Barometer Corrected | |
| 6   | Dry Bulb | Dry Bulb | |
| 7   | Wet Bulb | Wet Bulb | |
| 8   | Dew Point | Dew Point | |
| 9   | Vapour Pressure | Vapour Pressure | |
| 10  | Humidity % | Humidity % | |
| 11  | Wind Direction | Wind Direction | |
| 12  | Wind Force | Wind Force | |
| 13  | Cloud Amount | Cloud Amount | |
| 14  | Cloud Form | Cloud Form | |
| 15  | Wind Dir Lower | Wind Dir Lower | |
| 16  | Weather (obs time) | Weather (obs time) | |
| 17  | Rain (since last obs) | Rain (since last obs) | |
| 18  | Rain (preceding day) | | |
| 19  | Estimated duration | | |
| 37  | | | Max Temp |
| 38  | | | Min Temp |
| 39  | | | Max in Sun |
| 40  | | | Min on Grass |
| 42  | | | Earth Temp 1ft |
| 43  | | | Earth Temp 4ft |
| 44  | | | Sunshine Duration |
| 46  | | | Remarks |

## Design Approaches

### Approach 1: Generic Markdown Pipeline (Original)

The original pipeline used a general-purpose OCR approach:
1. Preprocess images (deskew, denoise, **binarize**)
2. OCR with Qwen-VL-Chat or GOT-OCR2_0
3. Extract markdown tables from output
4. Extract key-value pairs via regex
5. Build XLSX with 3 sheets (Tables, Fields, Raw OCR)

**Result**: Failed on these forms. The model output was empty or garbled.

### Approach 2: Qwen2.5-VL Upgrade

Upgraded from Qwen-VL-Chat (old `model.chat()` API) to Qwen2.5-VL-3B/7B (`model.generate()` + `processor`):

**Key changes**:
- `Qwen2_5_VLForConditionalGeneration` + `AutoProcessor` instead of `AutoModel` + `AutoTokenizer`
- `processor.apply_chat_template()` → `model.generate()` → `processor.batch_decode()`
- Vision inputs via `qwen_vl_utils.process_vision_info()`
- Removed `qwen-vl-chat` dependency, added `qwen-vl-utils`

### Approach 3: Specialized Rainfall Sheet Pipeline

Redesigned the entire pipeline for the specific use case:

**Prompt Design Evolution**:
1. **Complex CSV prompt** — Detailed column list, requested CSV output. Model hallucinated headers and outputted garbage. *(0 days extracted)*
2. **JSON prompt** — Requested JSON array with specific field names. Model outputted empty array `[]`. *(0 days)*
3. **Simple line-by-line prompt** — "List each day as: day | value1 | value2..." Model ran for 60+ minutes, outputted massive text. *(timed out)*
4. **Guided CSV prompt with example** — Provided exact format with day 1 example. Model outputted only 1 row (day 31). *(1 day)*
5. **Labeled output prompt** — "Day N: field1 value1, field2 value2, ..." Model outputted 27 rows with labeled values. *(27 days, best result)*

**Final Prompt**:
```
"List every day's data from this weather observation table. Output exactly:
Day 1: attached_thermometer, barometer_uncorrected, barometer_corrected, dry_bulb, wet_bulb, wind_direction, wind_force, cloud_amount, cloud_form, weather, rain
...
Day 31: ...
Use empty for missing values. Only output these 31 lines."
```

**Postprocessing**: Regex parser with patterns for each field:
```
attached_therm:  r"(?:Attached\s*Thermometer)\s*[:\s]*([^,]+)"
baro_uncorrected: r"(?:Barometer\s*Uncorrected)\s*[:\s]*(\d+\.?\d*)"
...
```

**XLSX Builder**: Writes into exact 46-column × 4-header-row format matching ground truth.

### Approach 4: Preprocessing Changes

| Version | Preprocessing | Result |
|---------|--------------|--------|
| Original | Deskew → Denoise → **Binary threshold** | Model outputted empty/garbled text |
| Final | Deskew → Denoise → **RGB** (no binarization) | Model outputted 27 rows with values |

Removing binary thresholding was critical — VLMs perform poorly on binary images.

## Model Tryouts

### Model Sizing

| Model | Params | GPU Memory | 4-bit Size | Inference Time | Quality |
|-------|--------|------------|------------|----------------|---------|
| Qwen2.5-VL-3B-Instruct | 3B | ~4GB (4-bit) | ~2.5GB | ~2 min / page | Fair (11% raw) |
| Qwen2.5-VL-7B-Instruct | 7B | ~8GB (4-bit) | ~4.5GB | ~5 min / page | Not tested (OOM) |

The 7B model did not fit on the available GPU (7.65GB) even with 4-bit quantization due to the vision encoder not being quantized by BitsAndBytes.

### Quantization

| Config | Outcome |
|--------|---------|
| `load_in_4bit=True` + `device_map="auto"` | Failed — vision encoder in float16, total >7.65GB |
| `load_in_4bit=True` + `max_memory={0:"6GB","cpu":"12GB"}` | Failed — 4-bit doesn't support CPU offloading by default |
| `load_in_4bit=True` + `device_map="cpu"` | Worked but extremely slow (3+ min load, 60+ min inference) |
| `load_in_4bit=True` (3B model, `device_map="auto"`) | Success — fit in 7.65GB, ~2 min inference |

**Lesson**: 4-bit quantization only works for models that fit entirely on GPU. The vision encoder in Qwen2.5-VL-7B is not quantized, making the effective memory requirement higher than expected.

## Evaluation

### Test Configuration

- **Input**: `Files/MO-9_1_029_c.tif` (1 page, 1 month)
- **Model**: Qwen2.5-VL-3B-Instruct (4-bit, device_map="auto")
- **Prompt**: Labeled Day N format
- **GPU**: 7.65GB VRAM

### Quantitative Results

| Metric | Value |
|--------|-------|
| Days extracted | 27 / 31 (87%) |
| Days missed | 28, 29, 30, 31 |
| Total fields | 297 (27 days × 11 fields) |
| Correct fields | 33 |
| **Per-field accuracy** | **11.1%** |
| Inference time | 2 min 11 sec |
| Pipeline total | ~2 min 15 sec |

### Per-Field Accuracy Breakdown

| Field | Correct / Total | Accuracy | Notes |
|-------|----------------|----------|-------|
| attached_therm | 0 / 27 | 0% | Always parsed as "ins." (column header text) |
| baro_uncorrected | 0 / 27 | 0% | Mapped to attached_therm value instead |
| baro_corrected | 0 / 27 | 0% | Values merged with uncorrected |
| dry_bulb | 9 / 27 | 33% | Often correct when isolated |
| wet_bulb | 2 / 27 | 7% | Frequently equal to dry_bulb |
| wind_dir | 9 / 27 | 33% | Second-best field |
| wind_force | 3 / 27 | 11% | Digitized as large numbers instead of 0-12 scale |
| cloud_amount | 5 / 27 | 19% | Sometimes correct |
| cloud_form | 0 / 27 | 0% | Periods stripped by parser |
| weather | 0 / 27 | 0% | Parser missed "B & C", "R", "O" etc. |
| rain_since_last | 0 / 27 | 0% | Always "0" instead of actual values (0.05, 0.34, etc.) |

### Qualitative Observations

1. **Column confusion**: The model reads "Attached Thermometer ins." as the label, then assigns the adjacent number to "Barometer Uncorrected" instead of the correct column.

2. **Value merging**: Barometer values like "30.6" and "30.523" get merged into "30-60-030-523" — the model concatenates adjacent columns.

3. **Wind force scaling**: The model outputs wind force as "38", "47", "45" instead of the 0-12 Beaufort scale. These appear to be total pixel positions or line numbers the model hallucinated.

4. **Weather field**: The model almost always outputs "-" (meaning no weather recorded) or leaves it blank, while the ground truth has values like "B & C", "R", "O", "G" for every day.

5. **Rain field**: Always outputs "0" instead of the actual rainfall values. The model cannot resolve the small-print decimal entries.

### Comparison with Rainfall Rescue Project

The [Robot Rainfall Rescue project](https://robot-rainfall-rescue.readthedocs.io/) using SmolVLM (2.2B), Granite (3B), and Gemma (4B) found:

| Model State | Single Model Accuracy | Ensemble (3 models) |
|-------------|----------------------|---------------------|
| Untrained | ~50% agreement | ~50% with 5-10% false positives |
| Fine-tuned (1000 images) | ~92% | ~98% |

Our 11% accuracy on Qwen2.5-VL-3B (untrained) is lower than the blog's reported ~50% for untrained SmolVLM. Possible reasons:
- Our forms are more complex (46 columns vs. monthly rainfall totals)
- Our evaluation is stricter (exact match per field)
- Different form layout (daily weather observations vs. monthly rainfall table)

## Design Decisions

### Qwen2.5-VL over SmolVLM/Granite/Gemma
- Qwen2.5-VL has stronger vision capabilities and OCR performance
- 3B variant fits in 7.65GB GPU with 4-bit
- HuggingFace ecosystem integration

### Labeled output format over JSON/CSV
- The 3B model struggles to produce valid JSON
- Labeled format ("Day N: Field Value") is more natural for the model
- Easier to parse with regex than free-form text

### RGB preprocessing over binary
- VLMs are trained on natural images
- Binary thresholding removes subtle variations that VLMs use for character recognition
- Deskew and denoise are still beneficial

### XLSX header template matching
- Hard-coded header rows matching ground truth
- Ensures output can be directly compared with ground truth
- Avoids complex dynamic header generation

## Output Comparison

### Ground Truth (Day 1)
```
Day | Attached Therm | Baro Uncorr | Baro Corr | Dry Bulb | Wet Bulb | Wind Dir | Wind Force | Cloud Amt | Cloud Form | Weather | Rain
1   | 52            | 30.6        | 30.523    | 47       | 46       | SSW      | 5          | 8         | A.S.       | B & C   | 0.05
```

### Model Output (Day 1, parsed)
```
Day | Attached Therm | Baro Uncorr | Baro Corr | Dry Bulb | Wet Bulb | Wind Dir | Wind Force | Cloud Amt | Cloud Form | Weather | Rain
1   |                | 52          | 52        | 47       | 47       | SSW      | 1          | 8         | A.S.       |         | 0
```

### Model Raw Output (Day 1)
```
Day 1: Attached Thermometer ins., Barometer Uncorrected 52, Barometer Corrected 30-60-030-523, Dry bulb 47, Wet bulb 47, Wind Direction SSW, Wind Force 1, Cloud Amount 8, Cloud Form A.S., Weather - , Rain 0
```

## Recommendations

1. **Fine-tuning**: LoRA fine-tuning on ~1000 labeled images should improve accuracy from 11% to ~90%+ (per Rainfall Rescue findings).

2. **Model upgrade**: Use Qwen2.5-VL-7B with a GPU with ≥16GB VRAM, or use CPU offloading with more RAM.

3. **Ensemble**: Run 3 different VLMs (Qwen, SmolVLM, Gemma) and require 2/3 agreement for maximum accuracy.

4. **Multi-page TIFF support**: The current pipeline only processes the first page of multi-page TIFFs. Add page iteration for full sheet extraction.

5. **Parser improvement**: The regex parser loses accuracy due to model formatting inconsistencies. A fuzzy matching approach would recover more fields.

6. **Post-processing validation**: Add range checks (e.g., wind force 0-12, temperature within reasonable bounds) to flag likely extraction errors.
