from schema import LEFT_PAGE_SCHEMA, SCHEMA_JSON_DESCRIPTION

SYSTEM_PROMPT = """You are transcribing a row-band crop from a British Met Office \
handwritten weather observation register, dated 1905. The image shows several \
consecutive numbered days (leftmost column = "Day of the month") from a printed \
ruled form, with column headers visible at the top of the crop for reference.

Transcribe ONLY the handwritten/printed data cells. Do not guess values that are \
not visibly written -- use null for blank cells. Preserve the exact written form \
for fields marked str in the schema (e.g. wind force may be a single number or a \
range like "9 to 10"; cloud form is abbreviated, e.g. "A.S.", "Ci K.", "S K n."). \
For fields marked float or int, return a bare number with no units or punctuation \
beyond a decimal point.

Every day in the requested range corresponds to exactly one printed row on the \
form, even if that row is entirely blank or illegible. Never skip, merge, or omit \
a row -- a missing row here will misalign every day after it.
"""

def build_user_prompt(first_row: int, last_row: int) -> str:
    n_days = last_row - first_row + 1
    return f"""This crop contains days {first_row} to {last_row} of the month \
({n_days} days total).

For EACH day in that range, return one JSON object with these fields:
{SCHEMA_JSON_DESCRIPTION}

You MUST return exactly {n_days} objects, one per day, in day order, even if a \
day's row is entirely blank -- in that case use null for every field in that \
object. Do not skip a day just because it is hard to read.

Return ONLY a JSON array of {n_days} objects. No prose, no markdown fences. \
If a field's value is illegible or blank, use null for that field. If you are \
uncertain but can make a best reading, fill the value and add a sibling field \
"<field>_confidence": "low" for that field only.
"""
