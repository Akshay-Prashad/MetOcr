from schema import LEFT_PAGE_SCHEMA, SCHEMA_JSON_DESCRIPTION

SYSTEM_PROMPT = """You are transcribing a row-band crop from a British Met Office \
handwritten weather observation register, dated 1905. The image shows several \
consecutive numbered days (leftmost column = "Day of the month") from a printed \
ruled form, with column headers visible at the top of the crop for reference.

Transcribe ONLY the handwritten/printed data cells. Do not guess values that are \
not visibly written -- use null for blank cells. Preserve the exact written form \
for fields marked str (e.g. wind force may be a single number or a range like \
"9 to 10"; cloud form is abbreviated, e.g. "A.S.", "Ci K.", "S K n.").
"""

def build_user_prompt(first_row: int, last_row: int) -> str:
    return f"""This crop contains days {first_row} to {last_row} of the month.

For EACH day in that range, return one JSON object with these fields:
{SCHEMA_JSON_DESCRIPTION}

Return ONLY a JSON array of objects, one per day, in day order. No prose, no \
markdown fences. If a row's value for a field is illegible or blank, use null. \
If you are uncertain but can make a best reading, fill the value and add a \
sibling field "<field>_confidence": "low" for that field only.
"""
