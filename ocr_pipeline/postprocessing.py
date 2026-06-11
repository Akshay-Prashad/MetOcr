import logging
import re
from io import StringIO

import pandas as pd

logger = logging.getLogger(__name__)

SEPARATOR_RE = re.compile(r"^\| ?:?-{3,}:? ?(?:\| ?:?-{3,}:? ?)*\|$")
TABLE_RE = re.compile(
    r"(\|.+\|\n\| ?:?-{3,}:? ?(?:\| ?:?-{3,}:? ?)*\|\n(?:\|.+\|\n?)+)",
    re.MULTILINE,
)


def clean_markdown(md: str) -> str:
    md = re.sub(r"-{3,}", "", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = re.sub(r"[^\S\n]+", " ", md)
    md = md.strip()
    return md


def extract_tables_from_markdown(md_text: str) -> list[pd.DataFrame]:
    tables = []
    for match in TABLE_RE.finditer(md_text):
        raw = match.group(0)
        lines = [
            l for l in raw.strip().splitlines()
            if not SEPARATOR_RE.match(l)
        ]
        csv_lines = [
            re.sub(r"^\||\|$", "", l).replace("|", ",")
            for l in lines
        ]
        try:
            df = pd.read_csv(StringIO("\n".join(csv_lines)))
            tables.append(df)
        except Exception as e:
            logger.warning("Failed to parse table: %s", e)
            continue
    return tables


def extract_key_values(md_text: str) -> dict:
    kv = {}
    patterns = [
        r"^([A-Za-z][^.\n]{2,40}?)\s*\.{3,}\s*(.+)$",
        r"^([A-Za-z][^.:\n]{2,40}?)\s*[:\-–]\s*(.+)$",
        r"^(Month\s+and\s+Year)\s+(.+)$",
        r"^([A-Z][a-z]{2,})\s+(?=.*[0-9])(.+)$",
    ]
    for line in md_text.splitlines():
        line = line.strip()
        for pat in patterns:
            m = re.match(pat, line)
            if m:
                kv[m.group(1).strip()] = m.group(2).strip()
                break
    return kv


def postprocess_markdown(
    md_pages: list[str],
    tables_only: bool = False,
) -> tuple[list[pd.DataFrame], dict]:
    all_tables, all_kv = [], {}
    for md in md_pages:
        all_tables.extend(extract_tables_from_markdown(md))
        if not tables_only:
            md_clean = clean_markdown(md)
            all_kv.update(extract_key_values(md_clean))
    return all_tables, all_kv
