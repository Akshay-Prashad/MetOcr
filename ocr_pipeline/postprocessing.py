import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import re

logger = logging.getLogger(__name__)

def postprocess_markdown(column_data: List[Dict[str, List[str]]]) -> pd.DataFrame:
    """
    Merges column-by-column extraction results into a single DataFrame.
    
    Args:
        column_data: List of dicts (one per page). 
                     Each dict: {col_name: [31 values]}
    
    Returns:
        DataFrame with rows as days (1-31) and columns as extracted fields.
    """
    if not column_data:
        return pd.DataFrame()

    page_results = column_data[0]
    
    # Initialize data for 31 days
    data = []
    for day in range(1, 32):
        day_row = {"Day": day}
        
        # Extract each defined column
        for col_name, values in page_results.items():
            try:
                val = values[day-1] if len(values) >= day else ""
            except (IndexError, TypeError):
                val = ""
                
            val = str(val).strip()
            if val in ["-", "null", "nan", "None", ""]:
                val = None
            
            # Type casting for specific columns with validation
            val = _clean_value(col_name, val)
            day_row[col_name] = val
            
        data.append(day_row)
        
    return pd.DataFrame(data)

def _clean_value(col_name: str, val: Any) -> Optional[Any]:
    if val is None:
        return None
    val = str(val).strip()
    if val in ["-", "null", "nan", "None", ""]:
        return None
    
    # Numeric columns
    if col_name in ["attached_thermometer", "dry_bulb", "wet_bulb", "wind_force", "cloud_amount"]:
        return _to_int(val)
    elif col_name in ["barometer_uncorrected", "barometer_corrected"]:
        numeric = _to_float(val)
        return numeric
    
    # Wind direction - compass abbreviations
    if col_name == "wind_direction":
        # Normalize common abbreviations
        abbreviations = {
            "N": "N", "NNE": "NNE", "NE": "NE", "ENE": "ENE",
            "E": "E", "ESE": "ESE", "SE": "SE", "SSE": "SSE",
            "S": "S", "SSW": "SSW", "SW": "SW", "WSW": "WSW",
            "W": "W", "WNW": "WNW", "NW": "NW", "NNW": "NNW",
        }
        return abbreviations.get(val.upper(), val)
    
    return val

def _to_int(val: Any) -> Optional[int]:
    if val is None: return None
    match = re.search(r'(-?\d+)', str(val))
    return int(match.group(1)) if match else None

def _to_float(val: Any) -> Optional[float]:
    if val is None: return None
    match = re.search(r'(-?\d*\.?\d+)', str(val))
    return float(match.group(1)) if match else None