"""
Column schema for the Met Office historical observation register,
matching the structure found in MO_9_2_029_c.xlsx (left page / 9 A.M. block).

Each field maps to: (xlsx_column_letter, human description, expected type)
Extend RIGHT_PAGE_SCHEMA for the second observation block / extra columns
once you start running this against the right-hand page too.
"""

LEFT_PAGE_SCHEMA = {
    "day":               ("B", "Day of the month", "int"),
    "attached_therm":    ("C", "Attached Thermometer", "float"),
    "baro_uncorrected":  ("D", "Barometer Uncorrected", "float"),
    "baro_corrected":    ("E", "Barometer corrected & reduced to 32F mean sea level", "float"),
    "temp_dry_bulb":     ("F", "Temperature Dry bulb", "float"),
    "temp_wet_bulb":     ("G", "Temperature Wet bulb", "float"),
    "dew_point":         ("H", "Dew Point", "float"),
    "vap_pressure":      ("I", "Vapour Pressure", "float"),
    "humidity_pct":      ("J", "Humidity Per Cent.", "float"),
    "wind_direction":    ("K", "Wind Direction", "str"),
    "wind_force":        ("L", "Wind Force (0-12)", "str"),   # sometimes a range e.g. "9 to 10"
    "cloud_amount":      ("M", "Cloud Amount (0-10)", "str"),
    "cloud_form":        ("N", "Cloud Form", "str"),
    "cloud_direction":   ("O", "Direction of lower stratum", "str"),
    "weather_at_obs":    ("P", "Weather at time of observation", "str"),
    "weather_since":     ("Q", "Weather since last observation", "str"),
    "rain_entered":      ("R", "Rain entered to preceding day", "float"),
    "rain_duration":     ("S", "Rain estimated duration", "str"),
    "earth_max":         ("AK", "Earth/extra Temperature Max", "float"),
    "earth_min":         ("AL", "Earth/extra Temperature Min", "float"),
    "sunshine_duration": ("AR", "Duration of Sunshine", "float"),
    "remarks":           ("AT", "Remarks", "str"),
}

# Row 18 in the xlsx = Day 1. Day N -> xlsx row (N + 17)
def day_to_row(day: int) -> int:
    return day + 17

SCHEMA_JSON_DESCRIPTION = "\n".join(
    f'  "{k}": {t}  # {desc}' for k, (_, desc, t) in LEFT_PAGE_SCHEMA.items()
)
