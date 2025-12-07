
# --- Helper for timezone offsets (UTC-based) ---
TIMEZONE_OFFSETS = {
    "UTC": 0,
    "GMT": 0,

    # North America
    "EST": -5,  # Eastern Standard Time
    "EDT": -4,  # Eastern Daylight Time
    "CST": -6,  # Central Standard Time
    "CDT": -5,  # Central Daylight Time
    "MST": -7,  # Mountain Standard Time
    "MDT": -6,  # Mountain Daylight Time
    "PST": -8,  # Pacific Standard Time
    "PDT": -7,  # Pacific Daylight Time
    "AKST": -9,  # Alaska Standard Time
    "AKDT": -8,  # Alaska Daylight Time

    # South America
    "ART": -3,  # Argentina Time
    "BRT": -3,  # Brasilia Time
    "CLT": -4,  # Chile Standard Time
    "CLST": -3, # Chile Summer Time

    # Europe
    "CET": 1,   # Central European Time
    "CEST": 2,  # Central European Summer Time
    "EET": 2,   # Eastern European Time
    "EEST": 3,  # Eastern European Summer Time
    "WET": 0,   # Western European Time
    "WEST": 1,  # Western European Summer Time
    "MSK": 3,   # Moscow Time

    # Africa
    "WAT": 1,   # West Africa Time
    "CAT": 2,   # Central Africa Time
    "EAT": 3,   # East Africa Time

    # Asia
    "IST": 5.5,  # India Standard Time
    "PKT": 5,    # Pakistan Standard Time
    "BST": 6,    # Bangladesh Standard Time
    "ICT": 7,    # Indochina Time
    "CST-Asia": 8,  # China Standard Time
    "JST": 9,    # Japan Standard Time
    "KST": 9,    # Korea Standard Time

    # Australia & Oceania
    "AWST": 8,   # Australian Western Standard Time
    "ACST": 9.5, # Australian Central Standard Time
    "ACDT": 10.5,# Australian Central Daylight Time1
    "AEST": 10,  # Australian Eastern Standard Time
    "AEDT": 11,  # Australian Eastern Daylight Time
    "NZST": 12,  # New Zealand Standard Time
    "NZDT": 13,  # New Zealand Daylight Time

    # Other
    "AST": -4,   # Atlantic Standard Time
    "ADT": -3,   # Atlantic Daylight Time
    "CHAST": 12.75, # Chatham Standard Time
    "CHADT": 13.75, # Chatham Daylight Time
}