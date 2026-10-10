"""Kannada number words. Data only; the algorithm lives in normalize.py (schema: see hi.py)."""

# 0-19 are irregular; 20-99 = tens stem (ends in -ತ್ತ) + a sandhi-joined unit suffix.
_UNDER_20 = [
    "ಸೊನ್ನೆ", "ಒಂದು", "ಎರಡು", "ಮೂರು", "ನಾಲ್ಕು", "ಐದು", "ಆರು", "ಏಳು", "ಎಂಟು", "ಒಂಬತ್ತು",
    "ಹತ್ತು", "ಹನ್ನೊಂದು", "ಹನ್ನೆರಡು", "ಹದಿಮೂರು", "ಹದಿನಾಲ್ಕು", "ಹದಿನೈದು", "ಹದಿನಾರು",
    "ಹದಿನೇಳು", "ಹದಿನೆಂಟು", "ಹತ್ತೊಂಬತ್ತು",
]
_TENS = ["ಇಪ್ಪತ್ತು", "ಮೂವತ್ತು", "ನಲವತ್ತು", "ಐವತ್ತು", "ಅರವತ್ತು", "ಎಪ್ಪತ್ತು", "ಎಂಬತ್ತು", "ತೊಂಬತ್ತು"]
# unit joined to a tens stem (ಇಪ್ಪತ್ತ + ಒಂದು = ಇಪ್ಪತ್ತೊಂದು)
_JOIN = ["", "ೊಂದು", "ೆರಡು", "ಮೂರು", "ನಾಲ್ಕು", "ೈದು", "ಾರು", "ೇಳು", "ೆಂಟು", "ೊಂಬತ್ತು"]

UNITS = list(_UNDER_20)
for _t in _TENS:
    UNITS.append(_t)
    UNITS += [_t[:-1] + _j for _j in _JOIN[1:]]
assert len(UNITS) == 100 and len(set(UNITS)) == 100

DIGITS = UNITS[:10]

# Indian numbering. Scale words take an oblique form before more digits:
# ಸಾವಿರ -> ಸಾವಿರದ, ಲಕ್ಷ -> ಲಕ್ಷದ, ಕೋಟಿ -> ಕೋಟಿಯ. "ಒಂದು ಸಾವಿರ" is kept explicit.
SCALES = [
    (10**7, "ಕೋಟಿ", "ಕೋಟಿಯ"),
    (10**5, "ಲಕ್ಷ", "ಲಕ್ಷದ"),
    (10**3, "ಸಾವಿರ", "ಸಾವಿರದ"),
]

HUNDRED = "ನೂರು"
# 100..900 standalone / with a remainder (ನೂರು -> ನೂರ ಐವತ್ತು)
HUNDRED_ALONE = {1: "ನೂರು", 2: "ಇನ್ನೂರು", 3: "ಮುನ್ನೂರು", 4: "ನಾನೂರು", 5: "ಐನೂರು",
                 6: "ಆರುನೂರು", 7: "ಏಳುನೂರು", 8: "ಎಂಟುನೂರು", 9: "ಒಂಬೈನೂರು"}
HUNDRED_LINK = {1: "ನೂರ", 2: "ಇನ್ನೂರ", 3: "ಮುನ್ನೂರ", 4: "ನಾನೂರ", 5: "ಐನೂರ",
                6: "ಆರುನೂರ", 7: "ಏಳುನೂರ", 8: "ಎಂಟುನೂರ", 9: "ಒಂಬೈನೂರ"}
YEAR_HUNDREDS = False  # Kannada reads 1947 as "ಒಂದು ಸಾವಿರದ ಒಂಬೈನೂರ ನಲವತ್ತೇಳು"
LIMIT = 10**9
POINT = "ದಶಮಾಂಶ"
PERCENT = "ಪ್ರತಿಶತ"
PLUS = "ಪ್ಲಸ್"
RUPEES = "ರೂಪಾಯಿ"
PAISE = "ಪೈಸೆ"

MONTHS = [
    "ಜನವರಿ", "ಫೆಬ್ರವರಿ", "ಮಾರ್ಚ್", "ಏಪ್ರಿಲ್", "ಮೇ", "ಜೂನ್",
    "ಜುಲೈ", "ಆಗಸ್ಟ್", "ಸೆಪ್ಟೆಂಬರ್", "ಅಕ್ಟೋಬರ್", "ನವೆಂಬರ್", "ಡಿಸೆಂಬರ್",
]

CODE_WORDS = ["ಓಟಿಪಿ", "OTP", "otp", "ಕೋಡ್", "ಪಿನ್", "PIN", "pin", "ಪಾಸ್ವರ್ಡ್", "ಒಟಿಪಿ", "ಓಟೀಪೀ"]
CURRENCY_SYMBOLS = ["ರೂ."]
