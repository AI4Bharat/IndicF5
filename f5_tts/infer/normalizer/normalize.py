"""Expand numbers into spoken words before the text reaches the TTS model.

IndicF5 has no numeral -> speech mapping, so digits come out as noise. See
NUMBER_NORMALIZATION_PLAN.md. One generic algorithm here; per-language word data lives in
one module per language (hi.py is the reference / template).
"""

import re
import unicodedata

from f5_tts.infer.normalizer import as_, bn, gu, hi, kn, ml, mr, or_, pa, ta, te

LANG_MODULES = {
    "hi": hi, "bn": bn, "as": as_, "gu": gu, "kn": kn, "ml": ml,
    "mr": mr, "or": or_, "pa": pa, "ta": ta, "te": te,
}
_NAMES = {
    "hindi": "hi", "bengali": "bn", "assamese": "as", "gujarati": "gu", "kannada": "kn",
    "malayalam": "ml", "marathi": "mr", "odia": "or", "punjabi": "pa", "tamil": "ta", "telugu": "te",
}

UNIVERSAL_CURRENCY = ["₹", "Rs.", "Rs", "INR"]

# 2,36,000 (Indian) | 1,234,567 (western) | 1500 ; optional fraction
_AMOUNT = r"(?:\d{1,3}(?:,\d{3})+(?!\d)|\d{1,3}(?:,\d{2})*,\d{3}(?!\d)|\d+)(?:\.\d+)?"

# ---------------------------------------------------------------- language detection

_SCRIPTS = {  # unicode block -> language code (ambiguous scripts resolved below)
    "hi": ("ऀ", "ॿ"),  # Devanagari: Hindi / Marathi
    "bn": ("ঀ", "৿"),  # Bengali script: Bengali / Assamese
    "pa": ("਀", "੿"),
    "gu": ("઀", "૿"),
    "or": ("଀", "୿"),
    "ta": ("஀", "௿"),
    "te": ("ఀ", "౿"),
    "kn": ("ಀ", "೿"),
    "ml": ("ഀ", "ൿ"),
}
_MARATHI_WORDS = {"आहे", "आहेत", "आणि", "नाही", "मला", "तुम्ही", "काय", "आपण", "होते", "आम्ही", "म्हणून", "पण"}
_HINDI_WORDS = {"है", "हैं", "और", "नहीं", "का", "की", "के", "में", "को", "से", "मैं", "आप", "हम", "था", "थी", "यह", "पर"}


def detect_lang(text):
    """Guess the language code from the script; None if the text has no Indic script."""
    counts = {code: 0 for code in _SCRIPTS}
    for c in text:
        for code, (lo, hi_) in _SCRIPTS.items():
            if lo <= c <= hi_:
                counts[code] += 1
                break
    code = max(counts, key=counts.get)
    if counts[code] == 0:
        return None
    if code == "hi":  # Hindi vs Marathi
        words = re.findall(r"[ऀ-ॿ]+", text)
        marathi = sum(w in _MARATHI_WORDS for w in words) + 2 * text.count("ळ")
        hindi = sum(w in _HINDI_WORDS for w in words)
        return "mr" if marathi > hindi else "hi"
    if code == "bn":  # Bengali vs Assamese: Assamese uses ৰ / ৱ
        return "as" if ("ৰ" in text or "ৱ" in text) else "bn"
    return code


# ---------------------------------------------------------------- number -> words

def _digits(s, L):
    return " ".join(L.DIGITS[int(c)] for c in s)


def _mult(q, L):
    """Form of q used in front of a scale word (e.g. 'ஒரு' instead of 'ஒன்று')."""
    return getattr(L, "MULT", {}).get(q, L.UNITS[q])


def _hundred(h, linked, L):
    """Word for h*100. linked=True when a remainder follows (some languages change form)."""
    table = getattr(L, "HUNDRED_LINK" if linked else "HUNDRED_ALONE", {})
    return table.get(h) or f"{_mult(h, L)} {L.HUNDRED}"


def _scale_part(q, spec, linked, L):
    """spec = (value, word[, linked_word[, omit_one]])."""
    word = spec[1]
    if linked and len(spec) > 2:
        word = spec[2]
    if q == 1 and len(spec) > 3 and spec[3]:  # e.g. Tamil 1000 = "ஆயிரம்", not "ஒரு ஆயிரம்"
        return word
    return f"{_mult(q, L)} {word}"


def _cardinal(n, L):
    total = n
    if n < 100:
        return L.UNITS[n]
    parts = []
    for spec in L.SCALES:  # thousand and above, largest first
        q, n = divmod(n, spec[0])
        if q:
            parts.append(_scale_part(q, spec, n > 0, L))
    if n >= 100:  # the sub-thousand tail stays one group: "पाँच सौ सड़सठ"
        h, n = divmod(n, 100)
        parts.append(_hundred(h, n > 0, L) + (f" {L.UNITS[n]}" if n else ""))
    elif n:
        parts.append(L.UNITS[n])
    # comma = pause cue for the model; only for lakh and above ("दो लाख, छत्तीस हज़ार")
    return (", " if total >= 10**5 else " ").join(parts)


def _number(s, L):
    """Digit string (commas removed) -> words. Leading zeros / huge values are read digit by digit."""
    if (len(s) > 1 and s[0] == "0") or len(s) > 20 or int(s) >= L.LIMIT:
        return _digits(s, L)
    return _cardinal(int(s), L)


def _amount(a, L):
    a = a.replace(",", "")
    whole, dot, frac = a.partition(".")
    if dot:
        return f"{_number(whole, L)} {L.POINT} {_digits(frac, L)}"
    return _number(whole, L)


def _money(a, L):
    whole, _, paise = a.replace(",", "").partition(".")
    out = f"{_number(whole, L)} {L.RUPEES}"
    paise = paise[:2].ljust(2, "0") if paise else ""
    if paise and int(paise):
        out += f" {_number(str(int(paise)), L)} {L.PAISE}"
    return out


def _year(y, L):
    if getattr(L, "YEAR_HUNDREDS", False) and 1100 <= y <= 1999:  # "उन्नीस सौ सैंतालीस"
        rest = f" {L.UNITS[y % 100]}" if y % 100 else ""
        return f"{L.UNITS[y // 100]} {L.HUNDRED}{rest}"
    return _cardinal(y, L)


def normalize_numbers(text, lang=None):
    """Replace every number in `text` with words.

    lang: language name ("hindi") or code ("hi"); auto-detected from the script when None.
    Text in an unsupported script (e.g. English) is returned unchanged.
    """
    # if lang param value is being provided
    if lang:
        lang = lang.lower()
        code = _NAMES.get(lang, lang)  # "hindi" -> "hi"; codes pass through
    # detect the language
    else:
        code = detect_lang(text)
    L = LANG_MODULES.get(code)
    if L is None:
        return text

    # 1. native digits (०१२...) -> ascii
    text = "".join(str(unicodedata.digit(c)) if c.isdigit() else c for c in text)

    # 2. dates dd/mm/yyyy
    def date(m):
        d, mo, y = int(m[1]), int(m[2]), int(m[3])
        if not (1 <= d <= 31 and 1 <= mo <= 12):
            return m[0]
        return f"{_cardinal(d, L)} {L.MONTHS[mo - 1]} {_year(y, L)}"

    text = re.sub(r"(?<![\d/.-])(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})(?![\d/-])(?!\.\d)", date, text)

    # 3. currency with a leading symbol: ₹1,500.50
    sym = "|".join(re.escape(s) for s in UNIVERSAL_CURRENCY + list(L.CURRENCY_SYMBOLS))
    text = re.sub(rf"(?:{sym})\s*({_AMOUNT})", lambda m: _money(m[1], L), text)

    # 4. phone numbers: 10 digits starting 6-9, optional +91, read as 5 + 5
    def phone(m):
        d = m[2]
        prefix = f"{L.PLUS} {_digits('91', L)}, " if m[1] else ""
        return f"{prefix}{_digits(d[:5], L)}, {_digits(d[5:], L)}"

    text = re.sub(r"(?<![\d,.])(\+91[\s-]?)?([6-9]\d{9})(?!\d)", phone, text)

    # 5. codes after a keyword: "OTP is 4582" -> digit by digit
    kw = "|".join(re.escape(w) for w in L.CODE_WORDS)
    text = re.sub(rf"((?:{kw})[^\d\n]{{0,20}}?)(\d{{4,8}})(?!\d)",
                  lambda m: m[1] + _digits(m[2], L), text)

    # 6. percent
    text = re.sub(rf"({_AMOUNT})\s*%", lambda m: f"{_amount(m[1], L)} {L.PERCENT}", text)

    # 7. everything else (plain integers, decimals)
    text = re.sub(_AMOUNT, lambda m: _amount(m[0], L), text)
    return text
