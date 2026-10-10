"""Hindi number words. Data only; the algorithm lives in normalize.py.
THIS FILE IS THE TEMPLATE for every other language module.

Required attributes
    UNITS      list of 100 words for 0..99 (standalone/counting form; all distinct)
    DIGITS     list of 10 words for 0..9, used to read OTPs / phone numbers digit by digit
    SCALES     thousand-and-above, LARGEST FIRST: (value, word[, word_when_more_follows[, omit_one]])
                 omit_one=True -> "1 x scale" is just the scale word (Tamil "ஆயிரம்", not "ஒரு ஆயிரம்")
    HUNDRED    the word for "hundred" (default composition: "<MULT[h]> <HUNDRED>")
    LIMIT      numbers >= LIMIT are read digit by digit (10**13 if the language has a word up to
               kharab, 10**9 if its largest scale is crore)
    POINT PERCENT PLUS RUPEES PAISE   "point", "percent", "plus", "rupees", "paise"
    MONTHS     12 month names, January first
    CODE_WORDS words meaning OTP / PIN / code / password (+ the Latin spellings "OTP", "PIN")
    CURRENCY_SYMBOLS  extra prefixes besides the universal ₹ / Rs. / Rs / INR (may be empty)

Optional attributes
    MULT            {q: form} of q used directly before a scale/hundred word (e.g. {1: "ஒரு"})
    HUNDRED_ALONE   {h: word} for h*100 with nothing after it        (h = 1..9)
    HUNDRED_LINK    {h: word} for h*100 when a remainder follows      (h = 1..9)
                    (Dravidian languages change the form: Tamil நூறு -> நூற்று)
    YEAR_HUNDREDS   True -> 1947 is read "nineteen hundred forty-seven"; False -> plain cardinal

Words are written with a decomposed nukta (base letter + U+093C), which is how the
model's vocab.txt stores it.
"""

# 0-99. Hindi has an irregular word for each of these.
UNITS = [
    "शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ",
    "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस",
    "बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस",
    "तीस", "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस",
    "चालीस", "इकतालीस", "बयालीस", "तैंतालीस", "चौवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास",
    "पचास", "इक्यावन", "बावन", "तिरपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ",
    "साठ", "इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर",
    "सत्तर", "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उनासी",
    "अस्सी", "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सत्तासी", "अट्ठासी", "नवासी",
    "नब्बे", "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे",
]
assert len(UNITS) == 100

DIGITS = UNITS[:10]

# Indian numbering, largest first.
SCALES = [
    (10**11, "खरब"),
    (10**9, "अरब"),
    (10**7, "करोड़"),
    (10**5, "लाख"),
    (10**3, "हज़ार"),
]

HUNDRED = "सौ"
YEAR_HUNDREDS = True  # 1947 -> "उन्नीस सौ सैंतालीस"
LIMIT = 10**13
POINT = "दशमलव"
PERCENT = "प्रतिशत"
PLUS = "प्लस"
RUPEES = "रुपये"
PAISE = "पैसे"

MONTHS = [
    "जनवरी", "फरवरी", "मार्च", "अप्रैल", "मई", "जून",
    "जुलाई", "अगस्त", "सितंबर", "अक्तूबर", "नवंबर", "दिसंबर",
]

# Words that mean the following 4-8 digit number is a code, to be read digit by digit.
CODE_WORDS = ["ओटीपी", "OTP", "otp", "कोड", "पिन", "PIN", "pin", "पासवर्ड", "ओटिपी", "ओ टी पी"]
CURRENCY_SYMBOLS = ["₹", "Rs.", "Rs", "INR", "रु."]
