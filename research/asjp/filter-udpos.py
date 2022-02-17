# whitelist = {}
# with open("whitelist.txt") as f:
#     for line in f:
#         k, v = line.rstrip().split()
#         whitelist[k] = v

# print(whitelist)

# pred
# whitelist = [
#     'AFRIKAANS', 'AKKADIAN', 'AKUNTSU', 'ALBANIAN', 'ANCIENT_GREEK', 'APURINA', 'ARABIC', 'ARMENIAN', 'ASSYRIAN',
#     'BAMBARA', 'BASQUE', 'BELARUSIAN', 'BHOJPURI', 'BRETON', 'BULGARIAN', 'BURYAT', 'CANTONESE', 'CATALAN', 'CHINESE',
#     'CHUKCHI', 'CLASSICAL_CHINESE', 'CROATIAN', 'CZECH', 'DANISH', 'DUTCH', 'ENGLISH', 'ERZYA', 'ESTONIAN', 'FAROESE',
#     'FINNISH', 'FRENCH', 'GALICIAN', 'GERMAN', 'GOTHIC', 'GREEK', 'GUAJAJARA', 'HEBREW', 'HINDI', 'HUNGARIAN',
#     'ICELANDIC', 'INDONESIAN', 'IRISH', 'ITALIAN', 'JAPANESE', 'KAAPOR', 'KANGRI', 'KARELIAN', 'KAZAKH', 'KHUNSARI',
#     'KICHE', 'KOMI_PERMYAK', 'KOMI_ZYRIAN', 'KOREAN', 'KURMANJI', 'LATIN', 'LATVIAN', 'LITHUANIAN', 'LIVVI',
#     'LOW_SAXON', 'MAKURAP', 'MALTESE', 'MANX', 'MARATHI', 'MBYA_GUARANI', 'MOKSHA', 'MUNDURUKU', 'NAIJA', 'NAYINI',
#     'NORTH_SAMI', 'NORWEGIAN', 'OLD_CHURCH_SLAVONIC', 'OLD_EAST_SLAVIC', 'OLD_FRENCH', 'OLD_TURKISH', 'PERSIAN',
#     'POLISH', 'PORTUGUESE', 'ROMANIAN', 'RUSSIAN', 'SANSKRIT', 'SCOTTISH_GAELIC', 'SERBIAN', 'SKOLT_SAMI', 'SLOVAK',
#     'SLOVENIAN', 'SOUTH_LEVANTINE_ARABIC', 'SPANISH', 'SWEDISH', 'SWISS_GERMAN', 'TAGALOG', 'TAMIL', 'TELUGU', 'THAI',
#     'TUPINAMBA', 'TURKISH', 'UKRAINIAN', 'UPPER_SORBIAN', 'URDU', 'UYGHUR', 'VIETNAMESE', 'WARLPIRI', 'WELSH',
#     'WESTERN_ARMENIAN', 'WOLOF', 'YORUBA'
# ]

# train
whitelist = [
    'AFRIKAANS', 'ANCIENT_GREEK', 'ARABIC', 'ARMENIAN', 'BASQUE', 'BELARUSIAN', 'BULGARIAN', 'CATALAN', 'CHINESE',
    'CLASSICAL_CHINESE', 'CROATIAN', 'CZECH', 'DANISH', 'DUTCH', 'ENGLISH', 'ESTONIAN', 'FAROESE', 'FINNISH', 'FRENCH',
    'GALICIAN', 'GERMAN', 'GOTHIC', 'GREEK', 'HEBREW', 'HINDI', 'HUNGARIAN', 'ICELANDIC', 'INDONESIAN', 'IRISH',
    'ITALIAN', 'JAPANESE', 'KOREAN', 'LATIN', 'LATVIAN', 'LITHUANIAN', 'MALTESE', 'MARATHI', 'NAIJA', 'NORTH_SAMI',
    'NORWEGIAN', 'OLD_CHURCH_SLAVONIC', 'OLD_EAST_SLAVIC', 'OLD_FRENCH', 'PERSIAN', 'POLISH', 'PORTUGUESE', 'ROMANIAN',
    'RUSSIAN', 'SANSKRIT', 'SCOTTISH_GAELIC', 'SERBIAN', 'SLOVAK', 'SLOVENIAN', 'SPANISH', 'SWEDISH', 'TAMIL', 'TELUGU',
    'TURKISH', 'UKRAINIAN', 'URDU', 'UYGHUR', 'VIETNAMESE', 'WELSH', 'WESTERN_ARMENIAN', 'WOLOF'
]

langmap = {
    "GREEK_ANCIENT": "ANCIENT_GREEK",
    "CAIRO_ARABIC": "ARABIC",
    "STANDARD_GERMAN": "GERMAN",
    "NORWEGIAN_BOKMAAL": "NORWEGIAN",
    "MANDARIN": "CHINESE",
    "NORTHERN_LOW_SAXON": "LOW_SAXON",
    "ASSYRIAN_NEO_ARAMAIC": "ASSYRIAN",
    "OLD_CHINESE": "CLASSICAL_CHINESE",
    "GAELIC_SCOTTISH": "SCOTTISH_GAELIC",
    "SERBOCROATIAN": "SERBIAN",
}

f1 = open("lists.txt", encoding="ISO-8859-1")
f2 = open("lists-udpos.txt", "w", encoding="ISO-8859-1")

content = False
include = False
for line in f1:
    if not content:
        f2.write(line)
        if line == "                                 \n":
            content = True
        continue

    if "{" in line and line.endswith("}\n"):
        lang_id = line.split("{")[0]
        lang_id = langmap.pop(lang_id, lang_id)

        if lang_id in whitelist:
            include = True
            f2.write(f"{lang_id}\n")
            whitelist.remove(lang_id)
        else:
            include = False
        continue

    if include:
        f2.write(line)
f2.write("     \n")

if len(langmap) > 0:
    print("WARNING: Some languages were not used")
    print(langmap)

if len(whitelist) > 0:
    print("WARNING: Some languages were not found")
    print(whitelist)
