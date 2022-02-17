# whitelist = {}
# with open("whitelist.txt") as f:
#     for line in f:
#         k, v = line.rstrip().split()
#         whitelist[k] = v

# print(whitelist)

whitelist = {
    "english": "en",
    "afrikaans": "af",
    "standard_arabic": "ar",
    "bulgarian": "bg",
    "bengali": "bn",
    "standard_german": "de",
    "greek": "el",
    "spanish": "es",
    "estonian": "et",
    "basque": "eu",
    "persian": "fa",
    "finnish_romani": "fi",
    "french": "fr",
    "hebrew": "he",
    "hindi": "hi",
    "hungarian": "hu",
    "indonesian": "id",
    "italian": "it",
    "japanese": "ja",
    "yogyakarta": "jv",
    "georgian": "ka",
    "kazakh": "kk",
    "korean": "ko",
    "malayalam": "ml",
    "marathi": "mr",
    "malay": "ms",
    "burmese": "my",
    "dutch": "nl",
    "portuguese": "pt",
    "russian": "ru",
    "swahili": "sw",
    "tamil": "ta",
    "telugu": "te",
    "thai": "th",
    "tagalog": "tl",
    "turkish": "tr",
    "urdu": "ur",
    "vietnamese": "vi",
    "yoruba": "yo",
    "mandarin": "zh",
}

f1 = open("lists.txt", encoding="ISO-8859-1")
f2 = open("lists2.txt", "w", encoding="ISO-8859-1")

content = False
include = False
for line in f1:
    if not content:
        f2.write(line)
        if line == "                                 \n":
            content = True
        continue

    if "{" in line and line.endswith("}\n"):
        lang_id = line.split("{")[0].lower()
        if lang_id in whitelist:
            include = True
            f2.write(f"{whitelist.pop(lang_id)}\n")
        else:
            include = False
        continue
    
    if include:
        f2.write(line)
f2.write("     \n")

if len(whitelist) > 0:
    print("WARNING: Some languages were not found")
    print(whitelist)
