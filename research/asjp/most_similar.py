import sys
import json

tgt_lang = sys.argv[1]

f = open("ldnd.txt")
next(f)
next(f)
next(f)

langs = next(f).strip().split()[1:]

tgt_col = langs.index(tgt_lang)

next(f)

distances = {}
for line in f:
    row = line.strip().split()
    if line == "\n":
        break
    if row[0] == tgt_lang:
        for lang, d in zip(langs, row[1:]):
            distances[lang] = float(d)
    elif len(row) > tgt_col + 1:
        distances[row[0]] = float(row[tgt_col + 1])

langs = sorted(langs, key=distances.get)
print(json.dumps(distances, indent=2))
print(json.dumps(langs))
print(json.dumps([distances[l] for l in langs]))
