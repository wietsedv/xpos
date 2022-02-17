import sys
import json
import numpy as np
import pandas as pd

ud_df = pd.read_csv("../../exports/udpos-langs.csv")
ud_langs = {row["language"].lower().replace(" ", "_"): row["language_id"] for _, row in ud_df.iterrows()}

f = open("ldnd-udpos.txt")
next(f)
next(f)
next(f)

langs = next(f).strip().split()[1:]
lang_names = []

next(f)

distances = np.empty((len(langs), len(langs)))

for i, line in enumerate(f):
    row = line.strip().split()
    if line == "\n":
        break
    lang_names.append(row[0].lower())
    for j, d in enumerate(row[1:]):
        distances[i, j] = float(d)
        distances[j, i] = float(d)

known = []
sorted_distances = np.argsort(distances)
for i, idx in enumerate(sorted_distances):
    nearest = sorted([lang_names[j] for j in idx if distances[i,j] < 80])
    if len(nearest) == 1 or sorted(nearest) in known:
        continue
    known.append(sorted(nearest))

    print('"' + "-".join([ud_langs[l] for l in nearest]) + "\",  #", " / ".join(nearest))

# langs = sorted(langs, key=distances.get)
# print(json.dumps(distances, indent=2))
# print(json.dumps(langs))
# print(json.dumps([distances[l] for l in langs]))
