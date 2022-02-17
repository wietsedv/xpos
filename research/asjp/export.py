""" Usage: python research/asjp/export.py en """

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

parser = ArgumentParser()
parser.add_argument("language")
parser.add_argument("-e", "--export", default=None, type=Path)
args = parser.parse_args()

ldnd_distances = {}
with open("research/asjp/ldnd.txt") as f:
    next(f)
    next(f)
    next(f)

    langs = next(f).strip().split()[1:]
    tgt_col = langs.index(args.language)
    next(f)

    for line in f:
        row = line.strip().split()
        if line == "\n":
            break
        if row[0] == args.language:
            for lang, d in zip(langs, row[1:]):
                ldnd_distances[lang] = float(d)
        elif len(row) > tgt_col + 1:
            ldnd_distances[row[0]] = float(row[tgt_col + 1])

ldnd_distances = sorted(ldnd_distances.items())
df = pd.DataFrame(ldnd_distances, columns=["language", "ldnd"])
print(df)
if args.export:
    df.to_csv(args.export)
