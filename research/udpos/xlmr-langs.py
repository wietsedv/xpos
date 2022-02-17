from argparse import ArgumentParser
from pathlib import Path
import unicodedata
import pandas as pd
import wikipedia
from typing import List

parser = ArgumentParser()
parser.add_argument("-e", "--export", default=None, type=Path)
args = parser.parse_args()


def get_script(txt):
    scripts = []
    try:
        for c in txt:
            s = unicodedata.name(c).split()
            if s[0] == "OLD":
                s = s[0] + " " + s[1]
            else:
                s = s[0]
            scripts.append(s)
    except ValueError:
        pass

    scripts = [
        s for s in scripts if s not in {
            "FULL",
            "LEFT-POINTING",
            "RIGHT-POINTING",
            "RIGHT",
            "LEFT",
            "DIGIT",
            "QUOTATION",
            "COMMA",
            "COLON",
            "HYPHEN-MINUS",
            'GREATER-THAN',
            'SOLIDUS',
            'AMPERSAND',
            'PLUS',
            'APOSTROPHE',
            'LESS-THAN',
            'VERTICAL',
            'EQUALS',
            'EXCLAMATION',
            'TILDE',
            'QUESTION',
            'NUMBER',
            'DOLLAR',
            'IDEOGRAPHIC',
            'FULLWIDTH',
            'EM',
            'PERCENT',
            'MIDDLE',
            'LOW',
            'POUND',
            'MIDLINE',
            'HORIZONTAL',
            'BULLET',
            'BOX',
            'SEMICOLON',
            'SPACE',
            'MODIFIER',
            'INVERTED',
            'COMBINING',
            'ACUTE',
            'RIGHT-TO-LEFT',
            'TELEVISION',
            'SEEDLING',
            'ORANGE',
            'HEADPHONE',
            'EVERGREEN',
            'BLACK',
            'NEUTRAL',
            'SIGN',
            'RELIEVED',
            'YELLOW',
            'HIGH',
            'TEACUP',
            'GRINNING',
            'WHITE',
            'HOT',
            'FIRE',
            'WINKING',
            'OPEN',
            'ROUND',
            'UPWARDS',
            'ROLLING',
            'SMILING',
            'FALLEN',
            'HEAVY',
            'VARIATION',
            'COMMERCIAL',
            'EMOJI',
            'MOBILE',
            'DOUBLE',
            'SPARKLES',
            'NUMERO',
            'EN',
            'SPLASHING',
            'CLAPPER',
            'FACE',
            'VICTORY',
            'RUNNER',
        }
    ]
    unique_scripts = [s for s in sorted(set(scripts)) if scripts.count(s) > len(scripts) * 0.2]
    if len(unique_scripts) == 1:
        return unique_scripts[0].lower()

    for script in unique_scripts:
        n = scripts.count(script)
        if n >= 0.95 * len(scripts):
            return script.lower()

    if unique_scripts == ["CJK", "HIRAGANA"]:
        return "kana"
    raise Exception("AMBIGUOUS:", unique_scripts, [scripts.count(s) for s in unique_scripts])


def get_text(lid):
    assert lid in wikipedia.languages()
    wikipedia.set_lang(lid)

    txt = ""

    titles = wikipedia.random(3)
    for title in titles:
        try:
            p = wikipedia.page(title)
        except wikipedia.WikipediaException:
            return get_text(lid)
        txt += p.summary

    assert txt
    return txt



lines = []
with open("research/udpos/langs-xlmr.txt") as f:
    for line in f:
        lid, lname, ntokens, gib = line.rstrip().split()
        lname = lname.replace("_", " ")
        ntokens = float(ntokens)
        gib = float(gib)

        print(f"{lid:<5} {lname:<20}")

        if lname.endswith(" Romanized"):
            lname = lname.replace(" Romanized", "")
            s = "latin"
        else:
            s = None
            while s is None:
                try:
                    s = get_script(get_text(lid))
                except Exception as e:
                    print(e)

            if s == "cjk":
                s = "chinese"

        print(f"{s}\n")

        lines.append((lid, lname, ntokens, gib, s))

df = pd.DataFrame(lines, columns=["language_id", "language", "tokens", "size", "script"])
print(df)
print(df.script.value_counts())

if args.export:
    df.to_csv(args.export)
