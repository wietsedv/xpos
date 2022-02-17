from pathlib import Path
import numpy as np
import pandas as pd

ud_df = pd.read_csv("exports/udpos-langs.csv")
res_df = pd.read_csv("exports/udpos-ft-10000.csv")


langs = []
for i,row in ud_df.iterrows():
    src_lang_code = row.language_id
    src_lang_name = row.language

    src_res_df = res_df.loc[res_df.lang_train == src_lang_code]
    if len(src_res_df) == 0:
        continue

    langs.append((src_lang_name, src_lang_code))

langs = sorted(langs)

print([x for x, _ in langs])
print([x for _, x in langs])