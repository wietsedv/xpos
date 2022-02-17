from pathlib import Path
import numpy as np
import pandas as pd


card = """
---
language:
- {src_lang_code}
license: apache-2.0
library_name: transformers
tags:
- part-of-speech
- token-classification
datasets:
- universal_dependencies
metrics:
- accuracy

model-index:
- name: {model_id}
  results:
  - task: 
      type: token-classification
      name: Part-of-Speech Tagging
    dataset:
      type: universal_dependencies
      name: Universal Dependencies v2.8
    metrics:
{res}
---

# XLM-RoBERTa base Universal Dependencies v2.8 POS tagging: {src_lang_name}

This model is part of our paper called:

- Make the Best of Cross-lingual Transfer: Evidence from POS Tagging with over 100 Languages
    
Check the [Space]([Space](https://huggingface.co/spaces/wietsedv/xpos)) for more details.
"""

res = """      - type: accuracy
        name: {tgt_lang_name} Test accuracy
        value: {tgt_acc}"""

ud_df = pd.read_csv("exports/udpos-langs.csv")
res_df = pd.read_csv("exports/udpos-ft-10000.csv")

model_id = "xlm-roberta-base-ft-udpos28-{src_lang_code}"
model_dir = Path("models/1d6ca3e8/")

for i,row in ud_df.iterrows():
    src_lang_code = row.language_id
    src_lang_name = row.language

    m = Path(model_id.format(src_lang_code=src_lang_code))
    p = model_dir / m
    if not p.exists():
        continue

    src_res_df = res_df.loc[res_df.lang_train == src_lang_code]
    if len(src_res_df) == 0:
        continue

    r = "\n".join([res.format(tgt_lang_name=ud_df.loc[ud_df.language_id == x.lang_pred].iloc[0].language, tgt_acc=f"{x.score:.1f}") for _, x in src_res_df.iterrows()])

    t = card.format(src_lang_code=src_lang_code, src_lang_name=src_lang_name, model_id=m, res=r)

    with open(p / "README.md", "w") as f:
        f.write(t)

    print(p)
