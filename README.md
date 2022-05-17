
# Make the Best of Cross-lingual Transfer: Evidence from POS Tagging with over 100 Languages

[Wietse de Vries](https://scholar.google.nl/citations?user=gZkWURYAAAAJ) •
[Martijn Wieling](https://scholar.google.nl/citations?user=Fzv0QJAAAAAJ) •
[Malvina Nissim](https://scholar.google.nl/citations?user=hnTpEOAAAAAJ)

 > **Abstract**: Cross-lingual transfer learning with large multilingual pre-trained models can be an effective approach for low-resource languages with no labeled training data. Existing evaluations of zero-shot cross-lingual generalisability of large pre-trained models use datasets with English training data, and test data in a selection of target languages. We explore a more extensive transfer learning setup with 65 different source languages and 105 target languages for part-of-speech tagging. Through our analysis, we show that pre-training of both source and target language, as well as matching language families, writing systems, word order systems, and lexical-phonetic distance significantly impact cross-lingual performance. 

**Status:** Published at ACL 2022. Final version: [click here](https://aclanthology.org/2022.acl-long.529.pdf).

```bibtex
@inproceedings{de-vries-etal-2022-make,
    title = "Make the Best of Cross-lingual Transfer: Evidence from {POS} Tagging with over 100 Languages",
    author = "de Vries, Wietse  and
      Wieling, Martijn  and
      Nissim, Malvina",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.529",
    pages = "7676--7685",
}
```

## Demo and Models

The main results and discussion of the paper are based on the predictions of 65 fine-tuned XLM-RoBERTa models. A demo and the 65 models can be found in this Hugging Face Space:

- [🤗&nbsp; hf.co/spaces/wietsedv/xpos](https://huggingface.co/spaces/wietsedv/xpos)


## Code

### Dependencies
If you use Conda environments, you can replicate the exact dependency versions that were used for the experiments:

```bash
conda create -n xpos --file conda-linux-64.lock  # if 64-bit Linux
conda create -n xpos --file conda-osx-arm64.lock  # if Apple Silicon
conda activate xpos
```

### Training
You can then train the models with:

```bash
python src/train.py udpos --learning_rate=5e-5 --eval_steps=1000 --per_device_batch_size=10 --max_steps=1000 --multi
```

**Tip:** append `--dry_run` to the previous command to only download and cache base models and data without training any models.

**Tip**: append `--language_source {lang_code}` (to any command) to only train for one source language. Check language codes in `configs/udpos.yaml`.

### Cross-lingual prediction

Predictions for the best trained models for every target language can be generated with:

```bash
python src/predict.py udpos
```

**Tip:** append `--language_source {lang_code}` or `--language_target {lang_code}` to generate predictions for specific languages.

**Tip:** append `--digest {digest}` to generate predictions for a specific training configuration. The digest is the random string of 8 characters in the output path of each model.

### Cross-lingual results

Export a csv with test accuracies for every source/target combination with:

```bash
python src/results.py udpos -a -e results.csv
```

**Tip:** Just like with training and prediction, you can specify specific languages or a specific digest.

### Models

Export trained models with:

```bash
python src/export.py udpos -e models
```

**Tip:** Just like with training and prediction, you can specify specific languages or a specific digest.
