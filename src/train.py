#!/usr/bin/env python3

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from argparse import ArgumentParser, Namespace
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math

import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F
from colorama import Fore, Style
from datasets import Dataset, DatasetDict, load_from_disk
from seqeval.metrics.sequence_labeling import accuracy_score, precision_recall_fscore_support
from transformers.hf_argparser import HfArgumentParser
from transformers.trainer import Trainer
from transformers.trainer_utils import EvalPrediction, IntervalStrategy, set_seed
from transformers.training_args import TrainingArguments

from lib.benchmark import BenchmarkInstance, XlingBenchmark
from lib.retrieval import featurize
from lib.trainer import QuestionAnsweringTrainer
from lib.utils import get_data_collator, get_feature_columns, get_prepare_fn, output_dir_populated, print_instance
from lib.utils_qa import postprocess_qa_predictions


def parse_arguments(argv=None):
    parser = ArgumentParser(fromfile_prefix_chars="@")
    XlingBenchmark.add_argparse_arguments(parser)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_prep", action="store_true")
    parser.add_argument("--skip_layers", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("-d", "--digest", default=None)
    parser.add_argument("--cache_dir", default="cache", type=Path)
    parser.add_argument("--multi", action="store_true")
    args, argv = parser.parse_known_args(argv)
    return args, argv


def print_stage(txt):
    print(f"\n{Style.BRIGHT}{Fore.CYAN}▶ {txt}{Style.RESET_ALL}")


def get_training_digest(training_args: TrainingArguments, length=8, verbose=False) -> str:
    ignore_keys = {
        "output_dir", "logging_dir", "run_name", "push_to_hub_model_id", "overwrite_output_dir", "do_train", "do_eval",
        "do_predict", "local_rank"
    }
    training_cfg = {k: str(v) for k, v in vars(training_args).items() if k[0] != "_" and k not in ignore_keys}
    encoded = json.dumps(training_cfg, sort_keys=True).encode()
    digest = hashlib.md5(encoded).hexdigest()[:length]
    if verbose:
        print(json.dumps(training_cfg))
        print(f"\nTraining digest: {Fore.BLUE}{digest}{Fore.RESET}")
    return digest


def find_digest_train_args(digest: Optional[str], output_root: Path):
    if digest is None or digest == "23a338c8":
        return []
    for path in output_root.glob(f"**/{digest}/train.args"):
        with open(path) as f:
            argv = f.read().strip().split()
        _, argv = parse_arguments(argv)
        if argv == []:
            continue
        return argv
    print(f"Digest {digest} not found!")
    exit(1)


def get_training_args(instance: BenchmarkInstance, argv: List[str], verbose=False):
    hf_parser = HfArgumentParser(TrainingArguments)
    print(argv)
    training_args, remaining_args = hf_parser.parse_args_into_dataclasses(["--output_dir", str(instance.output_base)] + argv, return_remaining_strings=True)
    assert isinstance(training_args, TrainingArguments)

    if remaining_args:
        print("WARNING: Unused arguments:", remaining_args)

    training_args.save_strategy = IntervalStrategy.NO
    instance.training_digest = get_training_digest(training_args, verbose=verbose)
    training_args.run_name = "/".join(instance.output_path.parts[-2:])
    training_args.output_dir = str(instance.output_path)
    training_args.evaluation_strategy = IntervalStrategy.STEPS
    if training_args.load_best_model_at_end:
        training_args.save_strategy = IntervalStrategy.STEPS
        training_args.save_total_limit = 1
    training_args.eval_steps = training_args.eval_steps or 500
    if instance.task.task_type in ["question-answering", "text-classification"]:
        training_args.eval_steps = training_args.eval_steps * 10
    training_args.logging_steps = training_args.eval_steps // 5

    return training_args


def get_training_instance(benchmark: XlingBenchmark, args: Namespace) -> Union[BenchmarkInstance, List[BenchmarkInstance]]:
    instances = benchmark.instances(source=True, args=args)
    if len(instances) > 1:
        print("Found multiple train instances:\n", file=sys.stderr)
        for instance in instances:
            print_instance(instance, "train")
        if args.multi:
            return instances
        exit(1)
    elif len(instances) == 0:
        print("No train instances found, all available instances:\n", file=sys.stderr)
        for instance in benchmark.instances(source=True):
            print_instance(instance)
        exit(1)
    return instances[0]


def get_compute_metrics_fn(task_type: str, id2label: Optional[Dict[int, str]]):
    if task_type == "text-classification":

        def compute_text_classification_metrics(p: EvalPrediction):
            predictions = np.argmax(p.predictions, axis=1)
            acc = (predictions == p.label_ids).sum() / predictions.shape[0]
            return {
                "accuracy": acc,
            }

        return compute_text_classification_metrics

    if task_type == "token-classification":
        assert id2label is not None
        label_list = [id2label[i] for i in sorted(id2label)]
        seq_label_list = [l if l == "O" or l[:2] in ["B-", "I-"] else f"B-{l}" for l in label_list]

        def compute_token_classification_metrics(p: EvalPrediction):
            predictions = np.argmax(p.predictions, axis=2)
            labels_pred = [[seq_label_list[p] for p, l in zip(prediction, label) if l != -100]
                           for prediction, label in zip(predictions, p.label_ids)]
            labels_true = [[seq_label_list[l] for l in label if l != -100] for label in p.label_ids]

            acc = accuracy_score(y_true=labels_true, y_pred=labels_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true=labels_true, y_pred=labels_pred, average="macro")
            res = {
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "accuracy": acc,
            }
            return res

        return compute_token_classification_metrics

    if task_type == "question-answering":
        from datasets import load_metric
        metric = load_metric("squad")

        def compute_question_answering_metrics(p: EvalPrediction) -> Dict[str, Any]:
            return metric.compute(predictions=p.predictions, references=p.label_ids)  # type: ignore

        return compute_question_answering_metrics


def _prepare_training(instance: BenchmarkInstance):
    print_stage("Prepare data")
    data = instance.load_datasets()
    assert isinstance(data, DatasetDict)
    # print(data)

    task_type = instance.task.task_type
    text_cols, label_col, label_list = get_feature_columns(data["train"], task_type)

    print_stage("Prepare model")
    tokenizer = instance.load_tokenizer()
    data_collator = get_data_collator(task_type, tokenizer)
    model = instance.load_model(label_list)
    print(model.config)

    print_stage("Preprocess data")
    prepare_fn = get_prepare_fn(tokenizer, task_type, label_col, text_cols)
    train_dataset = data["train"].map(prepare_fn, batched=True, remove_columns=data["train"].column_names)
    if "validation" in data:
        prepare_eval_fn = get_prepare_fn(tokenizer, task_type, label_col, text_cols)
        valid_dataset = data["validation"].map(prepare_eval_fn,
                                            batched=True,
                                            remove_columns=data["validation"].column_names)
    else:
        valid_dataset = None
    print(train_dataset)
    print(valid_dataset)

    return model, data_collator, train_dataset, valid_dataset


def _init_wandb(instance: BenchmarkInstance, training_args: TrainingArguments, args: Namespace):
    import wandb

    tags = [
        instance.task.task_type, instance.task.task_name, instance.language, instance.model_id,
        instance.model_config.config_name
    ]

    wandb.init(project="xling-benchmarks", name=training_args.run_name, tags=tags, reinit=True)
    wandb.config.update(args)
    return wandb


def run_training(instance: BenchmarkInstance, training_args: TrainingArguments, args: Namespace):
    model, data_collator, train_dataset, valid_dataset = _prepare_training(instance)
    valid_split = "validation"
    if valid_dataset is None:
        valid_dataset, valid_split = train_dataset, "train"

    if train_dataset.num_rows == 0:
        print("Training dataset is empty")
        return

    if args.max_eval_samples is not None and valid_dataset is not None:
        valid_dataset = valid_dataset.select(range(args.max_eval_samples))

    print_stage("Prepare trainer")
    task_type = instance.task.task_type
    if task_type == "question-answering":
        eval_examples = instance.load_datasets(valid_split)

        def post_processing_function(examples, features, predictions, stage="eval"):
            # Post-processing: we match the start logits and end logits to answers in the original context.
            predictions = postprocess_qa_predictions(
                examples=examples,
                features=features,
                predictions=predictions,
                n_best_size=20,
                max_answer_length=30,
                null_score_diff_threshold=0.0,
                output_dir=training_args.output_dir,
                prefix=stage,
            )
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

            references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
            return EvalPrediction(predictions=formatted_predictions, label_ids=references)  # type: ignore

        trainer_cls = partial(QuestionAnsweringTrainer,
                              eval_examples=eval_examples,
                              post_process_function=post_processing_function)
    else:
        trainer_cls = Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=valid_dataset,  # type: ignore
        tokenizer=data_collator.tokenizer,
        data_collator=data_collator,
        compute_metrics=get_compute_metrics_fn(task_type, model.config.id2label),
    )

    if args.dry_run:
        print("Not training in dry run")
        return

    _init_wandb(instance, training_args, args)

    print_stage("Start training")
    train_result = trainer.train()

    print_stage("Save train result")
    trainer.save_model()
    trainer.save_metrics("train", train_result.metrics)
    print(train_result.metrics)

    print_stage("Evaluate")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    for ckpt_path in Path(training_args.output_dir).glob("checkpoint-*"):
        shutil.rmtree(ckpt_path)


@torch.inference_mode()
def train_retrieval(instance: BenchmarkInstance, training_args: TrainingArguments, args: Namespace):
    """ Currently, retrieval training must use STS-like data """

    print_stage("Load data")
    data = instance.load_datasets()
    assert isinstance(data, DatasetDict)
    # print(data)

    task_type = instance.task.task_type
    text_cols, label_col, _ = get_feature_columns(data["train"], task_type)

    print_stage("Loading model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = instance.load_model().to(device)
    model_fn = lambda: model
    tokenizer = instance.load_tokenizer()
    data_collator = get_data_collator(task_type, tokenizer)

    if args.dry_run:
        print("Not training in dry run")
        exit(0)

    n_layers = model.config.num_hidden_layers + 1  # type: ignore
    layers = list(range(args.skip_layers, n_layers))

    print_stage("Preparing data")
    data_map = partial(get_prepare_fn, tokenizer, "sentence-retrieval", label_col)
    train_X_data = data["train"].map(data_map(text_cols[0]), batched=True, remove_columns=data["train"].column_names)
    train_Y_data = data["train"].map(data_map(text_cols[1]), batched=True, remove_columns=data["train"].column_names)
    train_labels = train_X_data["labels"]
    assert isinstance(train_labels, list)

    featurize_fn = partial(featurize,
                           model_fn,
                           data_collator,
                           normalize=True,
                           batch_size=training_args.per_device_eval_batch_size,
                           verbose=True)
    print_stage("Featurizing training data sentence1")
    train_layers_X = featurize_fn(layer=layers, dataset=train_X_data)
    print_stage("Featurizing training data sentence2")
    train_layers_Y = featurize_fn(layer=layers, dataset=train_Y_data)

    train_results = {"pearsonr": {}, "spearmanr": {}}

    for l in layers:
        sent1_feats = train_layers_X[l]
        sent2_feats = train_layers_Y[l]
        sims = F.cosine_similarity(torch.tensor(sent1_feats), torch.tensor(sent2_feats))
        train_results["pearsonr"][str(l)] = scipy.stats.pearsonr(train_labels, sims)[0]
        train_results["spearmanr"][str(l)] = scipy.stats.spearmanr(train_labels, sims)[0]

    os.makedirs(training_args.output_dir, exist_ok=True)
    print(train_results)
    with open(os.path.join(training_args.output_dir, "train_results.json"), "w") as f:
        json.dump(train_results, f, indent=2)

    res = train_results[instance.task.metric]
    best_layer = int(max(res, key=res.get))  # type: ignore

    eval_results = _train_retrieval_eval(data["validation"], data_map, text_cols, featurize_fn, best_layer)
    print(eval_results)
    with open(os.path.join(training_args.output_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=2)

    model = instance.load_model(num_hidden_layers=best_layer)
    model.save_pretrained(training_args.output_dir)
    data_collator.tokenizer.save_pretrained(training_args.output_dir)


def _train_retrieval_eval(data: Dataset, data_map: Callable, text_cols: Tuple[str, ...], featurize_fn: Callable,
                          best_layer: int):
    X_data = data.map(data_map(text_cols[0]), batched=True, remove_columns=data.column_names)
    Y_data = data.map(data_map(text_cols[1]), batched=True, remove_columns=data.column_names)
    true_labels = X_data["labels"]
    assert isinstance(true_labels, list)

    print_stage("Featurizing validation data sentence1")
    eval_sent1_layer_feats = featurize_fn(layer=best_layer, dataset=X_data)
    print_stage("Featurizing validation data sentence2")
    eval_sent2_layer_feats = featurize_fn(layer=best_layer, dataset=Y_data)

    eval_results: Dict[str, Any] = {"layer": best_layer}

    sims = F.cosine_similarity(torch.tensor(eval_sent1_layer_feats), torch.tensor(eval_sent2_layer_feats))
    eval_results["eval_pearsonr"] = scipy.stats.pearsonr(true_labels, sims)[0]
    eval_results["eval_spearmanr"] = scipy.stats.spearmanr(true_labels, sims)[0]
    return eval_results


def load_raw_data(instance: BenchmarkInstance, args):
    cache_path, data = None, None
    if args.cache_dir:
        cache_name, data_subset = instance.task.data_name_subset(instance.language)
        if data_subset:
            cache_name = f"{cache_name}_{data_subset}"
        cache_path = args.cache_dir / f"{cache_name}"
        if cache_path.exists():
            print("Loading from cache:", cache_path)
            data = load_from_disk(cache_path)
            data.cleanup_cache_files()
    if data is None:
        if args.no_prep:
            print("Data is not prepared yet. Not doing anything")
            exit(1)
        data = instance.load_datasets()
        if cache_path:
            print("Writing to cache:", cache_path)
            data.save_to_disk(cache_path)
            data.cleanup_cache_files()
    assert isinstance(data, DatasetDict)
    return data


def train_language_modeling(instance: BenchmarkInstance, training_args: TrainingArguments, args: Namespace):
    data = None
    if not args.no_prep:
        print_stage("Prepare data")
        data = load_raw_data(instance, args)
        print(data)

    print_stage("Prepare model")
    tokenizer = instance.load_tokenizer(data["train"] if data else None)
    data_collator = get_data_collator(instance.task.task_type, tokenizer)
    model = instance.load_model()
    model.resize_token_embeddings(len(tokenizer))
    print(tokenizer)
    print(model.config)

    print_stage("Preprocess data")
    data_path = instance.output_base / "data"
    if not data_path.exists():
        max_seq_length = tokenizer.model_max_length

        if args.no_prep:
            print("Data is not preprocessed yet. Not doing anything")
            exit(1)
        assert data

        def group_texts(examples):
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            result = {
                k: [t[i:i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        prepare_fn = get_prepare_fn(tokenizer, instance.task.task_type, label_col=None, text_cols="text")
        data = data.map(prepare_fn,
                        batched=True,
                        num_proc=args.num_workers,
                        remove_columns=data["train"].column_names,
                        desc=f"Tokenizing data")
        data = data.map(group_texts,
                        batched=True,
                        num_proc=args.num_workers,
                        desc=f"Grouping texts in chunks of {max_seq_length}")
        print("Writing to disk:", data_path)
        data.save_to_disk(str(data_path))
        data.cleanup_cache_files()
    else:
        print("Loading from disk:", data_path)
        data = load_from_disk(str(data_path))
        data.cleanup_cache_files()
    # print(data["train"])
    # print(data["validation"])

    # print("Example:")
    # ex = data["train"][0]
    # print(ex)
    # print(tokenizer.decode(ex["input_ids"]))

    if args.dry_run:
        print("Not training in dry run")
        exit(0)

    # TODO if args.max_train_samples is not None: train_dataset = train_dataset.select(range(args.max_train_samples))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],  # type: ignore
        eval_dataset=data["validation"],  # type: ignore
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    _init_wandb(instance, training_args, args)

    print_stage("Start training")
    train_result = trainer.train()

    print_stage("Save train result")
    trainer.save_model()
    train_metrics = train_result.metrics
    try:
        train_metrics["train_perplexity"] = math.exp(train_metrics["train_loss"])
    except OverflowError:
        train_metrics["train_perplexity"] = float("inf")
    trainer.save_metrics("train", train_result.metrics)
    print(train_result.metrics)

    print_stage("Evaluate")
    eval_metrics = trainer.evaluate()
    try:
        eval_metrics["eval_perplexity"] = math.exp(eval_metrics["eval_loss"])
    except OverflowError:
        eval_metrics["eval_perplexity"] = float("inf")
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)


def main():
    args, training_argv = parse_arguments()

    # Retrieve training instance
    benchmark = XlingBenchmark.from_yaml(args.config)
    instances = get_training_instance(benchmark, args)
    if isinstance(instances, BenchmarkInstance):
        instances = [instances]
    
    for instance in instances:
        output_root = Path(instance.output_base.parts[0])
        training_argv = find_digest_train_args(args.digest, output_root) + training_argv
        print("Arguments:", " ".join(training_argv))

        training_args = get_training_args(instance, training_argv, verbose=args.verbose)
        assert instance.training_digest

        if args.digest:
            tgt_output_dir = Path(training_args.output_dir).parent / args.digest
            os.makedirs(tgt_output_dir.parent, exist_ok=True)
            if tgt_output_dir.exists() and os.path.islink(tgt_output_dir):
                os.remove(tgt_output_dir)
            if args.digest and instance.training_digest != args.digest:
                print(
                    f"WARNING: Actual digest ({instance.training_digest}) is different from target digest ({args.digest}). Adding symlink"
                )
                if tgt_output_dir.exists() and training_args.overwrite_output_dir:
                    old_tgt_output_dir = tgt_output_dir.parent / (tgt_output_dir.name + "-old")
                    if not args.dry_run:
                        os.rename(tgt_output_dir, old_tgt_output_dir)
                    print(f"Renaming {tgt_output_dir} to {old_tgt_output_dir}")
                
                if not args.dry_run:
                    os.symlink(instance.training_digest, tgt_output_dir)

        if output_dir_populated(training_args.output_dir):
            if not training_args.overwrite_output_dir:
                print(f"output dir {training_args.output_dir} already exists. use --overwrite_output_dir to overwrite")
                print(os.listdir(training_args.output_dir))
                continue
            out_dir = Path(training_args.output_dir)
            old_output_dir = out_dir.parent / (out_dir.name + "-old")
            if not args.dry_run:
                os.rename(out_dir, old_output_dir)
            print(f"Renaming {out_dir} to {old_output_dir}")

        if not args.dry_run:
            os.makedirs(training_args.output_dir, exist_ok=True)
            save_args = len(sys.argv) > 1 and sys.argv[1][0] != "@"
            if save_args:
                with open(os.path.join(training_args.output_dir, "train.args"), "w") as f:
                    f.write(" ".join(sys.argv[1:]))

        print(f"Output dir: {Fore.BLUE}{training_args.output_dir}{Fore.RESET}")

        set_seed(training_args.seed)

        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        if instance.task.task_type.endswith("-retrieval"):
            train_retrieval(instance, training_args, args)
        elif instance.task.task_type == "masked-language-modeling":
            train_language_modeling(instance, training_args, args)
        else:
            run_training(instance, training_args, args)


if __name__ == "__main__":
    main()
