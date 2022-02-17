#!/usr/bin/env python3

from argparse import ArgumentParser, Namespace
import json
import os
from functools import partial
from datasets.arrow_dataset import Dataset

from tqdm import tqdm
import numpy as np
import torch
from colorama import Fore, Style
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from lib.benchmark import BenchmarkInstance, NoDigestError, UnavailableInstanceError, XlingBenchmark
from lib.retrieval import search_pairs, featurize
from lib.utils import get_data_collator, get_prepare_fn, get_feature_columns, print_instance
from lib.utils_qa import get_qa_predictions


def print_stage(txt):
    print(f"\n{Style.BRIGHT}{Fore.CYAN}▶▶ {txt}{Style.RESET_ALL}")


@torch.inference_mode()
def predict_text_classification(dataset: Dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast, args):
    id2label = model.config.id2label
    assert isinstance(id2label, dict)
    data_collator = get_data_collator("text-classification", tokenizer)

    batches = list(range(0, dataset.num_rows, args.batch_size))
    for i in tqdm(batches, dynamic_ncols=True):
        batch = data_collator(dataset[i:i + args.batch_size])
        batch_labels = batch.pop("labels")
        batch = {k: t.to(model.device) for k, t in batch.items()}

        batch_preds = model(**batch).logits.cpu().argmax(-1)

        for j in range(batch_preds.shape[0]):
            label: int = batch_labels[j].tolist()  # type: ignore
            pred: int = batch_preds[j].tolist()
            yield id2label[label], id2label[pred]


@torch.inference_mode()
def predict_token_classification(dataset: Dataset, model: PreTrainedModel):
    id2label = model.config.id2label
    assert isinstance(id2label, dict)

    for tokenized_inputs in tqdm(dataset):
        labels = np.array(tokenized_inputs.pop("labels"))
        tokenized_inputs = {key: torch.tensor(t).to(model.device).unsqueeze(0) for key, t in tokenized_inputs.items()}
        predictions = model(**tokenized_inputs).logits.squeeze(0).cpu().numpy().argmax(-1)

        for true_id, pred_id in zip(labels, predictions):
            if true_id == -100:
                continue
            yield id2label[true_id], id2label[pred_id]


@torch.inference_mode()
def predict_question_answering(instance: BenchmarkInstance, dataset: Dataset, model: PreTrainedModel,
                               tokenizer: PreTrainedTokenizerFast, args):
    data_collator = get_data_collator("question-answering", tokenizer)

    examples = instance.load_datasets(args.split)
    assert isinstance(examples, Dataset)

    start_logits = np.zeros((dataset.num_rows, tokenizer.model_max_length), dtype=np.float32)
    end_logits = np.zeros((dataset.num_rows, tokenizer.model_max_length), dtype=np.float32)

    batches = list(range(0, dataset.num_rows, args.batch_size))
    for i in tqdm(batches, dynamic_ncols=True):
        batch_idx = np.arange(i, min(i + args.batch_size, dataset.num_rows))
        # batch_examples = examples.select(batch_idx, keep_in_memory=True)
        batch_features = dataset.select(batch_idx, keep_in_memory=True)

        batch_data = batch_features[:]
        del batch_data["example_id"]  # type: ignore
        del batch_data["offset_mapping"]  # type: ignore

        batch = data_collator(batch_data)
        batch = {k: t.to(model.device) for k, t in batch.items()}

        batch_start_logits = model(**batch).start_logits.cpu().numpy()
        batch_end_logits = model(**batch).end_logits.cpu().numpy()

        start_logits[i:i + batch_start_logits.shape[0], :batch_start_logits.shape[1]] = batch_start_logits
        end_logits[i:i + batch_end_logits.shape[0], :batch_end_logits.shape[1]] = batch_end_logits

    predictions = get_qa_predictions(examples, dataset, (start_logits, end_logits))
    for ex in examples:
        yield json.dumps({"id": ex["id"], "answers": ex["answers"]}), predictions[ex["id"]].replace("\n", " ")


LOADED_MODEL = None

@torch.inference_mode()
def run_predict(instance: BenchmarkInstance, args: Namespace) -> bool:
    global LOADED_MODEL
    print_stage("Load tokenizer")
    try:
        tokenizer = instance.load_tokenizer()
    except NoDigestError as e:
        print(" > Skipping, because digest is missing for:", e)
        return False
    except UnavailableInstanceError as e:
        print(" > Skipping, because imported instance is not available:", e)
        return False

    print_stage("Load data")
    data = instance.load_datasets(split=args.split)
    assert isinstance(data, Dataset)
    # print(data)
    if data.num_rows == 0:
        print(" > Skipping, because number of rows is 0")
        return False

    task_type = instance.task.task_type
    text_cols, label_col, _ = get_feature_columns(data, task_type)

    print_stage("Load model")
    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    model_path = instance.model_path
    if model_path and LOADED_MODEL and LOADED_MODEL[0] == model_path:
        print(f"Reusing previous model [{model_path}]")
        model = LOADED_MODEL[1]
    else:
        model = instance.load_model().to(device)
        if model_path:
            LOADED_MODEL = model_path, model

    print_stage("Prepare data")
    dataset = data.map(get_prepare_fn(tokenizer, task_type, label_col, text_cols),
                       batched=True,
                       remove_columns=data.column_names)
    # print(dataset)

    print_stage(f"Predicting labels")
    predict_path = instance.get_predict_path(split=args.split)
    os.makedirs(predict_path.parent, exist_ok=True)

    lines = []
    if task_type == "text-classification":
        predictor = predict_text_classification(dataset, model, tokenizer, args)
    elif task_type == "token-classification":
        predictor = predict_token_classification(dataset, model)
    elif task_type == "question-answering":
        predictor = predict_question_answering(instance, dataset, model, tokenizer, args)
    else:
        print("Cannot predict task type", task_type)
        exit(1)

    for cols in predictor:
        lines.append("\t".join(cols) + "\n")

    with open(predict_path, "w") as f:
        f.writelines(lines)
    
    return True


@torch.inference_mode()
def run_retrieval(instance: BenchmarkInstance, args: Namespace) -> bool:
    from transformers.utils.logging import set_verbosity_error
    set_verbosity_error()
    print_stage("Load tokenizer")
    try:
        target_tokenizer = instance.load_tokenizer()
    except NoDigestError as e:
        print(" > Skipping, because digest is missing for:", e)
        return False
    except UnavailableInstanceError as e:
        print(" > Skipping, because imported instance is not available:", e)
        return False

    print_stage("Load data")
    data = instance.load_datasets(split=args.split)
    # print(data)
    assert isinstance(data, Dataset)

    task_type = instance.task.task_type
    assert task_type == "sentence-retrieval"
    text_cols, label_col, _ = get_feature_columns(data, task_type)

    # cache_dir = instance.get_predict_path(split=args.split, ext="tmp")
    featurize_fn = partial(featurize, normalize=True, batch_size=args.batch_size, verbose=args.verbose)
    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")

    column_names = data.column_names

    def _featurize(model_fn, tokenizer: PreTrainedTokenizerFast, col_name: str):
        X_indices = np.array([i for i, x in enumerate(data[col_name]) if x is not None])
        X_data = data.filter(lambda x: x[col_name] is not None)
        assert X_indices.shape == (X_data.num_rows, )
        data_collator = get_data_collator(task_type, tokenizer)
        data_map = partial(get_prepare_fn, tokenizer, "sentence-retrieval", label_col)
        X_data = X_data.map(data_map(col_name), batched=True, remove_columns=column_names)

        target_featurize_fn = partial(featurize_fn, model_fn, data_collator)
        X = target_featurize_fn(dataset=X_data)
        assert isinstance(X, np.ndarray)
        return X, X_indices

    print_stage(f"Featurizing target [{instance.language}]")
    target_model_fn = lambda: instance.load_model(verbose=args.verbose).to(device)
    X, X_indices = _featurize(target_model_fn, target_tokenizer, text_cols[0])

    print_stage("Featurizing source [orig]")
    # source_model_fn = lambda: instance.source_instance.load_model(verbose=args.verbose, num_hidden_layers=11).to(device)
    source_model_fn = lambda: instance.load_model(verbose=args.verbose, no_transform=True).to(device)
    source_tokenizer = instance.source_instance.load_tokenizer()
    Y, Y_indices = _featurize(source_model_fn, source_tokenizer, text_cols[1])

    print_stage("Searching pairs")
    pred_x2y_indices, pred_x2y_distances = search_pairs(
        X,
        Y,
        metric="cosine",
        margin="ratio",
        margin_k=4,
        # cache_dir=cache_dir,
        verbose=True)

    n_correct = 0
    lines = []
    for i in range(X.shape[0]):
        x = X_indices[i]
        if x not in Y_indices:
            x = "-"
        y = Y_indices[pred_x2y_indices[i]]
        if x == y:
            n_correct += 1
        d = pred_x2y_distances[i]
        lines.append(f"{x}\t{y}\t{d}\n")

    path = instance.get_predict_path(split=args.split)
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(lines)

    if args.verbose:
        print(f"\nAccuracy: {n_correct/len(lines):.1%}")

    return True


def main():
    parser = ArgumentParser()
    XlingBenchmark.add_argparse_arguments(parser)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--split", default="test")
    parser.add_argument("-d", "--digest", default=None)
    parser.add_argument("-i", "--interactive", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    not args.verbose or print("Loading instances")

    benchmark = XlingBenchmark.from_yaml(args.config)
    instances = benchmark.instances(args=args)
    if not instances:
        print("No instances found with filter:", args)
        exit(1)

    not args.verbose or print(f"{len(instances)} instances")

    if args.interactive:
        ok = input("Start predicting? [Y/n]: ").lower()
        if ok == "n":
            exit(0)
    
    if args.reverse:
        instances = list(reversed(instances))

    n_new, n_skipped, n_unavailable = 0, 0, 0
    for instance in instances:
        if args.digest:
            instance.source_instance.training_digest = args.digest
            instance.training_digest = args.digest
        else:
            instance.set_best_digest()

        if not instance.training_digest:
            continue

        try:
            predict_path = instance.get_predict_path(split=args.split)
        except UnavailableInstanceError as e:
            print(f"WARNING: {e}")
            n_unavailable += 1
            continue
        if not args.force and predict_path.exists() and os.path.getsize(predict_path) > 0:
            if args.verbose:
                print(f"{predict_path} already exists")
            continue

        results_path = instance.get_predict_path(split=args.split, ext="results.json")
        if results_path.exists():
            os.remove(results_path)

        print(f"\n{Style.BRIGHT}{Fore.GREEN}▶ {predict_path}{Style.RESET_ALL}:")

        if instance.task.task_type.endswith("-retrieval"):
            ok = run_retrieval(instance, args)
        else:
            ok = run_predict(instance, args)
        
        if ok:
            n_new += 1
        else:
            n_skipped += 1
    
    print("\nDone!")
    print(f"{n_new:,} new predictions")
    print(f"{n_skipped:,} skipped predictions")
    print(f"{n_unavailable:,} unavailable predictions")


if __name__ == "__main__":
    main()
