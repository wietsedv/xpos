from collections import defaultdict
from functools import partial
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple
from colorama.ansi import Style
import numpy as np
from colorama import Fore
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from lib.benchmark import BenchmarkInstance, UnavailableInstanceError, XlingBenchmark
from lib.utils import print_instance


def read_preds(path):
    res = None
    with open(path) as f:
        for line in f:
            vals = line.rstrip("\n").split("\t")
            if res is None:
                res = [[] for _ in vals]
            assert len(vals) == len(res)
            for i in range(len(vals)):
                res[i].append(vals[i] if vals[i] != "-" else "")
    if res is None:
        print(f"WARNING: {path} is empty", file=sys.stderr)
    return res


def bucc_threshold(y_true, y_pred, distances):
    ntrue = sum(bool(x) for x in y_true)
    npred, ncorrect = 0, 0
    best_f1 = 0
    threshold = 0
    for dist, true, pred in sorted(zip(distances, y_true, y_pred)):
        npred += 1
        if true and true == pred:
            ncorrect += 1
        if ncorrect > 0:
            p = ncorrect / npred
            r = ncorrect / ntrue
            f1 = 2 * p * r / (p + r)
            if f1 > best_f1:
                best_f1 = f1
                threshold = dist
    return threshold


def bucc_f1(y_true, y_pred, distances=None, threshold=None):
    if distances is not None:
        distances = [float(d) for d in distances]
    if threshold is None:
        threshold = bucc_threshold(y_true, y_pred, distances)

    ntrue, npred, ncorrect = 0, 0, 0
    for i in range(len(y_true)):
        true = y_true[i]
        pred = y_pred[i] if distances is None or distances[i] <= threshold else None

        if not (true or pred):
            continue
        if true:
            ntrue += 1
        if pred:
            npred += 1
        if true == pred:
            ncorrect += 1
    p = ncorrect / npred if npred > 0 else 0
    r = ncorrect / ntrue if ntrue > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return f1


def squad_score(y_true, y_pred, em=True, f1=True):
    from datasets import load_metric
    metric = load_metric("squad")

    references = [json.loads(y) for y in y_true]
    predictions = [{"id": ex["id"], "prediction_text": pred_text} for ex, pred_text in zip(references, y_pred)]

    assert len(references) == len(predictions)

    res = metric.compute(predictions=predictions, references=references)
    assert res is not None

    if em and f1:
        score = (res["exact_match"] + res["f1"]) / 2
    elif em:
        score = res["exact_match"]
    elif f1:
        score = res["f1"]
    else:
        raise ValueError("enable either em or f1")

    return score / 100


METRIC_FN = {
    "accuracy": lambda *x: accuracy_score(x[0], x[1]),
    "f1": f1_score,
    "micro_f1": partial(f1_score, average="micro"),
    "macro_f1": partial(f1_score, average="macro"),
    "bucc_f1": bucc_f1,
    "squad_em": partial(squad_score, em=True, f1=False),
    "squad_f1": partial(squad_score, em=False, f1=True),
    "squad_em_f1": partial(squad_score, em=True, f1=True),
    "precision": precision_score,
    "recall": recall_score,
}


def read_results(instances: List[BenchmarkInstance], args):
    model_types = []
    results = defaultdict(lambda: {})
    task_digests: Dict[Tuple[str, str], List[str]] = defaultdict(lambda: [])

    for instance in tqdm(instances, dynamic_ncols=True, leave=False):
        instance: BenchmarkInstance

        best_eval_score = None
        if args.digest:
            for d in args.digest:
                instance.source_instance.training_digest = d
                instance.training_digest = d
                best_eval_score = instance.get_score(d)
                if best_eval_score is not None:
                    break
        else:
            _, best_eval_score = instance.set_best_digest()

        source_instance = instance.source_instance

        row = (
            instance.task.task_type,
            instance.task.task_name,
            *([instance.digest] if args.digest is not False else []),
            source_instance.language,
            *([] if args.pool else [instance.language]),
        )

        # model_type = f"{instance.model_id}_{train_instance.model_config.config_name}"
        model_type = (instance.model_id, instance.model_config.config_name)
        if model_type not in model_types:
            model_types.append(model_type)

        if args.source:
            if np.isnan(best_eval_score):
                best_score = best_eval_score
            else:
                best_score = instance.get_score(train=True) if args.train else best_eval_score
                assert best_score is not None
            if instance.task.minimize:
                best_score = -best_score
            results[row][model_type] = [best_score]
            continue

        # There is a trained model
        if instance.training_digest:
            task_id = (source_instance.task.task_type, source_instance.task.task_name)
            task_digests[task_id].append(instance.training_digest)

            score = None

            try:
                results_path = instance.get_predict_path(split=args.split, ext="results.json")
                if results_path.exists():
                    with open(results_path) as f:
                        score = json.load(f).get(instance.task.metric, None)
                predict_path = instance.get_predict_path(split=args.split)
            except UnavailableInstanceError:
                predict_path = None
                results_path = None

            if score is None and predict_path and predict_path.exists():
                if score is None:
                    res = read_preds(predict_path)
                    if res is None:
                        score = np.nan
                    else:
                        score = METRIC_FN[instance.task.metric](*res) * 100
                        if results_path:
                            with open(results_path, "w") as f:
                                json.dump({instance.task.metric: score}, f)

            if score is not None:
                if instance.task.minimize:
                    score = -score
                if args.pool:
                    if model_type not in results[row]:
                        results[row][model_type] = []
                    elif not isinstance(results[row][model_type], list):
                        continue
                    results[row][model_type].append(score)
                else:
                    results[row][model_type] = score
            elif model_type in results[row]:
                if isinstance(results[row][model_type], list):
                    results[row][model_type] = np.nan
            # else:

    suggestions = []
    if not args.quiet:
        for instance in instances:
            source_instance = instance.source_instance
            task_id = (source_instance.task.task_type, source_instance.task.task_name)
            instance_digests = source_instance.get_training_digests(only_populated=True, include_symlinks=True, include_ignored=True)  # not args.ignore_unpopulated
            for digest in task_digests[task_id]:
                if digest not in instance_digests and (source_instance, digest) not in suggestions:
                    suggestions.append((source_instance, digest))

    return dict(results), model_types, suggestions


def tabulate_results(results, model_types, args):
    headers = ["task_type", "task_name", *(["digest"] if args.digest is not False else []), "lang_train"]
    if args.pool:
        if args.pool == "mean":
            agg_fn = np.mean
        elif args.pool == "std":
            agg_fn = np.std
        elif args.pool == "median":
            agg_fn = np.median
        else:
            raise ValueError
        for row in results:
            for model_type in results[row]:
                results[row][model_type] = agg_fn(results[row][model_type])
    else:
        headers.append("lang_pred")

    rows = [list(row) + [results[row].get(model_type, None) for model_type in model_types] for row in results]

    # Remove lang_train
    if args.language_source and "," not in args.language_source:
        for row in rows:
            row.pop(headers.index("lang_train"))
        headers.remove("lang_train")

    n_headers = len(headers)

    # Sort rows
    if args.sort_first:
        rows = sorted(rows, key=lambda x: 0 if x[n_headers] is None or np.isnan(x[n_headers]) else x[n_headers], reverse=True)

    # Remove row repetition
    for i in range(len(rows) - 1, 0, -1):
        for j in range(n_headers, 0, -1):
            if rows[i][:j] == rows[i - 1][:j]:
                rows[i][:j] = [None] * j
    rows = [row for row in rows if set(row) != {None}]

    # Add global aggregate
    if rows:
        agg_row = [None] * n_headers
        for i in range(n_headers, n_headers + len(model_types)):
            agg_row.append(np.mean([row[i] if row[i] is not None else np.nan for row in rows]))
        # rows.append([None] * n_headers + ["---"] * len(model_types))  # type: ignore
        # rows.append([None] * (n_headers + len(model_types)))
        rows.append(agg_row)

    for i in range(len(rows)):
        if set(rows[i][n_headers:]) == {None}:
            if args.drop_na:
                rows[i] = None  # type: ignore
            continue

        for k in range(n_headers, len(rows[i])):
            if rows[i][k] and isinstance(rows[i][k], np.floating):
                rows[i][k] = np.absolute(rows[i][k])

        row_values = np.array([(v[0] if type(v) == tuple else v) or np.nan for v in rows[i][n_headers:]])
        try:
            j = np.nanargmax(row_values) + n_headers
        except ValueError:
            j = None

        any_nan = np.isnan(row_values).any()
        # if args.drop_na and any_nan:
        #     rows[i] = None  # type: ignore
        #     continue

        # Add percentages
        if args.percent:
            ref_score = rows[i][n_headers] if args.sort_first else rows[i][j] if j is not None else None
            for k in range(n_headers, len(rows[i])):
                score = rows[i][k]
                if score is None or np.isnan(score):
                    continue
                elif ref_score is None or np.isnan(ref_score):
                    rows[i][k] = f"{score:.1f}"
                else:
                    rows[i][k] = f"{score:.1f} ({score / ref_score:.0%})"

                # if not args.no_color and k != j and round(score / ref_score,2) >= 0.95:
                #     rows[i][k] = f"{Fore.BLUE}{rows[i][k]}{Fore.RESET}"

        # Highlight missing values
        if not args.no_color:
            for k in range(n_headers, len(rows[i])):
                if rows[i][k] is None:
                    rows[i][k] = ""
                elif type(rows[i][k]) == str:
                    pass
                elif np.isnan(rows[i][k]):
                    rows[i][k] = f"{Fore.RED}nan{Fore.RESET}"
                else:
                    rows[i][k] = f"{rows[i][k]:.1f}"


        # Add color (best)
        if j is not None:
            if type(rows[i][j]) != str and np.isnan(rows[i][j]):
                continue
            if not args.no_color:
                rows[i][j] = f"{Fore.BLUE if any_nan else Fore.GREEN}{rows[i][j]}{Fore.RESET}"
    
    rows = [row for row in rows if row is not None]

    model_ids = sorted({mi for mi, _ in model_types})
    model_id_colors = [Fore.BLUE, Fore.YELLOW, Fore.MAGENTA, Fore.CYAN]
    def colorize_id(mi: str):
        i = model_ids.index(mi)
        if i >= len(model_id_colors):
            return mi
        return f"{model_id_colors[i]}{mi}{Fore.RESET}"

    def colorize_type(mt: str):
        if "-" in mt:
            mt1, mt2 = mt.rsplit("-", maxsplit=1)
            return f"{colorize_type(mt1)}-{Fore.MAGENTA}{mt2}{Fore.RESET}"
        return f"{Fore.CYAN}{mt}{Fore.RESET}"

    if not args.no_color:
        model_types = [(colorize_id(mi), colorize_type(mt)) for mi, mt in model_types]
    headers.extend(["\n".join(m) for m in model_types])

    # Remove redundant task info
    skip_cols = 0
    if args.task_name is not None and "*" not in args.task_name and "|" not in args.task_name:
        skip_cols = 2
    elif args.task_type is not None and "*" not in args.task_type and "|" not in args.task_type:
        skip_cols = 1
    if skip_cols:
        headers = headers[skip_cols:]
        rows = [row[skip_cols:] for row in rows]
    return headers, rows


def export_results(results: dict, model_types: list, path: Path, args):
    import pandas as pd

    headers = ["task_type", "task_name", *(["digest"] if args.digest is not False else []), "lang_train", *(["lang_pred"] if args.pool is None else []), "model_id", "model_type", "score"]
    rows = []
    for row in results:
        for model_type in model_types:
            score = results[tuple(row)].get(model_type, None)
            if score:
                score = np.absolute(score)
            full_row = [*row, *model_type, score]
            rows.append(full_row)

    df = pd.DataFrame(rows, columns=headers)
    df.to_csv(path)
    print(f"Exported to {path}")


def main():
    from argparse import ArgumentParser
    from tabulate import tabulate

    parser = ArgumentParser()
    parser.add_argument("--source", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--percent", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--no_color", action="store_true")
    parser.add_argument("--sort_first", action="store_true")
    parser.add_argument("--drop_na", action="store_true")
    parser.add_argument("-a", "--all", action="store_true")
    # parser.add_argument("--ignore_unpopulated", action="store_true")
    parser.add_argument("-p", "--pool", default="mean", choices=["mean", "std", "median"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--train_cmd", default=os.getenv("XLING_TRAIN_CMD"))
    parser.add_argument("-e", "--export", default=None, type=Path)
    parser.add_argument("-d", "--digest", default=False, nargs="*")
    XlingBenchmark.add_argparse_arguments(parser)
    args = parser.parse_args()

    if args.train and not args.source:
        print("--train only makes sense together with --source")
        exit(1)

    if args.digest is not False:
        args.digest = [d_ for d in args.digest for d_ in d.split(",")]

    if args.all:
        args.pool = None

    not args.verbose or print("Loading benchmark")
    benchmark = XlingBenchmark.from_yaml(args.config, strict=False, verbose=args.verbose)
    not args.verbose or print("Loading instances")
    instances = benchmark.instances(source=args.source, args=args, verbose=args.verbose)
    if len(instances) == 0:
        print("No matching instances found")
        exit(1)

    not args.verbose or print("Aggregating results")
    results, model_types, suggestions = read_results(instances, args)

    headers, rows = tabulate_results(results, model_types, args)
    print(tabulate(rows, headers=headers, floatfmt=".1f"))

    if args.export:
        export_results(results, model_types, args.export, args)

    if not args.quiet:
        print(f"\n{len(instances):,} instances")

        # print('\nMissing outputs:')
        # for instance, digest in suggestions:
        #     instance: BenchmarkInstance
        #     instance.training_digest = digest
        #     print(instance.output_path)

        print('\nSuggested runs:')
        for instance, digest in suggestions:
            print_instance(instance, args.train_cmd, digest)


if __name__ == "__main__":
    main()
