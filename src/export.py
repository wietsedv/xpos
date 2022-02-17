#!/usr/bin/env python3

from argparse import ArgumentParser
import os
from pathlib import Path
import shutil
from typing import Any

from transformers.pipelines import pipeline

from lib.benchmark import XlingBenchmark


def main():
    parser = ArgumentParser()
    XlingBenchmark.add_argparse_arguments(parser)
    parser.add_argument("-d", "--digest", required=True)
    parser.add_argument("-e", "--export", default="models", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    benchmark = XlingBenchmark.from_yaml(args.config)
    instances = benchmark.instances(args=args)
    if not instances:
        print("No instances found")
        exit(1)

    out_dir = args.export / args.digest
    os.makedirs(out_dir, exist_ok=True)

    model_ids = set()

    for instance in instances:
        p = Path(instance.model_path)  # type: ignore
        if not (p / "pytorch_model.bin").exists():
            print(f"{instance.model_path} does not exist")
            exit(1)

        instance.source_instance.training_digest = args.digest
        instance.training_digest = args.digest

        model_id = f"{instance.model_id}-{instance.model_config.config_name}-{instance.task.task_name}-{instance.source_instance.language}"
        if model_id in model_ids:
            continue
        model_ids.add(model_id)

        save_path = out_dir / model_id

        # Model files
        if args.force or not (save_path / "pytorch_model.bin").exists():
            model = instance.load_model()
            tokenizer: Any = instance.load_tokenizer()
            pipe = pipeline(instance.task.task_type, model=model, tokenizer=tokenizer)
            pipe.save_pretrained(save_path)

            shutil.copyfile(p / "train.args", save_path / "train.args")

        print(f"{instance.model_path} => {save_path}")


if __name__ == "__main__":
    main()
