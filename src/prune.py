#!/usr/bin/env python3

from argparse import ArgumentParser
import os
from pathlib import Path
import shutil
from glob import glob

from lib.benchmark import XlingBenchmark
from lib.utils import output_dir_populated

def main():
    parser = ArgumentParser()
    XlingBenchmark.add_argparse_arguments(parser)
    parser.add_argument("-y", "--yes", action="store_true")
    args = parser.parse_args()

    benchmark = XlingBenchmark.from_yaml(args.config)
    instances = benchmark.instances(train=True, args=args)
    print(f"{len(instances)} instances")

    train_dirs, orphans = [], []
    populated, unpopulated = [], []
    best, underperforming = [], []
    for instance in instances:
        if not instance.output_base.exists():
            continue
    
        train_dirs.append(instance.output_base)

        best_digest, _ = instance.set_best_digest()

        for output_dir in instance.output_base.glob("*"):
            if output_dir_populated(output_dir):
                populated.append(output_dir)
                if output_dir.name == best_digest:
                    best.append(output_dir)
                elif (output_dir / "pytorch_model.bin").exists():
                    underperforming.append(output_dir)
            else:
                unpopulated.append(output_dir)
    
    for train_dir in glob(benchmark.output_base_fmt.format(output_name="*")):
        train_dir = Path(train_dir)
        if train_dir not in train_dirs:
            orphans.append(train_dir)

    # print(f"\nPopulated ({len(populated)}):")
    # for path in populated:
    #     print(path)
    # if unpopulated:
    #     print(f"\nUnpopulated ({len(unpopulated)}):")
    #     for path in unpopulated:
    #         print(path)
    #     if input("\nRemove unpopulated output directories? [Yn]: ").lower() != "n":
    #         for output_dir in unpopulated:
    #             shutil.rmtree(output_dir)
    #             if not os.listdir(output_dir.parent):
    #                 os.rmdir(output_dir.parent)
    #             print(f"- Removed {output_dir}")
    # else:
    #     print("\nNo unpopulated output directories")

    # print(f"\nBest checkpoints ({len(best)}):")
    # for path in best:
    #     print(path)
    if underperforming:
        print(f"\nUnderperforming checkpoints ({len(underperforming)}):")
        for path in underperforming:
            print(path)
        if args.yes or input("\nRemove underperforming checkpoints? [Yn]: ").lower() != "n":
            for output_dir in underperforming:
                ckpt_path = output_dir / "pytorch_model.bin"
                if ckpt_path.exists():
                    os.remove(ckpt_path)
                print(f"- Removed {ckpt_path}")
    else:
        print("\nNo underperforming checkpoints")


    # print(f"\nTarget train dirs ({len(train_dirs)}):")
    # for path in train_dirs:
    #     print(path)
    # if orphans:
    #     print(f"\nOrphan train dirs ({len(orphans)}):")
    #     for path in orphans:
    #         print(path)
    #     if input("\nRemove orphan outputs? [yN]: ").lower() == "y":
    #         for train_dir in orphans:
    #             shutil.rmtree(train_dir)
    #             print(f"- Removed {train_dir}")
    # else:
    #     print("\nNo orphan dirs")


if __name__ == "__main__":
    main()
