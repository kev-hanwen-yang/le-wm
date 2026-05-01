#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import torch


# Script: evaluate saved probes on the held-out test split.
#
# This script assumes the encoded test split already exists under
# `.stable-wm/probes/encoded/` and that probe checkpoints already exist under
# `.stable-wm/probes/`. It produces a small Markdown table plus a JSON file.
#
# When `--probe-seeds` is provided, it evaluates checkpoints under
# `.stable-wm/probes/seed{probe_seed}/` and reports mean ± sample std across
# those probe-training seeds. The dataset split seed remains `--seed`.


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probe.evaluate import (  # noqa: E402
    DEFAULT_PROBE_TYPES,
    DEFAULT_TARGET_NAMES,
    evaluate_probe_grid,
    format_markdown_table,
    format_seed_level_markdown_table,
    load_encoded_split,
    save_report,
)


def eval_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encoded_cache_path(cache_dir, dataset_name, policy_name, split_name, seed, img_size):
    safe_policy_name = policy_name.replace("/", "_")
    filename = (
        f"{dataset_name}_{safe_policy_name}_{split_name}_"
        f"seed{seed}_img{img_size}_encoded.pt"
    )
    return cache_dir / "probes" / "encoded" / filename


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate saved Push-T probe checkpoints on cached test embeddings."
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--dataset-name", default="pusht_expert_train")
    parser.add_argument("--policy-name", default="pusht/lewm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGET_NAMES))
    parser.add_argument("--probe-types", nargs="+", default=list(DEFAULT_PROBE_TYPES))
    parser.add_argument("--probe-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--report-name", default=None)
    return parser.parse_args()


def default_cache_dir():
    project_cache_dir = REPO_ROOT / ".stable-wm"
    if project_cache_dir.exists():
        return project_cache_dir
    import stable_worldmodel as swm

    return Path(swm.data.utils.get_cache_dir())


def main():
    args = parse_args()
    cache_dir = args.cache_dir or default_cache_dir()
    device = eval_device() if args.device == "auto" else args.device

    test_cache_path = encoded_cache_path(
        cache_dir,
        args.dataset_name,
        args.policy_name,
        "test",
        args.seed,
        args.img_size,
    )
    if not test_cache_path.exists():
        raise FileNotFoundError(
            f"Missing cached test embeddings: {test_cache_path}. "
            "Run probe.py once first so the train/val/test encoded splits are cached."
        )

    test_encoded = load_encoded_split(test_cache_path)
    rows = evaluate_probe_grid(
        test_encoded,
        cache_dir=cache_dir,
        device=device,
        batch_size=args.batch_size,
        target_names=args.targets,
        probe_types=args.probe_types,
        probe_seeds=args.probe_seeds,
    )

    if args.report_name is not None:
        report_name = args.report_name
    elif args.probe_seeds is not None:
        seed_text = "_".join(str(seed) for seed in args.probe_seeds)
        report_name = (
            f"probe_test_report_split_seed{args.seed}_"
            f"probe_seeds{seed_text}_img{args.img_size}"
        )
    else:
        report_name = f"probe_test_report_seed{args.seed}_img{args.img_size}"
    markdown_path, seed_markdown_path, json_path = save_report(
        rows,
        output_dir=cache_dir / "probes" / "reports",
        report_name=report_name,
    )

    print(format_markdown_table(rows))
    print("\nSeed-level summary:")
    print(format_seed_level_markdown_table(rows))
    print(f"\nSaved Markdown report to {markdown_path}")
    print(f"Saved seed-level Markdown report to {seed_markdown_path}")
    print(f"Saved JSON report to {json_path}")


if __name__ == "__main__":
    main()
