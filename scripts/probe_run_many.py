#!/usr/bin/env python
import argparse
import subprocess
import sys
from pathlib import Path


# Script: sequential multi-seed probe runner.
#
# This is intentionally small and boring: it loops over target names, probe
# types, and probe-training seeds, then launches `probe.py` once per combination.
# The dataset split seed is kept separate as `--split-seed`; each `--probe-seed`
# only changes probe initialization, DataLoader shuffle order, and MLP dropout.
# This keeps the final std interpretable as probe-training variability on a
# fixed train/val/test episode split.


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = ("agent_location", "block_location", "block_angle")
DEFAULT_PROBE_TYPES = ("linear", "mlp")


def parse_args():
    parser = argparse.ArgumentParser(description="Train many Push-T probes sequentially.")
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument("--probe-types", nargs="+", default=list(DEFAULT_PROBE_TYPES))
    parser.add_argument("--probe-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--mlp-num-hidden-layers", type=int, default=1)
    parser.add_argument("--mlp-dropout", type=float, default=0.1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_cache_dir():
    return REPO_ROOT / ".stable-wm"


def checkpoint_path(cache_dir, target_name, probe_type, probe_seed):
    return cache_dir / "probes" / f"seed{probe_seed}" / f"pusht_{target_name}_{probe_type}.pt"


def build_command(args, target_name, probe_type, probe_seed):
    command = [
        args.python,
        "probe.py",
        f"seed={args.split_seed}",
        f"+target_name={target_name}",
        f"+probe_type={probe_type}",
        f"+probe_seed={probe_seed}",
    ]
    if args.cache_dir is not None:
        command.append(f"+cache_dir={args.cache_dir}")
    if probe_type == "mlp":
        command.extend(
            [
                f"+mlp_hidden_dim={args.mlp_hidden_dim}",
                f"+mlp_num_hidden_layers={args.mlp_num_hidden_layers}",
                f"+mlp_dropout={args.mlp_dropout}",
            ]
        )
    return command


def main():
    args = parse_args()
    cache_dir = args.cache_dir or default_cache_dir()

    for probe_seed in args.probe_seeds:
        for target_name in args.targets:
            for probe_type in args.probe_types:
                path = checkpoint_path(cache_dir, target_name, probe_type, probe_seed)
                if args.skip_existing and path.exists():
                    print(f"skip existing: {path}")
                    continue

                command = build_command(args, target_name, probe_type, probe_seed)
                print("run:", " ".join(str(part) for part in command))
                if args.dry_run:
                    continue
                subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
