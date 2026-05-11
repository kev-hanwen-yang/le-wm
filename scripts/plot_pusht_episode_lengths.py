#!/usr/bin/env python3
"""Plot histogram of episode lengths (ep_len) from pusht_expert_train.h5."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5",
        type=Path,
        default=None,
        help="Path to pusht_expert_train.h5 (default: <repo>/.stable-wm/pusht_expert_train.h5)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: <repo>/figures/pusht_expert_train_ep_len_histogram.png)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of histogram bins (default: 40)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    h5_path = args.h5 or (repo_root / ".stable-wm" / "pusht_expert_train.h5")
    out_path = args.out or (
        repo_root / "figures" / "pusht_expert_train_ep_len_histogram.png"
    )

    if not h5_path.is_file():
        raise SystemExit(f"HDF5 not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        ep_len = f["ep_len"][:].astype(np.int64)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(ep_len, bins=args.bins, color="steelblue", edgecolor="white", alpha=0.9)
    ax.set_xlabel("Episode length (rows / timesteps)")
    ax.set_ylabel("Count (episodes)")
    ax.set_title(f"pusht_expert_train — episode length distribution (N={len(ep_len):,})")
    ax.axvline(float(np.mean(ep_len)), color="darkred", linestyle="--", linewidth=1.5, label=f"mean = {np.mean(ep_len):.1f}")
    ax.axvline(float(np.median(ep_len)), color="darkgreen", linestyle=":", linewidth=1.5, label=f"median = {np.median(ep_len):.0f}")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
