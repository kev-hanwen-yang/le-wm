# Fork note — Long-horizon rollout characterization of LeWM

> **This is a fork of [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm).**
> The upstream README (model, training, planning) is preserved below. This fork
> **adds an empirical characterization of LeWM's predictor under long-horizon
> open-loop rollout** on Push-T. All analysis here uses the released checkpoint
> **without retraining**.
>
> **Blog post:** [Diagnosing LeWM's Long-Horizon Rollouts: Stable Latents, Wrong Futures](https://kev-hanwen-yang.github.io/blog/2026/lewm-latent-dynamics/)

## What this fork adds

Upstream LeWM evaluates the model with frozen-encoder probes at horizon zero and
aggregate CEM planning success. This fork asks a different question: when the
predictor is rolled forward open-loop for many steps — feeding on its own
predictions — does it preserve the task-relevant physical state needed for
planning?

The short answer: the predicted latents stay globally well-formed (their norm
stays inside the encoder's reference shell) but lose target-specific physical
state — they move too slowly, too straight, and drift off the manifold of real
latents. A teacher-forced control localizes the main cause to the one-step map rather
than to compounding feedback. See the blog post for the full argument.

All numbers reported in the blog are read from the JSON report files committed in
this repo, not from the charts. Every figure can be regenerated from the scripts
below.

## What is mine vs. upstream

**Added in this fork (analysis):**

- `probe.py` — Entry point for the diagnostic probes
- `probe/ scripts/` — scripts for running the diagnostics and reports
- `diagnostic_results/` — committed JSON reports and generated charts (source of truth for all reported numbers)
- `test_...py` — tests for the diagnostics results

**Upstream:** `jepa.py`, `module.py`, `train.py`, `eval.py`, `utils.py`, `config/`, `assets/`. I added comments to explain the architecture and the flow of training/inference.

## Reproducing the characterization

A full command-by-command walkthrough — exact arguments, run order, expected
runtimes — will live in [reproduce.md](./reproduce.md).

## Note on attribution

All credit for LeWM (the model, training objective, and original codebase) belongs
to the original authors (Maes, Le Lidec, Scieur, LeCun, Balestriero). This fork's
contribution is limited to the rollout-characterization analysis described above.
The original README follows.

---

# LeWorldModel

### Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

[Lucas Maes\*](https://x.com/lucasmaes_), [Quentin Le Lidec\*](https://quentinll.github.io/), [Damien Scieur](https://scholar.google.com/citations?user=hNscQzgAAAAJ&hl=fr), [Yann LeCun](https://yann.lecun.com/) and [Randall Balestriero](https://randallbalestriero.github.io/)

**Abstract:** Joint Embedding Predictive Architectures (JEPAs) offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pretrained encoders, or auxiliary supervision to avoid representation collapse. In this work, we introduce LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings. This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative. With ~15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48× faster than foundation-model-based world models while remaining competitive across diverse 2D and 3D control tasks. Beyond control, we show that LeWM's latent space encodes meaningful physical structure through probing of physical quantities. Surprise evaluation confirms that the model reliably detects physically implausible events.

<p align="center">
   <b>[ <a href="https://arxiv.org/pdf/2603.19312v1">Paper</a> | <a href="https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e?usp=sharing">Checkpoints</a> | <a href="https://huggingface.co/collections/quentinll/lewm">Data</a> | <a href="https://le-wm.github.io/">Website</a> ]</b>
</p>

<br>

<p align="center">
  <img src="assets/lewm.gif" width="80%">
</p>

If you find this code useful, please reference it in your paper:

```
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## Using the code

This codebase builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for environment management, planning, and evaluation, and [stable-pretraining](https://github.com/galilai-group/stable-pretraining) for training. Together they reduce this repository to its core contribution: the model architecture and training objective.

**Installation:**

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

## Data

Datasets use the HDF5 format for fast loading. Download the data from [HuggingFace](https://huggingface.co/collections/quentinll/lewm) and decompress with:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the extracted `.h5` files under `$STABLEWM_HOME` (defaults to `~/.stable-wm/`).

For a self-contained project-local setup, point it at a `.stable-wm/` folder inside this repo:

```bash
export STABLEWM_HOME="$PWD/.stable-wm"
mkdir -p "$STABLEWM_HOME"
```

You can still override it to any other shared storage path if you prefer.

Dataset names are specified without the `.h5` extension. For example, `config/train/data/pusht.yaml` references `pusht_expert_train`, which resolves to `$STABLEWM_HOME/pusht_expert_train.h5`.

## Training

`jepa.py` contains the PyTorch implementation of LeWM. Training is configured via [Hydra](https://hydra.cc/) config files under `config/train/`.

Before training, set your WandB `entity` and `project` in `config/train/lewm.yaml`:

```yaml
wandb:
  config:
    entity: your_entity
    project: your_project
```

To launch training:

```bash
python train.py data=pusht
```

Checkpoints are saved to `$STABLEWM_HOME` upon completion.

For baseline scripts, see the stable-worldmodel [scripts](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train) folder.

## Planning

Evaluation configs live under `config/eval/`. Set the `policy` field to the checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix:

```bash
# ✓ correct
python eval.py --config-name=pusht.yaml policy=pusht/lewm

# ✗ incorrect
python eval.py --config-name=pusht.yaml policy=pusht/lewm_object.ckpt
```

## Pretrained Checkpoints

Pre-trained checkpoints are available on [Google Drive](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e). Download the checkpoint archive and place the extracted files under `$STABLEWM_HOME/`.

<div align="center">

|    Method     | two-room | pusht | cube | reacher |
| :-----------: | :------: | :---: | :--: | :-----: |
|     pldm      |    ✓     |   ✓   |  ✓   |    ✓    |
|    lejepa     |    ✓     |   ✓   |  ✓   |    ✓    |
|      ivl      |    ✓     |   ✓   |  ✓   |    —    |
|      iql      |    ✓     |   ✓   |  ✓   |    —    |
|     gcbc      |    ✓     |   ✓   |  ✓   |    —    |
|    dinowm     |    ✓     |   ✓   |  —   |    —    |
| dinowm_noprop |    ✓     |   ✓   |  ✓   |    ✓    |

</div>

## Loading a checkpoint

Each tar archive contains two files per checkpoint:

- `<name>_object.ckpt` — a serialized Python object for convenient loading; this is what `eval.py` and the `stable_worldmodel` API use
- `<name>_weight.ckpt` — a weights-only checkpoint (`state_dict`) for cases where you want to load weights into your own model instance

To load the object checkpoint via the `stable_worldmodel` API:

```python
import stable_worldmodel as swm

# Load the cost model (for MPC)
cost = swm.policy.AutoCostModel('pusht/lewm')
```

This function accepts:

- `run_name` — checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix
- `cache_dir` — optional override for the checkpoint root (defaults to `$STABLEWM_HOME`)

The returned module is in `eval` mode with its PyTorch weights accessible via `.state_dict()`.

## Contact & Contributions

Feel free to open [issues](https://github.com/lucas-maes/le-wm/issues)! For questions or collaborations, please contact `lucas.maes@mila.quebec`
