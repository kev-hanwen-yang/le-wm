import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


# core training/validation step logic for LeWM: It takes one batch of frames and actions, 
# encodes frame and actions into latent embeddings, predicts the next latent frame embeddings, 
# computes the two LeWM losses, logs them, and returns a dict containing the total loss.
def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size # how many past timesteps are given as context (last 3 frames' embeddings + actions)
    n_preds = cfg.wm.num_preds # how many steps the predictor predicts into the future (1 = "next" frame's embedding)
    lambd = cfg.loss.sigreg.weight # how strongly to weight of the SIGReg loss (0.09 = 9% of the total loss)

    # Replace NaN values with 0 (occurs at sequence boundaries) to keep the forward pass numerical and matches "no action" at padded positions 
    # Here is the combined action token (e.g. [233, 711, 333, 711, 433, 711, 533, 711, 633, 711])
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    # encode the frames and actions into embeddings
    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D): Batch size, Sequence length (3), Latent dimension (192 for ViT-Tiny) - Latent representation of image frames
    act_emb = output["act_emb"] # Latent representation of action tokens, encoded by the action encoder. (combined action tokens)

    # This is the most important part of the training step:
    # with ctx_len = 3 and n_preds = 1, we are training the model to predict next embedding at each timestep from the available history up to that timestep,
    # so the later predictions use more context:
    # - emb contains     [e0, e1, e2, e3]
    # - ctx_emb contains [e0, e1, e2]
    # - ctx_act contains [a0, a1, a2]
    # - tgt_emb contains [e1, e2, e3]
    # So the predictor learns a shifted next-embedding prediction, the pred_emb and tgt_emb ends up with the same shape [e1, e2, e3]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]
    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean() # MSE in latent space, pred_loss gives gradients to predictor/action encoder too
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1)) # SIGReg only regularize frame embeddings, not action embeddings, it expects shape (T, B, D), so the transpose changes from (B, T, D) to (T, B, D).
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output # It returns: 1. emb 2. act_emb 3. pred_loss 4. sigreg_loss  5. loss 

# Full training entrypoint for LeWM: it sets up the dataset, model, optimizer, and trainer, and runs the training loop.
@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    # data preprocessing: for non-action columns like pixels, state, proprio, the dataset does data = data[::frameskip], so they are downsampled,
    # for action: it does not downsample, but just combines the 5 raw actions into a single token.
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None) 
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    # split the dataset into train and validation sets
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    # ViT backbone image encoder:
    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False, # we don't use pretrained weights & features for the encoder
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size # hidden dimension of the encoder (192 for ViT-Tiny)
    embed_dim = cfg.wm.get("embed_dim", hidden_dim) # latent dimension used by LeWM.
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim # action token size after accounting for frameskip.
    # PushT uses frameskip: 5, action_dim = 2, then each model step / each frame corresponds to 5 low-level actions (e.g. a0:[233, 711], a1:[333, 711], a2:[433, 711], a3:[533, 711], a4:[633, 711])
    # then it combines these five raw actions into a single action token so the action input effectively has dimension 5 * 2 = 10
    # This is important: the world model predicts at the temporally downsampled step size, 
    # so the action token has to summarize all low-level actions between two frames.
    # The action tokens are already combiend into a single token at the data preprocessing step,
    # The purpose of it is just to tell the action encoder what data format to expect.

    # Autoregressive predictor for next-step embedding prediction:
    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    # Action encoder: turns the combined action vectors (5 raw actions) into learned action embeddings
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    # Projectors: maps encoder output into the latent space used for JEPA training
    # Why project?:
    # The image encoder and predictor uses hidden_dim, while LeWM uses a separate latent training space embed_dim, so need to project both.
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    # Projector: maps predictor output into the same latent space as targets
    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    # Assemble the JEPA model
    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    max_steps = max(1, len(train) * int(cfg.trainer.max_epochs))
    warmup_steps = max(1, int(0.01 * max_steps))

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": warmup_steps,
                "max_steps": max_steps,
                "warmup_start_lr": 0.0,
                "eta_min": 0.0,
            },
            "interval": "step",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
