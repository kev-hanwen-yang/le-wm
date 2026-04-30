import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torchvision.transforms import v2 as transforms


# Component: frozen encoder inference and encoded-embedding caching.
#
# This module owns everything between raw Push-T image frames and reusable
# encoded probe data. It loads the released LeWM checkpoint as a frozen model,
# applies the same image preprocessing used during evaluation, runs encoder
# inference without gradients, and saves CPU tensors containing aligned
# embeddings/state/proprio/episode_idx/step_idx for fast probe training later.


def eval_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_image_transform(img_size: int):
    # Matches eval image preprocessing: image conversion, float scaling,
    # ImageNet normalization, then resizing to the configured model input size.
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def load_frozen_encoder(policy_name: str, device: str, cache_dir=None):
    model = swm.policy.AutoCostModel(policy_name, cache_dir=cache_dir)
    model = model.to(device)
    model.eval()  # Switch off training-time behavior such as dropout.
    model.requires_grad_(False)  # Probe training must not update encoder weights.
    model.interpolate_pos_encoding = True
    return model


def preprocess_pixels(pixels, image_transform, device):
    # HDF5Dataset returns pixels as (B, T, C, H, W), e.g. (64, 1, 3, 224, 224).
    # Torchvision transforms expect image tensors as (N, C, H, W), so flatten
    # B and T before preprocessing, then restore the model's expected sequence shape.
    b, t = pixels.shape[:2]
    pixels = pixels.reshape(b * t, *pixels.shape[2:])
    pixels = image_transform(pixels)
    pixels = pixels.reshape(b, t, *pixels.shape[1:])
    return pixels.to(device)


def make_probe_loader(dataset, indices, batch_size, shuffle):
    subset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def encode_batch(batch, model, device, image_transform):
    pixels = preprocess_pixels(batch["pixels"], image_transform, device)
    with torch.no_grad():
        emb = model.encode({"pixels": pixels})["emb"][:, 0]
    return emb.detach()


def extract_embeddings(dataset, indices, model, device, image_transform, batch_size=256):
    # Legacy helper: collects encoded embeddings and aligned labels by concatenating
    # batches. `precompute_encoded_split` is preferred because it preallocates output
    # tensors, but this remains useful for small debugging runs.
    loader = make_probe_loader(dataset, indices, batch_size=batch_size, shuffle=False)

    embeddings = []
    states = []
    proprios = []
    episode_ids = []
    step_ids = []

    with torch.inference_mode():  # Disable gradient tracking for efficient frozen encoding.
        for batch in loader:
            emb = encode_batch(batch, model, device, image_transform)
            embeddings.append(emb.cpu())  # (B, T, E) -> after [:, 0]: (B, 192)
            states.append(batch["state"][:, 0].float())
            proprios.append(batch["proprio"][:, 0].float())
            episode_ids.append(batch["episode_idx"][:, 0])
            step_ids.append(batch["step_idx"][:, 0])

    return {
        "emb": torch.cat(embeddings, dim=0),
        "state": torch.cat(states, dim=0),
        "proprio": torch.cat(proprios, dim=0),
        "episode_idx": torch.cat(episode_ids, dim=0),
        "step_idx": torch.cat(step_ids, dim=0),
    }


def precompute_encoded_split(
    dataset,
    indices,
    model,
    device,
    image_transform,
    split_name,
    batch_size,
    log_every=100,
):
    loader = make_probe_loader(dataset, indices, batch_size=batch_size, shuffle=False)
    n_samples = len(indices)  # Count how many frames are in this split.

    # Preallocate CPU storage so the expensive encoder only runs once per split.
    # Each row stays aligned: emb[i] comes from the same frame as state[i].
    encoded = {
        "emb": torch.empty((n_samples, 192), dtype=torch.float32),
        "state": torch.empty((n_samples, 7), dtype=torch.float32),
        "proprio": torch.empty((n_samples, 4), dtype=torch.float32),
        "episode_idx": torch.empty((n_samples,), dtype=torch.long),
        "step_idx": torch.empty((n_samples,), dtype=torch.long),
    }

    write_pos = 0
    with torch.inference_mode():  # Disables gradients during embedding extraction.
        for batch_idx, batch in enumerate(loader, start=1):
            emb = encode_batch(batch, model, device, image_transform)

            # The final batch is often smaller than `batch_size`; write only
            # the slice that corresponds to the actual batch length.
            batch_size_actual = emb.size(0)
            target_slice = slice(write_pos, write_pos + batch_size_actual)

            encoded["emb"][target_slice] = emb.cpu()
            encoded["state"][target_slice] = batch["state"][:, 0].float()
            encoded["proprio"][target_slice] = batch["proprio"][:, 0].float()
            encoded["episode_idx"][target_slice] = batch["episode_idx"][:, 0].long()
            encoded["step_idx"][target_slice] = batch["step_idx"][:, 0].long()

            write_pos += batch_size_actual
            if batch_idx % log_every == 0 or write_pos == n_samples:
                print(
                    f"precompute {split_name}: {write_pos}/{n_samples} "
                    f"frames encoded"
                )

    return encoded


def encoded_cache_path(cache_dir, dataset_name, policy_name, split_name, seed, img_size):
    safe_policy_name = policy_name.replace("/", "_")
    filename = (
        f"{dataset_name}_{safe_policy_name}_{split_name}_"
        f"seed{seed}_img{img_size}_encoded.pt"
    )
    return cache_dir / "probes" / "encoded" / filename


def load_or_precompute_encoded_split(
    dataset,
    indices,
    model,
    device,
    image_transform,
    split_name,
    cache_path,
    metadata,
    batch_size,
):
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        print(f"loaded cached {split_name} embeddings from {cache_path}")
        return payload["encoded"]

    encoded = precompute_encoded_split(
        dataset,
        indices,
        model,
        device,
        image_transform,
        split_name=split_name,
        batch_size=batch_size,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "encoded": encoded}, cache_path)
    print(f"saved cached {split_name} embeddings to {cache_path}")
    return encoded
