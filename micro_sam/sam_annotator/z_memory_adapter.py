from __future__ import annotations

from collections import OrderedDict
import pickle
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def mask_to_memory_tensor(mask, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(mask):
        mask_tensor = mask.to(device=device, dtype=torch.float32)
    else:
        mask_tensor = torch.as_tensor(np.asarray(mask), dtype=torch.float32, device=device)

    if mask_tensor.ndim == 2:
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    elif mask_tensor.ndim == 3:
        mask_tensor = mask_tensor.unsqueeze(1)
    elif mask_tensor.ndim != 4:
        raise ValueError(f"Unsupported mask shape for memory update: {tuple(mask_tensor.shape)}")

    return mask_tensor


def _normalize_checkpoint_key(key: str) -> str:
    prefixes = (
        "module.model.sam.",
        "module.sam.",
        "model.sam.",
        "sam_model.",
        "module.model.",
        "module.",
        "model.",
        "sam.",
    )

    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
                break
    return key


def load_checkpoint_state_dict(
    checkpoint_path: str,
    map_location: str | torch.device = "cpu",
) -> OrderedDict:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    except pickle.UnpicklingError as err:
        if "Weights only load failed" not in str(err):
            raise
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )

    if isinstance(checkpoint, torch.nn.Module):
        return OrderedDict(checkpoint.state_dict())

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    if "model_state" in checkpoint and isinstance(checkpoint["model_state"], dict):
        return OrderedDict(checkpoint["model_state"])
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        return OrderedDict(checkpoint["state_dict"])

    return OrderedDict(checkpoint)


def prepare_sam_state_dict(state_dict: Dict[str, torch.Tensor]) -> OrderedDict:
    normalized_state = OrderedDict()
    for key, value in state_dict.items():
        normalized_state[_normalize_checkpoint_key(key)] = value
    return normalized_state


def remove_memory_adapter_keys_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> OrderedDict:
    sam_state = OrderedDict()
    for key, value in state_dict.items():
        if not key.startswith("memory_adapter."):
            sam_state[key] = value
    return sam_state


def extract_memory_adapter_state_dict(state_dict: Dict[str, torch.Tensor]) -> OrderedDict:
    adapter_state = OrderedDict()

    for key, value in state_dict.items():
        normalized_key = _normalize_checkpoint_key(key)
        if normalized_key.startswith("memory_adapter."):
            adapter_state[normalized_key[len("memory_adapter."):]] = value

    if adapter_state:
        return adapter_state

    adapter_roots = ("down_proj.", "spatial_align.", "up_proj.", "gamma")
    for key, value in state_dict.items():
        normalized_key = _normalize_checkpoint_key(key)
        if normalized_key.startswith(adapter_roots) or normalized_key == "gamma":
            adapter_state[normalized_key] = value

    return adapter_state


@torch.no_grad()
def predict_from_memory_state(
    predictor,
    memory_state: Optional[Dict[str, torch.Tensor]],
    image=None,
    image_embeddings=None,
    index: Optional[int] = None,
    mask_threshold: float = 0.5,
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
    from micro_sam import util as sam_util

    sam_model = predictor.model
    if not hasattr(sam_model, "memory_adapter"):
        raise AttributeError("The predictor model does not have a memory_adapter.")

    if image_embeddings is not None:
        sam_util.set_precomputed(predictor, image_embeddings, i=index)
    elif image is not None:
        predictor.set_image(image)
    elif not getattr(predictor, "is_image_set", False):
        raise ValueError("Pass either `image` or `image_embeddings`/`index` before memory decoding.")

    current_embed = predictor.features
    sparse_embeddings, dense_embeddings = sam_model.memory_adapter.get_prompts(
        image_embeddings=current_embed,
        memory_state=memory_state,
    )

    low_res_masks, iou_predictions = sam_model.mask_decoder(
        image_embeddings=current_embed,
        image_pe=sam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    masks = sam_model.postprocess_masks(
        low_res_masks,
        input_size=predictor.input_size,
        original_size=predictor.original_size,
    )
    mask_prob = torch.sigmoid(masks)
    pred_mask = (mask_prob > mask_threshold)[0, 0].detach().cpu().numpy().astype(bool)
    return pred_mask, current_embed, mask_prob, iou_predictions


class ZMemoryAdapter(nn.Module):
    def __init__(self, embed_dim: int = 256, detach_memory: bool = True):
        super().__init__()
        bottleneck_dim = embed_dim // 4

        self.down_proj = nn.Conv2d(embed_dim, bottleneck_dim, kernel_size=1)
        self.spatial_align = nn.Conv2d(bottleneck_dim * 2, bottleneck_dim, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.up_proj = nn.Conv2d(bottleneck_dim, embed_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.embed_dim = embed_dim
        self.detach_memory = detach_memory

    def init_memory(self, batch_size, device):
        return None

    def update_memory(self, memory_state, current_image_embeddings, current_mask):
        current_mask = mask_to_memory_tensor(current_mask, current_image_embeddings.device)
        mask_64 = F.interpolate(
            current_mask.float(),
            size=current_image_embeddings.shape[-2:],
            mode="area",
        )

        if self.detach_memory:
            current_image_embeddings = current_image_embeddings.detach()
            mask_64 = mask_64.detach()

        return {
            "prev_embed": current_image_embeddings,
            "prev_mask": mask_64,
        }

    def get_prompts(self, image_embeddings, memory_state):
        if memory_state is None:
            batch_size = image_embeddings.shape[0]
            dense_embeddings = torch.zeros_like(image_embeddings)
            sparse_embeddings = torch.empty(batch_size, 0, self.embed_dim, device=image_embeddings.device)
            return sparse_embeddings, dense_embeddings

        prev_embed = memory_state["prev_embed"]
        prev_mask = memory_state["prev_mask"]
        if prev_embed.shape[0] != image_embeddings.shape[0]:
            raise ValueError(
                "Mismatch between memory batch size and image embedding batch size: "
                f"{prev_embed.shape[0]} vs {image_embeddings.shape[0]}"
            )

        memory_feature = prev_embed * prev_mask

        curr_proj = self.down_proj(image_embeddings)
        mem_proj = self.down_proj(memory_feature)

        concat_feat = torch.cat([curr_proj, mem_proj], dim=1)
        aligned_feat = self.act(self.spatial_align(concat_feat))

        dense_embeddings = self.up_proj(aligned_feat) * self.gamma
        sparse_embeddings = torch.empty(
            image_embeddings.shape[0], 0, self.embed_dim, device=image_embeddings.device
        )

        return sparse_embeddings, dense_embeddings


def get_embedding_tensor(image_embeddings, t):
    if isinstance(image_embeddings, dict) and "features" in image_embeddings:
        features = image_embeddings["features"]
        if torch.is_tensor(features):
            return features[t]
        return torch.as_tensor(features[t])
    raise ValueError("The provided image_embeddings do not contain the expected 'features' key.")
