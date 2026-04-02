"""
基于 micro-sam + ZMemoryAdapter 的 3D Z-stack 跨层追踪评估脚本。

特点：
1. 使用 micro-sam 接口加载原生 SAM predictor。
2. 手动注入 ZMemoryAdapter 到 sam_model.memory_adapter。
3. 支持从 `best.pt` 这类训练 checkpoint 中安全加载 `model_state`，
   同时兼容单独保存的 adapter checkpoint（包括 `torch.save(module)` 和 state_dict）。
4. 第 0 层使用 GT 掩码中心点作为正点提示，后续层仅依赖上一层 memory_state 自回归推理。
5. 返回每一层 IoU，并提供折线图可视化。

推荐放在 Jupyter / VS Code 中按 cell 运行，或直接作为脚本执行：

python notebooks/cross_slice_tracking_evaluation.py ^
    --image-volume path/to/image_volume.npy ^
    --gt-volume path/to/gt_volume.npy ^
    --model-type vit_b ^
    --finetuned-checkpoint best.pt ^
    --adapter-checkpoint best_mem_adapter.pt ^
    --output-dir outputs/cross_slice_eval
"""

# %%
from __future__ import annotations

import argparse
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from micro_sam import util as sam_util
from micro_sam.sam_annotator.z_memory_adapter import ZMemoryAdapter


# %%
def _normalize_checkpoint_key(key: str) -> str:
    """统一去掉训练时可能带上的前缀，便于后续做兼容加载。"""
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


def _load_checkpoint_state_dict(
    checkpoint_path: str,
    map_location: str | torch.device = "cpu",
) -> OrderedDict:
    """
    兼容多种 checkpoint 存储格式，统一提取成 state_dict。

    支持：
    1. torch_em / micro-sam 训练格式：{"model_state": ...}
    2. 直接保存的 state_dict
    3. torch.save(module) 直接保存的 nn.Module
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if isinstance(checkpoint, torch.nn.Module):
        return OrderedDict(checkpoint.state_dict())

    if not isinstance(checkpoint, dict):
        raise TypeError(f"无法识别的 checkpoint 格式: {type(checkpoint)}")

    if "model_state" in checkpoint and isinstance(checkpoint["model_state"], dict):
        return OrderedDict(checkpoint["model_state"])
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        return OrderedDict(checkpoint["state_dict"])

    return OrderedDict(checkpoint)


def _extract_adapter_state_dict(state_dict: Dict[str, torch.Tensor]) -> OrderedDict:
    """从大 checkpoint 中提取 memory_adapter 子模块权重，也兼容纯 adapter state_dict。"""
    adapter_state = OrderedDict()

    for key, value in state_dict.items():
        normalized_key = _normalize_checkpoint_key(key)
        if normalized_key.startswith("memory_adapter."):
            adapter_state[normalized_key[len("memory_adapter."):]] = value

    if adapter_state:
        return adapter_state

    # 兼容单独保存的 adapter state_dict。
    adapter_roots = ("down_proj.", "spatial_align.", "up_proj.", "gamma")
    for key, value in state_dict.items():
        normalized_key = _normalize_checkpoint_key(key)
        if normalized_key.startswith(adapter_roots) or normalized_key == "gamma":
            adapter_state[normalized_key] = value

    return adapter_state


def _prepare_sam_state_dict(state_dict: Dict[str, torch.Tensor]) -> OrderedDict:
    """对 checkpoint 键名做规范化，供 sam_model.load_state_dict(strict=False) 使用。"""
    normalized_state = OrderedDict()
    for key, value in state_dict.items():
        normalized_state[_normalize_checkpoint_key(key)] = value
    return normalized_state


def _remove_adapter_keys_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> OrderedDict:
    """只保留 SAM 主干权重，显式排除 memory_adapter.*。"""
    sam_state = OrderedDict()
    for key, value in state_dict.items():
        if not key.startswith("memory_adapter."):
            sam_state[key] = value
    return sam_state


# %%
def load_sam_with_memory_adapter(
    model_type: str = "vit_b",
    device: Optional[str] = None,
    base_sam_checkpoint_path: Optional[str] = None,
    finetuned_checkpoint_path: Optional[str] = "best.pt",
    adapter_checkpoint_path: Optional[str] = None,
    embed_dim: int = 256,
) -> Tuple[torch.nn.Module, object, Dict[str, List[str]]]:
    """
    加载原生 SAM，并强制注入 ZMemoryAdapter。

    参数说明：
    - base_sam_checkpoint_path:
        原生 SAM / micro-sam encoder 权重路径。若为 None，则通过 model_type 自动下载/缓存。
    - finetuned_checkpoint_path:
        含 `model_state` 的微调 checkpoint，例如 `best.pt`。
        会先用 `strict=False` 加载到 SAM，再单独提取 `memory_adapter.*` 给 adapter。
    - adapter_checkpoint_path:
        单独保存的 memory adapter 权重，优先级高于 finetuned_checkpoint_path 中的 adapter 权重。
    """
    predictor = sam_util.get_sam_model(
        model_type=model_type,
        device=device,
        checkpoint_path=base_sam_checkpoint_path,
    )
    sam_model = predictor.model

    memory_adapter = ZMemoryAdapter(embed_dim=embed_dim).to(sam_model.device)
    sam_model.memory_adapter = memory_adapter

    load_report: Dict[str, List[str]] = {
        "sam_missing_keys": [],
        "sam_unexpected_keys": [],
        "adapter_missing_keys": [],
        "adapter_unexpected_keys": [],
    }

    if finetuned_checkpoint_path:
        finetuned_state = _load_checkpoint_state_dict(
            finetuned_checkpoint_path,
            map_location=sam_model.device,
        )
        normalized_state = _prepare_sam_state_dict(finetuned_state)
        sam_state = _remove_adapter_keys_from_state_dict(normalized_state)

        sam_missing, sam_unexpected = sam_model.load_state_dict(sam_state, strict=False)
        load_report["sam_missing_keys"] = [
            key for key in sam_missing if not key.startswith("memory_adapter.")
        ]
        load_report["sam_unexpected_keys"] = list(sam_unexpected)

        adapter_state = _extract_adapter_state_dict(normalized_state)
        if adapter_state:
            adapter_missing, adapter_unexpected = sam_model.memory_adapter.load_state_dict(
                adapter_state,
                strict=False,
            )
            load_report["adapter_missing_keys"] = list(adapter_missing)
            load_report["adapter_unexpected_keys"] = list(adapter_unexpected)

    if adapter_checkpoint_path:
        adapter_raw_state = _load_checkpoint_state_dict(
            adapter_checkpoint_path,
            map_location=sam_model.device,
        )
        adapter_state = _extract_adapter_state_dict(adapter_raw_state)
        if not adapter_state:
            raise RuntimeError(
                "单独的 adapter checkpoint 中没有找到可识别的 memory_adapter 权重。"
            )

        adapter_missing, adapter_unexpected = sam_model.memory_adapter.load_state_dict(
            adapter_state,
            strict=False,
        )
        load_report["adapter_missing_keys"] = list(adapter_missing)
        load_report["adapter_unexpected_keys"] = list(adapter_unexpected)

    sam_model.eval()
    sam_model.memory_adapter.eval()

    return sam_model, predictor, load_report


# %%
def normalize_slice_to_rgb(image_slice: np.ndarray) -> np.ndarray:
    """
    将单通道显微镜切片转成 SAM 可接受的 8-bit RGB。

    这里不用依赖 micro-sam 内部私有函数，采用标准 numpy 逻辑完成。
    """
    if image_slice.ndim != 2:
        raise ValueError(f"期望输入单层切片为 2D，实际得到形状: {image_slice.shape}")

    image_slice = np.asarray(image_slice, dtype=np.float32)
    finite_mask = np.isfinite(image_slice)
    if not finite_mask.any():
        raise ValueError("当前切片全是非有限值，无法归一化。")

    valid_values = image_slice[finite_mask]
    vmin, vmax = np.percentile(valid_values, [1.0, 99.0])
    if vmax <= vmin:
        vmin = float(valid_values.min())
        vmax = float(valid_values.max())

    image_slice = np.clip(image_slice, vmin, vmax)
    image_slice = (image_slice - vmin) / max(vmax - vmin, 1e-8)
    image_slice = (255.0 * image_slice).astype(np.uint8)
    image_rgb = np.repeat(image_slice[..., None], 3, axis=-1)
    return image_rgb


def get_mask_center_point(mask: np.ndarray) -> np.ndarray:
    """
    计算 GT 掩码的中心点，并返回 SAM 所需的 [x, y] 坐标。

    为了保证点一定落在前景内，这里会取“距离质心最近的前景像素”。
    """
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        raise ValueError("初始化切片的 GT 掩码为空，无法生成正点提示。")

    centroid = coords.mean(axis=0)
    distances = np.sum((coords - centroid[None, :]) ** 2, axis=1)
    y, x = coords[np.argmin(distances)]
    return np.array([[float(x), float(y)]], dtype=np.float32)


def mask_to_tensor(mask: np.ndarray, device: torch.device) -> torch.Tensor:
    """将二值 mask 转成 [B, 1, H, W] 的 float tensor。"""
    return torch.from_numpy(mask.astype(np.float32))[None, None].to(device)


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """计算单层 IoU；若 pred 和 gt 都为空，则记为 1.0。"""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0
    return float(intersection / union)


# %%
@torch.no_grad()
def predict_first_slice_with_gt_point(
    predictor,
    gt_slice: np.ndarray,
    image_slice: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """
    第 0 层初始化：
    - 使用 GT 中心点作为正点 prompt
    - 获取初始预测 mask
    - 获取当前层 image embeddings
    """
    point_coords = get_mask_center_point(gt_slice)
    point_labels = np.array([1], dtype=np.int32)

    predictor.set_image(normalize_slice_to_rgb(image_slice))
    masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False,
        return_logits=False,
    )

    pred_mask = masks[0].astype(bool)
    current_embed = predictor.features
    return pred_mask, point_coords, current_embed


@torch.no_grad()
def predict_slice_from_memory(
    sam_model: torch.nn.Module,
    predictor,
    image_slice: np.ndarray,
    memory_state: Dict[str, torch.Tensor],
    mask_threshold: float = 0.5,
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    后续层推理：
    - 不使用任何 point / box prompt
    - 仅依赖上一层 memory_state 生成 dense prompt
    """
    predictor.set_image(normalize_slice_to_rgb(image_slice))
    current_embed = predictor.features

    sparse_embeddings, dense_embeddings = sam_model.memory_adapter.get_prompts(
        image_embeddings=current_embed,
        memory_state=memory_state,
    )

    low_res_masks, _ = sam_model.mask_decoder(
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
    mask_tensor = (mask_prob > mask_threshold).float()
    pred_mask = mask_tensor[0, 0].detach().cpu().numpy().astype(bool)
    return pred_mask, current_embed


# %%
@torch.no_grad()
def evaluate_cross_slice_tracking(
    sam_model: torch.nn.Module,
    predictor,
    image_volume: np.ndarray,
    gt_volume: np.ndarray,
    mask_threshold: float = 0.5,
) -> Dict[str, object]:
    """
    核心零样本跨层追踪评估循环。

    输入：
    - image_volume: [Z, H, W]
    - gt_volume: [Z, H, W]，单细胞 GT

    输出：
    - pred_volume: [Z, H, W]
    - iou_per_slice: 长度为 Z 的 IoU 列表
    - init_point: 第 0 层使用的 GT 中心点
    """
    if image_volume.ndim != 3 or gt_volume.ndim != 3:
        raise ValueError(
            f"image_volume 和 gt_volume 都必须是 [Z, H, W]。"
            f"当前形状分别为 {image_volume.shape} 和 {gt_volume.shape}"
        )
    if image_volume.shape != gt_volume.shape:
        raise ValueError(
            f"image_volume 与 gt_volume 的形状不一致: "
            f"{image_volume.shape} vs {gt_volume.shape}"
        )

    gt_volume = (gt_volume > 0).astype(np.uint8)
    z_slices = image_volume.shape[0]

    pred_volume = np.zeros_like(gt_volume, dtype=np.uint8)
    iou_per_slice: List[float] = []

    memory_state = sam_model.memory_adapter.init_memory(
        batch_size=1,
        device=sam_model.device,
    )

    # Z = 0: GT 中心点初始化
    first_pred_mask, init_point, current_embed = predict_first_slice_with_gt_point(
        predictor=predictor,
        gt_slice=gt_volume[0],
        image_slice=image_volume[0],
    )
    pred_volume[0] = first_pred_mask.astype(np.uint8)
    iou_per_slice.append(compute_iou(first_pred_mask, gt_volume[0]))

    first_mask_tensor = mask_to_tensor(first_pred_mask, sam_model.device)
    memory_state = sam_model.memory_adapter.update_memory(
        memory_state=memory_state,
        current_image_embeddings=current_embed,
        current_mask=first_mask_tensor,
    )

    # Z = 1 ... Z-1: 仅依赖 memory_state 自回归
    for z in range(1, z_slices):
        pred_mask, current_embed = predict_slice_from_memory(
            sam_model=sam_model,
            predictor=predictor,
            image_slice=image_volume[z],
            memory_state=memory_state,
            mask_threshold=mask_threshold,
        )

        pred_volume[z] = pred_mask.astype(np.uint8)
        iou_per_slice.append(compute_iou(pred_mask, gt_volume[z]))

        current_mask_tensor = mask_to_tensor(pred_mask, sam_model.device)
        memory_state = sam_model.memory_adapter.update_memory(
            memory_state=memory_state,
            current_image_embeddings=current_embed,
            current_mask=current_mask_tensor,
        )

    return {
        "pred_volume": pred_volume,
        "iou_per_slice": iou_per_slice,
        "init_point": init_point,
    }


# %%
def plot_slice_iou_curve(
    iou_per_slice: List[float],
    title: str = "Cross-slice Tracking IoU",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """绘制 Z-slice vs IoU 曲线，用于观察记忆衰减。"""
    z_axis = np.arange(len(iou_per_slice))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(z_axis, iou_per_slice, marker="o", linewidth=2)
    ax.set_xlabel("Z-slice")
    ax.set_ylabel("IoU")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, ax


# %%
def print_load_report(load_report: Dict[str, List[str]]) -> None:
    """打印 checkpoint 加载摘要，便于在服务器日志里快速排查。"""
    print("=" * 80)
    print("Checkpoint Load Report")
    print("=" * 80)
    print(f"SAM missing keys      : {len(load_report['sam_missing_keys'])}")
    print(f"SAM unexpected keys   : {len(load_report['sam_unexpected_keys'])}")
    print(f"Adapter missing keys  : {len(load_report['adapter_missing_keys'])}")
    print(f"Adapter unexpected keys: {len(load_report['adapter_unexpected_keys'])}")

    if load_report["sam_missing_keys"]:
        print("First 20 SAM missing keys:")
        print(load_report["sam_missing_keys"][:20])
    if load_report["sam_unexpected_keys"]:
        print("First 20 SAM unexpected keys:")
        print(load_report["sam_unexpected_keys"][:20])
    if load_report["adapter_missing_keys"]:
        print("Adapter missing keys:")
        print(load_report["adapter_missing_keys"])
    if load_report["adapter_unexpected_keys"]:
        print("Adapter unexpected keys:")
        print(load_report["adapter_unexpected_keys"])


def run_cross_slice_tracking_evaluation(
    image_volume: np.ndarray,
    gt_volume: np.ndarray,
    model_type: str = "vit_b",
    device: Optional[str] = None,
    base_sam_checkpoint_path: Optional[str] = None,
    finetuned_checkpoint_path: Optional[str] = "best.pt",
    adapter_checkpoint_path: Optional[str] = None,
    mask_threshold: float = 0.5,
    plot_title: str = "Cross-slice Tracking IoU",
    curve_save_path: Optional[str] = None,
) -> Dict[str, object]:
    """
    一站式入口：
    1. 初始化模型
    2. 执行跨层追踪评估
    3. 绘制 IoU 曲线
    """
    sam_model, predictor, load_report = load_sam_with_memory_adapter(
        model_type=model_type,
        device=device,
        base_sam_checkpoint_path=base_sam_checkpoint_path,
        finetuned_checkpoint_path=finetuned_checkpoint_path,
        adapter_checkpoint_path=adapter_checkpoint_path,
    )
    print_load_report(load_report)

    results = evaluate_cross_slice_tracking(
        sam_model=sam_model,
        predictor=predictor,
        image_volume=image_volume,
        gt_volume=gt_volume,
        mask_threshold=mask_threshold,
    )

    fig, ax = plot_slice_iou_curve(
        results["iou_per_slice"],
        title=plot_title,
        save_path=curve_save_path,
    )
    results["figure"] = fig
    results["axes"] = ax
    results["load_report"] = load_report
    return results


# %%
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 ZMemoryAdapter 的跨层追踪表现")
    parser.add_argument("--image-volume", type=str, required=True, help="3D 图像体 `.npy` 路径，形状 [Z, H, W]")
    parser.add_argument("--gt-volume", type=str, required=True, help="3D GT 掩码 `.npy` 路径，形状 [Z, H, W]")
    parser.add_argument("--model-type", type=str, default="vit_b", help="micro-sam 模型类型，如 vit_b / vit_l")
    parser.add_argument("--device", type=str, default=None, help="推理设备，如 cuda / cpu")
    parser.add_argument(
        "--base-sam-checkpoint",
        type=str,
        default=None,
        help="可选：原生 SAM 或 micro-sam 基础 checkpoint 路径",
    )
    parser.add_argument(
        "--finetuned-checkpoint",
        type=str,
        default="best.pt",
        help="可选：含 `model_state` 的微调 checkpoint，例如 best.pt",
    )
    parser.add_argument(
        "--adapter-checkpoint",
        type=str,
        default=None,
        help="可选：单独保存的 memory adapter checkpoint，优先覆盖 finetuned checkpoint 中的 adapter 权重",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="预测 mask 的二值化阈值")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录，用于保存结果与曲线")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_volume = np.load(args.image_volume)
    gt_volume = np.load(args.gt_volume)

    curve_save_path = None
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        curve_save_path = os.path.join(args.output_dir, "iou_curve.png")

    results = run_cross_slice_tracking_evaluation(
        image_volume=image_volume,
        gt_volume=gt_volume,
        model_type=args.model_type,
        device=args.device,
        base_sam_checkpoint_path=args.base_sam_checkpoint,
        finetuned_checkpoint_path=args.finetuned_checkpoint,
        adapter_checkpoint_path=args.adapter_checkpoint,
        mask_threshold=args.mask_threshold,
        curve_save_path=curve_save_path,
    )

    print("Per-slice IoU:")
    print(results["iou_per_slice"])
    print(f"Mean IoU: {np.mean(results['iou_per_slice']):.4f}")

    if args.output_dir is not None:
        np.save(os.path.join(args.output_dir, "pred_volume.npy"), results["pred_volume"].astype(np.uint8))
        np.save(os.path.join(args.output_dir, "iou_per_slice.npy"), np.asarray(results["iou_per_slice"], dtype=np.float32))
        print(f"结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
