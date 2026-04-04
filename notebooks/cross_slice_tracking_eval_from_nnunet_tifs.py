from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
import re
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage as ndi


def _load_tracking_eval_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("cross_slice_tracking_module", module_path)
    tracking_eval = importlib.util.module_from_spec(spec)
    sys.modules["cross_slice_tracking_module"] = tracking_eval
    assert spec.loader is not None
    spec.loader.exec_module(tracking_eval)
    return tracking_eval


def _collect_slice_pairs(dataset_root: Path) -> Tuple[str, List[int], List[Path], List[Path]]:
    images_dir = dataset_root / "imagesTr"
    labels_dir = dataset_root / "labelsTr"
    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {labels_dir}")

    image_pattern = re.compile(r"^(?P<prefix>.+)_(?P<slice>\d+)_0000\.tif$")
    label_pattern = re.compile(r"^(?P<prefix>.+)_(?P<slice>\d+)\.tif$")

    image_entries: Dict[Tuple[str, int], Path] = {}
    for path in images_dir.glob("*.tif"):
        match = image_pattern.match(path.name)
        if match is None:
            continue
        key = (match.group("prefix"), int(match.group("slice")))
        image_entries[key] = path

    label_entries: Dict[Tuple[str, int], Path] = {}
    for path in labels_dir.glob("*.tif"):
        match = label_pattern.match(path.name)
        if match is None:
            continue
        key = (match.group("prefix"), int(match.group("slice")))
        label_entries[key] = path

    shared_keys = sorted(set(image_entries) & set(label_entries), key=lambda item: (item[0], item[1]))
    if not shared_keys:
        raise RuntimeError(f"No matched tif pairs found under: {dataset_root}")

    prefixes = sorted({prefix for prefix, _ in shared_keys})
    if len(prefixes) != 1:
        raise RuntimeError(f"Expected a single volume prefix, but found: {prefixes}")

    prefix = prefixes[0]
    ordered_keys = [key for key in shared_keys if key[0] == prefix]
    slice_ids = [slice_id for _, slice_id in ordered_keys]
    image_paths = [image_entries[key] for key in ordered_keys]
    label_paths = [label_entries[key] for key in ordered_keys]
    return prefix, slice_ids, image_paths, label_paths


def _load_binary_label_volume(label_paths: Sequence[Path]) -> np.ndarray:
    return np.stack([(tifffile.imread(path) > 0).astype(np.uint8) for path in label_paths], axis=0)


def _get_component_stats(
    component_volume: np.ndarray,
    num_components: int,
    slice_ids: Sequence[int],
    min_voxels: int,
    min_z_span: int,
) -> List[Dict[str, int]]:
    component_slices = ndi.find_objects(component_volume)
    component_sizes = np.bincount(component_volume.ravel())

    candidates: List[Dict[str, int]] = []
    for component_id in range(1, num_components + 1):
        bbox = component_slices[component_id - 1]
        if bbox is None:
            continue

        voxels = int(component_sizes[component_id])
        if voxels < min_voxels:
            continue

        z_slice = bbox[0]
        z_start = int(z_slice.start)
        z_stop = int(z_slice.stop)
        z_span = z_stop - z_start
        if z_span < min_z_span:
            continue

        candidates.append(
            {
                "component_id": component_id,
                "voxels": voxels,
                "z_start_index": z_start,
                "z_stop_index": z_stop,
                "z_span": z_span,
                "slice_id_start": int(slice_ids[z_start]),
                "slice_id_end": int(slice_ids[z_stop - 1]),
            }
        )

    return candidates


def _choose_components(
    candidates: Sequence[Dict[str, int]],
    num_objects: int,
    seed: int,
    preferred_max_z_span: int | None,
) -> List[Dict[str, int]]:
    if not candidates:
        raise RuntimeError("No candidate objects matched the current filtering settings.")

    pool = list(candidates)
    if preferred_max_z_span is not None:
        preferred_pool = [candidate for candidate in pool if candidate["z_span"] <= preferred_max_z_span]
        if preferred_pool:
            pool = preferred_pool

    rng = np.random.default_rng(seed)
    num_to_select = min(num_objects, len(pool))
    selected_indices = rng.choice(len(pool), size=num_to_select, replace=False)
    selected = [pool[int(index)] for index in selected_indices]
    selected.sort(key=lambda candidate: (candidate["slice_id_start"], candidate["component_id"]))
    return selected


def _load_image_subvolume(image_paths: Sequence[Path], z_start: int, z_stop: int) -> np.ndarray:
    return np.stack([tifffile.imread(path) for path in image_paths[z_start:z_stop]], axis=0)


def _save_prediction_preview(
    image_volume: np.ndarray,
    gt_volume: np.ndarray,
    pred_volume: np.ndarray,
    iou_per_slice: Sequence[float],
    save_path: Path,
) -> None:
    if image_volume.shape[0] == 0:
        return

    z_indices = np.linspace(0, image_volume.shape[0] - 1, num=min(5, image_volume.shape[0]), dtype=int)
    z_indices = list(dict.fromkeys(int(z) for z in z_indices))

    fig, axes = plt.subplots(len(z_indices), 3, figsize=(12, 3 * len(z_indices)))
    if len(z_indices) == 1:
        axes = np.asarray([axes])

    for row, z_index in enumerate(z_indices):
        axes[row, 0].imshow(image_volume[z_index], cmap="gray")
        axes[row, 0].set_title(f"Image z={z_index}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt_volume[z_index], cmap="gray")
        axes[row, 1].set_title(f"GT z={z_index}")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(image_volume[z_index], cmap="gray")
        axes[row, 2].imshow(pred_volume[z_index], cmap="jet", alpha=0.35)
        axes[row, 2].set_title(f"Prediction z={z_index} | IoU={iou_per_slice[z_index]:.3f}")
        axes[row, 2].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cross-slice tracking on nnUNet tif stacks.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--tracking-module", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--tracking-mode",
        type=str,
        default="memory",
        choices=("memory", "plain_recursive_prompt"),
    )
    parser.add_argument("--model-type", type=str, default="vit_l")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--base-sam-checkpoint", type=Path, default=None)
    parser.add_argument("--finetuned-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, default=None)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--box-padding", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-objects", type=int, default=3)
    parser.add_argument("--min-z-span", type=int, default=10)
    parser.add_argument("--preferred-max-z-span", type=int, default=80)
    parser.add_argument("--min-voxels", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root))

    module_path = args.tracking_module
    if module_path is None:
        module_path = repo_root / "notebooks" / "cross_slice_tracking_evaluation.py"
    module_path = module_path.resolve()
    tracking_eval = _load_tracking_eval_module(module_path)

    prefix, slice_ids, image_paths, label_paths = _collect_slice_pairs(args.dataset_root)
    label_volume = _load_binary_label_volume(label_paths)
    component_volume, num_components = ndi.label(label_volume)

    candidates = _get_component_stats(
        component_volume=component_volume,
        num_components=num_components,
        slice_ids=slice_ids,
        min_voxels=args.min_voxels,
        min_z_span=args.min_z_span,
    )
    selected_components = _choose_components(
        candidates=candidates,
        num_objects=args.num_objects,
        seed=args.seed,
        preferred_max_z_span=args.preferred_max_z_span,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.tracking_mode == "memory":
        sam_model, predictor, load_report = tracking_eval.load_sam_with_memory_adapter(
            model_type=args.model_type,
            device=args.device,
            base_sam_checkpoint_path=str(args.base_sam_checkpoint) if args.base_sam_checkpoint is not None else None,
            finetuned_checkpoint_path=str(args.finetuned_checkpoint),
            adapter_checkpoint_path=str(args.adapter_checkpoint) if args.adapter_checkpoint is not None else None,
        )
    else:
        sam_model, predictor, load_report = tracking_eval.load_plain_sam(
            model_type=args.model_type,
            device=args.device,
            base_sam_checkpoint_path=str(args.base_sam_checkpoint) if args.base_sam_checkpoint is not None else None,
            finetuned_checkpoint_path=str(args.finetuned_checkpoint),
        )
    tracking_eval.print_load_report(load_report)

    evaluation_summaries = []
    for rank, component in enumerate(selected_components, start=1):
        z_start = component["z_start_index"]
        z_stop = component["z_stop_index"]
        component_id = component["component_id"]

        image_volume = _load_image_subvolume(image_paths, z_start=z_start, z_stop=z_stop)
        gt_volume = (component_volume[z_start:z_stop] == component_id).astype(np.uint8)

        if args.tracking_mode == "memory":
            results = tracking_eval.evaluate_cross_slice_tracking(
                sam_model=sam_model,
                predictor=predictor,
                image_volume=image_volume,
                gt_volume=gt_volume,
                mask_threshold=args.mask_threshold,
            )
        else:
            results = tracking_eval.evaluate_cross_slice_tracking_without_memory(
                predictor=predictor,
                image_volume=image_volume,
                gt_volume=gt_volume,
                box_padding=args.box_padding,
            )

        object_dir = args.output_dir / (
            f"object_{rank:02d}_comp_{component_id}_slice_{component['slice_id_start']}_{component['slice_id_end']}"
        )
        object_dir.mkdir(parents=True, exist_ok=True)

        curve_path = object_dir / "iou_curve.png"
        fig, _ = tracking_eval.plot_slice_iou_curve(
            results["iou_per_slice"],
            title=f"{prefix} component {component_id}",
            save_path=str(curve_path),
        )
        plt.close(fig)

        preview_path = object_dir / "prediction_preview.png"
        _save_prediction_preview(
            image_volume=image_volume,
            gt_volume=gt_volume,
            pred_volume=results["pred_volume"],
            iou_per_slice=results["iou_per_slice"],
            save_path=preview_path,
        )

        np.save(object_dir / "pred_volume.npy", results["pred_volume"].astype(np.uint8))
        np.save(object_dir / "gt_volume.npy", gt_volume.astype(np.uint8))
        np.save(object_dir / "iou_per_slice.npy", np.asarray(results["iou_per_slice"], dtype=np.float32))

        summary = {
            "component_id": component_id,
            "voxels": component["voxels"],
            "slice_id_start": component["slice_id_start"],
            "slice_id_end": component["slice_id_end"],
            "z_span": component["z_span"],
            "mean_iou": float(np.mean(results["iou_per_slice"])),
            "min_iou": float(np.min(results["iou_per_slice"])),
            "max_iou": float(np.max(results["iou_per_slice"])),
            "num_slices": int(len(results["iou_per_slice"])),
            "init_point": results["init_point"].tolist(),
            "curve_path": str(curve_path),
            "preview_path": str(preview_path),
        }
        (object_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        evaluation_summaries.append(summary)

        print(json.dumps(summary, indent=2))

    run_summary = {
        "dataset_root": str(args.dataset_root),
        "prefix": prefix,
        "tracking_mode": args.tracking_mode,
        "num_total_slices": len(slice_ids),
        "num_components": int(num_components),
        "num_candidates": len(candidates),
        "seed": args.seed,
        "num_selected_objects": len(selected_components),
        "selected_components": evaluation_summaries,
        "load_report": load_report,
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
