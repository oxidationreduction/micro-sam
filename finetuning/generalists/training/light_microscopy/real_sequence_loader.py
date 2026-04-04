from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import relabel_sequential
import tifffile
import torch
from torch.utils.data import Dataset
from torch_em.transform.label import PerObjectDistanceTransform


@dataclass(frozen=True)
class VolumeEntry:
    name: str
    slice_ids: Tuple[int, ...]
    image_paths: Tuple[Path, ...]
    label_paths: Tuple[Path, ...]


@dataclass(frozen=True)
class SequenceWindow:
    volume_index: int
    start_index: int
    slice_id_start: int
    slice_id_end: int
    volume_name: str


class RealSequenceTifDataset(Dataset):
    """Dataset for real tif stacks with numerically sorted, consecutive slices."""

    def __init__(
        self,
        dataset_root: str | Path,
        seq_len: int,
        patch_shape: Tuple[int, int],
        split: str,
        val_fraction: float = 0.2,
        min_size: int = 10,
        min_tracking_length: int = 4,
        max_sampling_attempts: int = 32,
        seed: int = 17,
        label_mode: str = "auto",
        require_consecutive_slices: bool = True,
        require_full_track: bool = True,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"Invalid split: {split}")
        if seq_len < 2:
            raise ValueError(f"Expected seq_len >= 2, got {seq_len}")
        if label_mode not in ("auto", "instance", "binary_3d_cc"):
            raise ValueError(f"Unsupported label_mode: {label_mode}")

        self.dataset_root = Path(dataset_root)
        self.seq_len = int(seq_len)
        self.patch_shape = tuple(int(value) for value in patch_shape)
        self.split = split
        self.val_fraction = float(val_fraction)
        self.min_tracking_length = max(2, min(int(min_tracking_length), self.seq_len))
        self.max_sampling_attempts = int(max_sampling_attempts)
        self.seed = int(seed)
        self.label_mode = label_mode
        self.require_consecutive_slices = require_consecutive_slices
        self.require_full_track = require_full_track
        self.label_transform = PerObjectDistanceTransform(
            distances=True,
            boundary_distances=True,
            directed_distances=False,
            foreground=True,
            instances=True,
            min_size=min_size,
        )

        self.volume_entries = self._collect_volumes(self.dataset_root, self.seq_len, self.require_consecutive_slices)
        self.volumes = [self._load_volume(entry) for entry in self.volume_entries]
        self.windows = self._build_windows()
        if not self.windows:
            raise RuntimeError(
                "No valid sequence windows were found for the selected split. "
                f"dataset_root={self.dataset_root}, split={self.split}, seq_len={self.seq_len}"
            )

        total_slices = sum(len(entry.slice_ids) for entry in self.volume_entries)
        total_instances = sum(volume["num_instances"] for volume in self.volumes)
        print(
            f"[RealSequenceTifDataset] split={self.split} root={self.dataset_root} "
            f"volumes={len(self.volume_entries)} slices={total_slices} "
            f"instances={total_instances} windows={len(self.windows)} "
            f"patch_shape={self.patch_shape} label_mode={self.label_mode}"
        )

    @staticmethod
    def _collect_volumes(
        dataset_root: Path,
        seq_len: int,
        require_consecutive_slices: bool,
    ) -> List[VolumeEntry]:
        images_dir = dataset_root / "imagesTr"
        labels_dir = dataset_root / "labelsTr"
        if not images_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {images_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"Missing label directory: {labels_dir}")

        image_pattern = re.compile(r"^(?P<prefix>.+)_(?P<slice>\d+)_0000\.tif$")
        label_pattern = re.compile(r"^(?P<prefix>.+)_(?P<slice>\d+)\.tif$")

        image_entries = {}
        for path in images_dir.glob("*.tif"):
            match = image_pattern.match(path.name)
            if match is None:
                continue
            key = (match.group("prefix"), int(match.group("slice")))
            image_entries[key] = path

        label_entries = {}
        for path in labels_dir.glob("*.tif"):
            match = label_pattern.match(path.name)
            if match is None:
                continue
            key = (match.group("prefix"), int(match.group("slice")))
            label_entries[key] = path

        shared_keys = sorted(set(image_entries) & set(label_entries), key=lambda item: (item[0], item[1]))
        if not shared_keys:
            raise RuntimeError(f"No matched tif pairs found in {dataset_root}")

        keys_by_prefix = {}
        for prefix, slice_id in shared_keys:
            keys_by_prefix.setdefault(prefix, []).append((slice_id, image_entries[(prefix, slice_id)], label_entries[(prefix, slice_id)]))

        volumes = []
        for prefix, entries in sorted(keys_by_prefix.items()):
            runs = RealSequenceTifDataset._split_into_runs(entries, require_consecutive_slices)
            for run_index, run in enumerate(runs):
                if len(run) < seq_len:
                    continue

                slice_ids = tuple(slice_id for slice_id, _, _ in run)
                image_paths = tuple(image_path for _, image_path, _ in run)
                label_paths = tuple(label_path for _, _, label_path in run)
                run_name = prefix if len(runs) == 1 else f"{prefix}__run{run_index:02d}"
                volumes.append(
                    VolumeEntry(
                        name=run_name,
                        slice_ids=slice_ids,
                        image_paths=image_paths,
                        label_paths=label_paths,
                    )
                )

        if not volumes:
            raise RuntimeError(
                "No valid consecutive tif volumes were found. "
                f"dataset_root={dataset_root}, seq_len={seq_len}"
            )
        return volumes

    @staticmethod
    def _split_into_runs(entries, require_consecutive_slices: bool):
        if not require_consecutive_slices:
            return [entries]

        runs = []
        current_run = [entries[0]]
        for previous, current in zip(entries[:-1], entries[1:]):
            prev_slice = previous[0]
            curr_slice = current[0]
            if curr_slice == prev_slice + 1:
                current_run.append(current)
            else:
                runs.append(current_run)
                current_run = [current]
        runs.append(current_run)
        return runs

    def _load_volume(self, entry: VolumeEntry):
        image_volume = np.stack([self._read_image(path) for path in entry.image_paths], axis=0)
        label_volume = np.stack([tifffile.imread(path) for path in entry.label_paths], axis=0)
        instance_volume, num_instances = self._to_instance_volume(label_volume)

        label_dtype = np.uint16 if num_instances <= np.iinfo(np.uint16).max else np.uint32
        instance_volume = instance_volume.astype(label_dtype, copy=False)
        return {
            "entry": entry,
            "image_volume": image_volume,
            "instance_volume": instance_volume,
            "num_instances": int(num_instances),
        }

    def _to_instance_volume(self, label_volume: np.ndarray):
        label_volume = np.asarray(label_volume)
        unique_values = np.unique(label_volume)
        positive_values = unique_values[unique_values > 0]
        is_binary = len(positive_values) <= 1

        resolved_mode = self.label_mode
        if resolved_mode == "auto":
            resolved_mode = "binary_3d_cc" if is_binary else "instance"

        if resolved_mode == "binary_3d_cc":
            instance_volume, num_instances = ndi.label(label_volume > 0)
            return instance_volume, int(num_instances)

        if resolved_mode == "instance":
            return self._relabel_instance_volume(label_volume)

        raise ValueError(f"Unsupported resolved label mode: {resolved_mode}")

    @staticmethod
    def _relabel_instance_volume(label_volume: np.ndarray):
        label_volume = label_volume.astype(np.int64, copy=False)
        unique_ids = np.unique(label_volume)
        unique_ids = unique_ids[unique_ids > 0]

        instance_volume = np.zeros_like(label_volume, dtype=np.int64)
        next_id = 1
        for label_id in unique_ids:
            component_volume, num_components = ndi.label(label_volume == label_id)
            if num_components == 0:
                continue

            component_mask = component_volume > 0
            component_volume[component_mask] += next_id - 1
            instance_volume[component_mask] = component_volume[component_mask]
            next_id += int(num_components)

        return instance_volume, int(next_id - 1)

    @staticmethod
    def _read_image(path: Path) -> np.ndarray:
        image = tifffile.imread(path)
        if image.ndim != 2:
            raise ValueError(f"Expected a 2D tif image, got shape {image.shape} for {path}")
        if image.dtype == np.uint8:
            return image

        image = image.astype(np.float32, copy=False)
        image -= image.min()
        image /= max(float(image.max()), 1e-7)
        return (255.0 * image).astype(np.uint8)

    def _build_windows(self) -> List[SequenceWindow]:
        windows = []
        for volume_index, volume in enumerate(self.volumes):
            entry = volume["entry"]
            valid_starts = self._compute_valid_starts(volume["instance_volume"])
            if not valid_starts:
                continue

            if len(valid_starts) == 1:
                train_starts = valid_starts
                val_starts = valid_starts
            else:
                cutoff = int(round((1.0 - self.val_fraction) * len(valid_starts)))
                cutoff = min(max(cutoff, 1), len(valid_starts) - 1)
                train_starts = valid_starts[:cutoff]
                val_starts = valid_starts[cutoff:]

            selected_starts = train_starts if self.split == "train" else val_starts
            for start in selected_starts:
                windows.append(
                    SequenceWindow(
                        volume_index=volume_index,
                        start_index=start,
                        slice_id_start=entry.slice_ids[start],
                        slice_id_end=entry.slice_ids[start + self.seq_len - 1],
                        volume_name=entry.name,
                    )
                )

        return windows

    def _compute_valid_starts(self, instance_volume: np.ndarray) -> List[int]:
        bboxes = ndi.find_objects(instance_volume)
        valid_starts = set()

        for object_id, bbox in enumerate(bboxes, start=1):
            if bbox is None:
                continue

            z_slice, y_slice, x_slice = bbox
            z_start, z_stop = int(z_slice.start), int(z_slice.stop)
            if z_stop - z_start < self.seq_len:
                continue

            subvolume = instance_volume[z_start:z_stop, y_slice, x_slice] == object_id
            presence = np.any(subvolume, axis=(1, 2))

            run_start = None
            for offset, is_present in enumerate(presence):
                if is_present and run_start is None:
                    run_start = offset
                elif not is_present and run_start is not None:
                    self._add_valid_range(valid_starts, z_start, run_start, offset)
                    run_start = None
            if run_start is not None:
                self._add_valid_range(valid_starts, z_start, run_start, len(presence))

        return sorted(valid_starts)

    def _add_valid_range(self, valid_starts: set[int], z_start: int, run_start: int, run_stop: int) -> None:
        run_length = run_stop - run_start
        if run_length < self.seq_len:
            return

        global_start = z_start + run_start
        global_stop = z_start + run_stop - self.seq_len
        valid_starts.update(range(global_start, global_stop + 1))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        window = self.windows[index]
        volume = self.volumes[window.volume_index]
        start = window.start_index
        stop = start + self.seq_len

        sequence_images = volume["image_volume"][start:stop]
        sequence_labels = volume["instance_volume"][start:stop]

        rng = np.random.default_rng(self.seed + index if self.split == "val" else None)
        top, left = self._choose_crop(sequence_labels, rng)

        patch_h, patch_w = self.patch_shape
        sequence_images = sequence_images[:, top:top + patch_h, left:left + patch_w]
        sequence_labels = sequence_labels[:, top:top + patch_h, left:left + patch_w]
        sequence_labels = relabel_sequential(sequence_labels)[0].astype(np.int64, copy=False)

        image_tensor = self._to_image_tensor(sequence_images)
        label_tensor = self._to_label_tensor(sequence_labels)
        return image_tensor, label_tensor

    def _choose_crop(self, sequence_labels: np.ndarray, rng: np.random.Generator) -> Tuple[int, int]:
        height, width = sequence_labels.shape[-2:]
        patch_h, patch_w = self.patch_shape
        if patch_h > height or patch_w > width:
            raise ValueError(
                f"Patch shape {self.patch_shape} is larger than image shape {(height, width)}"
            )

        max_top = height - patch_h
        max_left = width - patch_w
        required_length = sequence_labels.shape[0] if self.require_full_track else self.min_tracking_length

        for _ in range(self.max_sampling_attempts):
            top = int(rng.integers(0, max_top + 1)) if max_top > 0 else 0
            left = int(rng.integers(0, max_left + 1)) if max_left > 0 else 0
            crop = sequence_labels[:, top:top + patch_h, left:left + patch_w]
            if self._is_trackable_crop(crop, required_length):
                return top, left

        candidate_ids = self._get_trackable_ids(sequence_labels, required_length)
        if candidate_ids.size == 0:
            return max_top // 2, max_left // 2

        areas = [(sequence_labels[0] == object_id).sum() for object_id in candidate_ids]
        target_id = int(candidate_ids[int(np.argmax(areas))])
        coords = np.argwhere(sequence_labels[0] == target_id)
        center_y, center_x = coords.mean(axis=0)

        top = int(np.clip(round(center_y) - patch_h // 2, 0, max_top))
        left = int(np.clip(round(center_x) - patch_w // 2, 0, max_left))
        return top, left

    def _is_trackable_crop(self, sequence_labels: np.ndarray, required_length: int) -> bool:
        return self._get_trackable_ids(sequence_labels, required_length).size > 0

    def _get_trackable_ids(self, sequence_labels: np.ndarray, required_length: int) -> np.ndarray:
        first_slice_ids = np.unique(sequence_labels[0])
        first_slice_ids = first_slice_ids[first_slice_ids > 0]
        if len(first_slice_ids) == 0:
            return np.empty(0, dtype=np.int64)

        trackable_ids = []
        required_length = min(required_length, sequence_labels.shape[0])
        for object_id in first_slice_ids:
            presence = np.any(sequence_labels[:required_length] == object_id, axis=(1, 2))
            if presence.all():
                trackable_ids.append(int(object_id))
        return np.asarray(trackable_ids, dtype=np.int64)

    @staticmethod
    def _to_image_tensor(sequence_images: np.ndarray) -> torch.Tensor:
        image_tensor = np.repeat(sequence_images[:, None, :, :], 3, axis=1).astype(np.float32, copy=False)
        return torch.from_numpy(image_tensor)

    def _to_label_tensor(self, sequence_labels: np.ndarray) -> torch.Tensor:
        transformed = [
            self.label_transform(label_slice.astype(np.int64, copy=False)).astype(np.float32, copy=False)
            for label_slice in sequence_labels
        ]
        label_tensor = np.stack(transformed, axis=0)
        return torch.from_numpy(label_tensor)
