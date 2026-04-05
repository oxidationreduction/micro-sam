from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import relabel_sequential
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class _CremiVolume:
    volume_name: str
    file_path: Path
    raw_key: str
    label_key: str
    raw_start: Tuple[int, int, int]
    label_shape: Tuple[int, int, int]


@dataclass(frozen=True)
class _CremiSliceRecord:
    volume_name: str
    file_path: Path
    raw_key: str
    label_key: str
    raw_z: int
    label_z: int


class CremiAlignedSliceDataset(Dataset):
    """Sample aligned 2D patches from CREMI hdf files with padded raw volumes."""

    RAW_KEY = "volumes/raw"
    LABEL_KEY = "volumes/labels/neuron_ids"
    DEFAULT_Z_RANGES = {
        "train": {"A": slice(0, 75), "B": slice(0, 75), "C": slice(0, 75)},
        "val": {"A": slice(75, 100), "B": slice(75, 100), "C": slice(75, 100)},
    }
    CANONICAL_FILES = {
        "A": "sampleA.h5",
        "B": "sampleB.h5",
        "C": "sampleC.h5",
    }
    PADDED_FILES = {
        "A": "sample_A_padded_20160501.hdf",
        "B": "sample_B_padded_20160501.hdf",
        "C": "sample_C_padded_20160501.hdf",
    }

    def __init__(
        self,
        dataset_root: str | Path,
        patch_shape: Tuple[int, int],
        split: str,
        raw_transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        sampler: Optional[Callable] = None,
        n_samples: Optional[int] = None,
        z_ranges: Optional[Dict[str, Dict[str, slice]]] = None,
        max_sampling_attempts: int = 32,
        seed: int = 17,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"Invalid split: {split}")
        if len(patch_shape) != 2:
            raise ValueError(f"Expected 2D patch_shape=(H, W), got {patch_shape}")

        self.dataset_root = Path(dataset_root)
        self.patch_shape = tuple(int(value) for value in patch_shape)
        self.split = split
        self.raw_transform = raw_transform
        self.label_transform = label_transform
        self.sampler = sampler
        self.n_samples = int(n_samples) if n_samples is not None else None
        self.max_sampling_attempts = int(max_sampling_attempts)
        self.seed = int(seed)
        self._file_cache: Dict[Path, h5py.File] = {}

        split_ranges = self.DEFAULT_Z_RANGES if z_ranges is None else z_ranges
        if split not in split_ranges:
            raise ValueError(f"Missing z-range definition for split={split}")

        self.volumes = self._collect_volumes(self.dataset_root)
        self._volume_by_path = {volume.file_path: volume for volume in self.volumes}
        self.slice_records = self._collect_slice_records(split_ranges[split])
        self._close_cached_files()
        if not self.slice_records:
            raise RuntimeError(
                f"No usable CREMI slices found for split={self.split} in {self.dataset_root}"
            )

        print(
            f"[CremiAlignedSliceDataset] split={self.split} root={self.dataset_root} "
            f"volumes={len(self.volumes)} slices={len(self.slice_records)} patch_shape={self.patch_shape} "
            f"label_transform={self.label_transform is not None}"
        )

    def __len__(self) -> int:
        return len(self.slice_records) if self.n_samples is None else self.n_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        record = self.slice_records[index % len(self.slice_records)]
        rng = np.random.default_rng(self.seed + index if self.split == "val" else None)

        raw_slice, label_slice = self._load_slice(record)
        top, left = self._choose_crop(raw_slice, label_slice, rng)

        patch_h, patch_w = self.patch_shape
        raw_patch = raw_slice[top:top + patch_h, left:left + patch_w]
        label_patch = label_slice[top:top + patch_h, left:left + patch_w]
        label_patch = self._relabel_instance_mask(label_patch)

        raw_tensor = self._to_image_tensor(raw_patch)
        label_tensor = self._to_label_tensor(label_patch)
        return raw_tensor, label_tensor

    def __del__(self) -> None:
        self._close_cached_files()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file_cache"] = {}
        return state

    @classmethod
    def _collect_volumes(cls, dataset_root: Path) -> List[_CremiVolume]:
        if not dataset_root.exists():
            raise FileNotFoundError(f"Missing CREMI root: {dataset_root}")

        volumes = []
        for volume_name in ("A", "B", "C"):
            file_path = cls._resolve_volume_path(dataset_root, volume_name)
            raw_start, label_shape = cls._compute_alignment(file_path)
            volumes.append(
                _CremiVolume(
                    volume_name=volume_name,
                    file_path=file_path,
                    raw_key=cls.RAW_KEY,
                    label_key=cls.LABEL_KEY,
                    raw_start=raw_start,
                    label_shape=label_shape,
                )
            )
        return volumes

    @classmethod
    def _resolve_volume_path(cls, dataset_root: Path, volume_name: str) -> Path:
        canonical_path = dataset_root / cls.CANONICAL_FILES[volume_name]
        if canonical_path.exists():
            return canonical_path

        padded_path = dataset_root / cls.PADDED_FILES[volume_name]
        if padded_path.exists():
            return padded_path

        raise FileNotFoundError(
            f"Could not find CREMI volume {volume_name} under {dataset_root}. "
            f"Expected one of {canonical_path.name} or {padded_path.name}."
        )

    @classmethod
    def _compute_alignment(cls, file_path: Path) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        with h5py.File(file_path, "r") as handle:
            raw = handle[cls.RAW_KEY]
            labels = handle[cls.LABEL_KEY]

            raw_offset = np.asarray(raw.attrs.get("offset", np.zeros(raw.ndim)), dtype=np.float64)
            raw_resolution = np.asarray(raw.attrs.get("resolution", np.ones(raw.ndim)), dtype=np.float64)
            label_offset = np.asarray(labels.attrs.get("offset", np.zeros(labels.ndim)), dtype=np.float64)
            label_resolution = np.asarray(labels.attrs.get("resolution", raw_resolution), dtype=np.float64)

            if not np.allclose(raw_resolution, label_resolution):
                raise ValueError(
                    f"Resolution mismatch in {file_path}: raw={raw_resolution}, label={label_resolution}"
                )

            raw_start = np.rint((label_offset - raw_offset) / raw_resolution).astype(int)
            raw_stop = raw_start + np.asarray(labels.shape)
            if np.any(raw_start < 0) or np.any(raw_stop > np.asarray(raw.shape)):
                raise ValueError(
                    f"Aligned label ROI is out of bounds for {file_path}: start={tuple(raw_start)}, "
                    f"stop={tuple(raw_stop)}, raw_shape={tuple(raw.shape)}, label_shape={tuple(labels.shape)}"
                )

            return tuple(int(value) for value in raw_start), tuple(int(value) for value in labels.shape)

    def _collect_slice_records(self, split_ranges: Dict[str, slice]) -> List[_CremiSliceRecord]:
        records = []
        for volume in self.volumes:
            z_range = split_ranges.get(volume.volume_name)
            if z_range is None:
                continue

            z_start, z_stop, z_step = z_range.indices(volume.label_shape[0])
            if z_step != 1:
                raise ValueError("CREMI z-ranges must use contiguous slices.")

            label_ds = self._get_dataset(volume.file_path, volume.label_key)
            for label_z in range(z_start, z_stop):
                label_slice = label_ds[label_z]
                if not np.any(label_slice > 0):
                    continue
                records.append(
                    _CremiSliceRecord(
                        volume_name=volume.volume_name,
                        file_path=volume.file_path,
                        raw_key=volume.raw_key,
                        label_key=volume.label_key,
                        raw_z=volume.raw_start[0] + label_z,
                        label_z=label_z,
                    )
                )

        return records

    def _get_dataset(self, file_path: Path, key: str):
        handle = self._file_cache.get(file_path)
        if handle is None:
            handle = h5py.File(file_path, "r")
            self._file_cache[file_path] = handle
        return handle[key]

    def _close_cached_files(self) -> None:
        for handle in self._file_cache.values():
            try:
                handle.close()
            except Exception:
                pass
        self._file_cache.clear()

    def _load_slice(self, record: _CremiSliceRecord) -> Tuple[np.ndarray, np.ndarray]:
        raw_ds = self._get_dataset(record.file_path, record.raw_key)
        label_ds = self._get_dataset(record.file_path, record.label_key)

        volume = self._volume_by_path[record.file_path]
        y_start, x_start = volume.raw_start[1], volume.raw_start[2]
        y_stop = y_start + volume.label_shape[1]
        x_stop = x_start + volume.label_shape[2]

        raw_slice = np.asarray(raw_ds[record.raw_z, y_start:y_stop, x_start:x_stop])
        label_slice = np.asarray(label_ds[record.label_z])
        return raw_slice, label_slice

    def _choose_crop(
        self,
        raw_slice: np.ndarray,
        label_slice: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[int, int]:
        height, width = label_slice.shape
        patch_h, patch_w = self.patch_shape
        if patch_h > height or patch_w > width:
            raise ValueError(
                f"Patch shape {self.patch_shape} is larger than aligned CREMI slice shape {(height, width)}"
            )

        max_top = height - patch_h
        max_left = width - patch_w

        for _ in range(self.max_sampling_attempts):
            top = int(rng.integers(0, max_top + 1)) if max_top > 0 else 0
            left = int(rng.integers(0, max_left + 1)) if max_left > 0 else 0
            raw_patch = raw_slice[top:top + patch_h, left:left + patch_w]
            label_patch = label_slice[top:top + patch_h, left:left + patch_w]
            if self._accept_patch(raw_patch, label_patch):
                return top, left

        return self._fallback_crop(label_slice)

    def _accept_patch(self, raw_patch: np.ndarray, label_patch: np.ndarray) -> bool:
        if self.sampler is None:
            return np.any(label_patch > 0)
        return bool(self.sampler(raw_patch, label_patch))

    def _fallback_crop(self, label_slice: np.ndarray) -> Tuple[int, int]:
        patch_h, patch_w = self.patch_shape
        height, width = label_slice.shape
        max_top = height - patch_h
        max_left = width - patch_w

        relabeled = self._relabel_instance_mask(label_slice)
        object_ids = np.unique(relabeled)
        object_ids = object_ids[object_ids > 0]
        if len(object_ids) == 0:
            return max_top // 2, max_left // 2

        largest_id = max(object_ids, key=lambda object_id: int((relabeled == object_id).sum()))
        coords = np.argwhere(relabeled == largest_id)
        center_y, center_x = coords.mean(axis=0)

        top = int(np.clip(round(center_y) - patch_h // 2, 0, max_top))
        left = int(np.clip(round(center_x) - patch_w // 2, 0, max_left))
        return top, left

    @staticmethod
    def _relabel_instance_mask(label_slice: np.ndarray) -> np.ndarray:
        label_slice = np.asarray(label_slice).astype(np.int64, copy=False)
        unique_ids = np.unique(label_slice)
        unique_ids = unique_ids[unique_ids > 0]

        instance_mask = np.zeros_like(label_slice, dtype=np.int64)
        next_id = 1
        for label_id in unique_ids:
            components, n_components = ndi.label(label_slice == label_id)
            if n_components == 0:
                continue

            component_mask = components > 0
            components[component_mask] += next_id - 1
            instance_mask[component_mask] = components[component_mask]
            next_id += int(n_components)

        return relabel_sequential(instance_mask)[0].astype(np.int64, copy=False)

    def _to_image_tensor(self, raw_patch: np.ndarray) -> torch.Tensor:
        raw = raw_patch
        if self.raw_transform is not None:
            raw = self.raw_transform(raw)

        raw = np.asarray(raw)
        if raw.ndim == 2:
            raw = np.repeat(raw[None, :, :], 3, axis=0)
        elif raw.ndim == 3 and raw.shape[0] == 1:
            raw = np.repeat(raw, 3, axis=0)
        elif raw.ndim == 3 and raw.shape[-1] == 3 and raw.shape[0] != 3:
            raw = np.transpose(raw, (2, 0, 1))
        elif raw.ndim != 3:
            raise ValueError(f"Unsupported CREMI raw shape after transform: {raw.shape}")

        return torch.from_numpy(raw.astype(np.float32, copy=False))

    def _to_label_tensor(self, label_patch: np.ndarray) -> torch.Tensor:
        if self.label_transform is None:
            label_tensor = label_patch[None, :, :].astype(np.float32, copy=False)
            return torch.from_numpy(label_tensor)

        transformed = self.label_transform(label_patch).astype(np.float32, copy=False)
        return torch.from_numpy(transformed)
