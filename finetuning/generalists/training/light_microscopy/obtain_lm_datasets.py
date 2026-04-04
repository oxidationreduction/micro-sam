import os

import numpy as np
from sklearn.model_selection import train_test_split

import torch

import torch_em
from torch_em.data import datasets
from torch_em.transform.raw import normalize
from torch_em.data import MinInstanceSampler, ConcatDataset
from torch_em.transform.label import PerObjectDistanceTransform
from torch_em.data.datasets.light_microscopy.neurips_cell_seg import to_rgb

from finetuning.generalists.training.light_microscopy.real_sequence_loader import (
    RealSequenceTifDataset,
    SequenceWindowToFrameDataset,
)
from finetuning.generalists.training.light_microscopy.seq_wrapper import SyntheticSequenceWrapper
from micro_sam.training.util import ResizeRawTrafo, ResizeLabelTrafo


def _to_8bit(raw):
    "Ensures three channels for inputs and rescale them to 8 bit."
    if raw.ndim == 3 and raw.shape[0] == 1:  # If the inputs have 1 channel, we triplicate it.
        raw = np.concatenate([raw] * 3, axis=0)

    raw = to_rgb(raw)  # Ensure all images are in 3-channels: triplicate one channel to three channels.
    raw = normalize(raw) * 255
    return raw


def _identity(x):
    "Ensures three channels for inputs and avoids rescaling inputs."
    x = to_rgb(x)
    return x


def _cellpose_raw_trafo(x):
    """Transforms input images to desired format.

    NOTE: The input channel logic is arranged a bit strangely in `cyto` dataset.
    This function takes care of it here.
    """
    r, g, b = x

    assert g.max() != 0
    if r.max() == 0:
        # The image is 1 channel and exists in green channel only.
        assert b.max() == 0
        x = np.concatenate([g[None]] * 3, axis=0)

    elif r.max() != 0 and g.max() != 0:
        # The image is 2 channels and we sort the channels such that - 0: cell, 1: nucleus
        x = np.stack([g, r, np.zeros_like(b)], axis=0)

    x = to_rgb(x)  # Ensures three channels for inputs and avoids rescaling inputs.

    return x


def get_concat_lm_datasets(input_path, patch_shape, split_choice):
    assert split_choice in ["train", "val"]

    label_dtype = torch.float32
    sampler = MinInstanceSampler(min_size=10)

    def _get_label_transform():
        label_transform = PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True,
        )
        return label_transform
        # return None

    def get_livecell_datasets():
        "Datasets for cell segmentation in phase contrast microscopy images."
        all_livecell_datasets = [
            datasets.get_livecell_dataset(
                path=os.path.join(input_path, "livecell"), split=split_choice, patch_shape=patch_shape,
                sampler=sampler, label_dtype=label_dtype, raw_transform=_identity, download=True, cell_types=[ctype],
                label_transform=_get_label_transform(), n_samples=200 if split_choice == "train" else 50,
            ) for ctype in datasets.livecell.CELL_TYPES
        ]
        return all_livecell_datasets

    def get_embedseg_datasets():
        "Datasets for cell and nuclei segmentation in fluorescence microscopy images."
        names = [
            "Mouse-Organoid-Cells-CBG",
            "Mouse-Skull-Nuclei-CBG",
            "Platynereis-ISH-Nuclei-CBG",
            "Platynereis-Nuclei-CBG",
        ]
        all_embedseg_datasets = [
            datasets.get_embedseg_dataset(
                path=os.path.join(input_path, "embedseg"), name=name, patch_shape=(1, *patch_shape),
                download=True, n_samples=500 if split_choice == "train" else 100, raw_transform=_to_8bit,
                label_dtype=label_dtype, label_transform=_get_label_transform(), ndim=2,
                sampler=MinInstanceSampler(min_num_instances=3, min_size=10),
            ) for name in names
        ]
        return all_embedseg_datasets

    def get_yeaz_dataset():
        "Datasets for yeast segmentation in phase contrast and brightfield microscopy images."
        names = ["bf", "phc"]
        all_yeaz_datasets = [
            datasets.get_yeaz_dataset(
                path=os.path.join(input_path, "yeaz"), patch_shape=patch_shape, raw_transform=_to_8bit,
                ndim=2, download=False, split=split_choice, choice=name, label_transform=_get_label_transform(),
                sampler=sampler, label_dtype=label_dtype,
            ) for name in names
        ]
        return all_yeaz_datasets

    def get_cvz_dataset(stain_choice):
        "Datasets for cell and nuclei segmentation in fluorescence microscopy images."
        # NOTE: We create random splits for this dataset for training the generalist.
        raw_paths, label_paths = datasets.cvz_fluo.get_cvz_fluo_paths(
            path=os.path.join(input_path, "cvz"), stain_choice=stain_choice, download=True
        )
        train_raw_paths, test_raw_paths, train_label_paths, test_label_paths = train_test_split(
            raw_paths, label_paths, test_size=0.2, random_state=42,
        )
        ds = torch_em.default_segmentation_dataset(
            raw_paths=train_raw_paths if split_choice == "train" else test_raw_paths, raw_key=None,
            label_paths=train_label_paths if split_choice == "train" else test_label_paths,
            label_key=None, is_seg_dataset=False, patch_shape=patch_shape, sampler=sampler,
            raw_transform=_identity, label_transform=_get_label_transform(), label_dtype=label_dtype,
        )
        return [ds]

    def get_ctc_datasets():
        "Datasets for cell segmentation in different modalities."
        all_ctc_datasets = []
        for dataset_name in datasets.ctc.CTC_CHECKSUMS["train"].keys():
            if dataset_name in ["Fluo-N2DH-GOWT1", "Fluo-N2DL-HeLa"]:
                continue

            all_ctc_datasets.append(
                datasets.get_ctc_segmentation_dataset(
                    path=os.path.join(input_path, "ctc"), dataset_name=dataset_name, patch_shape=(1, *patch_shape),
                    sampler=sampler, raw_transform=_to_8bit, label_transform=_get_label_transform(),
                    download=True, label_dtype=label_dtype,
                )
            )
        return all_ctc_datasets

    _datasets = [
        # cell segmentation in tissue microscopy images.
        # datasets.get_tissuenet_dataset(
        #     path=os.path.join(input_path, "tissuenet"), split=split_choice, download=True, patch_shape=patch_shape,
        #     raw_channel="rgb", label_channel="cell", label_transform=ResizeLabelTrafo(patch_shape),
        #     # NOTE: Below is done for TissueNet: to work with the valid channels (i.e. the first and second channels).
        #     raw_transform=ResizeRawTrafo((3, *patch_shape), do_rescaling=True, valid_channels=(0, 1)),
        #     n_samples=500 if split_choice == "train" else 100, sampler=sampler, label_dtype=label_dtype,
        # ),
        # bacteria segmentation in label-free microscopy images.
        datasets.get_deepbacs_dataset(
            path=os.path.join(input_path, "deepbacs"), split=split_choice, patch_shape=patch_shape,
            raw_transform=_to_8bit, label_transform=_get_label_transform(), label_dtype=label_dtype,
            download=True, sampler=MinInstanceSampler(min_num_instances=4, min_size=10)
        ),
        # cell segmentation in confocal microscopy images.
        # datasets.get_plantseg_dataset(
        #     path=os.path.join(input_path, "plantseg"), name="root", n_samples=500 if split_choice == "train" else 100,
        #     patch_shape=(1, *patch_shape), download=True, ndim=3, raw_transform=ResizeRawTrafo((3, *patch_shape)),
        #     sampler=MinInstanceSampler(min_num_instances=4, min_size=10), split=split_choice, label_dtype=label_dtype,
        #     label_transform=ResizeLabelTrafo(patch_shape),
        # ),
        # cell segmentation in multi-modal microscopy images.
        datasets.get_neurips_cellseg_supervised_dataset(
            root=os.path.join(input_path, "neurips_cellseg"), split=split_choice, label_dtype=label_dtype,
            patch_shape=patch_shape, raw_transform=_to_8bit, label_transform=_get_label_transform(),
            sampler=MinInstanceSampler(min_num_instances=3, min_size=10), download=False,
        ),
        # nuclei segmentation in fluorescence microscopy images.
        datasets.get_dsb_dataset(
            path=os.path.join(input_path, "dsb"), split=split_choice if split_choice == "train" else "test",
            patch_shape=patch_shape, label_transform=_get_label_transform(), sampler=sampler,
            label_dtype=label_dtype, download=True, raw_transform=_identity,
        ),
        # nuclei segmentation in fluorescence microscopy images.
        # datasets.get_dynamicnuclearnet_dataset(
        #     path=os.path.join(input_path, "dynamicnuclearnet"), patch_shape=patch_shape, download=True, sampler=sampler,
        #     split=split_choice, n_samples=500 if split_choice == "train" else 100, label_dtype=label_dtype,
        #     raw_transform=_to_8bit, label_transform=_get_label_transform(),
        # ),
        # cell segmentation in multiple microscopy imaging modalities.
        # datasets.get_cellpose_dataset(
        #     path=os.path.join(input_path, "cellpose"), patch_shape=patch_shape, choice="cyto", sampler=sampler,
        #     download=True, split=split_choice if split_choice == "train" else "test", label_dtype=label_dtype,
        #     label_transform=_get_label_transform(), raw_transform=_cellpose_raw_trafo,
        # ),
        # bacteria segmentation in phase contrast and fluorescence microscopy images.
        # worm segmentation in brightfield microscopy images.
        datasets.get_omnipose_dataset(
            path=os.path.join(input_path, "omnipose"), patch_shape=patch_shape, download=True,
            split=split_choice if split_choice == "train" else "test", sampler=sampler,
            label_dtype=label_dtype, raw_transform=_to_8bit, label_transform=_get_label_transform(),
        ),
        # # organoid segmentation in brightfield microscopy images.
        datasets.get_orgasegment_dataset(
            path=os.path.join(input_path, "orgasegment"), patch_shape=patch_shape, download=True, split=split_choice,
            raw_transform=_identity, label_transform=_get_label_transform(), label_dtype=label_dtype, sampler=sampler,
        ),
    ]

    # Add LIVECell dataset: cell segmentation for phase contrast microscopy images.
    # _datasets.extend(get_livecell_datasets())

    # Add EmbedSeg datasets: cell and nuclei segmentation for fluorescence microscopy images.
    # _datasets.extend(get_embedseg_datasets())

    # Add YeaZ datasets: yeast segmentation for brightfield and phase contrast microscopy images.
    # _datasets.extend(get_yeaz_dataset())

    # Add CVZ Fluo datasets: cell and nuclei segmentation for fluorescence microscopy images.
    # _datasets.extend(get_cvz_dataset("cell"))
    # _datasets.extend(get_cvz_dataset("dapi"))

    # Add CTC datasets: cell segmentation for various
    if split_choice == "train":  # NOTE: We use CTC only for training.
        _datasets.extend(get_ctc_datasets())

    generalist_dataset = ConcatDataset(*_datasets)

    # Increasing the sampling attempts for the NeurIPS CellSeg dataset.
    # generalist_dataset.datasets[1].max_sampling_attempts = 5000

    return generalist_dataset


def get_generalist_lm_loaders(input_path, patch_shape, batch_size=2, num_workers=24):
    """This returns the concatenated light microscopy datasets implemented in `torch_em`:
    https://github.com/constantinpape/torch-em/tree/main/torch_em/data/datasets/light_microscopy.
    It will automatically download all the datasets, except:
    - TissueNet (see `torch_em/data/datasets/light_microscopy/tissuenet.py` for details)
    - DynamicNuclearNet (see `torch_em/data/datasets/light_microscopy/dynamicnuclearnet.py` for details)
    - CellPose (see `torch_em/data/datasets/light_microscopy/cellpose.py` for details)
    - YeaZ (see `torch_em/data/datasets/light_microscopy/yeaz.py` for details)

    NOTE: To remove / replace the datasets with another dataset, you need to add the datasets (for train and val splits)
    in `get_concat_lm_dataset`. The labels have to be in a label mask instance segmentation format,
    i.e. the tensors (inputs & masks) should be of same spatial shape, with each object in the mask having it's own ID.
    IMPORTANT: the ID 0 is reserved for background, and the IDs must be consecutive.
    """
    # Get the datasets.
    patch_shape_2d = patch_shape[1:]  if len(patch_shape) == 3 else patch_shape
    generalist_train_dataset = get_concat_lm_datasets(input_path, patch_shape_2d, "train")
    generalist_val_dataset = get_concat_lm_datasets(input_path, patch_shape_2d, "val")

    if len(patch_shape) == 3:
        # If the patch shape is 3D, we wrap the datasets in a sequence wrapper that creates synthetic sequences.
        generalist_train_dataset = SyntheticSequenceWrapper(generalist_train_dataset, seq_len=patch_shape[0])
        generalist_val_dataset = SyntheticSequenceWrapper(generalist_val_dataset, seq_len=patch_shape[0])

    # Get the dataloaders.
    train_loader = torch_em.get_data_loader(
        generalist_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch_em.get_data_loader(
        generalist_val_dataset, batch_size=max(1, batch_size >> 1), shuffle=True, num_workers=num_workers
    )

    return train_loader, val_loader


def get_real_sequence_lm_loaders(
    dataset_root,
    patch_shape,
    batch_size=1,
    num_workers=0,
    val_fraction=0.2,
    min_size=10,
    min_tracking_length=4,
    seed=17,
    label_mode="auto",
    require_consecutive_slices=True,
    require_full_track=True,
):
    """Build loaders for a real tif sequence dataset.

    Expected layout:
    - imagesTr/<prefix>_<slice_id>_0000.tif
    - labelsTr/<prefix>_<slice_id>.tif

    Slice ids are sorted numerically. Multiple prefixes are supported and gaps in the
    slice ids are split into separate volumes when `require_consecutive_slices=True`.
    """
    if len(patch_shape) != 3:
        raise ValueError(f"Expected patch_shape=(seq_len, H, W), got {patch_shape}")

    seq_len, patch_h, patch_w = patch_shape
    dataset_kwargs = {
        "seq_len": seq_len,
        "patch_shape": (patch_h, patch_w),
        "val_fraction": val_fraction,
        "min_size": min_size,
        "min_tracking_length": min_tracking_length,
        "seed": seed,
        "label_mode": label_mode,
        "require_consecutive_slices": require_consecutive_slices,
        "require_full_track": require_full_track,
    }

    if isinstance(dataset_root, (list, tuple)):
        dataset_roots = [str(root) for root in dataset_root if str(root).strip()]
    else:
        dataset_roots = [root.strip() for root in str(dataset_root).split(",") if root.strip()]
    if not dataset_roots:
        raise ValueError("Expected at least one sequence dataset root.")

    train_datasets = [
        RealSequenceTifDataset(dataset_root=root, split="train", **dataset_kwargs)
        for root in dataset_roots
    ]
    val_datasets = [
        RealSequenceTifDataset(dataset_root=root, split="val", **dataset_kwargs)
        for root in dataset_roots
    ]

    train_dataset = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(*train_datasets)
    val_dataset = val_datasets[0] if len(val_datasets) == 1 else ConcatDataset(*val_datasets)

    train_loader = torch_em.get_data_loader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch_em.get_data_loader(
        val_dataset, batch_size=max(1, batch_size >> 1), shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader


def get_real_sequence_frame_lm_loaders(
    dataset_root,
    patch_shape,
    batch_size=4,
    num_workers=0,
    val_fraction=0.2,
    min_size=10,
    min_tracking_length=4,
    seed=17,
    label_mode="auto",
    require_consecutive_slices=True,
    require_full_track=True,
    frame_indices=None,
):
    """Build 2D loaders from the same real tif sequence windows used by memory training.

    The underlying sequence windows are identical to `get_real_sequence_lm_loaders`, but each
    sampled window is flattened into independent 2D frames so that plain-SAM finetuning can use
    the same data source without temporal prompts.
    """
    if len(patch_shape) != 3:
        raise ValueError(f"Expected patch_shape=(seq_len, H, W), got {patch_shape}")

    seq_len, patch_h, patch_w = patch_shape
    dataset_kwargs = {
        "seq_len": seq_len,
        "patch_shape": (patch_h, patch_w),
        "val_fraction": val_fraction,
        "min_size": min_size,
        "min_tracking_length": min_tracking_length,
        "seed": seed,
        "label_mode": label_mode,
        "require_consecutive_slices": require_consecutive_slices,
        "require_full_track": require_full_track,
    }

    if isinstance(dataset_root, (list, tuple)):
        dataset_roots = [str(root) for root in dataset_root if str(root).strip()]
    else:
        dataset_roots = [root.strip() for root in str(dataset_root).split(",") if root.strip()]
    if not dataset_roots:
        raise ValueError("Expected at least one sequence dataset root.")

    train_datasets = [
        SequenceWindowToFrameDataset(
            RealSequenceTifDataset(dataset_root=root, split="train", **dataset_kwargs),
            frame_indices=frame_indices,
        )
        for root in dataset_roots
    ]
    val_datasets = [
        SequenceWindowToFrameDataset(
            RealSequenceTifDataset(dataset_root=root, split="val", **dataset_kwargs),
            frame_indices=frame_indices,
        )
        for root in dataset_roots
    ]

    train_dataset = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(*train_datasets)
    val_dataset = val_datasets[0] if len(val_datasets) == 1 else ConcatDataset(*val_datasets)

    train_loader = torch_em.get_data_loader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch_em.get_data_loader(
        val_dataset, batch_size=max(1, batch_size >> 1), shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader
