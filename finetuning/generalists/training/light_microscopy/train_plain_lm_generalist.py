import argparse
import os

import torch
import torch.distributed as dist

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model

from obtain_lm_datasets import get_generalist_lm_loaders, get_real_sequence_frame_lm_loaders


def _resolve_device():
    if not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        return torch.device("cpu"), 0

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if "LOCAL_RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        print(f"[Process {local_rank}] Using GPU: {torch.cuda.current_device()}")

    return device, local_rank


def _get_freeze_parts(train_full_model: bool):
    if train_full_model:
        return None
    return ["image_encoder", "prompt_encoder"]


def finetune_lm_generalist_plain(args):
    """Train a plain SAM baseline on either standard LM datasets or real tif sequence data."""
    device, _ = _resolve_device()

    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    n_objects_per_batch = args.n_objects
    sequence_patch_shape = (args.seq_len, args.patch_shape[0], args.patch_shape[1])
    spatial_patch_shape = (args.patch_shape[0], args.patch_shape[1])
    freeze_parts = _get_freeze_parts(args.train_full_model)
    checkpoint_name = (
        f"{model_type}/lm_real_sequence_plain_sam"
        if args.sequence_dataset_root is not None
        else f"{model_type}/lm_generalist_plain_sam"
    )

    if args.sequence_dataset_root is not None:
        batch_size = 4 if args.batch_size is None else args.batch_size
        num_workers = 0 if args.num_workers is None else args.num_workers
        print(
            "Using real tif sequence data for plain-SAM training. "
            "Sequence windows are flattened into independent 2D samples."
        )
        print(f"Sequence dataset root(s): {args.sequence_dataset_root}")
        print(
            f"Sequence label mode: {args.sequence_label_mode}, "
            f"require_full_track={not args.allow_partial_tracks}, "
            f"require_consecutive_slices={not args.allow_slice_gaps}"
        )
        print(
            f"Freeze parts: {freeze_parts if freeze_parts is not None else 'none'}; "
            f"segmentation_decoder={args.with_segmentation_decoder}"
        )
        train_loader, val_loader = get_real_sequence_frame_lm_loaders(
            dataset_root=args.sequence_dataset_root,
            patch_shape=sequence_patch_shape,
            batch_size=batch_size,
            num_workers=num_workers,
            val_fraction=args.val_fraction,
            min_size=args.min_instance_size,
            min_tracking_length=args.min_tracking_length,
            seed=args.seed,
            label_mode=args.sequence_label_mode,
            require_consecutive_slices=not args.allow_slice_gaps,
            require_full_track=not args.allow_partial_tracks,
        )
    else:
        batch_size = 4 if args.batch_size is None else args.batch_size
        num_workers = 24 if args.num_workers is None else args.num_workers
        print("Using the standard 2D LM generalist datasets for plain-SAM training.")
        print(
            f"Freeze parts: {freeze_parts if freeze_parts is not None else 'none'}; "
            f"segmentation_decoder={args.with_segmentation_decoder}"
        )
        train_loader, val_loader = get_generalist_lm_loaders(
            input_path=args.input_path,
            patch_shape=spatial_patch_shape,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    scheduler_kwargs = {"mode": "min", "factor": 0.9, "patience": 5}
    sam_training.train_sam(
        name=checkpoint_name,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        early_stopping=None,
        n_objects_per_batch=n_objects_per_batch,
        checkpoint_path=checkpoint_path,
        with_segmentation_decoder=args.with_segmentation_decoder,
        freeze=freeze_parts,
        device=device,
        lr=args.lr,
        n_iterations=args.iterations,
        save_root=args.save_root,
        scheduler_kwargs=scheduler_kwargs,
        verify_n_labels_in_loader=10,
        box_distortion_factor=0.05,
        mask_prob=0.5,
    )

    if args.export_path is not None:
        best_checkpoint_path = os.path.join(
            "" if args.save_root is None else args.save_root,
            "checkpoints",
            checkpoint_name,
            "best.pt",
        )
        export_custom_sam_model(
            checkpoint_path=best_checkpoint_path,
            model_type=model_type,
            save_path=args.export_path,
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Finetune a plain SAM LM baseline. "
            "When `--sequence_dataset_root` is given, the training uses the same real tif sequence windows "
            "as the memory-adapter pipeline, flattened into 2D frame samples."
        )
    )
    parser.add_argument(
        "--input_path", "-i", default="/home/mira/Downloads/micro-sam/data/light_microscopy/",
        help="Path to the LM datasets used for standard 2D generalist training."
    )
    parser.add_argument(
        "--sequence_dataset_root",
        type=str,
        default=None,
        help=(
            "Optional comma-separated path list to real tif sequence datasets with imagesTr/ and labelsTr/. "
            "If set, the script trains on these real sequence datasets instead of the default 2D LM datasets."
        ),
    )
    parser.add_argument(
        "--sequence_label_mode",
        type=str,
        default="auto",
        choices=("auto", "instance", "binary_3d_cc"),
        help=(
            "How to interpret labels for real tif sequence datasets. "
            "'instance' keeps instance ids, 'binary_3d_cc' builds 3D connected components, "
            "'auto' selects between them based on the label values."
        ),
    )
    parser.add_argument(
        "--allow_partial_tracks",
        action="store_true",
        help="Allow crops where an object is present only in the first few slices instead of the full sequence window.",
    )
    parser.add_argument(
        "--allow_slice_gaps",
        action="store_true",
        help="Do not split a tif stack when numeric slice ids have gaps. By default gaps break a volume.",
    )
    parser.add_argument(
        "--train_full_model",
        action="store_true",
        help=(
            "Train the full SAM model. By default the script freezes image_encoder and prompt_encoder "
            "to better match the memory-adapter comparison setup."
        ),
    )
    parser.add_argument(
        "--with_segmentation_decoder",
        action="store_true",
        help=(
            "Also train the additional UNETR segmentation decoder. Disabled by default to keep the baseline "
            "closer to the memory-adapter comparison."
        ),
    )
    parser.add_argument(
        "--model_type", "-m", default="vit_l_lm",
        help="The model type to use for fine-tuning. Either 'vit_t', 'vit_b', 'vit_l' or 'vit_h'."
    )
    parser.add_argument(
        "--save_root", "-s", default="/home/mira/Downloads/micro-sam/data/models/tmp",
        help="Where to save checkpoints and logs."
    )
    parser.add_argument(
        "--checkpoint_path", "-c", type=str, default=None,
        help="Path to the pre-finetuned micro-sam model used as initialization."
    )
    parser.add_argument(
        "--iterations", type=int, default=int(2e3),
        help="The number of training iterations."
    )
    parser.add_argument(
        "--export_path", "-e", default="/home/mira/Downloads/micro-sam/data/models/results",
        help="Where to export the finetuned model."
    )
    parser.add_argument(
        "--n_objects", type=int, default=1,
        help="The number of instances per batch used for fine-tuning."
    )
    parser.add_argument(
        "--seq_len", type=int, default=64,
        help="Sequence window length used to sample real tif training windows."
    )
    parser.add_argument(
        "--patch_shape", type=int, nargs=2, default=[512, 512],
        help="Spatial patch shape for training as H W."
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override the default batch size. Defaults to 4 for both 2D and real-sequence baseline training."
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="Override the default number of data loader workers. Defaults to 24 for 2D and 0 for real sequences."
    )
    parser.add_argument(
        "--val_fraction", type=float, default=0.2,
        help="Validation split fraction used for the real tif sequence dataset."
    )
    parser.add_argument(
        "--min_instance_size", type=int, default=10,
        help="Minimum instance size passed to the distance-transform label target for the real sequence dataset."
    )
    parser.add_argument(
        "--min_tracking_length", type=int, default=4,
        help="Minimum persistence required for a candidate object when partial tracks are allowed."
    )
    parser.add_argument(
        "--seed", type=int, default=17,
        help="Random seed used by the real tif sequence dataset."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5,
        help="Learning rate for plain-SAM finetuning."
    )
    args = parser.parse_args()
    finetune_lm_generalist_plain(args)


if __name__ == "__main__":
    main()
