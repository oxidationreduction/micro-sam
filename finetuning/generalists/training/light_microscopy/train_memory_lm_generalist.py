import argparse
import os

import torch
import torch.distributed as dist

import micro_sam.training as sam_training
from micro_sam.sam_annotator.z_memory_adapter import ZMemoryAdapter
from micro_sam.util import export_custom_sam_model

from obtain_lm_datasets import get_generalist_lm_loaders, get_real_sequence_lm_loaders


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


def finetune_lm_generalist_memory(args):
    """Code for finetuning SAM with a memory adapter on light microscopy datasets."""
    device, _ = _resolve_device()

    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    patch_shape = (args.seq_len, args.patch_shape[0], args.patch_shape[1])
    n_objects_per_batch = args.n_objects
    checkpoint_name = (
        f"{model_type}/lm_real_sequence_memory_sam"
        if args.sequence_dataset_root is not None
        else f"{model_type}/lm_generalist_memory_sam"
    )

    freeze_parts = ["image_encoder", "prompt_encoder"]
    memory_adapter = ZMemoryAdapter(
        embed_dim=256,
        detach_memory=not args.keep_memory_gradients,
    ).to(device)

    if args.sequence_dataset_root is not None:
        batch_size = 1 if args.batch_size is None else args.batch_size
        num_workers = 0 if args.num_workers is None else args.num_workers
        print(
            "Using real tif sequence data for memory training. "
            "This bypasses the synthetic sequence wrapper used for 2D LM datasets."
        )
        print(f"Sequence dataset root(s): {args.sequence_dataset_root}")
        print(
            f"Sequence label mode: {args.sequence_label_mode}, "
            f"require_full_track={not args.allow_partial_tracks}, "
            f"require_consecutive_slices={not args.allow_slice_gaps}"
        )
        train_loader, val_loader = get_real_sequence_lm_loaders(
            dataset_root=args.sequence_dataset_root,
            patch_shape=patch_shape,
            batch_size=batch_size,
            num_workers=num_workers,
            val_fraction=args.val_fraction,
            min_size=args.min_instance_size,
            min_tracking_length=args.min_tracking_length,
            seed=args.seed,
            label_mode=args.sequence_label_mode,
            require_consecutive_slices=not args.allow_slice_gaps,
            require_full_track=not args.allow_partial_tracks,
            use_label_transform=False,
        )
    else:
        batch_size = 9 if args.batch_size is None else args.batch_size
        num_workers = 24 if args.num_workers is None else args.num_workers
        print(
            "Using synthetic sequences created from 2D LM datasets. "
            "These are not true consecutive microscopy slices."
        )
        train_loader, val_loader = get_generalist_lm_loaders(
            input_path=args.input_path,
            patch_shape=patch_shape,
            batch_size=batch_size,
            num_workers=num_workers,
            use_label_transform=False,
            include_cremi=args.include_cremi,
            cremi_input_path=args.cremi_input_path,
        )

    scheduler_kwargs = {"mode": "min", "factor": 0.9, "patience": 5}
    sam_training.train_mem_sam(
        name=checkpoint_name,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        memory_adapter=memory_adapter,
        seq_len=args.seq_len,
        early_stopping=None,
        n_objects_per_batch=n_objects_per_batch,
        checkpoint_path=checkpoint_path,
        freeze=freeze_parts,
        device=device,
        lr=1e-4,
        decoder_lr=1e-6,
        n_iterations=args.iterations,
        save_root=args.save_root,
        scheduler_kwargs=scheduler_kwargs,
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
    parser = argparse.ArgumentParser(description="Finetune Segment Anything with a memory adapter.")
    parser.add_argument(
        "--input_path", "-i", default="/home/mira/Downloads/micro-sam/data/light_microscopy/",
        help="Path to the LM datasets used for synthetic-sequence training."
    )
    parser.add_argument(
        "--include_cremi",
        action="store_true",
        help="Also include CREMI in the synthetic generalist training set when sequence_dataset_root is not set.",
    )
    parser.add_argument(
        "--cremi_input_path",
        type=str,
        default="/mnt/mira_datacenter/liuhongyu2024/project/micro-sam/data/cremi/",
        help="Path to the CREMI dataset root used when --include_cremi is enabled.",
    )
    parser.add_argument(
        "--sequence_dataset_root",
        type=str,
        default=None,
        help=(
            "Optional comma-separated path list to real tif sequence datasets with imagesTr/ and labelsTr/. "
            "If set, the script trains on these real sequence datasets instead of synthetic sequences."
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
        "--keep_memory_gradients",
        action="store_true",
        help="Keep gradients through memory states across slices. By default memory states are detached to save VRAM.",
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
        help="The number of consecutive frames or slices to load for memory tracking."
    )
    parser.add_argument(
        "--patch_shape", type=int, nargs=2, default=[512, 512],
        help="Spatial patch shape for sequence training as H W."
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override the default batch size. Defaults to 9 for synthetic sequences and 1 for real sequences."
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="Override the default number of data loader workers. Defaults to 24 for synthetic sequences and 0 for real sequences."
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
    args = parser.parse_args()
    finetune_lm_generalist_memory(args)


if __name__ == "__main__":
    main()
