import argparse
import os

import torch
import torch.distributed as dist

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model

from obtain_lm_datasets import get_generalist_lm_loaders


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


def finetune_lm_generalist(args):
    """Code for finetuning SAM on multiple light microscopy datasets."""
    device, _ = _resolve_device()

    checkpoint_name = f"{args.model_type}/lm_generalist_sam"
    train_loader, val_loader = get_generalist_lm_loaders(
        input_path=args.input_path,
        patch_shape=tuple(args.patch_shape),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_label_transform=True,
        include_cremi=args.include_cremi,
        cremi_input_path=args.cremi_input_path,
    )
    scheduler_kwargs = {"mode": "min", "factor": 0.9, "patience": 5}

    sam_training.train_sam(
        name=checkpoint_name,
        model_type=args.model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        early_stopping=None,
        n_objects_per_batch=args.n_objects,
        checkpoint_path=args.checkpoint_path,
        with_segmentation_decoder=True,
        device=device,
        lr=1e-5,
        n_iterations=args.iterations,
        save_root=args.save_root,
        scheduler_kwargs=scheduler_kwargs,
        verify_n_labels_in_loader=10,
        box_distortion_factor=0.05,
    )

    if args.export_path is not None:
        checkpoint_path = os.path.join(
            "" if args.save_root is None else args.save_root,
            "checkpoints",
            checkpoint_name,
            "best.pt",
        )
        export_custom_sam_model(
            checkpoint_path=checkpoint_path,
            model_type=args.model_type,
            save_path=args.export_path,
        )


def main():
    parser = argparse.ArgumentParser(description="Finetune Segment Anything for the LM datasets.")
    parser.add_argument(
        "--input_path", "-i", default="/home/mira/Downloads/micro-sam/data/light_microscopy/",
        help="The filepath to all the respective LM datasets. If the data does not exist yet it will be downloaded.",
    )
    parser.add_argument(
        "--checkpoint_path", "-c", type=str, default=None,
        help="Path to a checkpoint used as initialization for LM generalist finetuning.",
    )
    parser.add_argument(
        "--include_cremi",
        action="store_true",
        help="Also include CREMI in the generalist training dataset.",
    )
    parser.add_argument(
        "--cremi_input_path",
        type=str,
        default="/mnt/mira_datacenter/liuhongyu2024/project/micro-sam/data/cremi/",
        help="Path to the CREMI dataset root used when --include_cremi is enabled.",
    )
    parser.add_argument(
        "--model_type", "-m", default="vit_l_lm",
        help="The model type to use for fine-tuning. Either 'vit_t', 'vit_b', 'vit_l' or 'vit_h'.",
    )
    parser.add_argument(
        "--patch_shape", type=int, nargs=2, default=[512, 512],
        help="Spatial patch shape for 2D LM finetuning as H W.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size used for LM generalist finetuning.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=24,
        help="Number of data loader workers used for LM generalist finetuning.",
    )
    parser.add_argument(
        "--save_root", "-s", default="/home/mira/Downloads/micro-sam/data/models/",
        help="Where to save the checkpoint and logs. By default they will be saved where this script is run from.",
    )
    parser.add_argument(
        "--iterations", type=int, default=int(2e3),
        help="For how many iterations should the model be trained? By default 2000.",
    )
    parser.add_argument(
        "--export_path", "-e", default="/home/mira/Downloads/micro-sam/data/models/",
        help="Where to export the finetuned model to. The exported model can be used in the annotation tools.",
    )
    parser.add_argument(
        "--n_objects", type=int, default=1,
        help="The number of instances (objects) per batch used for finetuning.",
    )
    args = parser.parse_args()
    finetune_lm_generalist(args)


if __name__ == "__main__":
    main()
