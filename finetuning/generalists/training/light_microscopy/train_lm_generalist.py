import os
import argparse

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
    torch.cuda.set_device(int(local_rank))
    device = torch.device(f"cuda:{int(local_rank)}")

    if "LOCAL_RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        print(f"[Process {local_rank}] Using GPU: {torch.cuda.current_device()}")

    return device, local_rank


def finetune_lm_generalist(args):
    """Code for finetuning SAM on multiple Light Microscopy datasets."""
    device, _ = _resolve_device()

    # 2. 【最关键的一步】强制当前进程只使用它对应的逻辑 GPU

    # 3. 定义 device 变量，后续模型和数据都 .to(device)

    # 4. 初始化分布式进程组 (如果是真正的 DDP 训练，必须有这一步)

    # training settings:
    model_type = args.model_type
    checkpoint_path = None  # override this to start training from a custom checkpoint
    patch_shape = (512, 512)  # the patch shape for training
    n_objects_per_batch = args.n_objects  # this is the number of objects per batch that will be sampled (default: 25)
    checkpoint_name = f"{(model_type)}/lm_generalist_sam"

    # all the stuff we need for training
    train_loader, val_loader = get_generalist_lm_loaders(input_path=args.input_path, patch_shape=patch_shape, batch_size=4)
    scheduler_kwargs = {"mode": "min", "factor": 0.9, "patience": 5}

    # Run training.
    sam_training.train_sam(
        name=checkpoint_name,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        early_stopping=None,  # Avoid early stopping for training the generalist model.
        n_objects_per_batch=n_objects_per_batch,
        checkpoint_path=checkpoint_path,
        with_segmentation_decoder=True,
        device=device,
        lr=1e-5,
        n_iterations=args.iterations,
        save_root=args.save_root,
        scheduler_kwargs=scheduler_kwargs,
        verify_n_labels_in_loader=10,  # Verifies all labels in the loader(s).
        box_distortion_factor=0.05,
    )

    if args.export_path is not None:
        checkpoint_path = os.path.join(
            "" if args.save_root is None else args.save_root, "checkpoints", checkpoint_name, "best.pt"
        )
        export_custom_sam_model(
            checkpoint_path=checkpoint_path, model_type=model_type, save_path=args.export_path
        )


def main():
    parser = argparse.ArgumentParser(description="Finetune Segment Anything for the LM datasets.")
    parser.add_argument(
        "--input_path", "-i", default="/home/mira/Downloads/micro-sam/data/light_microscopy/",
        help="The filepath to all the respective LM datasets. If the data does not exist yet it will be downloaded"
    )
    parser.add_argument(
        "--model_type", "-m", default="vit_l_lm",
        help="The model type to use for fine-tuning. Either 'vit_t', 'vit_b', 'vit_l' or 'vit_h'."
    )
    parser.add_argument(
        "--save_root", "-s", default="/home/mira/Downloads/micro-sam/data/models/",
        help="Where to save the checkpoint and logs. By default they will be saved where this script is run from."
    )
    parser.add_argument(
        "--iterations", type=int, default=int(2e3),
        help="For how many iterations should the model be trained? By default 250k."
    )
    parser.add_argument(
        "--export_path", "-e", default="/home/mira/Downloads/micro-sam/data/models/",
        help="Where to export the finetuned model to. The exported model can be used in the annotation tools."
    )
    parser.add_argument(
        "--n_objects", type=int, default=1, help="The number of instances (objects) per batch used for finetuning."
    )
    args = parser.parse_args()
    finetune_lm_generalist(args)


if __name__ == "__main__":
    main()
