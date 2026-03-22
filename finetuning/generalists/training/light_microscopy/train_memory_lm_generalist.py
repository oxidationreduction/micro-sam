import os
import argparse

import torch
import torch.distributed as dist

import micro_sam.training as sam_training
from micro_sam.sam_annotator.z_memory_adapter import ZMemoryAdapter
from micro_sam.util import export_custom_sam_model

from obtain_lm_datasets import get_generalist_lm_loaders


def finetune_lm_generalist_memory(args):
    """Code for finetuning SAM with Memory Adapter on multiple Light Microscopy datasets."""
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 2. 【最关键的一步】强制当前进程只使用它对应的逻辑 GPU
    torch.cuda.set_device(local_rank)

    # 3. 定义 device 变量，后续模型和数据都 .to(device)
    device = torch.device(f"cuda:{local_rank}")

    # 4. 初始化分布式进程组 (如果是真正的 DDP 训练，必须有这一步)
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        print(f"[Process {local_rank}] Use GPU: {torch.cuda.current_device()}")

    # training settings:
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path  # 必须加载预微调好的基础模型权重

    # 3D 序列格式，支持 seq_len
    patch_shape = (args.seq_len, 512, 512)
    n_objects_per_batch = args.n_objects
    checkpoint_name = f"{model_type}/lm_generalist_memory_sam"

    # 冻结特征提取模块
    freeze_parts = ["image_encoder", "prompt_encoder"]

    # 实例化 Memory Adapter
    # (注：SAM的特征维度均为256，因此可以直接传递 embed_dim=256)
    memory_adapter = ZMemoryAdapter(embed_dim=256).to(device)

    # 获取数据 loaders
    train_loader, val_loader = get_generalist_lm_loaders(input_path=args.input_path, patch_shape=patch_shape, batch_size=8)
    scheduler_kwargs = {"mode": "min", "factor": 0.9, "patience": 5}

    # --- 修复核心：直接调用我们封装好的 train_mem_sam 高层接口 ---
    sam_training.train_mem_sam(
        name=checkpoint_name,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        memory_adapter=memory_adapter,
        seq_len=args.seq_len,
        early_stopping=None,  # Generalist模型通常避免过早停止
        n_objects_per_batch=n_objects_per_batch,
        checkpoint_path=checkpoint_path,
        freeze=freeze_parts,
        device=device,
        lr=1e-4,  # Memory Adapter 的主学习率
        decoder_lr=1e-6,  # 保护原生 Mask Decoder 的微小学习率
        n_iterations=args.iterations,
        save_root=args.save_root,
        scheduler_kwargs=scheduler_kwargs,
        mask_prob=0.5
    )

    # 导出模型供标注工具使用
    if args.export_path is not None:
        best_checkpoint_path = os.path.join(
            "" if args.save_root is None else args.save_root, "checkpoints", checkpoint_name, "best.pt"
        )
        export_custom_sam_model(
            checkpoint_path=best_checkpoint_path, model_type=model_type, save_path=args.export_path
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
        "--checkpoint_path", "-c", type=str, default=None,
        help="Path to the pre-finetuned micro-sam generalist model (e.g., lm_generalist.pt)"
    )
    parser.add_argument(
        "--iterations", type=int, default=int(1e3),
        help="For how many iterations should the model be trained? By default 10k."
    )
    parser.add_argument(
        "--export_path", "-e", default="/home/mira/Downloads/micro-sam/data/models/",
        help="Where to export the finetuned model to. The exported model can be used in the annotation tools."
    )
    parser.add_argument(
        "--n_objects", type=int, default=1, help="The number of instances (objects) per batch used for finetuning."
    )
    parser.add_argument(
        "--seq_len", type=int, default=64, help="The number of consecutive frames/slices to load for memory tracking."
    )
    args = parser.parse_args()
    finetune_lm_generalist_memory(args)


if __name__ == "__main__":
    main()
