import os
import argparse
import torch

# 使用 torch_em 的通用数据加载器来加载自定义 EM 数据
from torch_em.default_segmentation_dataset import get_data_loader
from torch_em.transform.label import PerObjectDistanceTransform
import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model

def get_dataloaders(patch_shape, data_path, batch_size, n_objects):
    """
    加载自定义的 EM 实例分割数据集。
    假设你的数据目录结构如下：
    data_path/
      ├── train/
      │   ├── images/ (存放原图，例如 .tif)
      │   └── labels/ (存放实例掩码，每个细胞一个独立ID，0为背景)
      └── val/
          ├── images/
          └── labels/
    """
    # 标签转换：为实例分割生成边界、距离变换等（micro-sam 微调必须项）
    label_transform = PerObjectDistanceTransform(
        distances=True, boundary_distances=True, directed_distances=False,
        foreground=True, instances=True, min_size=25
    )
    raw_transform = sam_training.identity  # 保持原图输入（SAM自带归一化）

    train_image_dir = os.path.join(data_path, "train", "images")
    train_label_dir = os.path.join(data_path, "train", "labels")
    val_image_dir = os.path.join(data_path, "val", "images")
    val_label_dir = os.path.join(data_path, "val", "labels")

    train_loader = get_data_loader(
        image_paths=train_image_dir, image_pattern="*.tif",
        label_paths=train_label_dir, label_pattern="*.tif",
        patch_shape=patch_shape, batch_size=batch_size,
        label_transform=label_transform, raw_transform=raw_transform,
        num_workers=8, shuffle=True, is_seg_dataset=True
    )

    val_loader = get_data_loader(
        image_paths=val_image_dir, image_pattern="*.tif",
        label_paths=val_label_dir, label_pattern="*.tif",
        patch_shape=patch_shape, batch_size=1, # 验证集通常 batch_size=1
        label_transform=label_transform, raw_transform=raw_transform,
        num_workers=8, shuffle=False, is_seg_dataset=True
    )

    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser(description="EM Cell Instance Segmentation Finetuning for SAM")
    parser.add_argument("-i", "--input_path", required=True, help="自定义 EM 数据集的根目录")
    parser.add_argument("-m", "--model_type", default="vit_b", help="基础 SAM 模型类型: vit_t, vit_b, vit_l, vit_h")
    parser.add_argument("-s", "--save_root", required=True, help="保存 checkpoints 和 logs 的路径")
    parser.add_argument("-e", "--export_path", required=True, help="微调结束后导出供 UI 使用的 .pth 路径")
    parser.add_argument("--iterations", type=int, default=100000, help="微调的迭代次数")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--n_objects", type=int, default=25, help="每个 batch 采样的最大细胞实例数量")
    parser.add_argument("--patch_shape", type=int, nargs=2, default=[1024, 1024], help="训练时的裁剪尺寸 (H, W)")
    args = parser.parse_args()

    # 1. 获取 DataLoader
    print("Initializing DataLoaders...")
    train_loader, val_loader = get_dataloaders(
        patch_shape=tuple(args.patch_shape),
        data_path=args.input_path,
        batch_size=args.batch_size,
        n_objects=args.n_objects
    )

    # 2. 核心微调逻辑
    print(f"Starting SAM finetuning for model: {args.model_type}")
    sam_training.train_sam(
        name=f"em_cell_finetuning_{args.model_type}",
        model_type=args.model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=None,               # 由 iterations 控制
        n_iterations=args.iterations,
        save_root=args.save_root,
        n_objects_per_batch=args.n_objects,
        with_segmentation_decoder=True, # 训练专门用于实例分割的解码器
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # 3. 导出模型（使其能够被 micro-sam 的 Napari 插件直接加载）
    print(f"Exporting finetuned model to {args.export_path}...")
    checkpoint_path = os.path.join(args.save_root, "checkpoints", f"em_cell_finetuning_{args.model_type}", "best.pt")
    export_custom_sam_model(
        checkpoint_path=checkpoint_path,
        model_type=args.model_type,
        save_path=args.export_path,
    )
    print("Done!")

if __name__ == "__main__":
    main()