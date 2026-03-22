import torch
import torchvision.transforms.functional as TF
import random
from torch.utils.data import Dataset


class SyntheticSequenceWrapper(Dataset):
    def __init__(self, original_dataset, seq_len=3, max_translation=20, max_rotation=10):
        """
        将 2D Dataset 包装为输出 [seq_len, C, H, W] 的 3D Sequence Dataset
        """
        self.original_dataset = original_dataset
        self.seq_len = seq_len
        self.max_translation = max_translation
        self.max_rotation = max_rotation

    def __len__(self):
        return len(self.original_dataset)

    def _apply_random_affine(self, image, label):
        """对图像和标签应用完全相同的随机仿射变换，模拟切片间的偏移/形变"""
        angle = random.uniform(-self.max_rotation, self.max_rotation)
        translate_x = random.randint(-self.max_translation, self.max_translation)
        translate_y = random.randint(-self.max_translation, self.max_translation)
        scale = random.uniform(0.95, 1.05)
        shear = 0.0

        # 注意：image 和 label 必须应用相同的变换参数
        # image 形状为 [C, H, W], label 形状为 [C_label, H, W]
        img_t = TF.affine(image, angle=angle, translate=(translate_x, translate_y), scale=scale, shear=shear)

        # label 通常包含多通道 (如 foreground, distance_map, instance_id)
        # 这里为了保持 instance_id 不受插值破坏，需要使用最近邻插值 (Nearest)
        lbl_t = TF.affine(label, angle=angle, translate=(translate_x, translate_y), scale=scale, shear=shear,
                          interpolation=TF.InterpolationMode.NEAREST)
        return img_t, lbl_t

    def __getitem__(self, idx):
        # 1. 从原始的 torch_em 2D 数据集中获取一对 [C, H, W] 的图像和标签
        x_orig, y_orig = self.original_dataset[idx]

        # 将 numpy 转换为 tensor (如果 torch_em 输出的是 tensor 则跳过)
        if not isinstance(x_orig, torch.Tensor):
            x_orig = torch.from_numpy(x_orig).float()
            y_orig = torch.from_numpy(y_orig).float()

        seq_x = []
        seq_y = []

        # 2. 构造连续序列
        for i in range(self.seq_len):
            if i == 0:
                # 第一帧：原始图像
                seq_x.append(x_orig)
                seq_y.append(y_orig)
            else:
                # 后续帧：在上一帧的基础上叠加微小的随机仿射变换，模拟真实的物理切片漂移或细胞运动
                x_prev, y_prev = seq_x[-1], seq_y[-1]
                x_t, y_t = self._apply_random_affine(x_prev, y_prev)
                seq_x.append(x_t)
                seq_y.append(y_t)

        # 3. 堆叠成 [seq_len, C, H, W]
        x_stacked = torch.stack(seq_x, dim=0)
        y_stacked = torch.stack(seq_y, dim=0)

        return x_stacked, y_stacked