import torch
import torch.nn as nn
import torch.nn.functional as F


class ZMemoryAdapter(nn.Module):
    def __init__(self, embed_dim=256):
        """
        embed_dim: SAM 默认的 image embedding 维度通常是 256
        """
        super().__init__()
        # 降维以减少计算量 (Bottleneck 设计，经典的高效微调策略)
        self.down_proj = nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=1)

        # 跨切片空间注意力 (Spatial Attention)
        # 用于对齐 Z 和 Z-1 的形变
        self.spatial_align = nn.Conv2d((embed_dim // 4) * 2, embed_dim // 4, kernel_size=3, padding=1)

        # 升维恢复
        self.up_proj = nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=1)

        # 可学习的缩放因子，初始化为 0。
        # 这样在刚开始训练时，Adapter 不会破坏原始 SAM 的特征。
        self.gamma = nn.Parameter(torch.zeros(1))
        self.act = nn.GELU()

    def forward(self, curr_embed, prev_embed, prev_mask):
        """
        curr_embed: Z 层的特征 [B, 256, 64, 64]
        prev_embed: Z-1 层的特征 [B, 256, 64, 64]
        prev_mask: Z-1 层的分割结果 [B, 1, 64, 64] (需要 downsample 到 64x64)
        """
        # 1. 对 Z-1 层的特征进行 Mask 过滤，只保留前景（目标神经元）的记忆
        memory_feature = prev_embed * prev_mask

        # 2. 降维
        curr_proj = self.down_proj(curr_embed)
        mem_proj = self.down_proj(memory_feature)

        # 3. 特征拼接与对齐注意力
        concat_feat = torch.cat([curr_proj, mem_proj], dim=1)
        aligned_mem = self.act(self.spatial_align(concat_feat))

        # 4. 升维
        mem_residual = self.up_proj(aligned_mem)

        # 5. 残差连接注入到当前切片特征中
        fused_embed = curr_embed + self.gamma * mem_residual

        return fused_embed


def get_embedding_tensor(image_embeddings, t):
    if isinstance(image_embeddings, dict) and "features" in image_embeddings:
        return torch.tensor(image_embeddings["features"][t])
    else:
        raise ValueError("The provided image_embeddings do not contain the expected 'features' key.")
