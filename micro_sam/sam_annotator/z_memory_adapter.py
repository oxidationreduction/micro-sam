import torch
import torch.nn as nn
import torch.nn.functional as F


class ZMemoryAdapter(nn.Module):
    def __init__(self, embed_dim=256):
        """
        记忆适配器：用于跨 Z 轴（时间轴）传递掩码和特征。
        embed_dim: SAM 默认的 image embedding 维度通常是 256
        """
        super().__init__()
        # 降维以减少计算量 (Bottleneck 设计)
        self.down_proj = nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=1)

        # 跨切片空间注意力 (Spatial Alignment)
        # 输入通道为 (embed_dim//4)*2，因为我们要拼接 当前帧特征 和 上一帧记忆特征
        self.spatial_align = nn.Conv2d((embed_dim // 4) * 2, embed_dim // 4, kernel_size=3, padding=1)
        self.act = nn.GELU()

        # 升维恢复到 embed_dim，输出的特征将作为 SAM 的 Dense Prompt Embedding
        self.up_proj = nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=1)

        # 可学习的缩放因子，初始化为零（Zero-initialization）
        # 作用：在训练初期，Adapter 输出全0，不干扰预训练好的 SAM 权重，随着训练逐渐发挥作用。
        self.gamma = nn.Parameter(torch.zeros(1))

    def init_memory(self, batch_size, device):
        """初始化序列的记忆状态。第一帧 (t=0) 之前没有记忆。"""
        return None

    def update_memory(self, memory_state, current_image_embeddings, current_mask):
        """
        更新记忆状态，将当前帧(t)的信息打包，留给下一帧(t+1)使用。

        参数:
        current_image_embeddings: [B, 256, 64, 64]
        current_mask: 预测或真实的掩码，形状可能是 [B, 1, 256, 256] 或 [B, 1, 512, 512]
        """
        # 将高分辨率的 Mask 缩小到与 Image Embedding 一致的 64x64
        # 这样才能在 get_prompts 中进行逐像素特征过滤
        mask_64 = F.interpolate(
            current_mask.float(),
            size=current_image_embeddings.shape[-2:],
            mode="area"  # 掩码下采样推荐使用 area 或 bilinear
        )

        # 将它们存入字典。
        # 注意：这里我们保留了梯度。这样计算图就可以沿着时间轴 (BPTT) 传递。
        # 如果您在训练中遇到了 OOM (显存溢出)，可以在这里加上 .detach() 打断时间轴的梯度回传。
        return {
            "prev_embed": current_image_embeddings,
            "prev_mask": mask_64
        }

    def get_prompts(self, image_embeddings, memory_state):
        """
        利用当前帧特征和上一帧记忆，生成伪装成 SAM 提示词的特征。

        参数:
        image_embeddings: [B, 256, 64, 64]
        """

        # 理论上 t>0 时 memory_state 不会为空，防御性判定
        if memory_state is None:
            N_total = image_embeddings.shape[0]
            dense_embeddings = torch.zeros_like(image_embeddings)
            sparse_embeddings = torch.empty(N_total, 0, 256, device=image_embeddings.device)
            return sparse_embeddings, dense_embeddings
        # prev_embed -> (B, 256, 64, 64), prev_mask -> (B, n_obj, 256, 64, 64)
        prev_embed = memory_state["prev_embed"]
        prev_mask = memory_state["prev_mask"]

        # --- 核心融合逻辑 ---
        # 1. 对 Z-1 层的特征进行 Mask 过滤，只保留前景（目标细胞）的记忆特征
        # 背景区域的特征会被压制为 0
        #
        memory_feature = prev_embed * prev_mask

        # 2. 降维计算 (Bottleneck)
        curr_proj = self.down_proj(image_embeddings)
        mem_proj = self.down_proj(memory_feature)

        # 3. 特征拼接与空间对齐
        concat_feat = torch.cat([curr_proj, mem_proj], dim=1)
        aligned_feat = self.act(self.spatial_align(concat_feat))

        # 4. 升维并乘以 gamma 门控因子
        # 输出形状：[B, 256, 64, 64]
        dense_embeddings = self.up_proj(aligned_feat) * self.gamma

        # 5. 生成空的 Sparse Prompt
        # 因为我们的记忆全部通过 Dense 稠密空间特征传递，不需要点/框提示
        # SAM 要求 sparse_embeddings 的形状为 [B, N, 256]，我们传 N=0
        sparse_embeddings = torch.empty(image_embeddings.shape[0], 0, 256, device=image_embeddings.device)

        return sparse_embeddings, dense_embeddings


def get_embedding_tensor(image_embeddings, t):
    if isinstance(image_embeddings, dict) and "features" in image_embeddings:
        return torch.tensor(image_embeddings["features"][t])
    else:
        raise ValueError("The provided image_embeddings do not contain the expected 'features' key.")
