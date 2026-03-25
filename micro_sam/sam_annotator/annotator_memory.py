from typing import Optional, Tuple, Union

import napari
import numpy as np

import torch

from micro_sam import util
from micro_sam.sam_annotator import _widgets as widgets
from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam.sam_annotator._annotator import _AnnotatorBase
from micro_sam.sam_annotator.util import _initialize_parser, _sync_embedding_widget
from magicgui import magicgui

from micro_sam.sam_annotator.z_memory_adapter import ZMemoryAdapter


class AnnotatorMemory(_AnnotatorBase):
    def __init__(self, viewer: "napari.viewer.Viewer", reset_state: bool = True) -> None:
        # 关键修改 1：将 ndim 设为 3，让 Napari 适配 Z 轴滚动
        super().__init__(viewer=viewer, ndim=3)

        # Set the expected annotator class to the state.
        state = AnnotatorState()

        # Reset the state.
        if reset_state:
            state.reset_state()

        state.annotator = self

    def _get_widgets(self):
        autosegment = widgets.AutoSegmentWidget(
            self._viewer, with_decoder=AnnotatorState().decoder is not None, volumetric=False
        )

        # 关键修改 2：新增 Z 轴记忆追踪按钮
        @magicgui(call_button="Track Forward Across Z")
        def track_forward_widget():
            self._run_memory_tracking()

        return {
            "segment": widgets.segment(),
            "autosegment": autosegment,
            "commit": widgets.commit(),
            "clear": widgets.clear(),
            "track_forward": track_forward_widget,  # 挂载追踪按钮
        }

    def _check_stop_condition(self, z_index: int) -> bool:
        """
        终止条件判定逻辑：如果发现当前层有用户标记的负提示（红点），则终止追踪。
        """
        points = self.point_layer.data
        if len(points) == 0:
            return False

        # 获取点的正负标签 (1 为正，0 为负)
        labels = self.point_layer.features.get("label", np.ones(len(points)))

        # napari 的 3D 坐标格式为 [z, y, x]
        points_z = np.round(points[:, 0]).astype(int)

        # 找到属于当前 z_index 层的点
        mask_z = (points_z == z_index)

        if not np.any(mask_z):
            return False  # 这层没有任何用户的点提示，安全，继续追踪

        labels_z = labels[mask_z]

        # 如果当前层只有负提示 (全为 0)，或者包含了某种特殊信号
        # 这就是用户明确下达的“追踪到此结束”指令
        if np.all(labels_z == 0):
            return True

        return False

    def _run_memory_tracking(self):
        """执行沿 Z 轴的特征记忆传递"""
        state = AnnotatorState()
        predictor = state.predictor
        if predictor is None:
            napari.utils.notifications.show_warning("Please compute image embeddings first.")
            return

        sam_model = predictor.model
        if not hasattr(sam_model, "memory_adapter"):
            napari.utils.notifications.show_error("Model does not have a Memory Adapter!")
            return

        memory_adapter = sam_model.memory_adapter

        # 1. 获取用户当前所在的 Z 轴起点
        current_z = int(self._viewer.dims.current_step[0])
        total_z = self.image.shape[0]

        # 2. 提取当前层的分割结果 (作为记忆的起点)
        curr_mask_np = self.segmentation_layer.data[current_z]
        if not np.any(curr_mask_np):
            napari.utils.notifications.show_warning(
                f"No segmentation found on Z={current_z}. Please segment an object first.")
            return

        print(f"Start tracking from Z={current_z}...")

        # 3. 初始化记忆状态
        memory_state = memory_adapter.init_memory(batch_size=1, device=sam_model.device)
        curr_mask_tensor = torch.tensor(curr_mask_np > 0, dtype=torch.float32, device=sam_model.device).unsqueeze(
            0).unsqueeze(0)

        # 获取起点层的图像特征 (利用 SAM 原生 predictor 会自动缓存特征的特性)
        predictor.set_image(self.image[current_z])
        curr_embed = predictor.features

        memory_state = memory_adapter.update_memory(
            memory_state=memory_state,
            current_image_embeddings=curr_embed,
            current_mask=curr_mask_tensor
        )

        # 4. 无限 for 循环向下追踪
        for z in range(current_z + 1, total_z):

            # --- 终止条件 A：人为打断 ---
            if self._check_stop_condition(z):
                print(f"Tracking gracefully stopped at Z={z} due to User Negative Prompt.")
                napari.utils.notifications.show_info(f"Tracking stopped at Z={z} (Negative prompt reached).")
                break

            # 提取当前层图像的特征与尺寸
            predictor.set_image(self.image[z])
            curr_embed = predictor.features
            input_size = predictor.input_size
            original_size = predictor.original_size

            # 生成提示词
            sparse_prompt, dense_prompt = memory_adapter.get_prompts(curr_embed, memory_state)

            # 调用模型解码
            with torch.no_grad():
                low_res_masks, iou_predictions = sam_model.mask_decoder(
                    image_embeddings=curr_embed,
                    image_pe=sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_prompt,
                    dense_prompt_embeddings=dense_prompt,
                    multimask_output=False
                )

                # 恢复到原图尺寸
                masks = sam_model.postprocess_masks(
                    low_res_masks,
                    input_size=input_size,
                    original_size=original_size
                )

            # 二值化
            mask_prob = torch.sigmoid(masks)
            curr_mask_tensor = (mask_prob > 0.5).float()
            curr_mask_np = curr_mask_tensor[0, 0].cpu().numpy()

            # --- 终止条件 B：自动消失 ---
            if np.sum(curr_mask_np) < 128:
                print(f"Tracking automatically stopped at Z={z}: Object disappeared.")
                napari.utils.notifications.show_info(f"Tracking stopped at Z={z}: Object disappeared.")
                break

            # 更新 Napari 界面
            self.segmentation_layer.data[z] = curr_mask_np
            self.segmentation_layer.refresh()
            self._viewer.dims.current_step = (z, *self._viewer.dims.current_step[1:])  # 界面同步滚动

            # 击鼓传花：更新记忆给下一层使用
            memory_state = memory_adapter.update_memory(
                memory_state=memory_state,
                current_image_embeddings=curr_embed,
                current_mask=curr_mask_tensor
            )

        print("Tracking finished successfully!")


def annotator_memory_3d(
        image: np.ndarray,
        embedding_path: Optional[Union[str, util.ImageEmbeddings]] = None,
        segmentation_result: Optional[np.ndarray] = None,
        model_type: str = util._DEFAULT_MODEL,
        tile_shape: Optional[Tuple[int, int]] = None,
        halo: Optional[Tuple[int, int]] = None,
        return_viewer: bool = False,
        viewer: Optional["napari.viewer.Viewer"] = None,
        checkpoint_path: Optional[str] = None,
):
    # 初始化状态
    state = AnnotatorState()
    state.image_shape = image.shape[:-1] if image.ndim == 4 else image.shape

    # 创建 3D viewer
    if viewer is None:
        viewer = napari.Viewer()
    viewer.add_image(image, name="image")

    # 实例化我们的专属追踪插件
    annotator = AnnotatorMemory(viewer=viewer)

    # 1. 初始化原生的 SAM 预测器 (如果没传 checkpoint，这里加载的是官方预训练权重)
    # 注意：我们在这里先不传 checkpoint_path 给原生函数，由我们自己接管加载逻辑
    state.initialize_predictor(
        image, model_type=model_type, save_path=embedding_path,
        ndim=3, device=util.get_device(),
        tile_shape=tile_shape, halo=halo,
    )

    sam_model = state.predictor.model

    # 2. 核心注入：实例化 ZMemoryAdapter 并挂载到模型上
    print("Injecting ZMemoryAdapter into SAM...")
    memory_adapter = ZMemoryAdapter(embed_dim=256).to(sam_model.device)
    sam_model.memory_adapter = memory_adapter

    # 3. 手动加载您微调的 Checkpoint
    if checkpoint_path is not None:
        print(f"Loading custom fine-tuned weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=sam_model.device)

        # micro-sam/torch_em 保存的权重通常在 "model_state" 键下
        model_state = checkpoint.get("model_state", checkpoint)

        # 加载权重
        # strict=False 是必须的，因为我们可能只微调了 Adapter 和部分 Encoder
        missing_keys, unexpected_keys = sam_model.load_state_dict(model_state, strict=False)

        # 如果模型训练时将 memory_adapter 的权重独立保存了，尝试单独加载
        # (如果在微调代码中 adapter 是直接挂载在 model 里的，上面那句 load_state_dict 就已经加载成功了)
        try:
            # 兼容独立的 adapter 权重保存格式
            adapter_keys = {k.replace("memory_adapter.", ""): v for k, v in model_state.items() if
                            "memory_adapter" in k}
            if len(adapter_keys) > 0:
                sam_model.memory_adapter.load_state_dict(adapter_keys, strict=False)
                print("Memory Adapter weights loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load specific adapter weights. {e}")

        print("Custom Checkpoint loading complete!")

    # 刷新图像和图层
    annotator._update_image(segmentation_result=segmentation_result)
    viewer.window.add_dock_widget(annotator)

    if return_viewer:
        return viewer

    napari.run()


def main():
    """@private"""
    parser = _initialize_parser(description="Run Memory Tracking Segmentation for an image volume.")
    args = parser.parse_args()

    # 加载 3D 图像
    image = util.load_image_data(args.input, key=args.key)

    if args.segmentation_result is None:
        segmentation_result = None
    else:
        segmentation_result = util.load_image_data(args.segmentation_result, key=args.segmentation_key)

    annotator_memory_3d(
        image, embedding_path=args.embedding_path,
        segmentation_result=segmentation_result,
        model_type=args.model_type, tile_shape=args.tile_shape, halo=args.halo,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()