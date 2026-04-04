"""Multi-dimensional segmentation with segment anything.
"""

import os
import multiprocessing as mp
from concurrent import futures
from typing import Dict, List, Optional, Union, Tuple

import networkx as nx
import nifty
import numpy as np
import torch
from scipy.ndimage import binary_closing
from skimage.measure import label, regionprops
from skimage.segmentation import relabel_sequential

import elf.segmentation as seg_utils
import elf.tracking.tracking_utils as track_utils
from elf.tracking.motile_tracking import recolor_segmentation

from segment_anything.predictor import SamPredictor

from .sam_annotator.z_memory_adapter import get_module_device, mask_to_memory_tensor, predict_from_memory_state

try:
    from napari.utils import progress as tqdm
except ImportError:
    from tqdm import tqdm

try:
    from trackastra.model import Trackastra
    from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks
except ImportError:
    Trackastra = None
    graph_to_ctc = None
    graph_to_napari_tracks = None


from . import util
from .prompt_based_segmentation import segment_from_mask
from .instance_segmentation import AMGBase


PROJECTION_MODES = ("box", "mask", "points", "points_and_mask", "single_point")


def _validate_projection(projection):
    use_single_point = False
    if isinstance(projection, str):
        if projection == "mask":
            use_box, use_mask, use_points = True, True, False
        elif projection == "points":
            use_box, use_mask, use_points = False, False, True
        elif projection == "box":
            use_box, use_mask, use_points = True, False, False
        elif projection == "points_and_mask":
            use_box, use_mask, use_points = False, True, True
        elif projection == "single_point":
            use_box, use_mask, use_points = False, False, True
            use_single_point = True
        else:
            raise ValueError(
                "Choose projection method from 'mask' / 'points' / 'box' / 'points_and_mask' / 'single_point'. "
                f"You have passed the invalid option {projection}."
            )
    elif isinstance(projection, dict):
        assert len(projection.keys()) == 3, "There should be three parameters assigned for the projection method."
        use_box, use_mask, use_points = projection["use_box"], projection["use_mask"], projection["use_points"]
    else:
        raise ValueError(f"{projection} is not a supported projection method.")
    return use_box, use_mask, use_points, use_single_point


# Advanced stopping criterions.
# In practice these did not make a big difference, so we do not use this at the moment.
# We still leave it here for reference.
def _advanced_stopping_criteria(
    z, seg_z, seg_prev, z_start, z_increment, segmentation, criterion_choice, score, increment
):
    def _compute_mean_iou_for_n_slices(z, increment, seg_z, n_slices):
        iou_list = [
            util.compute_iou(segmentation[z - increment * _slice], seg_z) for _slice in range(1, n_slices+1)
        ]
        return np.mean(iou_list)

    if criterion_choice == 1:
        # 1. current metric: iou of current segmentation and the previous slice
        iou = util.compute_iou(seg_prev, seg_z)
        criterion = iou

    elif criterion_choice == 2:
        # 2. combining SAM iou + iou: curr. slice & first segmented slice + iou: curr. slice vs prev. slice
        iou = util.compute_iou(seg_prev, seg_z)
        ff_iou = util.compute_iou(segmentation[z_start], seg_z)
        criterion = 0.5 * iou + 0.3 * score + 0.2 * ff_iou

    elif criterion_choice == 3:
        # 3. iou of current segmented slice w.r.t the previous n slices
        criterion = _compute_mean_iou_for_n_slices(z, increment, seg_z, min(5, abs(z - z_start)))

    return criterion


def _segment_mask_in_volume_legacy(
    segmentation: np.ndarray,
    predictor: SamPredictor,
    image_embeddings: util.ImageEmbeddings,
    segmented_slices: np.ndarray,
    stop_lower: bool,
    stop_upper: bool,
    iou_threshold: float,
    projection: Union[str, dict],
    update_progress: Optional[callable] = None,
    box_extension: float = 0.0,
    verbose: bool = False,
    memory_adapter = None
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Segment an object mask in volumetric data.

    Args:
        segmentation: The initial segmentation for the object.
        predictor: The Segment Anything predictor.
        image_embeddings: The precomputed image embeddings for the volume.
        segmented_slices: List of slices for which this object has already been segmented.
        stop_lower: Whether to stop at the lowest segmented slice.
        stop_upper: Whether to stop at the topmost segmented slice.
        iou_threshold: The IOU threshold for continuing segmentation across 3d.
        projection: The projection method to use. One of 'box', 'mask', 'points', 'points_and_mask' or 'single point'.
            Pass a dictionary to choose the exact combination of projection modes.
        update_progress: Callback to update an external progress bar.
        box_extension: Extension factor for increasing the box size after projection.
            By default, does not increase the projected box size.
        verbose: Whether to print details about the segmentation steps. By default, set to 'True'.

    Returns:
        Array with the volumetric segmentation.
        Tuple with the first and last segmented slice.
    """
    return segment_mask_in_volume(
        segmentation=segmentation,
        predictor=predictor,
        image_embeddings=image_embeddings,
        segmented_slices=segmented_slices,
        stop_lower=stop_lower,
        stop_upper=stop_upper,
        iou_threshold=iou_threshold,
        projection=projection,
        update_progress=update_progress,
        box_extension=box_extension,
        verbose=verbose,
        memory_adapter=memory_adapter,
    )
    # 验证projection模式，确定使用哪些prompt类型（mask, box, points等）
    # 这决定了如何从上一层的分割结果生成prompt传递给SAM
    use_box, use_mask, use_points, use_single_point = _validate_projection(projection)

    # 如果没有提供进度更新函数，定义一个空函数
    if update_progress is None:
        def update_progress(*args):
            pass

    # 定义内部函数segment_range，用于在指定范围内逐步分割slice
    # 思路：从z_start开始，按照increment方向（+1或-1）逐步向z_stop扩展
    # 每次使用上一层的分割结果作为mask prompt，通过SAM分割当前层
    # 如果IOU低于阈值，则停止扩展，以避免错误传播
    def segment_range(z_start, z_stop, increment, stopping_criterion, threshold=None, verbose=False):
        z = z_start + increment  # 从起始点开始，移动到下一层
        while True:
            if verbose:
                print(f"Segment {z_start} to {z_stop}: segmenting slice {z}")
            # 获取上一层的分割结果，作为prompt
            seg_prev = segmentation[z - increment]
            if memory_adapter is not None:
                device = "cuda"  # 确保一切都在 GPU 上运行
                with torch.no_grad():
                    # 1. 提取当前层 (z) 和 参考层 (z - increment) 的原始特征
                    curr_embed = get_embedding_tensor(image_embeddings, z).to(device)
                    prev_embed = get_embedding_tensor(image_embeddings, z - increment).to(device)

                    # 2. 将参考层的 2D Mask 转换为 Adapter 需要的低分辨率张量
                    # 先转成 tensor 并增加 batch 和 channel 维度: [1, 1, H, W]
                    prev_mask_tensor = torch.tensor(seg_prev).unsqueeze(0).unsqueeze(0).float().to(device)
                    # 使用双线性插值降采样到 SAM 特征图的默认尺寸 64x64
                    prev_mask_low_res = torch.nn.functional.interpolate(
                        prev_mask_tensor, size=(64, 64), mode='bilinear'
                    )
                    # 二值化，确保它是一个清晰的 0/1 掩码
                    prev_mask_low_res = (prev_mask_low_res > 0.5).float()

                    # 3. 调用 Adapter 网络计算 fused_embed！
                    fused_embed = memory_adapter(curr_embed, prev_embed, prev_mask_low_res)

                    # 4. 强行把 SAM Predictor 肚子里的特征替换成我们的增强版特征
                    predictor.features = fused_embed
                    predictor.is_image_set = True
            # 使用SAM的segment_from_mask函数，基于上一层mask生成当前层的分割
            # projection模式控制如何使用mask（例如，是否结合box或points）
            seg_z, score, _ = segment_from_mask(
                predictor, seg_prev, image_embeddings=image_embeddings, i=z, use_mask=use_mask,
                use_box=use_box, use_points=use_points, box_extension=box_extension, return_all=True,
                use_single_point=use_single_point,
            )
            # 如果设置了阈值，计算IOU，如果低于阈值则停止
            if threshold is not None:
                iou = util.compute_iou(seg_prev, seg_z)
                if iou < threshold:
                    if verbose:
                        msg = f"Segmentation stopped at slice {z} due to IOU {iou} < {threshold}."
                        print(msg)
                    break

            # 将分割结果保存到segmentation数组
            segmentation[z] = seg_z
            # 移动到下一层
            z += increment
            # 检查是否达到停止条件（例如，z >= z_stop）
            if stopping_criterion(z, z_stop):
                if verbose:
                    print(f"Segment {z_start} to {z_stop}: stop at slice {z}")
                break
            # 更新进度
            update_progress(1)

        # 返回最后分割的slice索引
        return z - increment

    # 获取已分割slice的范围：z0是最小slice索引，z1是最大slice索引
    z0, z1 = int(segmented_slices.min()), int(segmented_slices.max())

    # 第一步：分割低于z0的slice（向下扩展）
    # 思路：如果z0 > 0且不强制停止，则从z0向下扩展到slice 0
    # 使用segment_range函数，increment=-1（向下），直到z < 0
    if z0 > 0 and not stop_lower:
        z_min = segment_range(z0, 0, -1, np.less, iou_threshold, verbose=verbose)
    else:
        z_min = z0  # 如果不能扩展，z_min设为z0

    # 第二步：分割高于z1的slice（向上扩展）
    # 思路：如果z1 < 总slice数-1且不强制停止，则从z1向上扩展到最后slice
    # 使用segment_range函数，increment=1（向上），直到z > 最后索引
    if z1 < segmentation.shape[0] - 1 and not stop_upper:
        z_max = segment_range(z1, segmentation.shape[0] - 1, 1, np.greater, iou_threshold, verbose=verbose)
    else:
        z_max = z1  # 如果不能扩展，z_max设为z1

    # 第三步：分割z0和z1之间的slice（填充中间gap）
    # 思路：遍历segmented_slices的相邻对，处理不同间距情况
    # 目的是确保所有slice都被分割，即使有gap
    if z0 != z1:
        for z_start, z_stop in zip(segmented_slices[:-1], segmented_slices[1:]):
            slice_diff = z_stop - z_start  # 计算间距
            z_mid = int((z_start + z_stop) // 2)  # 中间slice索引

            if slice_diff == 1:  # 相邻slice，无需操作
                pass

            elif z_start == z0 and stop_lower:  # 下边界是stop，从上向下分割
                segment_range(z_stop, z_start, -1, np.less_equal, verbose=verbose)

            elif z_stop == z1 and stop_upper:  # 上边界是stop，从下向上分割
                segment_range(z_start, z_stop, 1, np.greater_equal, verbose=verbose)

            elif slice_diff == 2:  # 间距为2，只有一个中间slice，使用组合mask
                z = z_start + 1  # 中间slice
                # 组合z_start和z_stop的mask，作为prompt（逻辑或操作）
                seg_prompt = np.logical_or(segmentation[z_start] == 1, segmentation[z_stop] == 1)
                # 使用组合mask分割中间slice
                segmentation[z] = segment_from_mask(
                    predictor, seg_prompt, image_embeddings=image_embeddings, i=z,
                    use_mask=use_mask, use_box=use_box, use_points=use_points,
                    box_extension=box_extension
                )
                update_progress(1)

            else:  # 间距>2，范围分割
                # 从下向上分割到z_mid
                segment_range(
                    z_start, z_mid, 1, np.greater_equal if slice_diff % 2 == 0 else np.greater, verbose=verbose
                )
                # 从上向下分割到z_mid
                segment_range(z_stop, z_mid, -1, np.less_equal, verbose=verbose)
                # 如果间距偶数，中间slice未被分割，使用相邻slice的组合mask
                if slice_diff % 2 == 0:
                    seg_prompt = np.logical_or(segmentation[z_mid - 1] == 1, segmentation[z_mid + 1] == 1)
                    segmentation[z_mid] = segment_from_mask(
                        predictor, seg_prompt, image_embeddings=image_embeddings, i=z_mid,
                        use_mask=use_mask, use_box=use_box, use_points=use_points,
                        box_extension=box_extension
                    )
                    update_progress(1)

    # 返回更新后的segmentation和扩展后的范围(z_min, z_max)
    return segmentation, (z_min, z_max)


def segment_mask_in_volume(
    segmentation: np.ndarray,
    predictor: SamPredictor,
    image_embeddings: util.ImageEmbeddings,
    segmented_slices: np.ndarray,
    stop_lower: bool,
    stop_upper: bool,
    iou_threshold: float,
    projection: Union[str, dict],
    update_progress: Optional[callable] = None,
    box_extension: float = 0.0,
    verbose: bool = False,
    memory_adapter=None
) -> Tuple[np.ndarray, Tuple[int, int]]:
    use_box, use_mask, use_points, use_single_point = _validate_projection(projection)

    if update_progress is None:
        def update_progress(*args):
            pass

    model_device = get_module_device(predictor.model)
    if memory_adapter is not None:
        predictor.model.memory_adapter = memory_adapter

    def _init_memory_state(z_index: int):
        if memory_adapter is None:
            return None

        memory_state = memory_adapter.init_memory(batch_size=1, device=model_device)
        util.set_precomputed(predictor, image_embeddings, z_index)
        return memory_adapter.update_memory(
            memory_state=memory_state,
            current_image_embeddings=predictor.features,
            current_mask=mask_to_memory_tensor(segmentation[z_index], model_device),
        )

    def segment_range(z_start, z_stop, increment, stopping_criterion, threshold=None, verbose=False):
        z = z_start + increment
        memory_state = _init_memory_state(z_start)

        while True:
            if verbose:
                print(f"Segment {z_start} to {z_stop}: segmenting slice {z}")

            seg_prev = segmentation[z - increment]
            if memory_adapter is not None:
                seg_z, curr_embed, _, _ = predict_from_memory_state(
                    predictor=predictor,
                    memory_state=memory_state,
                    image_embeddings=image_embeddings,
                    index=z,
                    mask_threshold=0.5,
                )
                seg_z = seg_z.astype(segmentation.dtype, copy=False)
            else:
                seg_z, _, _ = segment_from_mask(
                    predictor, seg_prev, image_embeddings=image_embeddings, i=z, use_mask=use_mask,
                    use_box=use_box, use_points=use_points, box_extension=box_extension, return_all=True,
                    use_single_point=use_single_point,
                )

            if threshold is not None:
                iou = util.compute_iou(seg_prev, seg_z)
                if iou < threshold:
                    if verbose:
                        print(f"Segmentation stopped at slice {z} due to IOU {iou} < {threshold}.")
                    break

            segmentation[z] = seg_z
            if memory_adapter is not None:
                memory_state = memory_adapter.update_memory(
                    memory_state=memory_state,
                    current_image_embeddings=curr_embed,
                    current_mask=mask_to_memory_tensor(seg_z, model_device),
                )

            z += increment
            if stopping_criterion(z, z_stop):
                if verbose:
                    print(f"Segment {z_start} to {z_stop}: stop at slice {z}")
                break
            update_progress(1)

        return z - increment

    z0, z1 = int(segmented_slices.min()), int(segmented_slices.max())

    if z0 > 0 and not stop_lower:
        z_min = segment_range(z0, 0, -1, np.less, iou_threshold, verbose=verbose)
    else:
        z_min = z0

    if z1 < segmentation.shape[0] - 1 and not stop_upper:
        z_max = segment_range(z1, segmentation.shape[0] - 1, 1, np.greater, iou_threshold, verbose=verbose)
    else:
        z_max = z1

    if z0 != z1:
        for z_start, z_stop in zip(segmented_slices[:-1], segmented_slices[1:]):
            slice_diff = z_stop - z_start
            z_mid = int((z_start + z_stop) // 2)

            if slice_diff == 1:
                pass
            elif z_start == z0 and stop_lower:
                segment_range(z_stop, z_start, -1, np.less_equal, verbose=verbose)
            elif z_stop == z1 and stop_upper:
                segment_range(z_start, z_stop, 1, np.greater_equal, verbose=verbose)
            elif slice_diff == 2 and memory_adapter is not None:
                segment_range(z_start, z_start + 1, 1, np.greater_equal, verbose=verbose)
            elif slice_diff == 2:
                z = z_start + 1
                seg_prompt = np.logical_or(segmentation[z_start] == 1, segmentation[z_stop] == 1)
                segmentation[z] = segment_from_mask(
                    predictor, seg_prompt, image_embeddings=image_embeddings, i=z,
                    use_mask=use_mask, use_box=use_box, use_points=use_points,
                    box_extension=box_extension
                )
                update_progress(1)
            else:
                segment_range(
                    z_start, z_mid, 1, np.greater_equal if slice_diff % 2 == 0 else np.greater, verbose=verbose
                )
                segment_range(z_stop, z_mid, -1, np.less_equal, verbose=verbose)
                if slice_diff % 2 == 0:
                    seg_prompt = np.logical_or(segmentation[z_mid - 1] == 1, segmentation[z_mid + 1] == 1)
                    segmentation[z_mid] = segment_from_mask(
                        predictor, seg_prompt, image_embeddings=image_embeddings, i=z_mid,
                        use_mask=use_mask, use_box=use_box, use_points=use_points,
                        box_extension=box_extension
                    )
                    update_progress(1)

    return segmentation, (z_min, z_max)


def _preprocess_closing(slice_segmentation, gap_closing, pbar_update):
    binarized = slice_segmentation > 0
    # Use a structuring element that only closes elements in z, to avoid merging objects in-plane.
    structuring_element = np.zeros((3, 1, 1))
    structuring_element[:, 0, 0] = 1
    closed_segmentation = binary_closing(binarized, iterations=gap_closing, structure=structuring_element)

    new_segmentation = np.zeros_like(slice_segmentation)
    n_slices = new_segmentation.shape[0]

    def process_slice(z, offset):
        seg_z = slice_segmentation[z]

        # Closing does not work for the first and last gap slices
        if z < gap_closing or z >= (n_slices - gap_closing):
            seg_z, _, _ = relabel_sequential(seg_z, offset=offset)
            offset = int(seg_z.max()) + 1
            return seg_z, offset

        # Apply connected components to the closed segmentation.
        closed_z = label(closed_segmentation[z])

        # Map objects in the closed and initial segmentation.
        # We take objects from the closed segmentation unless they
        # have overlap with more than one object from the initial segmentation.
        # This indicates wrong merging of closeby objects that we want to prevent.
        matches = nifty.ground_truth.overlap(closed_z, seg_z)
        matches = {
            seg_id: matches.overlapArrays(seg_id, sorted=False)[0] for seg_id in range(1, int(closed_z.max() + 1))
        }
        matches = {k: v[v != 0] for k, v in matches.items()}

        ids_initial, ids_closed = [], []
        for seg_id, matched in matches.items():
            if len(matched) > 1:
                ids_initial.extend(matched.tolist())
            else:
                ids_closed.append(seg_id)

        seg_new = np.zeros_like(seg_z)
        closed_mask = np.isin(closed_z, ids_closed)
        seg_new[closed_mask] = closed_z[closed_mask]

        if ids_initial:
            initial_mask = np.isin(seg_z, ids_initial)
            seg_new[initial_mask] = relabel_sequential(seg_z[initial_mask], offset=seg_new.max() + 1)[0]

        seg_new, _, _ = relabel_sequential(seg_new, offset=offset)
        max_z = seg_new.max()
        if max_z > 0:
            offset = int(max_z) + 1

        return seg_new, offset

    # Further optimization: parallelize
    offset = 1
    for z in range(n_slices):
        new_segmentation[z], offset = process_slice(z, offset)
        pbar_update(1)

    return new_segmentation


def _filter_z_extent(segmentation, min_z_extent):
    props = regionprops(segmentation)
    filter_ids = []
    for prop in props:
        box = prop.bbox
        z_extent = box[3] - box[0]
        if z_extent < min_z_extent:
            filter_ids.append(prop.label)
    if filter_ids:
        segmentation[np.isin(segmentation, filter_ids)] = 0
    return segmentation


def merge_instance_segmentation_3d(
    slice_segmentation: np.ndarray,
    beta: float = 0.5,
    with_background: bool = True,
    gap_closing: Optional[int] = None,
    min_z_extent: Optional[int] = None,
    verbose: bool = True,
    pbar_init: Optional[callable] = None,
    pbar_update: Optional[callable] = None,
) -> np.ndarray:
    """Merge stacked 2d instance segmentations into a consistent 3d segmentation.

    Solves a multicut problem based on the overlap of objects to merge across z.

    Args:
        slice_segmentation: The stacked segmentation across the slices.
            We assume that the segmentation is labeled consecutive across z.
        beta: The bias term for the multicut. Higher values lead to a larger
            degree of over-segmentation and vice versa. by default, set to '0.5'.
        with_background: Whether this is a segmentation problem with background.
            In that case all edges connecting to the background are set to be repulsive.
            By default, set to 'True'.
        gap_closing: If given, gaps in the segmentation are closed with a binary closing
            operation. The value is used to determine the number of iterations for the closing.
        min_z_extent: Require a minimal extent in z for the segmented objects.
            This can help to prevent segmentation artifacts.
        verbose: Verbosity flag. By default, set to 'True'.
        pbar_init: Callback to initialize an external progress bar. Must accept number of steps and description.
            Can be used together with pbar_update to handle napari progress bar in other thread.
            To enables using this function within a threadworker.
        pbar_update: Callback to update an external progress bar.

    Returns:
        The merged segmentation.
    """
    _, pbar_init, pbar_update, pbar_close = util.handle_pbar(verbose, pbar_init, pbar_update)

    if gap_closing is not None and gap_closing > 0:
        pbar_init(slice_segmentation.shape[0] + 1, "Merge segmentation")
        slice_segmentation = _preprocess_closing(slice_segmentation, gap_closing, pbar_update)
    else:
        pbar_init(1, "Merge segmentation")

    # Extract the overlap between slices.
    edges = track_utils.compute_edges_from_overlap(slice_segmentation, verbose=False)
    if len(edges) == 0:  # Nothing to merge.
        return slice_segmentation

    uv_ids = np.array([[edge["source"], edge["target"]] for edge in edges])
    overlaps = np.array([edge["score"] for edge in edges])

    n_nodes = int(slice_segmentation.max() + 1)
    graph = nifty.graph.undirectedGraph(n_nodes)
    graph.insertEdges(uv_ids)

    costs = seg_utils.multicut.compute_edge_costs(overlaps)
    # Set background weights to be maximally repulsive.
    if with_background:
        bg_edges = (uv_ids == 0).any(axis=1)
        costs[bg_edges] = -8.0

    node_labels = seg_utils.multicut.multicut_decomposition(graph, 1.0 - costs, beta=beta)

    segmentation = nifty.tools.take(node_labels, slice_segmentation)
    if min_z_extent is not None and min_z_extent > 0:
        segmentation = _filter_z_extent(segmentation, min_z_extent)

    pbar_update(1)
    pbar_close()

    return segmentation


def _segment_slices(
    data, predictor, segmentor, embedding_path, verbose, tile_shape, halo, batch_size=1, **kwargs
):
    assert data.ndim == 3

    image_embeddings = util.precompute_image_embeddings(
        predictor=predictor,
        input_=data,
        save_path=embedding_path,
        ndim=3,
        tile_shape=tile_shape,
        halo=halo,
        verbose=verbose,
        batch_size=batch_size,
    )

    offset = 0
    segmentation = np.zeros(data.shape, dtype="uint32")

    for i in tqdm(range(segmentation.shape[0]), desc="Segment slices", disable=not verbose):
        segmentor.initialize(data[i], image_embeddings=image_embeddings, verbose=False, i=i)
        seg = segmentor.generate(**kwargs)

        # Set offset for instance per slice.
        max_z = int(seg.max())
        if max_z == 0:
            continue
        seg[seg != 0] += offset
        offset = max_z + offset
        segmentation[i] = seg

    return segmentation, image_embeddings


def automatic_3d_segmentation(
    volume: np.ndarray,
    predictor: SamPredictor,
    segmentor: AMGBase,
    embedding_path: Optional[Union[str, os.PathLike]] = None,
    with_background: bool = True,
    gap_closing: Optional[int] = None,
    min_z_extent: Optional[int] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    verbose: bool = True,
    return_embeddings: bool = False,
    batch_size: int = 1,
    **kwargs,
) -> np.ndarray:
    """Automatically segment objects in a volume.

    First segments slices individually in 2d and then merges them across 3d
    based on overlap of objects between slices.

    Args:
        volume: The input volume.
        predictor: The Segment Anything predictor.
        segmentor: The instance segmentation class.
        embedding_path: The path to save pre-computed embeddings.
        with_background: Whether the segmentation has background. By default, set to 'True'.
        gap_closing: If given, gaps in the segmentation are closed with a binary closing
            operation. The value is used to determine the number of iterations for the closing.
        min_z_extent: Require a minimal extent in z for the segmented objects.
            This can help to prevent segmentation artifacts.
        tile_shape: Shape of the tiles for tiled prediction. By default prediction is run without tiling.
        halo: Overlap of the tiles for tiled prediction. By default prediction is run without tiling.
        verbose: Verbosity flag. By default, set to 'True'.
        return_embeddings: Whether to return the precomputed image embeddings. By default, set to 'False'.
        batch_size: The batch size to compute image embeddings over planes. By default, set to '1'.
        kwargs: Keyword arguments for the 'generate' method of the 'segmentor'.

    Returns:
        The segmentation.
    """
    segmentation, image_embeddings = _segment_slices(
        data=volume,
        predictor=predictor,
        segmentor=segmentor,
        embedding_path=embedding_path,
        verbose=verbose,
        tile_shape=tile_shape,
        halo=halo,
        batch_size=batch_size,
        **kwargs
    )
    segmentation = merge_instance_segmentation_3d(
        segmentation,
        beta=0.5,
        with_background=with_background,
        gap_closing=gap_closing,
        min_z_extent=min_z_extent,
        verbose=verbose,
    )
    if return_embeddings:
        return segmentation, image_embeddings
    else:
        return segmentation


def _filter_tracks(tracking_result, min_track_length):
    props = regionprops(tracking_result)
    discard_ids = []
    for prop in props:
        label_id = prop.label
        z_start, z_stop = prop.bbox[0], prop.bbox[3]
        if z_stop - z_start < min_track_length:
            discard_ids.append(label_id)
    tracking_result[np.isin(tracking_result, discard_ids)] = 0
    tracking_result, _, _ = relabel_sequential(tracking_result)
    return tracking_result


def _extract_tracks_and_lineages(segmentations, track_data, parent_graph):
    # The track data has the following layout: n_tracks x 4
    # With the following columns:
    # track_id - id of the track (= result from trackastra)
    # timepoint
    # y coordinate
    # x coordinate

    # Use the last three columns to index the segmentation and get the segmentation id.
    index = np.round(track_data[:, 1:], 0).astype("int32")
    index = tuple(index[:, i] for i in range(index.shape[1]))
    segmentation_ids = segmentations[index]

    # Find the mapping of nodes (= segmented objects) to track-ids.
    track_ids = track_data[:, 0].astype("int32")
    assert len(segmentation_ids) == len(track_ids)
    node_to_track = {k: v for k, v in zip(segmentation_ids, track_ids)}

    # Find the lineages as connected components in the parent graph.
    # First, we build a proper graph.
    lineage_graph = nx.Graph()
    for k, v in parent_graph.items():
        lineage_graph.add_edge(k, v)

    # Then, find the connected components, and compute the lineage representation expected by micro-sam from it:
    # E.g. if we have three lineages, the first consisting of three tracks and the second and third of one track each:
    # [
    #   {1: [2, 3]},  lineage with a dividing cell
    #   {4: []}, lineage with just one cell
    #   {5: []}, lineage with just one cell
    # ]

    # First, we fill the lineages which have one or more divisions, i.e. trees with more than one node.
    lineages = []
    for component in nx.connected_components(lineage_graph):
        root = next(iter(component))
        lineage_dict = {}

        def dfs(node, parent):
            # Avoid revisiting the parent node
            children = [n for n in lineage_graph[node] if n != parent]
            lineage_dict[node] = children
            for child in children:
                dfs(child, node)

        dfs(root, None)
        lineages.append(lineage_dict)

    # Then add single node lineages, which are not reflected in the original graph.
    all_tracks = set(track_ids.tolist())
    lineage_tracks = []
    for lineage in lineages:
        for k, v in lineage.items():
            lineage_tracks.append(k)
            lineage_tracks.extend(v)
    singleton_tracks = list(all_tracks - set(lineage_tracks))
    lineages.extend([{track: []} for track in singleton_tracks])

    # Make sure node_to_track contains everything.
    all_seg_ids = np.unique(segmentations)
    missing_seg_ids = np.setdiff1d(all_seg_ids, list(node_to_track.keys()))
    node_to_track.update({seg_id: 0 for seg_id in missing_seg_ids})
    return node_to_track, lineages


def _filter_lineages(lineages, tracking_result):
    track_ids = set(np.unique(tracking_result)) - {0}
    filtered_lineages = []
    for lineage in lineages:
        filtered_lineage = {k: v for k, v in lineage.items() if k in track_ids}
        if filtered_lineage:
            filtered_lineages.append(filtered_lineage)
    return filtered_lineages


def _tracking_impl(timeseries, segmentation, mode, min_time_extent, output_folder=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Trackastra.from_pretrained("general_2d", device=device)
    lineage_graph, _ = model.track(timeseries, segmentation, mode=mode)
    track_data, parent_graph, _ = graph_to_napari_tracks(lineage_graph)
    node_to_track, lineages = _extract_tracks_and_lineages(segmentation, track_data, parent_graph)
    tracking_result = recolor_segmentation(segmentation, node_to_track)

    if output_folder is not None:  # Store tracking results in CTC format.
        graph_to_ctc(lineage_graph, segmentation, outdir=output_folder)

    # TODO
    # We should check if trackastra supports this already.
    # Filter out short tracks / lineages.
    if min_time_extent is not None and min_time_extent > 0:
        raise NotImplementedError

    # Filter out pruned lineages.
    # Mmay either be missing due to track filtering or non-consectutive track numbering in trackastra.
    lineages = _filter_lineages(lineages, tracking_result)

    return tracking_result, lineages


def track_across_frames(
    timeseries: np.ndarray,
    segmentation: np.ndarray,
    gap_closing: Optional[int] = None,
    min_time_extent: Optional[int] = None,
    verbose: bool = True,
    pbar_init: Optional[callable] = None,
    pbar_update: Optional[callable] = None,
    output_folder: Optional[Union[os.PathLike, str]] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """Track segmented objects over time.

    This function uses Trackastra: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf
    for tracking. Please cite it if you use the automated tracking functionality.

    Args:
        timeseries: The input timeseries of images.
        segmentation: The segmentation. Expect segmentation results per frame
            that are relabeled so that segmentation ids don't overlap.
        gap_closing: If given, gaps in the segmentation are closed with a binary closing
            operation. The value is used to determine the number of iterations for the closing.
        min_time_extent: Require a minimal extent in time for the tracked objects.
        verbose: Verbosity flag. By default, set to 'True'.
        pbar_init: Function to initialize the progress bar.
        pbar_update: Function to update the progress bar.
        output_folder: The folder where the tracking results are stored in CTC format.

    Returns:
        The tracking result. Each object is colored by its track id.
        The lineages, which correspond to the cell divisions. Lineages are represented by a list of dicts,
            with each dict encoding a lineage, where keys correspond to parent track ids.
            Each key either maps to a list with two child track ids (cell division) or to an empty list (no division).
    """
    if Trackastra is None:
        raise RuntimeError(
            "The automatic tracking functionality requires trackastra. You can install it via 'pip install trackastra'."
        )

    _, pbar_init, pbar_update, pbar_close = util.handle_pbar(verbose, pbar_init=pbar_init, pbar_update=pbar_update)

    if gap_closing is not None and gap_closing > 0:
        segmentation = _preprocess_closing(segmentation, gap_closing, pbar_update)

    segmentation, lineage = _tracking_impl(
        timeseries=np.asarray(timeseries),
        segmentation=segmentation,
        mode="greedy",
        min_time_extent=min_time_extent,
        output_folder=output_folder,
    )
    return segmentation, lineage


def automatic_tracking_implementation(
    timeseries: np.ndarray,
    predictor: SamPredictor,
    segmentor: AMGBase,
    embedding_path: Optional[Union[str, os.PathLike]] = None,
    gap_closing: Optional[int] = None,
    min_time_extent: Optional[int] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    verbose: bool = True,
    return_embeddings: bool = False,
    batch_size: int = 1,
    output_folder: Optional[Union[os.PathLike, str]] = None,
    **kwargs,
) -> Tuple[np.ndarray, List[Dict]]:
    """Automatically track objects in a timesries based on per-frame automatic segmentation.

    This function uses Trackastra: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf
    for tracking. Please cite it if you use the automated tracking functionality.

    Args:
        timeseries: The input timeseries of images.
        predictor: The SAM model.
        segmentor: The instance segmentation class.
        embedding_path: The path to save pre-computed embeddings.
        gap_closing: If given, gaps in the segmentation are closed with a binary closing
            operation. The value is used to determine the number of iterations for the closing.
        min_time_extent: Require a minimal extent in time for the tracked objects.
        tile_shape: Shape of the tiles for tiled prediction. By default prediction is run without tiling.
        halo: Overlap of the tiles for tiled prediction. By default prediction is run without tiling.
        verbose: Verbosity flag. By default, set to 'True'.
        return_embeddings: Whether to return the precomputed image embeddings. By default, set to 'False'.
        batch_size: The batch size to compute image embeddings over planes. By default, set to '1'.
        output_folder: The folder where the tracking results are stored in CTC format.
        kwargs: Keyword arguments for the 'generate' method of the 'segmentor'.

    Returns:
        The tracking result. Each object is colored by its track id.
        The lineages, which correspond to the cell divisions. Lineages are represented by a list of dicts,
            with each dict encoding a lineage, where keys correspond to parent track ids.
            Each key either maps to a list with two child track ids (cell division) or to an empty list (no division).
    """
    if Trackastra is None:
        raise RuntimeError(
            "Automatic tracking requires trackastra. You can install it via 'pip install trackastra'."
        )

    segmentation, image_embeddings = _segment_slices(
        timeseries, predictor, segmentor, embedding_path, verbose,
        tile_shape=tile_shape, halo=halo, batch_size=batch_size,
        **kwargs,
    )

    segmentation, lineage = track_across_frames(
        timeseries=timeseries,
        segmentation=segmentation,
        gap_closing=gap_closing,
        min_time_extent=min_time_extent,
        verbose=verbose,
        output_folder=output_folder,
    )

    if return_embeddings:
        return segmentation, lineage, image_embeddings
    else:
        return segmentation, lineage


def get_napari_track_data(
    segmentation: np.ndarray, lineages: List[Dict], n_threads: Optional[int] = None
) -> Tuple[np.ndarray, Dict[int, List]]:
    """Derive the inputs for the napari tracking layer from a tracking result.

    Args:
        segmentation: The segmentation, after relabeling with track ids.
        lineages: The lineage information.
        n_threads: Number of threads for extracting the track data from the segmentation.

    Returns:
        The array with the track data expected by napari.
        The parent dictionary for napari.
    """
    if n_threads is None:
        n_threads = mp.cpu_count()

    def compute_props(t):
        props = regionprops(segmentation[t])
        # Create the track data representation for napari, which expects:
        # track_id, timepoint, y, x
        track_data = np.array([[prop.label, t] + list(prop.centroid) for prop in props])
        return track_data

    with futures.ThreadPoolExecutor(n_threads) as tp:
        track_data = list(tp.map(compute_props, range(segmentation.shape[0])))
    track_data = [data for data in track_data if data.size > 0]
    track_data = np.concatenate(track_data)

    # The graph representation of napari uses the children as keys and the parents as values,
    # whereas our representation uses parents as keys and children as values.
    # Hence, we need to translate the representation.
    parent_graph = {
        child: [parent] for lineage in lineages for parent, children in lineage.items() for child in children
    }

    return track_data, parent_graph
