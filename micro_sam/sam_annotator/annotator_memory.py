import sys
from typing import Optional, Tuple, Union

import napari
import numpy as np
import torch
from magicgui import magicgui

from micro_sam import util
from micro_sam.sam_annotator import _widgets as widgets
from micro_sam.sam_annotator._annotator import _AnnotatorBase
from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam.sam_annotator.z_memory_adapter import (
    ZMemoryAdapter,
    extract_memory_adapter_state_dict,
    get_module_device,
    load_checkpoint_state_dict,
    mask_to_memory_tensor,
    predict_from_memory_state,
    prepare_sam_state_dict,
    remove_memory_adapter_keys_from_state_dict,
)


class AnnotatorMemory(_AnnotatorBase):
    def __init__(self, viewer: "napari.viewer.Viewer", reset_state: bool = True) -> None:
        super().__init__(viewer=viewer, ndim=3)

        state = AnnotatorState()
        if reset_state:
            state.reset_state()

        state.annotator = self
        print("Memory Annotator")

    def _get_widgets(self):
        autosegment = widgets.AutoSegmentWidget(
            self._viewer, with_decoder=AnnotatorState().decoder is not None, volumetric=False
        )

        @magicgui(call_button="Track Forward Across Z")
        def track_forward_widget():
            self._run_memory_tracking()

        return {
            "segment": widgets.segment(),
            "autosegment": autosegment,
            "commit": widgets.commit(),
            "clear": widgets.clear(),
            "track_forward": track_forward_widget,
        }

    def _check_stop_condition(self, z_index: int) -> bool:
        points = self.point_layer.data
        if len(points) == 0:
            return False

        labels = self.point_layer.features.get("label", np.ones(len(points)))
        points_z = np.round(points[:, 0]).astype(int)
        mask_z = points_z == z_index
        if not np.any(mask_z):
            return False

        labels_z = labels[mask_z]
        return bool(np.all(labels_z == 0))

    def _run_memory_tracking(self):
        state = AnnotatorState()
        predictor = state.predictor
        if predictor is None:
            napari.utils.notifications.show_warning("Please compute image embeddings first.")
            return

        sam_model = predictor.model
        if not hasattr(sam_model, "memory_adapter"):
            napari.utils.notifications.show_error("Model does not have a Memory Adapter.")
            return

        memory_adapter = sam_model.memory_adapter
        model_device = get_module_device(sam_model)

        current_z = int(self._viewer.dims.current_step[0])
        total_z = self.image.shape[0]

        curr_mask_np = self.segmentation_layer.data[current_z]
        if not np.any(curr_mask_np):
            napari.utils.notifications.show_warning(
                f"No segmentation found on Z={current_z}. Please segment an object first."
            )
            return

        print(f"Start tracking from Z={current_z}...")

        memory_state = memory_adapter.init_memory(batch_size=1, device=model_device)
        predictor.set_image(self.image[current_z])
        curr_embed = predictor.features
        curr_mask_tensor = mask_to_memory_tensor(curr_mask_np > 0, model_device)
        memory_state = memory_adapter.update_memory(
            memory_state=memory_state,
            current_image_embeddings=curr_embed,
            current_mask=curr_mask_tensor,
        )

        for z in range(current_z + 1, total_z):
            if self._check_stop_condition(z):
                print(f"Tracking gracefully stopped at Z={z} due to User Negative Prompt.")
                napari.utils.notifications.show_info(f"Tracking stopped at Z={z} (negative prompt reached).")
                break

            curr_mask_np, curr_embed, _, _ = predict_from_memory_state(
                predictor=predictor,
                memory_state=memory_state,
                image=self.image[z],
                mask_threshold=0.5,
            )

            if np.sum(curr_mask_np) < 128:
                print(f"Tracking automatically stopped at Z={z}: object disappeared.")
                napari.utils.notifications.show_info(f"Tracking stopped at Z={z}: object disappeared.")
                break

            self.segmentation_layer.data[z] = curr_mask_np
            self.segmentation_layer.refresh()
            self._viewer.dims.current_step = (z, *self._viewer.dims.current_step[1:])

            memory_state = memory_adapter.update_memory(
                memory_state=memory_state,
                current_image_embeddings=curr_embed,
                current_mask=mask_to_memory_tensor(curr_mask_np, model_device),
            )

        print("Tracking finished successfully.")


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
    state = AnnotatorState()
    state.image_shape = image.shape[:-1] if image.ndim == 4 else image.shape
    print("State initialized")

    sys.argv = [sys.argv[0]]
    if viewer is None:
        viewer = napari.Viewer()
    viewer.add_image(image, name="image")

    annotator = AnnotatorMemory(viewer=viewer)
    state.initialize_predictor(
        image,
        model_type=model_type,
        save_path=embedding_path,
        ndim=3,
        device=util.get_device(),
        tile_shape=tile_shape,
        halo=halo,
    )

    sam_model = state.predictor.model
    model_device = get_module_device(sam_model)
    sam_model.memory_adapter = ZMemoryAdapter(embed_dim=256).to(model_device)

    if checkpoint_path is not None:
        print(f"Loading custom fine-tuned weights from: {checkpoint_path}")
        raw_state = load_checkpoint_state_dict(checkpoint_path, map_location=model_device)
        normalized_state = prepare_sam_state_dict(raw_state)
        sam_state = remove_memory_adapter_keys_from_state_dict(normalized_state)

        missing_keys, unexpected_keys = sam_model.load_state_dict(sam_state, strict=False)
        adapter_state = extract_memory_adapter_state_dict(normalized_state)
        if adapter_state:
            adapter_missing, adapter_unexpected = sam_model.memory_adapter.load_state_dict(
                adapter_state,
                strict=False,
            )
            print(
                "Memory Adapter weights loaded successfully. "
                f"missing={len(adapter_missing)}, unexpected={len(adapter_unexpected)}"
            )

        filtered_missing = [key for key in missing_keys if not key.startswith("memory_adapter.")]
        print(
            "Custom checkpoint loading complete. "
            f"sam_missing={len(filtered_missing)}, sam_unexpected={len(unexpected_keys)}"
        )

    annotator._update_image(segmentation_result=segmentation_result)
    viewer.window.add_dock_widget(annotator)

    if return_viewer:
        return viewer

    napari.run()


def main():
    input_path = "/mnt/mira_datacenter/liuhongyu2024/ninanjie/3_cell/raw_0/"
    model_type = "vit_l"
    checkpoint = "/home/mira/Downloads/micro-sam/data/models/checkpoints/vit_l_lm/lm_generalist_memory_sam/best.pt"
    image = util.load_image_data(input_path, key=None)

    annotator_memory_3d(
        image,
        embedding_path=None,
        segmentation_result=None,
        model_type=model_type,
        tile_shape=None,
        halo=None,
        checkpoint_path=checkpoint,
    )


if __name__ == "__main__":
    main()
