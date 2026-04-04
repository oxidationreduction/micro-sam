import os
import time
from typing import Optional, Union

import torch
import torch.nn as nn
from torch_em.trainer.wandb_logger import WandbLogger
from tqdm import tqdm

from micro_sam.training.sam_trainer import SamTrainer


class MemorySamTrainer(SamTrainer):
    def __init__(
        self,
        memory_adapter: nn.Module,
        seq_len: int = 3,
        teacher_forcing_prob: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.memory_adapter = memory_adapter
        self.seq_len = seq_len
        self.teacher_forcing_prob = teacher_forcing_prob

    @staticmethod
    def _extract_instance_channel(y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 4:
            return y[:, :1]
        if y.ndim == 3:
            return y.unsqueeze(1)
        raise ValueError(f"Unsupported memory label shape: {tuple(y.shape)}")

    def _preprocess_memory_batch(self, batched_inputs, y, sampled_ids):
        if sampled_ids is None:
            raise ValueError("sampled_ids must be initialized before preprocessing memory targets.")

        instance_targets = self._extract_instance_channel(y)
        batched_inputs, y_one_hot = super()._preprocess_batch(batched_inputs, instance_targets, sampled_ids)
        n_objects = y_one_hot.shape[1]
        sampled_ids = [ids[:n_objects] for ids in sampled_ids]
        return batched_inputs, y_one_hot, sampled_ids

    @staticmethod
    def _flatten_per_object(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 5:
            raise ValueError(f"Expected a 5D tensor, got shape {tuple(tensor.shape)}")
        return tensor.reshape(-1, *tensor.shape[-3:])

    @staticmethod
    def _expand_embeddings_by_object(image_embeddings: torch.Tensor, sampled_ids):
        counts = [len(ids) for ids in sampled_ids]
        if not counts or min(counts) == 0:
            raise RuntimeError("Encountered a training batch without trackable objects.")

        expanded = [
            image_embeddings[i].unsqueeze(0).expand(count, -1, -1, -1)
            for i, count in enumerate(counts)
        ]
        return torch.cat(expanded, dim=0), counts

    def _collect_memory_masks(self, batched_outputs, batched_iou_predictions, y_one_hot, is_val):
        if not is_val and torch.rand(1).item() < self.teacher_forcing_prob:
            return self._flatten_per_object(y_one_hot)

        if batched_outputs[0]["masks"].shape[1] > 1:
            best_masks, _ = self._get_best_masks(batched_outputs, batched_iou_predictions)
            return self._flatten_per_object(best_masks)

        predicted_masks = torch.stack(
            [torch.sigmoid(batch_output["masks"]) for batch_output in batched_outputs],
            dim=0,
        )
        return self._flatten_per_object(predicted_masks)

    def _forward_step(self, x_t, y_t, t, memory_state, sampled_ids, iteration, is_val=False):
        batch_size = x_t.shape[0]
        x_t = x_t.to(self.device)

        input_images = torch.stack([self.model.sam.preprocess(img) for img in x_t], dim=0)
        target_dtype = next(self.model.sam.image_encoder.parameters()).dtype
        input_images = input_images.to(dtype=target_dtype)

        if not any(p.requires_grad for p in self.model.sam.image_encoder.parameters()):
            with torch.no_grad():
                image_embeddings = self.model.sam.image_encoder(input_images)
        else:
            image_embeddings = self.model.sam.image_encoder(input_images)

        batched_outputs = []

        if t == 0:
            if is_val:
                n_pos, n_neg, get_boxes, multimask_output = self._get_prompt_and_multimasking_choices_for_val(iteration)
            else:
                n_pos, n_neg, get_boxes, multimask_output = self._get_prompt_and_multimasking_choices(iteration)

            batched_inputs, sampled_ids = self.convert_inputs(
                x_t, y_t, n_pos, n_neg, get_boxes, self.n_objects_per_batch,
            )
            batched_inputs, y_one_hot, sampled_ids = self._preprocess_memory_batch(
                batched_inputs, y_t, sampled_ids,
            )

            for batched_input in batched_inputs:
                for key, value in batched_input.items():
                    if isinstance(value, torch.Tensor):
                        batched_input[key] = value.to(self.device)
            y_one_hot = y_one_hot.to(self.device)

            for image_record, curr_embedding in zip(batched_inputs, image_embeddings):
                points = None
                if "point_coords" in image_record:
                    points = (image_record["point_coords"], image_record["point_labels"])

                sparse_embeddings, dense_embeddings = self.model.sam.prompt_encoder(
                    points=points,
                    boxes=image_record.get("boxes", None),
                    masks=image_record.get("mask_inputs", None),
                )
                n_objects = dense_embeddings.shape[0]
                curr_img_embed = curr_embedding.unsqueeze(0).expand(n_objects, -1, -1, -1)

                low_res_masks, iou_predictions = self.model.sam.mask_decoder(
                    image_embeddings=curr_img_embed,
                    image_pe=self.model.sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )

                masks = self.model.sam.postprocess_masks(
                    low_res_masks,
                    input_size=image_record["image"].shape[-2:],
                    original_size=image_record["original_size"],
                )
                batched_outputs.append(
                    {
                        "masks": masks,
                        "iou_predictions": iou_predictions,
                        "low_res_masks": low_res_masks,
                    }
                )

        else:
            dummy_inputs = [{} for _ in range(batch_size)]
            _, y_one_hot, sampled_ids = self._preprocess_memory_batch(dummy_inputs, y_t, sampled_ids)

            expanded_curr_embeddings, num_objs = self._expand_embeddings_by_object(image_embeddings, sampled_ids)
            sparse_embeddings, dense_embeddings = self.memory_adapter.get_prompts(
                image_embeddings=expanded_curr_embeddings,
                memory_state=memory_state,
            )

            dense_splits = torch.split(dense_embeddings, num_objs, dim=0)
            sparse_splits = torch.split(sparse_embeddings, num_objs, dim=0)

            for i in range(batch_size):
                curr_dense = dense_splits[i]
                curr_sparse = sparse_splits[i]
                curr_img_embed = image_embeddings[i].unsqueeze(0).expand(num_objs[i], -1, -1, -1)

                low_res_masks, iou_predictions = self.model.sam.mask_decoder(
                    image_embeddings=curr_img_embed,
                    image_pe=self.model.sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=curr_sparse,
                    dense_prompt_embeddings=curr_dense,
                    multimask_output=False,
                )

                masks = self.model.sam.postprocess_masks(
                    low_res_masks,
                    input_size=x_t.shape[-2:],
                    original_size=y_t.shape[-2:],
                )
                batched_outputs.append(
                    {
                        "masks": masks,
                        "iou_predictions": iou_predictions,
                        "low_res_masks": low_res_masks,
                    }
                )

        if y_one_hot.shape[1] == 0:
            raise RuntimeError("Encountered a memory-training batch with zero sampled objects.")

        loss, mask_loss, iou_regression_loss = self._compute_loss(batched_outputs, y_one_hot)

        with torch.no_grad():
            batched_iou_predictions = torch.stack(
                [batch_output["iou_predictions"] for batch_output in batched_outputs],
                dim=0,
            )
            model_iou = batched_iou_predictions.mean()

        mask_for_memory = self._collect_memory_masks(
            batched_outputs=batched_outputs,
            batched_iou_predictions=batched_iou_predictions,
            y_one_hot=y_one_hot,
            is_val=is_val,
        )
        expanded_mem_embeddings, _ = self._expand_embeddings_by_object(image_embeddings, sampled_ids)
        memory_state = self.memory_adapter.update_memory(
            memory_state=memory_state,
            current_image_embeddings=expanded_mem_embeddings,
            current_mask=mask_for_memory,
        )

        return loss, mask_loss, iou_regression_loss, model_iou, y_one_hot, sampled_ids, memory_state

    def _interactive_train_iteration(self, x, y):
        total_loss, total_mask_loss, total_iou_loss, total_model_iou = 0.0, 0.0, 0.0, 0.0
        seq_len = min(self.seq_len, x.shape[1])

        memory_state = self.memory_adapter.init_memory(x.shape[0], device=self.device)
        sampled_ids = None
        y_one_hot_t0 = None

        for t in range(seq_len):
            x_t = x[:, t]
            y_t = y[:, t]

            loss, mask_loss, iou_regression_loss, model_iou, y_one_hot, sampled_ids, memory_state = self._forward_step(
                x_t, y_t, t, memory_state, sampled_ids, self._iteration, is_val=False
            )

            total_loss += loss
            total_mask_loss += mask_loss
            total_iou_loss += iou_regression_loss
            total_model_iou += model_iou

            if t == 0:
                y_one_hot_t0 = y_one_hot

        return (
            total_loss / seq_len,
            total_mask_loss / seq_len,
            total_iou_loss / seq_len,
            total_model_iou / seq_len,
            y_one_hot_t0,
        )

    def _interactive_val_iteration(self, x, y, val_iteration):
        total_loss, total_mask_loss, total_iou_loss, total_model_iou = 0.0, 0.0, 0.0, 0.0
        seq_len = min(self.seq_len, x.shape[1])

        memory_state = self.memory_adapter.init_memory(x.shape[0], device=self.device)
        sampled_ids = None
        y_one_hot_t0 = None

        for t in range(seq_len):
            x_t = x[:, t]
            y_t = y[:, t]

            loss, mask_loss, iou_regression_loss, model_iou, y_one_hot, sampled_ids, memory_state = self._forward_step(
                x_t, y_t, t, memory_state, sampled_ids, val_iteration, is_val=True
            )

            total_loss += loss
            total_mask_loss += mask_loss
            total_iou_loss += iou_regression_loss
            total_model_iou += model_iou

            if t == 0:
                y_one_hot_t0 = y_one_hot

        metric = total_mask_loss / seq_len
        return (
            total_loss / seq_len,
            total_mask_loss / seq_len,
            total_iou_loss / seq_len,
            total_model_iou / seq_len,
            y_one_hot_t0,
            metric,
        )

    def fit(
        self,
        iterations: Optional[int] = None,
        load_from_checkpoint: Optional[Union[os.PathLike, str]] = None,
        epochs: Optional[int] = None,
        save_every_kth_epoch: Optional[int] = None,
        progress=None,
        overwrite_training: bool = True,
    ):
        print("######################")
        print("# Training MemorySam #")
        print("######################")
        best_metric = self._initialize(iterations, load_from_checkpoint, epochs)

        if not overwrite_training:
            if load_from_checkpoint is not None:
                raise ValueError(
                    "We do not support 'overwrite_training=False' and 'load_from_checkpoint' at the same time."
                )

            if self._verify_if_training_completed():
                print(
                    f"The model is trained for {self.max_iteration} iterations / {self.max_epoch} epochs "
                    "and 'overwrite_training' is set to 'False'."
                )
                print(f"The checkpoints are located at '{os.path.abspath(self.checkpoint_folder)}'.")
                return

        print(
            "Start fitting for",
            self.max_iteration - self._iteration,
            "iterations / ",
            self.max_epoch - self._epoch,
            "epochs",
        )
        print("with", len(self.train_loader), "iterations per epoch")

        if self.mixed_precision:
            train_epoch = self._train_epoch_mixed
            validate = self._validate_mixed
            print("Training with mixed precision")
        else:
            train_epoch = self._train_epoch
            validate = self._validate
            print("Training with single precision")

        total_iterations = epochs * len(self.train_loader) if iterations is None else iterations
        if progress is None:
            progress = tqdm(total=total_iterations, desc=f"Epoch {self._epoch}", leave=True)
        else:
            progress.total = total_iterations
            progress.set_description(f"Epoch {self._epoch}")

        msg = "Epoch %i: average [s/it]: %f, current metric: %f, best metric: %f"
        train_epochs = self.max_epoch - self._epoch
        t_start = time.time()
        for epoch in range(train_epochs):
            try:
                self.train_loader.sampler.set_epoch(epoch)
            except AttributeError:
                pass

            t_per_iter = train_epoch(progress)
            current_metric = validate()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(current_metric)

            total_train_time = (time.time() - t_start) + self.train_time

            if current_metric < best_metric:
                best_metric = current_metric
                self._best_epoch = self._epoch
                self.save_checkpoint("best", current_metric, best_metric, train_time=total_train_time)
                torch.save(self.memory_adapter, os.path.join(self.checkpoint_folder, "best_mem_adapter.pt"))

            self.save_checkpoint("latest", current_metric, best_metric, train_time=total_train_time)
            torch.save(self.memory_adapter, os.path.join(self.checkpoint_folder, "latest_mem_adapter.pt"))

            if save_every_kth_epoch is not None and (self._epoch + 1) % save_every_kth_epoch == 0:
                self.save_checkpoint(
                    f"epoch-{self._epoch + 1}", current_metric, best_metric, train_time=total_train_time
                )

            if self.early_stopping is not None:
                epochs_since_best = self._epoch - self._best_epoch
                if epochs_since_best > self.early_stopping:
                    print("Stopping training because there has been no improvement for", self.early_stopping, "epochs")
                    break

            self._epoch += 1
            progress.set_description(msg % (self._epoch, t_per_iter, current_metric, best_metric), refresh=True)

        print(f"Finished training after {self._epoch} epochs / {self._iteration} iterations.")
        print(f"The best epoch is number {self._best_epoch}.")

        if self._generate_name:
            self.name = None

        self.train_time = total_train_time

        if isinstance(self.logger, WandbLogger):
            self.logger.get_wandb().finish()
