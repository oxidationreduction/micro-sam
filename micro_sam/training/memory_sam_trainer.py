import torch
import torch.nn as nn
from torch import overrides
from torch_em.trainer.wandb_logger import WandbLogger
from tqdm import tqdm
import time
import os
from typing import Union, Optional

from micro_sam.training.sam_trainer import SamTrainer

class MemorySamTrainer(SamTrainer):
    def __init__(
            self,
            memory_adapter: nn.Module,
            seq_len: int = 3,
            teacher_forcing_prob: float = 0.5,
            **kwargs
    ):
        # 将基础参数透传给父类 SamTrainer
        super().__init__(**kwargs)
        self.memory_adapter = memory_adapter
        self.seq_len = seq_len
        self.teacher_forcing_prob = teacher_forcing_prob

    def _forward_step(self, x_t, y_t, t, memory_state, sampled_ids, iteration, is_val=False):
        """处理序列中单帧 (t) 的正向传播与记忆更新"""
        batch_size = x_t.shape[0]

        # ==========================================================
        # 1. 图像预处理与特征提取 (修复 FP16 vs FP32 的核心地带)
        # 使用 SAM 原生的 preprocess 方法，它会自动将输入的 FP32 (float) 张量
        # 转换为与冻结模型参数一致的 FP16 (Half) 张量，并应用归一化！
        # ==========================================================
        x_t = x_t.to(self.device)

        input_images = torch.stack([self.model.sam.preprocess(img) for img in x_t], dim=0)

        target_dtype = next(self.model.sam.image_encoder.parameters()).dtype
        input_images = input_images.to(dtype=target_dtype)

        if not any(p.requires_grad for p in self.model.sam.image_encoder.parameters()):
            with torch.no_grad():
                image_embeddings = self.model.sam.image_encoder(input_images)
        else:
            image_embeddings = self.model.sam.image_encoder(input_images)

        # ==========================================================
        # 2. SAM 提示与解码
        # ==========================================================
        batched_outputs = []
        if t == 0:
            # === [第 0 帧] 传统 Prompt 交互逻辑 ===
            if is_val:
                n_pos, n_neg, get_boxes, multimask_output = self._get_prompt_and_multimasking_choices_for_val(iteration)
            else:
                n_pos, n_neg, get_boxes, multimask_output = self._get_prompt_and_multimasking_choices(iteration)

            # 解析第一帧的 prompt，并获取抽样到的细胞 ID
            batched_inputs, sampled_ids = self.convert_inputs(x_t, y_t, n_pos, n_neg, get_boxes,
                                                              self.n_objects_per_batch)
            _, y_one_hot = self._preprocess_batch(batched_inputs, y_t, sampled_ids)

            for batched_input in batched_inputs:
                for k in batched_input:
                    batched_input[k] = batched_input[k].to(self.device) if isinstance(batched_input[k], torch.Tensor) else batched_input[k]
            y_one_hot = y_one_hot.to(self.device)


            # 【重要修复】：由于我们已经提前提取了 image_embeddings，不能再使用 self.model(...)
            # 必须像 SAM 内部那样，手动跑完解码流水线以复用特征
            for i, (image_record, curr_embedding) in enumerate(zip(batched_inputs, image_embeddings)):
                points = (image_record["point_coords"],
                          image_record["point_labels"]) if "point_coords" in image_record else None

                sparse_embeddings, dense_embeddings = self.model.sam.prompt_encoder(
                    points=points,
                    boxes=image_record.get("boxes", None),
                    masks=image_record.get("mask_inputs", None),
                )
                n_i = dense_embeddings.shape[0]
                curr_img_embed_exp = curr_embedding.unsqueeze(0).expand(n_i, -1, -1, -1)

                low_res_masks, iou_predictions = self.model.sam.mask_decoder(
                    image_embeddings=curr_img_embed_exp,
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
                for j in range(n_i):
                    batched_outputs.append({
                        "masks": masks[j:j+1],
                        "iou_predictions": iou_predictions[j:j+1],
                        "low_res_masks": low_res_masks[j:j+1]
                    })

        else:
            # === [后续帧] 时序追踪逻辑 ===
            dummy_inputs = [{} for _ in range(batch_size)]
            _, y_one_hot = self._preprocess_batch(dummy_inputs, y_t, sampled_ids)

            num_objs = [len(ids) for ids in sampled_ids]
            expanded_curr_embeddings = []
            for i, n in enumerate(num_objs):
                expanded_curr_embeddings.append(image_embeddings[i].unsqueeze(0).expand(n, -1, -1, -1))
            expanded_curr_embeddings = torch.cat(expanded_curr_embeddings, dim=0)

            # 利用记忆适配器生成 Prompt
            sparse_embeddings, dense_embeddings = self.memory_adapter.get_prompts(
                image_embeddings=expanded_curr_embeddings,
                memory_state=memory_state
            )

            dense_splits = torch.split(dense_embeddings, num_objs, dim=0)
            sparse_splits = torch.split(sparse_embeddings, num_objs, dim=0)

            for i in range(batch_size):
                n_i = num_objs[i]
                if n_i == 0:
                    continue

                curr_dense = dense_splits[i]
                curr_sparse = sparse_splits[i]
                # 同样地，把图片特征复制 n_i 份以匹配对象的数量
                curr_img_embed_exp = image_embeddings[i].unsqueeze(0).expand(n_i, -1, -1, -1)

                low_res_masks, iou_predictions = self.model.sam.mask_decoder(
                    image_embeddings=curr_img_embed_exp,
                    image_pe=self.model.sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=curr_sparse,
                    dense_prompt_embeddings=curr_dense,
                    multimask_output=False,
                )

                masks = self.model.sam.postprocess_masks(
                    low_res_masks,
                    input_size=x_t.shape[-2:],
                    original_size=y_t.shape[-2:]
                )

                for j in range(n_i):
                    batched_outputs.append({
                        "masks": masks[j:j+1],
                        "iou_predictions": iou_predictions[j:j+1],
                        "low_res_masks": low_res_masks[j:j+1]
                    })

        # ==========================================================
        # 3. 计算 Loss 与 更新记忆
        # ==========================================================
        loss, mask_loss, iou_regression_loss = self._compute_loss(batched_outputs, y_one_hot)

        with torch.no_grad():
            batched_iou_predictions = torch.stack([m["iou_predictions"] for m in batched_outputs])
            model_iou = torch.mean(batched_iou_predictions)

        # 记忆更新
        if not is_val and torch.rand(1).item() < self.teacher_forcing_prob:
            mask_for_memory = y_one_hot.squeeze(1)  # (N_total, 1, H, W)
        else:
            # 【核心修复】：处理 SAM 输出多层 Mask (multimask_output=True) 的情况
            best_masks = []
            for m in batched_outputs:
                if m["masks"].shape[-3] > 1:
                    # 如果有多个 Mask 通道（比如 3 个），选出预测 IoU 最高的那一层的索引
                    best_idx = torch.argmax(m["iou_predictions"])
                    # 提取该层，并通过切片保持 (1, H, W) 的维度
                    best_mask = m["masks"][..., best_idx: best_idx + 1, :, :]
                else:
                    best_mask = m["masks"]
                best_masks.append(best_mask)

            # 堆叠后一定能保证是 (N_total, 1, H, W)
            masks_tensor = torch.concat(best_masks, dim=0)
            mask_for_memory = torch.sigmoid(masks_tensor)

        # 【同步展开】: 把记忆更新时所绑定的图像特征，也同步扁平化成 N_total
        num_objs = [len(ids) for ids in sampled_ids]
        expanded_mem_embeddings = []
        for i, n in enumerate(num_objs):
            expanded_mem_embeddings.append(image_embeddings[i].unsqueeze(0).expand(n, -1, -1, -1))
        expanded_mem_embeddings = torch.cat(expanded_mem_embeddings, dim=0)

        memory_state = self.memory_adapter.update_memory(
            memory_state=memory_state,
            current_image_embeddings=expanded_mem_embeddings,
            current_mask=mask_for_memory
        )

        return loss, mask_loss, iou_regression_loss, model_iou, y_one_hot, sampled_ids, memory_state
    # ====================================================
    # 彻底覆盖父类的单步执行接口，对接 torch_em 底层
    # ====================================================

    def _interactive_train_iteration(self, x, y):
        """训练模式下的序列迭代"""
        batch_size = x.shape[0]
        total_loss, total_mask_loss, total_iou_loss, total_model_iou = 0.0, 0.0, 0.0, 0.0

        memory_state = self.memory_adapter.init_memory(batch_size, device=self.device)
        sampled_ids = None
        y_one_hot_t0 = None

        # Z 轴 / 时间轴 循环
        for t in range(self.seq_len):
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
                y_one_hot_t0 = y_one_hot  # 用于可视化日志记录第一帧

        # 将平均指标返回给父类的外层记录器
        return (
            total_loss / self.seq_len,
            total_mask_loss / self.seq_len,
            total_iou_loss / self.seq_len,
            total_model_iou / self.seq_len,
            y_one_hot_t0
        )

    def _interactive_val_iteration(self, x, y, val_iteration):
        """验证模式下的序列迭代 (无 Teacher Forcing)"""
        batch_size = x.shape[0]
        total_loss, total_mask_loss, total_iou_loss, total_model_iou = 0.0, 0.0, 0.0, 0.0

        memory_state = self.memory_adapter.init_memory(batch_size, device=self.device)
        sampled_ids = None
        y_one_hot_t0 = None

        for t in range(self.seq_len):
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

        # 验证集需要多返回一个 metric (原代码使用 mask_loss 即 Dice Loss)
        metric = total_mask_loss / self.seq_len

        return (
            total_loss / self.seq_len,
            total_mask_loss / self.seq_len,
            total_iou_loss / self.seq_len,
            total_model_iou / self.seq_len,
            y_one_hot_t0,
            metric
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
        """Run neural network training.

        Exactly one of 'iterations' or 'epochs' has to be passed.

        Args:
            iterations: How long to train, specified in iterations.
            load_from_checkpoint: Path to a checkpoint from where training should be continued .
            epochs: How long to train, specified in epochs.
            save_every_kth_epoch: Save checkpoints after every kth epoch in a separate file.
                The corresponding checkpoints will be saved with the naming scheme 'epoch-{epoch}.pt'.
            progress: Optional progress bar for integration with external tools. Expected to follow the tqdm interface.
            overwrite_training: Whether to overwrite existing checkpoints in the save directory.
        """
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

            # Ensure data is shuffled differently at each epoch.
            try:
                self.train_loader.sampler.set_epoch(epoch)
            except AttributeError:
                pass

            # Run training and validation for this epoch
            t_per_iter = train_epoch(progress)
            current_metric = validate()

            # perform all the post-epoch steps:

            # apply the learning rate scheduler
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(current_metric)

            # how long did we train in total?
            total_train_time = (time.time() - t_start) + self.train_time

            # save this checkpoint as the new best checkpoint if
            # it has the best overall validation metric
            if current_metric < best_metric:
                best_metric = current_metric
                self._best_epoch = self._epoch
                self.save_checkpoint("best", current_metric, best_metric, train_time=total_train_time)
                torch.save(self.memory_adapter, os.path.join(self.checkpoint_folder, "best_mem_adapter.pt"))

            # save this checkpoint as the latest checkpoint
            self.save_checkpoint("latest", current_metric, best_metric, train_time=total_train_time)
            torch.save(self.memory_adapter, os.path.join(self.checkpoint_folder, "latest_mem_adapter.pt"))

            # if we save after every k-th epoch then check if we need to save now
            if save_every_kth_epoch is not None and (self._epoch + 1) % save_every_kth_epoch == 0:
                self.save_checkpoint(
                    f"epoch-{self._epoch + 1}", current_metric, best_metric, train_time=total_train_time
                )

            # if early stopping has been specified then check if the stopping condition is met
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

        # Update the train time
        self.train_time = total_train_time

        # TODO save the model to wandb if we have the wandb logger
        if isinstance(self.logger, WandbLogger):
            self.logger.get_wandb().finish()
