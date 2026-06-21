# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Callbacks for saving augmented dataset and prediction preview grids."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import Callback, LightningModule, Trainer


class DatasetGridCallback(Callback):
    """Save train/validation dataset grids once at the beginning of training."""

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save dataset grids at the beginning of the first training epoch.

        Args:
            trainer: Active PyTorch Lightning trainer.
            pl_module: Lightning module being trained.
        """
        datamodule: Any | None = getattr(trainer, "datamodule", None)
        save_grids = getattr(datamodule, "_maybe_save_dataset_grids", None)
        if callable(save_grids):
            save_grids(epoch=int(trainer.current_epoch))


class PredictionGridCallback(Callback):
    """Save validation prediction grids with ground-truth and predicted boxes."""

    def __init__(self, output_dir: str, max_batches: int = 3, max_images: int = 9) -> None:
        """Initialize the callback.

        Args:
            output_dir: Training output directory.
            max_batches: Maximum number of validation batches to save per epoch.
            max_images: Maximum number of images to draw from each validation batch.
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.max_batches = max_batches
        self.max_images = max_images

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Save prediction preview grids from the first few validation batches."""
        if not trainer.is_global_zero or batch_idx >= self.max_batches or dataloader_idx != 0:
            return
        epoch = int(trainer.current_epoch)
        if not isinstance(outputs, dict) or "results" not in outputs or "targets" not in outputs:
            return
        try:
            self._save_prediction_grid(batch, outputs, epoch=epoch, batch_idx=batch_idx)
        except Exception:
            from rfdetr.utilities.logger import get_logger

            get_logger().warning(
                "Failed to save prediction grid; training will continue without it.",
                exc_info=True,
            )

    def _save_prediction_grid(self, batch: Any, outputs: dict[str, Any], *, epoch: int, batch_idx: int) -> None:
        """Render and save a grid from validation images, targets, and predictions."""
        import matplotlib.pyplot as plt
        import numpy as np
        import supervision as sv
        import torchvision.transforms as T  # noqa: N812

        from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy

        samples, _ = batch
        images = samples.tensors
        results = outputs["results"]
        targets = outputs["targets"]
        count = min(self.max_images, images.shape[0], len(results), len(targets))
        if count <= 0:
            return

        inv_normalize = T.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_scale=0.4, text_padding=2)

        columns = 3
        rows = max(1, (count + columns - 1) // columns)
        fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 5 * rows))
        fig.suptitle(f"validation predictions, epoch {epoch}, batch {batch_idx}")
        axes_array = np.asarray(axes, dtype=object).reshape(-1)
        for axis in axes_array:
            axis.axis("off")

        for axis, index in zip(axes_array, range(count), strict=False):
            target = targets[index]
            result = results[index]
            image_tensor = images[index]
            size = target.get("size")
            height, width = (
                (int(size[0]), int(size[1]))
                if isinstance(size, torch.Tensor)
                else (int(image_tensor.shape[-2]), int(image_tensor.shape[-1]))
            )
            image = inv_normalize(image_tensor)[:, :height, :width].detach().cpu().numpy()
            scene = np.ascontiguousarray((np.clip(image.transpose(1, 2, 0), 0.0, 1.0) * 255).astype(np.uint8))

            gt_boxes = target.get("boxes", torch.zeros((0, 4), dtype=torch.float32))
            if isinstance(gt_boxes, torch.Tensor) and gt_boxes.numel() > 0:
                gt_xyxy = box_cxcywh_to_xyxy(gt_boxes.detach().cpu()) * torch.tensor(
                    [width, height, width, height],
                    dtype=torch.float32,
                )
                class_ids = target["labels"].detach().cpu().numpy().astype(int)
                detections = sv.Detections(xyxy=gt_xyxy.numpy().astype(np.float32), class_id=class_ids)
                scene = box_annotator.annotate(scene=scene, detections=detections)
                scene = label_annotator.annotate(
                    scene=scene,
                    detections=detections,
                    labels=[f"gt {class_id}" for class_id in class_ids],
                )

            pred_boxes = result.get("boxes", torch.zeros((0, 4), dtype=torch.float32))
            if isinstance(pred_boxes, torch.Tensor) and pred_boxes.numel() > 0:
                pred_xyxy = pred_boxes.detach().cpu().numpy().astype(np.float32)
                orig_size = target.get("orig_size", size)
                if isinstance(orig_size, torch.Tensor):
                    orig_height, orig_width = int(orig_size[0]), int(orig_size[1])
                    if orig_height > 0 and orig_width > 0 and (orig_height, orig_width) != (height, width):
                        pred_xyxy[:, [0, 2]] *= width / orig_width
                        pred_xyxy[:, [1, 3]] *= height / orig_height
                scores = result.get("scores", torch.zeros((pred_xyxy.shape[0],), dtype=torch.float32))
                labels = result.get("labels", torch.zeros((pred_xyxy.shape[0],), dtype=torch.int64))
                class_ids = labels.detach().cpu().numpy().astype(int)
                confidences = scores.detach().cpu().numpy().astype(float)
                detections = sv.Detections(xyxy=pred_xyxy, class_id=class_ids, confidence=confidences)
                scene = box_annotator.annotate(scene=scene, detections=detections)
                scene = label_annotator.annotate(
                    scene=scene,
                    detections=detections,
                    labels=[
                        f"pred {class_id} {confidence:.2f}"
                        for class_id, confidence in zip(class_ids, confidences, strict=False)
                    ],
                )

            axis.imshow(scene)
            axis.set_title(f"sample {index}: GT + predictions", fontsize=10)
            axis.axis("off")

        fig.tight_layout()
        output_dir = self.output_dir / "prediction_grids"
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"val_epoch{epoch:04d}_batch{batch_idx:04d}_predictions.jpg", dpi=200)
        plt.close(fig)
