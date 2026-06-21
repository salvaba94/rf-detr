# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Callback for saving augmented dataset preview grids."""

from __future__ import annotations

from typing import Any

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
