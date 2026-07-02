# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for build_trainer() — callback stack and config coercion."""

from unittest.mock import Mock

import pytest
from pytorch_lightning.callbacks import RichProgressBar, TQDMProgressBar

from rfdetr.training import build_trainer
from rfdetr.training.callbacks import DatasetGridCallback, PredictionGridCallback

# ---------------------------------------------------------------------------
# TestProgressBarCallbacks — verifies the correct callback is installed
# ---------------------------------------------------------------------------


class TestProgressBarCallbacks:
    """build_trainer() must install the right progress bar callback for each mode."""

    def test_rich_progress_bar_installed_for_rich(self, base_model_config, base_train_config):
        """progress_bar='rich' must add RichProgressBar and not TQDMProgressBar."""
        mc = base_model_config()
        tc = base_train_config(progress_bar="rich")
        trainer = build_trainer(tc, mc, accelerator="cpu")
        cb_types = [type(cb) for cb in trainer.callbacks]
        assert RichProgressBar in cb_types
        assert TQDMProgressBar not in cb_types

    def test_tqdm_progress_bar_installed_for_tqdm(self, base_model_config, base_train_config):
        """progress_bar='tqdm' must add TQDMProgressBar and not RichProgressBar."""
        mc = base_model_config()
        tc = base_train_config(progress_bar="tqdm")
        trainer = build_trainer(tc, mc, accelerator="cpu")
        cb_types = [type(cb) for cb in trainer.callbacks]
        assert TQDMProgressBar in cb_types
        assert RichProgressBar not in cb_types

    def test_progress_bar_refresh_rate_is_five(self, base_model_config, base_train_config):
        """The installed progress bar callback should refresh every five batches."""
        mc = base_model_config()
        tc = base_train_config(progress_bar="tqdm")
        trainer = build_trainer(tc, mc, accelerator="cpu")
        progress_bar = next(cb for cb in trainer.callbacks if isinstance(cb, TQDMProgressBar))

        assert progress_bar.refresh_rate == 5

    def test_no_progress_bar_callback_for_none(self, base_model_config, base_train_config):
        """progress_bar=None must not add any progress bar callback."""
        mc = base_model_config()
        tc = base_train_config(progress_bar=None)
        trainer = build_trainer(tc, mc, accelerator="cpu")
        cb_types = [type(cb) for cb in trainer.callbacks]
        assert RichProgressBar not in cb_types
        assert TQDMProgressBar not in cb_types


class TestDatasetGridCallback:
    """build_trainer() wires dataset grid saving at train epoch start."""

    def test_dataset_grid_callback_installed_when_enabled(self, base_model_config, base_train_config):
        """save_dataset_grids=True must add dataset and prediction grid callbacks."""
        mc = base_model_config()
        tc = base_train_config(save_dataset_grids=True)
        trainer = build_trainer(tc, mc, accelerator="cpu")

        assert any(isinstance(cb, DatasetGridCallback) for cb in trainer.callbacks)
        assert any(isinstance(cb, PredictionGridCallback) for cb in trainer.callbacks)

    def test_prediction_grid_callback_uses_validation_config(self, base_model_config, base_train_config):
        """Prediction grid confidence filter and cap come from validation TrainConfig fields."""
        mc = base_model_config()
        tc = base_train_config(
            save_dataset_grids=True,
            validation_prediction_grid_score_threshold=0.4,
            validation_prediction_grid_max_predictions=12,
        )
        trainer = build_trainer(tc, mc, accelerator="cpu")

        callback = next(cb for cb in trainer.callbacks if isinstance(cb, PredictionGridCallback))
        assert callback.score_threshold == 0.4
        assert callback.max_predictions == 12

    def test_dataset_grid_callback_skipped_when_disabled(self, base_model_config, base_train_config):
        """save_dataset_grids=False must not add image grid callbacks."""
        mc = base_model_config()
        tc = base_train_config(save_dataset_grids=False)
        trainer = build_trainer(tc, mc, accelerator="cpu")

        assert not any(isinstance(cb, DatasetGridCallback) for cb in trainer.callbacks)
        assert not any(isinstance(cb, PredictionGridCallback) for cb in trainer.callbacks)

    def test_dataset_grid_callback_saves_on_train_epoch_start(self):
        """DatasetGridCallback delegates saving to the attached datamodule at epoch start."""
        callback = DatasetGridCallback()
        datamodule = Mock()
        trainer = Mock(datamodule=datamodule, current_epoch=3)

        callback.on_train_epoch_start(trainer, Mock())

        datamodule._maybe_save_dataset_grids.assert_called_once_with(epoch=3)

    def test_prediction_grid_callback_skips_sanity_check(self):
        """Prediction grids are saved only for real validation, not Lightning sanity batches."""
        callback = PredictionGridCallback(output_dir="output")
        callback._save_prediction_grid = Mock()
        trainer = Mock(is_global_zero=True, sanity_checking=True, current_epoch=0)
        outputs = {"results": [], "targets": []}

        callback.on_validation_batch_end(trainer, Mock(), outputs, batch=None, batch_idx=0)

        callback._save_prediction_grid.assert_not_called()


# ---------------------------------------------------------------------------
# TestCoerceLegacyProgressBar — backward-compat validator on TrainConfig
# ---------------------------------------------------------------------------


class TestCoerceLegacyProgressBar:
    """_coerce_legacy_progress_bar must normalise legacy bool values."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            pytest.param(True, "tqdm", id="True->tqdm"),
            pytest.param(False, None, id="False->None"),
            pytest.param("rich", "rich", id="rich_passthrough"),
            pytest.param("tqdm", "tqdm", id="tqdm_passthrough"),
            pytest.param(None, None, id="None_passthrough"),
        ],
    )
    def test_coerce(self, base_train_config, value, expected):
        """progress_bar field normalises legacy bool and passes through string/None."""
        tc = base_train_config(progress_bar=value)
        assert tc.progress_bar == expected
