# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for RFDETRCli — PTL Ch4/T4.

Verifies that the CLI module is correctly structured: importable, subclasses LightningCLI, overrides
add_arguments_to_parser, and exposes a callable main() entry point.  CLI integration / smoke tests (--help subprocess,
YAML roundtrip) live in T4-7.
"""

import pytest

# ---------------------------------------------------------------------------
# Structure and importability
# ---------------------------------------------------------------------------


class TestRFDETRCliStructure:
    """RFDETRCli is correctly structured and importable."""

    def test_cli_module_importable(self):
        """rfdetr.training.cli imports without error."""
        import rfdetr.training.cli  # noqa: F401

    def test_rfdetr_cli_importable(self):
        """RFDETRCli can be imported from rfdetr.training.cli."""
        from rfdetr.training.cli import RFDETRCli  # noqa: F401

    def test_main_importable(self):
        """Main() can be imported from rfdetr.training.cli."""
        from rfdetr.training.cli import main  # noqa: F401

    def test_rfdetr_cli_is_lightning_cli_subclass(self):
        """RFDETRCli must subclass pytorch_lightning LightningCLI."""
        from pytorch_lightning.cli import LightningCLI

        from rfdetr.training.cli import RFDETRCli

        assert issubclass(RFDETRCli, LightningCLI)

    def test_main_is_callable(self):
        """Main must be a callable (function, not e.g. a string)."""
        from rfdetr.training.cli import main

        assert callable(main)

    def test_add_arguments_to_parser_is_overridden(self):
        """RFDETRCli overrides add_arguments_to_parser from LightningCLI."""
        from pytorch_lightning.cli import LightningCLI

        from rfdetr.training.cli import RFDETRCli

        assert RFDETRCli.add_arguments_to_parser is not LightningCLI.add_arguments_to_parser

    def test_exported_from_lit_package(self):
        """RFDETRCli is exported from rfdetr.training (appears in __all__)."""
        import rfdetr.training as lit

        assert hasattr(lit, "RFDETRCli")
        assert "RFDETRCli" in lit.__all__


# ---------------------------------------------------------------------------
# Argument linking
# ---------------------------------------------------------------------------


class TestRFDETRCliArgumentLinking:
    """add_arguments_to_parser registers the expected argument links."""

    def _collect_links(self):
        """Instantiate a minimal parser and collect registered link sources."""
        import unittest.mock as mock

        from rfdetr.training.cli import RFDETRCli

        captured = []

        class _FakeParser:
            def link_arguments(self, source, target, **kwargs):
                captured.append({"source": source, "target": target, **kwargs})

            # LightningArgumentParser methods that may be called during setup
            def __getattr__(self, name):
                return mock.MagicMock()

        cli = RFDETRCli.__new__(RFDETRCli)
        cli.add_arguments_to_parser(_FakeParser())
        return captured

    def test_model_config_link_registered(self):
        """model.model_config is linked to data.model_config."""
        links = self._collect_links()
        sources = [lnk["source"] for lnk in links]
        assert "model.model_config" in sources

    def test_train_config_link_registered(self):
        """model.train_config is linked to data.train_config."""
        links = self._collect_links()
        sources = [lnk["source"] for lnk in links]
        assert "model.train_config" in sources

    @pytest.mark.parametrize(
        "source, expected_target",
        [
            pytest.param("model.model_config", "data.model_config", id="model_config"),
            pytest.param("model.train_config", "data.train_config", id="train_config"),
        ],
    )
    def test_link_target(self, source, expected_target):
        """Each link points to the correct data.* target."""
        links = self._collect_links()
        match = next((lnk for lnk in links if lnk["source"] == source), None)
        assert match is not None, f"No link registered for source {source!r}"
        assert match["target"] == expected_target


class TestRFDETRCliConfigParsing:
    """RFDETRCli accepts the supported config file shapes."""

    def _make_model_parser(self):
        """Create a parser with the RFDETRModelModule arguments registered."""
        from pytorch_lightning.cli import LightningArgumentParser

        from rfdetr.training.module_model import RFDETRModelModule

        parser = LightningArgumentParser(exit_on_error=False)
        parser.add_lightning_class_args(RFDETRModelModule, "model")
        return parser

    def test_model_config_class_path_init_args_are_accepted(self):
        """Variant model_config sections may use class_path/init_args."""
        parser = self._make_model_parser()

        parsed = parser.parse_object(
            {
                "model": {
                    "model_config": {
                        "class_path": "rfdetr.config.RFDETRNanoConfig",
                        "init_args": {"num_classes": 6},
                    },
                    "train_config": {
                        "class_path": "rfdetr.config.TrainConfig",
                        "init_args": {"dataset_dir": "."},
                    },
                }
            }
        )

        assert parsed.model.model_config.class_path == "rfdetr.config.RFDETRNanoConfig"
        assert parsed.model.model_config.init_args.num_classes == 6
        assert parsed.model.model_config.init_args.num_windows == 2
        assert parsed.model.model_config.init_args.resolution == 384
        assert parsed.model.model_config.init_args.positional_encoding_size == 24
        assert parsed.model.model_config.init_args.num_queries == 300
        assert parsed.model.model_config.init_args.num_select == 300
        assert parsed.model.train_config.dataset_dir == "."

    def test_flat_model_config_with_train_config_class_path_is_accepted(self):
        """Custom base model_config sections may be flat while train_config uses class_path/init_args."""
        parser = self._make_model_parser()

        parsed = parser.parse_object(
            {
                "model": {
                    "model_config": {
                        "encoder": "dinov2_windowed_small",
                        "out_feature_indexes": [3, 6, 9, 12],
                        "dec_layers": 2,
                        "projector_scale": ["P4"],
                        "hidden_dim": 256,
                        "patch_size": 16,
                        "num_windows": 4,
                        "sa_nheads": 8,
                        "ca_nheads": 16,
                        "dec_n_points": 2,
                        "num_classes": 6,
                        "resolution": 640,
                        "positional_encoding_size": 40,
                    },
                    "train_config": {
                        "class_path": "rfdetr.config.TrainConfig",
                        "init_args": {"dataset_dir": ".", "batch_size": 32},
                    },
                }
            }
        )

        assert parsed.model.model_config.resolution == 640
        assert parsed.model.model_config.num_classes == 6
        assert parsed.model.train_config.batch_size == 32


class TestRFDETRCliTrainerInstantiation:
    """RFDETRCli builds Trainer from RF-DETR TrainConfig semantics."""

    def test_instantiate_trainer_uses_build_trainer_without_lightning_epoch_defaults(self):
        """Lightning default max_epochs must not override TrainConfig.epochs."""
        from types import SimpleNamespace
        from unittest.mock import sentinel

        from jsonargparse import Namespace

        from rfdetr.config import RFDETRNanoConfig, TrainConfig
        from rfdetr.training.cli import RFDETRCli

        model_config = RFDETRNanoConfig(pretrain_weights=None)
        train_config = TrainConfig(dataset_dir=".", epochs=10, grad_accum_steps=2)

        cli = RFDETRCli.__new__(RFDETRCli)
        cli.subcommand = None
        cli.config_init = Namespace(
            trainer=Namespace(
                accelerator="gpu",
                devices="auto",
                max_epochs=1000,
                accumulate_grad_batches=1,
            )
        )
        cli.model = SimpleNamespace(model_config=model_config, train_config=train_config)

        with pytest.MonkeyPatch.context() as monkeypatch:
            calls = []

            def fake_build_trainer(*args, **kwargs):
                calls.append((args, kwargs))
                return sentinel.trainer

            monkeypatch.setattr("rfdetr.training.cli.build_trainer", fake_build_trainer)
            trainer = cli.instantiate_trainer()

        assert trainer is sentinel.trainer
        assert calls == [
            (
                (train_config, model_config),
                {"accelerator": "gpu", "devices": "auto"},
            )
        ]
