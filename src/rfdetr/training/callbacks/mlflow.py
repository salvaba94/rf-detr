# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""MLflow artifact logging callback for RF-DETR training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

from pytorch_lightning import Callback, LightningModule, Trainer

from rfdetr.utilities.logger import get_logger

logger = get_logger()


class MLFlowSystemMonitorCallback(Callback):
    """Start MLflow's native system metrics monitor for the active Lightning run."""

    def __init__(self) -> None:
        """Initialize the callback."""
        super().__init__()
        self.system_monitor: Any | None = None

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Start MLflow system metric collection for the active MLflow run."""
        if not trainer.is_global_zero:
            return
        mlflow_logger = RFDETRMLflowArtifactCallback._get_mlflow_logger(trainer)
        if mlflow_logger is None:
            logger.warning("MLflow system metrics disabled: no MLFlowLogger is attached to the trainer.")
            return
        try:
            from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor
        except ModuleNotFoundError as exc:
            logger.warning("MLflow system metrics disabled: %s. Install MLflow system metrics dependencies.", exc)
            return
        except Exception as exc:  # pragma: no cover - defensive optional dependency guard
            logger.warning("MLflow system metrics disabled: %s", exc)
            return

        try:
            self.system_monitor = SystemMetricsMonitor(run_id=mlflow_logger.run_id)
            self.system_monitor.start()
            logger.info("Started MLflow system metrics monitor for run %s.", mlflow_logger.run_id)
        except Exception as exc:  # pragma: no cover - defensive optional dependency guard
            self.system_monitor = None
            logger.warning("MLflow system metrics disabled: %s", exc)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Stop MLflow system metric collection."""
        if not trainer.is_global_zero or self.system_monitor is None:
            return
        try:
            self.system_monitor.finish()
            logger.info("Stopped MLflow system metrics monitor.")
        except Exception as exc:  # pragma: no cover - defensive optional dependency guard
            logger.warning("Failed to stop MLflow system metrics monitor cleanly: %s", exc)
        finally:
            self.system_monitor = None


class RFDETRMLflowArtifactCallback(Callback):
    """Upload RF-DETR run artifacts to the active MLflow run.

    Args:
        output_dir: Training output directory that contains metrics, checkpoints,
            dataset grids, and RF-DETR-managed artifacts.
    """

    def __init__(self, output_dir: str) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self._logged_paths: set[Path] = set()

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log static run artifacts after datamodule setup has completed."""
        if not trainer.is_global_zero:
            return
        self._write_and_log_config_snapshot(trainer, pl_module)
        self._log_directory(trainer, self.output_dir / "dataset_grids", artifact_path="dataset_grids")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log end-of-run artifacts such as metrics and checkpoints."""
        if not trainer.is_global_zero:
            return
        self._write_metric_plots()
        self._log_directory(trainer, self.output_dir / "dataset_grids", artifact_path="dataset_grids")
        self._log_directory(trainer, self.output_dir / "prediction_grids", artifact_path="prediction_grids")
        self._log_directory(trainer, self.output_dir / "plots", artifact_path="plots")
        self._log_file(trainer, self.output_dir / "metrics.csv", artifact_path="metrics")
        self._log_matching_files(
            trainer,
            patterns=("*.ckpt", "*.pth", "*.pt"),
            artifact_path="checkpoints",
        )

    def _write_and_log_config_snapshot(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Write resolved RF-DETR configs to disk and upload them."""
        config_dir = self.output_dir / "mlflow_artifacts"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "resolved_config.json"
        payload = {
            "model_config": self._model_dump(getattr(pl_module, "model_config", None)),
            "train_config": self._model_dump(getattr(pl_module, "train_config", None)),
        }
        config_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        self._log_file(trainer, config_path, artifact_path="config")

    @staticmethod
    def _model_dump(config: Any) -> Any:
        """Return a JSON-serialisable dump for a Pydantic config object."""
        if hasattr(config, "model_dump"):
            return config.model_dump(mode="json")
        return config

    def _log_matching_files(self, trainer: Trainer, *, patterns: Iterable[str], artifact_path: str) -> None:
        """Upload files in ``output_dir`` matching any of the given glob patterns."""
        if not self.output_dir.exists():
            return
        for pattern in patterns:
            for path in sorted(self.output_dir.glob(pattern)):
                self._log_file(trainer, path, artifact_path=artifact_path)

    def _log_directory(self, trainer: Trainer, path: Path, *, artifact_path: str) -> None:
        """Upload a directory to MLflow when it exists."""
        if not path.is_dir():
            return
        for file_path in sorted(path.rglob("*")):
            if not file_path.is_file():
                continue
            relative_parent = file_path.parent.relative_to(path)
            nested_artifact_path = Path(artifact_path)
            if str(relative_parent) != ".":
                nested_artifact_path /= relative_parent
            self._log_file(trainer, file_path, artifact_path=str(nested_artifact_path))

    def _write_metric_plots(self) -> None:
        """Render training metric plots from ``metrics.csv`` when plotting deps are installed."""
        metrics_path = self.output_dir / "metrics.csv"
        if not metrics_path.is_file():
            return
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_specs = (
            ("metrics.png", "plot_metrics", {"loss_log_scale": True}),
            ("loss.png", "plot_loss_metrics", {"loss_log_scale": True}),
            ("map.png", "plot_map_metrics", {}),
        )
        for filename, function_name, kwargs in plot_specs:
            output_path = plots_dir / filename
            if output_path.resolve() in self._logged_paths:
                continue
            try:
                from rfdetr.visualize import training as training_plots

                figure = getattr(training_plots, function_name)(
                    str(metrics_path),
                    output_path=str(output_path),
                    **kwargs,
                )
                try:
                    import matplotlib.pyplot as plt

                    plt.close(figure)
                except Exception:  # pragma: no cover - best-effort figure cleanup
                    pass
            except (FileNotFoundError, ImportError, ValueError) as exc:
                logger.warning("MLflow metrics plot skipped for %s: %s", function_name, exc)
            except Exception as exc:  # pragma: no cover - defensive logging, not control flow
                logger.warning("MLflow metrics plot skipped for %s: %s", function_name, exc)

    def _log_file(self, trainer: Trainer, path: Path, *, artifact_path: str) -> None:
        """Upload one file to MLflow when it exists."""
        resolved_path = path.resolve()
        if not path.is_file() or resolved_path in self._logged_paths:
            return
        mlflow_logger = self._get_mlflow_logger(trainer)
        if mlflow_logger is None:
            return
        try:
            mlflow_logger.experiment.log_artifact(
                mlflow_logger.run_id,
                str(path),
                artifact_path=artifact_path,
            )
            self._logged_paths.add(resolved_path)
        except Exception as exc:  # pragma: no cover - defensive logging, not control flow
            logger.warning("MLflow artifact upload skipped for %s: %s", path, exc)

    @staticmethod
    def _get_mlflow_logger(trainer: Trainer) -> Optional[Any]:
        """Return the trainer's MLflow logger, if present."""
        for candidate in getattr(trainer, "loggers", []) or []:
            if candidate.__class__.__name__ == "MLFlowLogger":
                return candidate
        return None
