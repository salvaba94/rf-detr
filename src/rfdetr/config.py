# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import functools
import importlib
import json
import os
import warnings
from collections.abc import Callable, Mapping
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional, TypeAlias, Union

import torch
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator
from pydantic_core import PydanticUndefined
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

EncoderName: TypeAlias = Literal["dinov2_windowed_small", "dinov2_windowed_base", "dinov2_registers_windowed_small"]
PathLikeStr: TypeAlias = str | Path

__all__ = [
    "AugmentationBackend",
    "ModelConfig",
    "RFDETRBaseConfig",
    "RFDETRLargeDeprecatedConfig",
    "RFDETRNanoConfig",
    "RFDETRSmallConfig",
    "RFDETRMediumConfig",
    "RFDETRLargeConfig",
    "RFDETRSegPreviewConfig",
    "RFDETRSegNanoConfig",
    "RFDETRSegSmallConfig",
    "RFDETRSegMediumConfig",
    "RFDETRSegLargeConfig",
    "RFDETRSegXLargeConfig",
    "RFDETRSeg2XLargeConfig",
    "RFDETRKeypointPreviewConfig",
    "TrainConfig",
    "SegmentationTrainConfig",
    "KeypointTrainConfig",
]

#: Legacy augmentation-backend string aliases, mapped to their current form.
_LEGACY_AUGMENTATION_BACKEND_ALIASES: Dict[str, str] = {
    "gpu": "kornia",
    "tv": "torchvision",
    "albu": "albumentations",
}


def _package_importable(module_name: str) -> bool:
    """Return ``True`` when *module_name* can be imported.

    Args:
        module_name: Dotted module path to probe (e.g. ``"kornia.augmentation"``).

    Returns:
        ``True`` if the import succeeds, ``False`` on ``ImportError``.
    """
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


class AugmentationBackend(str, Enum):
    """Concrete augmentation backend selector for ``TrainConfig.augmentation_backend``.

    Only holds directly-usable, concrete backends — ``TV`` (torchvision), ``ALBU`` (Albumentations), and ``KORNIA``.
    ``GPU`` is a Python enum alias for ``KORNIA`` (same value ``"kornia"``): Kornia augmentation always runs on-device
    (GPU), so the two names refer to the same backend; ``GPU`` exists only so legacy ``augmentation_backend="gpu"``
    strings keep resolving correctly.

    ``"cpu"`` and ``"auto"`` are accepted as *input* strings (on ``TrainConfig.augmentation_backend`` and by
    :meth:`from_str`) but are never stored or returned as a member of this enum — they are auto-pick sentinels resolved
    to a concrete member at :meth:`from_str` call time. Resolution stays late (re-checked at dataset-build time against
    whatever is installed in the current environment) rather than baked into ``TrainConfig`` at construction time, so a
    saved config using ``"cpu"``/``"auto"`` remains portable across environments with different optional packages
    installed. Pass a concrete value (``"torchvision"``, ``"albumentations"``, or ``"kornia"``) explicitly to pin the
    backend regardless of environment.
    """

    TV = "torchvision"
    ALBU = "albumentations"
    KORNIA = "kornia"
    GPU = "kornia"  # alias for KORNIA — backward compat name; kornia is always the GPU-side path

    @classmethod
    def from_str(cls, value: str, *, has_cuda: bool = False) -> "AugmentationBackend":
        """Resolve a string to a concrete backend, auto-picking the best installed one.

        Legacy string aliases (``"gpu"``, ``"tv"``, ``"albu"``) are mapped to their current form
        first. ``"cpu"`` auto-picks the best *installed* CPU backend: Albumentations > Kornia
        (CPU) > torchvision. ``"auto"`` additionally prefers Kornia first when ``has_cuda=True``
        and Kornia is installed, then falls back to the same CPU priority. The concrete backend
        ``"cpu"``/``"auto"`` resolve to can therefore vary across environments — pass
        ``"torchvision"`` explicitly to force torchvision regardless of what's installed.

        Args:
            value: Backend name string.
            has_cuda: Whether a CUDA device is available. Only consulted for ``"auto"`` — callers
                that care about CUDA-gated GPU selection (e.g. dataset builders) compute this via
                their own fork-safe CUDA check and pass it in; this function does not probe CUDA
                itself to avoid importing device-detection code from other modules.

        Returns:
            Concrete ``AugmentationBackend`` member.

        Raises:
            ValueError: When *value* is not a recognised backend name.

        Examples:
            >>> AugmentationBackend.from_str("torchvision")
            <AugmentationBackend.TV: 'torchvision'>
            >>> AugmentationBackend.from_str("gpu")
            <AugmentationBackend.KORNIA: 'kornia'>
        """
        value = _LEGACY_AUGMENTATION_BACKEND_ALIASES.get(value, value)
        if value in ("cpu", "auto"):
            if value == "auto" and has_cuda and cls._is_kornia_available():
                return cls.KORNIA
            if cls._is_albu_available():
                return cls.ALBU
            if cls._is_kornia_available():
                return cls.KORNIA
            return cls.TV
        try:
            return cls(value)
        except ValueError:
            raise ValueError(
                f"Unknown augmentation_backend {value!r}; expected one of 'cpu', 'auto', 'torchvision', "
                "'albumentations', 'kornia'."
            ) from None

    @classmethod
    @functools.lru_cache(maxsize=None)
    def _is_albu_available(cls) -> bool:
        """Return ``True`` when Albumentations is importable.

        Cached for the process lifetime — package installation state does not change at runtime.
        Tests that need to simulate "not installed" should patch this method directly (e.g.
        ``patch.object(AugmentationBackend, "_is_albu_available", return_value=False)``) rather
        than blocking the underlying import, since the cache is keyed on this method, not on the
        import machinery.

        Returns:
            ``True`` if ``albumentations`` can be imported.
        """
        return _package_importable("albumentations")

    @classmethod
    @functools.lru_cache(maxsize=None)
    def _is_kornia_available(cls) -> bool:
        """Return ``True`` when Kornia's augmentation module is importable.

        Cached for the process lifetime — see :meth:`_is_albu_available` for the caching and test
        rationale.

        Returns:
            ``True`` if ``kornia.augmentation`` can be imported.
        """
        return _package_importable("kornia.augmentation")

    @classmethod
    def _is_tv_available(cls) -> bool:
        """Return ``True`` — torchvision is a hard (non-optional) RF-DETR dependency.

        Not cached: the result is a compile-time constant, not worth the caching machinery.

        Returns:
            Always ``True``.
        """
        return True


class PretrainWeightsCompatibilityWarning(UserWarning):
    """Warning emitted when ``ModelConfig`` overrides are likely to prevent the variant's published pretrained weights
    from loading into the model — leaving large portions of the model randomly initialized and typically producing much
    lower accuracy."""


def _detect_device() -> str:
    """Detect the best available device **without** initialising the CUDA runtime.

    ``torch.cuda.is_available()`` creates a CUDA driver context that makes ``_is_in_bad_fork()`` return ``True`` in
    child processes.  This breaks fork-based DDP strategies (e.g. ``ddp_notebook``) in notebook environments.

    We defer to :func:`torch.accelerator.current_accelerator` (PyTorch ≥ 2.4) when available — it queries the driver
    through NVML without creating a primary context.  On older builds we fall back to ``torch.cuda.is_available()``.

    ``check_available=True`` is required: without it ``current_accelerator()`` only reports the *compile-time*
    accelerator, so the default CUDA wheel on a machine without an NVIDIA driver yields ``"cuda"`` and every model build
    crashes with "Found no NVIDIA driver".  The runtime check is NVML-backed and still avoids creating a CUDA context.
    Builds whose ``current_accelerator`` predates the ``check_available`` kwarg get the same runtime verification via
    ``torch.accelerator.is_available``.
    """
    accelerator = getattr(torch, "accelerator", None)
    current_accelerator = getattr(accelerator, "current_accelerator", None)
    if current_accelerator is not None:
        try:
            try:
                accel = current_accelerator(check_available=True)
            except TypeError:
                accel = current_accelerator()
                if accel is not None and not accelerator.is_available():
                    accel = None
            if accel is not None:
                return str(accel)
            return "cpu"
        except RuntimeError:
            return "cpu"
    # Fallback for PyTorch < 2.4 — this DOES create a CUDA driver context.
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE: str = _detect_device()
_OPTIMIZER_MANAGED_KWARGS = {"params", "lr", "weight_decay", "fused"}


def _resolve_native_optimizer(name: str) -> type[Optimizer]:
    """Resolve a bare optimizer short name to a ``torch.optim`` optimizer class.

    Only native ``torch.optim`` optimizers may be selected by short name; the match
    is case-insensitive (``"adamw"`` → ``torch.optim.AdamW``, ``"sgd"`` → ``torch.optim.SGD``).
    Any other optimizer must be given as a full dotted import path or a callable.

    Args:
        name: A bare optimizer name (no dotted import path).

    Returns:
        The matching ``torch.optim`` optimizer class.

    Raises:
        ValueError: If ``name`` is not a native ``torch.optim`` optimizer.

    Examples:
        >>> _resolve_native_optimizer("adamw") is torch.optim.AdamW
        True
    """
    target = name.strip().lower()
    for attribute in dir(torch.optim):
        candidate = getattr(torch.optim, attribute)
        if isinstance(candidate, type) and issubclass(candidate, Optimizer) and attribute.lower() == target:
            return candidate
    raise ValueError(
        f"Unknown native optimizer {name!r}. Short names must name a torch.optim optimizer "
        "(e.g. 'adamw', 'sgd', 'adam'); use a full dotted import path or a callable for anything else."
    )


def _is_managed_optimizer_name(optimizer: object) -> bool:
    """Return whether an optimizer config selects RF-DETR's managed construction.

    Managed mode covers bare ``torch.optim`` short names (e.g. ``"adamw"``, ``"sgd"``);
    RF-DETR injects ``lr`` and a signature-aware ``weight_decay`` there. A dotted import
    path or a callable selects explicit mode, where the optimizer is built only from
    ``optimizer_kwargs`` (or the callable's own bound arguments).

    Args:
        optimizer: The ``TrainConfig.optimizer`` value.

    Returns:
        ``True`` for managed short-name strings, ``False`` for dotted paths and callables.

    Examples:
        >>> _is_managed_optimizer_name("sgd")
        True
        >>> _is_managed_optimizer_name("torch.optim.AdamW")
        False
    """
    return isinstance(optimizer, str) and "." not in optimizer


def _desugar_optimizer_callable(
    optimizer: Callable[..., Optimizer],
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Decompose a callable optimizer into a serializable ``(dotted_path, kwargs)`` form.

    Reconstructable callables — an importable top-level class or function, optionally
    wrapped in ``functools.partial`` with JSON-serializable keyword arguments and no
    positional arguments — desugar to a dotted import path plus keyword arguments that
    round-trip through ``training_config.json``.

    Args:
        optimizer: A callable or ``functools.partial`` given as ``TrainConfig.optimizer``.

    Returns:
        ``(dotted_path, kwargs, None)`` when reconstructable, otherwise
        ``(None, None, reason)`` where ``reason`` explains how to make it compatible.
    """
    func: Any = optimizer
    extracted_kwargs: dict[str, Any] = {}
    if isinstance(optimizer, functools.partial):
        if optimizer.args:
            return None, None, "pass every functools.partial argument as a keyword, not positionally"
        func = optimizer.func
        extracted_kwargs = dict(optimizer.keywords or {})

    module = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", None)
    if module is None or qualname is None or "<" in qualname:
        return (
            None,
            None,
            "define the optimizer as an importable top-level class or function (no lambda or nested definition)",
        )

    try:
        json.dumps(extracted_kwargs)
    except (TypeError, ValueError):
        return (
            None,
            None,
            "use only JSON-serializable functools.partial keyword arguments (no tensors, modules, or callables)",
        )

    return f"{module}.{qualname}", extracted_kwargs, None


_MANAGED_SCHEDULER_PRESETS = {"step", "cosine"}
_DEPRECATED_LR_FIELD_KWARGS = {"lr_drop": "lr_drop", "lr_min_factor": "min_factor"}
# Keys the managed "step" / "cosine" presets actually consume from lr_scheduler_kwargs.
_MANAGED_SCHEDULER_KWARGS = {"min_factor", "lr_drop"}

# ReduceLROnPlateau does not subclass LRScheduler but is a supported explicit scheduler.
SchedulerType: TypeAlias = LRScheduler | ReduceLROnPlateau


def _is_managed_scheduler_name(lr_scheduler: object) -> bool:
    """Return whether an lr_scheduler config selects an RF-DETR managed preset.

    Managed presets are the built-in ``"step"`` and ``"cosine"`` schedules, which own warmup
    and total-step sizing. A dotted import path or a callable instead selects an explicit
    scheduler built from ``lr_scheduler_kwargs`` (or the callable's own bound arguments).

    Args:
        lr_scheduler: The ``TrainConfig.lr_scheduler`` value.

    Returns:
        ``True`` for managed preset short names, ``False`` for dotted paths and callables.

    Examples:
        >>> _is_managed_scheduler_name("cosine")
        True
        >>> _is_managed_scheduler_name("torch.optim.lr_scheduler.StepLR")
        False
    """
    return isinstance(lr_scheduler, str) and lr_scheduler.strip().lower() in _MANAGED_SCHEDULER_PRESETS


def _desugar_scheduler_callable(
    lr_scheduler: Callable[..., SchedulerType],
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Decompose a callable lr_scheduler into a serializable ``(dotted_path, kwargs)`` form.

    Reconstructable callables — an importable top-level class or function, optionally wrapped in
    ``functools.partial`` with JSON-serializable keyword arguments and no positional arguments —
    desugar to a dotted import path plus keyword arguments that round-trip through
    ``training_config.json``. The optimizer is supplied at build time, never baked into the callable.

    Args:
        lr_scheduler: A callable or ``functools.partial`` given as ``TrainConfig.lr_scheduler``.

    Returns:
        ``(dotted_path, kwargs, None)`` when reconstructable, otherwise
        ``(None, None, reason)`` where ``reason`` explains how to make it compatible.
    """
    func: Any = lr_scheduler
    extracted_kwargs: dict[str, Any] = {}
    if isinstance(lr_scheduler, functools.partial):
        if lr_scheduler.args:
            return None, None, "pass every functools.partial argument as a keyword, not positionally"
        func = lr_scheduler.func
        extracted_kwargs = dict(lr_scheduler.keywords or {})

    module = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", None)
    if module is None or qualname is None or "<" in qualname:
        return (
            None,
            None,
            "define the lr_scheduler as an importable top-level class or function (no lambda or nested definition)",
        )

    try:
        json.dumps(extracted_kwargs)
    except (TypeError, ValueError):
        return (
            None,
            None,
            "use only JSON-serializable functools.partial keyword arguments (no tensors, modules, or callables)",
        )

    return f"{module}.{qualname}", extracted_kwargs, None


class BaseConfig(BaseModel):
    """Base configuration class that validates input parameters against the defined model schema.

    If any unknown fields are provided, a ValueError is raised listing the unknown and available parameters.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", validate_assignment=True)

    @model_validator(mode="before")
    @classmethod
    def catch_typo_kwargs(cls, values: Any) -> Any:
        if not isinstance(values, Mapping):
            return values
        if cls.model_config.get("extra") != "forbid":
            return values
        allowed_params = set(cls.model_fields.keys())
        provided_params = set(values)
        unknown_params = provided_params - allowed_params
        if unknown_params:
            unknown_params_list = ", ".join(f"'{param}'" for param in sorted(unknown_params))
            allowed_params_list = ", ".join(sorted(allowed_params))
            raise ValueError(
                f"Unknown parameter(s): {unknown_params_list}. Available parameter(s): {allowed_params_list}."
            )
        return values

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or name in type(self).model_fields:
            super().__setattr__(name, value)
            return
        raise ValueError(f"Unknown attribute: '{name}'.")


class StalConfig(BaseConfig):
    """Small-target geometry relaxation for Hungarian matching."""

    enabled: bool = False
    small_box_threshold: float = Field(default=8.0, ge=0.0)
    expanded_box_size: float = Field(default=16.0, ge=0.0)

    @model_validator(mode="after")
    def _validate_expanded_box_size(self) -> "StalConfig":
        """Require relaxed dimensions to be no smaller than the trigger threshold."""
        if self.expanded_box_size < self.small_box_threshold:
            raise ValueError("stal.expanded_box_size must be greater than or equal to stal.small_box_threshold")
        return self


class EarlyStoppingConfig(BaseConfig):
    """Early-stopping callback options."""

    enabled: bool = False
    patience: int = Field(default=10, ge=1)
    min_delta: float = Field(default=0.001, ge=0.0)
    use_ema: bool = False


class MultiScaleConfig(BaseConfig):
    """Multi-scale training resize options."""

    enabled: bool = True
    expanded_scales: bool = True
    min_offset: Optional[int] = None
    max_offset: Optional[int] = None
    random_resize_via_padding: bool = False

    @model_validator(mode="after")
    def _validate_offset_range(self) -> "MultiScaleConfig":
        """Validate explicit multi-scale offset bounds."""
        if self.min_offset is not None and self.max_offset is not None and self.min_offset > self.max_offset:
            raise ValueError(
                "multi_scale.min_offset must be less than or equal to multi_scale.max_offset, "
                f"got {self.min_offset} > {self.max_offset}."
            )
        return self


class EmaConfig(BaseConfig):
    """Exponential moving average training options."""

    enabled: bool = True
    decay: float = 0.993
    tau: int = Field(default=100, ge=1)
    update_interval: int = Field(default=1, ge=1)
    auto_batch_headroom: float = Field(default=0.7, gt=0.0, le=1.0)


class OptimizerSchedulerConfig(BaseConfig):
    """Learning-rate scheduler options grouped under optimizer configuration."""

    name: Literal["step", "cosine"] = "step"
    min_factor: float = 0.0
    warmup_epochs: float = 0.0
    drop_epoch: int = 100


class OptimizerConfig(BaseConfig):
    """Optimizer hyperparameters grouped away from TrainConfig's legacy flat fields."""

    lr: float = 1e-4
    lr_encoder: float = 1.5e-4
    weight_decay: float = 1e-4
    momentum: float = Field(default=0.95, ge=0.0, lt=1.0)
    nesterov: bool = True
    muon_lr_scale: float = Field(default=0.2, ge=0.0)
    fallback_lr_scale: float = Field(default=1.0, ge=0.0)
    ns_coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315)
    eps: float = Field(default=1e-7, gt=0.0)
    ns_steps: int = Field(default=5, ge=1)
    adjust_lr_fn: Optional[Literal["original", "match_rms_adamw"]] = None
    scheduler: OptimizerSchedulerConfig = Field(default_factory=OptimizerSchedulerConfig)


class SahiValidationConfig(BaseConfig):
    """Fixed-grid SAHI validation options."""

    slice_height: Optional[int] = Field(default=None, ge=1)
    slice_width: Optional[int] = Field(default=None, ge=1)
    overlap_height_ratio: float = Field(default=0.2, ge=0.0, lt=1.0)
    overlap_width_ratio: float = Field(default=0.2, ge=0.0, lt=1.0)
    nms_iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    merge_metric: Literal["iou", "ios", "diou", "cdn"] = "iou"
    source_aware_duplicate_suppression: bool = False
    include_full_image: bool = False
    tile_batch_size: Optional[int] = Field(default=None, ge=1)
    batch_across_images: bool = False
    tile_input_dtype: Literal["auto", "fp32", "bf16", "fp16"] = "auto"
    tile_memory_format: Literal["contiguous", "channels_last"] = "contiguous"
    score_threshold: float = Field(default=0.001, ge=0.0, le=1.0)


class AsahiValidationConfig(BaseConfig):
    """Adaptive SAHI validation options."""

    short_side_threshold: int = Field(default=1280, ge=1)
    low_patch_count: Literal[6] = 6
    high_patch_count: Literal[6, 12] = 12
    overlap_ratio: float = Field(default=0.15, ge=0.0, lt=1.0)
    include_full_image: bool = True
    source_mode: Literal["adaptive", "full", "full_adaptive"] = "full_adaptive"
    window_resize_longest_side: Optional[int] = Field(default=None, ge=1)
    window_resize_policy: Literal[
        "aspect_longest_side",
        "aspect_valid_target",
        "square_stretch",
        "square_letterbox",
        "letterbox_valid_target",
        "letterbox_canvas_target",
        "square_context",
    ] = "square_context"
    nms_iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    merge_metric: Literal["iou", "ios", "diou", "cdn"] = "cdn"
    source_aware_duplicate_suppression: bool = False
    tile_batch_size: Optional[int] = Field(default=None, ge=1)
    batch_across_images: bool = False
    tile_input_dtype: Literal["auto", "fp32", "bf16", "fp16"] = "auto"
    tile_memory_format: Literal["contiguous", "channels_last"] = "contiguous"
    score_threshold: float = Field(default=0.001, ge=0.0, le=1.0)


class GsahiValidationConfig(BaseConfig):
    """Guided SAHI coarse-to-fine validation options."""

    coarse_slice_size: int = Field(default=640, ge=1)
    fine_slice_size: int = Field(default=256, ge=1)
    coarse_overlap: float = Field(default=0.2, ge=0.0, lt=1.0)
    fine_overlap: float = Field(default=0.2, ge=0.0, lt=1.0)
    include_full_image: bool = True
    roi_score_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    roi_expansion_ratio: float = Field(default=0.25, ge=0.0)
    roi_max_regions: int = Field(default=32, ge=1)
    nms_iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    merge_metric: Literal["iou", "ios", "diou", "cdn"] = "iou"
    merge_sources: Literal["full_fine", "full_coarse_fine", "fine"] = "full_fine"
    source_aware_duplicate_suppression: bool = False
    tile_batch_size: Optional[int] = Field(default=None, ge=1)
    tile_input_dtype: Literal["auto", "fp32", "bf16", "fp16"] = "auto"
    tile_memory_format: Literal["contiguous", "channels_last"] = "contiguous"
    score_threshold: float = Field(default=0.001, ge=0.0, le=1.0)


class ModelConfig(BaseConfig):
    """Core architecture configuration for RF-DETR models.

    Concrete subclasses (e.g. ``RFDETRBaseConfig``, ``RFDETRLargeConfig``) must supply every field
    that has no default; direct instantiation of ``ModelConfig`` is unsupported.

    Attributes:
        encoder: Vision-transformer backbone identifier. Must be provided by concrete subclass.
        out_feature_indexes: Encoder layer indices whose feature maps are forwarded to the decoder.
            Must be provided by concrete subclass.
        dec_layers: Number of transformer decoder layers. Must be provided by concrete subclass.
        projector_scale: Feature-pyramid levels fed to the decoder cross-attention (subset of
            ``["P3", "P4", "P5"]``). Must be provided by concrete subclass.
        hidden_dim: Width of the decoder hidden state. Must be provided by concrete subclass.
        patch_size: ViT patch size used by the backbone. Must be provided by concrete subclass.
        num_windows: Number of windowed-attention windows in the backbone. Must be provided by
            concrete subclass.
        sa_nheads: Number of heads in decoder self-attention. Must be provided by concrete
            subclass.
        ca_nheads: Number of heads in decoder cross-attention. Must be provided by concrete
            subclass.
        dec_n_points: Deformable attention points per head per level in the decoder. Must be
            provided by concrete subclass.
        resolution: Square input resolution (pixels). Must be provided by concrete subclass.
        positional_encoding_size: Side length (in patches) of the sinusoidal positional grid.
            Must be provided by concrete subclass.
        num_queries: Number of object queries used during inference (and per group during
            training). Defaults to ``300``.
        num_classes: Number of output classes (background-free). Defaults to ``90`` (COCO).
        group_detr: Number of duplicate query groups used during training for GroupPose-style
            convergence acceleration. ``num_queries * group_detr`` predictions are produced in
            training mode; ``num_queries`` in eval mode. ``num_queries`` must be divisible by
            ``group_detr``. Defaults to ``13``.
        amp: Enable automatic mixed precision (bfloat16/float16). Defaults to ``True``.
        compile: Compile the model with ``torch.compile`` for faster throughput. Defaults to
            ``False``.
        pretrain_weights: Path or URL to pretrained checkpoint. ``None`` trains from scratch.
        device: Target device string (e.g. ``"cuda"``, ``"cpu"``). Auto-detected if not set.
        gradient_checkpointing: Trade compute for memory by checkpointing activations. Defaults
            to ``False``.
    """

    encoder: EncoderName
    out_feature_indexes: list[int]
    dec_layers: int
    two_stage: bool = True
    projector_scale: list[Literal["P3", "P4", "P5"]]
    hidden_dim: int
    patch_size: int
    num_windows: int
    sa_nheads: int
    ca_nheads: int
    dec_n_points: int
    num_queries: int = 300
    # ModelConfig is the sole owner of `num_select` for PTL/inference; it is read via `_namespace_from_configs`.
    num_select: int = 300
    postprocess_trace_alpha: float = Field(default=0.2, ge=0.0)
    bbox_reparam: bool = True
    lite_refpoint_refine: bool = True
    layer_norm: bool = True
    amp: bool = True
    num_channels: int = Field(default=3, ge=1)
    num_classes: int = 90
    pretrain_weights: PathLikeStr | None = None
    # torch.device values are accepted at validation time and normalized to string.
    device: str = DEVICE
    resolution: int
    group_detr: int = 13
    gradient_checkpointing: bool = False
    compile: bool = False
    fused_optimizer: bool = True
    positional_encoding_size: int
    ia_bce_loss: bool = True
    segmentation_head: bool = False
    use_grouppose_keypoints: bool = False
    keypoint_cross_attn: bool = True
    inter_instance_kp_attn: bool = False
    grouppose_keypoint_dim_downscale: int = 1
    dual_projector: bool = False
    dual_projector_kp_only: bool = False
    num_keypoints_per_class: list[int] = Field(default_factory=list)
    num_decoder_registers: int = 0
    mask_downsample_ratio: int = 4
    backbone_lora: bool = False
    freeze_encoder: bool = False
    license: str = "Apache-2.0"
    model_name: str | None = Field(
        default=None,
        description=(
            'Name of the model class stored in training checkpoints (e.g. ``"RFDETRLarge"``). '
            "Set automatically by ``RFDETR.train()`` before saving. "
            "Used by ``RFDETR.from_checkpoint()`` to resolve the correct subclass directly "
            "without inspecting ``pretrain_weights``."
        ),
    )

    @model_validator(mode="after")
    def _sync_pe_with_resolution(self) -> "ModelConfig":
        """Auto-update positional_encoding_size when resolution is explicitly provided.

        When a user provides a custom ``resolution`` at construction time (e.g., ``RFDETRLarge(resolution=640)``),
        ``positional_encoding_size`` is updated proportionally, provided the class-default PE is formula-derived
        (``default_pe == default_resolution // patch_size``).

        Configs with a pretrained-specific PE (e.g., ``RFDETRBaseConfig`` with ``positional_encoding_size=37`` for
        DINOv2's native 518 px grid, while ``resolution=560``) are left unchanged.
        """
        if "resolution" not in self.model_fields_set or "positional_encoding_size" in self.model_fields_set:
            return self

        cls = type(self)
        default_resolution = cls.model_fields["resolution"].default
        default_pe = cls.model_fields["positional_encoding_size"].default
        default_patch_size = cls.model_fields["patch_size"].default

        # Skip when any relevant default is not a concrete integer (abstract base
        # class fields have no defaults; required fields use PydanticUndefined,
        # not int).
        if (
            not isinstance(default_resolution, int)
            or not isinstance(default_pe, int)
            or not isinstance(default_patch_size, int)
        ):
            return self

        # Only update PE when the class default is formula-derived from the class
        # default resolution and patch size.
        if default_pe == default_resolution // default_patch_size:
            self.positional_encoding_size = self.resolution // self.patch_size

        return self

    @model_validator(mode="after")
    def _warn_pretrain_compatibility(self) -> "ModelConfig":
        """Warn when overrides are likely to prevent published pretrained weights from loading.

        Three cases:

        1. ``pretrain_weights`` was explicitly set to ``None`` and the variant
           has a non-``None`` default → warn that the model is being initialised from scratch.
        2. ``pretrain_weights`` was explicitly set to a non-``None`` custom path
           → suppress the architecture-override check (we cannot know the architecture stored in a user-supplied
           checkpoint at config time). The load-time partial-load detector in
           :func:`rfdetr.models.weights.load_pretrain_weights` covers this case by inspecting the checkpoint contents
           directly.
        3. ``pretrain_weights`` is the variant's published default → check
           architecture-affecting fields against the variant defaults and emit a single consolidated warning listing
           every load-breaking override.

        The warning class is :class:`PretrainWeightsCompatibilityWarning` (a :class:`UserWarning` subclass), silenceable
        via the standard ``warnings.filterwarnings`` machinery.
        """
        cls = type(self)
        fields_set = self.model_fields_set
        pretrain_user_set = "pretrain_weights" in fields_set

        if pretrain_user_set and self.pretrain_weights is None:
            default_pretrain = cls.model_fields["pretrain_weights"].default
            if default_pretrain is not PydanticUndefined and default_pretrain is not None:
                warnings.warn(
                    f"{cls.__name__} was instantiated with pretrain_weights=None. "
                    f"The model will be initialised from scratch, which typically "
                    f"produces lower accuracy than fine-tuning from the published "
                    f"checkpoint ({default_pretrain!r}).",
                    PretrainWeightsCompatibilityWarning,
                    stacklevel=2,
                )
            return self

        if pretrain_user_set and self.pretrain_weights is not None:
            # Custom checkpoint: architecture overrides may match what the
            # checkpoint was trained with.  Defer to the load-time partial-load
            # detector which can read the file.
            # Exception: when the user explicitly passes the variant's own
            # published-default path string (e.g. ``"rf-detr-nano.pth"``), it
            # IS the published checkpoint — treat it as case 3 so architecture-
            # override checks still apply.  Compare after expand_path so bare
            # filenames resolve to the same cache-dir path as self.pretrain_weights.
            _default_pretrain = cls.model_fields["pretrain_weights"].default
            if _default_pretrain is not None and _default_pretrain is not PydanticUndefined:
                _expanded_default = cls.expand_path(_default_pretrain)
                if self.pretrain_weights != _expanded_default:
                    return self
                # Falls through to case-3 when the user passed the exact variant default.
            else:
                return self

        # `pretrain_weights` is the variant's published default — check
        # architecture overrides against the class defaults.
        # Skip entirely when this variant has no published checkpoint (default
        # is None/PydanticUndefined); warning would reference "(None)" which is
        # misleading and confusing for users of the abstract base config.
        _class_default_pretrain = cls.model_fields["pretrain_weights"].default
        if _class_default_pretrain is None or _class_default_pretrain is PydanticUndefined:
            return self

        overrides: list[tuple[str, Any, Any]] = []

        # Fields that, when explicitly overridden to any value other than the
        # variant default, prevent the published checkpoint from loading cleanly.
        # Includes major architecture knobs, "less obvious" knobs (bbox_reparam,
        # lite_refpoint_refine, layer_norm, two_stage), defense-in-depth for
        # fields that currently raise hard errors (patch_size, segmentation_head),
        # and num_channels (loads via heuristic but result isn't real pretrained
        # weights for the new input domain).
        breaking_fields: tuple[str, ...] = (
            "encoder",
            "hidden_dim",
            "dec_layers",
            "num_windows",
            "sa_nheads",
            "ca_nheads",
            "dec_n_points",
            "out_feature_indexes",
            "projector_scale",
            "bbox_reparam",
            "lite_refpoint_refine",
            "layer_norm",
            "two_stage",
            "patch_size",
            "segmentation_head",
            "num_channels",
        )
        # Fields where only an *increase* above the variant default is load-breaking:
        # num_queries / group_detr add slots whose shape differs — decrease is fine.
        breaking_on_increase: tuple[str, ...] = (
            "num_queries",
            "group_detr",
        )

        for name in breaking_fields:
            if name not in fields_set:
                continue
            field_info = cls.model_fields.get(name)
            if field_info is None or field_info.is_required():
                continue
            default = field_info.default
            if default is PydanticUndefined:
                continue
            current = getattr(self, name)
            if current != default:
                overrides.append((name, current, default))

        for name in breaking_on_increase:
            if name not in fields_set:
                continue
            field_info = cls.model_fields.get(name)
            if field_info is None or field_info.is_required():
                continue
            default = field_info.default
            if default is PydanticUndefined or not isinstance(default, int):
                continue
            current = getattr(self, name)
            if isinstance(current, int) and current > default:
                overrides.append((name, current, default))

        # ``mask_downsample_ratio`` only affects segmentation models — skip on
        # detector-only variants to avoid a misleading "weights won't load" warning.
        if "mask_downsample_ratio" in fields_set and self.segmentation_head:
            _mdr_info = cls.model_fields.get("mask_downsample_ratio")
            if _mdr_info is not None and not _mdr_info.is_required():
                _mdr_default = _mdr_info.default
                if _mdr_default is not PydanticUndefined:
                    _mdr_current = self.mask_downsample_ratio
                    if _mdr_current != _mdr_default:
                        overrides.append(("mask_downsample_ratio", _mdr_current, _mdr_default))

        if overrides:
            default_pretrain = cls.model_fields["pretrain_weights"].default
            lines = "\n".join(
                f"  {name}: {current!r} (variant default: {default!r})" for name, current, default in overrides
            )
            warnings.warn(
                f"{cls.__name__} was instantiated with overrides that differ from the variant "
                f"defaults in ways that prevent the published pretrained weights "
                f"({default_pretrain!r}) from loading correctly:\n"
                f"{lines}\n"
                "Loading the checkpoint with this configuration will leave significant portions "
                "of the model randomly initialised, which typically produces lower accuracy. "
                "To suppress this warning: revert the override(s), pick a variant whose defaults "
                "match, or pass pretrain_weights=None to acknowledge that you intend to train "
                "from scratch.",
                PretrainWeightsCompatibilityWarning,
                stacklevel=2,
            )

        return self

    @field_validator("pretrain_weights", mode="before")
    @classmethod
    def expand_path(cls, v: PathLikeStr | None) -> str | None:
        """Expand and resolve the pretrain_weights path.

        Bare filenames (no directory component, e.g. ``rf-detr-base.pth``) are resolved to the model cache directory so
        weights land in a stable, user-configurable location (``~/.roboflow/models`` by default, or the path set via the
        ``RF_HOME`` environment variable) instead of CWD.

        Paths that already contain a directory separator (e.g. ``~/models/x.pth``, ``/abs/path/x.pth``,
        ``models/x.pth``) are normalised with ``os.path.realpath`` as before.
        """
        if v is None:
            return v
        expanded = os.path.expanduser(os.fspath(v))
        if not os.path.dirname(expanded):
            # Bare filename → use model cache dir so weights don't land in CWD.
            from rfdetr.assets.model_weights import get_model_cache_dir

            return os.path.join(get_model_cache_dir(), expanded)
        return os.path.realpath(expanded)

    @field_validator("device", mode="before")
    @classmethod
    def _normalize_device(cls, v: Any) -> str:
        """Normalize supported device inputs to a canonical torch-style string.

        Args:
            v: Device specifier provided by callers. Supported values are
                ``str`` (for example ``"cpu"``, ``"cuda"``, ``"cuda:1"``) and ``torch.device``.

        Returns:
            Canonical string form of the parsed device (for example ``"cuda:1"``).

        Raises:
            ValueError: If a string value cannot be parsed as a valid torch device.
            ValueError: If ``v`` is not a string or ``torch.device``.
        """
        if isinstance(v, torch.device):
            return str(v)
        if isinstance(v, str):
            try:
                return str(torch.device(v))
            except (TypeError, ValueError, RuntimeError) as exc:
                raise ValueError(f"Invalid device specifier: {v!r}.") from exc
        raise ValueError("device must be a string or torch.device.")


class RFDETRBaseConfig(ModelConfig):
    """The configuration for an RF-DETR Base model."""

    encoder: EncoderName = "dinov2_windowed_small"
    hidden_dim: int = 256
    patch_size: int = 14
    num_windows: int = 4
    dec_layers: int = 3
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_queries: int = 300
    num_select: int = 300
    projector_scale: list[Literal["P3", "P4", "P5"]] = ["P4"]
    out_feature_indexes: list[int] = [2, 5, 8, 11]
    pretrain_weights: PathLikeStr | None = "rf-detr-base.pth"
    resolution: int = 560
    positional_encoding_size: int = 37


class RFDETRLargeDeprecatedConfig(RFDETRBaseConfig):
    """The configuration for an RF-DETR Large model."""

    encoder: EncoderName = "dinov2_windowed_base"
    hidden_dim: int = 384
    sa_nheads: int = 12
    ca_nheads: int = 24
    dec_n_points: int = 4
    projector_scale: list[Literal["P3", "P4", "P5"]] = ["P3", "P5"]
    pretrain_weights: PathLikeStr | None = "rf-detr-large.pth"


class RFDETRNanoConfig(RFDETRBaseConfig):
    """The configuration for an RF-DETR Nano model."""

    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 2
    patch_size: int = 16
    resolution: int = 384
    positional_encoding_size: int = 24
    pretrain_weights: PathLikeStr | None = "rf-detr-nano.pth"


class RFDETRSmallConfig(RFDETRBaseConfig):
    """The configuration for an RF-DETR Small model."""

    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 3
    patch_size: int = 16
    resolution: int = 512
    positional_encoding_size: int = 32
    pretrain_weights: PathLikeStr | None = "rf-detr-small.pth"


class RFDETRMediumConfig(RFDETRBaseConfig):
    """The configuration for an RF-DETR Medium model."""

    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 4
    patch_size: int = 16
    resolution: int = 576
    positional_encoding_size: int = 36
    pretrain_weights: PathLikeStr | None = "rf-detr-medium.pth"


# res 704, ps 16, 2 windows, 4 dec layers, 300 queries, ViT-S basis
class RFDETRLargeConfig(ModelConfig):
    """Configuration for the RF-DETR Large model variant."""

    encoder: Literal["dinov2_windowed_small"] = "dinov2_windowed_small"
    hidden_dim: int = 256
    dec_layers: int = 4
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_windows: int = 2
    patch_size: int = 16
    projector_scale: list[Literal["P4",]] = ["P4"]
    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_classes: int = 90
    positional_encoding_size: int = 704 // 16
    pretrain_weights: PathLikeStr | None = "rf-detr-large-2026.pth"
    resolution: int = 704
    # Explicit so populate_args and _build_args_from_configs agree.
    # ModelConfig does not define these fields; without them the legacy path
    # picks up populate_args defaults (num_select=100) while the PTL path falls
    # back to TrainConfig.num_select (300), causing a postprocess mismatch.
    num_queries: int = 300
    num_select: int = 300


class RFDETRSegPreviewConfig(RFDETRBaseConfig):
    """Configuration for the RF-DETR Segmentation Preview model."""

    segmentation_head: bool = True
    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 4
    patch_size: int = 12
    resolution: int = 432
    positional_encoding_size: int = 36
    num_queries: int = 200
    num_select: int = 200
    pretrain_weights: PathLikeStr | None = "rf-detr-seg-preview.pt"
    num_classes: int = 90


class RFDETRSegNanoConfig(RFDETRBaseConfig):
    """Configuration for the RF-DETR Segmentation Nano model variant."""

    segmentation_head: bool = True
    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_windows: int = 1
    dec_layers: int = 4
    patch_size: int = 12
    resolution: int = 312
    positional_encoding_size: int = 312 // 12
    num_queries: int = 100
    num_select: int = 100
    pretrain_weights: PathLikeStr | None = "rf-detr-seg-nano.pt"
    num_classes: int = 90


class RFDETRSegSmallConfig(RFDETRBaseConfig):
    """Configuration for the RF-DETR Segmentation Small model variant."""

    segmentation_head: bool = True
    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 4
    patch_size: int = 12
    resolution: int = 384
    positional_encoding_size: int = 384 // 12
    num_queries: int = 100
    num_select: int = 100
    pretrain_weights: PathLikeStr | None = "rf-detr-seg-small.pt"
    num_classes: int = 90


class RFDETRSegMediumConfig(RFDETRBaseConfig):
    """Configuration for the RF-DETR Segmentation Medium model variant."""

    segmentation_head: bool = True
    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 5
    patch_size: int = 12
    resolution: int = 432
    positional_encoding_size: int = 432 // 12
    num_queries: int = 200
    num_select: int = 200
    pretrain_weights: PathLikeStr | None = "rf-detr-seg-medium.pt"
    num_classes: int = 90


class RFDETRSegLargeConfig(RFDETRBaseConfig):
    """Configuration for the RF-DETR Segmentation Large model variant."""

    segmentation_head: bool = True
    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 5
    patch_size: int = 12
    resolution: int = 504
    positional_encoding_size: int = 504 // 12
    num_queries: int = 200
    num_select: int = 200
    pretrain_weights: PathLikeStr | None = "rf-detr-seg-large.pt"
    num_classes: int = 90


class RFDETRSegXLargeConfig(RFDETRBaseConfig):
    """Configuration for the RF-DETR Segmentation XLarge model variant."""

    segmentation_head: bool = True
    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 6
    patch_size: int = 12
    resolution: int = 624
    positional_encoding_size: int = 624 // 12
    num_queries: int = 300
    num_select: int = 300
    pretrain_weights: PathLikeStr | None = "rf-detr-seg-xlarge.pt"
    num_classes: int = 90


class RFDETRSeg2XLargeConfig(RFDETRBaseConfig):
    """Configuration for the RF-DETR Segmentation 2XLarge model variant."""

    segmentation_head: bool = True
    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 6
    patch_size: int = 12
    resolution: int = 768
    positional_encoding_size: int = 768 // 12
    num_queries: int = 300
    num_select: int = 300
    pretrain_weights: PathLikeStr | None = "rf-detr-seg-xxlarge.pt"
    num_classes: int = 90


class RFDETRKeypointPreviewConfig(RFDETRBaseConfig):
    """Configuration for the preview keypoint model."""

    use_grouppose_keypoints: bool = True
    dual_projector: bool = True
    dual_projector_kp_only: bool = True
    num_keypoints_per_class: list[int] = [17]
    keypoint_cross_attn: bool = True
    inter_instance_kp_attn: bool = False
    grouppose_keypoint_dim_downscale: int = 1
    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 4
    patch_size: int = 12
    resolution: int = 576
    positional_encoding_size: int = 576 // 12
    num_queries: int = 100
    num_select: int = 100
    pretrain_weights: PathLikeStr | None = "rf-detr-keypoint-preview-xlarge.pth"
    num_classes: int = 90


class TrainConfig(BaseConfig):
    """Training hyperparameters and auto-batching configuration.

    Notes:
        * ``auto_batch_target_effective`` is interpreted as the **per-device**
          effective batch size target, i.e. the number of images seen by a single process in one optimizer step after
          accounting for ``grad_accum_steps``. In multi-GPU / multi-node runs the global effective batch size is
          therefore:

            ``global_effective_batch = auto_batch_target_effective * devices * num_nodes``

          This avoids silently changing behavior when scaling from single-GPU to multi-GPU training.
    """

    # extra="forbid" arms BaseConfig.catch_typo_kwargs so typo'd train() kwargs (e.g. ``epoch`` instead of
    # ``epochs``) raise with a helpful message instead of being silently ignored.  Legacy kwargs handled by
    # RFDETR.train() (resolution/device/callbacks/start_epoch/do_benchmark) are popped before construction.
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", validate_assignment=True)

    lr: float = 1e-4
    lr_encoder: float = 1.5e-4
    batch_size: int | Literal["auto"] = 4
    grad_accum_steps: int = 4
    auto_batch_target_effective: int = 16  # per-device effective batch size target (before devices * num_nodes)
    # Auto-batch probe: worst-case assumptions when batch_size="auto".
    auto_batch_max_targets_per_image: int = 100
    epochs: int = 100
    resume: Optional[PathLikeStr] = None
    lr_drop: int = 100
    checkpoint_interval: int = Field(default=10, ge=1)
    skip_best_epochs: int = Field(default=0, ge=0)
    smooth_alpha: float = 0.0
    warmup_epochs: float = 0.0
    lr_vit_layer_decay: float = 0.8
    lr_component_decay: float = 0.7
    drop_path: float = 0.0
    cls_loss_coef: float = 1.0
    # Detection-vs-keypoint distinction is derived by callers via `include_keypoints`, not
    # stored on this field. See rfdetr.datasets.transforms.AlbumentationsWrapper.from_config
    # for the None/[]/[...] tri-state contract applied at the augmentation-pipeline boundary.
    keypoint_flip_pairs: list[int] = Field(default_factory=list)
    keypoint_l1_loss_coef: float = 0
    keypoint_findable_loss_coef: float = 0
    keypoint_visible_loss_coef: float = 0
    keypoint_nll_loss_coef: float = 0
    keypoint_oks_sigmas: list[float] | None = None
    dataset_file: Literal["coco", "o365", "roboflow", "yolo"] = "roboflow"
    square_resize_div_64: bool = True
    dataset_dir: PathLikeStr | None
    output_dir: PathLikeStr = "output"
    multi_scale: MultiScaleConfig = Field(default_factory=MultiScaleConfig)
    ema: EmaConfig = Field(default_factory=EmaConfig)
    num_workers: int = 2
    weight_decay: float = 1e-4
    optimizer: Literal["adamw", "sgd", "muadamw", "musgd"] = "adamw"
    optimizer_config: OptimizerConfig = Field(default_factory=OptimizerConfig)
    stal: StalConfig = Field(default_factory=StalConfig)
    amp_dtype: Literal["auto", "bf16", "fp16"] = Field(
        default="auto",
        description=(
            "Mixed-precision autocast dtype. "
            "'auto' selects bf16-mixed on Ampere+ CUDA, fp16 otherwise. "
            "'bf16' forces bfloat16 (falls back to fp16 with a warning if unsupported). "
            "'fp16' forces fp16. "
            "Has no effect when model_config.amp=False or when training on CPU."
        ),
    )
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    progress_bar: Optional[Literal["tqdm", "rich"]] = None  # Progress bar style: "rich", "tqdm", or None to disable.
    tensorboard: bool = True
    wandb: bool = False
    mlflow: bool | Dict[str, Any] = False
    mlflow_tracking_uri: Optional[str] = Field(default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI"))
    mlflow_log_artifacts: bool = True
    mlflow_log_system_metrics: bool = True
    clearml: bool = False  # Not yet implemented — reserved for future use.
    project: str | None = None
    run: str | None = None
    class_names: list[str] | None = None
    run_test: bool = False
    validate_before_fit: bool = False
    segmentation_head: bool = False
    eval_max_dets: int = 500
    eval_interval: int = 1
    eval_ema_only: bool = False
    eval_masks_head_resolution: bool = False
    log_per_class_metrics: bool = True
    validation_batch_size: Optional[int] = Field(default=None, ge=1)
    validation_mode: Literal[
        "standard",
        "sahi",
        "asahi",
        "gsahi",
    ] = "standard"
    sahi: SahiValidationConfig = Field(default_factory=SahiValidationConfig)
    asahi: AsahiValidationConfig = Field(default_factory=AsahiValidationConfig)
    gsahi: GsahiValidationConfig = Field(default_factory=GsahiValidationConfig)
    validation_score_threshold: float = Field(default=0.001, ge=0.0, le=1.0)
    validation_max_predictions: int = Field(default=500, ge=1)
    pre_resize_aug_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    aug_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    eval_pre_resize_aug_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    eval_aug_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    augmentation_backend: AugmentationBackend | Literal["cpu", "auto"] = "cpu"
    save_dataset_grids: bool = False
    save_prediction_grids: bool = False
    notes: Optional[Any] = Field(
        default=None,
        description=(
            "User-defined provenance metadata embedded in best-model .pth checkpoints "
            "under checkpoint['args']['notes'] and in exported ONNX files under the "
            "'rfdetr_notes' metadata property. Accepts any JSON-serialisable value "
            "(string, dict, list, int, float, bool). String values are stored verbatim; "
            "all other types are JSON-encoded."
        ),
    )

    @field_validator("augmentation_backend", mode="before")
    @classmethod
    def _coerce_augmentation_backend(cls, value: Any) -> Any:
        """Map legacy augmentation backend aliases to current names."""
        if isinstance(value, str):
            return _LEGACY_AUGMENTATION_BACKEND_ALIASES.get(value, value)
        return value

    @field_serializer("augmentation_backend")
    def _serialize_augmentation_backend(self, value: AugmentationBackend | str) -> str:
        """Serialize augmentation backend enums as JSON-safe strings."""
        return value.value if isinstance(value, AugmentationBackend) else value

    @property
    def use_ema(self) -> bool:
        """Return the legacy flat alias for structured EMA enablement."""
        return self.ema.enabled

    @property
    def ema_decay(self) -> float:
        """Return the legacy flat alias for structured EMA decay."""
        return self.ema.decay

    @property
    def ema_tau(self) -> int:
        """Return the legacy flat alias for structured EMA warmup."""
        return self.ema.tau

    @property
    def ema_update_interval(self) -> int:
        """Return the legacy flat alias for the structured EMA update interval."""
        return self.ema.update_interval

    @property
    def auto_batch_ema_headroom(self) -> float:
        """Return the legacy flat alias for structured EMA auto-batch headroom."""
        return self.ema.auto_batch_headroom

    @model_validator(mode="before")
    @classmethod
    def _expand_nested_ema_config(cls, data: Any) -> Any:
        """Normalize legacy EMA fields into the structured config.

        Args:
            data: Raw config data received by Pydantic.

        Returns:
            Config data with supported EMA keys under ``ema``.
        """
        if not isinstance(data, dict):
            return data
        expanded = dict(data)
        raw_ema = expanded.get("ema")
        legacy_mapping = {
            "use_ema": "enabled",
            "ema_decay": "decay",
            "ema_tau": "tau",
            "ema_update_interval": "update_interval",
            "auto_batch_ema_headroom": "auto_batch_headroom",
        }

        if isinstance(raw_ema, bool):
            normalized: dict[str, Any] = {"enabled": raw_ema}
        elif isinstance(raw_ema, dict):
            normalized = dict(raw_ema)
        else:
            normalized = {}

        for legacy_key, nested_key in legacy_mapping.items():
            if legacy_key in expanded:
                legacy_value = expanded.pop(legacy_key)
                if nested_key not in normalized:
                    normalized[nested_key] = legacy_value

        if normalized:
            expanded["ema"] = normalized
        return expanded

    @model_validator(mode="before")
    @classmethod
    def _expand_nested_multi_scale_config(cls, data: Any) -> Any:
        """Normalize legacy multi-scale fields into the structured config.

        Args:
            data: Raw config data received by Pydantic.

        Returns:
            Config data with supported multi-scale keys under ``multi_scale``.
        """
        if not isinstance(data, dict):
            return data
        expanded = dict(data)
        raw_multi_scale = expanded.get("multi_scale")
        legacy_mapping = {
            "expanded_scales": "expanded_scales",
            "multi_scale_min_offset": "min_offset",
            "multi_scale_max_offset": "max_offset",
            "do_random_resize_via_padding": "random_resize_via_padding",
        }

        if isinstance(raw_multi_scale, bool):
            normalized: dict[str, Any] = {"enabled": raw_multi_scale}
        elif isinstance(raw_multi_scale, dict):
            normalized = dict(raw_multi_scale)
        else:
            normalized = {}

        for legacy_key, nested_key in legacy_mapping.items():
            if legacy_key in expanded:
                legacy_value = expanded.pop(legacy_key)
                if nested_key not in normalized:
                    normalized[nested_key] = legacy_value

        if normalized:
            expanded["multi_scale"] = normalized
        return expanded

    @model_validator(mode="before")
    @classmethod
    def _expand_nested_optimizer_config(cls, data: Any) -> Any:
        """Keep backward-compatible nested optimizer selector syntax.

        Args:
            data: Raw config data received by Pydantic.

        Returns:
            Config data with ``optimizer.name`` moved to the selector field when present.
        """
        if not isinstance(data, dict):
            return data
        optimizer_config = data.get("optimizer_config")
        optimizer = data.get("optimizer")
        if not isinstance(optimizer, dict):
            return data
        expanded = dict(data)
        if "name" in optimizer:
            expanded["optimizer"] = optimizer["name"]
        optimizer_without_name = {key: value for key, value in optimizer.items() if key != "name"}
        if optimizer_without_name:
            if isinstance(optimizer_config, dict):
                expanded["optimizer_config"] = {**optimizer_without_name, **optimizer_config}
            else:
                expanded["optimizer_config"] = optimizer_without_name
        return expanded

    @model_validator(mode="after")
    def _sync_optimizer_config_aliases(self) -> "TrainConfig":
        """Synchronize legacy flat optimizer fields with the structured optimizer config."""
        optimizer_fields_set = "optimizer_config" in self.model_fields_set
        scheduler_fields_set = optimizer_fields_set and "scheduler" in self.optimizer_config.model_fields_set
        scheduler = self.optimizer_config.scheduler
        explicit_nonmanaged_scheduler = (
            "lr_scheduler" in self.model_fields_set and not _is_managed_scheduler_name(self.lr_scheduler)
        )
        synchronize_scheduler = not explicit_nonmanaged_scheduler and (
            scheduler_fields_set or _is_managed_scheduler_name(self.lr_scheduler)
        )

        if not optimizer_fields_set:
            self.optimizer_config.lr = self.lr
            self.optimizer_config.lr_encoder = self.lr_encoder
            self.optimizer_config.weight_decay = self.weight_decay
            if synchronize_scheduler:
                scheduler.name = self.lr_scheduler
                scheduler.min_factor = self.lr_min_factor
                scheduler.warmup_epochs = self.warmup_epochs
                scheduler.drop_epoch = self.lr_drop
            return self

        if "lr" not in self.optimizer_config.model_fields_set and "lr" in self.model_fields_set:
            self.optimizer_config.lr = self.lr
        if "lr_encoder" not in self.optimizer_config.model_fields_set and "lr_encoder" in self.model_fields_set:
            self.optimizer_config.lr_encoder = self.lr_encoder
        if "weight_decay" not in self.optimizer_config.model_fields_set and "weight_decay" in self.model_fields_set:
            self.optimizer_config.weight_decay = self.weight_decay

        if synchronize_scheduler and not scheduler_fields_set:
            if "lr_scheduler" in self.model_fields_set:
                scheduler.name = self.lr_scheduler
            if "lr_min_factor" in self.model_fields_set:
                scheduler.min_factor = self.lr_min_factor
            if "warmup_epochs" in self.model_fields_set:
                scheduler.warmup_epochs = self.warmup_epochs
            if "lr_drop" in self.model_fields_set:
                scheduler.drop_epoch = self.lr_drop
        elif synchronize_scheduler:
            if "name" not in scheduler.model_fields_set and "lr_scheduler" in self.model_fields_set:
                scheduler.name = self.lr_scheduler
            if "min_factor" not in scheduler.model_fields_set and "lr_min_factor" in self.model_fields_set:
                scheduler.min_factor = self.lr_min_factor
            if "warmup_epochs" not in scheduler.model_fields_set and "warmup_epochs" in self.model_fields_set:
                scheduler.warmup_epochs = self.warmup_epochs
            if "drop_epoch" not in scheduler.model_fields_set and "lr_drop" in self.model_fields_set:
                scheduler.drop_epoch = self.lr_drop

        object.__setattr__(self, "lr", self.optimizer_config.lr)
        object.__setattr__(self, "lr_encoder", self.optimizer_config.lr_encoder)
        object.__setattr__(self, "weight_decay", self.optimizer_config.weight_decay)
        if synchronize_scheduler:
            object.__setattr__(self, "lr_scheduler", scheduler.name)
            object.__setattr__(self, "lr_min_factor", scheduler.min_factor)
            object.__setattr__(self, "warmup_epochs", scheduler.warmup_epochs)
            object.__setattr__(self, "lr_drop", scheduler.drop_epoch)
        return self

    @model_validator(mode="before")
    @classmethod
    def _expand_nested_early_stopping_config(cls, data: Any) -> Any:
        """Normalize legacy early-stopping forms into the structured config.

        Args:
            data: Raw config data received by Pydantic.

        Returns:
            Config data with supported early-stopping keys under ``early_stopping``.
        """
        if not isinstance(data, dict):
            return data
        expanded = dict(data)
        raw_early_stopping = expanded.get("early_stopping")
        early_stopping_config = expanded.pop("early_stopping_config", None)
        legacy_mapping = {
            "early_stopping_patience": "patience",
            "early_stopping_min_delta": "min_delta",
            "early_stopping_use_ema": "use_ema",
        }

        if isinstance(raw_early_stopping, bool):
            normalized: dict[str, Any] = {"enabled": raw_early_stopping}
        elif isinstance(raw_early_stopping, dict):
            normalized = dict(raw_early_stopping)
        else:
            normalized = {}

        if isinstance(early_stopping_config, dict):
            normalized = {**normalized, **early_stopping_config}

        for legacy_key, nested_key in legacy_mapping.items():
            if legacy_key in expanded:
                normalized[nested_key] = expanded.pop(legacy_key)

        if normalized:
            expanded["early_stopping"] = normalized
        return expanded

    @model_validator(mode="before")
    @classmethod
    def _expand_nested_mlflow_config(cls, data: Any) -> Any:
        """Expand a nested ``mlflow`` YAML block into flat TrainConfig fields.

        Args:
            data: Raw config data received by Pydantic.

        Returns:
            Config data with supported ``mlflow`` keys mapped to internal fields.
        """
        if not isinstance(data, dict):
            return data
        mlflow = data.get("mlflow")
        if not isinstance(mlflow, dict):
            return data
        expanded = dict(data)
        mapping = {
            "enabled": "mlflow",
            "tracking_uri": "mlflow_tracking_uri",
            "log_artifacts": "mlflow_log_artifacts",
            "log_system_metrics": "mlflow_log_system_metrics",
        }
        for nested_key, field_name in mapping.items():
            if nested_key in mlflow:
                expanded[field_name] = mlflow[nested_key]
        if "enabled" not in mlflow:
            expanded["mlflow"] = True
        return expanded

    @model_validator(mode="before")
    @classmethod
    def _normalize_validation_method_configs(cls, data: Any) -> Any:
        """Normalize validation method aliases while keeping method options grouped.

        Args:
            data: Raw config data received by Pydantic.

        Returns:
            Config data with validation method settings stored under ``sahi``, ``asahi``, and ``gsahi``.
        """
        if not isinstance(data, dict):
            return data
        expanded = dict(data)
        if expanded.get("validation_mode") in {"guided_sahi", "dual_sahi", "coarse_to_fine", "gois"}:
            expanded["validation_mode"] = "gsahi"

        if isinstance(expanded.get("sahi"), dict):
            sahi = dict(expanded["sahi"])
            confidence_threshold = sahi.pop("confidence_threshold", None)
            if confidence_threshold is not None:
                if "validation_score_threshold" not in expanded:
                    expanded["validation_score_threshold"] = confidence_threshold
                sahi.setdefault("score_threshold", confidence_threshold)
            expanded["sahi"] = sahi

        alias_block = next(
            (
                expanded.pop(alias)
                for alias in ("guided_sahi", "dual_sahi", "coarse_to_fine", "gois")
                if isinstance(expanded.get(alias), dict)
            ),
            None,
        )
        if alias_block is not None and "gsahi" not in expanded:
            expanded["gsahi"] = alias_block

        return expanded

    @model_validator(mode="after")
    def _warn_deprecated_train_config_fields(self) -> "TrainConfig":
        """Emit DeprecationWarning for fields whose ownership is moving to ModelConfig.

        The following fields are duplicated between ``ModelConfig`` and ``TrainConfig`` but ``ModelConfig`` is the
        authoritative source (Item #3, v1.7.0).  Setting them on ``TrainConfig`` is deprecated.  The fields will be
        removed in v1.9.0.

        - ``group_detr``: query group count is an architecture decision → ``ModelConfig``
        - ``ia_bce_loss``: loss type is tied to architecture family → ``ModelConfig``
        - ``segmentation_head``: architecture flag → ``ModelConfig``
        - ``num_select``: postprocessor count is an architecture decision → ``ModelConfig``
        """
        _deprecated = ("group_detr", "ia_bce_loss", "segmentation_head", "num_select")
        for field in _deprecated:
            if field in self.model_fields_set:
                # stacklevel=2 points into Pydantic internals; unavoidable with
                # @model_validator(mode="after") in Pydantic v2.
                warnings.warn(
                    f"TrainConfig.{field} is deprecated since v1.7.0 and will be removed in v1.9.0. "
                    f"Set {field} on ModelConfig instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        return self

    @field_validator("progress_bar", mode="before")
    @classmethod
    def _coerce_legacy_progress_bar(cls, value: Any) -> Any:
        """Normalize legacy boolean progress_bar values to the new string/None representation.

        This preserves compatibility with older configs where ``progress_bar`` was a bool.
        """
        if isinstance(value, bool):
            return "tqdm" if value else None
        return value

    @field_validator("amp_dtype", mode="before")
    @classmethod
    def _coerce_amp_dtype(cls, value: Any) -> Any:
        """Fall back to ``'auto'`` (with a warning) for an unrecognised or wrong-typed ``amp_dtype``.

        Mixed precision is a best-effort speed/memory optimisation, so an invalid request degrades to the auto-selected
        dtype rather than failing the whole training run.
        """
        if value not in ("auto", "bf16", "fp16"):
            # stacklevel=2 points into Pydantic internals; unavoidable with @field_validator in Pydantic v2.
            warnings.warn(
                f"Unknown amp_dtype={value!r}; expected one of 'auto', 'bf16', 'fp16'. Falling back to 'auto'.",
                UserWarning,
                stacklevel=2,
            )
            return "auto"
        return value

    # Promoted from populate_args() — PTL migration (T4-2).
    # device is intentionally absent: PTL auto-detects accelerator via Trainer(accelerator="auto").
    accelerator: str = "auto"
    clip_max_norm: float = 0.1
    seed: int | None = None
    sync_bn: bool = False
    # strategy maps to PTL Trainer(strategy=...). Common values: "auto", "ddp",
    # "ddp_spawn", "fsdp", "deepspeed". Invalid values surface as PTL errors.
    strategy: str = "auto"
    devices: int | str = 1
    # num_nodes maps to PTL Trainer(num_nodes=...) for multi-machine training.
    # Single-machine DDP users should leave this at 1 (the default).
    num_nodes: int = 1
    fp16_eval: bool = False
    lr_scheduler: str | Callable[..., SchedulerType] = "step"
    lr_scheduler_kwargs: dict[str, Any] = Field(default_factory=dict)
    lr_scheduler_interval: Literal["step", "epoch"] = "step"
    lr_scheduler_monitor: str = "val/loss"
    # Deprecated aux LR knobs — kept for one cycle; folded into lr_scheduler_kwargs (see _map_deprecated_lr_fields).
    lr_min_factor: float = 0.0
    optimizer: str | Callable[..., Optimizer] = "adamw"
    optimizer_kwargs: dict[str, Any] = Field(default_factory=dict)
    dont_save_weights: bool = False
    # PTL runtime/perf tuning knobs.
    train_log_sync_dist: bool = False
    train_log_on_step: bool = False
    compute_train_metrics: bool = False
    compute_val_loss: bool = True
    compute_test_loss: bool = True
    pin_memory: bool | None = None
    persistent_workers: bool | None = None
    prefetch_factor: int | None = None

    @field_validator("batch_size", mode="after")
    @classmethod
    def validate_batch_size(cls, v: int | Literal["auto"]) -> int | Literal["auto"]:
        """Validate batch_size is a positive integer or the literal 'auto'."""
        if v == "auto":
            return v
        if v < 1:
            raise ValueError("batch_size must be >= 1, or 'auto'.")
        return v

    @field_validator(
        "grad_accum_steps", "auto_batch_target_effective", "auto_batch_max_targets_per_image", mode="after"
    )
    @classmethod
    def validate_positive_train_steps(cls, v: int) -> int:
        """Validate accumulation, target-effective batch, and max targets are >= 1."""
        if v < 1:
            raise ValueError(
                "grad_accum_steps, auto_batch_target_effective, and auto_batch_max_targets_per_image must be >= 1."
            )
        return v

    @field_validator("smooth_alpha", mode="after")
    @classmethod
    def validate_smooth_alpha(cls, v: float) -> float:
        """Validate smooth_alpha is in [0.0, 1.0)."""
        if not (0.0 <= v < 1.0):
            raise ValueError("smooth_alpha must be in [0.0, 1.0).")
        return v

    @field_validator("eval_interval", mode="after")
    @classmethod
    def validate_positive_intervals(cls, v: int) -> int:
        """Validate interval fields are >= 1."""
        if v < 1:
            raise ValueError("Interval fields must be >= 1.")
        return v

    @model_validator(mode="before")
    @classmethod
    def _desugar_callable_optimizer(cls, data: Any) -> Any:
        """Desugar a reconstructable callable optimizer into its serializable string form.

        A callable ``optimizer`` (a class or ``functools.partial``) that can be imported is rewritten to a dotted import
        path plus ``optimizer_kwargs`` so the config round-trips through ``training_config.json``. User-supplied
        ``optimizer_kwargs`` are ignored for callable optimizers (bake arguments into the callable instead). Non-
        reconstructable callables are kept as-is and only warned about.
        """
        if not isinstance(data, dict):
            return data
        optimizer = data.get("optimizer")
        if optimizer is None or isinstance(optimizer, str) or not callable(optimizer):
            return data

        if data.get("optimizer_kwargs"):
            warnings.warn(
                "optimizer_kwargs is ignored when optimizer is a callable; bake arguments into the "
                "callable (for example with functools.partial) instead.",
                stacklevel=2,
            )

        path, kwargs, reason = _desugar_optimizer_callable(optimizer)
        if reason is None:
            data["optimizer"] = path
            data["optimizer_kwargs"] = kwargs
        else:
            data["optimizer_kwargs"] = {}
            label = getattr(optimizer, "__qualname__", None) or repr(optimizer)
            warnings.warn(
                f"optimizer callable {label!r} cannot be saved to training_config.json and restored: "
                f"{reason}. Training proceeds with the in-memory callable; only saved-config "
                "reproducibility is affected.",
                stacklevel=2,
            )
        return data

    @field_validator("optimizer", mode="after")
    @classmethod
    def validate_optimizer_name(cls, v: str | Callable[..., Optimizer]) -> str | Callable[..., Optimizer]:
        """Validate a string optimizer: a bare name must be a native torch.optim optimizer."""
        if not isinstance(v, str):
            return v
        optimizer = v.strip()
        if not optimizer:
            raise ValueError("optimizer must be a non-empty string.")
        if optimizer.lower() in {"muadamw", "musgd"}:
            return optimizer.lower()
        # Bare short names must resolve to a torch.optim optimizer (checked eagerly).
        # Dotted import paths are validated lazily at train start (the module may be optional).
        if "." not in optimizer:
            _resolve_native_optimizer(optimizer)
        return optimizer

    @model_validator(mode="after")
    def validate_eval_ema_only(self) -> "TrainConfig":
        """``eval_ema_only`` has no EMA model to evaluate without ``use_ema=True``."""
        if self.eval_ema_only and not self.use_ema:
            raise ValueError("eval_ema_only=True requires use_ema=True.")
        return self

    @model_validator(mode="after")
    def validate_optimizer_kwargs(self) -> "TrainConfig":
        """Reserved optimizer kwargs are only rejected for managed (short-name) optimizers."""
        if _is_managed_optimizer_name(self.optimizer):
            reserved_present = _OPTIMIZER_MANAGED_KWARGS.intersection(self.optimizer_kwargs)
            if reserved_present:
                reserved = ", ".join(sorted(reserved_present))
                raise ValueError(f"optimizer_kwargs cannot include RF-DETR-managed key(s): {reserved}.")
        return self

    @model_validator(mode="after")
    def validate_lr_scheduler_kwargs(self) -> "TrainConfig":
        """Reject unknown ``lr_scheduler_kwargs`` keys for the managed ``"step"`` / ``"cosine"`` presets.

        Managed presets consume only ``min_factor`` and ``lr_drop``; any other key would be silently ignored, so surface
        it as an error (mirroring ``validate_optimizer_kwargs``). Explicit schedulers forward their kwargs verbatim to
        the constructor and are left unchecked here.
        """
        if _is_managed_scheduler_name(self.lr_scheduler):
            unknown = set(self.lr_scheduler_kwargs) - _MANAGED_SCHEDULER_KWARGS
            if unknown:
                allowed = ", ".join(sorted(_MANAGED_SCHEDULER_KWARGS))
                unknown_keys = ", ".join(sorted(unknown))
                raise ValueError(
                    f"lr_scheduler_kwargs for a managed preset ({self.lr_scheduler!r}) accepts only "
                    f"{{{allowed}}}; unknown key(s): {unknown_keys}."
                )
        return self

    @model_validator(mode="before")
    @classmethod
    def _map_deprecated_lr_fields(cls, data: Any) -> Any:
        """Fold the deprecated ``lr_drop`` / ``lr_min_factor`` fields into ``lr_scheduler_kwargs``.

        These loose knobs are deprecated in favor of ``lr_scheduler_kwargs``. When either is supplied with a non-default
        value for a managed preset (``"step"`` / ``"cosine"``), it is copied into ``lr_scheduler_kwargs`` (without
        overriding an explicit kwarg) and a ``FutureWarning`` is emitted. Default values are ignored silently so round-
        tripping a dumped config (which always carries these fields) never warns. For explicit (dotted-path / callable)
        schedulers the deprecated fields are preset-specific and left untouched.
        """
        if not isinstance(data, dict):
            return data
        # Only managed presets consume these knobs; default lr_scheduler ("step") is managed.
        if not _is_managed_scheduler_name(data.get("lr_scheduler", "step")):
            # Explicit / callable scheduler: these preset knobs are inert. Warn (never fold) when a non-default
            # value is set so a stale lr_drop / lr_min_factor carried over from a managed config is not silently
            # dropped — a reproducibility footgun when migrating a saved config to an explicit scheduler.
            for field_name in _DEPRECATED_LR_FIELD_KWARGS:
                if field_name in data and data[field_name] != cls.model_fields[field_name].default:
                    warnings.warn(
                        f"{field_name} is ignored for the explicit (non-managed) lr_scheduler "
                        f"{data.get('lr_scheduler')!r}; it only applies to the managed 'step'/'cosine' presets.",
                        FutureWarning,
                        stacklevel=2,
                    )
            return data
        kwargs = dict(data.get("lr_scheduler_kwargs") or {})
        for field_name, kwarg_name in _DEPRECATED_LR_FIELD_KWARGS.items():
            if field_name not in data:
                continue
            # A default value (common when reloading a dumped config) is a no-op: the managed builder falls back to
            # the same default. Skip silently so serialization round-trips don't emit spurious deprecation warnings.
            if data[field_name] == cls.model_fields[field_name].default:
                continue
            # Already migrated: a dumped config carries both the top-level field and the folded kwarg. If the kwarg
            # already holds this value, the field adds nothing — skip silently so migrated-config reloads never warn.
            if kwargs.get(kwarg_name) == data[field_name]:
                continue
            # Both set to different values: the kwarg wins (setdefault below is a no-op). Say so explicitly rather
            # than implying the deprecated field was migrated, which would mislead — the field value is discarded.
            if kwarg_name in kwargs:
                warnings.warn(
                    f"{field_name}={data[field_name]!r} is ignored because lr_scheduler_kwargs already sets "
                    f"{kwarg_name!r}={kwargs[kwarg_name]!r} (the kwarg wins); remove the deprecated {field_name}.",
                    FutureWarning,
                    stacklevel=2,
                )
                continue
            warnings.warn(
                f"{field_name} is deprecated; pass it via lr_scheduler_kwargs={{'{kwarg_name}': ...}} instead.",
                FutureWarning,
                stacklevel=2,
            )
            kwargs[kwarg_name] = data[field_name]
        if kwargs:
            data["lr_scheduler_kwargs"] = kwargs
        return data

    @model_validator(mode="before")
    @classmethod
    def _desugar_callable_lr_scheduler(cls, data: Any) -> Any:
        """Desugar a reconstructable callable lr_scheduler into its serializable string form.

        A callable ``lr_scheduler`` (a class or ``functools.partial``) that can be imported is rewritten to a dotted
        import path plus ``lr_scheduler_kwargs`` so the config round-trips through ``training_config.json``. User-
        supplied ``lr_scheduler_kwargs`` are ignored for callable schedulers (bake arguments into the callable instead).
        Non-reconstructable callables are kept as-is and only warned about.
        """
        if not isinstance(data, dict):
            return data
        lr_scheduler = data.get("lr_scheduler")
        if lr_scheduler is None or isinstance(lr_scheduler, str) or not callable(lr_scheduler):
            return data

        if data.get("lr_scheduler_kwargs"):
            warnings.warn(
                "lr_scheduler_kwargs is ignored when lr_scheduler is a callable; bake arguments into the "
                "callable (for example with functools.partial) instead.",
                stacklevel=2,
            )

        path, kwargs, reason = _desugar_scheduler_callable(lr_scheduler)
        if reason is None:
            data["lr_scheduler"] = path
            data["lr_scheduler_kwargs"] = kwargs
        else:
            data["lr_scheduler_kwargs"] = {}
            label = getattr(lr_scheduler, "__qualname__", None) or repr(lr_scheduler)
            warnings.warn(
                f"lr_scheduler callable {label!r} cannot be saved to training_config.json and restored: "
                f"{reason}. Training proceeds with the in-memory callable; only saved-config "
                "reproducibility is affected.",
                stacklevel=2,
            )
        return data

    @field_validator("lr_scheduler", mode="after")
    @classmethod
    def validate_lr_scheduler_name(cls, v: str | Callable[..., SchedulerType]) -> str | Callable[..., SchedulerType]:
        """Validate a string lr_scheduler: a bare name must be a managed preset, else use a dotted path."""
        if not isinstance(v, str):
            return v
        lr_scheduler = v.strip()
        if not lr_scheduler:
            raise ValueError("lr_scheduler must be a non-empty string.")
        # Bare names must be a managed preset; dotted import paths are validated lazily at train start.
        if "." not in lr_scheduler and not _is_managed_scheduler_name(lr_scheduler):
            presets = ", ".join(sorted(_MANAGED_SCHEDULER_PRESETS))
            raise ValueError(
                f"Unknown lr_scheduler {v!r}. Bare names must be a managed preset ({presets}); "
                "use a full dotted import path (e.g. 'torch.optim.lr_scheduler.StepLR') or a callable "
                "for anything else."
            )
        return lr_scheduler

    @field_validator("prefetch_factor", mode="after")
    @classmethod
    def validate_prefetch_factor(cls, v: int | None) -> int | None:
        """Validate prefetch_factor is None or >= 1."""
        if v is not None and v < 1:
            raise ValueError("prefetch_factor must be >= 1 when provided.")
        return v

    @field_validator("dataset_dir", "output_dir", mode="before")
    @classmethod
    def expand_paths(cls, v: PathLikeStr | None) -> str | None:
        """Expand and normalize dataset/output directory paths via ``os.fspath`` → ``expanduser`` → ``realpath``."""
        if v is None:
            return v
        return os.path.realpath(os.path.expanduser(os.fspath(v)))

    @field_validator("resume", mode="before")
    @classmethod
    def _coerce_resume_path(cls, v: PathLikeStr | None) -> str | None:
        """Normalise the resume checkpoint value to ``str`` without resolving it.

        Unlike ``dataset_dir``/``output_dir``, ``resume`` is forwarded verbatim to PyTorch Lightning's
        ``trainer.fit(ckpt_path=...)``, which also accepts sentinel values such as ``"last"``. Running
        ``os.path.realpath`` would rewrite those sentinels into spurious absolute paths, so this validator only coerces
        the type (``Path`` -> ``str``) and leaves the value untouched.
        """
        if v is None:
            return v
        return os.fspath(v)


class SegmentationTrainConfig(TrainConfig):
    """Training configuration for instance segmentation models.

    Extends :class:`TrainConfig` with segmentation-specific loss coefficients.

    Attributes:
        mask_point_sample_ratio: Number of points sampled per mask for point-based
            mask loss computation.
        mask_ce_loss_coef: Cross-entropy loss weight for mask prediction.
        mask_dice_loss_coef: Dice loss weight for mask prediction.
        cls_loss_coef: Classification loss weight. Defaults to ``1.0`` to match the
            effective pre-v1.7 value (the v1.7 TrainConfig ownership migration
            silently activated a dormant ``5.0``; this field restores the correct
            weight). To reproduce pre-fix segmentation behaviour pass
            ``cls_loss_coef=5.0`` explicitly.
    """

    mask_point_sample_ratio: int = 16
    mask_ce_loss_coef: float = 5.0
    mask_dice_loss_coef: float = 5.0
    cls_loss_coef: float = 1.0


class KeypointTrainConfig(TrainConfig):
    """Training configuration for keypoint detection models.

    Extends :class:`TrainConfig` with keypoint-specific loss coefficients and
    metric-smoothing defaults tuned for the NLL-Cholesky keypoint head, which
    produces noisy per-epoch OKS metrics during early fine-tuning.

    Attributes:
        cls_loss_coef: Classification loss weight.
        keypoint_l1_loss_coef: L1 regression loss weight for keypoint coordinates.
        keypoint_findable_loss_coef: Loss weight for the keypoint visibility head.
        keypoint_visible_loss_coef: Loss weight for the keypoint visibility score.
        keypoint_nll_loss_coef: NLL-Cholesky loss weight. Restored to ``1.0`` to
            align with the other keypoint loss terms (``keypoint_l1_loss_coef``,
            ``keypoint_findable_loss_coef``, ``keypoint_visible_loss_coef``).
            Previously set to ``0.5`` to dampen OKS@75 oscillation; reverted as
            the under-weighting was not beneficial in practice.
        smooth_alpha: EMA smoothing factor for :class:`BestModelCallback` metric
            comparison. Overrides the :class:`TrainConfig` default of ``0.0``
            (disabled) to ``0.5``, which balances responsiveness and noise
            suppression for noisy keypoint mAP curves.
        skip_best_epochs: Number of epochs to skip before checkpoint selection begins.
            Overrides the :class:`TrainConfig` default of ``0`` to ``10`` because
            ``val/keypoint_map_50_95`` under the NLL-Cholesky loss is noisy in early
            fine-tuning and can lock checkpoint selection to a transient peak.
    """

    cls_loss_coef: float = 2.0  # TODO: verify empirically before final release; ported as-is from internal recipe.
    keypoint_l1_loss_coef: float = 1
    keypoint_findable_loss_coef: float = 1
    keypoint_visible_loss_coef: float = 1
    keypoint_nll_loss_coef: float = 1.0
    smooth_alpha: float = 0.5
    skip_best_epochs: int = Field(default=10, ge=0)


RFDETRModelConfig: TypeAlias = Union[
    ModelConfig,
    RFDETRBaseConfig,
    RFDETRLargeDeprecatedConfig,
    RFDETRNanoConfig,
    RFDETRSmallConfig,
    RFDETRMediumConfig,
    RFDETRLargeConfig,
    RFDETRSegPreviewConfig,
    RFDETRSegNanoConfig,
    RFDETRSegSmallConfig,
    RFDETRSegMediumConfig,
    RFDETRSegLargeConfig,
    RFDETRSegXLargeConfig,
    RFDETRSeg2XLargeConfig,
    RFDETRKeypointPreviewConfig,
]
RFDETRTrainConfig: TypeAlias = Union[TrainConfig, SegmentationTrainConfig, KeypointTrainConfig]
