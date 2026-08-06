# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Common tiled validation primitives for RF-DETR."""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torchvision.ops import nms

from rfdetr.utilities.tensors import NestedTensor, nested_tensor_from_tensor_list

MergeMetric = Literal["iou", "ios", "diou", "cdn"]
TileInputDType = Literal["auto", "fp32", "bf16", "fp16"]
TileMemoryFormat = Literal["contiguous", "channels_last"]
WindowResizePolicy = Literal[
    "aspect_longest_side",
    "aspect_valid_target",
    "square_stretch",
    "square_letterbox",
    "letterbox_valid_target",
    "letterbox_canvas_target",
    "square_context",
]

SOURCE_STAGE_TILE = 0
SOURCE_STAGE_FULL = 1
SOURCE_STAGE_COARSE = 2
SOURCE_STAGE_FINE = 3
SOURCE_STAGE_ASAHI = 4


@dataclass(frozen=True)
class TileWindow:
    """Spatial crop window in ``xyxy`` image coordinates.

    Attributes:
        x0: Inclusive left coordinate.
        y0: Inclusive top coordinate.
        x1: Exclusive right coordinate.
        y1: Exclusive bottom coordinate.
    """

    x0: int
    y0: int
    x1: int
    y1: int


@dataclass
class TiledTimingStats:
    """Optional timing and tensor-shape counters for tiled inference."""

    enabled: bool = False
    device: torch.device | None = None
    buckets: dict[str, float] | None = None
    forward_calls: int = 0
    tile_batch_sizes: list[int] | None = None
    tile_spatial_shapes: list[tuple[int, int]] | None = None
    model_input_dtypes: list[str] | None = None
    model_output_dtypes: list[str] | None = None
    tile_input_bytes: int = 0
    cuda_peak_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        """Initialise mutable timing containers."""
        if self.buckets is None:
            self.buckets = {}
        if self.tile_batch_sizes is None:
            self.tile_batch_sizes = []
        if self.tile_spatial_shapes is None:
            self.tile_spatial_shapes = []
        if self.model_input_dtypes is None:
            self.model_input_dtypes = []
        if self.model_output_dtypes is None:
            self.model_output_dtypes = []

    @contextmanager
    def measure(self, bucket: str, device: torch.device | None = None) -> Iterator[None]:
        """Measure CPU or CUDA elapsed seconds for a named bucket."""
        if not self.enabled:
            yield
            return
        resolved_device = device or self.device
        if resolved_device is not None and resolved_device.type == "cuda" and torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                end.synchronize()
                assert self.buckets is not None
                self.buckets[bucket] = self.buckets.get(bucket, 0.0) + start.elapsed_time(end) / 1000.0
            return
        started_at = time.perf_counter()
        try:
            yield
        finally:
            assert self.buckets is not None
            self.buckets[bucket] = self.buckets.get(bucket, 0.0) + time.perf_counter() - started_at

    def record_forward(self, samples: NestedTensor, outputs: dict[str, torch.Tensor]) -> None:
        """Record model-forward tensor properties."""
        if not self.enabled:
            return
        tensor = samples.tensors
        self.forward_calls += 1
        assert self.tile_batch_sizes is not None
        assert self.tile_spatial_shapes is not None
        assert self.model_input_dtypes is not None
        assert self.model_output_dtypes is not None
        self.tile_batch_sizes.append(int(tensor.shape[0]))
        self.tile_spatial_shapes.append((int(tensor.shape[-2]), int(tensor.shape[-1])))
        self.model_input_dtypes.append(str(tensor.dtype).removeprefix("torch."))
        self.tile_input_bytes += int(tensor.numel() * tensor.element_size())
        output_dtype = next((value.dtype for value in outputs.values() if torch.is_tensor(value)), None)
        if output_dtype is not None:
            self.model_output_dtypes.append(str(output_dtype).removeprefix("torch."))

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-friendly timing statistics."""
        if (
            self.enabled
            and self.device is not None
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            self.cuda_peak_memory_bytes = int(torch.cuda.max_memory_allocated(self.device))
        return {
            "buckets": dict(self.buckets or {}),
            "forward_calls": int(self.forward_calls),
            "tile_batch_sizes": list(self.tile_batch_sizes or []),
            "tile_spatial_shapes": [list(shape) for shape in self.tile_spatial_shapes or []],
            "model_input_dtypes": sorted(set(self.model_input_dtypes or [])),
            "model_output_dtypes": sorted(set(self.model_output_dtypes or [])),
            "tile_input_bytes": int(self.tile_input_bytes),
            "cuda_peak_memory_bytes": self.cuda_peak_memory_bytes,
        }


PreparedWindowBatch = tuple[
    list[torch.Tensor],
    list[torch.Tensor | None],
    list[TileWindow],
    list[tuple[int, int]],
    list[tuple[int, int]],
    list[tuple[int, int]],
    list[tuple[int, int]],
    list[tuple[float, float]],
    list[tuple[float, float, float, float]],
    list[TileWindow],
    NestedTensor | None,
]


@dataclass(frozen=True)
class _TileTask:
    """One tile prediction task in a batch-spanning scheduler."""

    image_index: int
    image: torch.Tensor
    window: TileWindow
    inference_window: TileWindow
    full_size: tuple[int, int]


def generate_tile_windows(
    *,
    height: int,
    width: int,
    tile_height: int,
    tile_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
) -> list[TileWindow]:
    """Generate windows that fully cover an image.

    Args:
        height: Image height.
        width: Image width.
        tile_height: Requested tile height.
        tile_width: Requested tile width.
        overlap_height_ratio: Fractional vertical overlap in ``[0, 1)``.
        overlap_width_ratio: Fractional horizontal overlap in ``[0, 1)``.

    Returns:
        Tile windows covering the full image, including right/bottom borders.
    """
    if height <= 0 or width <= 0:
        return []
    tile_height = min(max(1, int(tile_height)), int(height))
    tile_width = min(max(1, int(tile_width)), int(width))
    y_starts = _tile_starts(height, tile_height, overlap_height_ratio)
    x_starts = _tile_starts(width, tile_width, overlap_width_ratio)
    return [
        TileWindow(x0=x0, y0=y0, x1=min(x0 + tile_width, width), y1=min(y0 + tile_height, height))
        for y0 in y_starts
        for x0 in x_starts
    ]


def predict_windows(
    *,
    model: torch.nn.Module,
    postprocess: Any,
    samples: NestedTensor,
    targets: list[dict[str, torch.Tensor]],
    windows_per_image: list[list[TileWindow]],
    nms_threshold: float,
    merge_metric: MergeMetric,
    score_threshold: float,
    max_predictions: int,
    tile_batch_size: int,
    block_size: int,
    segmentation: bool,
    source_aware_duplicate_suppression: bool = False,
    full_image_size: int | None = None,
    window_resize_longest_side: int | None = None,
    window_resize_policy: WindowResizePolicy = "aspect_longest_side",
    window_source_stage: int = SOURCE_STAGE_TILE,
    tile_input_dtype: TileInputDType = "auto",
    tile_memory_format: TileMemoryFormat = "contiguous",
    batch_across_images: bool = False,
    return_diagnostics: bool = False,
    timing_stats: TiledTimingStats | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Run validation for explicitly supplied windows.

    Args:
        model: RF-DETR module to evaluate.
        postprocess: RF-DETR postprocessor callable.
        samples: Batched validation images.
        targets: Per-image metric targets.
        windows_per_image: Crop windows for every image in the batch.
        nms_threshold: Duplicate suppression threshold.
        merge_metric: Box merge metric, or mask IoU/IoS when masks are present.
        score_threshold: Minimum score retained for metrics.
        max_predictions: Maximum predictions returned per image.
        tile_batch_size: Number of windows per model forward.
        block_size: Backbone padding multiple.
        segmentation: Whether segmentation masks are expected.
        full_image_size: Optional longest side for an additional resized full-image context pass.
        window_resize_longest_side: Optional longest side used to resize each crop before prediction.
        window_resize_policy: How resized crop tensors are prepared before prediction.
        window_source_stage: Source-stage label attached to each crop prediction.
        tile_input_dtype: Optional dtype conversion for tile tensors before model forward.
        tile_memory_format: Optional memory format conversion for 4D tile batches.
        batch_across_images: Whether to batch tiles across images when supported.
        return_diagnostics: Whether to attach debug-only projection metadata to predictions.
        timing_stats: Optional debug timing accumulator.

    Returns:
        Merged prediction dictionaries.
    """
    results: list[dict[str, torch.Tensor]] = []
    images = samples.tensors
    tile_batch_size = max(1, int(tile_batch_size))
    if batch_across_images and full_image_size is None and not segmentation and not return_diagnostics:
        return _predict_windows_across_images(
            model=model,
            postprocess=postprocess,
            images=images,
            targets=targets,
            windows_per_image=windows_per_image,
            nms_threshold=nms_threshold,
            merge_metric=merge_metric,
            source_aware_duplicate_suppression=source_aware_duplicate_suppression,
            score_threshold=score_threshold,
            max_predictions=max_predictions,
            tile_batch_size=tile_batch_size,
            block_size=block_size,
            resize_longest_side=window_resize_longest_side,
            resize_policy=window_resize_policy,
            source_stage=window_source_stage,
            tile_input_dtype=tile_input_dtype,
            tile_memory_format=tile_memory_format,
            timing_stats=timing_stats,
        )
    for index, (image_tensor, target) in enumerate(zip(images, targets)):
        image_size = target.get("size", target["orig_size"])
        image_h, image_w = [int(value) for value in image_size.detach().cpu().tolist()]
        image = image_tensor[:, :image_h, :image_w]
        windows = windows_per_image[index] if index < len(windows_per_image) else []
        full_size = (image_h, image_w)
        if full_image_size is None:
            result = _predict_image_windows(
                model=model,
                postprocess=postprocess,
                image=image,
                windows=windows,
                full_size=full_size,
                orig_size=target["orig_size"],
                score_threshold=score_threshold,
                max_predictions=max_predictions,
                nms_threshold=nms_threshold,
                merge_metric=merge_metric,
                source_aware_duplicate_suppression=source_aware_duplicate_suppression,
                tile_batch_size=tile_batch_size,
                block_size=block_size,
                segmentation=segmentation,
                resize_longest_side=window_resize_longest_side,
                resize_policy=window_resize_policy,
                source_stage=window_source_stage,
                tile_input_dtype=tile_input_dtype,
                tile_memory_format=tile_memory_format,
                return_diagnostics=return_diagnostics,
                timing_stats=timing_stats,
            )
        else:
            full_predictions = _collect_resized_full_image_predictions(
                model=model,
                postprocess=postprocess,
                image=image,
                full_size=full_size,
                resize_longest_side=full_image_size,
                score_threshold=score_threshold,
                max_predictions=max_predictions,
                block_size=block_size,
                segmentation=segmentation,
                return_diagnostics=return_diagnostics,
                timing_stats=timing_stats,
            )
            tile_predictions = _collect_window_predictions(
                model=model,
                postprocess=postprocess,
                image=image,
                windows=windows,
                full_size=full_size,
                score_threshold=score_threshold,
                max_predictions=max_predictions,
                tile_batch_size=tile_batch_size,
                block_size=block_size,
                segmentation=segmentation,
                resize_longest_side=window_resize_longest_side,
                resize_policy=window_resize_policy,
                source_stage=window_source_stage,
                tile_input_dtype=tile_input_dtype,
                tile_memory_format=tile_memory_format,
                return_diagnostics=return_diagnostics,
                timing_stats=timing_stats,
            )
            merge_timer = timing_stats.measure("merge_nms", image.device) if timing_stats is not None else nullcontext()
            with merge_timer:
                result = _merge_predictions(
                    predictions=[full_predictions, tile_predictions],
                    full_size=full_size,
                    orig_size=target["orig_size"],
                    score_threshold=score_threshold,
                    max_predictions=max_predictions,
                    nms_threshold=nms_threshold,
                    merge_metric=merge_metric,
                    source_aware_duplicate_suppression=source_aware_duplicate_suppression,
                    segmentation=segmentation,
                    return_diagnostics=return_diagnostics,
                )
        target_device = target["boxes"].device
        results.append(
            {
                key: value if not torch.is_tensor(value) or value.device == target_device else value.to(target_device)
                for key, value in result.items()
            }
        )
    return results


def _tile_starts(length: int, tile_length: int, overlap_ratio: float) -> list[int]:
    """Return start coordinates that cover one image axis."""
    if length <= tile_length:
        return [0]
    step = max(1, int(round(tile_length * (1.0 - float(overlap_ratio)))))
    last_start = length - tile_length
    starts = list(range(0, last_start + 1, step))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _predict_image_windows(
    *,
    model: torch.nn.Module,
    postprocess: Any,
    image: torch.Tensor,
    windows: list[TileWindow],
    full_size: tuple[int, int],
    orig_size: torch.Tensor,
    score_threshold: float,
    max_predictions: int,
    nms_threshold: float,
    merge_metric: MergeMetric,
    tile_batch_size: int,
    block_size: int,
    segmentation: bool,
    source_aware_duplicate_suppression: bool = False,
    resize_longest_side: int | None = None,
    resize_policy: WindowResizePolicy = "aspect_longest_side",
    source_stage: int = SOURCE_STAGE_TILE,
    tile_input_dtype: TileInputDType = "auto",
    tile_memory_format: TileMemoryFormat = "contiguous",
    return_diagnostics: bool = False,
    timing_stats: TiledTimingStats | None = None,
) -> dict[str, torch.Tensor]:
    """Predict and merge windows for one image."""
    predictions = _collect_window_predictions(
        model=model,
        postprocess=postprocess,
        image=image,
        windows=windows,
        full_size=full_size,
        score_threshold=score_threshold,
        max_predictions=max_predictions,
        tile_batch_size=tile_batch_size,
        block_size=block_size,
        segmentation=segmentation,
        resize_longest_side=resize_longest_side,
        resize_policy=resize_policy,
        source_stage=source_stage,
        tile_input_dtype=tile_input_dtype,
        tile_memory_format=tile_memory_format,
        return_diagnostics=return_diagnostics,
        timing_stats=timing_stats,
    )
    merge_timer = timing_stats.measure("merge_nms", orig_size.device) if timing_stats is not None else nullcontext()
    with merge_timer:
        return _merge_predictions(
            predictions=[predictions],
            full_size=full_size,
            orig_size=orig_size,
            score_threshold=score_threshold,
            max_predictions=max_predictions,
            nms_threshold=nms_threshold,
            merge_metric=merge_metric,
            source_aware_duplicate_suppression=source_aware_duplicate_suppression,
            segmentation=segmentation,
            return_diagnostics=return_diagnostics,
        )


def _predict_windows_across_images(
    *,
    model: torch.nn.Module,
    postprocess: Any,
    images: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    windows_per_image: list[list[TileWindow]],
    nms_threshold: float,
    merge_metric: MergeMetric,
    score_threshold: float,
    max_predictions: int,
    tile_batch_size: int,
    block_size: int,
    resize_longest_side: int | None,
    resize_policy: WindowResizePolicy,
    source_stage: int,
    tile_input_dtype: TileInputDType,
    tile_memory_format: TileMemoryFormat,
    source_aware_duplicate_suppression: bool = False,
    timing_stats: TiledTimingStats | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Predict tiles across images in global chunks, then merge per image."""
    tasks: list[_TileTask] = []
    full_sizes: list[tuple[int, int]] = []
    for image_index, (image_tensor, target) in enumerate(zip(images, targets)):
        image_size = target.get("size", target["orig_size"])
        image_h, image_w = [int(value) for value in image_size.detach().cpu().tolist()]
        image = image_tensor[:, :image_h, :image_w]
        full_size = (image_h, image_w)
        full_sizes.append(full_size)
        windows = windows_per_image[image_index] if image_index < len(windows_per_image) else []
        for window in windows:
            inference_window = _inference_window_for_policy(
                window=window,
                full_size=full_size,
                resize_policy=resize_policy,
            )
            tasks.append(
                _TileTask(
                    image_index=image_index,
                    image=image,
                    window=window,
                    inference_window=inference_window,
                    full_size=full_size,
                )
            )

    device = images.device
    boxes_by_image: list[list[torch.Tensor]] = [[] for _target in targets]
    scores_by_image: list[list[torch.Tensor]] = [[] for _target in targets]
    labels_by_image: list[list[torch.Tensor]] = [[] for _target in targets]
    source_windows_by_image: list[list[tuple[int, int, int, int]]] = [[] for _target in targets]
    source_counts_by_image: list[list[int]] = [[] for _target in targets]

    for start in range(0, len(tasks), tile_batch_size):
        batch_tasks = tasks[start : start + tile_batch_size]
        prepare_timer = timing_stats.measure("prepare_tiles", device) if timing_stats is not None else nullcontext()
        with prepare_timer:
            (
                tile_tensors,
                tile_masks,
                model_windows,
                model_sizes,
                _valid_sizes,
                target_sizes_for_postprocess,
                source_sizes,
                _resize_scales,
                _resize_paddings,
                inference_windows,
                prepared_samples,
            ) = _prepare_tile_task_batch(
                batch_tasks,
                resize_longest_side=resize_longest_side,
                resize_policy=resize_policy,
                timing_stats=timing_stats,
            )
        tile_sizes = torch.as_tensor(target_sizes_for_postprocess, dtype=torch.float32, device=device)
        tile_samples = prepared_samples or _nested_tensor_from_prepared_windows(
            tile_tensors,
            tile_masks,
            block_size=block_size,
        )
        tile_samples = _format_tile_samples(
            tile_samples,
            tile_input_dtype=tile_input_dtype,
            tile_memory_format=tile_memory_format,
        )
        forward_timer = timing_stats.measure("model_forward", device) if timing_stats is not None else nullcontext()
        with forward_timer:
            outputs = model(tile_samples)
        if timing_stats is not None:
            timing_stats.record_forward(tile_samples, outputs)
        postprocess_timer = timing_stats.measure("postprocess", device) if timing_stats is not None else nullcontext()
        with postprocess_timer:
            tile_results = postprocess(outputs, tile_sizes)
        shift_timer = timing_stats.measure("shift_project", device) if timing_stats is not None else nullcontext()
        with shift_timer:
            for tile_result, model_window, model_size, source_size, task, inference_window in zip(
                tile_results,
                model_windows,
                model_sizes,
                source_sizes,
                batch_tasks,
                inference_windows,
            ):
                shifted = _shift_tile_result(
                    tile_result,
                    window=model_window,
                    full_size=model_size,
                    score_threshold=score_threshold,
                    max_predictions=max_predictions,
                    segmentation=False,
                )
                shifted = _project_local_result_to_window(
                    shifted,
                    source_size=source_size,
                    window=inference_window,
                    full_size=task.full_size,
                    segmentation=False,
                )
                if shifted["scores"].numel() == 0:
                    continue
                image_index = task.image_index
                boxes_by_image[image_index].append(shifted["boxes"])
                scores_by_image[image_index].append(shifted["scores"])
                labels_by_image[image_index].append(shifted["labels"])
                source_windows_by_image[image_index].append(
                    (task.window.x0, task.window.y0, task.window.x1, task.window.y1)
                )
                source_counts_by_image[image_index].append(int(shifted["scores"].numel()))

    results: list[dict[str, torch.Tensor]] = []
    for image_index, target in enumerate(targets):
        if not scores_by_image[image_index]:
            collected = _empty_result(device=device)
        else:
            boxes = torch.cat(boxes_by_image[image_index])
            scores = torch.cat(scores_by_image[image_index])
            labels = torch.cat(labels_by_image[image_index])
            collected = {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
                "_source_windows": _source_windows_from_counts(
                    boxes,
                    source_windows_by_image[image_index],
                    source_counts_by_image[image_index],
                ),
                "_source_stages": labels.new_full((scores.shape[0],), int(source_stage)),
            }
        merge_timer = timing_stats.measure("merge_nms", device) if timing_stats is not None else nullcontext()
        with merge_timer:
            result = _merge_predictions(
                predictions=[collected],
                full_size=full_sizes[image_index],
                orig_size=target["orig_size"],
                score_threshold=score_threshold,
                max_predictions=max_predictions,
                nms_threshold=nms_threshold,
                merge_metric=merge_metric,
                source_aware_duplicate_suppression=source_aware_duplicate_suppression,
                segmentation=False,
            )
        target_device = target["boxes"].device
        results.append(
            {
                key: value if not torch.is_tensor(value) or value.device == target_device else value.to(target_device)
                for key, value in result.items()
            }
        )
    return results


def _collect_window_predictions(
    *,
    model: torch.nn.Module,
    postprocess: Any,
    image: torch.Tensor,
    windows: list[TileWindow],
    full_size: tuple[int, int],
    score_threshold: float,
    max_predictions: int,
    tile_batch_size: int,
    block_size: int,
    segmentation: bool,
    resize_longest_side: int | None = None,
    resize_policy: WindowResizePolicy = "aspect_longest_side",
    source_stage: int = SOURCE_STAGE_TILE,
    tile_input_dtype: TileInputDType = "auto",
    tile_memory_format: TileMemoryFormat = "contiguous",
    return_diagnostics: bool = False,
    timing_stats: TiledTimingStats | None = None,
) -> dict[str, torch.Tensor]:
    """Collect unmerged predictions shifted into full-image coordinates."""
    device = image.device
    if not windows:
        return _empty_result(device=device, size=torch.tensor(full_size, device=device) if segmentation else None)

    boxes_list: list[torch.Tensor] = []
    scores_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    masks_list: list[torch.Tensor] = []
    source_windows_list: list[torch.Tensor] = []
    source_window_rows: list[tuple[int, int, int, int]] = []
    source_window_counts: list[int] = []
    local_boxes_list: list[torch.Tensor] = []
    model_sizes_list: list[torch.Tensor] = []
    resize_scales_list: list[torch.Tensor] = []
    resize_paddings_list: list[torch.Tensor] = []
    inference_windows_list: list[torch.Tensor] = []
    valid_sizes_list: list[torch.Tensor] = []
    target_sizes_list: list[torch.Tensor] = []

    for start in range(0, len(windows), tile_batch_size):
        batch_windows = windows[start : start + tile_batch_size]
        prepare_timer = timing_stats.measure("prepare_tiles", device) if timing_stats is not None else nullcontext()
        with prepare_timer:
            (
                tile_tensors,
                tile_masks,
                model_windows,
                model_sizes,
                valid_sizes,
                target_sizes_for_postprocess,
                source_sizes,
                resize_scales,
                resize_paddings,
                inference_windows,
                prepared_samples,
            ) = _prepare_window_batch(
                image,
                windows=batch_windows,
                full_size=full_size,
                resize_longest_side=resize_longest_side,
                resize_policy=resize_policy,
                timing_stats=timing_stats,
            )
        tile_sizes = torch.as_tensor(
            target_sizes_for_postprocess,
            dtype=torch.float32,
            device=device,
        )
        tile_samples = prepared_samples or _nested_tensor_from_prepared_windows(
            tile_tensors,
            tile_masks,
            block_size=block_size,
        )
        tile_samples = _format_tile_samples(
            tile_samples,
            tile_input_dtype=tile_input_dtype,
            tile_memory_format=tile_memory_format,
        )
        forward_timer = timing_stats.measure("model_forward", device) if timing_stats is not None else nullcontext()
        with forward_timer:
            outputs = model(tile_samples)
        if timing_stats is not None:
            timing_stats.record_forward(tile_samples, outputs)
        postprocess_timer = timing_stats.measure("postprocess", device) if timing_stats is not None else nullcontext()
        with postprocess_timer:
            tile_results = postprocess(outputs, tile_sizes)
        shift_timer = timing_stats.measure("shift_project", device) if timing_stats is not None else nullcontext()
        with shift_timer:
            if not segmentation:
                shifted = _shift_project_detection_batch(
                    tile_results=tile_results,
                    model_sizes=model_sizes,
                    source_sizes=source_sizes,
                    windows=batch_windows,
                    inference_windows=inference_windows,
                    valid_sizes=valid_sizes,
                    target_sizes=target_sizes_for_postprocess,
                    resize_scales=resize_scales,
                    resize_paddings=resize_paddings,
                    full_size=full_size,
                    score_threshold=score_threshold,
                    max_predictions=max_predictions,
                    return_diagnostics=return_diagnostics,
                    device=device,
                )
                if shifted["scores"].numel() == 0:
                    pass
                else:
                    boxes_list.append(shifted["boxes"])
                    scores_list.append(shifted["scores"])
                    labels_list.append(shifted["labels"])
                    source_windows_list.append(shifted["_source_windows"])
                    if return_diagnostics:
                        if "_local_boxes" in shifted:
                            local_boxes_list.append(shifted["_local_boxes"])
                        model_sizes_list.append(shifted["_model_sizes"])
                        resize_scales_list.append(shifted["_resize_scales"])
                        resize_paddings_list.append(shifted["_resize_paddings"])
                        inference_windows_list.append(shifted["_inference_windows"])
                        valid_sizes_list.append(shifted["_valid_sizes"])
                        target_sizes_list.append(shifted["_target_sizes"])
            else:
                for (
                    tile_result,
                    model_window,
                    model_size,
                    valid_size,
                    target_size,
                    source_size,
                    scale,
                    padding,
                    window,
                    inference_window,
                ) in zip(
                    tile_results,
                    model_windows,
                    model_sizes,
                    valid_sizes,
                    target_sizes_for_postprocess,
                    source_sizes,
                    resize_scales,
                    resize_paddings,
                    batch_windows,
                    inference_windows,
                ):
                    shifted = _shift_tile_result(
                        tile_result,
                        window=model_window,
                        full_size=model_size,
                        score_threshold=score_threshold,
                        max_predictions=max_predictions,
                        segmentation=segmentation,
                    )
                    if return_diagnostics and shifted["scores"].numel() > 0:
                        shifted["_local_boxes"] = shifted["boxes"].clone()
                    shifted = _project_local_result_to_window(
                        shifted,
                        source_size=source_size,
                        window=inference_window,
                        full_size=full_size,
                        segmentation=segmentation,
                    )
                    if shifted["scores"].numel() == 0:
                        continue
                    boxes_list.append(shifted["boxes"])
                    scores_list.append(shifted["scores"])
                    labels_list.append(shifted["labels"])
                    source_window_rows.append((window.x0, window.y0, window.x1, window.y1))
                    source_window_counts.append(int(shifted["scores"].numel()))
                    if return_diagnostics:
                        if "_local_boxes" in shifted:
                            local_boxes_list.append(shifted["_local_boxes"])
                        model_sizes_list.append(
                            shifted["boxes"].new_tensor([[model_size[0], model_size[1]]]).expand(
                                shifted["scores"].numel(),
                                2,
                            )
                        )
                        resize_scales_list.append(
                            shifted["boxes"].new_tensor([[scale[0], scale[1]]]).expand(shifted["scores"].numel(), 2)
                        )
                        resize_paddings_list.append(
                            shifted["boxes"].new_tensor([padding]).expand(shifted["scores"].numel(), 4)
                        )
                        inference_window_tensor = shifted["boxes"].new_tensor(
                            [[inference_window.x0, inference_window.y0, inference_window.x1, inference_window.y1]]
                        )
                        inference_windows_list.append(inference_window_tensor.expand(shifted["scores"].numel(), 4))
                        valid_sizes_list.append(
                            shifted["boxes"].new_tensor([[valid_size[0], valid_size[1]]]).expand(
                                shifted["scores"].numel(),
                                2,
                            )
                        )
                        target_sizes_list.append(
                            shifted["boxes"].new_tensor([[target_size[0], target_size[1]]]).expand(
                                shifted["scores"].numel(),
                                2,
                            )
                        )
                    if "masks" in shifted:
                        masks_list.append(shifted["masks"])

    if not scores_list:
        return _empty_result(device=device, size=torch.tensor(full_size, device=device) if segmentation else None)

    boxes = torch.cat(boxes_list)
    scores = torch.cat(scores_list)
    labels = torch.cat(labels_list)
    source_windows = torch.cat(source_windows_list) if source_windows_list else _source_windows_from_counts(
        boxes,
        source_window_rows,
        source_window_counts,
    )
    source_stages = labels.new_full((scores.shape[0],), int(source_stage))
    masks = torch.cat(masks_list) if masks_list and len(masks_list) == len(scores_list) else None
    result: dict[str, torch.Tensor] = {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
        "_source_windows": source_windows,
        "_source_stages": source_stages,
    }
    if return_diagnostics:
        local_boxes = (
            torch.cat(local_boxes_list)
            if local_boxes_list and len(local_boxes_list) == len(scores_list)
            else None
        )
        if local_boxes is not None:
            result["_local_boxes"] = local_boxes
        if model_sizes_list:
            result["_model_sizes"] = torch.cat(model_sizes_list)
            result["_resize_scales"] = torch.cat(resize_scales_list)
            result["_resize_paddings"] = torch.cat(resize_paddings_list)
            result["_inference_windows"] = torch.cat(inference_windows_list)
            result["_valid_sizes"] = torch.cat(valid_sizes_list)
            result["_target_sizes"] = torch.cat(target_sizes_list)
    if masks is not None:
        result["masks"] = masks
    return result


def _prepare_window_batch(
    image: torch.Tensor,
    *,
    windows: list[TileWindow],
    full_size: tuple[int, int],
    resize_longest_side: int | None,
    resize_policy: WindowResizePolicy,
    timing_stats: TiledTimingStats | None = None,
) -> PreparedWindowBatch:
    """Prepare a batch of crop tensors and projection metadata."""
    inference_windows = [
        _inference_window_for_policy(
            window=window,
            full_size=full_size,
            resize_policy=resize_policy,
        )
        for window in windows
    ]
    prepared = _prepare_uniform_resized_window_batch(
        image,
        windows=windows,
        inference_windows=inference_windows,
        resize_longest_side=resize_longest_side,
        resize_policy=resize_policy,
        timing_stats=timing_stats,
    )
    if prepared is not None:
        return prepared

    tile_tensors: list[torch.Tensor] = []
    tile_masks: list[torch.Tensor | None] = []
    model_windows: list[TileWindow] = []
    model_sizes: list[tuple[int, int]] = []
    valid_sizes: list[tuple[int, int]] = []
    source_sizes: list[tuple[int, int]] = []
    target_sizes_for_postprocess: list[tuple[int, int]] = []
    resize_scales: list[tuple[float, float]] = []
    resize_paddings: list[tuple[float, float, float, float]] = []
    for inference_window in inference_windows:
        tile_tensor = image[:, inference_window.y0 : inference_window.y1, inference_window.x0 : inference_window.x1]
        (
            tile_tensor,
            tile_mask,
            model_size,
            valid_size,
            target_size,
            source_size,
            scale,
            padding,
        ) = _prepare_window_tensor(
            tile_tensor,
            resize_longest_side=resize_longest_side,
            resize_policy=resize_policy,
        )
        tile_tensors.append(tile_tensor)
        tile_masks.append(tile_mask)
        model_windows.append(TileWindow(0, 0, target_size[1], target_size[0]))
        model_sizes.append(model_size)
        valid_sizes.append(valid_size)
        target_sizes_for_postprocess.append(target_size)
        source_sizes.append(source_size)
        resize_scales.append(scale)
        resize_paddings.append(padding)
    return (
        tile_tensors,
        tile_masks,
        model_windows,
        model_sizes,
        valid_sizes,
        target_sizes_for_postprocess,
        source_sizes,
        resize_scales,
        resize_paddings,
        inference_windows,
        None,
    )


def _prepare_tile_task_batch(
    tasks: list[_TileTask],
    *,
    resize_longest_side: int | None,
    resize_policy: WindowResizePolicy,
    timing_stats: TiledTimingStats | None = None,
) -> PreparedWindowBatch:
    """Prepare a global batch of tile tasks."""
    prepared = _prepare_uniform_tile_task_batch(
        tasks,
        resize_longest_side=resize_longest_side,
        resize_policy=resize_policy,
        timing_stats=timing_stats,
    )
    if prepared is not None:
        return prepared

    tile_tensors: list[torch.Tensor] = []
    tile_masks: list[torch.Tensor | None] = []
    model_windows: list[TileWindow] = []
    model_sizes: list[tuple[int, int]] = []
    valid_sizes: list[tuple[int, int]] = []
    source_sizes: list[tuple[int, int]] = []
    target_sizes_for_postprocess: list[tuple[int, int]] = []
    resize_scales: list[tuple[float, float]] = []
    resize_paddings: list[tuple[float, float, float, float]] = []
    inference_windows: list[TileWindow] = []
    for task in tasks:
        inference_window = task.inference_window
        tile_tensor = task.image[:, inference_window.y0 : inference_window.y1, inference_window.x0 : inference_window.x1]
        (
            tile_tensor,
            tile_mask,
            model_size,
            valid_size,
            target_size,
            source_size,
            scale,
            padding,
        ) = _prepare_window_tensor(
            tile_tensor,
            resize_longest_side=resize_longest_side,
            resize_policy=resize_policy,
        )
        tile_tensors.append(tile_tensor)
        tile_masks.append(tile_mask)
        model_windows.append(TileWindow(0, 0, target_size[1], target_size[0]))
        model_sizes.append(model_size)
        valid_sizes.append(valid_size)
        target_sizes_for_postprocess.append(target_size)
        source_sizes.append(source_size)
        resize_scales.append(scale)
        resize_paddings.append(padding)
        inference_windows.append(inference_window)
    return (
        tile_tensors,
        tile_masks,
        model_windows,
        model_sizes,
        valid_sizes,
        target_sizes_for_postprocess,
        source_sizes,
        resize_scales,
        resize_paddings,
        inference_windows,
        None,
    )


def _prepare_uniform_resized_window_batch(
    image: torch.Tensor,
    *,
    windows: list[TileWindow],
    inference_windows: list[TileWindow],
    resize_longest_side: int | None,
    resize_policy: WindowResizePolicy,
    timing_stats: TiledTimingStats | None = None,
) -> PreparedWindowBatch | None:
    """Prepare same-shaped resized crop tensors with one batched interpolation."""
    if resize_longest_side is None or not windows:
        return None
    if resize_policy not in {"square_stretch", "aspect_longest_side", "aspect_valid_target"}:
        return None

    source_shapes = [
        (inference_window.y1 - inference_window.y0, inference_window.x1 - inference_window.x0)
        for inference_window in inference_windows
    ]
    if len(set(source_shapes)) != 1:
        return None
    source_h, source_w = source_shapes[0]
    if source_h <= 0 or source_w <= 0:
        return None

    target = max(1, int(resize_longest_side))
    if resize_policy == "square_stretch":
        resized_size = (target, target)
        scale = (float(target) / max(float(source_w), 1.0), float(target) / max(float(source_h), 1.0))
    else:
        resized_size = _longest_side_resized_size(source_h, source_w, longest_side=target)
        uniform_scale = float(max(resized_size)) / max(float(max(source_h, source_w)), 1.0)
        scale = (uniform_scale, uniform_scale)

    device = image.device
    stack_timer = timing_stats.measure("crop_stack", device) if timing_stats is not None else nullcontext()
    with stack_timer:
        crop_batch = torch.stack(
            [
                image[:, inference_window.y0 : inference_window.y1, inference_window.x0 : inference_window.x1]
                for inference_window in inference_windows
            ]
        )
    if tuple(crop_batch.shape[-2:]) != resized_size:
        resize_timer = timing_stats.measure("resize_interpolate", device) if timing_stats is not None else nullcontext()
        with resize_timer:
            crop_batch = F.interpolate(crop_batch, size=resized_size, mode="bilinear", align_corners=False)

    mask = torch.zeros(
        (crop_batch.shape[0], crop_batch.shape[-2], crop_batch.shape[-1]),
        dtype=torch.bool,
        device=crop_batch.device,
    )
    prepared_samples = NestedTensor(crop_batch, mask)
    tile_tensors = []
    tile_masks = [None for _window in windows]
    model_windows = [TileWindow(0, 0, resized_size[1], resized_size[0]) for _window in windows]
    model_sizes = [resized_size for _window in windows]
    valid_sizes = [resized_size for _window in windows]
    target_sizes_for_postprocess = [resized_size for _window in windows]
    source_sizes = [resized_size for _window in windows]
    resize_scales = [scale for _window in windows]
    resize_paddings = [(0.0, 0.0, 0.0, 0.0) for _window in windows]
    return (
        tile_tensors,
        tile_masks,
        model_windows,
        model_sizes,
        valid_sizes,
        target_sizes_for_postprocess,
        source_sizes,
        resize_scales,
        resize_paddings,
        inference_windows,
        prepared_samples,
    )


def _prepare_uniform_tile_task_batch(
    tasks: list[_TileTask],
    *,
    resize_longest_side: int | None,
    resize_policy: WindowResizePolicy,
    timing_stats: TiledTimingStats | None = None,
) -> PreparedWindowBatch | None:
    """Prepare same-shaped global tile tasks with one batched interpolation."""
    if resize_longest_side is None or not tasks:
        return None
    if resize_policy not in {"square_stretch", "aspect_longest_side", "aspect_valid_target"}:
        return None

    source_shapes = [
        (
            task.inference_window.y1 - task.inference_window.y0,
            task.inference_window.x1 - task.inference_window.x0,
        )
        for task in tasks
    ]
    if len(set(source_shapes)) != 1:
        return None
    source_h, source_w = source_shapes[0]
    if source_h <= 0 or source_w <= 0:
        return None

    target = max(1, int(resize_longest_side))
    if resize_policy == "square_stretch":
        resized_size = (target, target)
        scale = (float(target) / max(float(source_w), 1.0), float(target) / max(float(source_h), 1.0))
    else:
        resized_size = _longest_side_resized_size(source_h, source_w, longest_side=target)
        uniform_scale = float(max(resized_size)) / max(float(max(source_h, source_w)), 1.0)
        scale = (uniform_scale, uniform_scale)

    device = tasks[0].image.device
    stack_timer = timing_stats.measure("crop_stack", device) if timing_stats is not None else nullcontext()
    with stack_timer:
        crop_batch = torch.stack(
            [
                task.image[
                    :,
                    task.inference_window.y0 : task.inference_window.y1,
                    task.inference_window.x0 : task.inference_window.x1,
                ]
                for task in tasks
            ]
        )
    if tuple(crop_batch.shape[-2:]) != resized_size:
        resize_timer = timing_stats.measure("resize_interpolate", device) if timing_stats is not None else nullcontext()
        with resize_timer:
            crop_batch = F.interpolate(crop_batch, size=resized_size, mode="bilinear", align_corners=False)

    mask = torch.zeros(
        (crop_batch.shape[0], crop_batch.shape[-2], crop_batch.shape[-1]),
        dtype=torch.bool,
        device=crop_batch.device,
    )
    prepared_samples = NestedTensor(crop_batch, mask)
    inference_windows = [task.inference_window for task in tasks]
    tile_tensors: list[torch.Tensor] = []
    tile_masks = [None for _task in tasks]
    model_windows = [TileWindow(0, 0, resized_size[1], resized_size[0]) for _task in tasks]
    model_sizes = [resized_size for _task in tasks]
    valid_sizes = [resized_size for _task in tasks]
    target_sizes_for_postprocess = [resized_size for _task in tasks]
    source_sizes = [resized_size for _task in tasks]
    resize_scales = [scale for _task in tasks]
    resize_paddings = [(0.0, 0.0, 0.0, 0.0) for _task in tasks]
    return (
        tile_tensors,
        tile_masks,
        model_windows,
        model_sizes,
        valid_sizes,
        target_sizes_for_postprocess,
        source_sizes,
        resize_scales,
        resize_paddings,
        inference_windows,
        prepared_samples,
    )


def _format_tile_samples(
    samples: NestedTensor,
    *,
    tile_input_dtype: TileInputDType,
    tile_memory_format: TileMemoryFormat,
) -> NestedTensor:
    """Apply optional dtype and memory-format conversion to model input tiles."""
    tensor = samples.tensors
    dtype = _tile_input_torch_dtype(tile_input_dtype)
    memory_format = torch.channels_last if tile_memory_format == "channels_last" else torch.contiguous_format
    needs_dtype = dtype is not None and tensor.dtype != dtype
    needs_memory_format = tensor.ndim == 4 and not tensor.is_contiguous(memory_format=memory_format)
    if needs_dtype or needs_memory_format:
        if dtype is None:
            tensor = tensor.to(memory_format=memory_format)
        else:
            tensor = tensor.to(dtype=dtype, memory_format=memory_format)
    return NestedTensor(tensor, samples.mask)


def _tile_input_torch_dtype(tile_input_dtype: TileInputDType) -> torch.dtype | None:
    """Resolve a tile input dtype option to a torch dtype."""
    if tile_input_dtype == "auto":
        return None
    if tile_input_dtype == "fp32":
        return torch.float32
    if tile_input_dtype == "bf16":
        return torch.bfloat16
    if tile_input_dtype == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported tile input dtype: {tile_input_dtype!r}")


def _source_windows_from_counts(
    reference: torch.Tensor,
    windows: list[tuple[int, int, int, int]],
    counts: list[int],
) -> torch.Tensor:
    """Build repeated source-window metadata once per image."""
    if not windows:
        return torch.empty((0, 4), dtype=reference.dtype, device=reference.device)
    rows = reference.new_tensor(windows)
    repeats = torch.as_tensor(counts, dtype=torch.long, device=reference.device)
    return torch.repeat_interleave(rows, repeats, dim=0)


def _prepare_window_tensor(
    image: torch.Tensor,
    *,
    resize_longest_side: int | None,
    resize_policy: WindowResizePolicy,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
    tuple[float, float],
    tuple[float, float, float, float],
]:
    """Prepare one crop tensor for model inference and return projection metadata."""
    height, width = int(image.shape[-2]), int(image.shape[-1])
    if resize_longest_side is None:
        size = (height, width)
        return image, None, size, size, size, size, (1.0, 1.0), (0.0, 0.0, 0.0, 0.0)

    target = max(1, int(resize_longest_side))
    if resize_policy == "square_stretch":
        resized = F.interpolate(
            image.unsqueeze(0),
            size=(target, target),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        scale_x = float(target) / max(float(width), 1.0)
        scale_y = float(target) / max(float(height), 1.0)
        size = (target, target)
        return resized, None, size, size, size, size, (scale_x, scale_y), (0.0, 0.0, 0.0, 0.0)

    resized, resized_size = _resize_image_longest_side(image, longest_side=target)
    scale = float(max(resized_size)) / max(float(max(height, width)), 1.0)
    if resize_policy in {"aspect_longest_side", "aspect_valid_target"}:
        return resized, None, resized_size, resized_size, resized_size, resized_size, (scale, scale), (0.0, 0.0, 0.0, 0.0)
    if resize_policy in {"square_letterbox", "letterbox_valid_target", "letterbox_canvas_target", "square_context"}:
        resized_h, resized_w = resized_size
        padded = image.new_zeros((image.shape[0], target, target))
        mask = torch.ones((target, target), dtype=torch.bool, device=image.device)
        padded[:, :resized_h, :resized_w] = resized
        mask[:resized_h, :resized_w] = False
        padding = (0.0, 0.0, float(target - resized_w), float(target - resized_h))
        target_size = (target, target) if resize_policy == "letterbox_canvas_target" else resized_size
        source_size = resized_size
        if resize_policy == "square_context":
            source_size = resized_size
            target_size = resized_size
        return padded, mask, (target, target), resized_size, target_size, source_size, (scale, scale), padding
    raise ValueError(f"Unsupported window resize policy: {resize_policy!r}")


def _inference_window_for_policy(
    *,
    window: TileWindow,
    full_size: tuple[int, int],
    resize_policy: WindowResizePolicy,
) -> TileWindow:
    """Return the actual source crop used for a requested validation window."""
    if resize_policy != "square_context":
        return window
    full_h, full_w = full_size
    width = window.x1 - window.x0
    height = window.y1 - window.y0
    side = max(width, height)
    center_x = (window.x0 + window.x1) * 0.5
    center_y = (window.y0 + window.y1) * 0.5
    x0 = int(round(center_x - side * 0.5))
    y0 = int(round(center_y - side * 0.5))
    x0 = min(max(0, x0), max(0, full_w - side))
    y0 = min(max(0, y0), max(0, full_h - side))
    x1 = min(full_w, x0 + side)
    y1 = min(full_h, y0 + side)
    if x1 - x0 < side:
        x0 = max(0, x1 - side)
    if y1 - y0 < side:
        y0 = max(0, y1 - side)
    return TileWindow(x0=x0, y0=y0, x1=x1, y1=y1)


def _nested_tensor_from_prepared_windows(
    tensors: list[torch.Tensor],
    masks: list[torch.Tensor | None],
    *,
    block_size: int,
) -> NestedTensor:
    """Build a NestedTensor while preserving explicit letterbox masks."""
    if not any(mask is not None for mask in masks) and _can_stack_prepared_windows(tensors, block_size=block_size):
        batch = torch.stack(tensors)
        mask = torch.zeros(
            (batch.shape[0], batch.shape[-2], batch.shape[-1]),
            dtype=torch.bool,
            device=batch.device,
        )
        return NestedTensor(batch, mask)

    if not any(mask is not None for mask in masks):
        return nested_tensor_from_tensor_list(tensors, block_size=block_size)

    nested = nested_tensor_from_tensor_list(tensors, block_size=block_size)
    assert nested.mask is not None
    for index, mask in enumerate(masks):
        if mask is None:
            height, width = tensors[index].shape[-2:]
            nested.mask[index, :height, :width] = False
            continue
        height, width = mask.shape[-2:]
        nested.mask[index, :height, :width] = mask
    return nested


def _can_stack_prepared_windows(tensors: list[torch.Tensor], *, block_size: int) -> bool:
    """Return whether prepared windows can bypass generic padding."""
    if not tensors:
        return False
    reference_shape = tuple(tensors[0].shape)
    if any(tuple(tensor.shape) != reference_shape for tensor in tensors):
        return False
    height, width = int(reference_shape[-2]), int(reference_shape[-1])
    return block_size <= 1 or (height % int(block_size) == 0 and width % int(block_size) == 0)


def _collect_resized_full_image_predictions(
    *,
    model: torch.nn.Module,
    postprocess: Any,
    image: torch.Tensor,
    full_size: tuple[int, int],
    resize_longest_side: int,
    score_threshold: float,
    max_predictions: int,
    block_size: int,
    segmentation: bool,
    return_diagnostics: bool = False,
    timing_stats: TiledTimingStats | None = None,
) -> dict[str, torch.Tensor]:
    """Run one full-image context pass after resizing the longest side."""
    device = image.device
    if image.numel() == 0 or resize_longest_side <= 0:
        return _empty_result(device=device, size=torch.tensor(full_size, device=device) if segmentation else None)
    resized_image, resized_size = _resize_image_longest_side(image, longest_side=resize_longest_side)
    resized_window = TileWindow(0, 0, resized_size[1], resized_size[0])
    tile_samples = _letterbox_nested_tensor(resized_image, longest_side=resize_longest_side, block_size=block_size)
    target_sizes = torch.as_tensor([resized_size], dtype=torch.float32, device=device)
    forward_timer = timing_stats.measure("model_forward", device) if timing_stats is not None else nullcontext()
    with forward_timer:
        outputs = model(tile_samples)
    if timing_stats is not None:
        timing_stats.record_forward(tile_samples, outputs)
    postprocess_timer = timing_stats.measure("postprocess", device) if timing_stats is not None else nullcontext()
    with postprocess_timer:
        resized_result = postprocess(outputs, target_sizes)[0]
    filtered = _shift_tile_result(
        resized_result,
        window=resized_window,
        full_size=resized_size,
        score_threshold=score_threshold,
        max_predictions=max_predictions,
        segmentation=segmentation,
    )
    if return_diagnostics and filtered["scores"].numel() > 0:
        filtered["_local_boxes"] = filtered["boxes"].clone()
    scaled = _scale_to_original_size(
        filtered,
        image_size=resized_size,
        orig_size=torch.tensor(full_size, device=device),
        segmentation=segmentation,
    )
    if scaled["scores"].numel() > 0:
        full_h, full_w = full_size
        scaled["_source_windows"] = scaled["boxes"].new_tensor([[0, 0, full_w, full_h]]).expand(
            scaled["scores"].numel(),
            4,
        )
        scaled["_source_stages"] = scaled["labels"].new_full((scaled["scores"].numel(),), SOURCE_STAGE_FULL)
        if return_diagnostics:
            scaled["_model_sizes"] = scaled["boxes"].new_tensor([[resized_size[0], resized_size[1]]]).expand(
                scaled["scores"].numel(),
                2,
            )
            full_h, full_w = full_size
            scale_x = float(resized_size[1]) / max(float(full_w), 1.0)
            scale_y = float(resized_size[0]) / max(float(full_h), 1.0)
            scaled["_resize_scales"] = scaled["boxes"].new_tensor([[scale_x, scale_y]]).expand(
                scaled["scores"].numel(),
                2,
            )
            scaled["_resize_paddings"] = scaled["boxes"].new_zeros((scaled["scores"].numel(), 4))
    return scaled


def _resize_image_longest_side(image: torch.Tensor, *, longest_side: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Resize ``image`` so its longest side equals ``longest_side``."""
    height, width = int(image.shape[-2]), int(image.shape[-1])
    resized_h, resized_w = _longest_side_resized_size(height, width, longest_side=longest_side)
    if resized_h == height and resized_w == width:
        return image, (height, width)
    resized = F.interpolate(
        image.unsqueeze(0),
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return resized, (resized_h, resized_w)


def _longest_side_resized_size(height: int, width: int, *, longest_side: int) -> tuple[int, int]:
    """Return ``(height, width)`` after longest-side preserving resize."""
    if height <= 0 or width <= 0:
        return height, width
    scale = float(longest_side) / float(max(height, width))
    resized_h = max(1, int(round(float(height) * scale)))
    resized_w = max(1, int(round(float(width) * scale)))
    return resized_h, resized_w


def _letterbox_nested_tensor(image: torch.Tensor, *, longest_side: int, block_size: int) -> NestedTensor:
    """Pad one resized image to a square letterbox ``NestedTensor``."""
    height, width = int(image.shape[-2]), int(image.shape[-1])
    side = max(1, int(longest_side), height, width)
    if block_size > 1:
        side = int(math.ceil(float(side) / float(block_size)) * block_size)
    tensor = image.new_zeros((1, image.shape[0], side, side))
    mask = torch.ones((1, side, side), dtype=torch.bool, device=image.device)
    tensor[0, :, :height, :width] = image
    mask[0, :height, :width] = False
    return NestedTensor(tensor, mask)


def _merge_predictions(
    *,
    predictions: list[dict[str, torch.Tensor]],
    full_size: tuple[int, int],
    orig_size: torch.Tensor,
    score_threshold: float,
    max_predictions: int,
    nms_threshold: float,
    merge_metric: MergeMetric,
    segmentation: bool,
    source_aware_duplicate_suppression: bool = False,
    return_diagnostics: bool = False,
) -> dict[str, torch.Tensor]:
    """Merge one or more full-image prediction dictionaries."""
    non_empty = [prediction for prediction in predictions if prediction["scores"].numel() > 0]
    if not non_empty:
        return _empty_result(device=orig_size.device, size=orig_size if segmentation else None)

    boxes = torch.cat([prediction["boxes"] for prediction in non_empty])
    scores = torch.cat([prediction["scores"] for prediction in non_empty])
    labels = torch.cat([prediction["labels"] for prediction in non_empty])
    source_windows = None
    if all("_source_windows" in prediction for prediction in non_empty):
        source_windows = torch.cat([prediction["_source_windows"] for prediction in non_empty])
    source_stages = None
    if all("_source_stages" in prediction for prediction in non_empty):
        source_stages = torch.cat([prediction["_source_stages"] for prediction in non_empty])
    diagnostic_tensors: dict[str, torch.Tensor] = {}
    if return_diagnostics:
        for key in (
            "_local_boxes",
            "_model_sizes",
            "_resize_scales",
            "_resize_paddings",
            "_inference_windows",
            "_valid_sizes",
            "_target_sizes",
        ):
            if all(key in prediction for prediction in non_empty):
                diagnostic_tensors[key] = torch.cat([prediction[key] for prediction in non_empty])
    masks = None
    if all("masks" in prediction for prediction in non_empty):
        masks = torch.cat([prediction["masks"] for prediction in non_empty])

    keep = torch.nonzero(scores >= float(score_threshold), as_tuple=False).flatten()
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    if source_windows is not None:
        source_windows = source_windows[keep]
    if source_stages is not None:
        source_stages = source_stages[keep]
    diagnostic_tensors = {key: value[keep] for key, value in diagnostic_tensors.items()}
    if masks is not None:
        masks = masks[keep]
    if scores.numel() == 0:
        return _empty_result(device=boxes.device, size=orig_size if segmentation else None)

    keep = _classwise_nms(
        boxes=boxes,
        scores=scores,
        labels=labels,
        threshold=nms_threshold,
        metric=merge_metric,
        masks=masks,
    )
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    if source_windows is not None:
        source_windows = source_windows[keep]
    if source_stages is not None:
        source_stages = source_stages[keep]
    diagnostic_tensors = {key: value[keep] for key, value in diagnostic_tensors.items()}
    if masks is not None:
        masks = masks[keep]
    if source_aware_duplicate_suppression and source_windows is not None:
        keep = _source_aware_duplicate_suppression(
            boxes=boxes,
            scores=scores,
            labels=labels,
            source_windows=source_windows,
            center_margin_px=10.0,
            center_margin_ratio=0.15,
        )
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        source_windows = source_windows[keep]
        if source_stages is not None:
            source_stages = source_stages[keep]
        diagnostic_tensors = {key: value[keep] for key, value in diagnostic_tensors.items()}
        if masks is not None:
            masks = masks[keep]
    if scores.numel() > max_predictions:
        keep = torch.argsort(scores, descending=True)[:max_predictions]
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        if source_windows is not None:
            source_windows = source_windows[keep]
        if source_stages is not None:
            source_stages = source_stages[keep]
        diagnostic_tensors = {key: value[keep] for key, value in diagnostic_tensors.items()}
        if masks is not None:
            masks = masks[keep]
    result: dict[str, torch.Tensor] = {"boxes": boxes, "scores": scores, "labels": labels}
    if source_windows is not None:
        result["_source_windows"] = source_windows
    if source_stages is not None:
        result["_source_stages"] = source_stages
    result.update(diagnostic_tensors)
    if masks is not None:
        result["masks"] = masks
    return _scale_to_original_size(result, image_size=full_size, orig_size=orig_size, segmentation=segmentation)


def _shift_tile_result(
    result: dict[str, torch.Tensor],
    *,
    window: TileWindow,
    full_size: tuple[int, int],
    score_threshold: float,
    max_predictions: int,
    segmentation: bool,
) -> dict[str, torch.Tensor]:
    """Move one tile's predictions into full-image coordinates."""
    device = result.get("boxes", torch.empty(0)).device
    boxes = result.get("boxes", torch.empty((0, 4), dtype=torch.float32, device=device)).detach().float()
    scores = result.get("scores", torch.empty(0, dtype=torch.float32, device=device)).detach().float()
    labels = result.get("labels", torch.empty(0, dtype=torch.long, device=device)).detach().long()
    masks = result.get("masks")
    if scores.numel() == 0:
        return _empty_result(device=device, size=torch.tensor(full_size, device=device) if segmentation else None)
    keep = torch.isfinite(scores) & torch.isfinite(boxes).all(dim=1) & (scores >= float(score_threshold))
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    if torch.is_tensor(masks):
        masks = _normalize_tile_masks(masks.detach(), size=(window.y1 - window.y0, window.x1 - window.x0))
        masks = masks[keep] if masks.shape[:1] == keep.shape[:1] else None
    else:
        masks = None
    if scores.numel() > max_predictions:
        top = torch.topk(scores, max_predictions).indices
        boxes = boxes[top]
        scores = scores[top]
        labels = labels[top]
        if masks is not None:
            masks = masks[top]
    tile_h = float(window.y1 - window.y0)
    tile_w = float(window.x1 - window.x0)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0.0, tile_w) + float(window.x0)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0.0, tile_h) + float(window.y0)
    full_h, full_w = full_size
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0.0, float(full_w))
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0.0, float(full_h))
    valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid_boxes]
    scores = scores[valid_boxes]
    labels = labels[valid_boxes]
    shifted: dict[str, torch.Tensor] = {"boxes": boxes, "scores": scores, "labels": labels}
    if masks is not None:
        shifted["masks"] = _paste_tile_masks(masks[valid_boxes], window=window, full_size=full_size)
    return shifted


def _shift_project_detection_batch(
    *,
    tile_results: list[dict[str, torch.Tensor]],
    model_sizes: list[tuple[int, int]],
    source_sizes: list[tuple[int, int]],
    windows: list[TileWindow],
    inference_windows: list[TileWindow],
    valid_sizes: list[tuple[int, int]],
    target_sizes: list[tuple[int, int]],
    resize_scales: list[tuple[float, float]],
    resize_paddings: list[tuple[float, float, float, float]],
    full_size: tuple[int, int],
    score_threshold: float,
    max_predictions: int,
    return_diagnostics: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Shift and project a detection-only tile batch with batched tensor math."""
    boxes_parts: list[torch.Tensor] = []
    scores_parts: list[torch.Tensor] = []
    labels_parts: list[torch.Tensor] = []
    tile_index_parts: list[torch.Tensor] = []
    for tile_index, result in enumerate(tile_results):
        boxes = result.get("boxes")
        scores = result.get("scores")
        labels = result.get("labels")
        if not torch.is_tensor(boxes) or not torch.is_tensor(scores) or not torch.is_tensor(labels):
            continue
        if scores.numel() == 0:
            continue
        boxes = boxes.detach()
        scores = scores.detach()
        labels = labels.detach()
        if boxes.dtype != torch.float32:
            boxes = boxes.float()
        if scores.dtype != torch.float32:
            scores = scores.float()
        labels = labels.long()
        keep = torch.isfinite(scores) & torch.isfinite(boxes).all(dim=1) & (scores >= float(score_threshold))
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        if scores.numel() == 0:
            continue
        if scores.numel() > max_predictions:
            top = torch.topk(scores, max_predictions).indices
            boxes = boxes[top]
            scores = scores[top]
            labels = labels[top]
        boxes_parts.append(boxes)
        scores_parts.append(scores)
        labels_parts.append(labels)
        tile_index_parts.append(torch.full((scores.shape[0],), tile_index, dtype=torch.long, device=scores.device))

    if not scores_parts:
        return _empty_result(device=device)

    boxes = torch.cat(boxes_parts)
    scores = torch.cat(scores_parts)
    labels = torch.cat(labels_parts)
    tile_indices = torch.cat(tile_index_parts)
    metadata_device = boxes.device
    model_size_tensor = boxes.new_tensor(model_sizes)
    source_size_tensor = boxes.new_tensor(source_sizes)
    window_tensor = boxes.new_tensor([(window.x0, window.y0, window.x1, window.y1) for window in windows])
    inference_window_tensor = boxes.new_tensor(
        [(window.x0, window.y0, window.x1, window.y1) for window in inference_windows]
    )

    model_hw = model_size_tensor[tile_indices]
    local_boxes = boxes.clone()
    local_boxes[:, [0, 2]] = torch.minimum(local_boxes[:, [0, 2]].clamp_min(0.0), model_hw[:, 1:2])
    local_boxes[:, [1, 3]] = torch.minimum(local_boxes[:, [1, 3]].clamp_min(0.0), model_hw[:, 0:1])

    source_hw = source_size_tensor[tile_indices].clamp(min=1.0)
    inference_xyxy = inference_window_tensor[tile_indices]
    target_wh = (inference_xyxy[:, 2:4] - inference_xyxy[:, 0:2]).clamp(min=0.0)
    scale_x = target_wh[:, 0] / source_hw[:, 1]
    scale_y = target_wh[:, 1] / source_hw[:, 0]
    projected_boxes = local_boxes.clone()
    projected_boxes[:, [0, 2]] = projected_boxes[:, [0, 2]] * scale_x[:, None] + inference_xyxy[:, 0:1]
    projected_boxes[:, [1, 3]] = projected_boxes[:, [1, 3]] * scale_y[:, None] + inference_xyxy[:, 1:2]
    full_h, full_w = full_size
    projected_boxes[:, [0, 2]] = projected_boxes[:, [0, 2]].clamp(0.0, float(full_w))
    projected_boxes[:, [1, 3]] = projected_boxes[:, [1, 3]].clamp(0.0, float(full_h))
    valid_boxes = (projected_boxes[:, 2] > projected_boxes[:, 0]) & (projected_boxes[:, 3] > projected_boxes[:, 1])
    projected_boxes = projected_boxes[valid_boxes]
    if projected_boxes.numel() == 0:
        return _empty_result(device=metadata_device)

    tile_indices = tile_indices[valid_boxes]
    result: dict[str, torch.Tensor] = {
        "boxes": projected_boxes,
        "scores": scores[valid_boxes],
        "labels": labels[valid_boxes],
        "_source_windows": window_tensor[tile_indices],
    }
    if return_diagnostics:
        result["_local_boxes"] = local_boxes[valid_boxes]
        result["_model_sizes"] = model_size_tensor[tile_indices]
        result["_resize_scales"] = boxes.new_tensor(resize_scales)[tile_indices]
        result["_resize_paddings"] = boxes.new_tensor(resize_paddings)[tile_indices]
        result["_inference_windows"] = inference_window_tensor[tile_indices]
        result["_valid_sizes"] = boxes.new_tensor(valid_sizes)[tile_indices]
        result["_target_sizes"] = boxes.new_tensor(target_sizes)[tile_indices]
    return result


def _project_local_result_to_window(
    result: dict[str, torch.Tensor],
    *,
    source_size: tuple[int, int],
    window: TileWindow,
    full_size: tuple[int, int],
    segmentation: bool,
) -> dict[str, torch.Tensor]:
    """Project local crop predictions back to full-image coordinates."""
    scores = result["scores"]
    if scores.numel() == 0:
        return _empty_result(
            device=scores.device,
            size=torch.tensor(full_size, device=scores.device) if segmentation else None,
        )
    source_h, source_w = source_size
    target_h = window.y1 - window.y0
    target_w = window.x1 - window.x0
    scale_x = float(target_w) / max(float(source_w), 1.0)
    scale_y = float(target_h) / max(float(source_h), 1.0)
    boxes = result["boxes"].clone() if "_local_boxes" in result else result["boxes"]
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x + float(window.x0)
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y + float(window.y0)
    full_h, full_w = full_size
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0.0, float(full_w))
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0.0, float(full_h))
    valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    projected: dict[str, torch.Tensor] = {
        "boxes": boxes[valid_boxes],
        "scores": scores[valid_boxes],
        "labels": result["labels"][valid_boxes],
    }
    if "_local_boxes" in result:
        projected["_local_boxes"] = result["_local_boxes"][valid_boxes]
    masks = result.get("masks")
    if torch.is_tensor(masks):
        masks = masks[valid_boxes]
        if masks.numel() > 0:
            local_masks = F.interpolate(
                masks.unsqueeze(1).float(),
                size=(target_h, target_w),
                mode="nearest",
            ).squeeze(1)
            projected["masks"] = _paste_tile_masks(local_masks.bool(), window=window, full_size=full_size)
    return projected


def _normalize_tile_masks(masks: torch.Tensor, *, size: tuple[int, int]) -> torch.Tensor:
    """Convert postprocessor mask output to boolean ``N,H,W`` tile masks."""
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks.squeeze(1)
    if masks.ndim != 3:
        return torch.empty((0, size[0], size[1]), dtype=torch.bool, device=masks.device)
    if tuple(masks.shape[-2:]) != size:
        masks = F.interpolate(masks.unsqueeze(1).float(), size=size, mode="nearest").squeeze(1)
    return masks.bool()


def _paste_tile_masks(
    masks: torch.Tensor,
    *,
    window: TileWindow,
    full_size: tuple[int, int],
) -> torch.Tensor:
    """Paste tile masks into full-image mask tensors."""
    full_h, full_w = full_size
    pasted = torch.zeros((masks.shape[0], full_h, full_w), dtype=torch.bool, device=masks.device)
    pasted[:, window.y0 : window.y1, window.x0 : window.x1] = masks[:, : window.y1 - window.y0, : window.x1 - window.x0]
    return pasted


def _classwise_nms(
    *,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
    metric: MergeMetric,
    masks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run class-wise duplicate suppression."""
    keep_parts: list[torch.Tensor] = []
    for label in labels.unique():
        class_indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        if metric == "cdn":
            local_keep = _box_cluster_diou_nms(boxes[class_indices], scores[class_indices], threshold=threshold)
        elif metric == "diou":
            local_keep = _box_diou_nms(boxes[class_indices], scores[class_indices], threshold=threshold)
        elif masks is not None:
            mask_metric: Literal["iou", "ios"] = "ios" if metric == "ios" else "iou"
            local_keep = _mask_nms(masks[class_indices], scores[class_indices], threshold=threshold, metric=mask_metric)
        elif metric == "ios":
            local_keep = _box_ios_nms(boxes[class_indices], scores[class_indices], threshold=threshold)
        else:
            local_keep = nms(boxes[class_indices], scores[class_indices], threshold)
        keep_parts.append(class_indices[local_keep])
    if not keep_parts:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    keep = torch.cat(keep_parts)
    return keep[torch.argsort(scores[keep], descending=True)]


def _source_aware_duplicate_suppression(
    *,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    source_windows: torch.Tensor,
    center_margin_px: float,
    center_margin_ratio: float,
) -> torch.Tensor:
    """Suppress same-class duplicates produced by different source windows."""
    if scores.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    keep_mask = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
    order = torch.argsort(scores, descending=True)
    for order_position, index in enumerate(order):
        if not keep_mask[index]:
            continue
        remaining = order[order_position + 1 :]
        if remaining.numel() == 0:
            continue
        candidates = remaining[
            keep_mask[remaining]
            & (labels[remaining] == labels[index])
            & ((source_windows[remaining] != source_windows[index]).any(dim=1))
        ]
        if candidates.numel() == 0:
            continue
        duplicate = _mutual_center_in_expanded_boxes(
            boxes[index : index + 1],
            boxes[candidates],
            margin_px=center_margin_px,
            margin_ratio=center_margin_ratio,
        ).flatten()
        keep_mask[candidates[duplicate]] = False
    return torch.nonzero(keep_mask, as_tuple=False).flatten()


def _mutual_center_in_expanded_boxes(
    reference_boxes: torch.Tensor,
    candidate_boxes: torch.Tensor,
    *,
    margin_px: float,
    margin_ratio: float,
) -> torch.Tensor:
    """Return candidates whose center and reference center fall inside each other."""
    reference_contains = _center_in_expanded_boxes(
        centers=_box_centers(candidate_boxes),
        boxes=reference_boxes.expand(candidate_boxes.shape[0], 4),
        margin_px=margin_px,
        margin_ratio=margin_ratio,
    )
    candidate_contains = _center_in_expanded_boxes(
        centers=_box_centers(reference_boxes).expand(candidate_boxes.shape[0], 2),
        boxes=candidate_boxes,
        margin_px=margin_px,
        margin_ratio=margin_ratio,
    )
    return reference_contains & candidate_contains


def _box_centers(boxes: torch.Tensor) -> torch.Tensor:
    """Return xy centers for xyxy boxes."""
    return (boxes[:, :2] + boxes[:, 2:]) * 0.5


def _center_in_expanded_boxes(
    *,
    centers: torch.Tensor,
    boxes: torch.Tensor,
    margin_px: float,
    margin_ratio: float,
) -> torch.Tensor:
    """Return whether centers fall inside boxes expanded by a size-aware margin."""
    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0.0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0.0)
    margin_x = torch.maximum(
        widths * float(margin_ratio),
        widths.new_full(widths.shape, float(margin_px)),
    )
    margin_y = torch.maximum(
        heights * float(margin_ratio),
        heights.new_full(heights.shape, float(margin_px)),
    )
    return (
        (centers[:, 0] >= boxes[:, 0] - margin_x)
        & (centers[:, 0] <= boxes[:, 2] + margin_x)
        & (centers[:, 1] >= boxes[:, 1] - margin_y)
        & (centers[:, 1] <= boxes[:, 3] + margin_y)
    )


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU for ``xyxy`` boxes."""
    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))[:, None]
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))[None, :]
    return inter / (area1 + area2 - inter).clamp(min=1e-6)


def _box_ios_nms(boxes: torch.Tensor, scores: torch.Tensor, *, threshold: float) -> torch.Tensor:
    """Greedy NMS using intersection-over-smaller-box."""
    order = torch.argsort(scores, descending=True)
    keep: list[torch.Tensor] = []
    while order.numel() > 0:
        current = order[:1]
        keep.append(current)
        if order.numel() == 1:
            break
        rest = order[1:]
        overlap = _box_ios(boxes[current], boxes[rest]).squeeze(0)
        order = rest[overlap <= threshold]
    return torch.cat(keep) if keep else torch.empty(0, dtype=torch.long, device=scores.device)


def _box_diou_nms(boxes: torch.Tensor, scores: torch.Tensor, *, threshold: float) -> torch.Tensor:
    """Greedy NMS using DIoU as the suppression criterion."""
    order = torch.argsort(scores, descending=True)
    keep: list[torch.Tensor] = []
    while order.numel() > 0:
        current = order[:1]
        keep.append(current)
        if order.numel() == 1:
            break
        rest = order[1:]
        overlap = _box_diou(boxes[current], boxes[rest]).squeeze(0)
        order = rest[overlap <= threshold]
    return torch.cat(keep) if keep else torch.empty(0, dtype=torch.long, device=scores.device)


def _box_cluster_diou_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    *,
    threshold: float,
    block_size: int = 1024,
) -> torch.Tensor:
    """Run score-ordered Cluster-DIoU-NMS with bounded device-local workspace.

    The strictly upper-triangular DIoU matrix is evaluated in column blocks so
    tiled validation does not allocate a dense ``N x N`` tensor. A box is
    suppressed when any earlier score-sorted box exceeds the threshold, matching
    the D-FINE tiled-inference reference. Returned indices are local to the input
    tensors and remain score-sorted.
    """
    if scores.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    order = torch.argsort(scores, descending=True)
    sorted_boxes = boxes[order]
    count = int(order.numel())
    block_size = max(1, int(block_size))
    suppressed = torch.zeros(count, dtype=torch.bool, device=boxes.device)
    for column_start in range(0, count, block_size):
        column_end = min(column_start + block_size, count)
        overlap = _box_diou(
            sorted_boxes[:column_end],
            sorted_boxes[column_start:column_end],
        )
        row_ids = torch.arange(column_end, device=boxes.device)[:, None]
        column_ids = torch.arange(column_start, column_end, device=boxes.device)[None, :]
        suppressed[column_start:column_end] = (
            (overlap > threshold) & (row_ids < column_ids)
        ).any(dim=0)
    return order[~suppressed]


def _box_diou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Distance-IoU for ``xyxy`` boxes."""
    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))[:, None]
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))[None, :]
    union = (area1 + area2 - inter).clamp(min=1e-6)
    iou = inter / union

    center1 = (boxes1[:, :2] + boxes1[:, 2:]) / 2.0
    center2 = (boxes2[:, :2] + boxes2[:, 2:]) / 2.0
    center_distance = ((center1[:, None, :] - center2[None, :, :]) ** 2).sum(dim=-1)
    enclosing_lt = torch.minimum(boxes1[:, None, :2], boxes2[:, :2])
    enclosing_rb = torch.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
    enclosing_wh = (enclosing_rb - enclosing_lt).clamp(min=0)
    enclosing_diagonal = (enclosing_wh[..., 0] ** 2 + enclosing_wh[..., 1] ** 2).clamp(min=1e-6)
    return iou - center_distance / enclosing_diagonal


def _box_ios(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute intersection over smaller box area."""
    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))[:, None]
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))[None, :]
    return inter / torch.minimum(area1, area2).clamp(min=1e-6)


def _mask_nms(
    masks: torch.Tensor,
    scores: torch.Tensor,
    *,
    threshold: float,
    metric: MergeMetric,
) -> torch.Tensor:
    """Greedy mask NMS using mask IoU or IoS."""
    if masks.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    order = torch.argsort(scores, descending=True)
    keep: list[torch.Tensor] = []
    flat_masks = masks.flatten(1).bool()
    areas = flat_masks.sum(dim=1).float()
    while order.numel() > 0:
        current = order[:1]
        keep.append(current)
        if order.numel() == 1:
            break
        rest = order[1:]
        overlap = _mask_overlap(
            flat_masks[current],
            flat_masks[rest],
            areas[current],
            areas[rest],
            metric=metric,
        ).squeeze(0)
        order = rest[overlap <= threshold]
    return torch.cat(keep) if keep else torch.empty(0, dtype=torch.long, device=scores.device)


def _mask_overlap(
    masks1: torch.Tensor,
    masks2: torch.Tensor,
    areas1: torch.Tensor,
    areas2: torch.Tensor,
    *,
    metric: MergeMetric,
) -> torch.Tensor:
    """Compute pairwise mask IoU or IoS for flattened boolean masks."""
    inter = (masks1.float() @ masks2.float().T).float()
    if metric == "ios":
        denom = torch.minimum(areas1[:, None], areas2[None, :])
    else:
        denom = areas1[:, None] + areas2[None, :] - inter
    return inter / denom.clamp(min=1e-6)


def _scale_to_original_size(
    result: dict[str, torch.Tensor],
    *,
    image_size: tuple[int, int],
    orig_size: torch.Tensor,
    segmentation: bool,
) -> dict[str, torch.Tensor]:
    """Scale merged predictions from validation image size to original metric size."""
    image_h, image_w = image_size
    orig_h, orig_w = [int(value) for value in orig_size.detach().cpu().tolist()]
    if image_h <= 0 or image_w <= 0 or (image_h == orig_h and image_w == orig_w):
        if segmentation and "masks" not in result:
            result["masks"] = torch.empty((0, orig_h, orig_w), dtype=torch.bool, device=result["boxes"].device)
        return result
    scale = result["boxes"].new_tensor([orig_w / image_w, orig_h / image_h, orig_w / image_w, orig_h / image_h])
    result["boxes"] = result["boxes"] * scale
    if "_source_windows" in result:
        result["_source_windows"] = result["_source_windows"] * scale
    if "_inference_windows" in result:
        result["_inference_windows"] = result["_inference_windows"] * scale
    if "masks" in result:
        result["masks"] = (
            F.interpolate(result["masks"].unsqueeze(1).float(), size=(orig_h, orig_w), mode="nearest")
            .squeeze(1)
            .bool()
        )
    elif segmentation:
        result["masks"] = torch.empty((0, orig_h, orig_w), dtype=torch.bool, device=result["boxes"].device)
    return result


def _empty_result(*, device: torch.device, size: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Return an empty prediction dictionary."""
    result = {
        "boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
        "scores": torch.empty(0, dtype=torch.float32, device=device),
        "labels": torch.empty(0, dtype=torch.long, device=device),
    }
    if size is not None:
        height, width = [int(value) for value in size.detach().cpu().tolist()]
        result["masks"] = torch.empty((0, height, width), dtype=torch.bool, device=device)
    return result
