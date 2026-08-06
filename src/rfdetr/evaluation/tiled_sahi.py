# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Fixed-grid SAHI tiled validation."""

from __future__ import annotations

from typing import Any

import torch

from rfdetr.evaluation.tiled_core import (
    MergeMetric,
    TileInputDType,
    TileMemoryFormat,
    TileWindow,
    TiledTimingStats,
    WindowResizePolicy,
    generate_tile_windows,
    predict_windows,
)
from rfdetr.utilities.tensors import NestedTensor


def predict_tiled(
    *,
    model: torch.nn.Module,
    postprocess: Any,
    samples: NestedTensor,
    targets: list[dict[str, torch.Tensor]],
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
    include_full_image: bool,
    nms_threshold: float,
    merge_metric: MergeMetric,
    score_threshold: float,
    max_predictions: int,
    tile_batch_size: int,
    block_size: int,
    segmentation: bool,
    source_aware_duplicate_suppression: bool = False,
    full_image_size: int | None = None,
    window_resize_policy: WindowResizePolicy = "aspect_longest_side",
    tile_input_dtype: TileInputDType = "auto",
    tile_memory_format: TileMemoryFormat = "contiguous",
    batch_across_images: bool = False,
    return_diagnostics: bool = False,
    timing_stats: TiledTimingStats | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Run tiled validation directly on normalized tensors.

    Args:
        model: RF-DETR module to evaluate.
        postprocess: RF-DETR postprocessor callable.
        samples: Batched validation images.
        targets: Per-image metric targets.
        slice_height: Tile height.
        slice_width: Tile width.
        overlap_height_ratio: Fractional vertical tile overlap.
        overlap_width_ratio: Fractional horizontal tile overlap.
        include_full_image: Whether to include the full image as an additional context window.
        nms_threshold: Duplicate suppression threshold.
        merge_metric: Duplicate suppression metric: ``"iou"``, ``"ios"``, ``"diou"``, or ``"cdn"``.
        score_threshold: Minimum score retained for metrics.
        max_predictions: Maximum predictions returned per image.
        tile_batch_size: Number of tiles per model forward.
        block_size: Backbone padding multiple.
        segmentation: Whether segmentation masks are expected.
        full_image_size: Optional longest side for the resized full-image context pass.
        window_resize_policy: How adaptive crops are resized before prediction.
        tile_input_dtype: Optional dtype conversion for tile tensors before model forward.
        tile_memory_format: Optional memory format conversion for 4D tile batches.
        batch_across_images: Whether to batch tiles across images when supported.
        return_diagnostics: Whether to attach debug-only projection metadata to predictions.
        timing_stats: Optional debug timing accumulator.

    Returns:
        Postprocessed, merged prediction dictionaries.
    """
    windows_per_image: list[list[TileWindow]] = []
    for target in targets:
        image_size = target.get("size", target["orig_size"])
        image_h, image_w = [int(value) for value in image_size.detach().cpu().tolist()]
        windows = generate_tile_windows(
            height=image_h,
            width=image_w,
            tile_height=slice_height,
            tile_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
        )
        if include_full_image:
            full_window = TileWindow(0, 0, image_w, image_h)
            windows = [window for window in windows if window != full_window]
        windows_per_image.append(windows)
    resolved_full_image_size = full_image_size or max(slice_height, slice_width)
    return predict_windows(
        model=model,
        postprocess=postprocess,
        samples=samples,
        targets=targets,
        windows_per_image=windows_per_image,
        nms_threshold=nms_threshold,
        merge_metric=merge_metric,
        source_aware_duplicate_suppression=source_aware_duplicate_suppression,
        score_threshold=score_threshold,
        max_predictions=max_predictions,
        tile_batch_size=tile_batch_size,
        block_size=block_size,
        segmentation=segmentation,
        full_image_size=resolved_full_image_size if include_full_image else None,
        tile_input_dtype=tile_input_dtype,
        tile_memory_format=tile_memory_format,
        batch_across_images=batch_across_images,
        return_diagnostics=return_diagnostics,
        timing_stats=timing_stats,
    )
