# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""ASAHI adaptive tiled validation."""

from __future__ import annotations

import math
from typing import Any, Literal

import torch

from rfdetr.evaluation.tiled_core import (
    SOURCE_STAGE_ASAHI,
    MergeMetric,
    TileInputDType,
    TileMemoryFormat,
    TileWindow,
    TiledTimingStats,
    WindowResizePolicy,
    predict_windows,
)
from rfdetr.utilities.tensors import NestedTensor

AsahiSourceMode = Literal["adaptive", "full", "full_adaptive"]


def generate_asahi_windows(
    *,
    height: int,
    width: int,
    short_side_threshold: int,
    low_patch_count: int,
    high_patch_count: int,
    overlap_ratio: float,
    include_full_image: bool,
) -> list[TileWindow]:
    """Generate ASAHI-style adaptive grid windows.

    Args:
        height: Image height.
        width: Image width.
        short_side_threshold: Limiting-dimension cutoff for switching from low to high patch count.
        low_patch_count: Patch count used below the threshold. Supported value is ``6``.
        high_patch_count: Patch count used at or above the threshold. Supported value is ``12``.
        overlap_ratio: Approximate overlap ratio between neighboring grid cells.
        include_full_image: Whether to include the full image as an additional context window.

    Returns:
        Full-image context window plus adaptive grid windows when requested.
    """
    if height <= 0 or width <= 0:
        return []
    patch_count = low_patch_count if max(height, width) <= short_side_threshold else high_patch_count
    rows, cols = _asahi_grid_shape(height=height, width=width, patch_count=patch_count)
    windows = _grid_windows(height=height, width=width, rows=rows, cols=cols, overlap_ratio=overlap_ratio)
    if include_full_image:
        return [TileWindow(0, 0, width, height), *windows]
    return windows


def predict_asahi(
    *,
    model: torch.nn.Module,
    postprocess: Any,
    samples: NestedTensor,
    targets: list[dict[str, torch.Tensor]],
    short_side_threshold: int,
    low_patch_count: int,
    high_patch_count: int,
    overlap_ratio: float,
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
    window_resize_longest_side: int | None = None,
    window_resize_policy: WindowResizePolicy = "square_stretch",
    source_mode: AsahiSourceMode | None = None,
    tile_input_dtype: TileInputDType = "auto",
    tile_memory_format: TileMemoryFormat = "contiguous",
    batch_across_images: bool = False,
    return_diagnostics: bool = False,
    timing_stats: TiledTimingStats | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Run ASAHI-style adaptive tiled validation."""
    resolved_source_mode: AsahiSourceMode
    if source_mode is None:
        resolved_source_mode = "full_adaptive" if include_full_image else "adaptive"
    else:
        resolved_source_mode = source_mode
    if resolved_source_mode not in {"adaptive", "full", "full_adaptive"}:
        raise ValueError("ASAHI source_mode must be 'adaptive', 'full', or 'full_adaptive'.")

    windows_per_image: list[list[TileWindow]] = []
    if resolved_source_mode in {"adaptive", "full_adaptive"}:
        for target in targets:
            image_size = target.get("size", target["orig_size"])
            image_h, image_w = [int(value) for value in image_size.detach().cpu().tolist()]
            windows_per_image.append(
                generate_asahi_windows(
                    height=image_h,
                    width=image_w,
                    short_side_threshold=short_side_threshold,
                    low_patch_count=low_patch_count,
                    high_patch_count=high_patch_count,
                    overlap_ratio=overlap_ratio,
                    include_full_image=False,
                )
            )
    else:
        windows_per_image = [[] for _target in targets]

    resolved_full_image_size = full_image_size or short_side_threshold
    resolved_window_resize_longest_side = window_resize_longest_side or resolved_full_image_size
    include_resized_full_image = resolved_source_mode in {"full", "full_adaptive"}
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
        full_image_size=resolved_full_image_size if include_resized_full_image else None,
        window_resize_longest_side=resolved_window_resize_longest_side,
        window_resize_policy=window_resize_policy,
        window_source_stage=SOURCE_STAGE_ASAHI,
        tile_input_dtype=tile_input_dtype,
        tile_memory_format=tile_memory_format,
        batch_across_images=batch_across_images,
        return_diagnostics=return_diagnostics,
        timing_stats=timing_stats,
    )


def _asahi_grid_shape(*, height: int, width: int, patch_count: int) -> tuple[int, int]:
    """Return ASAHI grid rows/cols for supported patch counts."""
    landscape = width >= height
    if patch_count == 6:
        return (2, 3) if landscape else (3, 2)
    if patch_count == 12:
        return (3, 4) if landscape else (4, 3)
    raise ValueError("ASAHI patch counts must be 6 or 12.")


def _grid_windows(
    *,
    height: int,
    width: int,
    rows: int,
    cols: int,
    overlap_ratio: float,
) -> list[TileWindow]:
    """Generate a fixed-count overlapping grid covering the full image."""
    tile_h = _grid_tile_length(length=height, cells=rows, overlap_ratio=overlap_ratio)
    tile_w = _grid_tile_length(length=width, cells=cols, overlap_ratio=overlap_ratio)
    y_starts = _even_starts(length=height, tile_length=tile_h, cells=rows)
    x_starts = _even_starts(length=width, tile_length=tile_w, cells=cols)
    return [
        TileWindow(x0=x0, y0=y0, x1=min(x0 + tile_w, width), y1=min(y0 + tile_h, height))
        for y0 in y_starts
        for x0 in x_starts
    ]


def _grid_tile_length(*, length: int, cells: int, overlap_ratio: float) -> int:
    """Compute ASAHI tile length for a fixed-count grid with approximate overlap."""
    if cells <= 1:
        return length
    denominator = float(cells) - (float(cells) - 1.0) * float(overlap_ratio)
    return min(length, max(1, int(math.ceil(float(length) / max(denominator, 1e-6)) + 1)))


def _even_starts(*, length: int, tile_length: int, cells: int) -> list[int]:
    """Return exactly ``cells`` starts from zero to the last valid border start."""
    if cells <= 1 or length <= tile_length:
        return [0]
    last_start = max(0, length - tile_length)
    return [int(round(float(last_start) * index / float(cells - 1))) for index in range(cells)]
