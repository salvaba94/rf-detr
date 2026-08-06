# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""GSAHI coarse-to-fine tiled validation."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Literal

import torch

from rfdetr.evaluation.tiled_core import (
    SOURCE_STAGE_COARSE,
    SOURCE_STAGE_FINE,
    MergeMetric,
    TileInputDType,
    TileMemoryFormat,
    TileWindow,
    TiledTimingStats,
    _collect_resized_full_image_predictions,
    _collect_window_predictions,
    _empty_result,
    _merge_predictions,
    generate_tile_windows,
)
from rfdetr.utilities.tensors import NestedTensor

GsahiMergeSources = Literal["full_fine", "full_coarse_fine", "fine"]


def predict_gsahi(
    *,
    model: torch.nn.Module,
    postprocess: Any,
    samples: NestedTensor,
    targets: list[dict[str, torch.Tensor]],
    coarse_slice_size: int,
    fine_slice_size: int,
    coarse_overlap: float,
    fine_overlap: float,
    include_full_image: bool,
    roi_score_threshold: float,
    roi_expansion_ratio: float,
    roi_max_regions: int,
    nms_threshold: float,
    merge_metric: MergeMetric,
    score_threshold: float,
    max_predictions: int,
    tile_batch_size: int,
    block_size: int,
    segmentation: bool,
    source_aware_duplicate_suppression: bool = False,
    full_image_size: int | None = None,
    merge_sources: GsahiMergeSources = "full_fine",
    tile_input_dtype: TileInputDType = "auto",
    tile_memory_format: TileMemoryFormat = "contiguous",
    return_diagnostics: bool = False,
    timing_stats: TiledTimingStats | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Run GSAHI-style coarse-to-fine guided tiled validation."""
    results: list[dict[str, torch.Tensor]] = []
    images = samples.tensors
    tile_batch_size = max(1, int(tile_batch_size))
    for image_tensor, target in zip(images, targets):
        image_size = target.get("size", target["orig_size"])
        image_h, image_w = [int(value) for value in image_size.detach().cpu().tolist()]
        image = image_tensor[:, :image_h, :image_w]
        full_size = (image_h, image_w)
        coarse_windows = generate_tile_windows(
            height=image_h,
            width=image_w,
            tile_height=coarse_slice_size,
            tile_width=coarse_slice_size,
            overlap_height_ratio=coarse_overlap,
            overlap_width_ratio=coarse_overlap,
        )
        uses_full_predictions = include_full_image and merge_sources in {"full_fine", "full_coarse_fine"}
        if uses_full_predictions:
            full_predictions = _collect_resized_full_image_predictions(
                model=model,
                postprocess=postprocess,
                image=image,
                full_size=full_size,
                resize_longest_side=full_image_size or coarse_slice_size,
                score_threshold=score_threshold,
                max_predictions=max_predictions,
                block_size=block_size,
                segmentation=segmentation,
                return_diagnostics=return_diagnostics,
                timing_stats=timing_stats,
            )
        else:
            full_predictions = _empty_result(
                device=image.device,
                size=torch.tensor(full_size, device=image.device) if segmentation else None,
            )
        coarse_predictions = _collect_window_predictions(
            model=model,
            postprocess=postprocess,
            image=image,
            windows=coarse_windows,
            full_size=full_size,
            score_threshold=score_threshold,
            max_predictions=max_predictions,
            tile_batch_size=tile_batch_size,
            block_size=block_size,
            segmentation=segmentation,
            source_stage=SOURCE_STAGE_COARSE,
            tile_input_dtype=tile_input_dtype,
            tile_memory_format=tile_memory_format,
            return_diagnostics=return_diagnostics,
            timing_stats=timing_stats,
        )
        roi_timer = timing_stats.measure("roi_build", image.device) if timing_stats is not None else nullcontext()
        with roi_timer:
            rois = _build_gsahi_rois(
                predictions=coarse_predictions,
                full_size=full_size,
                score_threshold=roi_score_threshold,
                expansion_ratio=roi_expansion_ratio,
                max_regions=roi_max_regions,
            )
        fine_windows = _gsahi_fine_windows(
            rois=rois,
            full_size=full_size,
            fine_slice_size=fine_slice_size,
            fine_overlap=fine_overlap,
        )
        fine_predictions = _collect_window_predictions(
            model=model,
            postprocess=postprocess,
            image=image,
            windows=fine_windows,
            full_size=full_size,
            score_threshold=score_threshold,
            max_predictions=max_predictions,
            tile_batch_size=tile_batch_size,
            block_size=block_size,
            segmentation=segmentation,
            resize_longest_side=full_image_size or fine_slice_size,
            source_stage=SOURCE_STAGE_FINE,
            tile_input_dtype=tile_input_dtype,
            tile_memory_format=tile_memory_format,
            return_diagnostics=return_diagnostics,
            timing_stats=timing_stats,
        )
        merge_parts = []
        if merge_sources in {"full_fine", "full_coarse_fine"}:
            merge_parts.append(full_predictions)
        if merge_sources == "full_coarse_fine":
            merge_parts.append(coarse_predictions)
        merge_parts.append(fine_predictions)
        if merge_sources == "fine":
            merge_parts = [fine_predictions]
        merge_timer = timing_stats.measure("merge_nms", image.device) if timing_stats is not None else nullcontext()
        with merge_timer:
            merged = _merge_predictions(
                predictions=merge_parts,
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
        results.append({key: value.to(target["boxes"].device) for key, value in merged.items()})
    return results


def _build_gsahi_rois(
    *,
    predictions: dict[str, torch.Tensor],
    full_size: tuple[int, int],
    score_threshold: float,
    expansion_ratio: float,
    max_regions: int,
) -> list[TileWindow]:
    """Build class-agnostic GSAHI refinement ROIs from coarse predictions."""
    scores = predictions["scores"]
    if scores.numel() == 0 or max_regions <= 0:
        return []
    boxes = predictions["boxes"]
    keep = torch.nonzero(scores >= float(score_threshold), as_tuple=False).flatten()
    if keep.numel() == 0:
        return []
    boxes = boxes[keep]
    scores = scores[keep]
    if keep.numel() > max_regions * 4:
        top = torch.topk(scores, max_regions * 4).indices
        boxes = boxes[top]
        scores = scores[top]
    boxes = boxes.detach().cpu()
    scores = scores.detach().cpu()
    scored_rois = _expand_boxes_to_scored_windows(
        boxes=boxes,
        scores=scores,
        full_size=full_size,
        expansion_ratio=expansion_ratio,
    )
    return _merge_roi_windows(scored_rois=scored_rois, max_regions=max_regions)


def _expand_boxes_to_scored_windows(
    *,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    full_size: tuple[int, int],
    expansion_ratio: float,
) -> list[tuple[TileWindow, float]]:
    """Expand detection boxes into clipped integer ROI windows."""
    full_h, full_w = full_size
    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0)
    pad_x = widths * float(expansion_ratio)
    pad_y = heights * float(expansion_ratio)
    x0 = (boxes[:, 0] - pad_x).floor().clamp(0, full_w)
    y0 = (boxes[:, 1] - pad_y).floor().clamp(0, full_h)
    x1 = (boxes[:, 2] + pad_x).ceil().clamp(0, full_w)
    y1 = (boxes[:, 3] + pad_y).ceil().clamp(0, full_h)
    windows: list[tuple[TileWindow, float]] = []
    for left, top, right, bottom, score in zip(x0, y0, x1, y1, scores):
        if right > left and bottom > top:
            windows.append(
                (
                    TileWindow(int(left.item()), int(top.item()), int(right.item()), int(bottom.item())),
                    float(score.item()),
                )
            )
    return windows


def _merge_roi_windows(
    *,
    scored_rois: list[tuple[TileWindow, float]],
    max_regions: int,
) -> list[TileWindow]:
    """Merge overlapping GSAHI ROIs greedily and keep top regions."""
    if not scored_rois:
        return []
    scored_rois = sorted(scored_rois, key=lambda item: item[1], reverse=True)
    merged: list[tuple[TileWindow, float]] = []
    for roi, score in scored_rois:
        merged_index = _find_overlapping_roi(roi, [item[0] for item in merged])
        if merged_index is None:
            merged.append((roi, score))
        else:
            existing, existing_score = merged[merged_index]
            merged[merged_index] = (_union_windows(existing, roi), max(existing_score, score))
    merged.sort(key=lambda item: item[1], reverse=True)
    return [item[0] for item in merged[:max_regions]]


def _find_overlapping_roi(roi: TileWindow, rois: list[TileWindow]) -> int | None:
    """Return index of the first ROI with non-zero overlap."""
    for index, existing in enumerate(rois):
        if _window_intersection_area(roi, existing) > 0:
            return index
    return None


def _window_intersection_area(first: TileWindow, second: TileWindow) -> int:
    """Return intersection area between two windows."""
    width = max(0, min(first.x1, second.x1) - max(first.x0, second.x0))
    height = max(0, min(first.y1, second.y1) - max(first.y0, second.y0))
    return width * height


def _union_windows(first: TileWindow, second: TileWindow) -> TileWindow:
    """Return the union of two windows."""
    return TileWindow(
        x0=min(first.x0, second.x0),
        y0=min(first.y0, second.y0),
        x1=max(first.x1, second.x1),
        y1=max(first.y1, second.y1),
    )


def _gsahi_fine_windows(
    *,
    rois: list[TileWindow],
    full_size: tuple[int, int],
    fine_slice_size: int,
    fine_overlap: float,
) -> list[TileWindow]:
    """Generate fine windows inside GSAHI ROIs."""
    windows: list[TileWindow] = []
    for roi in rois:
        roi = _expand_window_to_min_size(
            roi,
            full_size=full_size,
            min_height=fine_slice_size,
            min_width=fine_slice_size,
        )
        local_windows = generate_tile_windows(
            height=roi.y1 - roi.y0,
            width=roi.x1 - roi.x0,
            tile_height=fine_slice_size,
            tile_width=fine_slice_size,
            overlap_height_ratio=fine_overlap,
            overlap_width_ratio=fine_overlap,
        )
        windows.extend(
            TileWindow(
                x0=roi.x0 + window.x0,
                y0=roi.y0 + window.y0,
                x1=roi.x0 + window.x1,
                y1=roi.y0 + window.y1,
            )
            for window in local_windows
        )
    return windows


def _expand_window_to_min_size(
    window: TileWindow,
    *,
    full_size: tuple[int, int],
    min_height: int,
    min_width: int,
) -> TileWindow:
    """Expand a window around its center to provide enough fine-stage context."""
    full_h, full_w = full_size
    target_w = min(full_w, max(window.x1 - window.x0, int(min_width)))
    target_h = min(full_h, max(window.y1 - window.y0, int(min_height)))
    center_x = (window.x0 + window.x1) * 0.5
    center_y = (window.y0 + window.y1) * 0.5
    x0 = int(round(center_x - target_w * 0.5))
    y0 = int(round(center_y - target_h * 0.5))
    x0 = min(max(0, x0), max(0, full_w - target_w))
    y0 = min(max(0, y0), max(0, full_h - target_h))
    return TileWindow(x0=x0, y0=y0, x1=x0 + target_w, y1=y0 + target_h)
