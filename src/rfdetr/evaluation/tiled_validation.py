# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Compatibility facade for tiled validation helpers.

Method-specific implementations live in ``tiled_sahi``, ``tiled_asahi``, and
``tiled_gsahi``. Common execution, projection, and merge primitives live in
``tiled_core``.
"""

from rfdetr.evaluation.tiled_asahi import AsahiSourceMode, generate_asahi_windows, predict_asahi
from rfdetr.evaluation.tiled_core import (
    SOURCE_STAGE_ASAHI,
    SOURCE_STAGE_COARSE,
    SOURCE_STAGE_FINE,
    SOURCE_STAGE_FULL,
    SOURCE_STAGE_TILE,
    MergeMetric,
    TileWindow,
    WindowResizePolicy,
    _box_cluster_diou_nms,
    _box_diou,
    _box_diou_nms,
    _box_ios,
    _box_ios_nms,
    _box_iou,
    _center_in_expanded_boxes,
    _classwise_nms,
    _collect_resized_full_image_predictions,
    _collect_window_predictions,
    _empty_result,
    _letterbox_nested_tensor,
    _mask_nms,
    _mask_overlap,
    _merge_predictions,
    _mutual_center_in_expanded_boxes,
    _nested_tensor_from_prepared_windows,
    _normalize_tile_masks,
    _paste_tile_masks,
    _prepare_window_tensor,
    _project_local_result_to_window,
    _resize_image_longest_side,
    _scale_to_original_size,
    _shift_tile_result,
    _source_aware_duplicate_suppression,
    _tile_starts,
    generate_tile_windows,
    predict_windows,
)
from rfdetr.evaluation.tiled_gsahi import (
    GsahiMergeSources,
    _build_gsahi_rois,
    _expand_boxes_to_scored_windows,
    _expand_window_to_min_size,
    _find_overlapping_roi,
    _gsahi_fine_windows,
    _merge_roi_windows,
    _union_windows,
    _window_intersection_area,
    predict_gsahi,
)
from rfdetr.evaluation.tiled_sahi import predict_tiled

__all__ = [
    "MergeMetric",
    "WindowResizePolicy",
    "AsahiSourceMode",
    "GsahiMergeSources",
    "SOURCE_STAGE_TILE",
    "SOURCE_STAGE_FULL",
    "SOURCE_STAGE_COARSE",
    "SOURCE_STAGE_FINE",
    "SOURCE_STAGE_ASAHI",
    "TileWindow",
    "generate_tile_windows",
    "generate_asahi_windows",
    "predict_windows",
    "predict_tiled",
    "predict_asahi",
    "predict_gsahi",
    "_prepare_window_tensor",
    "_project_local_result_to_window",
    "_gsahi_fine_windows",
    "_expand_window_to_min_size",
]
