# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for native tiled validation merge behavior."""

import pytest
import torch

from rfdetr.evaluation.tiled_core import (
    _box_cluster_diou_nms,
    _box_diou,
    _classwise_nms,
    _source_aware_duplicate_suppression,
)


def _dense_cluster_diou_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    """Return the D-FINE dense score-ordered DIoU result for comparison."""
    order = torch.argsort(scores, descending=True)
    diou = _box_diou(boxes[order], boxes[order]).triu(diagonal=1)
    return order[~(diou > threshold).any(dim=0)]


@pytest.mark.parametrize("block_size", [1, 7, 64, 1024])
@pytest.mark.parametrize("crowded", [False, True])
def test_blockwise_cluster_diou_matches_dense_reference(block_size: int, crowded: bool) -> None:
    """Blockwise CDN should preserve D-FINE's dense suppression rule exactly."""
    generator = torch.Generator().manual_seed(19)
    count = 137
    if crowded:
        centers = torch.full((count, 2), 500.0) + torch.randn(count, 2, generator=generator) * 4
    else:
        centers = torch.rand(count, 2, generator=generator) * 2000
    sizes = torch.rand(count, 2, generator=generator) * 40 + 2
    boxes = torch.cat((centers - sizes / 2, centers + sizes / 2), dim=1)
    scores = torch.rand(count, generator=generator)

    expected = _dense_cluster_diou_nms(boxes, scores, threshold=0.45)
    actual = _box_cluster_diou_nms(boxes, scores, threshold=0.45, block_size=block_size)

    assert torch.equal(actual, expected)


def test_classwise_mask_nms_uses_mask_iou_and_respects_classes() -> None:
    """Mask NMS should suppress overlapping masks only within the same class."""
    masks = torch.zeros(3, 8, 8, dtype=torch.bool)
    masks[0, 1:5, 1:5] = True
    masks[1, 2:6, 2:6] = True
    masks[2, 2:6, 2:6] = True
    boxes = torch.tensor([[1.0, 1.0, 5.0, 5.0], [2.0, 2.0, 6.0, 6.0], [2.0, 2.0, 6.0, 6.0]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    labels = torch.tensor([0, 0, 1])

    keep = _classwise_nms(boxes=boxes, scores=scores, labels=labels, threshold=0.3, metric="iou", masks=masks)

    assert keep.tolist() == [0, 2]


def test_classwise_mask_nms_can_use_mask_ios() -> None:
    """Mask IoS should suppress a smaller mask contained inside a larger mask."""
    masks = torch.zeros(2, 8, 8, dtype=torch.bool)
    masks[0, 1:7, 1:7] = True
    masks[1, 2:4, 2:4] = True
    boxes = torch.tensor([[1.0, 1.0, 7.0, 7.0], [2.0, 2.0, 4.0, 4.0]])
    scores = torch.tensor([0.9, 0.8])
    labels = torch.tensor([0, 0])

    keep = _classwise_nms(boxes=boxes, scores=scores, labels=labels, threshold=0.9, metric="ios", masks=masks)

    assert keep.tolist() == [0]


def test_classwise_box_nms_can_use_ios() -> None:
    """Box IoS NMS should suppress a smaller box contained inside a larger box."""
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [2.0, 2.0, 4.0, 4.0], [30.0, 30.0, 40.0, 40.0]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    labels = torch.tensor([0, 0, 0])

    keep = _classwise_nms(boxes=boxes, scores=scores, labels=labels, threshold=0.9, metric="ios")

    assert keep.tolist() == [0, 2]


def test_classwise_box_nms_can_use_diou_for_center_aware_suppression() -> None:
    """DIoU NMS should keep nearby objects that plain IoU NMS would suppress."""
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [2.0, 2.0, 12.0, 12.0],
            [0.2, 0.2, 10.2, 10.2],
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    labels = torch.tensor([0, 0, 0])

    keep_iou = _classwise_nms(boxes=boxes, scores=scores, labels=labels, threshold=0.45, metric="iou")
    keep_diou = _classwise_nms(boxes=boxes, scores=scores, labels=labels, threshold=0.45, metric="diou")

    assert keep_iou.tolist() == [0]
    assert keep_diou.tolist() == [0, 1]


def test_cluster_diou_nms_suppresses_against_every_earlier_box() -> None:
    """CDN should suppress a box when any earlier score-sorted box overlaps it."""
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.5, 0.5, 10.5, 10.5],
            [1.0, 1.0, 11.0, 11.0],
            [30.0, 30.0, 40.0, 40.0],
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])

    keep = _box_cluster_diou_nms(boxes=boxes, scores=scores, threshold=0.5)

    assert keep.tolist() == [0, 3]


def test_classwise_box_nms_can_use_cdn() -> None:
    """The public merge dispatcher should expose Cluster-DIoU-NMS."""
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.5, 0.5, 10.5, 10.5], [30.0, 30.0, 40.0, 40.0]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    labels = torch.tensor([0, 0, 0])

    keep = _classwise_nms(boxes=boxes, scores=scores, labels=labels, threshold=0.5, metric="cdn")

    assert keep.tolist() == [0, 2]


def test_source_aware_duplicate_suppression_removes_full_image_crop_duplicate() -> None:
    """Source-aware suppression should remove shifted full-image/crop duplicate boxes."""
    boxes = torch.tensor(
        [
            [1737.6, 937.4, 1765.2, 949.9],
            [1749.0, 927.0, 1767.5, 941.3],
            [1800.0, 1200.0, 1820.0, 1220.0],
        ]
    )
    scores = torch.tensor([0.78, 0.44, 0.7])
    labels = torch.tensor([3, 3, 3])
    source_windows = torch.tensor(
        [
            [1129.0, 614.0, 1883.0, 1434.0],
            [0.0, 0.0, 2448.0, 2048.0],
            [1129.0, 614.0, 1883.0, 1434.0],
        ]
    )

    keep = _source_aware_duplicate_suppression(
        boxes=boxes,
        scores=scores,
        labels=labels,
        source_windows=source_windows,
        center_margin_px=4.0,
        center_margin_ratio=0.15,
    )

    assert keep.tolist() == [0, 2]
