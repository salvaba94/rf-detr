# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for SetCriterion edge paths and diagnostics."""

import pytest
import torch

from rfdetr.models.criterion import SetCriterion


class _MatcherStub:
    """Minimal matcher that returns identity indices for every target in the batch."""

    def __call__(self, outputs, targets, group_detr=1):
        return [(torch.arange(len(t["labels"])), torch.arange(len(t["labels"]))) for t in targets]


def _bare_criterion() -> SetCriterion:
    """Return a SetCriterion with no losses so forward() is a no-op."""
    criterion = SetCriterion.__new__(SetCriterion)
    criterion.training = True
    criterion.group_detr = 1
    criterion.sum_group_losses = False
    criterion.losses = []
    criterion.weight_dict = {}
    criterion.matcher = _MatcherStub()
    criterion.num_keypoints_per_class = []
    return criterion


def _cardinality_criterion() -> SetCriterion:
    """Return a minimal initialized criterion for diagnostic loss tests."""
    return SetCriterion(
        num_classes=3,
        matcher=None,
        weight_dict={},
        focal_alpha=0.25,
        losses=["cardinality"],
    )


class TestOutputDevice:
    """Tests for SetCriterion._output_device — probes top-level tensor values only."""

    def test_returns_device_of_first_tensor(self):
        """Device inferred from the first tensor value in outputs."""
        outputs = {"pred_logits": torch.zeros(1, 1, 1)}

        device = SetCriterion._output_device(outputs)

        assert device == torch.device("cpu")

    def test_raises_when_no_tensor_present(self):
        """ValueError raised when no top-level value is a tensor."""
        outputs = {"meta": "string_value", "count": 42}

        with pytest.raises(ValueError, match="at least one tensor"):
            SetCriterion._output_device(outputs)

    def test_skips_non_tensor_values(self):
        """Non-tensor entries at the top level are skipped; first tensor wins."""
        outputs = {"meta": "ignored", "pred_logits": torch.zeros(1, 1, 1)}

        device = SetCriterion._output_device(outputs)

        assert device == torch.device("cpu")


class TestNumBoxesForTargets:
    """Tests for SetCriterion.num_boxes_for_targets — clamp and empty-target edge cases."""

    def test_returns_tensor_gte_one(self):
        """Result must be clamped to >= 1.0 to prevent division by zero."""
        criterion = _bare_criterion()
        outputs = {"pred_logits": torch.zeros(1, 1, 1)}
        targets = [{"labels": torch.tensor([0, 1])}]

        result = criterion.num_boxes_for_targets(outputs, targets)

        assert result.item() >= 1.0

    def test_clamps_zero_box_count_to_one(self):
        """Empty targets (no labels) must clamp to 1.0 to avoid zero denominator."""
        criterion = _bare_criterion()
        outputs = {"pred_logits": torch.zeros(1, 1, 1)}
        targets = [{"labels": torch.zeros(0, dtype=torch.int64)}]

        result = criterion.num_boxes_for_targets(outputs, targets)

        assert result.item() == pytest.approx(1.0)

    def test_clamps_empty_target_list(self):
        """Empty target list (batch_size=0 edge case) must also clamp to 1.0."""
        criterion = _bare_criterion()
        outputs = {"pred_logits": torch.zeros(1, 1, 1)}
        targets = []

        result = criterion.num_boxes_for_targets(outputs, targets)

        assert result.item() == pytest.approx(1.0)

    def test_counts_labels_correctly(self):
        """Box count equals total number of labels across all targets in the batch."""
        criterion = _bare_criterion()
        outputs = {"pred_logits": torch.zeros(1, 1, 1)}
        targets = [
            {"labels": torch.tensor([0, 1])},
            {"labels": torch.tensor([0])},
        ]

        result = criterion.num_boxes_for_targets(outputs, targets)

        # 2 + 1 = 3 boxes; single-process so no all-reduce
        assert result.item() == pytest.approx(3.0)

class TestCardinalityDiagnostic:
    """Tests for sigmoid-head cardinality diagnostics."""

    def test_counts_sigmoid_confident_queries(self) -> None:
        """Cardinality diagnostic counts queries above sigmoid threshold, not argmax classes."""
        criterion = _cardinality_criterion()
        logits = torch.full((1, 4, 3), -10.0, requires_grad=True)
        logits.data[0, 0, 1] = 10.0
        logits.data[0, 2, 2] = 10.0
        outputs = {"pred_logits": logits}
        targets = [{"labels": torch.tensor([0, 2])}]

        losses = criterion.loss_cardinality(outputs, targets, indices=None, num_boxes=None)

        assert losses["cardinality_error"].item() == 0.0
        assert not losses["cardinality_error"].requires_grad

    def test_low_logits_predict_zero_objects(self) -> None:
        """All-low sigmoid logits should produce zero predicted objects."""
        criterion = _cardinality_criterion()
        outputs = {"pred_logits": torch.full((1, 4, 3), -10.0)}
        targets = [{"labels": torch.tensor([0, 2])}]

        losses = criterion.loss_cardinality(outputs, targets, indices=None, num_boxes=None)

        assert losses["cardinality_error"].item() == 2.0
