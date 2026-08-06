# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for RF-DETR custom augmentation implementations."""

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms.v2 import Compose

from rfdetr.datasets.aug_configs import AUG_SAHI
from rfdetr.datasets.transforms import (
    AlbumentationsWrapper,
    CopyPaste,
    MixUp,
    TiledCroppingWithMasks,
    prepare_multi_image_augmentations,
)


class TestCopyPaste:
    """Tests for the RF-DETR-native CopyPaste augmentation."""

    def test_detection_only_adds_pasted_box_and_label(self):
        """BBox-only CopyPaste should paste a rectangular object and append its annotation."""
        transform = CopyPaste(p=1.0, max_paste_objects=1, max_iou=1.0, seed=0)
        image = Image.new("RGB", (64, 64), color="black")
        target = {
            "boxes": torch.tensor([[8.0, 8.0, 18.0, 20.0]], dtype=torch.float32),
            "labels": torch.tensor([2], dtype=torch.long),
            "area": torch.tensor([120.0], dtype=torch.float32),
            "iscrowd": torch.tensor([0], dtype=torch.long),
        }

        _, augmented = transform(image, target)

        assert augmented["boxes"].shape == (2, 4)
        assert augmented["labels"].tolist() == [2, 2]
        assert augmented["area"].shape == (2,)
        assert augmented["iscrowd"].tolist() == [0, 0]

    def test_mask_targets_append_pasted_mask_and_tight_box(self):
        """Mask-aware CopyPaste should append a pasted mask aligned with the new box."""
        transform = CopyPaste(p=1.0, max_paste_objects=1, max_iou=1.0, seed=1)
        image_np = np.zeros((64, 64, 3), dtype=np.uint8)
        image_np[10:20, 12:22] = [255, 0, 0]
        image = Image.fromarray(image_np)
        masks = torch.zeros((1, 64, 64), dtype=torch.bool)
        masks[0, 10:20, 12:22] = True
        target = {
            "boxes": torch.tensor([[12.0, 10.0, 22.0, 20.0]], dtype=torch.float32),
            "labels": torch.tensor([4], dtype=torch.long),
            "masks": masks,
            "area": torch.tensor([100.0], dtype=torch.float32),
        }

        _, augmented = transform(image, target)

        assert augmented["boxes"].shape == (2, 4)
        assert augmented["masks"].shape == (2, 64, 64)
        pasted_mask = augmented["masks"][1]
        ys, xs = torch.nonzero(pasted_mask, as_tuple=True)
        pasted_box = torch.tensor(
            [xs.min().item(), ys.min().item(), xs.max().item() + 1, ys.max().item() + 1],
            dtype=torch.float32,
        )
        torch.testing.assert_close(augmented["boxes"][1], pasted_box)

    def test_keypoints_are_appended_and_translated(self):
        """CopyPaste should preserve keypoint targets by duplicating translated keypoints."""
        transform = CopyPaste(p=1.0, max_paste_objects=1, max_iou=1.0, seed=2)
        image = Image.new("RGB", (64, 64), color="black")
        target = {
            "boxes": torch.tensor([[8.0, 8.0, 18.0, 20.0]], dtype=torch.float32),
            "labels": torch.tensor([2], dtype=torch.long),
            "keypoints": torch.tensor([[[10.0, 11.0, 2.0], [0.0, 0.0, 0.0]]], dtype=torch.float32),
        }

        _, augmented = transform(image, target)

        assert augmented["boxes"].shape == (2, 4)
        assert augmented["keypoints"].shape == (2, 2, 3)
        dx = augmented["boxes"][1, 0] - target["boxes"][0, 0]
        dy = augmented["boxes"][1, 1] - target["boxes"][0, 1]
        expected_visible = target["keypoints"][0, 0].clone()
        expected_visible[0] += dx
        expected_visible[1] += dy
        torch.testing.assert_close(augmented["keypoints"][1, 0], expected_visible)
        torch.testing.assert_close(augmented["keypoints"][1, 1], target["keypoints"][0, 1])

    def test_uses_additional_sample_provider_when_available(self):
        """CopyPaste should paste objects from a provided second sample."""
        transform = CopyPaste(p=1.0, max_paste_objects=1, max_iou=1.0, seed=4)
        base_image = Image.new("RGB", (64, 64), color="black")
        base_target = {
            "boxes": torch.tensor([[2.0, 2.0, 8.0, 8.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.long),
            "area": torch.tensor([36.0], dtype=torch.float32),
            "iscrowd": torch.tensor([0], dtype=torch.long),
        }
        source_image = Image.new("RGB", (64, 64), color="black")
        source_target = {
            "boxes": torch.tensor([[20.0, 20.0, 30.0, 32.0]], dtype=torch.float32),
            "labels": torch.tensor([9], dtype=torch.long),
            "area": torch.tensor([120.0], dtype=torch.float32),
            "iscrowd": torch.tensor([1], dtype=torch.long),
        }
        transform.set_additional_sample_provider(lambda: (source_image, source_target))

        _, augmented = transform(base_image, base_target)

        assert augmented["labels"].tolist() == [1, 9]
        assert augmented["iscrowd"].tolist() == [0, 1]

    def test_prepare_multi_image_augmentations_attaches_provider(self):
        """Dataset hook should attach providers to native multi-image transforms inside Compose."""
        transform = CopyPaste(p=1.0, max_paste_objects=1, max_iou=1.0, seed=5)
        composed = Compose([transform])

        class DatasetWithAdditionalSample:
            def _get_additional_sample(self, index):
                return Image.new("RGB", (8, 8)), {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.long),
                }

        prepare_multi_image_augmentations(composed, DatasetWithAdditionalSample(), 0)

        assert transform._additional_sample_provider is not None

    def test_prepare_multi_image_augmentations_tracks_used_indices(self):
        """Provider should exclude the base index and donors already used for the current sample."""
        transform = CopyPaste(p=1.0, max_paste_objects=1, max_iou=1.0, seed=5)
        composed = Compose([transform])

        class DatasetWithIndexedAdditionalSamples:
            def __init__(self):
                self.exclude_history = []

            def _get_additional_sample(self, index):
                raise AssertionError("_get_additional_sample_info should be preferred when available")

            def _get_additional_sample_info(self, index, exclude_indices=None):
                excluded = set(exclude_indices or set())
                self.exclude_history.append(excluded)
                for candidate in (1, 2, 3):
                    if candidate not in excluded:
                        return candidate, (
                            Image.new("RGB", (8, 8)),
                            {
                                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                                "labels": torch.zeros((0,), dtype=torch.long),
                            },
                        )
                return None

        dataset = DatasetWithIndexedAdditionalSamples()
        prepare_multi_image_augmentations(composed, dataset, 0)

        assert transform._additional_sample_provider is not None
        assert transform._additional_sample_provider() is not None
        assert transform._additional_sample_provider() is not None
        assert dataset.exclude_history == [{0}, {0, 1}]

    def test_from_config_instantiates_copy_paste(self):
        """CopyPaste should be available through RF-DETR aug_config."""
        transforms = AlbumentationsWrapper.from_config({"CopyPaste": {"p": 1.0, "seed": 0}})

        assert len(transforms) == 1
        assert isinstance(transforms[0], CopyPaste)


class TestMixUp:
    """Tests for the RF-DETR-native MixUp augmentation."""

    def test_uses_additional_sample_provider_and_merges_targets(self):
        """MixUp should blend images and append annotations from the second sample."""
        transform = MixUp(p=1.0, alpha=0.0, seed=0)
        base_image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        base_target = {
            "boxes": torch.tensor([[2.0, 2.0, 8.0, 8.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.long),
            "area": torch.tensor([36.0], dtype=torch.float32),
            "iscrowd": torch.tensor([0], dtype=torch.long),
        }
        mix_image = Image.fromarray(np.full((32, 32, 3), 200, dtype=np.uint8))
        mix_target = {
            "boxes": torch.tensor([[10.0, 10.0, 20.0, 22.0]], dtype=torch.float32),
            "labels": torch.tensor([9], dtype=torch.long),
            "area": torch.tensor([120.0], dtype=torch.float32),
            "iscrowd": torch.tensor([1], dtype=torch.long),
        }
        transform.set_additional_sample_provider(lambda: (mix_image, mix_target))

        mixed_image, mixed_target = transform(base_image, base_target)

        assert np.asarray(mixed_image).mean() == pytest.approx(100.0)
        assert mixed_target["boxes"].shape == (2, 4)
        assert mixed_target["labels"].tolist() == [1, 9]
        assert mixed_target["iscrowd"].tolist() == [0, 1]
        assert mixed_target["area"].tolist() == pytest.approx([36.0, 120.0])

    def test_resizes_second_sample_boxes_masks_and_keypoints(self):
        """MixUp should scale second-sample annotations when image sizes differ."""
        transform = MixUp(p=1.0, alpha=0.0, seed=0)
        base_image = Image.fromarray(np.zeros((40, 80, 3), dtype=np.uint8))
        base_masks = torch.zeros((1, 40, 80), dtype=torch.bool)
        base_masks[0, 2:8, 2:8] = True
        base_target = {
            "boxes": torch.tensor([[2.0, 2.0, 8.0, 8.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.long),
            "masks": base_masks,
            "keypoints": torch.tensor([[[4.0, 5.0, 2.0]]], dtype=torch.float32),
        }
        mix_image = Image.fromarray(np.full((20, 40, 3), 128, dtype=np.uint8))
        mix_masks = torch.zeros((1, 20, 40), dtype=torch.bool)
        mix_masks[0, 5:10, 10:20] = True
        mix_target = {
            "boxes": torch.tensor([[10.0, 5.0, 20.0, 10.0]], dtype=torch.float32),
            "labels": torch.tensor([2], dtype=torch.long),
            "masks": mix_masks,
            "keypoints": torch.tensor([[[15.0, 7.0, 2.0]]], dtype=torch.float32),
        }
        transform.set_additional_sample_provider(lambda: (mix_image, mix_target))

        _, mixed_target = transform(base_image, base_target)

        torch.testing.assert_close(mixed_target["boxes"][1], torch.tensor([20.0, 10.0, 40.0, 20.0]))
        torch.testing.assert_close(mixed_target["keypoints"][1, 0], torch.tensor([30.0, 14.0, 2.0]))
        assert mixed_target["masks"].shape == (2, 40, 80)
        assert mixed_target["masks"][1].sum().item() == 200

    def test_from_config_instantiates_mixup(self):
        """MixUp should be available through RF-DETR aug_config."""
        transforms = AlbumentationsWrapper.from_config({"MixUp": {"p": 1.0, "alpha": 0.0, "seed": 0}})

        assert len(transforms) == 1
        assert isinstance(transforms[0], MixUp)


class TestTiledCroppingWithMasks:
    """Tests for the tiled mask-aware crop augmentation."""

    def test_returns_requested_crop_size_when_foreground_exists(self):
        """TiledCroppingWithMasks should crop to the configured fixed size when possible."""
        wrapper = AlbumentationsWrapper(
            TiledCroppingWithMasks(height=50, width=50, p=1.0, deterministic=True, edge_bias_prob=0.0)
        )
        image = Image.new("RGB", (100, 100))
        target = {
            "boxes": torch.tensor([[60.0, 60.0, 90.0, 90.0]], dtype=torch.float32),
            "labels": torch.tensor([1]),
        }

        aug_image, aug_target = wrapper(image, target)

        assert aug_image.size == (50, 50)
        assert aug_target["boxes"].shape == (1, 4)
        assert aug_target["labels"].tolist() == [1]

    def test_detection_only_uses_box_foreground(self):
        """Detection-only targets should use rectangular masks generated from boxes."""
        wrapper = AlbumentationsWrapper(TiledCroppingWithMasks(height=40, width=40, p=1.0, edge_bias_prob=0.0))
        image = Image.new("RGB", (100, 100))
        target = {
            "boxes": torch.tensor([[70.0, 70.0, 90.0, 90.0]], dtype=torch.float32),
            "labels": torch.tensor([3]),
        }

        aug_image, aug_target = wrapper(image, target)

        assert aug_image.size == (40, 40)
        assert aug_target["boxes"].shape[0] >= 1
        assert aug_target["labels"].tolist() == [3]

    def test_filters_boxes_by_min_area(self):
        """Boxes below TiledCroppingWithMasks min_area should be removed after cropping."""
        wrapper = AlbumentationsWrapper(TiledCroppingWithMasks(height=50, width=50, min_area=10_000.0, p=1.0))
        image = Image.new("RGB", (100, 100))
        target = {
            "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]], dtype=torch.float32),
            "labels": torch.tensor([1]),
            "area": torch.tensor([100.0]),
        }

        _, aug_target = wrapper(image, target)

        assert aug_target["boxes"].shape == (0, 4)
        assert aug_target["labels"].shape == (0,)
        assert aug_target["area"].shape == (0,)

    def test_tightens_boxes_from_matching_masks(self):
        """When masks exist, final boxes should be recomputed from cropped masks."""
        wrapper = AlbumentationsWrapper(TiledCroppingWithMasks(height=80, width=80, p=1.0, edge_bias_prob=0.0))
        image = Image.new("RGB", (100, 100))
        masks = torch.zeros((1, 100, 100), dtype=torch.bool)
        masks[0, 20:40, 25:35] = True
        target = {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]], dtype=torch.float32),
            "labels": torch.tensor([1]),
            "masks": masks,
            "area": torch.tensor([1600.0]),
        }

        _, aug_target = wrapper(image, target)

        aug_mask = aug_target["masks"][0]
        ys, xs = torch.nonzero(aug_mask, as_tuple=True)
        expected_box = torch.tensor(
            [xs.min().item(), ys.min().item(), xs.max().item() + 1, ys.max().item() + 1],
            dtype=torch.float32,
        )
        torch.testing.assert_close(aug_target["boxes"][0], expected_box)
        expected_area = float((expected_box[2] - expected_box[0]) * (expected_box[3] - expected_box[1]))
        assert aug_target["area"].item() == pytest.approx(expected_area)

    def test_empty_masks_drop_matching_boxes_and_keep_fields_aligned(self):
        """A surviving cropped box with an empty matching mask should be dropped."""
        wrapper = AlbumentationsWrapper(TiledCroppingWithMasks(height=80, width=80, p=1.0, edge_bias_prob=0.0))
        image = Image.new("RGB", (100, 100))
        masks = torch.zeros((2, 100, 100), dtype=torch.bool)
        masks[0, 10:30, 10:30] = True
        target = {
            "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0], [12.0, 12.0, 28.0, 28.0]], dtype=torch.float32),
            "labels": torch.tensor([1, 2]),
            "masks": masks,
            "area": torch.tensor([400.0, 256.0]),
            "iscrowd": torch.tensor([0, 1]),
            "keypoints": torch.tensor([[[15.0, 15.0, 2.0]], [[20.0, 20.0, 2.0]]]),
        }

        _, aug_target = wrapper(image, target)

        assert aug_target["boxes"].shape == (1, 4)
        assert aug_target["labels"].tolist() == [1]
        assert aug_target["masks"].shape[0] == 1
        assert aug_target["iscrowd"].tolist() == [0]
        assert aug_target["keypoints"].shape == (1, 1, 3)

    def test_keypoints_are_translated_with_crop_and_invisible_points_preserved(self):
        """TiledCroppingWithMasks should use the standard Albumentations keypoint path."""
        wrapper = AlbumentationsWrapper(TiledCroppingWithMasks(height=50, width=50, p=1.0, edge_bias_prob=0.0))
        image = Image.new("RGB", (100, 100))
        target = {
            "boxes": torch.tensor([[60.0, 60.0, 90.0, 90.0]], dtype=torch.float32),
            "labels": torch.tensor([1]),
            "keypoints": torch.tensor([[[70.0, 75.0, 2.0], [0.0, 0.0, 0.0]]], dtype=torch.float32),
        }

        _, aug_target = wrapper(image, target)

        assert aug_target["boxes"].shape == (1, 4)
        assert aug_target["keypoints"].shape == (1, 2, 3)
        dx = aug_target["boxes"][0, 0] - target["boxes"][0, 0]
        dy = aug_target["boxes"][0, 1] - target["boxes"][0, 1]
        expected_visible = target["keypoints"][0, 0].clone()
        expected_visible[0] += dx
        expected_visible[1] += dy
        torch.testing.assert_close(aug_target["keypoints"][0, 0], expected_visible)
        torch.testing.assert_close(aug_target["keypoints"][0, 1], target["keypoints"][0, 1])

    def test_no_boxes_use_random_crop_fallback(self):
        """No-box inputs should still produce a fixed-size random tile."""
        wrapper = AlbumentationsWrapper(TiledCroppingWithMasks(height=40, width=40, p=1.0))
        image = Image.new("RGB", (100, 100))
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.long),
        }

        aug_image, aug_target = wrapper(image, target)

        assert aug_image.size == (40, 40)
        assert aug_target["boxes"].shape == (0, 4)
        assert aug_target["labels"].shape == (0,)

    def test_deterministic_crop_is_stable_and_keeps_foreground(self):
        """Deterministic tiled crops should be repeatable and retain object boxes."""
        wrapper = AlbumentationsWrapper(
            TiledCroppingWithMasks(height=50, width=50, p=1.0, deterministic=True, edge_bias_prob=0.0)
        )
        image = Image.new("RGB", (100, 100))
        target = {
            "boxes": torch.tensor([[60.0, 60.0, 90.0, 90.0]], dtype=torch.float32),
            "labels": torch.tensor([1]),
        }

        first_image, first_target = wrapper(image, target)
        second_image, second_target = wrapper(image, target)

        assert first_image.size == (50, 50)
        assert second_image.size == (50, 50)
        torch.testing.assert_close(first_target["boxes"], second_target["boxes"])
        assert first_target["labels"].tolist() == [1]

    def test_from_config_instantiates_tiled_cropping_with_masks(self):
        """TiledCroppingWithMasks should be available through RF-DETR aug_config."""
        transforms = AlbumentationsWrapper.from_config(
            {"TiledCroppingWithMasks": {"height": 32, "width": 32, "p": 1.0}}
        )

        assert len(transforms) == 1
        assert transforms[0].transform.transforms[0].__class__.__name__ == "TiledCroppingWithMasks"

    def test_tiled_cropping_alias_accepts_target_size_and_ranges(self):
        """DA-YOLO-style TiledCroppingWithMasks config should instantiate the canonical crop."""
        transforms = AlbumentationsWrapper.from_config(
            {
                "TiledCroppingWithMasks": {
                    "target_size": 32,
                    "height_range": (24, 32),
                    "width_range": (24, 32),
                    "p": 1.0,
                }
            }
        )

        tiled = transforms[0].transform.transforms[0]
        assert isinstance(tiled, TiledCroppingWithMasks)
        assert tiled.__class__.__name__ == "TiledCroppingWithMasks"
        assert tiled.base_height == 32
        assert tiled.height_range == (24, 32)

    def test_kornia_gpu_backend_reports_tiled_cropping_with_masks_as_unsupported(self):
        """The GPU augmentation backend should reject TiledCroppingWithMasks by name."""
        pytest.importorskip("kornia")
        from rfdetr.datasets.kornia_transforms import build_kornia_pipeline

        with pytest.raises(ValueError, match="TiledCroppingWithMasks"):
            build_kornia_pipeline(AUG_SAHI, 640)
