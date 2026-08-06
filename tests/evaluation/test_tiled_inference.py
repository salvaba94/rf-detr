# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for native tiled validation inference."""

import torch
from torch import nn

from rfdetr.evaluation.tiled_asahi import predict_asahi
from rfdetr.evaluation.tiled_core import (
    TileWindow,
    _prepare_window_tensor,
    _project_local_result_to_window,
)
from rfdetr.evaluation.tiled_gsahi import predict_gsahi
from rfdetr.evaluation.tiled_sahi import predict_tiled
from rfdetr.utilities.tensors import NestedTensor


class _RecordingTileModel(nn.Module):
    """Fake model that records tile batch sizes."""

    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []
        self.spatial_sizes: list[tuple[int, int]] = []
        self.valid_pixel_counts: list[int] = []
        self.dtypes: list[torch.dtype] = []
        self.channels_last: list[bool] = []

    def forward(self, samples: NestedTensor) -> dict[str, torch.Tensor]:
        """Record the batch size and return a tiny output marker."""
        batch_size = int(samples.tensors.shape[0])
        self.batch_sizes.append(batch_size)
        self.spatial_sizes.append(tuple(int(value) for value in samples.tensors.shape[-2:]))
        self.valid_pixel_counts.extend((~samples.mask).flatten(1).sum(dim=1).cpu().tolist())
        self.dtypes.append(samples.tensors.dtype)
        self.channels_last.append(samples.tensors.is_contiguous(memory_format=torch.channels_last))
        return {"batch_size": torch.tensor(batch_size, device=samples.tensors.device)}


class _FixedTilePostprocess:
    """Postprocess returning one centered tile prediction per tile."""

    def __call__(
        self,
        outputs: dict[str, torch.Tensor],
        target_sizes: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Return one local box for every tile."""
        results: list[dict[str, torch.Tensor]] = []
        for size in target_sizes:
            height, width = [float(value) for value in size.tolist()]
            results.append(
                {
                    "boxes": torch.tensor([[1.0, 1.0, min(3.0, width), min(3.0, height)]], device=size.device),
                    "scores": torch.tensor([0.9], device=size.device),
                    "labels": torch.tensor([0], dtype=torch.long, device=size.device),
                }
            )
        return results


class _GoisStagePostprocess:
    """Postprocess with distinct coarse and fine-stage predictions."""

    def __init__(self) -> None:
        self.call_index = 0

    def __call__(
        self,
        outputs: dict[str, torch.Tensor],
        target_sizes: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Return a coarse proposal first, then a shifted fine detection."""
        self.call_index += 1
        box = [2.0, 2.0, 4.0, 4.0] if self.call_index == 1 else [3.0, 3.0, 5.0, 5.0]
        return [
            {
                "boxes": torch.tensor([box], device=target_sizes.device),
                "scores": torch.tensor([0.9], device=target_sizes.device),
                "labels": torch.tensor([0], dtype=torch.long, device=target_sizes.device),
            }
            for _ in target_sizes
        ]


def test_predict_tiled_batches_tiles_and_shifts_boxes() -> None:
    """Native tiled prediction should batch tiles and shift local boxes into full-image coordinates."""
    model = _RecordingTileModel()
    samples = NestedTensor(torch.zeros(1, 3, 8, 8), torch.zeros(1, 8, 8, dtype=torch.bool))
    targets = [
        {
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty(0, dtype=torch.long),
            "orig_size": torch.tensor([8, 8]),
            "size": torch.tensor([8, 8]),
        }
    ]

    results = predict_tiled(
        model=model,
        postprocess=_FixedTilePostprocess(),
        samples=samples,
        targets=targets,
        slice_height=4,
        slice_width=4,
        overlap_height_ratio=0.0,
        overlap_width_ratio=0.0,
        include_full_image=False,
        nms_threshold=0.5,
        merge_metric="iou",
        score_threshold=0.1,
        max_predictions=10,
        tile_batch_size=2,
        block_size=4,
        segmentation=False,
    )

    assert model.batch_sizes == [2, 2]
    assert model.spatial_sizes == [(4, 4), (4, 4)]
    assert results[0]["boxes"].tolist() == [
        [1.0, 1.0, 3.0, 3.0],
        [5.0, 1.0, 7.0, 3.0],
        [1.0, 5.0, 3.0, 7.0],
        [5.0, 5.0, 7.0, 7.0],
    ]


def test_predict_asahi_resizes_adaptive_windows_to_model_resolution() -> None:
    """ASAHI adaptive crops should be resized to the model input size before prediction."""
    model = _RecordingTileModel()
    samples = NestedTensor(torch.zeros(1, 3, 8, 16), torch.zeros(1, 8, 16, dtype=torch.bool))
    targets = [
        {
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty(0, dtype=torch.long),
            "orig_size": torch.tensor([8, 16]),
            "size": torch.tensor([8, 16]),
        }
    ]

    results = predict_asahi(
        model=model,
        postprocess=_FixedTilePostprocess(),
        samples=samples,
        targets=targets,
        short_side_threshold=32,
        low_patch_count=6,
        high_patch_count=12,
        overlap_ratio=0.15,
        include_full_image=False,
        nms_threshold=1.0,
        merge_metric="iou",
        score_threshold=0.1,
        max_predictions=10,
        tile_batch_size=3,
        block_size=4,
        segmentation=False,
        full_image_size=12,
    )

    assert model.batch_sizes == [3, 3]
    assert all(max(size) == 12 for size in model.spatial_sizes)
    assert results[0]["_source_stages"].unique().tolist() == [4]


def test_predict_asahi_uses_explicit_adaptive_window_resize_size() -> None:
    """ASAHI adaptive crop size can differ from the resized full-image pass."""
    model = _RecordingTileModel()
    samples = NestedTensor(torch.zeros(1, 3, 8, 16), torch.zeros(1, 8, 16, dtype=torch.bool))
    targets = [
        {
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty(0, dtype=torch.long),
            "orig_size": torch.tensor([8, 16]),
            "size": torch.tensor([8, 16]),
        }
    ]

    results = predict_asahi(
        model=model,
        postprocess=_FixedTilePostprocess(),
        samples=samples,
        targets=targets,
        short_side_threshold=32,
        low_patch_count=6,
        high_patch_count=12,
        overlap_ratio=0.15,
        include_full_image=True,
        nms_threshold=1.0,
        merge_metric="iou",
        score_threshold=0.1,
        max_predictions=20,
        tile_batch_size=3,
        block_size=4,
        segmentation=False,
        full_image_size=12,
        window_resize_longest_side=16,
        source_mode="full_adaptive",
    )

    assert model.spatial_sizes[0] == (12, 12)
    assert all(size == (16, 16) for size in model.spatial_sizes[1:])
    assert set(results[0]["_source_stages"].unique().tolist()) == {1, 4}


def test_predict_asahi_batches_tiles_across_images_without_changing_predictions() -> None:
    """Global tile batching should scatter predictions back to the same per-image results."""
    samples = NestedTensor(torch.zeros(2, 3, 8, 16), torch.zeros(2, 8, 16, dtype=torch.bool))
    targets = [
        {
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty(0, dtype=torch.long),
            "orig_size": torch.tensor([8, 16]),
            "size": torch.tensor([8, 16]),
        },
        {
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty(0, dtype=torch.long),
            "orig_size": torch.tensor([8, 16]),
            "size": torch.tensor([8, 16]),
        },
    ]

    per_image_model = _RecordingTileModel()
    per_image_results = predict_asahi(
        model=per_image_model,
        postprocess=_FixedTilePostprocess(),
        samples=samples,
        targets=targets,
        short_side_threshold=32,
        low_patch_count=6,
        high_patch_count=12,
        overlap_ratio=0.15,
        include_full_image=False,
        nms_threshold=1.0,
        merge_metric="iou",
        score_threshold=0.1,
        max_predictions=20,
        tile_batch_size=4,
        block_size=4,
        segmentation=False,
        full_image_size=12,
        batch_across_images=False,
    )
    global_model = _RecordingTileModel()
    global_results = predict_asahi(
        model=global_model,
        postprocess=_FixedTilePostprocess(),
        samples=samples,
        targets=targets,
        short_side_threshold=32,
        low_patch_count=6,
        high_patch_count=12,
        overlap_ratio=0.15,
        include_full_image=False,
        nms_threshold=1.0,
        merge_metric="iou",
        score_threshold=0.1,
        max_predictions=20,
        tile_batch_size=4,
        block_size=4,
        segmentation=False,
        full_image_size=12,
        batch_across_images=True,
    )

    assert per_image_model.batch_sizes == [4, 2, 4, 2]
    assert global_model.batch_sizes == [4, 4, 4]
    for per_image_result, global_result in zip(per_image_results, global_results):
        assert torch.allclose(per_image_result["boxes"], global_result["boxes"])
        assert torch.allclose(per_image_result["scores"], global_result["scores"])
        assert torch.equal(per_image_result["labels"], global_result["labels"])
        assert torch.equal(per_image_result["_source_stages"], global_result["_source_stages"])


def test_predict_asahi_global_batching_falls_back_for_diagnostics() -> None:
    """Diagnostics need per-tile metadata, so cross-image batching should use the old path."""
    model = _RecordingTileModel()
    samples = NestedTensor(torch.zeros(2, 3, 8, 16), torch.zeros(2, 8, 16, dtype=torch.bool))
    targets = [
        {
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty(0, dtype=torch.long),
            "orig_size": torch.tensor([8, 16]),
            "size": torch.tensor([8, 16]),
        },
        {
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty(0, dtype=torch.long),
            "orig_size": torch.tensor([8, 16]),
            "size": torch.tensor([8, 16]),
        },
    ]

    results = predict_asahi(
        model=model,
        postprocess=_FixedTilePostprocess(),
        samples=samples,
        targets=targets,
        short_side_threshold=32,
        low_patch_count=6,
        high_patch_count=12,
        overlap_ratio=0.15,
        include_full_image=False,
        nms_threshold=1.0,
        merge_metric="iou",
        score_threshold=0.1,
        max_predictions=20,
        tile_batch_size=4,
        block_size=4,
        segmentation=False,
        full_image_size=12,
        batch_across_images=True,
        return_diagnostics=True,
    )

    assert model.batch_sizes == [4, 2, 4, 2]
    assert all("_local_boxes" in result for result in results)


def test_predict_asahi_global_batching_falls_back_for_full_image_fusion() -> None:
    """Full-image fusion should keep the existing per-image tiled path."""
    model = _RecordingTileModel()
    samples = NestedTensor(torch.zeros(2, 3, 8, 16), torch.zeros(2, 8, 16, dtype=torch.bool))
    targets = [
        {
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty(0, dtype=torch.long),
            "orig_size": torch.tensor([8, 16]),
            "size": torch.tensor([8, 16]),
        },
        {
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty(0, dtype=torch.long),
            "orig_size": torch.tensor([8, 16]),
            "size": torch.tensor([8, 16]),
        },
    ]

    predict_asahi(
        model=model,
        postprocess=_FixedTilePostprocess(),
        samples=samples,
        targets=targets,
        short_side_threshold=32,
        low_patch_count=6,
        high_patch_count=12,
        overlap_ratio=0.15,
        include_full_image=True,
        nms_threshold=1.0,
        merge_metric="iou",
        score_threshold=0.1,
        max_predictions=20,
        tile_batch_size=4,
        block_size=4,
        segmentation=False,
        full_image_size=12,
        source_mode="full_adaptive",
        batch_across_images=True,
    )

    assert model.batch_sizes == [1, 4, 2, 1, 4, 2]


def test_predict_tiled_global_batching_falls_back_for_segmentation() -> None:
    """Segmentation keeps the current per-image path because mask metadata is image-local."""
    model = _RecordingTileModel()
    samples = NestedTensor(torch.zeros(2, 3, 8, 8), torch.zeros(2, 8, 8, dtype=torch.bool))
    targets = [
        {
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty(0, dtype=torch.long),
            "orig_size": torch.tensor([8, 8]),
            "size": torch.tensor([8, 8]),
        },
        {
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty(0, dtype=torch.long),
            "orig_size": torch.tensor([8, 8]),
            "size": torch.tensor([8, 8]),
        },
    ]

    results = predict_tiled(
        model=model,
        postprocess=_FixedTilePostprocess(),
        samples=samples,
        targets=targets,
        slice_height=4,
        slice_width=4,
        overlap_height_ratio=0.0,
        overlap_width_ratio=0.0,
        include_full_image=False,
        nms_threshold=1.0,
        merge_metric="iou",
        score_threshold=0.1,
        max_predictions=20,
        tile_batch_size=3,
        block_size=4,
        segmentation=True,
        batch_across_images=True,
    )

    assert model.batch_sizes == [3, 1, 3, 1]
    assert all("masks" in result for result in results)


def test_predict_asahi_applies_tile_dtype_and_memory_format() -> None:
    """Benchmark flags should convert model input tiles without changing config defaults."""
    model = _RecordingTileModel()
    samples = NestedTensor(torch.zeros(1, 3, 8, 16), torch.zeros(1, 8, 16, dtype=torch.bool))
    targets = [
        {
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty(0, dtype=torch.long),
            "orig_size": torch.tensor([8, 16]),
            "size": torch.tensor([8, 16]),
        }
    ]

    predict_asahi(
        model=model,
        postprocess=_FixedTilePostprocess(),
        samples=samples,
        targets=targets,
        short_side_threshold=32,
        low_patch_count=6,
        high_patch_count=12,
        overlap_ratio=0.15,
        include_full_image=False,
        nms_threshold=1.0,
        merge_metric="iou",
        score_threshold=0.1,
        max_predictions=20,
        tile_batch_size=3,
        block_size=4,
        segmentation=False,
        full_image_size=12,
        tile_input_dtype="bf16",
        tile_memory_format="channels_last",
    )

    assert model.dtypes == [torch.bfloat16, torch.bfloat16]
    assert model.channels_last == [True, True]


def test_predict_asahi_source_modes_select_expected_prediction_sources() -> None:
    """ASAHI source modes should isolate adaptive, full, or fused prediction stages."""
    stage_sets: dict[str, set[int]] = {}
    batch_sizes: dict[str, list[int]] = {}
    for source_mode in ("adaptive", "full", "full_adaptive"):
        model = _RecordingTileModel()
        samples = NestedTensor(torch.zeros(1, 3, 8, 16), torch.zeros(1, 8, 16, dtype=torch.bool))
        targets = [
            {
                "boxes": torch.empty((0, 4)),
                "labels": torch.empty(0, dtype=torch.long),
                "orig_size": torch.tensor([8, 16]),
                "size": torch.tensor([8, 16]),
            }
        ]

        results = predict_asahi(
            model=model,
            postprocess=_FixedTilePostprocess(),
            samples=samples,
            targets=targets,
            short_side_threshold=32,
            low_patch_count=6,
            high_patch_count=12,
            overlap_ratio=0.15,
            include_full_image=True,
            nms_threshold=1.0,
            merge_metric="iou",
            score_threshold=0.1,
            max_predictions=20,
            tile_batch_size=3,
            block_size=4,
            segmentation=False,
            full_image_size=12,
            source_mode=source_mode,
        )
        stage_sets[source_mode] = set(results[0]["_source_stages"].unique().tolist())
        batch_sizes[source_mode] = model.batch_sizes

    assert stage_sets["adaptive"] == {4}
    assert stage_sets["full"] == {1}
    assert stage_sets["full_adaptive"] == {1, 4}
    assert batch_sizes["adaptive"] == [3, 3]
    assert batch_sizes["full"] == [1]
    assert batch_sizes["full_adaptive"] == [1, 3, 3]


def test_square_stretch_projection_maps_local_boxes_back_to_window() -> None:
    """Square-stretched ASAHI crops should project local boxes with independent x/y scales."""
    image = torch.zeros(3, 4, 8)
    _tensor, mask, model_size, valid_size, target_size, source_size, scale, padding = _prepare_window_tensor(
        image,
        resize_longest_side=16,
        resize_policy="square_stretch",
    )
    result = {
        "boxes": torch.tensor([[4.0, 4.0, 12.0, 12.0]]),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([0]),
        "_local_boxes": torch.tensor([[4.0, 4.0, 12.0, 12.0]]),
    }

    projected = _project_local_result_to_window(
        result,
        source_size=source_size,
        window=TileWindow(10, 20, 18, 24),
        full_size=(40, 40),
        segmentation=False,
    )

    assert mask is None
    assert model_size == (16, 16)
    assert valid_size == (16, 16)
    assert target_size == (16, 16)
    assert source_size == (16, 16)
    assert scale == (2.0, 4.0)
    assert padding == (0.0, 0.0, 0.0, 0.0)
    assert projected["boxes"].tolist() == [[12.0, 21.0, 16.0, 23.0]]
    assert projected["_local_boxes"].tolist() == [[4.0, 4.0, 12.0, 12.0]]


def test_square_letterbox_window_tensor_records_padding_mask() -> None:
    """Square-letterboxed ASAHI crops should expose valid pixels through the NestedTensor mask."""
    image = torch.ones(3, 4, 8)

    tensor, mask, model_size, valid_size, target_size, source_size, scale, padding = _prepare_window_tensor(
        image,
        resize_longest_side=16,
        resize_policy="square_letterbox",
    )

    assert tensor.shape == (3, 16, 16)
    assert mask is not None
    assert mask[:8, :16].eq(False).all()
    assert mask[8:, :].eq(True).all()
    assert model_size == (16, 16)
    assert valid_size == (8, 16)
    assert target_size == (8, 16)
    assert source_size == (8, 16)
    assert scale == (2.0, 2.0)
    assert padding == (0.0, 0.0, 0.0, 8.0)


def test_letterbox_canvas_target_projects_valid_content_from_square_canvas() -> None:
    """Letterbox canvas postprocess boxes should unproject through the valid content region."""
    image = torch.ones(3, 4, 8)
    tensor, mask, model_size, valid_size, target_size, source_size, scale, padding = _prepare_window_tensor(
        image,
        resize_longest_side=16,
        resize_policy="letterbox_canvas_target",
    )
    result = {
        "boxes": torch.tensor([[4.0, 2.0, 12.0, 6.0]]),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([0]),
        "_local_boxes": torch.tensor([[4.0, 2.0, 12.0, 6.0]]),
    }

    projected = _project_local_result_to_window(
        result,
        source_size=source_size,
        window=TileWindow(10, 20, 18, 24),
        full_size=(40, 40),
        segmentation=False,
    )

    assert tensor.shape == (3, 16, 16)
    assert mask is not None
    assert model_size == (16, 16)
    assert valid_size == (8, 16)
    assert target_size == (16, 16)
    assert source_size == (8, 16)
    assert scale == (2.0, 2.0)
    assert padding == (0.0, 0.0, 0.0, 8.0)
    assert projected["boxes"].tolist() == [[12.0, 21.0, 16.0, 23.0]]


def test_square_context_uses_expanded_square_inference_windows() -> None:
    """Square-context ASAHI should run a square source crop while preserving the requested source window."""
    model = _RecordingTileModel()
    samples = NestedTensor(torch.zeros(1, 3, 8, 16), torch.zeros(1, 8, 16, dtype=torch.bool))
    targets = [
        {
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty(0, dtype=torch.long),
            "orig_size": torch.tensor([8, 16]),
            "size": torch.tensor([8, 16]),
        }
    ]

    results = predict_asahi(
        model=model,
        postprocess=_FixedTilePostprocess(),
        samples=samples,
        targets=targets,
        short_side_threshold=32,
        low_patch_count=6,
        high_patch_count=12,
        overlap_ratio=0.15,
        include_full_image=False,
        nms_threshold=1.0,
        merge_metric="iou",
        score_threshold=0.1,
        max_predictions=20,
        tile_batch_size=3,
        block_size=4,
        segmentation=False,
        full_image_size=12,
        window_resize_policy="square_context",
        return_diagnostics=True,
    )

    source_windows = results[0]["_source_windows"]
    inference_windows = results[0]["_inference_windows"]
    source_sizes = source_windows[:, 2:4] - source_windows[:, 0:2]
    inference_sizes = inference_windows[:, 2:4] - inference_windows[:, 0:2]
    assert torch.any(source_sizes[:, 0] != source_sizes[:, 1])
    assert torch.allclose(inference_sizes[:, 0], inference_sizes[:, 1])
    assert results[0]["_valid_sizes"].unique(dim=0).tolist() == [[12.0, 12.0]]
    assert results[0]["_target_sizes"].unique(dim=0).tolist() == [[12.0, 12.0]]


def test_predict_tiled_can_include_resized_full_image_context() -> None:
    """SAHI prediction should optionally run one resized full-image context pass before tiled windows."""
    model = _RecordingTileModel()
    samples = NestedTensor(torch.zeros(1, 3, 8, 4), torch.zeros(1, 8, 4, dtype=torch.bool))
    targets = [
        {
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty(0, dtype=torch.long),
            "orig_size": torch.tensor([8, 4]),
            "size": torch.tensor([8, 4]),
        }
    ]

    predict_tiled(
        model=model,
        postprocess=_FixedTilePostprocess(),
        samples=samples,
        targets=targets,
        slice_height=4,
        slice_width=4,
        overlap_height_ratio=0.0,
        overlap_width_ratio=0.0,
        include_full_image=True,
        nms_threshold=0.5,
        merge_metric="iou",
        score_threshold=0.1,
        max_predictions=10,
        tile_batch_size=2,
        block_size=4,
        segmentation=False,
    )

    assert model.batch_sizes == [1, 2]
    assert model.spatial_sizes[0] == (4, 4)
    assert model.valid_pixel_counts[0] == 8


def test_predict_tiled_scales_boxes_to_original_size() -> None:
    """Predictions should be scaled when validation tensors differ from original image size."""
    model = _RecordingTileModel()
    samples = NestedTensor(torch.zeros(1, 3, 4, 4), torch.zeros(1, 4, 4, dtype=torch.bool))
    targets = [
        {
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty(0, dtype=torch.long),
            "orig_size": torch.tensor([8, 12]),
            "size": torch.tensor([4, 4]),
        }
    ]

    results = predict_tiled(
        model=model,
        postprocess=_FixedTilePostprocess(),
        samples=samples,
        targets=targets,
        slice_height=4,
        slice_width=4,
        overlap_height_ratio=0.0,
        overlap_width_ratio=0.0,
        include_full_image=False,
        nms_threshold=0.5,
        merge_metric="iou",
        score_threshold=0.1,
        max_predictions=10,
        tile_batch_size=1,
        block_size=4,
        segmentation=False,
    )

    assert results[0]["boxes"].tolist() == [[3.0, 2.0, 9.0, 6.0]]
    assert results[0]["_source_windows"].tolist() == [[0.0, 0.0, 12.0, 8.0]]


def test_predict_gsahi_uses_coarse_predictions_as_proposals_only_when_fine_runs() -> None:
    """GSAHI should not return duplicate coarse detections when a fine pass is produced."""
    model = _RecordingTileModel()
    samples = NestedTensor(torch.zeros(1, 3, 8, 8), torch.zeros(1, 8, 8, dtype=torch.bool))
    targets = [
        {
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty(0, dtype=torch.long),
            "orig_size": torch.tensor([8, 8]),
            "size": torch.tensor([8, 8]),
        }
    ]

    results = predict_gsahi(
        model=model,
        postprocess=_GoisStagePostprocess(),
        samples=samples,
        targets=targets,
        coarse_slice_size=8,
        fine_slice_size=8,
        coarse_overlap=0.0,
        fine_overlap=0.0,
        include_full_image=False,
        roi_score_threshold=0.1,
        roi_expansion_ratio=2.0,
        roi_max_regions=1,
        nms_threshold=1.0,
        merge_metric="iou",
        score_threshold=0.1,
        max_predictions=10,
        tile_batch_size=1,
        block_size=4,
        segmentation=False,
    )

    assert results[0]["boxes"].tolist() == [[3.0, 3.0, 5.0, 5.0]]
    assert results[0]["_source_stages"].tolist() == [3]


def test_predict_gsahi_resizes_fine_windows_to_model_resolution() -> None:
    """GSAHI fine-stage crops should be resized/projected back from model resolution."""
    model = _RecordingTileModel()
    samples = NestedTensor(torch.zeros(1, 3, 8, 8), torch.zeros(1, 8, 8, dtype=torch.bool))
    targets = [
        {
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty(0, dtype=torch.long),
            "orig_size": torch.tensor([8, 8]),
            "size": torch.tensor([8, 8]),
        }
    ]

    results = predict_gsahi(
        model=model,
        postprocess=_GoisStagePostprocess(),
        samples=samples,
        targets=targets,
        coarse_slice_size=8,
        fine_slice_size=4,
        coarse_overlap=0.0,
        fine_overlap=0.0,
        include_full_image=False,
        roi_score_threshold=0.1,
        roi_expansion_ratio=2.0,
        roi_max_regions=1,
        nms_threshold=1.0,
        merge_metric="iou",
        score_threshold=0.1,
        max_predictions=10,
        tile_batch_size=1,
        block_size=4,
        segmentation=False,
        full_image_size=8,
    )

    assert model.spatial_sizes[0] == (8, 8)
    assert all(size == (8, 8) for size in model.spatial_sizes[1:])
    assert results[0]["_source_stages"].unique().tolist() == [3]


def test_predict_gsahi_does_not_fallback_to_coarse_predictions_without_fine_rois() -> None:
    """GSAHI coarse detections should remain proposals, not final detections."""
    model = _RecordingTileModel()
    samples = NestedTensor(torch.zeros(1, 3, 8, 8), torch.zeros(1, 8, 8, dtype=torch.bool))
    targets = [
        {
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty(0, dtype=torch.long),
            "orig_size": torch.tensor([8, 8]),
            "size": torch.tensor([8, 8]),
        }
    ]

    results = predict_gsahi(
        model=model,
        postprocess=_GoisStagePostprocess(),
        samples=samples,
        targets=targets,
        coarse_slice_size=8,
        fine_slice_size=8,
        coarse_overlap=0.0,
        fine_overlap=0.0,
        include_full_image=False,
        roi_score_threshold=1.1,
        roi_expansion_ratio=2.0,
        roi_max_regions=1,
        nms_threshold=1.0,
        merge_metric="iou",
        score_threshold=0.1,
        max_predictions=10,
        tile_batch_size=1,
        block_size=4,
        segmentation=False,
    )

    assert results[0]["scores"].numel() == 0


def test_predict_gsahi_fine_merge_skips_unused_full_image_pass() -> None:
    """GSAHI fine-only merge should not compute discarded full-image predictions."""
    model = _RecordingTileModel()
    samples = NestedTensor(torch.zeros(1, 3, 8, 4), torch.zeros(1, 8, 4, dtype=torch.bool))
    targets = [
        {
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty(0, dtype=torch.long),
            "orig_size": torch.tensor([8, 4]),
            "size": torch.tensor([8, 4]),
        }
    ]

    predict_gsahi(
        model=model,
        postprocess=_GoisStagePostprocess(),
        samples=samples,
        targets=targets,
        coarse_slice_size=4,
        fine_slice_size=4,
        coarse_overlap=0.0,
        fine_overlap=0.0,
        include_full_image=True,
        roi_score_threshold=0.1,
        roi_expansion_ratio=2.0,
        roi_max_regions=1,
        nms_threshold=1.0,
        merge_metric="iou",
        score_threshold=0.1,
        max_predictions=10,
        tile_batch_size=2,
        block_size=4,
        segmentation=False,
        full_image_size=12,
        merge_sources="fine",
    )

    assert model.batch_sizes[0] == 2
