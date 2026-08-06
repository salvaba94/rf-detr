# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the first-batch tiled validation debug script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


def load_debug_module() -> ModuleType:
    """Load the debug script as a module for pure-function tests."""

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "debug_tiled_validation_first_batch.py"
    spec = importlib.util.spec_from_file_location("debug_tiled_validation_first_batch", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compare_predictions_counts_duplicates_and_unmatched_predictions() -> None:
    """Duplicate accounting flags extra same-class predictions around one GT."""

    module = load_debug_module()
    predictions = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0], [11.0, 11.0, 21.0, 21.0], [40.0, 40.0, 50.0, 50.0]]),
            "scores": torch.tensor([0.9, 0.8, 0.7]),
            "labels": torch.tensor([1, 1, 1]),
        }
    ]
    targets = [
        {
            "boxes": torch.tensor([[15.0, 15.0, 10.0, 10.0]]) / 100.0,
            "labels": torch.tensor([1]),
            "orig_size": torch.tensor([100, 100]),
        }
    ]

    comparison = module.compare_predictions_to_targets(predictions, targets, score_threshold=0.4)

    assert comparison["summary"]["matched_gt"] == 1
    assert comparison["summary"]["missed_gt"] == 0
    assert comparison["summary"]["duplicates"] == 1
    assert comparison["summary"]["duplicate_boxes"] == 1
    assert comparison["summary"]["unmatched_predictions"] == 1


def test_method_variants_sweeps_asahi_only() -> None:
    """Only ASAHI expands into threshold sweep variants."""

    module = load_debug_module()

    variants = module.method_variants(
        ["sahi", "asahi", "gsahi"],
        [0.45, 0.25],
        ["aspect_longest_side"],
        ["full_adaptive"],
        [None],
        [None],
        [None],
        None,
        ["full_fine"],
        [None],
        [None],
        [None],
        [None],
    )

    assert [variant.name for variant in variants] == [
        "sahi",
        "asahi_aspect_longest_side_full_adaptive_nms0.45",
        "asahi_aspect_longest_side_full_adaptive_nms0.25",
        "gsahi",
    ]


def test_method_variants_sweeps_gsahi_presets_and_sources() -> None:
    """GSAHI expands into requested GOIS preset and merge-source variants."""

    module = load_debug_module()

    variants = module.method_variants(
        ["gsahi"],
        [0.5],
        ["aspect_longest_side"],
        ["full_adaptive"],
        [None],
        [None],
        [None],
        ["c1", "c2"],
        ["full_fine", "fine"],
        [None],
        [None],
        [None],
        [None],
        [None],
    )

    assert [variant.name for variant in variants] == [
        "gsahi_c1_full_fine",
        "gsahi_c1_fine",
        "gsahi_c2_full_fine",
        "gsahi_c2_fine",
    ]
    assert [variant.nms_iou_threshold for variant in variants] == [0.3, 0.3, 0.4, 0.4]


def test_method_variants_sweeps_gsahi_nms_thresholds() -> None:
    """GSAHI can override GOIS preset NMS thresholds for benchmark sweeps."""

    module = load_debug_module()

    variants = module.method_variants(
        ["gsahi"],
        [0.5],
        ["aspect_longest_side"],
        ["full_adaptive"],
        [None],
        [None],
        [None],
        ["c2"],
        ["full_coarse_fine"],
        [None],
        [None],
        [None],
        [None],
        gsahi_nms_thresholds=[0.5],
    )

    assert [variant.name for variant in variants] == ["gsahi_c2_full_coarse_fine_nms0.50"]
    assert [variant.nms_iou_threshold for variant in variants] == [0.5]


def test_method_variants_sweeps_asahi_resize_policies() -> None:
    """ASAHI expands across requested NMS thresholds and resize policies."""

    module = load_debug_module()

    variants = module.method_variants(
        ["asahi"],
        [0.5],
        ["aspect_longest_side", "square_stretch"],
        ["full_adaptive"],
        [None],
        [None],
        [None],
        None,
        ["full_fine"],
        [None],
        [None],
        [None],
        [None],
    )

    assert [variant.name for variant in variants] == [
        "asahi_aspect_longest_side_full_adaptive_nms0.50",
        "asahi_square_stretch_full_adaptive_nms0.50",
    ]
    assert [variant.asahi_resize_policy for variant in variants] == ["aspect_longest_side", "square_stretch"]


def test_method_variants_sweeps_asahi_source_modes() -> None:
    """ASAHI variants should expose full/adaptive source isolation for geometry debugging."""
    module = load_debug_module()

    variants = module.method_variants(
        ["asahi"],
        [0.5],
        ["square_stretch"],
        ["adaptive", "full", "full_adaptive"],
        [None],
        [None],
        [None],
        None,
        ["full_fine"],
        [None],
        [None],
        [None],
        [None],
    )

    assert [variant.name for variant in variants] == [
        "asahi_square_stretch_adaptive_nms0.50",
        "asahi_square_stretch_full_nms0.50",
        "asahi_square_stretch_full_adaptive_nms0.50",
    ]
    assert [variant.asahi_source_mode for variant in variants] == ["adaptive", "full", "full_adaptive"]


def test_method_variants_sweeps_asahi_window_geometry() -> None:
    """ASAHI variants should expand overlap, threshold, and high-patch debug sweeps."""
    module = load_debug_module()

    variants = module.method_variants(
        ["asahi"],
        [0.5],
        ["square_context"],
        ["adaptive"],
        [0.1, 0.2],
        [1818],
        [6, 12],
        None,
        ["full_fine"],
        [None],
        [None],
        [None],
        [None],
    )

    assert [variant.name for variant in variants] == [
        "asahi_square_context_adaptive_nms0.50_ov0.10_thr1818_hp6",
        "asahi_square_context_adaptive_nms0.50_ov0.10_thr1818_hp12",
        "asahi_square_context_adaptive_nms0.50_ov0.20_thr1818_hp6",
        "asahi_square_context_adaptive_nms0.50_ov0.20_thr1818_hp12",
    ]
    assert [variant.asahi_overlap_ratio for variant in variants] == [0.1, 0.1, 0.2, 0.2]
    assert [variant.asahi_high_patch_count for variant in variants] == [6, 12, 6, 12]


def test_method_variants_sweeps_source_aware_duplicate_suppression() -> None:
    """Source-aware duplicate suppression should expand as a debug-only variant axis."""
    module = load_debug_module()

    variants = module.method_variants(
        ["sahi", "asahi"],
        [0.5],
        ["square_context"],
        ["adaptive"],
        [None],
        [None],
        [None],
        None,
        ["full_fine"],
        [None],
        [None],
        [None],
        [None],
        [True, False],
    )

    assert [variant.name for variant in variants] == [
        "sahi_sads_on",
        "sahi_sads_off",
        "asahi_square_context_adaptive_nms0.50_sads_on",
        "asahi_square_context_adaptive_nms0.50_sads_off",
    ]
    assert [variant.source_aware_duplicate_suppression for variant in variants] == [True, False, True, False]


def test_method_variants_sweeps_sahi_and_asahi_tile_batch_sizes() -> None:
    """SAHI and ASAHI variants should expose tile batch size as a speed sweep."""
    module = load_debug_module()

    variants = module.method_variants(
        ["sahi", "asahi"],
        [0.5],
        ["square_context"],
        ["adaptive"],
        [None],
        [None],
        [None],
        None,
        ["full_fine"],
        [None],
        [None],
        [None],
        [None],
        sahi_tile_batch_sizes=[4, 8],
        asahi_tile_batch_sizes=[6, 12],
    )

    assert [variant.name for variant in variants] == [
        "sahi_tb4",
        "sahi_tb8",
        "asahi_square_context_adaptive_nms0.50_tb6",
        "asahi_square_context_adaptive_nms0.50_tb12",
    ]
    assert [variant.sahi_tile_batch_size for variant in variants] == [4, 8, None, None]
    assert [variant.asahi_tile_batch_size for variant in variants] == [None, None, 6, 12]


def test_method_variants_sweeps_tile_dtype_memory_format_and_cross_image_batching() -> None:
    """SAHI and ASAHI speed variants should expose tile dtype, layout, and global batching."""
    module = load_debug_module()

    variants = module.method_variants(
        ["sahi", "asahi"],
        [0.5],
        ["square_context"],
        ["adaptive"],
        [None],
        [None],
        [None],
        None,
        ["full_fine"],
        [None],
        [None],
        [None],
        [None],
        tile_input_dtypes=["fp32", "bf16"],
        tile_memory_formats=["contiguous", "channels_last"],
        batch_across_images_values=[False, True],
        asahi_window_resize_longest_sides=[640, 768],
    )

    assert len(variants) == 24
    assert variants[0].name == "sahi_fp32_contiguous_ximg_off"
    assert variants[7].name == "sahi_bf16_channels_last_ximg_on"
    assert variants[8].name == "asahi_square_context_adaptive_nms0.50_rs640_fp32_contiguous_ximg_off"
    assert variants[-1].name == "asahi_square_context_adaptive_nms0.50_rs768_bf16_channels_last_ximg_on"
    assert {variant.tile_input_dtype for variant in variants} == {"fp32", "bf16"}
    assert {variant.tile_memory_format for variant in variants} == {"contiguous", "channels_last"}
    assert {variant.batch_across_images for variant in variants} == {False, True}
    assert {variant.asahi_window_resize_longest_side for variant in variants if variant.method == "asahi"} == {
        640,
        768,
    }


def test_quick_benchmark_sweep_preset_limits_variant_axes() -> None:
    """The quick benchmark preset should keep nuisance axes fixed."""

    module = load_debug_module()
    args = SimpleNamespace(sweep_preset="quick", methods=["sahi", "asahi", "gsahi"])

    module.apply_benchmark_sweep_preset(args)
    variants = module.method_variants(
        args.methods,
        args.asahi_nms_thresholds,
        args.asahi_resize_policies,
        args.asahi_source_modes,
        args.asahi_overlap_ratios,
        args.asahi_short_side_thresholds,
        args.asahi_high_patch_counts,
        args.gsahi_presets,
        args.gsahi_merge_sources,
        args.gsahi_roi_score_thresholds,
        args.gsahi_roi_max_regions,
        args.gsahi_fine_overlaps,
        args.gsahi_tile_batch_sizes,
        source_aware_duplicate_suppressions=module.source_aware_sweep_values(
            args.source_aware_duplicate_suppression
        ),
        asahi_tile_batch_sizes=args.asahi_tile_batch_sizes,
        gsahi_include_full_images=module.optional_bool_sweep_values(
            args.gsahi_include_full_image,
            name="GSAHI include-full-image",
        ),
        tile_input_dtypes=args.tile_input_dtypes,
        tile_memory_formats=args.tile_memory_formats,
        batch_across_images_values=module.optional_bool_sweep_values(
            args.batch_across_images,
            name="batch-across-images",
        ),
        asahi_window_resize_longest_sides=args.asahi_window_resize_longest_sides,
        gsahi_nms_thresholds=args.gsahi_nms_thresholds,
    )

    assert len(variants) == 13
    assert {variant.nms_iou_threshold for variant in variants if variant.method in {"asahi", "gsahi"}} == {0.5}
    assert {variant.asahi_source_mode for variant in variants if variant.method == "asahi"} == {"adaptive"}
    assert {variant.asahi_window_resize_longest_side for variant in variants if variant.method == "asahi"} == {
        640,
        768,
    }
    assert {variant.gsahi_include_full_image for variant in variants if variant.method == "gsahi"} == {False}


def test_asahi_full_benchmark_sweep_preset_compares_full_image_only_where_requested() -> None:
    """The ASAHI full-image preset should be compact and ASAHI-only."""

    module = load_debug_module()
    args = SimpleNamespace(sweep_preset="asahi_full", methods=["sahi", "asahi", "gsahi"])

    module.apply_benchmark_sweep_preset(args)
    variants = module.method_variants(
        args.methods,
        args.asahi_nms_thresholds,
        args.asahi_resize_policies,
        args.asahi_source_modes,
        args.asahi_overlap_ratios,
        args.asahi_short_side_thresholds,
        args.asahi_high_patch_counts,
        args.gsahi_presets,
        args.gsahi_merge_sources,
        args.gsahi_roi_score_thresholds,
        args.gsahi_roi_max_regions,
        args.gsahi_fine_overlaps,
        args.gsahi_tile_batch_sizes,
        source_aware_duplicate_suppressions=module.source_aware_sweep_values(
            args.source_aware_duplicate_suppression
        ),
        asahi_tile_batch_sizes=args.asahi_tile_batch_sizes,
        gsahi_include_full_images=module.optional_bool_sweep_values(
            args.gsahi_include_full_image,
            name="GSAHI include-full-image",
        ),
        tile_input_dtypes=args.tile_input_dtypes,
        tile_memory_formats=args.tile_memory_formats,
        batch_across_images_values=module.optional_bool_sweep_values(
            args.batch_across_images,
            name="batch-across-images",
        ),
        asahi_window_resize_longest_sides=args.asahi_window_resize_longest_sides,
        gsahi_nms_thresholds=args.gsahi_nms_thresholds,
    )

    assert len(variants) == 8
    assert {variant.method for variant in variants} == {"asahi"}
    assert {variant.asahi_source_mode for variant in variants} == {"adaptive", "full_adaptive"}


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        pytest.param("config", [None], id="config"),
        pytest.param("on", [True], id="on"),
        pytest.param("off", [False], id="off"),
        pytest.param("both", [True, False], id="both"),
    ],
)
def test_source_aware_sweep_values(mode: str, expected: list[bool | None]) -> None:
    """CLI source-aware mode should map to concrete variant values."""
    module = load_debug_module()

    assert module.source_aware_sweep_values(mode) == expected


def test_set_variant_config_applies_gsahi_sweep_fields() -> None:
    """GSAHI debug variants should be expressed as train_config changes."""
    module = load_debug_module()
    train_config = SimpleNamespace(
        validation_mode="standard",
        validation_score_threshold=0.0,
        sahi=SimpleNamespace(),
        asahi=SimpleNamespace(),
        gsahi=SimpleNamespace(
            coarse_slice_size=1,
            fine_slice_size=1,
            coarse_overlap=0.0,
            fine_overlap=0.0,
            include_full_image=True,
            nms_iou_threshold=0.0,
            merge_sources="full_fine",
            roi_score_threshold=None,
            roi_max_regions=1,
            tile_batch_size=1,
            source_aware_duplicate_suppression=False,
        ),
    )
    variant = module.MethodVariant(
        method="gsahi",
        name="gsahi_c2_full_coarse_fine",
        gsahi_preset="c2",
        gsahi_merge_sources="full_coarse_fine",
        gsahi_include_full_image=False,
        gsahi_roi_score_threshold=0.25,
        gsahi_roi_max_regions=4,
        gsahi_fine_overlap=0.15,
        gsahi_tile_batch_size=8,
        source_aware_duplicate_suppression=True,
    )

    module.set_variant_config(train_config, variant, 0.42)

    assert train_config.validation_mode == "gsahi"
    assert train_config.validation_score_threshold == pytest.approx(0.42)
    assert train_config.gsahi.coarse_slice_size == 640
    assert train_config.gsahi.fine_slice_size == 256
    assert train_config.gsahi.coarse_overlap == pytest.approx(0.2)
    assert train_config.gsahi.fine_overlap == pytest.approx(0.15)
    assert train_config.gsahi.nms_iou_threshold == pytest.approx(0.4)
    assert train_config.gsahi.merge_sources == "full_coarse_fine"
    assert train_config.gsahi.include_full_image is False
    assert train_config.gsahi.roi_score_threshold == pytest.approx(0.25)
    assert train_config.gsahi.roi_max_regions == 4
    assert train_config.gsahi.tile_batch_size == 8
    assert train_config.gsahi.source_aware_duplicate_suppression is True


def test_set_variant_config_applies_sahi_and_asahi_tile_batch_sizes() -> None:
    """Tile batch size debug variants should update the matching method config."""
    module = load_debug_module()
    train_config = SimpleNamespace(
        validation_mode="standard",
        validation_score_threshold=0.0,
        sahi=SimpleNamespace(tile_batch_size=4, tile_input_dtype="auto", tile_memory_format="contiguous"),
        asahi=SimpleNamespace(
            tile_batch_size=4,
            window_resize_longest_side=12,
            tile_input_dtype="auto",
            tile_memory_format="contiguous",
            batch_across_images=False,
        ),
        gsahi=SimpleNamespace(),
    )

    module.set_variant_config(
        train_config,
        module.MethodVariant(method="sahi", name="sahi_tb16", sahi_tile_batch_size=16),
        0.42,
    )
    assert train_config.validation_mode == "sahi"
    assert train_config.sahi.tile_batch_size == 16
    assert train_config.asahi.tile_batch_size == 4

    module.set_variant_config(
        train_config,
        module.MethodVariant(
            method="asahi",
            name="asahi_tb12",
            asahi_tile_batch_size=12,
            asahi_window_resize_longest_side=768,
            tile_input_dtype="bf16",
            tile_memory_format="channels_last",
            batch_across_images=True,
        ),
        0.37,
    )
    assert train_config.validation_mode == "asahi"
    assert train_config.validation_score_threshold == pytest.approx(0.37)
    assert train_config.sahi.tile_batch_size == 16
    assert train_config.asahi.tile_batch_size == 12
    assert train_config.asahi.window_resize_longest_side == 768
    assert train_config.asahi.tile_input_dtype == "bf16"
    assert train_config.asahi.tile_memory_format == "channels_last"
    assert train_config.asahi.batch_across_images is True


def test_run_variant_prediction_delegates_to_module_prediction_helper() -> None:
    """Variant prediction should reuse RF-DETR's validation implementation."""
    module = load_debug_module()
    batch = ("samples", [{"image_id": 1}])
    calls = []

    def predict_validation_batch_with_model(
        model_arg: object,
        batch_arg: object,
        *,
        return_diagnostics: bool = False,
        timing_stats: object | None = None,
    ) -> dict[str, list[str]]:
        calls.append((model_arg, batch_arg, return_diagnostics, timing_stats))
        return {"results": ["prediction"], "targets": ["target"]}

    dummy_module = SimpleNamespace(
        model="model",
        predict_validation_batch_with_model=predict_validation_batch_with_model,
    )

    assert module.run_variant_prediction(dummy_module, batch) == {
        "results": ["prediction"],
        "targets": ["target"],
    }
    assert calls == [("model", batch, False, None)]

    module.run_variant_prediction(dummy_module, batch, return_diagnostics=True)
    assert calls[-1] == ("model", batch, True, None)


def test_prediction_timing_reports_average_seconds_and_fps() -> None:
    """CUDA timing summaries should average repeats and compute images per second."""
    module = load_debug_module()

    timing = module.prediction_timing(
        6,
        [2.0, 4.0],
        device=torch.device("cuda"),
        amp_dtype=torch.bfloat16,
        include_transfer_time=False,
    )

    assert timing["prediction_seconds"] == pytest.approx(3.0)
    assert timing["images_per_second"] == pytest.approx(2.0)
    assert timing["timing_device"] == "cuda"
    assert timing["fps_valid"] is True
    assert timing["amp_dtype"] == "bfloat16"
    assert timing["include_transfer_time"] is False
    assert timing["num_images"] == 6
    assert timing["timing_repeats"] == 2
    assert timing["prediction_seconds_per_repeat"] == [2.0, 4.0]


def test_prediction_timing_does_not_report_cpu_fps() -> None:
    """CPU debug runs should not emit invalid FPS comparisons."""
    module = load_debug_module()

    timing = module.prediction_timing(
        6,
        [2.0, 4.0],
        device=torch.device("cpu"),
        amp_dtype=None,
        include_transfer_time=True,
    )

    assert timing["prediction_seconds"] == pytest.approx(3.0)
    assert timing["images_per_second"] is None
    assert timing["timing_device"] == "cpu"
    assert timing["fps_valid"] is False
    assert timing["include_transfer_time"] is True
    assert timing["amp_dtype"] == "off"


def test_parse_args_defaults_to_ten_warmups(monkeypatch: pytest.MonkeyPatch) -> None:
    """FPS benchmarks should default to ten untimed warmup repeats."""
    module = load_debug_module()

    monkeypatch.setattr(sys, "argv", ["debug_tiled_validation_first_batch.py", "--checkpoint", "checkpoint.pth"])

    args = module.parse_args()

    assert args.timing_warmup == 10


def test_effective_warmup_repeats_enforces_minimum() -> None:
    """Explicitly lower warmup requests should still run ten repeats."""
    module = load_debug_module()

    assert module.effective_warmup_repeats(0) == 10
    assert module.effective_warmup_repeats(1) == 10
    assert module.effective_warmup_repeats(10) == 10
    assert module.effective_warmup_repeats(12) == 12


def test_sync_cuda_if_needed_calls_cuda_synchronize(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA timing should synchronize before and after measured prediction."""
    module = load_debug_module()
    calls = []

    monkeypatch.setattr(module.torch.cuda, "synchronize", lambda device: calls.append(device))

    module.sync_cuda_if_needed(torch.device("cuda"))

    assert calls == [torch.device("cuda")]


def test_batch_artifact_stem_preserves_first_batch_name() -> None:
    """Single-batch artifacts should keep their historical first_batch stem."""
    module = load_debug_module()

    assert module.batch_artifact_stem("sahi", batch_index=0, num_batches=1) == "first_batch_sahi"
    assert module.batch_artifact_stem("sahi", batch_index=1, num_batches=3) == "batch001_sahi"


def test_compare_predictions_emits_geometry_diagnostics() -> None:
    """Prediction details should include local crop and resize metadata when present."""

    module = load_debug_module()
    predictions = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1]),
            "_local_boxes": torch.tensor([[1.0, 1.0, 3.0, 3.0]]),
            "_model_sizes": torch.tensor([[16.0, 16.0]]),
            "_resize_scales": torch.tensor([[2.0, 2.0]]),
            "_resize_paddings": torch.tensor([[0.0, 0.0, 0.0, 8.0]]),
        }
    ]
    targets = [
        {
            "boxes": torch.tensor([[15.0, 15.0, 10.0, 10.0]]) / 100.0,
            "labels": torch.tensor([1]),
            "orig_size": torch.tensor([100, 100]),
        }
    ]

    comparison = module.compare_predictions_to_targets(predictions, targets, score_threshold=0.4)

    geometry = comparison["images"][0]["gt"][0]["matched_predictions"][0]["geometry"]
    gt_detail = comparison["images"][0]["gt"][0]
    assert geometry["local_box"] == [1.0, 1.0, 3.0, 3.0]
    assert geometry["model_size"] == [16.0, 16.0]
    assert geometry["resize_scale"] == [2.0, 2.0]
    assert geometry["resize_padding"] == [0.0, 0.0, 0.0, 8.0]
    assert gt_detail["center_delta"] == [0.0, 0.0]
    assert comparison["summary"]["mean_center_delta_x"] == 0.0
    assert comparison["summary"]["mean_center_delta_y"] == 0.0


def test_compare_predictions_reports_signed_box_shape_errors() -> None:
    """Comparison summaries should expose signed size errors and box ratios."""
    module = load_debug_module()
    predictions = [
        {
            "boxes": torch.tensor([[9.0, 11.0, 21.0, 19.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1]),
        }
    ]
    targets = [
        {
            "boxes": torch.tensor([[15.0, 15.0, 10.0, 10.0]]) / 100.0,
            "labels": torch.tensor([1]),
            "orig_size": torch.tensor([100, 100]),
        }
    ]

    comparison = module.compare_predictions_to_targets(predictions, targets, score_threshold=0.4)
    summary = comparison["summary"]
    gt_detail = comparison["images"][0]["gt"][0]

    assert summary["mean_signed_width_error"] == pytest.approx(2.0)
    assert summary["mean_signed_height_error"] == pytest.approx(-2.0)
    assert summary["mean_width_ratio"] == pytest.approx(1.2)
    assert summary["mean_height_ratio"] == pytest.approx(0.8)
    assert summary["mean_area_ratio"] == pytest.approx(0.96)
    assert gt_detail["signed_width_error"] == pytest.approx(2.0)
    assert gt_detail["signed_height_error"] == pytest.approx(-2.0)
    assert gt_detail["width_ratio"] == pytest.approx(1.2)
    assert gt_detail["height_ratio"] == pytest.approx(0.8)
    assert gt_detail["area_ratio"] == pytest.approx(0.96)


def test_build_per_gt_comparison_rows_collects_best_prediction_geometry() -> None:
    """Per-GT rows should expose best match geometry across variants."""
    module = load_debug_module()
    payloads = [
        {
            "variant": "asahi_square_context",
            "method": "asahi",
            "images": [
                {
                    "image_index": 0,
                    "gt": [
                        {
                            "gt_index": 0,
                            "class": 1,
                            "best_iou": 0.8,
                            "center_offset": 1.0,
                            "center_delta": [1.0, 0.0],
                            "width_ratio": 1.1,
                            "height_ratio": 0.9,
                            "area_ratio": 0.99,
                            "signed_width_error": 1.0,
                            "signed_height_error": -1.0,
                            "best_source_stage": "adaptive",
                            "best_prediction_index": 3,
                            "matched_predictions": [
                                {
                                    "prediction_index": 3,
                                    "source_window": [0.0, 0.0, 10.0, 10.0],
                                    "geometry": {
                                        "inference_window": [0.0, 0.0, 12.0, 12.0],
                                        "target_size": [640.0, 640.0],
                                        "valid_size": [640.0, 640.0],
                                    },
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    rows = module.build_per_gt_comparison_rows(payloads)

    assert rows == [
        {
            "variant": "asahi_square_context",
            "method": "asahi",
            "image_index": 0,
            "gt_index": 0,
            "class": 1,
            "best_iou": 0.8,
            "center_offset": 1.0,
            "center_delta": [1.0, 0.0],
            "width_ratio": 1.1,
            "height_ratio": 0.9,
            "area_ratio": 0.99,
            "signed_width_error": 1.0,
            "signed_height_error": -1.0,
            "best_source_stage": "adaptive",
            "source_window": [0.0, 0.0, 10.0, 10.0],
            "inference_window": [0.0, 0.0, 12.0, 12.0],
            "target_size": [640.0, 640.0],
            "valid_size": [640.0, 640.0],
        }
    ]
