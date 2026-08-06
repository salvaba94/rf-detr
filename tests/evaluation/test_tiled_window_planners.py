# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for native tiled validation window planners."""

from rfdetr.evaluation.tiled_asahi import generate_asahi_windows
from rfdetr.evaluation.tiled_core import TileWindow, generate_tile_windows
from rfdetr.evaluation.tiled_gsahi import _gsahi_fine_windows


def test_generate_tile_windows_cover_right_and_bottom_edges() -> None:
    """Tile generation should cover image borders even when stride does not divide the size."""
    windows = generate_tile_windows(
        height=10,
        width=11,
        tile_height=4,
        tile_width=5,
        overlap_height_ratio=0.25,
        overlap_width_ratio=0.2,
    )

    assert windows[0].x0 == 0
    assert windows[0].y0 == 0
    assert max(window.x1 for window in windows) == 11
    assert max(window.y1 for window in windows) == 10


def test_generate_asahi_windows_uses_landscape_low_patch_grid() -> None:
    """ASAHI low patch mode should use a 3x2 landscape grid plus full-image context."""
    windows = generate_asahi_windows(
        height=720,
        width=1280,
        short_side_threshold=1280,
        low_patch_count=6,
        high_patch_count=12,
        overlap_ratio=0.15,
        include_full_image=True,
    )

    assert len(windows) == 7
    assert windows[0] == TileWindow(0, 0, 1280, 720)
    assert max(window.x1 for window in windows[1:]) == 1280
    assert max(window.y1 for window in windows[1:]) == 720


def test_generate_asahi_windows_uses_portrait_high_patch_grid() -> None:
    """ASAHI high patch mode should use a 3x4 portrait grid above the limiting dimension."""
    windows = generate_asahi_windows(
        height=1600,
        width=900,
        short_side_threshold=1280,
        low_patch_count=6,
        high_patch_count=12,
        overlap_ratio=0.15,
        include_full_image=False,
    )

    assert len(windows) == 12
    assert len({window.x0 for window in windows}) == 3
    assert len({window.y0 for window in windows}) == 4
    assert max(window.x1 for window in windows) == 900
    assert max(window.y1 for window in windows) == 1600


def test_airborne_2448x2048_validation_windows_match_config_geometry() -> None:
    """Configured tiled validation planners should cover 2448x2048 airborne frames predictably."""
    sahi_windows = generate_tile_windows(
        height=2048,
        width=2448,
        tile_height=640,
        tile_width=640,
        overlap_height_ratio=0.25,
        overlap_width_ratio=0.25,
    )
    asahi_windows = generate_asahi_windows(
        height=2048,
        width=2448,
        short_side_threshold=1818,
        low_patch_count=6,
        high_patch_count=12,
        overlap_ratio=0.15,
        include_full_image=True,
    )

    assert len(sahi_windows) == 20
    assert len({window.x0 for window in sahi_windows}) == 5
    assert len({window.y0 for window in sahi_windows}) == 4
    assert max(window.x1 for window in sahi_windows) == 2448
    assert max(window.y1 for window in sahi_windows) == 2048
    assert len(asahi_windows) == 13
    assert asahi_windows[0] == TileWindow(0, 0, 2448, 2048)
    assert len({window.x0 for window in asahi_windows[1:]}) == 4
    assert len({window.y0 for window in asahi_windows[1:]}) == 3
    assert asahi_windows[1].x1 - asahi_windows[1].x0 == 691
    assert asahi_windows[1].y1 - asahi_windows[1].y0 == 760


def test_asahi_visdrone_like_resolution_uses_high_patch_grid_and_covers_edges() -> None:
    """A VisDrone-like 1920x1080 frame should use the 12-patch landscape grid above the limiting dimension."""
    windows = generate_asahi_windows(
        height=1080,
        width=1920,
        short_side_threshold=1818,
        low_patch_count=6,
        high_patch_count=12,
        overlap_ratio=0.15,
        include_full_image=False,
    )

    assert len(windows) == 12
    assert len({window.x0 for window in windows}) == 4
    assert len({window.y0 for window in windows}) == 3
    assert min(window.x0 for window in windows) == 0
    assert min(window.y0 for window in windows) == 0
    assert max(window.x1 for window in windows) == 1920
    assert max(window.y1 for window in windows) == 1080
    assert windows[0].x1 - windows[0].x0 == 542
    assert windows[0].y1 - windows[0].y0 == 401


def test_gsahi_fine_windows_are_generated_inside_rois() -> None:
    """GSAHI fine windows should stay inside selected full-image ROIs."""
    windows = _gsahi_fine_windows(
        rois=[TileWindow(10, 20, 110, 120)],
        full_size=(200, 200),
        fine_slice_size=64,
        fine_overlap=0.25,
    )

    assert windows
    assert min(window.x0 for window in windows) >= 10
    assert min(window.y0 for window in windows) >= 20
    assert max(window.x1 for window in windows) <= 110
    assert max(window.y1 for window in windows) <= 120


def test_gsahi_fine_windows_expand_tiny_rois_to_fine_slice_context() -> None:
    """Tiny coarse boxes should seed a full fine-stage context crop, not a micro-crop."""
    windows = _gsahi_fine_windows(
        rois=[TileWindow(45, 45, 55, 55)],
        full_size=(100, 100),
        fine_slice_size=64,
        fine_overlap=0.25,
    )

    assert windows == [TileWindow(18, 18, 82, 82)]
