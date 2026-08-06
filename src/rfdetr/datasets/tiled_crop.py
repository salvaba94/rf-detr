# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""RF-DETR custom tiled foreground crop augmentation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

try:
    import albumentations as alb
except ImportError:
    alb = None  # type: ignore[assignment]
import numpy as np


if alb is not None:
    from albumentations.augmentations.crops.transforms import CropNonEmptyMaskIfExists

    class TiledCroppingWithMasks(CropNonEmptyMaskIfExists):
        """Crop around foreground using Albumentations' non-empty-mask crop.

        RF-DETR supplies a foreground mask from real instance masks when available, or from bounding boxes for
        detection-only targets. Albumentations handles crop geometry, boxes, masks, and keypoints through its standard
        processors.
        """

        def __init__(
            self,
            height: int | None = None,
            width: int | None = None,
            target_size: int | tuple[int, int] | list[int] | None = None,
            height_range: tuple[int, int] | list[int] | None = None,
            width_range: tuple[int, int] | list[int] | None = None,
            ignore_values: list[int] | None = None,
            ignore_channels: list[int] | None = None,
            allow_empty: bool = False,
            deterministic: bool = False,
            min_visibility: float = 0.05,
            min_area: float = 1.0,
            p: float = 0.5,
            **_: Any,
        ) -> None:
            """Initialize a SAHI-style tiled foreground crop."""
            if target_size is not None:
                if isinstance(target_size, (int, float)):
                    height = width = int(target_size)
                elif isinstance(target_size, Sequence) and len(target_size) == 2:
                    height, width = int(target_size[0]), int(target_size[1])
                else:
                    raise ValueError("TiledCroppingWithMasks target_size must be an int or (height, width)")
            if height is None or width is None:
                raise ValueError("TiledCroppingWithMasks requires height/width or target_size")
            if height <= 0 or width <= 0:
                raise ValueError("TiledCroppingWithMasks height and width must be positive")
            super().__init__(
                height=int(height),
                width=int(width),
                ignore_values=ignore_values,
                ignore_channels=ignore_channels,
                p=p,
            )
            self.base_height = int(height)
            self.base_width = int(width)
            self.height_range = self._validate_size_range(height_range, "height_range")
            self.width_range = self._validate_size_range(width_range, "width_range")
            self.allow_empty = bool(allow_empty)
            self.deterministic_crop = bool(deterministic)
            self.min_visibility = float(min_visibility)
            self.min_area = float(min_area)

        @staticmethod
        def _validate_size_range(
            value: tuple[int, int] | list[int] | None,
            name: str,
        ) -> tuple[int, int] | None:
            """Validate an optional inclusive crop-size range."""
            if value is None:
                return None
            if not isinstance(value, Sequence) or len(value) != 2:
                raise ValueError(f"TiledCroppingWithMasks {name} must be a two-value sequence")
            min_value, max_value = int(value[0]), int(value[1])
            if min_value <= 0 or max_value <= 0 or min_value > max_value:
                raise ValueError(f"TiledCroppingWithMasks {name} must contain positive values with min <= max")
            return min_value, max_value

        def _random_size(self) -> None:
            """Select a crop size for this call, following DA-YOLO tiled cropping."""
            if self.height_range is None:
                self.height = self.base_height
            else:
                self.height = self.py_random.randint(self.height_range[0], self.height_range[1])
            if self.width_range is None:
                self.width = self.base_width
            else:
                self.width = self.py_random.randint(self.width_range[0], self.width_range[1])

        def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
            """Preprocess masks and clamp oversized crop dimensions."""
            mask_height, mask_width = mask.shape[:2]

            if self.ignore_values is not None:
                ignore_values_np = np.array(self.ignore_values)
                mask = np.where(np.isin(mask, ignore_values_np), 0, mask)

            if mask.ndim == 3 and self.ignore_channels is not None:
                target_channels = np.array([ch for ch in range(mask.shape[-1]) if ch not in self.ignore_channels])
                mask = np.take(mask, target_channels, axis=-1)

            self.height = min(self.height, mask_height)
            self.width = min(self.width, mask_width)
            return mask

        def _clamp_crop_size(self, image_height: int, image_width: int) -> None:
            """Ensure crop dimensions fit the current image."""
            self.height = min(self.height, image_height)
            self.width = min(self.width, image_width)

        def _random_crop_coords(self, image_height: int, image_width: int) -> dict[str, tuple[int, int, int, int]]:
            """Return a random crop when no foreground guidance is available."""
            self._clamp_crop_size(image_height, image_width)
            max_x = max(image_width - self.width, 0)
            max_y = max(image_height - self.height, 0)
            x_min = self.py_random.randint(0, max_x) if max_x > 0 else 0
            y_min = self.py_random.randint(0, max_y) if max_y > 0 else 0
            return {"crop_coords": (x_min, y_min, x_min + self.width, y_min + self.height)}

        def _deterministic_crop_coords(self, mask: np.ndarray) -> dict[str, tuple[int, int, int, int]]:
            """Return a stable foreground-centered crop for evaluation."""
            mask_2d = mask.any(axis=-1) if mask.ndim == 3 else mask
            ys, xs = np.nonzero(mask_2d)
            if len(xs) == 0 or len(ys) == 0:
                image_height, image_width = mask_2d.shape[:2]
                return self._random_crop_coords(image_height, image_width)

            image_height, image_width = mask_2d.shape[:2]
            center_x = int(round((int(xs.min()) + int(xs.max())) / 2))
            center_y = int(round((int(ys.min()) + int(ys.max())) / 2))
            x_min = min(max(center_x - self.width // 2, 0), max(image_width - self.width, 0))
            y_min = min(max(center_y - self.height // 2, 0), max(image_height - self.height, 0))
            return {"crop_coords": (x_min, y_min, x_min + self.width, y_min + self.height)}

        def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
            """Select random size and then delegate crop sampling to Albumentations."""
            self._random_size()
            mask = data.get("mask")
            masks = data.get("masks")
            if mask is None and masks is not None and len(masks) > 0:
                mask = np.copy(masks[0])
                for candidate in masks[1:]:
                    mask |= candidate
            if mask is None:
                if self.allow_empty:
                    return super().get_params_dependent_on_data(params, data)
                image_height, image_width = params["shape"][:2]
                return self._random_crop_coords(image_height, image_width)

            processed_mask = self._preprocess_mask(np.asarray(mask).copy())
            if not self.allow_empty and not processed_mask.any():
                image_height, image_width = params["shape"][:2]
                return self._random_crop_coords(image_height, image_width)
            if self.deterministic_crop:
                return self._deterministic_crop_coords(processed_mask)
            data = dict(data)
            data["mask"] = processed_mask
            return super().get_params_dependent_on_data(params, data)

else:
    TiledCroppingWithMasks = None  # type: ignore[assignment]

SAHIMaskCrop = TiledCroppingWithMasks
