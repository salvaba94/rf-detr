# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""RF-DETR-native multi-image augmentations."""

from __future__ import annotations

from typing import Any

import numpy as np
import PIL
import torch
from PIL import Image

from rfdetr.utilities.logger import get_logger

logger = get_logger()


class CopyPaste:
    """Copy annotated objects within the same image and paste them at new locations."""

    def __init__(
        self,
        p: float = 0.5,
        max_paste_objects: int = 3,
        paste_attempts: int = 20,
        min_area: float = 1.0,
        max_iou: float = 0.3,
        use_masks: bool = True,
        seed: int | None = None,
    ) -> None:
        """Initialize CopyPaste augmentation."""
        if not 0.0 <= p <= 1.0:
            raise ValueError("CopyPaste p must be in [0, 1]")
        if max_paste_objects <= 0:
            raise ValueError("CopyPaste max_paste_objects must be positive")
        if paste_attempts <= 0:
            raise ValueError("CopyPaste paste_attempts must be positive")
        self.p = float(p)
        self.max_paste_objects = int(max_paste_objects)
        self.paste_attempts = int(paste_attempts)
        self.min_area = float(min_area)
        self.max_iou = float(max_iou)
        self.use_masks = bool(use_masks)
        self._rng = np.random.default_rng(seed)
        self._additional_sample_provider = None

    def set_additional_sample_provider(self, provider: Any | None) -> None:
        """Attach a callable that returns an additional ``(image, target)`` sample."""
        self._additional_sample_provider = provider

    @staticmethod
    def _clip_boxes(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
        """Clip XYXY boxes to image bounds."""
        clipped = boxes.clone().to(dtype=torch.float32)
        clipped[:, 0::2].clamp_(0, width)
        clipped[:, 1::2].clamp_(0, height)
        return clipped

    @staticmethod
    def _box_iou_one_to_many(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU between one XYXY box and many XYXY boxes."""
        if boxes.numel() == 0:
            return torch.zeros((0,), dtype=torch.float32)
        lt = torch.maximum(box[:2], boxes[:, :2])
        rb = torch.minimum(box[2:], boxes[:, 2:])
        wh = (rb - lt).clamp(min=0)
        intersection = wh[:, 0] * wh[:, 1]
        box_area = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
        boxes_area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        union = box_area + boxes_area - intersection
        return torch.where(union > 0, intersection / union, torch.zeros_like(union))

    @staticmethod
    def _mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        """Return tight XYXY bounds for a binary mask."""
        ys, xs = np.nonzero(mask)
        if xs.size == 0 or ys.size == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    def _source_object_bounds(
        self,
        box: torch.Tensor,
        mask: np.ndarray | None,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int] | None:
        """Return the source crop bounds for an object."""
        if mask is not None and mask.any():
            return self._mask_bounds(mask)
        x1, y1, x2, y2 = box.round().to(dtype=torch.int64).tolist()
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _sample_destination(
        self,
        object_width: int,
        object_height: int,
        image_width: int,
        image_height: int,
        existing_boxes: torch.Tensor,
    ) -> tuple[int, int, torch.Tensor] | None:
        """Sample a valid destination top-left and pasted box."""
        if object_width <= 0 or object_height <= 0 or object_width > image_width or object_height > image_height:
            return None
        max_x = image_width - object_width
        max_y = image_height - object_height
        for _ in range(self.paste_attempts):
            dst_x = int(self._rng.integers(0, max_x + 1)) if max_x > 0 else 0
            dst_y = int(self._rng.integers(0, max_y + 1)) if max_y > 0 else 0
            new_box = torch.tensor(
                [dst_x, dst_y, dst_x + object_width, dst_y + object_height],
                dtype=torch.float32,
            )
            ious = self._box_iou_one_to_many(new_box, existing_boxes)
            max_iou = float(ious.max().item()) if ious.numel() > 0 else 0.0
            if max_iou <= self.max_iou:
                return dst_x, dst_y, new_box
        return None

    @staticmethod
    def _append_tensor_field(value: torch.Tensor, appended: list[torch.Tensor]) -> torch.Tensor:
        """Append per-instance tensor values preserving dtype and device."""
        if not appended:
            return value
        return torch.cat([value, *[item.to(device=value.device, dtype=value.dtype) for item in appended]], dim=0)

    @staticmethod
    def _translate_keypoints(
        keypoints: torch.Tensor,
        source_index: int,
        dx: int,
        dy: int,
        image_width: int,
        image_height: int,
    ) -> torch.Tensor:
        """Copy one instance's keypoints and translate visible coordinates."""
        pasted_keypoints = keypoints[source_index].clone()
        visibility = None
        if pasted_keypoints.ndim >= 2 and pasted_keypoints.shape[-1] >= 3:
            visibility = pasted_keypoints[..., 2] > 0
        if visibility is None:
            pasted_keypoints[..., 0] = pasted_keypoints[..., 0] + dx
            pasted_keypoints[..., 1] = pasted_keypoints[..., 1] + dy
        else:
            pasted_keypoints[..., 0] = torch.where(
                visibility,
                pasted_keypoints[..., 0] + dx,
                pasted_keypoints[..., 0],
            )
            pasted_keypoints[..., 1] = torch.where(
                visibility,
                pasted_keypoints[..., 1] + dy,
                pasted_keypoints[..., 1],
            )
        pasted_keypoints[..., 0].clamp_(0, image_width)
        pasted_keypoints[..., 1].clamp_(0, image_height)
        return pasted_keypoints.unsqueeze(0)

    def __call__(
        self, image: PIL.Image.Image, target: dict[str, Any] | None = None
    ) -> tuple[PIL.Image.Image, dict[str, Any] | None]:
        """Apply CopyPaste using an additional sample when available."""
        if target is None or self._rng.random() > self.p or "boxes" not in target or "labels" not in target:
            return image, target

        image_np = np.array(image).copy()
        height, width = image_np.shape[:2]
        boxes = self._clip_boxes(target["boxes"], width, height)
        labels = target["labels"]
        num_boxes = int(boxes.shape[0])

        source_image = image
        source_target = target
        if self._additional_sample_provider is not None:
            try:
                provided = self._additional_sample_provider()
            except Exception as exc:
                logger.debug("CopyPaste additional sample provider failed: %s", exc)
                provided = None
            if provided is not None:
                source_image, source_target = provided

        source_image_np = np.array(source_image).copy()
        source_height, source_width = source_image_np.shape[:2]
        if source_target is None or "boxes" not in source_target or "labels" not in source_target:
            return image, target
        source_boxes = self._clip_boxes(source_target["boxes"], source_width, source_height)
        source_labels = source_target["labels"]
        source_num_boxes = int(source_boxes.shape[0])
        if source_num_boxes == 0:
            return image, target

        masks_tensor = target.get("masks")
        source_masks_tensor = source_target.get("masks")
        masks_np = None
        if (
            self.use_masks
            and torch.is_tensor(source_masks_tensor)
            and source_masks_tensor.ndim == 3
            and source_masks_tensor.shape[0] == source_num_boxes
        ):
            masks_np = source_masks_tensor.cpu().numpy().astype(bool, copy=False)

        valid_areas = (source_boxes[:, 2] - source_boxes[:, 0]).clamp(min=0) * (
            source_boxes[:, 3] - source_boxes[:, 1]
        ).clamp(min=0)
        valid_indices = torch.nonzero(valid_areas >= self.min_area, as_tuple=False).flatten().cpu().numpy()
        if valid_indices.size == 0:
            return image, target

        num_to_paste = int(self._rng.integers(1, min(self.max_paste_objects, valid_indices.size) + 1))
        source_indices = self._rng.choice(valid_indices, size=num_to_paste, replace=False)
        existing_boxes = boxes.clone()
        new_boxes: list[torch.Tensor] = []
        new_labels: list[torch.Tensor] = []
        new_masks: list[torch.Tensor] = []
        new_area: list[torch.Tensor] = []
        new_keypoints: list[torch.Tensor] = []
        copied_fields: dict[str, list[torch.Tensor]] = {}
        handled_fields = {"boxes", "labels", "masks", "area", "orig_size", "size", "image_id", "keypoints"}
        keypoints_tensor = target.get("keypoints")
        source_keypoints_tensor = source_target.get("keypoints")
        has_keypoints = (
            torch.is_tensor(keypoints_tensor)
            and keypoints_tensor.ndim >= 2
            and keypoints_tensor.shape[0] == num_boxes
            and torch.is_tensor(source_keypoints_tensor)
            and source_keypoints_tensor.ndim >= 2
            and source_keypoints_tensor.shape[0] == source_num_boxes
        )

        for source_index in source_indices.tolist():
            mask = masks_np[source_index] if masks_np is not None else None
            bounds = self._source_object_bounds(source_boxes[source_index], mask, source_width, source_height)
            if bounds is None:
                continue
            x1, y1, x2, y2 = bounds
            object_width = x2 - x1
            object_height = y2 - y1
            if object_width * object_height < self.min_area:
                continue
            sampled = self._sample_destination(object_width, object_height, width, height, existing_boxes)
            if sampled is None:
                continue
            dst_x, dst_y, new_box = sampled
            object_pixels = source_image_np[y1:y2, x1:x2]
            destination = image_np[dst_y : dst_y + object_height, dst_x : dst_x + object_width]

            pasted_mask_tensor: torch.Tensor | None = None
            if mask is not None:
                object_mask = mask[y1:y2, x1:x2]
                destination[object_mask] = object_pixels[object_mask]
                pasted_mask = np.zeros((height, width), dtype=bool)
                pasted_mask[dst_y : dst_y + object_height, dst_x : dst_x + object_width] = object_mask
                pasted_mask_tensor = torch.as_tensor(pasted_mask, dtype=torch.bool)
                tight_bounds = self._mask_bounds(pasted_mask)
                if tight_bounds is None:
                    continue
                new_box = torch.tensor(tight_bounds, dtype=torch.float32)
            else:
                destination[:, :] = object_pixels
                if torch.is_tensor(masks_tensor):
                    pasted_mask = np.zeros((height, width), dtype=bool)
                    pasted_mask[dst_y : dst_y + object_height, dst_x : dst_x + object_width] = True
                    pasted_mask_tensor = torch.as_tensor(pasted_mask, dtype=torch.bool)

            existing_boxes = torch.cat([existing_boxes, new_box.unsqueeze(0)], dim=0)
            new_boxes.append(new_box.unsqueeze(0))
            new_labels.append(source_labels[source_index].reshape(1))
            if has_keypoints:
                new_keypoints.append(
                    self._translate_keypoints(
                        source_keypoints_tensor,
                        source_index,
                        dst_x - x1,
                        dst_y - y1,
                        width,
                        height,
                    )
                )
            if pasted_mask_tensor is not None:
                new_masks.append(pasted_mask_tensor.unsqueeze(0))
            box_area = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])
            new_area.append(box_area.reshape(1))

            for key, value in source_target.items():
                if key in handled_fields or key not in target:
                    continue
                base_value = target[key]
                if (
                    torch.is_tensor(value)
                    and torch.is_tensor(base_value)
                    and value.ndim >= 1
                    and base_value.ndim >= 1
                    and value.shape[0] == source_num_boxes
                    and base_value.shape[0] == num_boxes
                ):
                    copied_fields.setdefault(key, []).append(value[source_index].unsqueeze(0))

        if not new_boxes:
            return image, target

        target_out = target.copy()
        target_out["boxes"] = torch.cat([boxes, *new_boxes], dim=0).to(dtype=torch.float32)
        target_out["labels"] = self._append_tensor_field(labels, new_labels)
        if "area" in target_out:
            target_out["area"] = self._append_tensor_field(target_out["area"], new_area)
        if torch.is_tensor(masks_tensor):
            if new_masks:
                target_out["masks"] = self._append_tensor_field(masks_tensor.bool(), new_masks)
            else:
                target_out["masks"] = masks_tensor.bool()
        if has_keypoints:
            target_out["keypoints"] = self._append_tensor_field(keypoints_tensor, new_keypoints)
        for key, values in copied_fields.items():
            target_out[key] = self._append_tensor_field(target_out[key], values)

        return Image.fromarray(image_np), target_out


class MixUp:
    """Blend an image with an additional training sample and merge annotations."""

    def __init__(self, alpha: float = 1.0, p: float = 0.5, seed: int | None = None, **_: Any) -> None:
        """Initialize MixUp augmentation."""
        if alpha < 0:
            raise ValueError(f"MixUp alpha must be >= 0, got {alpha}")
        if not 0.0 <= p <= 1.0:
            raise ValueError("MixUp p must be in [0, 1]")
        self.alpha = float(alpha)
        self.p = float(p)
        self._rng = np.random.default_rng(seed)
        self._additional_sample_provider = None

    def set_additional_sample_provider(self, provider: Any | None) -> None:
        """Attach a callable that returns an additional ``(image, target)`` sample."""
        self._additional_sample_provider = provider

    @staticmethod
    def _scale_boxes(boxes: torch.Tensor, scale_x: float, scale_y: float, width: int, height: int) -> torch.Tensor:
        """Scale absolute XYXY boxes and clip to the target image size."""
        scaled = boxes.clone().to(dtype=torch.float32)
        scaled[:, [0, 2]] *= float(scale_x)
        scaled[:, [1, 3]] *= float(scale_y)
        scaled[:, 0::2].clamp_(0, width)
        scaled[:, 1::2].clamp_(0, height)
        return scaled

    @staticmethod
    def _scale_keypoints(
        keypoints: torch.Tensor,
        scale_x: float,
        scale_y: float,
        width: int,
        height: int,
    ) -> torch.Tensor:
        """Scale visible keypoints and clamp coordinates to the target image size."""
        scaled = keypoints.clone().to(dtype=torch.float32)
        visibility = scaled[..., 2] > 0 if scaled.shape[-1] >= 3 else torch.ones_like(scaled[..., 0], dtype=torch.bool)
        scaled[..., 0] = torch.where(visibility, scaled[..., 0] * float(scale_x), scaled[..., 0])
        scaled[..., 1] = torch.where(visibility, scaled[..., 1] * float(scale_y), scaled[..., 1])
        scaled[..., 0].clamp_(0, width)
        scaled[..., 1].clamp_(0, height)
        return scaled

    @staticmethod
    def _resize_masks(masks: torch.Tensor, width: int, height: int) -> torch.Tensor:
        """Resize binary masks with nearest-neighbor interpolation."""
        resized = []
        for mask in masks.bool().cpu().numpy():
            mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
            mask_image = mask_image.resize((width, height), resample=Image.Resampling.NEAREST)
            resized.append(np.asarray(mask_image) > 0)
        if not resized:
            return torch.zeros((0, height, width), dtype=torch.bool)
        return torch.as_tensor(np.stack(resized), dtype=torch.bool)

    @staticmethod
    def _append_tensor_field(value: torch.Tensor, appended: torch.Tensor) -> torch.Tensor:
        """Append tensor values preserving dtype and device."""
        return torch.cat([value, appended.to(device=value.device, dtype=value.dtype)], dim=0)

    def _sample_lambda(self) -> float:
        """Sample the MixUp blend weight."""
        if self.alpha > 0:
            return float(self._rng.beta(self.alpha, self.alpha))
        return 0.5

    def __call__(
        self, image: PIL.Image.Image, target: dict[str, Any] | None = None
    ) -> tuple[PIL.Image.Image, dict[str, Any] | None]:
        """Apply MixUp using a dataset-provided additional sample."""
        if target is None or self._rng.random() > self.p or self._additional_sample_provider is None:
            return image, target
        if "boxes" not in target or "labels" not in target:
            return image, target

        try:
            provided = self._additional_sample_provider()
        except Exception as exc:
            logger.debug("MixUp additional sample provider failed: %s", exc)
            return image, target
        if provided is None:
            return image, target

        mix_image, mix_target = provided
        if mix_target is None or "boxes" not in mix_target or "labels" not in mix_target:
            return image, target

        image_np = np.array(image).copy()
        mix_np = np.array(mix_image).copy()
        height, width = image_np.shape[:2]
        mix_height, mix_width = mix_np.shape[:2]
        if height <= 0 or width <= 0 or mix_height <= 0 or mix_width <= 0:
            return image, target

        if (mix_width, mix_height) != (width, height):
            mix_np = np.array(Image.fromarray(mix_np).resize((width, height), resample=Image.Resampling.BILINEAR))
        scale_x = width / mix_width
        scale_y = height / mix_height
        lam = self._sample_lambda()
        mixed_np = lam * image_np.astype(np.float32) + (1.0 - lam) * mix_np.astype(np.float32)
        if image_np.dtype == np.uint8:
            mixed_np = np.clip(mixed_np, 0, 255).astype(np.uint8)
        else:
            mixed_np = mixed_np.astype(image_np.dtype, copy=False)

        target_out = target.copy()
        boxes = target["boxes"].to(dtype=torch.float32)
        mix_boxes = self._scale_boxes(mix_target["boxes"], scale_x, scale_y, width, height)
        target_out["boxes"] = torch.cat([boxes, mix_boxes], dim=0)
        target_out["labels"] = self._append_tensor_field(target["labels"], mix_target["labels"])

        if "area" in target_out:
            boxes_out = target_out["boxes"]
            target_out["area"] = (boxes_out[:, 2] - boxes_out[:, 0]) * (boxes_out[:, 3] - boxes_out[:, 1])

        target_masks = target.get("masks")
        mix_masks = mix_target.get("masks")
        if torch.is_tensor(target_masks) and torch.is_tensor(mix_masks) and mix_masks.ndim == 3:
            resized_mix_masks = self._resize_masks(mix_masks, width, height)
            target_out["masks"] = self._append_tensor_field(target_masks.bool(), resized_mix_masks)

        target_keypoints = target.get("keypoints")
        mix_keypoints = mix_target.get("keypoints")
        if (
            torch.is_tensor(target_keypoints)
            and torch.is_tensor(mix_keypoints)
            and target_keypoints.ndim >= 2
            and mix_keypoints.ndim >= 2
        ):
            scaled_mix_keypoints = self._scale_keypoints(mix_keypoints, scale_x, scale_y, width, height)
            target_out["keypoints"] = self._append_tensor_field(target_keypoints, scaled_mix_keypoints)

        global_fields = {"boxes", "labels", "masks", "area", "orig_size", "size", "image_id", "keypoints"}
        num_target = int(boxes.shape[0])
        num_mix = int(mix_boxes.shape[0])
        for key, value in mix_target.items():
            if key in global_fields or key not in target:
                continue
            base_value = target[key]
            if (
                torch.is_tensor(base_value)
                and torch.is_tensor(value)
                and base_value.ndim >= 1
                and value.ndim >= 1
                and base_value.shape[0] == num_target
                and value.shape[0] == num_mix
            ):
                target_out[key] = self._append_tensor_field(base_value, value)

        return Image.fromarray(mixed_np), target_out


def prepare_multi_image_augmentations(transforms: Any, dataset: Any, index: int) -> None:
    """Attach dataset-backed additional-sample providers to multi-image transforms."""

    def iter_transforms(transform: Any) -> Any:
        yield transform
        for child in getattr(transform, "transforms", []) or []:
            yield from iter_transforms(child)

    if not hasattr(dataset, "_get_additional_sample"):
        return

    used_indices = {index}

    def provider(dataset: Any = dataset, index: int = index, used_indices: set[int] = used_indices) -> Any | None:
        sampler_with_info = getattr(dataset, "_get_additional_sample_info", None)
        if callable(sampler_with_info):
            sampled = sampler_with_info(index, exclude_indices=used_indices)
            if sampled is None:
                return None
            sample_idx, sample = sampled
            used_indices.add(int(sample_idx))
            return sample
        return dataset._get_additional_sample(index)

    for transform in iter_transforms(transforms):
        setter = getattr(transform, "set_additional_sample_provider", None)
        if setter is not None:
            setter(provider)
