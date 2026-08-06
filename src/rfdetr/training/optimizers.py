# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Optimizer factory helpers for RF-DETR training."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import torch
from torch import optim

OptimizerName = Literal["adamw", "sgd", "muadamw", "musgd"]


class DualOptimizer(optim.Optimizer):
    """Wrap two optimizers behind a single Optimizer-compatible facade."""

    def __init__(self, primary: optim.Optimizer, secondary: optim.Optimizer) -> None:
        """Initialize the facade with two already-built optimizers.

        Args:
            primary: First optimizer to step.
            secondary: Second optimizer to step.
        """
        super().__init__(secondary.param_groups, {})
        combined = primary.param_groups + secondary.param_groups
        self.param_groups[:] = combined
        self.primary = primary
        self.secondary = secondary

    def state_dict(self) -> dict[str, Any]:
        """Return sub-optimizer state dictionaries."""
        return {
            "primary": self.primary.state_dict(),
            "secondary": self.secondary.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load sub-optimizer state dictionaries.

        Args:
            state_dict: State previously returned by :meth:`state_dict`.
        """
        if "primary" not in state_dict or "secondary" not in state_dict:
            raise ValueError("DualOptimizer state must contain 'primary' and 'secondary' entries.")
        self.primary.load_state_dict(state_dict["primary"])
        self.secondary.load_state_dict(state_dict["secondary"])
        self.param_groups[:] = self.primary.param_groups + self.secondary.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients in both sub-optimizers."""
        self.primary.zero_grad(set_to_none=set_to_none)
        self.secondary.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Any = None) -> Any:
        """Step both sub-optimizers."""
        primary_result = self.primary.step(closure)
        secondary_result = self.secondary.step(closure)
        return primary_result if primary_result is not None else secondary_result


def build_optimizer(
    *,
    name: OptimizerName | str,
    params: Iterable[dict[str, Any]],
    lr: float,
    weight_decay: float,
    fused_adamw: bool = False,
    momentum: float = 0.95,
    nesterov: bool = True,
    muon_lr_scale: float = 0.2,
    fallback_lr_scale: float = 1.0,
    ns_coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315),
    eps: float = 1e-7,
    ns_steps: int = 5,
    adjust_lr_fn: str | None = None,
) -> optim.Optimizer:
    """Build an RF-DETR optimizer by name.

    Args:
        name: Optimizer name: ``adamw``, ``sgd``, ``muadamw``, or ``musgd``.
        params: Parameter groups, typically from :func:`rfdetr.training.param_groups.get_param_dict`.
        lr: Default learning rate.
        weight_decay: Default weight decay.
        fused_adamw: Whether plain AdamW should use PyTorch's fused implementation.
        momentum: Momentum for SGD and Muon-backed optimizers.
        nesterov: Whether SGD/Muon-backed optimizers should use Nesterov momentum.
        muon_lr_scale: Learning-rate multiplier for the Muon optimizer.
        fallback_lr_scale: Learning-rate multiplier for the fallback optimizer.
        ns_coefficients: Newton-Schulz iteration coefficients for Muon updates.
        eps: Numerical stability epsilon for Muon orthogonalization.
        ns_steps: Number of Newton-Schulz iterations.
        adjust_lr_fn: Optional PyTorch Muon learning-rate adjustment mode.

    Returns:
        Configured optimizer.
    """
    optimizer_name = str(name).lower()
    groups = _normalize_param_groups(params)
    if optimizer_name == "adamw":
        return optim.AdamW(groups, lr=lr, weight_decay=weight_decay, fused=fused_adamw)
    if optimizer_name == "sgd":
        return optim.SGD(groups, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
    if optimizer_name == "muadamw":
        return _build_muon_dual_optimizer(
            groups=groups,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            muon_lr_scale=muon_lr_scale,
            fallback_lr_scale=fallback_lr_scale,
            ns_coefficients=ns_coefficients,
            eps=eps,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
            fallback="adamw",
        )
    if optimizer_name == "musgd":
        return _build_muon_dual_optimizer(
            groups=groups,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            muon_lr_scale=muon_lr_scale,
            fallback_lr_scale=fallback_lr_scale,
            ns_coefficients=ns_coefficients,
            eps=eps,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
            fallback="sgd",
        )
    raise ValueError("Unknown optimizer %r. Choose from: adamw, sgd, muadamw, musgd." % name)


def _build_muon_dual_optimizer(
    *,
    groups: list[dict[str, Any]],
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    muon_lr_scale: float,
    fallback_lr_scale: float,
    ns_coefficients: tuple[float, float, float],
    eps: float,
    ns_steps: int,
    adjust_lr_fn: str | None,
    fallback: Literal["adamw", "sgd"],
) -> optim.Optimizer:
    """Build scaled Muon for 2-D params plus a scaled fallback optimizer for all params."""
    muon_cls = getattr(optim, "Muon", None)
    if muon_cls is None:
        raise RuntimeError(
            "torch.optim.Muon is required for muadamw/musgd but is not available in this PyTorch build."
        )

    muon_groups = _scaled_param_groups(_muon_param_groups(groups), muon_lr_scale)
    fallback_groups = _scaled_param_groups(groups, fallback_lr_scale)
    if not muon_groups:
        raise ValueError("Muon optimizers require at least one 2-D trainable parameter.")

    muon_optimizer = muon_cls(
        muon_groups,
        lr=lr * muon_lr_scale,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        ns_coefficients=ns_coefficients,
        eps=eps,
        ns_steps=ns_steps,
        adjust_lr_fn=adjust_lr_fn,
    )
    if fallback == "adamw":
        fallback_optimizer = optim.AdamW(fallback_groups, lr=lr * fallback_lr_scale, weight_decay=weight_decay)
    else:
        fallback_optimizer = optim.SGD(
            fallback_groups,
            lr=lr * fallback_lr_scale,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
        )
    return DualOptimizer(muon_optimizer, fallback_optimizer)


def _muon_param_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return param groups containing only 2-D Muon params."""
    muon_groups: list[dict[str, Any]] = []
    for group in groups:
        base = {key: value for key, value in group.items() if key != "params"}
        muon_params = [param for param in group["params"] if param.ndim == 2]
        if muon_params:
            muon_groups.append({**base, "params": muon_params})
    return muon_groups


def _scaled_param_groups(groups: list[dict[str, Any]], lr_scale: float) -> list[dict[str, Any]]:
    """Return shallow-copied param groups with explicit scheduler LR fields scaled."""
    scaled_groups: list[dict[str, Any]] = []
    for group in groups:
        scaled_group = dict(group)
        for lr_key in ("lr", "initial_lr"):
            if lr_key in scaled_group:
                scaled_group[lr_key] = scaled_group[lr_key] * lr_scale
        scaled_groups.append(scaled_group)
    return scaled_groups


def _normalize_param_groups(params: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return non-empty param groups whose ``params`` entry is always a list."""
    normalized: list[dict[str, Any]] = []
    for group in params:
        group_params = group["params"]
        if isinstance(group_params, torch.Tensor):
            param_list = [group_params]
        else:
            param_list = list(group_params)
        param_list = [param for param in param_list if param.requires_grad]
        if param_list:
            normalized.append({**group, "params": param_list})
    if not normalized:
        raise ValueError("Optimizer received no trainable parameters.")
    return normalized
