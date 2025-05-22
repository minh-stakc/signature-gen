"""Shared utility functions: visualisation, I/O, learning-rate scheduling."""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

import config


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> str:
    """Create *path* if it does not exist and return it."""
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def save_image_grid(
    images: np.ndarray,
    path: str,
    nrow: int = 8,
    padding: int = 2,
    bg_value: int = 255,
) -> None:
    """Save a batch of images as a single grid PNG.

    Parameters
    ----------
    images : np.ndarray
        Shape ``(N, H, W, 1)`` or ``(N, H, W)`` with values in [0, 255] uint8.
    path : str
        Destination file path.
    nrow : int
        Number of images per row in the grid.
    """
    if images.ndim == 4 and images.shape[-1] == 1:
        images = images[..., 0]

    n, h, w = images.shape[:3]
    ncol = nrow
    nrow_actual = int(np.ceil(n / ncol))

    grid_h = nrow_actual * h + (nrow_actual + 1) * padding
    grid_w = ncol * w + (ncol + 1) * padding
    grid = np.full((grid_h, grid_w), bg_value, dtype=np.uint8)

    for idx in range(n):
        row, col = divmod(idx, ncol)
        y = padding + row * (h + padding)
        x = padding + col * (w + padding)
        grid[y : y + h, x : x + w] = images[idx]

    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, grid)


def denormalize(images: tf.Tensor) -> np.ndarray:
    """Map images from [-1, 1] to [0, 255] uint8."""
    out = ((images + 1.0) / 2.0 * 255.0)
    if isinstance(out, tf.Tensor):
        out = out.numpy()
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

class LinearDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linearly decay the learning rate from *initial_lr* to 0 between
    ``decay_start_step`` and ``decay_end_step``."""

    def __init__(
        self,
        initial_lr: float,
        decay_start_step: int,
        decay_end_step: int,
    ):
        super().__init__()
        self.initial_lr = initial_lr
        self.decay_start_step = float(decay_start_step)
        self.decay_end_step = float(decay_end_step)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        lr = tf.cond(
            step < self.decay_start_step,
            lambda: self.initial_lr,
            lambda: self.initial_lr * (
                1.0
                - (step - self.decay_start_step)
                / (self.decay_end_step - self.decay_start_step + 1e-8)
            ),
        )
        return tf.maximum(lr, 0.0)

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "decay_start_step": self.decay_start_step,
            "decay_end_step": self.decay_end_step,
        }


def build_lr_schedules(
    steps_per_epoch: int,
) -> tuple[LinearDecaySchedule, LinearDecaySchedule]:
    """Return ``(g_schedule, d_schedule)`` using config-defined boundaries."""
    g_sched = LinearDecaySchedule(
        config.LEARNING_RATE_G,
        config.LR_DECAY_START_EPOCH * steps_per_epoch,
        config.LR_DECAY_END_EPOCH * steps_per_epoch,
    )
    d_sched = LinearDecaySchedule(
        config.LEARNING_RATE_D,
        config.LR_DECAY_START_EPOCH * steps_per_epoch,
        config.LR_DECAY_END_EPOCH * steps_per_epoch,
    )
    return g_sched, d_sched


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def plot_losses(
    d_losses: list[float],
    g_losses: list[float],
    save_path: Optional[str] = None,
) -> None:
    """Plot generator and discriminator loss curves."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(d_losses, label="Discriminator", alpha=0.8)
    ax.plot(g_losses, label="Generator", alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def set_seed(seed: int = config.SEED) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
