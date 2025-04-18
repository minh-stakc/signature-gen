"""Training script for the Signature Generator cGAN.

Usage
-----
    python train.py --epochs 200 --batch_size 32

If no real data exists under ``config.DATA_DIR``, the script falls back to
synthetic placeholder data so the full pipeline can be validated end-to-end.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import tensorflow as tf

import config
from utils import (
    ensure_dir,
    save_image_grid,
    denormalize,
    build_lr_schedules,
    plot_losses,
    set_seed,
)
from data.dataset import SignatureDataset, build_dataset
from models.cgan import ConditionalGAN, build_cgan


def _build_synthetic_dataset(
    num_samples: int = 500,
    batch_size: int = config.BATCH_SIZE,
) -> tf.data.Dataset:
    """Create a synthetic dataset of random images for pipeline validation."""
    images = np.random.uniform(
        -1, 1,
        (num_samples, config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
    ).astype(np.float32)
    labels = np.random.randint(0, config.NUM_WRITERS, num_samples).astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.shuffle(num_samples).repeat().batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


class SampleCallback(tf.keras.callbacks.Callback):
    """Generate and save a grid of samples at the end of every N-th epoch."""

    def __init__(self, cgan: ConditionalGAN, every_n: int = 5, output_dir: str = config.OUTPUT_DIR):
        super().__init__()
        self.cgan = cgan
        self.every_n = every_n
        self.output_dir = output_dir
        # Fixed noise for consistent visual comparison
        self.fixed_z = tf.random.normal((16, config.LATENT_DIM))
        self.fixed_style = tf.random.normal((16, config.STYLE_DIM))
        self.fixed_wids = tf.constant([i % config.NUM_WRITERS for i in range(16)], dtype=tf.int32)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n != 0:
            return
        images = self.cgan.generator(
            (self.fixed_z, self.fixed_wids, self.fixed_style), training=False
        )
        images = denormalize(images)
        path = os.path.join(self.output_dir, "samples", f"epoch_{epoch + 1:04d}.png")
        save_image_grid(images, path, nrow=4)
        print(f"  [sample] Saved grid -> {path}")


class LossPlotCallback(tf.keras.callbacks.Callback):
    """Accumulate per-epoch losses and save a plot."""

    def __init__(self, output_dir: str = config.OUTPUT_DIR):
        super().__init__()
        self.output_dir = output_dir
        self.d_losses: list[float] = []
        self.g_losses: list[float] = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.d_losses.append(float(logs.get("d_loss", 0)))
        self.g_losses.append(float(logs.get("g_loss", 0)))
        plot_losses(
            self.d_losses,
            self.g_losses,
            os.path.join(self.output_dir, "loss_curves.png"),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Signature cGAN")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--data_dir", type=str, default=config.DATA_DIR)
    parser.add_argument("--checkpoint_dir", type=str, default=config.CHECKPOINT_DIR)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()

    ensure_dir(args.checkpoint_dir)
    ensure_dir(config.OUTPUT_DIR)
    ensure_dir(config.LOG_DIR)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    sig_ds = SignatureDataset(data_dir=args.data_dir, augment=True)
    if len(sig_ds) > 0:
        print(f"Found {len(sig_ds)} images from {len(set(sig_ds.labels))} writers.")
        dataset = sig_ds.build(batch_size=args.batch_size)
        steps_per_epoch = max(len(sig_ds) // args.batch_size, 1)
    else:
        print("No real data found -- using synthetic placeholder data for validation.")
        dataset = _build_synthetic_dataset(batch_size=args.batch_size)
        steps_per_epoch = 500 // args.batch_size

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    g_sched, d_sched = build_lr_schedules(steps_per_epoch)

    cgan = ConditionalGAN()
    cgan.compile(
        g_optimizer=tf.keras.optimizers.Adam(g_sched, beta_1=config.BETA_1, beta_2=config.BETA_2),
        d_optimizer=tf.keras.optimizers.Adam(d_sched, beta_1=config.BETA_1, beta_2=config.BETA_2),
    )

    # ------------------------------------------------------------------
    # Checkpoint restoration
    # ------------------------------------------------------------------
    ckpt = tf.train.Checkpoint(
        generator=cgan.generator,
        discriminator=cgan.discriminator,
        g_optimizer=cgan.g_optimizer,
        d_optimizer=cgan.d_optimizer,
    )
    ckpt_manager = tf.train.CheckpointManager(ckpt, args.checkpoint_dir, max_to_keep=5)
    initial_epoch = 0

    if args.resume and ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        # Infer epoch from checkpoint name (ckpt-<number>)
        initial_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1])
        print(f"Restored from {ckpt_manager.latest_checkpoint} (epoch {initial_epoch})")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    callbacks = [
        SampleCallback(cgan, every_n=5),
        LossPlotCallback(),
        tf.keras.callbacks.TensorBoard(log_dir=config.LOG_DIR, update_freq="epoch"),
    ]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    print(f"\nStarting training for {args.epochs} epochs "
          f"(steps/epoch={steps_per_epoch}, batch={args.batch_size})")
    t0 = time.time()

    cgan.fit(
        dataset,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed / 60:.1f} minutes.")

    # Save final checkpoint
    ckpt_manager.save()
    print(f"Final checkpoint saved to {args.checkpoint_dir}")

    # Save generator weights separately for easy loading during generation
    gen_path = os.path.join(args.checkpoint_dir, "generator_final.weights.h5")
    cgan.generator.save_weights(gen_path)
    print(f"Generator weights saved to {gen_path}")


if __name__ == "__main__":
    main()
