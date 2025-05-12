"""Generate synthetic signatures from a trained cGAN.

Usage
-----
    python generate.py --writer_id 5 --num_samples 10 --output_dir outputs/generated

Supports:
- Single writer or batch of writers
- Explicit style vectors or random sampling
- Optional style interpolation between two endpoints
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import tensorflow as tf

import config
from models.generator import build_generator
from models.cgan import ConditionalGAN
from utils import ensure_dir, denormalize, save_image_grid, set_seed


def load_model(checkpoint_dir: str) -> ConditionalGAN:
    """Load a ConditionalGAN from the latest checkpoint."""
    cgan = ConditionalGAN()
    ckpt = tf.train.Checkpoint(
        generator=cgan.generator, discriminator=cgan.discriminator
    )
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest:
        ckpt.restore(latest).expect_partial()
        print(f"Loaded checkpoint: {latest}")
    else:
        # Try loading standalone generator weights
        gen_path = os.path.join(checkpoint_dir, "generator_final.weights.h5")
        if os.path.exists(gen_path):
            cgan.generator.load_weights(gen_path)
            print(f"Loaded generator weights: {gen_path}")
        else:
            print("WARNING: No checkpoint found -- generating with random weights.")
    return cgan


def generate_samples(
    cgan: ConditionalGAN,
    writer_ids: np.ndarray,
    num_samples: int,
    style: np.ndarray | None = None,
) -> np.ndarray:
    """Generate *num_samples* images and return uint8 array."""
    return cgan.generate(writer_ids, num_samples=num_samples, style=style)


def interpolate_styles(
    cgan: ConditionalGAN,
    writer_id: int,
    steps: int = 10,
) -> np.ndarray:
    """Linearly interpolate between two random style endpoints.

    Returns an array of shape ``(steps, H, W, 1)`` uint8.
    """
    style_a = np.random.randn(1, config.STYLE_DIM).astype(np.float32)
    style_b = np.random.randn(1, config.STYLE_DIM).astype(np.float32)
    alphas = np.linspace(0, 1, steps).reshape(-1, 1)
    styles = style_a * (1 - alphas) + style_b * alphas  # (steps, STYLE_DIM)
    wids = np.full(steps, writer_id, dtype=np.int32)

    z = tf.random.normal((steps, config.LATENT_DIM))
    images = cgan.generator(
        (z, tf.constant(wids), tf.constant(styles, dtype=tf.float32)),
        training=False,
    )
    return denormalize(images)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic signatures")
    p.add_argument("--writer_id", type=int, default=0, help="Writer identity index")
    p.add_argument("--num_samples", type=int, default=16, help="Number of signatures to generate")
    p.add_argument("--checkpoint_dir", type=str, default=config.CHECKPOINT_DIR)
    p.add_argument("--output_dir", type=str, default=os.path.join(config.OUTPUT_DIR, "generated"))
    p.add_argument("--interpolate", action="store_true", help="Generate a style interpolation strip")
    p.add_argument("--interp_steps", type=int, default=10, help="Interpolation steps")
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    cgan = load_model(args.checkpoint_dir)

    if args.interpolate:
        print(f"Generating style interpolation for writer {args.writer_id} "
              f"({args.interp_steps} steps) ...")
        images = interpolate_styles(cgan, args.writer_id, steps=args.interp_steps)
        path = os.path.join(args.output_dir, f"interp_w{args.writer_id}.png")
        save_image_grid(images, path, nrow=args.interp_steps)
        print(f"Saved interpolation strip -> {path}")
    else:
        print(f"Generating {args.num_samples} signatures for writer {args.writer_id} ...")
        wids = np.full(args.num_samples, args.writer_id, dtype=np.int32)
        images = generate_samples(cgan, wids, args.num_samples)

        # Save grid
        grid_path = os.path.join(args.output_dir, f"grid_w{args.writer_id}.png")
        save_image_grid(images, grid_path, nrow=min(8, args.num_samples))
        print(f"Saved grid -> {grid_path}")

        # Save individual images
        ind_dir = ensure_dir(os.path.join(args.output_dir, f"writer_{args.writer_id}"))
        for i, img in enumerate(images):
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img[..., 0]
            import cv2
            cv2.imwrite(os.path.join(ind_dir, f"sig_{i:04d}.png"), img)
        print(f"Saved {len(images)} individual images -> {ind_dir}")


if __name__ == "__main__":
    main()
