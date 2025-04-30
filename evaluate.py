"""Evaluation script: FID, SSIM, and human-in-the-loop assessment.

Usage
-----
    python evaluate.py --checkpoint_dir checkpoints/ --num_samples 1000

Metrics implemented
-------------------
- **FID** (Frechet Inception Distance): measures distributional similarity
  between real and generated image feature embeddings.  Lower is better.
- **SSIM** (Structural Similarity Index): measures structural fidelity of
  individual generated images against their nearest real neighbours.
- **Human-in-the-loop assessment**: generates a labelled batch and writes an
  HTML report that a human reviewer can score.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
from scipy import linalg
from skimage.metrics import structural_similarity as compare_ssim

import config
from models.generator import build_generator
from models.discriminator import build_discriminator
from models.cgan import ConditionalGAN
from data.dataset import SignatureDataset
from utils import ensure_dir, denormalize, set_seed


# ======================================================================
# Feature extraction (lightweight -- no Inception dependency)
# ======================================================================

def _build_feature_extractor() -> tf.keras.Model:
    """Build a lightweight CNN feature extractor for FID computation.

    For production use, replace this with an InceptionV3 feature extractor.
    This version uses a small trainable-free random CNN so the pipeline runs
    without downloading external weights.
    """
    inp = tf.keras.Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS))
    # Upsample to 3-channel 75x75 (minimum for a reasonable feature space)
    x = tf.keras.layers.Resizing(75, 75)(inp)
    x = tf.keras.layers.Concatenate()([x, x, x])  # 1-ch -> 3-ch

    for filters in [32, 64, 128, 256]:
        x = tf.keras.layers.Conv2D(filters, 3, strides=2, padding="same")(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.Model(inp, x, name="feature_extractor")
    return model


# ======================================================================
# FID
# ======================================================================

def _compute_statistics(
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of a feature matrix (N, D)."""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def compute_fid(
    real_features: np.ndarray, fake_features: np.ndarray
) -> float:
    """Frechet Inception Distance between two feature sets.

    FID = ||mu_r - mu_f||^2 + Tr(Sigma_r + Sigma_f - 2*sqrt(Sigma_r @ Sigma_f))
    """
    mu_r, sigma_r = _compute_statistics(real_features)
    mu_f, sigma_f = _compute_statistics(fake_features)

    diff = mu_r - mu_f
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)

    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r + sigma_f - 2.0 * covmean)
    return float(fid)


# ======================================================================
# SSIM
# ======================================================================

def compute_mean_ssim(
    real_images: np.ndarray, fake_images: np.ndarray
) -> float:
    """Mean SSIM between each fake image and its nearest real neighbour.

    Images should be uint8, shape (N, H, W) or (N, H, W, 1).
    """
    if real_images.ndim == 4:
        real_images = real_images[..., 0]
    if fake_images.ndim == 4:
        fake_images = fake_images[..., 0]

    ssim_scores = []
    for fake in fake_images:
        best = -1.0
        # Compare against a random subset for efficiency
        indices = np.random.choice(len(real_images), min(50, len(real_images)), replace=False)
        for idx in indices:
            score = compare_ssim(real_images[idx], fake, data_range=255)
            best = max(best, score)
        ssim_scores.append(best)

    return float(np.mean(ssim_scores))


# Backward-compatible alias
compute_ssim = compute_mean_ssim


# ======================================================================
# Human-in-the-loop HTML report
# ======================================================================

def generate_human_eval_report(
    fake_images: np.ndarray,
    writer_ids: np.ndarray,
    output_dir: str,
) -> str:
    """Save generated images and an HTML page for manual scoring.

    Returns the path to the HTML report.
    """
    report_dir = ensure_dir(os.path.join(output_dir, "human_eval"))
    img_dir = ensure_dir(os.path.join(report_dir, "images"))

    # Save individual images
    entries = []
    for i, (img, wid) in enumerate(zip(fake_images, writer_ids)):
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]
        fname = f"sample_{i:04d}_w{wid}.png"
        cv2.imwrite(os.path.join(img_dir, fname), img)
        entries.append({"index": i, "writer_id": int(wid), "file": fname})

    # Build HTML
    html_lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Signature GAN -- Human Evaluation</title>",
        "<style>",
        "body{font-family:sans-serif;margin:2em}",
        "table{border-collapse:collapse}",
        "td,th{border:1px solid #ccc;padding:8px;text-align:center}",
        "img{max-height:80px}",
        "select{font-size:1em}",
        "</style></head><body>",
        "<h1>Human Evaluation: Signature Quality</h1>",
        "<p>Rate each generated signature on a 1-5 scale "
        "(1=clearly fake, 5=indistinguishable from real).</p>",
        '<form id="evalForm">',
        "<table><tr><th>#</th><th>Writer</th><th>Image</th><th>Score</th></tr>",
    ]
    for e in entries:
        html_lines.append(
            f"<tr><td>{e['index']}</td><td>{e['writer_id']}</td>"
            f"<td><img src='images/{e['file']}'></td>"
            f"<td><select name='score_{e['index']}'>"
            "<option value=''>--</option>"
            "<option>1</option><option>2</option><option>3</option>"
            "<option>4</option><option>5</option>"
            "</select></td></tr>"
        )
    html_lines += [
        "</table><br>",
        "<button type='button' onclick='exportScores()'>Export Scores (JSON)</button>",
        "<pre id='output'></pre>",
        "<script>",
        "function exportScores(){",
        "  const fd=new FormData(document.getElementById('evalForm'));",
        "  const obj={};fd.forEach((v,k)=>{obj[k]=v});",
        "  document.getElementById('output').textContent=JSON.stringify(obj,null,2);",
        "}",
        "</script></form></body></html>",
    ]

    html_path = os.path.join(report_dir, "eval_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))

    # Also dump metadata
    meta_path = os.path.join(report_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(entries, f, indent=2)

    return html_path


# ======================================================================
# Main evaluation routine
# ======================================================================

def load_real_images(
    data_dir: str, num: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load *num* real images and their labels as uint8 arrays."""
    ds = SignatureDataset(data_dir=data_dir, augment=False)
    if len(ds) == 0:
        # Synthetic fallback
        imgs = np.random.randint(0, 256, (num, config.IMG_HEIGHT, config.IMG_WIDTH, 1), dtype=np.uint8)
        lbls = np.random.randint(0, config.NUM_WRITERS, num, dtype=np.int32)
        return imgs, lbls

    tf_ds = ds.build(batch_size=num, shuffle=True, repeat=False)
    for batch_images, batch_labels in tf_ds.take(1):
        images = denormalize(batch_images)
        labels = batch_labels.numpy().astype(np.int32)
    return images, labels


def evaluate(
    checkpoint_dir: str,
    data_dir: str = config.DATA_DIR,
    num_fid_samples: int = config.FID_NUM_SAMPLES,
    num_ssim_samples: int = config.SSIM_NUM_SAMPLES,
    num_human_samples: int = config.HUMAN_EVAL_NUM_SAMPLES,
    output_dir: str = config.OUTPUT_DIR,
) -> dict:
    """Run the full evaluation suite and return a results dict."""
    set_seed()

    # Load model
    cgan = ConditionalGAN()
    ckpt = tf.train.Checkpoint(
        generator=cgan.generator, discriminator=cgan.discriminator
    )
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest:
        ckpt.restore(latest).expect_partial()
        print(f"Restored checkpoint: {latest}")
    else:
        print("WARNING: No checkpoint found -- evaluating with random weights.")

    feature_extractor = _build_feature_extractor()
    results: dict = {}

    # ---- FID ----
    print(f"\nComputing FID over {num_fid_samples} samples ...")
    real_imgs, real_lbls = load_real_images(data_dir, num_fid_samples)
    writer_ids = np.random.randint(0, config.NUM_WRITERS, num_fid_samples, dtype=np.int32)
    fake_imgs = cgan.generate(writer_ids, num_samples=num_fid_samples)

    real_normed = real_imgs.astype(np.float32) / 127.5 - 1.0
    fake_normed = fake_imgs.astype(np.float32) / 127.5 - 1.0

    real_feats = feature_extractor.predict(real_normed, batch_size=64, verbose=0)
    fake_feats = feature_extractor.predict(fake_normed, batch_size=64, verbose=0)

    fid = compute_fid(real_feats, fake_feats)
    results["fid"] = fid
    print(f"  FID = {fid:.2f}")

    # ---- SSIM ----
    print(f"\nComputing mean SSIM over {num_ssim_samples} samples ...")
    ssim_fake = fake_imgs[:num_ssim_samples]
    ssim_real = real_imgs[:num_ssim_samples]
    mean_ssim = compute_mean_ssim(ssim_real, ssim_fake)
    results["mean_ssim"] = mean_ssim
    print(f"  Mean SSIM = {mean_ssim:.4f}")

    # ---- Human evaluation report ----
    print(f"\nGenerating human evaluation report ({num_human_samples} samples) ...")
    human_wids = np.random.randint(0, config.NUM_WRITERS, num_human_samples, dtype=np.int32)
    human_imgs = cgan.generate(human_wids, num_samples=num_human_samples)
    html_path = generate_human_eval_report(human_imgs, human_wids, output_dir)
    results["human_eval_report"] = html_path
    print(f"  Report saved -> {html_path}")

    # ---- Save results ----
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump({k: (v if isinstance(v, str) else float(v)) for k, v in results.items()}, f, indent=2)
    print(f"\nAll results saved to {results_path}")

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the Signature cGAN")
    p.add_argument("--checkpoint_dir", type=str, default=config.CHECKPOINT_DIR)
    p.add_argument("--data_dir", type=str, default=config.DATA_DIR)
    p.add_argument("--num_fid_samples", type=int, default=config.FID_NUM_SAMPLES)
    p.add_argument("--num_ssim_samples", type=int, default=config.SSIM_NUM_SAMPLES)
    p.add_argument("--output_dir", type=str, default=config.OUTPUT_DIR)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        num_fid_samples=args.num_fid_samples,
        num_ssim_samples=args.num_ssim_samples,
        output_dir=args.output_dir,
    )
