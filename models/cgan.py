"""Combined Conditional GAN (cGAN) model with a full training loop.

Implements WGAN-GP (Gulrajani et al., 2017) loss with a projection-conditioned
discriminator and conditional batch-norm generator.  The training loop supports:

- Gradient penalty for Lipschitz constraint
- Separate optimiser schedules for G and D
- Periodic image sampling and metric logging
- Checkpoint saving / resumption
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

import config
from models.generator import Generator, build_generator
from models.discriminator import Discriminator, build_discriminator


class ConditionalGAN(keras.Model):
    """Full cGAN wrapper that owns the generator, discriminator, and
    training step logic."""

    def __init__(
        self,
        generator: Optional[Generator] = None,
        discriminator: Optional[Discriminator] = None,
        latent_dim: int = config.LATENT_DIM,
        style_dim: int = config.STYLE_DIM,
        lambda_gp: float = config.LAMBDA_GP,
        lambda_l1: float = config.LAMBDA_L1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.generator = generator or build_generator()
        self.discriminator = discriminator or build_discriminator()
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.lambda_gp = lambda_gp
        self.lambda_l1 = lambda_l1

    def compile(
        self,
        g_optimizer: Optional[keras.optimizers.Optimizer] = None,
        d_optimizer: Optional[keras.optimizers.Optimizer] = None,
        **kwargs,
    ):
        super().compile(**kwargs)
        self.g_optimizer = g_optimizer or keras.optimizers.Adam(
            config.LEARNING_RATE_G, beta_1=config.BETA_1, beta_2=config.BETA_2,
        )
        self.d_optimizer = d_optimizer or keras.optimizers.Adam(
            config.LEARNING_RATE_D, beta_1=config.BETA_1, beta_2=config.BETA_2,
        )

        # Metrics
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        self.gp_metric = keras.metrics.Mean(name="gradient_penalty")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric, self.gp_metric]

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_noise(self, batch_size: int):
        z = tf.random.normal((batch_size, self.latent_dim))
        style = tf.random.normal((batch_size, self.style_dim))
        return z, style

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    @staticmethod
    def _wasserstein_d_loss(real_logits: tf.Tensor, fake_logits: tf.Tensor):
        return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

    @staticmethod
    def _wasserstein_g_loss(fake_logits: tf.Tensor):
        return -tf.reduce_mean(fake_logits)

    def _gradient_penalty(
        self,
        real_images: tf.Tensor,
        fake_images: tf.Tensor,
        writer_ids: tf.Tensor,
    ) -> tf.Tensor:
        """Compute the gradient penalty (GP) for WGAN-GP."""
        batch_size = tf.shape(real_images)[0]
        alpha = tf.random.uniform((batch_size, 1, 1, 1), 0.0, 1.0)
        interpolated = alpha * real_images + (1.0 - alpha) * fake_images

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator((interpolated, writer_ids), training=True)

        grads = gp_tape.gradient(pred, interpolated)
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-8)
        return tf.reduce_mean(tf.square(grad_norm - 1.0))

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, data):
        """Single training step over one batch.

        Parameters
        ----------
        data : tuple
            ``(real_images, writer_ids)`` coming from the ``tf.data`` pipeline.
        """
        real_images, writer_ids = data
        batch_size = tf.shape(real_images)[0]
        z, style = self._sample_noise(batch_size)

        # ---- Discriminator step ----
        with tf.GradientTape() as d_tape:
            fake_images = self.generator(
                (z, writer_ids, style), training=True
            )
            real_logits = self.discriminator(
                (real_images, writer_ids), training=True
            )
            fake_logits = self.discriminator(
                (fake_images, writer_ids), training=True
            )
            d_loss = self._wasserstein_d_loss(real_logits, fake_logits)
            gp = self._gradient_penalty(real_images, fake_images, writer_ids)
            d_total = d_loss + self.lambda_gp * gp

        d_grads = d_tape.gradient(d_total, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_variables)
        )

        # ---- Generator step ----
        z, style = self._sample_noise(batch_size)
        with tf.GradientTape() as g_tape:
            fake_images = self.generator(
                (z, writer_ids, style), training=True
            )
            fake_logits = self.discriminator(
                (fake_images, writer_ids), training=True
            )
            g_adv = self._wasserstein_g_loss(fake_logits)

            # Optional L1 reconstruction term (only useful if paired data available)
            g_l1 = tf.reduce_mean(tf.abs(fake_images - real_images))
            g_total = config.LAMBDA_ADV * g_adv + self.lambda_l1 * g_l1

        g_grads = g_tape.gradient(g_total, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_grads, self.generator.trainable_variables)
        )

        # ---- Metrics ----
        self.d_loss_metric.update_state(d_total)
        self.g_loss_metric.update_state(g_total)
        self.gp_metric.update_state(gp)
        return {m.name: m.result() for m in self.metrics}

    # ------------------------------------------------------------------
    # Generation helper
    # ------------------------------------------------------------------

    def generate(
        self,
        writer_ids: np.ndarray,
        num_samples: int = 1,
        style: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate synthetic signatures.

        Parameters
        ----------
        writer_ids : array-like of int, shape (num_samples,)
        num_samples : int
        style : optional array (num_samples, STYLE_DIM)

        Returns
        -------
        np.ndarray  shape (num_samples, H, W, 1) in [0, 255] uint8
        """
        z = tf.random.normal((num_samples, self.latent_dim))
        if style is None:
            style = tf.random.normal((num_samples, self.style_dim))
        else:
            style = tf.constant(style, dtype=tf.float32)

        writer_ids = tf.constant(writer_ids, dtype=tf.int32)
        images = self.generator((z, writer_ids, style), training=False)
        images = ((images + 1.0) / 2.0 * 255.0).numpy().astype(np.uint8)
        return images


def build_cgan(**kwargs) -> ConditionalGAN:
    """Factory: build and compile a ConditionalGAN ready for training."""
    cgan = ConditionalGAN(**kwargs)
    cgan.compile()
    return cgan
