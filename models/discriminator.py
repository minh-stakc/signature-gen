"""Discriminator network for the conditional GAN.

Architecture overview
---------------------
A PatchGAN discriminator with *projection-based* conditioning (Miyato &
Koyama, 2018).  Instead of naively concatenating the label to the input,
the writer embedding is projected into the final feature space and combined
via an inner product, which gives a much stronger conditioning signal.

The discriminator also uses spectral normalisation on every convolutional
layer for training stability.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config


class SpectralNormalization(layers.Wrapper):
    """Spectral normalisation wrapper for a Dense or Conv2D layer."""

    def __init__(self, layer: layers.Layer, power_iterations: int = 1, **kwargs):
        super().__init__(layer, **kwargs)
        self.power_iterations = power_iterations

    def build(self, input_shape):
        super().build(input_shape)
        kernel = self.layer.kernel
        self.u = self.add_weight(
            "sn_u",
            shape=(1, kernel.shape[-1]),
            initializer="truncated_normal",
            trainable=False,
        )

    def call(self, inputs, **kwargs):
        kernel = self.layer.kernel
        k_shape = kernel.shape
        k_2d = tf.reshape(kernel, [-1, k_shape[-1]])

        u_hat = self.u
        for _ in range(self.power_iterations):
            v_hat = tf.nn.l2_normalize(tf.matmul(u_hat, k_2d, transpose_b=True))
            u_hat = tf.nn.l2_normalize(tf.matmul(v_hat, k_2d))

        sigma = tf.matmul(tf.matmul(v_hat, k_2d), u_hat, transpose_b=True)
        self.u.assign(u_hat)
        self.layer.kernel.assign(kernel / sigma)
        return self.layer(inputs, **kwargs)


def _sn_conv(filters, kernel_size, strides=1, padding="same"):
    """Convenience: spectrally-normalised Conv2D."""
    return SpectralNormalization(
        layers.Conv2D(
            filters, kernel_size, strides=strides, padding=padding,
            kernel_initializer="he_normal", use_bias=False,
        )
    )


class DiscriminatorBlock(layers.Layer):
    """Down-sampling block: SN-Conv -> LeakyReLU -> SN-Conv -> AvgPool."""

    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.conv1 = _sn_conv(self.filters, 3)
        self.conv2 = _sn_conv(self.filters, 3)
        self.shortcut = _sn_conv(self.filters, 1)
        self.pool = layers.AveragePooling2D(2)
        super().build(input_shape)

    def call(self, x):
        h = tf.nn.leaky_relu(self.conv1(x), alpha=0.2)
        h = tf.nn.leaky_relu(self.conv2(h), alpha=0.2)
        h = self.pool(h)
        shortcut = self.pool(self.shortcut(x))
        return h + shortcut


class Discriminator(keras.Model):
    """Projection-conditioned PatchGAN discriminator.

    Returns a scalar ``(B, 1)`` realness score per sample.
    """

    def __init__(
        self,
        num_writers: int = config.NUM_WRITERS,
        writer_embed_dim: int = config.WRITER_EMBED_DIM,
        base_filters: int = config.DISC_BASE_FILTERS,
        num_layers: int = config.DISC_NUM_LAYERS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers

        # Convolutional backbone
        self.blocks = []
        filters = base_filters
        for i in range(num_layers):
            self.blocks.append(DiscriminatorBlock(filters, name=f"d_block_{i}"))
            filters = min(filters * 2, base_filters * 8)

        self.activation = layers.LeakyReLU(0.2)
        self.global_pool = layers.GlobalAveragePooling2D()

        # Linear output (unconditional part)
        self.fc = SpectralNormalization(layers.Dense(1))

        # Projection conditioning
        self.writer_embedding = layers.Embedding(num_writers, filters // 2)
        self.embed_dense = SpectralNormalization(
            layers.Dense(filters // 2, use_bias=False)
        )

    def call(self, inputs, training=False):
        """
        Parameters
        ----------
        inputs : tuple
            (image, writer_id) where
            - image : (B, H, W, 1)  in [-1, 1]
            - writer_id : (B,) int
        """
        image, writer_id = inputs

        h = image
        for block in self.blocks:
            h = block(h)

        h = self.activation(h)
        h = self.global_pool(h)  # (B, C)

        # Unconditional output
        out = self.fc(h)  # (B, 1)

        # Projection conditioning: inner product with writer embedding
        w_emb = self.writer_embedding(writer_id)  # (B, embed_dim)
        h_proj = self.embed_dense(h)               # (B, embed_dim)
        proj = tf.reduce_sum(h_proj * w_emb, axis=-1, keepdims=True)  # (B, 1)

        return out + proj


def build_discriminator(**kwargs) -> Discriminator:
    """Factory function that returns an initialised Discriminator."""
    disc = Discriminator(**kwargs)
    dummy_img = tf.zeros((1, config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS))
    dummy_wid = tf.zeros((1,), dtype=tf.int32)
    disc((dummy_img, dummy_wid), training=False)
    return disc
