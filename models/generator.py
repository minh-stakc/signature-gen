"""Generator network for the conditional GAN.

Architecture overview
---------------------
The generator follows a U-Net-inspired encoder-decoder layout with skip
connections and *conditional* batch normalisation (CBN).  The writer identity
and style vector are projected and injected into every CBN layer so that the
network can modulate features on a per-writer, per-style basis.

Input:  latent vector z (LATENT_DIM) + writer_id (int) + style (STYLE_DIM)
Output: grayscale image (IMG_HEIGHT x IMG_WIDTH x 1) in [-1, 1]
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config


class ConditionalBatchNorm(layers.Layer):
    """Conditional Batch Normalisation (de Vries et al., 2017).

    Learns per-sample affine parameters (gamma, beta) from a conditioning
    vector instead of using fixed learnable parameters.
    """

    def __init__(self, num_features: int, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.bn = layers.BatchNormalization(
            scale=False, center=False, momentum=0.9, epsilon=1e-5
        )

    def build(self, input_shape):
        # These dense layers will be connected to the condition vector at call time
        self.gamma_dense = layers.Dense(self.num_features, kernel_initializer="ones")
        self.beta_dense = layers.Dense(self.num_features, kernel_initializer="zeros")
        super().build(input_shape)

    def call(self, x, condition, training=False):
        """
        Parameters
        ----------
        x : tf.Tensor  -- (B, H, W, C)
        condition : tf.Tensor -- (B, cond_dim)
        """
        normalized = self.bn(x, training=training)
        gamma = self.gamma_dense(condition)  # (B, C)
        beta = self.beta_dense(condition)    # (B, C)
        # Reshape for broadcasting over spatial dims
        gamma = tf.reshape(gamma, [-1, 1, 1, self.num_features])
        beta = tf.reshape(beta, [-1, 1, 1, self.num_features])
        return normalized * (1.0 + gamma) + beta


class ResBlockUp(layers.Layer):
    """Residual block with 2x upsampling and conditional batch norm."""

    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.cbn1 = ConditionalBatchNorm(input_shape[-1])
        self.conv1 = layers.Conv2DTranspose(
            self.filters, 3, strides=2, padding="same",
            kernel_initializer="he_normal", use_bias=False,
        )
        self.cbn2 = ConditionalBatchNorm(self.filters)
        self.conv2 = layers.Conv2D(
            self.filters, 3, padding="same",
            kernel_initializer="he_normal", use_bias=False,
        )
        # Shortcut: upsample + 1x1 conv to match dimensions
        self.shortcut_up = layers.UpSampling2D(size=2, interpolation="bilinear")
        self.shortcut_conv = layers.Conv2D(
            self.filters, 1, padding="same", use_bias=False,
        )
        super().build(input_shape)

    def call(self, x, condition, training=False):
        h = tf.nn.relu(self.cbn1(x, condition, training=training))
        h = self.conv1(h)
        h = tf.nn.relu(self.cbn2(h, condition, training=training))
        h = self.conv2(h)
        shortcut = self.shortcut_conv(self.shortcut_up(x))
        return h + shortcut


class SelfAttention(layers.Layer):
    """Self-attention layer (Zhang et al., SAGAN)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        channels = input_shape[-1]
        self.query = layers.Conv2D(channels // 8, 1, use_bias=False)
        self.key = layers.Conv2D(channels // 8, 1, use_bias=False)
        self.value = layers.Conv2D(channels, 1, use_bias=False)
        self.gamma = self.add_weight("gamma", shape=[1], initializer="zeros")
        super().build(input_shape)

    def call(self, x):
        batch, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], x.shape[-1]
        q = tf.reshape(self.query(x), [batch, -1, c // 8])
        k = tf.reshape(self.key(x), [batch, -1, c // 8])
        v = tf.reshape(self.value(x), [batch, -1, c])

        attn = tf.nn.softmax(tf.matmul(q, k, transpose_b=True), axis=-1)
        out = tf.matmul(attn, v)
        out = tf.reshape(out, [batch, h, w, c])
        return x + self.gamma * out


class Generator(keras.Model):
    """Conditional generator network.

    Takes a latent vector *z*, an integer writer identity, and a continuous
    style vector.  Returns a single-channel image in [-1, 1].
    """

    def __init__(
        self,
        latent_dim: int = config.LATENT_DIM,
        num_writers: int = config.NUM_WRITERS,
        writer_embed_dim: int = config.WRITER_EMBED_DIM,
        style_dim: int = config.STYLE_DIM,
        base_filters: int = config.GEN_BASE_FILTERS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        cond_dim = writer_embed_dim + style_dim

        # Condition path
        self.writer_embedding = layers.Embedding(num_writers, writer_embed_dim)

        # Project & reshape: target (4, 16, base*8)
        self.init_h = 4
        self.init_w = 16
        self.init_filters = base_filters * 8
        self.fc = layers.Dense(
            self.init_h * self.init_w * self.init_filters,
            use_bias=False,
            kernel_initializer="he_normal",
        )

        # Condition projection (latent + cond -> combined)
        self.cond_project = layers.Dense(cond_dim, activation="relu")

        # Upsampling blocks: 4x16 -> 8x32 -> 16x64 -> 32x128 -> 64x256
        self.up1 = ResBlockUp(base_filters * 4)
        self.up2 = ResBlockUp(base_filters * 2)
        self.attn = SelfAttention()
        self.up3 = ResBlockUp(base_filters)
        self.up4 = ResBlockUp(base_filters // 2)

        self.final_bn = layers.BatchNormalization(momentum=0.9)
        self.final_conv = layers.Conv2D(
            1, 3, padding="same", activation="tanh",
            kernel_initializer="glorot_normal",
        )

    def call(self, inputs, training=False):
        """
        Parameters
        ----------
        inputs : tuple
            (z, writer_id, style) where
            - z : (B, LATENT_DIM)
            - writer_id : (B,)  int
            - style : (B, STYLE_DIM)
        """
        z, writer_id, style = inputs

        # Build condition vector
        w_emb = self.writer_embedding(writer_id)  # (B, embed_dim)
        cond = tf.concat([w_emb, style], axis=-1)
        cond = self.cond_project(cond)

        # Concatenate z and condition
        h = tf.concat([z, cond], axis=-1)
        h = self.fc(h)
        h = tf.reshape(h, [-1, self.init_h, self.init_w, self.init_filters])

        h = self.up1(h, cond, training=training)   # 8x32
        h = self.up2(h, cond, training=training)    # 16x64
        h = self.attn(h)
        h = self.up3(h, cond, training=training)    # 32x128
        h = self.up4(h, cond, training=training)    # 64x256

        h = tf.nn.relu(self.final_bn(h, training=training))
        return self.final_conv(h)


def build_generator(**kwargs) -> Generator:
    """Factory function that returns an initialised Generator."""
    gen = Generator(**kwargs)
    # Build with dummy data
    z = tf.zeros((1, config.LATENT_DIM))
    wid = tf.zeros((1,), dtype=tf.int32)
    style = tf.zeros((1, config.STYLE_DIM))
    gen((z, wid, style), training=False)
    return gen
