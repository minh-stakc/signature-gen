"""Signature dataset loader built on top of ``tf.data``.

Expected directory layout::

    data/signatures/
        0/          # writer id 0
            001.png
            002.png
            ...
        1/          # writer id 1
            ...

Each image is loaded as grayscale, resized, normalised to [-1, 1] and
optionally augmented via :class:`augmentation.AugmentationPipeline`.
"""

from __future__ import annotations

import os
import pathlib
from typing import Optional

import numpy as np
import tensorflow as tf

import config
from data.augmentation import AugmentationPipeline


class SignatureDataset:
    """Thin wrapper that builds a ``tf.data.Dataset`` from a directory tree.

    Attributes
    ----------
    file_paths : list[str]
        Absolute paths to every image discovered.
    labels : list[int]
        Corresponding integer writer-identity labels.
    """

    def __init__(
        self,
        data_dir: str = config.DATA_DIR,
        img_height: int = config.IMG_HEIGHT,
        img_width: int = config.IMG_WIDTH,
        augment: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.img_height = img_height
        self.img_width = img_width
        self.augment = augment
        self.augmenter = AugmentationPipeline() if augment else None

        self.file_paths: list[str] = []
        self.labels: list[int] = []
        self._scan_directory()

    # ------------------------------------------------------------------
    # Directory scanning
    # ------------------------------------------------------------------

    def _scan_directory(self) -> None:
        """Walk ``data_dir`` and collect (path, writer_id) pairs."""
        if not os.path.isdir(self.data_dir):
            return
        for writer_dir in sorted(pathlib.Path(self.data_dir).iterdir()):
            if not writer_dir.is_dir():
                continue
            try:
                writer_id = int(writer_dir.name)
            except ValueError:
                continue
            for img_path in sorted(writer_dir.glob("*")):
                if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif"):
                    self.file_paths.append(str(img_path))
                    self.labels.append(writer_id)

    # ------------------------------------------------------------------
    # tf.data helpers
    # ------------------------------------------------------------------

    def _load_and_preprocess(
        self, path: tf.Tensor, label: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Read, decode, resize, and normalise a single image."""
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=1, expand_animations=False)
        img = tf.image.resize(img, [self.img_height, self.img_width])
        img = tf.cast(img, tf.float32) / 255.0  # [0, 1]
        return img, label

    def _augment_py(
        self, img: tf.Tensor, label: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Apply the numpy augmentation pipeline inside ``tf.py_function``."""

        def _aug(image: np.ndarray) -> np.ndarray:
            assert self.augmenter is not None
            return self.augmenter(image)

        aug_img = tf.py_function(_aug, [img], tf.float32)
        aug_img.set_shape([self.img_height, self.img_width, 1])
        return aug_img, label

    @staticmethod
    def _normalise(
        img: tf.Tensor, label: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Scale from [0, 1] to [-1, 1] (tanh output range)."""
        return img * 2.0 - 1.0, label

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        batch_size: int = config.BATCH_SIZE,
        shuffle: bool = True,
        repeat: bool = True,
    ) -> tf.data.Dataset:
        """Return a batched, prefetched ``tf.data.Dataset``.

        Each element is a tuple ``(images, labels)`` where images have shape
        ``(B, H, W, 1)`` in ``[-1, 1]`` and labels are int32 scalars.
        """
        ds = tf.data.Dataset.from_tensor_slices(
            (self.file_paths, self.labels)
        )

        if shuffle:
            ds = ds.shuffle(buffer_size=max(len(self.file_paths), 1), seed=config.SEED)

        ds = ds.map(self._load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

        if self.augment and self.augmenter is not None:
            ds = ds.map(self._augment_py, num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.map(self._normalise, num_parallel_calls=tf.data.AUTOTUNE)

        if repeat:
            ds = ds.repeat()

        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def __len__(self) -> int:
        return len(self.file_paths)


def build_dataset(
    data_dir: Optional[str] = None,
    batch_size: int = config.BATCH_SIZE,
    augment: bool = True,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Convenience factory that returns a ready-to-iterate dataset."""
    data_dir = data_dir or config.DATA_DIR
    loader = SignatureDataset(data_dir=data_dir, augment=augment)
    return loader.build(batch_size=batch_size, shuffle=shuffle)
