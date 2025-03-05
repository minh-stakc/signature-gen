"""Central configuration for the Signature Generator project.

All hyperparameters, paths, and training settings live here so that every
module imports a single source of truth.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "signatures")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# ---------------------------------------------------------------------------
# Image settings
# ---------------------------------------------------------------------------
IMG_HEIGHT = 64
IMG_WIDTH = 256
IMG_CHANNELS = 1  # grayscale signatures

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
LATENT_DIM = 128
NUM_WRITERS = 50          # number of distinct writer identities
WRITER_EMBED_DIM = 64     # dimension of the writer-identity embedding
STYLE_DIM = 16            # continuous style vector dimension

# Generator
GEN_BASE_FILTERS = 64
GEN_NUM_DOWNSAMPLE = 3
GEN_NUM_RESBLOCKS = 6

# Discriminator
DISC_BASE_FILTERS = 64
DISC_NUM_LAYERS = 4       # PatchGAN depth

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE_G = 2e-4
LEARNING_RATE_D = 1e-4
BETA_1 = 0.0
BETA_2 = 0.9

# Loss weights
LAMBDA_ADV = 1.0
LAMBDA_L1 = 10.0          # pixel-level reconstruction
LAMBDA_GP = 10.0           # gradient penalty coefficient

# Learning-rate schedule
LR_DECAY_START_EPOCH = 100
LR_DECAY_END_EPOCH = 200

# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
AUG_ROTATION_RANGE = 5.0          # degrees
AUG_SCALE_RANGE = (0.9, 1.1)
AUG_ELASTIC_ALPHA = 20.0
AUG_ELASTIC_SIGMA = 3.0
AUG_SHEAR_RANGE = 3.0             # degrees
AUG_BRIGHTNESS_DELTA = 0.1
AUG_MORPHO_KERNEL_RANGE = (2, 4)  # erosion / dilation kernel sizes

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
FID_NUM_SAMPLES = 1000
SSIM_NUM_SAMPLES = 200
HUMAN_EVAL_NUM_SAMPLES = 50

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
SEED = 42
NUM_WORKERS = 4
