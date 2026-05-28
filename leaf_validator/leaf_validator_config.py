"""
leaf_validator_config.py
========================
Central configuration for the Leaf Validator transfer-learning pipeline.
All other leaf_stage*.py scripts import from here.
"""

import os
import gc
import logging
import numpy as np
import tensorflow as tf

# ============================================================
# PROJECT PATHS
# ============================================================
# This file lives in:  MobiKD/leaf_validator/leaf_validator_config.py
# PROJECT_ROOT points to the MobiKD/ root (one level up)
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)          # MobiKD/

MODELS_DIR    = os.path.join(PROJECT_ROOT, 'models', 'leaf_validator')
PLOTS_DIR     = os.path.join(PROJECT_ROOT, 'plots',  'leaf_validator')
RESULTS_DIR   = os.path.join(PROJECT_ROOT, 'results','leaf_validator')
DEPLOY_DIR    = os.path.join(PROJECT_ROOT, 'deployment', 'leaf_validator')
LOG_FILE      = os.path.join(PROJECT_ROOT, 'leaf_validator_pipeline.log')

# Dataset lives in data/ after reorganization
DATASET_DIR   = os.path.join(PROJECT_ROOT, 'data', 'lead and not leaf')
LEAF_DIR      = os.path.join(DATASET_DIR, 'leaf')
NOT_LEAF_DIR  = os.path.join(DATASET_DIR, 'non_leaf')

# Existing backbone model (from LiteRob pipeline — read-only, never overwrite)
BACKBONE_PATH = os.path.join(PROJECT_ROOT, 'models', 'literob', 'kd_mobilenetv2.keras')

# Leaf validator outputs (all go into models/leaf_validator/)
LEAF_KERAS_PATH  = os.path.join(MODELS_DIR, 'kd_mobilenetv2_leaf_validator.keras')
LEAF_FP32_PATH   = os.path.join(MODELS_DIR, 'kd_mobilenetv2_leaf_validator_fp32.tflite')
LEAF_F16_PATH    = os.path.join(MODELS_DIR, 'kd_mobilenetv2_leaf_validator_float16.tflite')
LEAF_CKPT_PATH   = os.path.join(MODELS_DIR, 'leaf_validator_ckpt.keras')

# ============================================================
# CLASS MAPPING  (class 0 = not_leaf, class 1 = leaf)
# ============================================================
CLASS_NAMES   = ['not_leaf', 'leaf']
NUM_CLASSES   = 2

# ============================================================
# IMAGE & TRAINING HYPER-PARAMETERS
# ============================================================
IMAGE_SIZE    = 32          # must match backbone input
CHANNELS      = 3
BATCH_SIZE    = 32

# Phase 1: train only the new head
P1_LR         = 1e-3
P1_EPOCHS     = 50
P1_PATIENCE   = 10

# Phase 2: fine-tune top backbone layers
P2_LR         = 1e-5
P2_EPOCHS     = 100
P2_PATIENCE   = 15
P2_UNFREEZE   = 30          # number of backbone layers to unfreeze from the top

# Dataset split ratios
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
TEST_RATIO    = 0.15

# Confidence threshold for deployment inference
CONFIDENCE_THRESHOLD = 0.70

# ============================================================
# LOGGING
# ============================================================
def setup_logging(name='LeafValidator'):
    os.makedirs(PROJECT_ROOT, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='a'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)

# ============================================================
# GPU SETUP
# ============================================================
def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            print(f'  GPU ready: {gpus[0].name}')
        except RuntimeError as e:
            print(f'  GPU config error: {e}')
    else:
        print('  WARNING: No GPU detected. Training on CPU.')
    np.random.seed(42)
    tf.random.set_seed(42)

# ============================================================
# MEMORY
# ============================================================
def clear_memory():
    gc.collect()
    tf.keras.backend.clear_session()
    print('  Memory cleared.')

# ============================================================
# DEGRADATION FUNCTIONS (for robustness evaluation)
# ============================================================
import cv2

def apply_blur(img):
    u8 = (img * 255).astype(np.uint8)
    return cv2.GaussianBlur(u8, (5, 5), 0).astype(np.float32) / 255.0

def apply_noise(img):
    return np.clip(
        img + np.random.normal(0, 0.15, img.shape).astype(np.float32), 0.0, 1.0
    )

def apply_low_light(img):
    return np.power(img, 2.5).astype(np.float32)

def apply_combined(img):
    return apply_low_light(apply_noise(apply_blur(img)))

def degrade_set(images, func, desc):
    print(f'  Building {desc}...', end=' ', flush=True)
    out = np.array([func(i) for i in images])
    print('done')
    return out

# ============================================================
# UTILITY BANNERS
# ============================================================
def banner(text):
    print('\n' + '=' * 65)
    print(f'  {text}')
    print('=' * 65)

def section(text):
    print(f'\n── {text} ──')
