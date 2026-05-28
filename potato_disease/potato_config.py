"""
potato_config.py
================
Central configuration for the Potato Disease Detection pipeline.
All potato_stage*.py scripts import from here.

Pipeline position:
    Input → [Leaf Validator] → IS leaf → [Potato Disease Detector]
                                          Early Blight | Late Blight | Healthy
"""

import os
import gc
import logging
import numpy as np
import tensorflow as tf
import cv2

# ============================================================
# PROJECT PATHS
# ============================================================
# This file lives in:  MobiKD/potato_disease/potato_config.py
# PROJECT_ROOT points to the MobiKD/ root (one level up)
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)          # MobiKD/

MODELS_DIR   = os.path.join(PROJECT_ROOT, 'models',  'potato_disease')
PLOTS_DIR    = os.path.join(PROJECT_ROOT, 'plots',   'potato_disease')
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'results', 'potato_disease')
DEPLOY_DIR   = os.path.join(PROJECT_ROOT, 'deployment', 'potato_pipeline')
LOG_FILE     = os.path.join(PROJECT_ROOT, 'potato_pipeline.log')

# Dataset — extracted from archive.zip by extract_potato.ps1
DATASET_DIR       = os.path.join(PROJECT_ROOT, 'data', 'potato')
EARLY_BLIGHT_DIR  = os.path.join(DATASET_DIR, 'early_blight')
HEALTHY_DIR       = os.path.join(DATASET_DIR, 'healthy')
LATE_BLIGHT_DIR   = os.path.join(DATASET_DIR, 'late_blight')
NOT_POTATO_DIR    = os.path.join(DATASET_DIR, 'not_potato_leaf')

# Backbone: shared LiteRob MobileNetV2 (read-only, never overwrite)
BACKBONE_PATH = os.path.join(PROJECT_ROOT, 'models', 'literob', 'kd_mobilenetv2.keras')

# Potato disease model outputs
POTATO_KERAS_PATH = os.path.join(MODELS_DIR, 'kd_mobilenetv2_potato_disease.keras')
POTATO_FP32_PATH  = os.path.join(MODELS_DIR, 'kd_mobilenetv2_potato_disease_fp32.tflite')
POTATO_F16_PATH   = os.path.join(MODELS_DIR, 'kd_mobilenetv2_potato_disease_float16.tflite')
POTATO_CKPT_PATH  = os.path.join(MODELS_DIR, 'potato_disease_ckpt.keras')

# Leaf validator model (for the combined pipeline)
LEAF_VALIDATOR_PATH = os.path.join(
    PROJECT_ROOT, 'models', 'leaf_validator',
    'kd_mobilenetv2_leaf_validator_float16.tflite'
)

# ============================================================
# CLASS MAPPING  (alphabetical order — matches folder sort)
# ============================================================
#  0 = early_blight    (Potato early blight)
#  1 = healthy         (Potato healthy)
#  2 = late_blight     (Potato late blight)
#  3 = not_potato_leaf (Other plant leaves)
CLASS_NAMES  = ['early_blight', 'healthy', 'late_blight', 'not_potato_leaf']
NUM_CLASSES  = 4

# Human-readable labels for reports and inference output
CLASS_LABELS = {
    0: 'Potato: Early Blight',
    1: 'Potato: Healthy',
    2: 'Potato: Late Blight',
    3: 'Not a Potato Leaf',
}

# Confidence threshold for deployment
CONFIDENCE_THRESHOLD = 0.65

# ============================================================
# IMAGE & TRAINING HYPER-PARAMETERS
# ============================================================
IMAGE_SIZE  = 96        # upgraded to 96x96 for fine texture details
CHANNELS    = 3
BATCH_SIZE  = 32

# Phase 1: train only the new head (backbone frozen)
P1_LR       = 1e-3
P1_EPOCHS   = 50
P1_PATIENCE = 10

# Phase 2: fine-tune top backbone layers
P2_LR       = 1e-5
P2_EPOCHS   = 100
P2_PATIENCE = 15
P2_UNFREEZE = 30        # unfreeze top N backbone layers

# Dataset split
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ============================================================
# LOGGING
# ============================================================
def setup_logging(name='PotatoDisease'):
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
    print(f'\n-- {text} --')
