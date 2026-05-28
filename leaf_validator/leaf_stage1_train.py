"""
leaf_stage1_train.py
====================
Transfer-learning training pipeline for LEAF vs NOT_LEAF classifier.

Uses the distilled kd_mobilenetv2.keras backbone (from MobiKD project)
and attaches a new binary classification head.

Run:
    python leaf_stage1_train.py

Saves:
    models/kd_mobilenetv2_leaf_validator.keras
    models/leaf_validator_ckpt.keras
    results/leaf/phase1_log.csv
    results/leaf/phase2_log.csv
    results/leaf/final_metrics.json
"""

import os, sys, time, json, warnings
os.environ['TF_GPU_ALLOCATOR']          = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL']      = '2'
warnings.filterwarnings('ignore')

import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, balanced_accuracy_score, precision_score, recall_score
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

from leaf_validator_config import (
    setup_logging, setup_gpu, clear_memory, banner, section,
    BACKBONE_PATH, LEAF_KERAS_PATH, LEAF_CKPT_PATH,
    LEAF_DIR, NOT_LEAF_DIR, MODELS_DIR, RESULTS_DIR,
    IMAGE_SIZE, CHANNELS, BATCH_SIZE, NUM_CLASSES, CLASS_NAMES,
    P1_LR, P1_EPOCHS, P1_PATIENCE,
    P2_LR, P2_EPOCHS, P2_PATIENCE, P2_UNFREEZE,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)

logger = setup_logging()

# ============================================================
# 1. DATA LOADING
# ============================================================
def load_dataset():
    banner('STAGE 1 — Data Loading')

    images, labels = [], []
    class_dirs = [
        (NOT_LEAF_DIR, 0),  # class 0 = not_leaf
        (LEAF_DIR,     1),  # class 1 = leaf
    ]
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    for folder, label in class_dirs:
        if not os.path.isdir(folder):
            print(f'  ERROR: {folder} not found.')
            sys.exit(1)
        paths = [p for p in Path(folder).rglob('*') if p.suffix.lower() in valid_exts]
        print(f'  Class {label} ({CLASS_NAMES[label]}): {len(paths)} images from {folder}')
        for p in paths:
            try:
                img = Image.open(p).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0
                images.append(arr)
                labels.append(label)
            except Exception as e:
                print(f'    Skip {p.name}: {e}')

    if len(images) == 0:
        print('  ERROR: No images loaded. Check dataset paths.')
        sys.exit(1)

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    print(f'\n  Total images loaded : {len(X)}')
    print(f'  Class distribution  : not_leaf={np.sum(y==0)}, leaf={np.sum(y==1)}')
    print(f'  Image shape         : {X.shape[1:]}  dtype={X.dtype}')
    print(f'  Pixel range         : [{X.min():.3f}, {X.max():.3f}]')
    return X, y


def split_data(X, y):
    section('Train / Val / Test Split')
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, stratify=y, random_state=42
    )
    val_frac = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, stratify=y_temp, random_state=42
    )
    print(f'  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}')

    # Compute class weights for imbalanced datasets
    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: w for i, w in enumerate(cw)}
    print(f'  Class weights: {class_weights}')

    return X_train, X_val, X_test, y_train, y_val, y_test, class_weights


# ============================================================
# 2. AUGMENTATION
# ============================================================
def build_augmentation_pipeline():
    """Keras Sequential augmentation block — applied during training only."""
    return keras.Sequential([
        layers.RandomFlip('horizontal_and_vertical'),
        layers.RandomRotation(0.15),          # ±15% of 360° = ±54°
        layers.RandomZoom((-0.2, 0.2)),        # zoom ±20%
        layers.RandomBrightness(0.3),
        layers.RandomContrast(0.3),
        layers.RandomTranslation(0.1, 0.1),   # translate ±10%
        # NOTE: GaussianNoise removed — at 32×32 it fires on every image
        # and destroys clean-accuracy learning. Noise is handled probabilistically
        # by numpy_augment (30% of images, moderate sigma).
    ], name='augmentation')


def numpy_augment(img):
    """Probabilistic blur/noise augmentation — applied to 60% of images.
    Noise sigma 0.03-0.15 gives robustness without destroying clean features."""
    r = np.random.random()
    if r < 0.30:  # 30% blur
        k = np.random.choice([3, 5])
        img = cv2.GaussianBlur((img * 255).astype(np.uint8), (k, k), 0).astype(np.float32) / 255.0
    elif r < 0.60:  # 30% noise — moderate sigma, not too destructive at 32x32
        sigma = np.random.uniform(0.03, 0.15)
        img = np.clip(img + np.random.normal(0, sigma, img.shape).astype(np.float32), 0.0, 1.0)
    # remaining 40% — no extra degradation (clean path)
    return img


def tf_numpy_augment(img, label):
    """Wrapper to call numpy_augment inside a tf.data pipeline.
    Note: tf.numpy_function passes plain numpy arrays to func — do NOT call .numpy()."""
    def _do_augment(x):
        # x is already a numpy ndarray here
        return numpy_augment(x)

    img_aug = tf.numpy_function(func=_do_augment, inp=[img], Tout=tf.float32)
    img_aug.set_shape(img.shape)
    return img_aug, label


def create_tf_dataset(X, y, augment=False, shuffle=True):
    """Create a tf.data.Dataset pipeline."""
    aug_pipeline = build_augmentation_pipeline() if augment else None

    def preprocess(img, label):
        img = tf.cast(img, tf.float32)
        label_oh = tf.one_hot(label, NUM_CLASSES)
        return img, label_oh

    def augment_fn(img, label_oh):
        img = aug_pipeline(img, training=True)
        return img, label_oh

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=42)
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_fn,         num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(tf_numpy_augment,   num_parallel_calls=tf.data.AUTOTUNE)  # blur/noise
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ============================================================
# 3. MODEL BUILDING — TRANSFER LEARNING
# ============================================================
def build_leaf_validator():
    """
    Load kd_mobilenetv2.keras backbone, strip its head, add new binary head.

    Architecture:
        backbone (up to GlobalAveragePooling2D output)
        → BatchNormalization
        → Dense(128, relu)
        → Dropout(0.4)
        → Dense(64, relu)
        → Dropout(0.3)
        → Dense(2, softmax)   →  [not_leaf, leaf]
    """
    section('Building Leaf Validator Model')

    if not os.path.exists(BACKBONE_PATH):
        print(f'  ERROR: Backbone not found at {BACKBONE_PATH}')
        sys.exit(1)

    print(f'  Loading backbone from: {BACKBONE_PATH}')
    backbone_full = keras.models.load_model(BACKBONE_PATH)
    print(f'  Backbone loaded: {backbone_full.name}')
    print(f'  Backbone params: {backbone_full.count_params():,}')
    print(f'  Input  shape   : {backbone_full.input_shape}')
    print(f'  Output shape   : {backbone_full.output_shape}')

    # ── Find the GAP output (before the old Dense head) ──────────────────────
    # We search for the GlobalAveragePooling2D layer output
    gap_layer = None
    for layer in backbone_full.layers:
        if isinstance(layer, keras.layers.GlobalAveragePooling2D):
            gap_layer = layer
            break
        # Also catch it inside a nested model (MobileNetV2 sub-model)
        if hasattr(layer, 'layers'):
            for sub in layer.layers:
                if isinstance(sub, keras.layers.GlobalAveragePooling2D):
                    gap_layer = sub  # found inside sub-model

    # Approach: use all layers up to (and including) the BatchNormalization
    # that follows GAP in the original kd_mobilenetv2.keras
    # Original head: GAP → BN → Dense(128,relu) → Dropout(0.3) → Dense(2,softmax)
    # We keep everything up to BN output as our feature vector.
    bn_layer = None
    found_gap = False
    for layer in backbone_full.layers:
        if isinstance(layer, keras.layers.GlobalAveragePooling2D):
            found_gap = True
        if found_gap and isinstance(layer, keras.layers.BatchNormalization):
            bn_layer = layer
            break

    # Build feature extractor up to BN
    # Keras 3 compatible: use backbone_full.get_layer(name).output
    # instead of bn_layer.output / bn_layer.output_shape (removed in Keras 3)
    if bn_layer is not None:
        feature_extractor = Model(
            inputs  = backbone_full.input,
            outputs = backbone_full.get_layer(bn_layer.name).output,
            name    = 'MobiKD_Backbone'
        )
        print(f'  Feature cutpoint: {bn_layer.name}')
    else:
        # Fallback: cut just before the first Dense in the top head
        dense_layers = [l for l in backbone_full.layers
                        if isinstance(l, keras.layers.Dense)]
        cut_layer    = backbone_full.layers[
            backbone_full.layers.index(dense_layers[0]) - 1
        ]
        feature_extractor = Model(
            inputs  = backbone_full.input,
            outputs = backbone_full.get_layer(cut_layer.name).output,
            name    = 'MobiKD_Backbone'
        )
        print(f'  Feature cutpoint (fallback): {cut_layer.name}')

    print(f'  Feature extractor output shape: {feature_extractor.output_shape}')

    # ── Freeze the entire backbone for Phase 1 ────────────────────────────────
    feature_extractor.trainable = False
    print(f'  Backbone frozen (Phase 1).')

    # ── Attach new classification head ────────────────────────────────────────
    inp = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), name='leaf_input')
    x   = feature_extractor(inp, training=False)

    x = layers.Dense(128, activation='relu', name='head_dense1')(x)
    x = layers.Dropout(0.4, name='head_drop1')(x)
    x = layers.Dense(64, activation='relu', name='head_dense2')(x)
    x = layers.Dropout(0.3, name='head_drop2')(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax', name='leaf_output')(x)

    model = Model(inp, out, name='LeafValidator')
    print(f'  Full model params: {model.count_params():,}')
    trainable_p = sum(
        np.prod(v.shape) for v in model.trainable_variables
    )
    print(f'  Trainable params (Phase 1): {trainable_p:,}')

    return model, feature_extractor


# ============================================================
# 4. TRAINING
# ============================================================
def get_callbacks(phase, csv_path, ckpt_path):
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=P1_PATIENCE if phase == 1 else P2_PATIENCE,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=5, min_lr=1e-7, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path, monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
        keras.callbacks.CSVLogger(csv_path, append=False),
    ]


def compute_metrics(y_true, y_pred_probs):
    y_pred = np.argmax(y_pred_probs, axis=1)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    f1      = f1_score(y_true, y_pred, average='macro')
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec    = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec     = recall_score(y_true, y_pred, average='macro', zero_division=0)
    cm      = confusion_matrix(y_true, y_pred)
    return {
        'accuracy'          : float(np.mean(y_pred == y_true)),
        'f1_macro'          : float(f1),
        'precision_macro'   : float(prec),
        'recall_macro'      : float(rec),
        'balanced_accuracy' : float(bal_acc),
        'confusion_matrix'  : cm.tolist(),
        'per_class'         : report
    }


def train_phase(model, train_ds, val_ds, phase, lr, epochs, csv_path):
    banner(f'PHASE {phase} TRAINING')
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=lr),
        loss      = 'categorical_crossentropy',
        metrics   = [
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ]
    )
    total = model.count_params()
    train = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f'  Total params    : {total:,}')
    print(f'  Trainable params: {train:,}')
    print(f'  Frozen params   : {total - train:,}')
    print(f'  Learning rate   : {lr}')
    print(f'  Max epochs      : {epochs}')

    cbs = get_callbacks(phase, csv_path, LEAF_CKPT_PATH)
    history = model.fit(
        train_ds,
        epochs          = epochs,
        validation_data = val_ds,
        callbacks       = cbs,
        verbose         = 1
    )
    return history


# ============================================================
# 5. MAIN
# ============================================================
def run():
    t0 = time.time()
    banner('LEAF VALIDATOR — Transfer Learning Pipeline')
    setup_gpu()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    X, y = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test, class_weights = split_data(X, y)

    # Save test set for later stages
    np.savez_compressed(
        os.path.join(RESULTS_DIR, 'test_data.npz'),
        X_test=X_test, y_test=y_test
    )
    print(f'\n  Test set saved → results/leaf/test_data.npz')

    # ── Build tf.data pipelines ───────────────────────────────────────────────
    section('Building tf.data Pipelines')
    train_ds = create_tf_dataset(X_train, y_train, augment=True,  shuffle=True)
    val_ds   = create_tf_dataset(X_val,   y_val,   augment=False, shuffle=False)
    test_ds  = create_tf_dataset(X_test,  y_test,  augment=False, shuffle=False)

    # ── Build model ───────────────────────────────────────────────────────────
    model, feature_extractor = build_leaf_validator()

    # ── Phase 1: Train head only ──────────────────────────────────────────────
    p1_csv = os.path.join(RESULTS_DIR, 'phase1_log.csv')
    h1 = train_phase(model, train_ds, val_ds, phase=1,
                     lr=P1_LR, epochs=P1_EPOCHS, csv_path=p1_csv)

    # Evaluate after Phase 1
    section('Phase 1 Evaluation')
    val_preds_p1 = model.predict(val_ds, verbose=0)
    m1 = compute_metrics(y_val, val_preds_p1)
    print(f'  Val Accuracy : {m1["accuracy"]*100:.2f}%')
    print(f'  Val F1-macro : {m1["f1_macro"]:.4f}')
    print(f'  Balanced Acc : {m1["balanced_accuracy"]*100:.2f}%')

    # ── Phase 2: Unfreeze top backbone layers ─────────────────────────────────
    section('Unfreezing Top Backbone Layers for Phase 2')
    feature_extractor.trainable = True
    # Freeze all but the last P2_UNFREEZE layers
    for layer in feature_extractor.layers[:-P2_UNFREEZE]:
        layer.trainable = False
    # Keep all BN layers in inference mode to avoid NaN
    for layer in feature_extractor.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
        if hasattr(layer, 'layers'):
            for sub in layer.layers:
                if isinstance(sub, keras.layers.BatchNormalization):
                    sub.trainable = False

    unfrozen = sum(1 for v in model.trainable_variables)
    print(f'  Unfrozen variable tensors: {unfrozen}')
    print(f'  BN layers kept frozen (inference-mode) to prevent NaN.')

    p2_csv = os.path.join(RESULTS_DIR, 'phase2_log.csv')
    h2 = train_phase(model, train_ds, val_ds, phase=2,
                     lr=P2_LR, epochs=P2_EPOCHS, csv_path=p2_csv)

    # ── Final evaluation on test set ──────────────────────────────────────────
    banner('FINAL TEST SET EVALUATION')
    test_preds = model.predict(test_ds, verbose=0)
    # Trim to actual test size (last batch may pad)
    test_preds = test_preds[:len(y_test)]
    metrics = compute_metrics(y_test, test_preds)

    print(f'  Test Accuracy       : {metrics["accuracy"]*100:.2f}%')
    print(f'  F1 (macro)          : {metrics["f1_macro"]:.4f}')
    print(f'  Precision (macro)   : {metrics["precision_macro"]:.4f}')
    print(f'  Recall (macro)      : {metrics["recall_macro"]:.4f}')
    print(f'  Balanced Accuracy   : {metrics["balanced_accuracy"]*100:.2f}%')
    print(f'\n  Confusion Matrix:')
    cm = np.array(metrics['confusion_matrix'])
    print(f'    {cm[0]}')
    print(f'    {cm[1]}')

    # ── Save final Keras model ────────────────────────────────────────────────
    section('Saving Final Keras Model')
    model.save(LEAF_KERAS_PATH)
    size_mb = os.path.getsize(LEAF_KERAS_PATH) / 1024 / 1024
    print(f'  Saved : {LEAF_KERAS_PATH}')
    print(f'  Size  : {size_mb:.2f} MB')

    # ── Save metrics JSON ──────────────────────────────────────────────────────
    metrics['model_path']  = LEAF_KERAS_PATH
    metrics['model_size_mb'] = round(size_mb, 2)
    metrics['elapsed_min'] = round((time.time() - t0) / 60, 2)
    metrics['phase1_best_val_acc'] = float(max(h1.history.get('val_accuracy', [0])))
    metrics['phase2_best_val_acc'] = float(max(h2.history.get('val_accuracy', [0])))

    metrics_path = os.path.join(RESULTS_DIR, 'final_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'  Metrics saved → {metrics_path}')

    elapsed = (time.time() - t0) / 60
    banner(f'DONE — Elapsed: {elapsed:.1f} min')
    print('  Next: python leaf_stage2_export.py')


if __name__ == '__main__':
    run()
