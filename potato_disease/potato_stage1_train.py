"""
potato_stage1_train.py
======================
Transfer-learning training pipeline for Potato Disease Detection.
3-class classifier: Early Blight | Healthy | Late Blight

Uses the shared kd_mobilenetv2.keras backbone from the LiteRob project.

Run:
    python potato_stage1_train.py

Saves:
    models/potato_disease/kd_mobilenetv2_potato_disease.keras
    models/potato_disease/potato_disease_ckpt.keras
    results/potato_disease/phase1_log.csv
    results/potato_disease/phase2_log.csv
    results/potato_disease/final_metrics.json
    results/potato_disease/test_data.npz
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

from potato_config import (
    setup_logging, setup_gpu, clear_memory, banner, section,
    BACKBONE_PATH, POTATO_KERAS_PATH, POTATO_CKPT_PATH,
    EARLY_BLIGHT_DIR, LATE_BLIGHT_DIR, HEALTHY_DIR, NOT_POTATO_DIR,
    MODELS_DIR, RESULTS_DIR,
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
    banner('STAGE 1 -- Data Loading')

    paths, labels = [], []
    class_dirs = [
        (EARLY_BLIGHT_DIR, 0),   # early_blight
        (HEALTHY_DIR,      1),   # healthy
        (LATE_BLIGHT_DIR,  2),   # late_blight
        (NOT_POTATO_DIR,   3),   # not_potato_leaf
    ]
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    for folder, label in class_dirs:
        if not os.path.isdir(folder):
            print(f'  ERROR: {folder} not found.')
            print(f'  Run extract_potato.ps1 first to extract the dataset.')
            sys.exit(1)
        fps = [str(p) for p in Path(folder).rglob('*') if p.suffix.lower() in valid_exts]
        paths.extend(fps)
        labels.extend([label] * len(fps))

    if len(paths) == 0:
        print('  ERROR: No images loaded. Check dataset paths.')
        sys.exit(1)

    X_paths = np.array(paths)
    y       = np.array(labels, dtype=np.int32)
    print(f'\n  Total images found  : {len(X_paths)}')
    print(f'  Class distribution  : ' + ' | '.join(
        f'{CLASS_NAMES[i]}={np.sum(y==i)}' for i in range(NUM_CLASSES)
    ))
    return X_paths, y


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

    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: w for i, w in enumerate(cw)}
    print(f'  Class weights: {class_weights}')
    return X_train, X_val, X_test, y_train, y_val, y_test, class_weights


# ============================================================
# 2. AUGMENTATION
# ============================================================
def build_augmentation_pipeline():
    return keras.Sequential([
        layers.RandomFlip('horizontal_and_vertical'),
        layers.RandomRotation(0.15),
        layers.RandomZoom((-0.2, 0.2)),
        layers.RandomBrightness(0.3),
        layers.RandomContrast(0.3),
        layers.RandomTranslation(0.1, 0.1),
    ], name='augmentation')


def numpy_augment(img):
    """Probabilistic blur, noise, and low light — applied independently to create combinations."""
    # 50% chance of Blur
    if np.random.random() < 0.50:
        k = np.random.choice([3, 5])
        img = cv2.GaussianBlur(
            (img * 255).astype(np.uint8), (k, k), 0
        ).astype(np.float32) / 255.0
        
    # 50% chance of Noise
    if np.random.random() < 0.50:
        sigma = np.random.uniform(0.05, 0.20)
        img = np.clip(
            img + np.random.normal(0, sigma, img.shape).astype(np.float32),
            0.0, 1.0
        )
        
    # 50% chance of Low Light
    if np.random.random() < 0.50:
        gamma = np.random.uniform(1.5, 3.0)
        img = np.power(img, gamma).astype(np.float32)
        
    return img


def tf_numpy_augment(img, label):
    """Wrapper to apply numpy_augment inside tf.data pipeline."""
    def _do_augment(x):
        return numpy_augment(x)
    img_aug = tf.numpy_function(func=_do_augment, inp=[img], Tout=tf.float32)
    img_aug.set_shape(img.shape)
    return img_aug, label


def create_tf_dataset(paths, y, augment=False, shuffle=True):
    aug_pipeline = build_augmentation_pipeline() if augment else None

    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        label_oh = tf.one_hot(label, NUM_CLASSES)
        return img, label_oh

    def augment_fn(img, label_oh):
        img = aug_pipeline(img, training=True)
        return img, label_oh

    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=42)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_fn,       num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(tf_numpy_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ============================================================
# 3. MODEL BUILDING
# ============================================================
def build_potato_detector():
    """
    Load ImageNet-pretrained MobileNetV2, add custom head for potato disease classification.

    Architecture:
        MobileNetV2 (include_top=False, 96x96x3) -> GlobalAveragePooling2D -> (None, 1280)
        -> Dense(256, relu) -> Dropout(0.4)
        -> Dense(128, relu) -> Dropout(0.3)
        -> Dense(NUM_CLASSES, softmax)
    """
    section('Building Potato Disease Detector')

    print('  Loading MobileNetV2 (ImageNet weights, include_top=False)...')
    feature_extractor = keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
        include_top=False,
        weights='imagenet'
    )
    print(f'  Backbone params: {feature_extractor.count_params():,}')

    # Freeze backbone for Phase 1
    feature_extractor.trainable = False

    inp = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), name='potato_input')
    # MobileNetV2 expects inputs in [-1, 1]. Our data pipeline provides [0, 1].
    x   = layers.Rescaling(scale=2.0, offset=-1.0, name='mobilenet_preprocess')(inp)
    x   = feature_extractor(x, training=False)
    x   = layers.GlobalAveragePooling2D(name='head_gap')(x)
    x   = layers.Dense(256, activation='relu',  name='head_dense1')(x)
    x   = layers.Dropout(0.4,                   name='head_drop1')(x)
    x   = layers.Dense(128, activation='relu',  name='head_dense2')(x)
    x   = layers.Dropout(0.3,                   name='head_drop2')(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax', name='potato_output')(x)

    model = Model(inp, out, name='PotatoDiseaseDetector')
    trainable_p = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f'  Full model params     : {model.count_params():,}')
    print(f'  Trainable (Phase 1)   : {trainable_p:,}')

    return model, feature_extractor


# ============================================================
# 4. TRAINING
# ============================================================
def get_callbacks(phase, csv_path):
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=P1_PATIENCE if phase == 1 else P2_PATIENCE,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=POTATO_CKPT_PATH, monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
        keras.callbacks.CSVLogger(csv_path, append=False),
    ]


def compute_metrics(y_true, y_pred_probs):
    y_pred  = np.argmax(y_pred_probs, axis=1)
    report  = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
    )
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
        'per_class'         : report,
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
    total   = model.count_params()
    train_n = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f'  Total params    : {total:,}')
    print(f'  Trainable params: {train_n:,}')
    print(f'  Frozen params   : {total - train_n:,}')
    print(f'  Learning rate   : {lr}')
    print(f'  Max epochs      : {epochs}')

    cbs = get_callbacks(phase, csv_path)
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
    banner('POTATO DISEASE DETECTOR -- Transfer Learning Pipeline')
    setup_gpu()

    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load & split data (Paths only to save memory)
    X_paths, y = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test, class_weights = split_data(X_paths, y)

    # Load test images into memory so export & evaluation scripts can use them directly
    print('  Loading test images into memory to save test_data.npz...')
    X_test_imgs = []
    for p in X_test:
        img = Image.open(p).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        X_test_imgs.append(np.array(img, dtype=np.float32) / 255.0)
    X_test_imgs = np.array(X_test_imgs, dtype=np.float32)

    np.savez_compressed(
        os.path.join(RESULTS_DIR, 'test_data.npz'),
        X_test=X_test_imgs, y_test=y_test
    )
    print(f'\n  Test set saved -> results/potato_disease/test_data.npz')

    # Build tf.data pipelines (Lazy loading)
    section('Building tf.data Pipelines')
    train_ds = create_tf_dataset(X_train, y_train, augment=True,  shuffle=True)
    val_ds   = create_tf_dataset(X_val,   y_val,   augment=False, shuffle=False)
    test_ds  = create_tf_dataset(X_test,  y_test,  augment=False, shuffle=False)

    # Build model
    model, feature_extractor = build_potato_detector()

    # Phase 1: train head only
    p1_csv = os.path.join(RESULTS_DIR, 'phase1_log.csv')
    h1 = train_phase(model, train_ds, val_ds, phase=1,
                     lr=P1_LR, epochs=P1_EPOCHS, csv_path=p1_csv)

    section('Phase 1 Evaluation')
    val_preds_p1 = model.predict(val_ds, verbose=0)
    m1 = compute_metrics(y_val, val_preds_p1)
    print(f'  Val Accuracy : {m1["accuracy"]*100:.2f}%')
    print(f'  Val F1-macro : {m1["f1_macro"]:.4f}')
    print(f'  Balanced Acc : {m1["balanced_accuracy"]*100:.2f}%')

    # Phase 2: unfreeze top backbone layers
    section('Unfreezing Top Backbone Layers for Phase 2')
    feature_extractor.trainable = True
    for layer in feature_extractor.layers[:-P2_UNFREEZE]:
        layer.trainable = False
    for layer in feature_extractor.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
        if hasattr(layer, 'layers'):
            for sub in layer.layers:
                if isinstance(sub, keras.layers.BatchNormalization):
                    sub.trainable = False

    unfrozen = sum(1 for v in model.trainable_variables)
    print(f'  Unfrozen variable tensors: {unfrozen}')
    print(f'  BN layers kept frozen to prevent NaN.')

    p2_csv = os.path.join(RESULTS_DIR, 'phase2_log.csv')
    h2 = train_phase(model, train_ds, val_ds, phase=2,
                     lr=P2_LR, epochs=P2_EPOCHS, csv_path=p2_csv)

    # Final test evaluation
    banner('FINAL TEST SET EVALUATION')
    test_preds = model.predict(test_ds, verbose=0)[:len(y_test)]
    metrics    = compute_metrics(y_test, test_preds)

    print(f'  Test Accuracy       : {metrics["accuracy"]*100:.2f}%')
    print(f'  F1 (macro)          : {metrics["f1_macro"]:.4f}')
    print(f'  Precision (macro)   : {metrics["precision_macro"]:.4f}')
    print(f'  Recall (macro)      : {metrics["recall_macro"]:.4f}')
    print(f'  Balanced Accuracy   : {metrics["balanced_accuracy"]*100:.2f}%')
    print(f'\n  Confusion Matrix (early_blight | healthy | late_blight):')
    cm = np.array(metrics['confusion_matrix'])
    for row in cm:
        print(f'    {row}')

    section('Saving Final Keras Model')
    model.save(POTATO_KERAS_PATH)
    size_mb = os.path.getsize(POTATO_KERAS_PATH) / 1024 / 1024
    print(f'  Saved : {POTATO_KERAS_PATH}')
    print(f'  Size  : {size_mb:.2f} MB')

    metrics['model_path']             = POTATO_KERAS_PATH
    metrics['model_size_mb']          = round(size_mb, 2)
    metrics['elapsed_min']            = round((time.time() - t0) / 60, 2)
    metrics['phase1_best_val_acc']    = float(max(h1.history.get('val_accuracy', [0])))
    metrics['phase2_best_val_acc']    = float(max(h2.history.get('val_accuracy', [0])))
    metrics['class_names']            = CLASS_NAMES

    metrics_path = os.path.join(RESULTS_DIR, 'final_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'  Metrics saved -> {metrics_path}')

    elapsed = (time.time() - t0) / 60
    banner(f'DONE -- Elapsed: {elapsed:.1f} min')
    print('  Next: python potato_stage2_export.py')


if __name__ == '__main__':
    run()
