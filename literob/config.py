import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import gc
import time
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = f'{PROJECT_ROOT}/models'
RESULTS_DIR  = f'{PROJECT_ROOT}/results'
DATA_DIR     = f'{PROJECT_ROOT}/data'
PLOTS_DIR    = f'{PROJECT_ROOT}/plots'
LOG_FILE     = f'{PROJECT_ROOT}/pipeline.log'

# ============================================================
# HYPER-PARAMETERS
# ============================================================
TREE_FINE_LABELS  = [47, 52, 56, 59, 96]
IMAGE_SIZE        = 64
BATCH_SIZE        = 8
TEACHER_BATCH_SIZE = 8
TEACHER_LR        = 1e-4
STUDENT_LR        = 1e-4
TEMPERATURE       = 4.0
ALPHA             = 0.3
TEACHER_P1_EPOCHS = 20
BASELINE_EPOCHS   = 200
KD_EPOCHS         = 200
LITEROB_EPOCHS    = 200
PATIENCE          = 10

# ============================================================
# LOGGING
# ============================================================
def setup_logging():
    os.makedirs(PROJECT_ROOT, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='a'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('LiteRob')

# ============================================================
# GPU SETUP
# ============================================================
def setup_gpu(logger=None):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            print(f'  GPU: {gpus[0].name} ready')
            if logger:
                logger.info(f'GPU: {gpus[0].name}')
        except RuntimeError as e:
            print(f'  GPU config error: {e}')
    else:
        print('  WARNING: No GPU found. Training will be slow on CPU.')
    np.random.seed(42)
    tf.random.set_seed(42)

# ============================================================
# MEMORY MANAGEMENT
# ============================================================
def clear_gpu_memory():
    gc.collect()
    tf.keras.backend.clear_session()
    print('  GPU + TF memory cleared.')

# ============================================================
# DEGRADATION FUNCTIONS
# ============================================================
import cv2

def apply_blur(img):
    u8 = (img * 255).astype(np.uint8)
    return cv2.GaussianBlur(u8, (5, 5), 0).astype(np.float32) / 255.0

def apply_noise(img):
    return np.clip(
        img + np.random.normal(0, 0.15, img.shape).astype(np.float32), 0, 1
    )

def apply_low_light(img):
    return np.power(img, 2.5).astype(np.float32)

def apply_combined(img):
    return apply_low_light(apply_noise(apply_blur(img)))

def degrade_dataset(images, func, desc):
    print(f'  Creating {desc}...', end=' ', flush=True)
    out = np.array([func(i) for i in images])
    print('done')
    return out

# ============================================================
# SHARED UTILITIES
# ============================================================
def get_sample_weights(yb, cw_arr):
    labels = tf.argmax(yb, axis=1)
    sw = tf.where(
        tf.equal(labels, 1),
        tf.fill(tf.shape(labels), float(cw_arr[1])),
        tf.fill(tf.shape(labels), float(cw_arr[0]))
    )
    return tf.cast(sw, tf.float32)

def freeze_model(model):
    model.trainable = False
    for layer in model.layers:
        layer.trainable = False


def soften_probabilities(probs, temperature):
    """Apply KD temperature scaling to probability outputs."""
    logits = tf.math.log(tf.clip_by_value(probs, 1e-8, 1.0))
    return tf.nn.softmax(logits / temperature, axis=-1)

# ============================================================
# MODEL BUILDERS
# ============================================================
def build_teacher_model():
    inp = keras.Input(shape=(32, 32, 3))
    x = layers.Resizing(IMAGE_SIZE, IMAGE_SIZE)(inp)

    # ResNet50 Caffe Preprocessing via Conv2D (fully serializable)
    x = layers.Rescaling(255.0)(x)
    prep_conv = layers.Conv2D(3, 1, use_bias=True, trainable=False, name='caffe_prep')
    x = prep_conv(x)
    W = np.zeros((1, 1, 3, 3), dtype=np.float32)
    W[0, 0, 0, 2] = 1.0  # R -> B channel
    W[0, 0, 1, 1] = 1.0  # G -> G channel
    W[0, 0, 2, 0] = 1.0  # B -> R channel
    b = np.array([-103.939, -116.779, -123.68], dtype=np.float32)
    prep_conv.set_weights([W, b])

    base = ResNet50(weights='imagenet', include_top=False,
                    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(2, activation='softmax')(x)
    return Model(inp, out, name='Teacher_ResNet50'), base


def build_student(name):
    inp = keras.Input(shape=(32, 32, 3))
    x = layers.Resizing(IMAGE_SIZE, IMAGE_SIZE)(inp)
    # MobileNetV2 preprocessing: [0,1] -> [-1,1]
    x = layers.Rescaling(scale=2.0, offset=-1.0)(x)
    base = MobileNetV2(weights='imagenet', include_top=False,
                       input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    # Freeze first ~125 layers; only fine-tune the last block + conv_pw_13
    # This cuts backward-pass activation memory by ~4x on a 1.7 GB GPU
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(2, activation='softmax')(x)
    return Model(inp, out, name=name)


def load_teacher():
    model = keras.models.load_model(f'{MODELS_DIR}/teacher_resnet50.keras')
    freeze_model(model)
    return model

# ============================================================
# KNOWLEDGE DISTILLATION
# ============================================================
def kd_loss(model, teacher, xb, yb, cw_arr):
    ce_fn = keras.losses.CategoricalCrossentropy()
    kl_fn = keras.losses.KLDivergence()
    sw = get_sample_weights(yb, cw_arr)

    student_probs = model(xb, training=True)
    teacher_probs = tf.stop_gradient(teacher(xb, training=False))

    ce = ce_fn(yb, student_probs, sample_weight=sw)
    kl = kl_fn(
        soften_probabilities(teacher_probs, TEMPERATURE),
        soften_probabilities(student_probs, TEMPERATURE)
    )
    loss = ALPHA * ce + (1 - ALPHA) * (TEMPERATURE ** 2) * kl
    return loss, student_probs


def custom_train_kd(student, teacher, data, max_epochs, name, log_path,
                    checkpoint_path=None):
    """Train MobileNetV2 student with ResNet50 teacher distillation."""
    freeze_model(teacher)

    optimizer = keras.optimizers.Adam(STUDENT_LR)
    cw_arr    = data['class_weights']
    x_train   = data['x_train']
    y_train   = data['y_train']
    x_val     = data['x_val_clean'] if 'x_val_clean' in data else data['x_test_clean']
    y_val     = data['y_val'] if 'y_val' in data else data['y_test']
    n         = x_train.shape[0]
    n_batches = int(np.ceil(n / BATCH_SIZE))
    best_val  = 0.0
    pat_count = 0
    best_w    = None

    ce_fn = keras.losses.CategoricalCrossentropy()
    kl_fn = keras.losses.KLDivergence()

    @tf.function(reduce_retracing=True)
    def train_step(xb, yb):
        xb     = tf.cast(xb, tf.float32)
        yb     = tf.cast(yb, tf.float32)
        sw     = get_sample_weights(yb, cw_arr)

        teacher_probs = tf.stop_gradient(teacher(xb, training=False))

        with tf.GradientTape() as tape:
            student_probs = student(xb, training=True)
            ce = ce_fn(yb, student_probs, sample_weight=sw)
            kl = kl_fn(
                soften_probabilities(teacher_probs, TEMPERATURE),
                soften_probabilities(student_probs, TEMPERATURE)
            )
            loss = ALPHA * ce + (1 - ALPHA) * (TEMPERATURE ** 2) * kl
        grads = tape.gradient(loss, student.trainable_variables)
        optimizer.apply_gradients(zip(grads, student.trainable_variables))
        acc = tf.reduce_mean(keras.metrics.categorical_accuracy(yb, student_probs))
        return loss, acc

    @tf.function(reduce_retracing=True)
    def val_step(xb):
        return student(tf.cast(xb, tf.float32), training=False)

    log_f = open(log_path, 'w')
    log_f.write('epoch,loss,acc,val_acc\n')
    print(f'\n  Training {name} up to {max_epochs} epochs...')
    print(f'  KD settings: alpha={ALPHA}  temperature={TEMPERATURE}')

    for epoch in range(max_epochs):
        idx    = np.random.permutation(n)
        ep_loss, ep_acc = [], []

        for b in range(n_batches):
            s = b * BATCH_SIZE
            e = s + BATCH_SIZE
            batch_idx = idx[s:e]
            loss, acc = train_step(x_train[batch_idx], y_train[batch_idx])
            ep_loss.append(float(loss))
            ep_acc.append(float(acc))

        # Batched validation
        val_preds_list = []
        val_steps = int(np.ceil(len(x_val) / BATCH_SIZE))
        for i in range(val_steps):
            val_xb = x_val[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            val_preds_list.append(val_step(val_xb))
        val_preds = tf.concat(val_preds_list, axis=0)
        val_acc = float(tf.reduce_mean(
            keras.metrics.categorical_accuracy(
                tf.constant(y_val, dtype=tf.float32), val_preds
            )
        ))

        mean_loss = float(np.mean(ep_loss))
        mean_acc  = float(np.mean(ep_acc))
        log_f.write(f'{epoch+1},{mean_loss:.6f},{mean_acc:.6f},{val_acc:.6f}\n')
        log_f.flush()
        print(f'  Epoch {epoch+1:3d}/{max_epochs} | '
              f'loss: {mean_loss:.4f} | '
              f'acc: {mean_acc*100:.2f}% | '
              f'val_acc: {val_acc*100:.2f}%')

        if val_acc > best_val:
            best_val  = val_acc
            best_w    = student.get_weights()
            pat_count = 0
            if checkpoint_path is not None:
                student.save(checkpoint_path)
        else:
            pat_count += 1
            if pat_count >= PATIENCE:
                print(f'\n  Early stopping at epoch {epoch+1}. '
                      f'Best val_acc: {best_val*100:.2f}%')
                break

    log_f.close()
    if best_w is not None:
        student.set_weights(best_w)
        print(f'  Best weights restored. Val acc: {best_val*100:.2f}%')
    return student

# ============================================================
# EVALUATION
# ============================================================
def evaluate_model(model, data, is_tflite=False, tflite_path=None):
    test_sets = {
        'Clean'    : data['x_test_clean'],
        'Blur'     : data['x_test_blur'],
        'Noise'    : data['x_test_noise'],
        'Low Light': data['x_test_lowlight'],
        'Combined' : data['x_test_combined'],
    }
    y_test  = data['y_test']
    results = {}

    if is_tflite:
        interp = tf.lite.Interpreter(model_path=tflite_path)
        interp.allocate_tensors()
        inp_idx = interp.get_input_details()[0]['index']
        out_idx = interp.get_output_details()[0]['index']
        for name, x in test_sets.items():
            correct = 0
            for img, label in zip(x, y_test):
                inp = img[np.newaxis].astype(np.float32)
                interp.set_tensor(inp_idx, inp)
                interp.invoke()
                pred = interp.get_tensor(out_idx)[0]
                if np.argmax(pred) == np.argmax(label):
                    correct += 1
            results[name] = correct / len(y_test)
    else:
        for name, x in test_sets.items():
            preds = model.predict(x, batch_size=BATCH_SIZE, verbose=0)
            results[name] = float(
                np.mean(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1))
            )
    return results
