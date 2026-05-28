"""
STAGE 5 TFLite FIX - Stable TFLite Export Pipeline
====================================================
Run:    python3 stage5_tflite_fix.py
Needs:  models/literob_mobilenet.keras  (stage5_literob.py must have run)
Saves:  models/literob_fp32.tflite
        models/literob_float16_fixed.tflite  (only if FP32 passes)
"""
import os, sys, time
os.environ['TF_GPU_ALLOCATOR']      = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')

KERAS_PATH = os.path.join(MODELS_DIR, 'literob_mobilenet.keras')
FP32_PATH  = os.path.join(MODELS_DIR, 'literob_fp32.tflite')
F16_PATH   = os.path.join(MODELS_DIR, 'literob_float16_fixed.tflite')

BATCH_SIZE = 32

# ── Utility ───────────────────────────────────────────────────────────────────
def banner(t):
    print('\n' + '='*65)
    print(f'  {t}')
    print('='*65)

def section(t):
    print(f'\n── {t} ──')

def check_sanity(preds, label, n_classes=2):
    nan_n  = int(np.sum(np.isnan(preds)))
    inf_n  = int(np.sum(np.isinf(preds)))
    sums   = np.sum(preds, axis=1)
    sum_ok = bool(np.allclose(sums, 1.0, atol=1e-4))
    n_cls  = len(np.unique(np.argmax(preds, axis=1)))
    print(f'  [{label}] NaN={nan_n}  Inf={inf_n}  '
          f'SumTo1={sum_ok}  ClassesHit={n_cls}/{n_classes}')
    print(f'  [{label}] prob-sum  min={sums.min():.6f}  max={sums.max():.6f}')
    print(f'  [{label}] output    min={preds.min():.6f}  max={preds.max():.6f}')
    if nan_n > 0:
        raise ValueError(f'FAIL [{label}]: {nan_n} NaN(s) detected!')
    if inf_n > 0:
        raise ValueError(f'FAIL [{label}]: {inf_n} Inf(s) detected!')
    if not sum_ok:
        raise ValueError(f'FAIL [{label}]: probs do not sum to 1 '
                         f'(range {sums.min():.4f}–{sums.max():.4f})')
    if n_cls < 2:
        raise ValueError(f'FAIL [{label}]: model predicts only class 0 — collapsed!')
    print(f'  [{label}] ✓ Sanity passed.')

def debug_bn(model):
    section('BatchNorm Diagnostics')
    count = 0
    def _inspect(lyr):
        nonlocal count
        if isinstance(lyr, keras.layers.BatchNormalization):
            mu  = lyr.moving_mean.numpy()
            var = lyr.moving_variance.numpy()
            gam = lyr.gamma.numpy()
            bet = lyr.beta.numpy()
            bad = np.any(var <= 0)
            print(f'  BN {lyr.name:40s} '
                  f'mu=[{mu.min():.3f},{mu.max():.3f}] '
                  f'var=[{var.min():.3f},{var.max():.3f}]'
                  + (' ← VAR<=0 WARNING' if bad else ''))
            count += 1
    for layer in model.layers:
        _inspect(layer)
        if hasattr(layer, 'layers'):
            for sub in layer.layers:
                _inspect(sub)
    print(f'  Total BN layers: {count}')

def tflite_predict(path, x_samples):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    print(f'  TFLite input  shape={inp_d["shape"].tolist()}  dtype={inp_d["dtype"].__name__}')
    print(f'  TFLite output shape={out_d["shape"].tolist()}  dtype={out_d["dtype"].__name__}')
    preds = []
    for img in x_samples:
        interp.set_tensor(inp_d['index'], img[np.newaxis].astype(np.float32))
        interp.invoke()
        preds.append(interp.get_tensor(out_d['index'])[0].copy())
    return np.array(preds, dtype=np.float32)

def tflite_accuracy(path, x, y, name):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp_i = interp.get_input_details()[0]['index']
    out_i = interp.get_output_details()[0]['index']
    correct = 0
    for img, lbl in zip(x, y):
        interp.set_tensor(inp_i, img[np.newaxis].astype(np.float32))
        interp.invoke()
        p = interp.get_tensor(out_i)[0]
        if np.argmax(p) == np.argmax(lbl):
            correct += 1
    acc = correct / len(y)
    print(f'  {name}: {acc*100:.2f}%  ({correct}/{len(y)})')
    return acc

# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    banner('STAGE 5 TFLite FIX — Stable Export Pipeline')
    t0 = time.time()

    # GPU setup
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── 1. Load Keras model ───────────────────────────────────────────────────
    section('1. Loading Keras Model')
    if not os.path.exists(KERAS_PATH):
        print(f'  ERROR: {KERAS_PATH} not found.')
        sys.exit(1)
    model = keras.models.load_model(KERAS_PATH)
    print(f'  Parameters : {model.count_params():,}')
    print(f'  Input  shape: {model.input_shape}')
    print(f'  Output shape: {model.output_shape}')

    # Force every layer to non-trainable (inference only)
    model.trainable = False
    for layer in model.layers:
        layer.trainable = False

    # ── 2. BatchNorm diagnostics ──────────────────────────────────────────────
    debug_bn(model)

    # ── 3. Load sample data ───────────────────────────────────────────────────
    section('2. Loading Sample Data')
    data_path = os.path.join(DATA_DIR, 'cifar100_tree_data.npz')
    if not os.path.exists(data_path):
        print('  WARNING: data file missing — using random noise for checks.')
        x_sample = np.random.rand(10, 32, 32, 3).astype(np.float32)
        y_sample  = None
        x_full = x_sample
        y_full = None
    else:
        data     = np.load(data_path)
        x_full   = data['x_test_clean']
        y_full   = data['y_test']
        cls_ids  = np.argmax(y_full, axis=1)
        
        # Ensure we have a balanced 10-sample batch (5 class 0, 5 class 1)
        idx_0 = np.where(cls_ids == 0)[0][:5]
        idx_1 = np.where(cls_ids == 1)[0][:5]
        idx_sample = np.concatenate([idx_0, idx_1])
        
        x_sample = x_full[idx_sample]
        y_sample = y_full[idx_sample]
        
        print(f'  Test set  : {x_full.shape}')
        print(f'  Not-tree  : {np.sum(cls_ids==0)}   Tree: {np.sum(cls_ids==1)}')
        print(f'  Pixel range: {x_sample.min():.4f} – {x_sample.max():.4f}')

    # ── 4. Rebuild clean inference graph ─────────────────────────────────────
    section('3. Rebuilding Inference-Only Graph (training=False)')
    print('  Strategy: keras.Input → model(inputs, training=False) → keras.Model')
    inp = keras.Input(shape=(32, 32, 3), name='image_input')
    out = model(inp, training=False)
    inference_model = keras.Model(inp, out, name='LiteRob_Inference')
    print(f'  Inference model output: {inference_model.output_shape}')

    # ── 5. Pre-conversion Keras sanity check ──────────────────────────────────
    section('4. Pre-Conversion Keras Sanity Check (10 samples)')
    kp = inference_model.predict(x_sample, batch_size=10, verbose=0)
    print('  Keras predictions (first 5):')
    for i, p in enumerate(kp[:5]):
        print(f'    [{i}] {p}  → class {np.argmax(p)}')
    print(f'  Preprocessing path: [0,1] → Resizing(64) → Rescaling(×2,-1) → [-1,1]')
    check_sanity(kp, 'Keras-Inference')

    # ── 6. FP32 TFLite conversion ─────────────────────────────────────────────
    section('5. Pure FP32 TFLite Conversion')
    print('  converter = TFLiteConverter.from_keras_model(inference_model)')
    print('  No optimizations · No float16 · No int8')

    def _do_convert(conv):
        try:
            return conv.convert()
        except Exception as e:
            print(f'  Conversion error: {e}')
            print('  Retrying with SELECT_TF_OPS fallback...')
            conv.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            return conv.convert()

    conv_fp32 = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    conv_fp32.optimizations           = []
    conv_fp32.experimental_new_converter = True
    conv_fp32.experimental_new_quantizer = True
    buf_fp32 = _do_convert(conv_fp32)

    with open(FP32_PATH, 'wb') as f:
        f.write(buf_fp32)
    fp32_mb = os.path.getsize(FP32_PATH) / 1024 / 1024
    print(f'  Saved : {FP32_PATH}')
    print(f'  Size  : {fp32_mb:.2f} MB')
    if fp32_mb < 1.0:
        print(f'  CRITICAL: Only {fp32_mb:.2f} MB — MobileNetV2 expects ~9–14 MB.')
        print('  The graph may not contain model weights.')

    # ── 7. Post-conversion FP32 verification ─────────────────────────────────
    section('6. Post-Conversion FP32 Verification (10 samples)')
    tp_fp32 = tflite_predict(FP32_PATH, x_sample)
    print('\n  TFLite FP32 predictions (first 5):')
    for i, p in enumerate(tp_fp32[:5]):
        print(f'    [{i}] {p}  → class {np.argmax(p)}')
    diff = float(np.max(np.abs(kp - tp_fp32)))
    print(f'\n  Max |Keras − TFLite FP32| = {diff:.8f}')
    if diff > 0.01:
        print('  WARNING: Large prediction gap between Keras and TFLite!')
    else:
        print('  ✓ Predictions match closely.')
    try:
        check_sanity(tp_fp32, 'TFLite-FP32')
        fp32_ok = True
    except ValueError as e:
        print(f'\n  FP32 REJECTED: {e}')
        fp32_ok = False

    # ── 8. FP32 full accuracy ─────────────────────────────────────────────────
    fp32_acc = None
    keras_acc = None
    if fp32_ok and y_full is not None:
        section('7. FP32 Full Test-Set Accuracy')
        fp32_acc  = tflite_accuracy(FP32_PATH, x_full, y_full, 'TFLite FP32')
        kp_full   = inference_model.predict(x_full, batch_size=BATCH_SIZE, verbose=0)
        keras_acc = float(np.mean(
            np.argmax(kp_full, axis=1) == np.argmax(y_full, axis=1)))
        print(f'  Keras reference : {keras_acc*100:.2f}%')
        print(f'  Accuracy gap    : {(keras_acc - fp32_acc)*100:.2f}%')

    # ── 9. Optional float16 ───────────────────────────────────────────────────
    f16_acc  = None
    f16_mb   = None
    f16_ok   = False
    if not fp32_ok:
        print('\n  Skipping float16 — FP32 failed. Fix FP32 first.')
    else:
        section('8. Float16 TFLite Conversion (Optional)')
        try:
            conv16 = tf.lite.TFLiteConverter.from_keras_model(inference_model)
            conv16.optimizations              = [tf.lite.Optimize.DEFAULT]
            conv16.target_spec.supported_types = [tf.float16]
            conv16.experimental_new_converter  = True
            conv16.experimental_new_quantizer  = True
            buf16 = _do_convert(conv16)

            with open(F16_PATH, 'wb') as f:
                f.write(buf16)
            f16_mb = os.path.getsize(F16_PATH) / 1024 / 1024
            print(f'  Saved : {F16_PATH}')
            print(f'  Size  : {f16_mb:.2f} MB')

            section('9. Float16 Verification (10 samples)')
            tp_f16 = tflite_predict(F16_PATH, x_sample)
            print('  Float16 predictions (first 5):')
            for i, p in enumerate(tp_f16[:5]):
                print(f'    [{i}] {p}  → class {np.argmax(p)}')
            try:
                check_sanity(tp_f16, 'TFLite-Float16')
                f16_ok = True
                if y_full is not None:
                    section('10. Float16 Full Test-Set Accuracy')
                    f16_acc = tflite_accuracy(F16_PATH, x_full, y_full, 'TFLite Float16')
            except ValueError as e:
                print(f'  Float16 REJECTED: {e}')
                os.remove(F16_PATH)
                print(f'  Deleted broken file: {F16_PATH}')
        except Exception as e:
            print(f'  Float16 conversion failed: {e}')

    # ── 10. Deployment report ─────────────────────────────────────────────────
    banner('DEPLOYMENT REPORT')
    elapsed = (time.time() - t0) / 60

    print(f'\n  {"Model":<30} {"Size":>8}   {"Accuracy":>10}   Status')
    print(f'  {"-"*70}')
    if keras_acc is not None:
        print(f'  {"Keras LiteRob":<30} {"—":>8}   {keras_acc*100:>9.2f}%   reference')
    fp32_status = '✓ SAFE' if fp32_ok else '✗ BROKEN'
    fp32_acc_s  = f'{fp32_acc*100:.2f}%' if fp32_acc is not None else '—'
    print(f'  {"FP32 TFLite":<30} {fp32_mb:>7.2f}MB   {fp32_acc_s:>10}   {fp32_status}')
    if f16_mb is not None:
        f16_status = '✓ SAFE' if f16_ok else '✗ BROKEN'
        f16_acc_s  = f'{f16_acc*100:.2f}%' if f16_acc is not None else '—'
        print(f'  {"Float16 TFLite":<30} {f16_mb:>7.2f}MB   {f16_acc_s:>10}   {f16_status}')

    print(f'\n  Deployment safe : {"YES ✓" if fp32_ok else "NO ✗"}')
    print(f'  Elapsed         : {elapsed:.1f} min')

    if fp32_ok:
        print('\n  ✓ literob_fp32.tflite is numerically stable.')
        print('  ✓ Ready for mobile deployment.')
        if f16_ok:
            print('  ✓ literob_float16_fixed.tflite also validated.')
    else:
        print('\n  ✗ Export FAILED. Do not deploy. Check model weights.')
        sys.exit(1)

if __name__ == '__main__':
    run()
