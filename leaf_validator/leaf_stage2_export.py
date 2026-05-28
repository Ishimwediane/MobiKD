"""
leaf_stage2_export.py
=====================
TFLite export + validation for the Leaf Validator model.

Run after leaf_stage1_train.py:
    python leaf_stage2_export.py

Saves:
    models/kd_mobilenetv2_leaf_validator_fp32.tflite
    models/kd_mobilenetv2_leaf_validator_float16.tflite

Performs:
    - Keras sanity check (NaN, Inf, class diversity, sum-to-1)
    - FP32 TFLite export + full accuracy on test set
    - Float16 TFLite export + full accuracy on test set
    - Numerical comparison: |Keras - TFLite| max diff
    - Deployment report printed to console
"""

import os, sys, time
os.environ['TF_GPU_ALLOCATOR']          = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL']      = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras

from leaf_validator_config import (
    setup_logging, setup_gpu, banner, section,
    LEAF_KERAS_PATH, LEAF_FP32_PATH, LEAF_F16_PATH,
    RESULTS_DIR, MODELS_DIR, IMAGE_SIZE, BATCH_SIZE,
    CLASS_NAMES, NUM_CLASSES
)

logger = setup_logging()

# ============================================================
# UTILITIES
# ============================================================
def check_sanity(preds, label):
    nan_n  = int(np.sum(np.isnan(preds)))
    inf_n  = int(np.sum(np.isinf(preds)))
    sums   = np.sum(preds, axis=1)
    sum_ok = bool(np.allclose(sums, 1.0, atol=1e-3))
    n_cls  = len(np.unique(np.argmax(preds, axis=1)))
    print(f'  [{label}] NaN={nan_n}  Inf={inf_n}  '
          f'SumTo1={sum_ok}  ClassesHit={n_cls}/{NUM_CLASSES}')
    print(f'  [{label}] prob-sum  min={sums.min():.6f}  max={sums.max():.6f}')
    print(f'  [{label}] output    min={preds.min():.6f}  max={preds.max():.6f}')
    if nan_n > 0:
        raise ValueError(f'FAIL [{label}]: {nan_n} NaN(s) detected!')
    if inf_n > 0:
        raise ValueError(f'FAIL [{label}]: {inf_n} Inf(s) detected!')
    if not sum_ok:
        raise ValueError(f'FAIL [{label}]: probs do not sum to 1.0')
    if n_cls < NUM_CLASSES:
        raise ValueError(f'FAIL [{label}]: only {n_cls} class(es) predicted — model collapsed!')
    print(f'  [{label}] ✓ Sanity PASSED.')


def debug_bn(model):
    section('BatchNorm Diagnostics')
    count = 0
    def _inspect(lyr):
        nonlocal count
        if isinstance(lyr, keras.layers.BatchNormalization):
            mu  = lyr.moving_mean.numpy()
            var = lyr.moving_variance.numpy()
            bad = np.any(var <= 0)
            print(f'  BN {lyr.name:40s} '
                  f'mu=[{mu.min():.3f},{mu.max():.3f}] '
                  f'var=[{var.min():.5f},{var.max():.5f}]'
                  + (' ← VAR≤0 WARNING' if bad else ''))
            count += 1
    for layer in model.layers:
        _inspect(layer)
        if hasattr(layer, 'layers'):
            for sub in layer.layers:
                _inspect(sub)
                if hasattr(sub, 'layers'):
                    for ssub in sub.layers:
                        _inspect(ssub)
    print(f'  Total BN layers inspected: {count}')


def tflite_predict_batch(path, x_samples):
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


def tflite_accuracy(path, x, y_true, name):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp_i = interp.get_input_details()[0]['index']
    out_i = interp.get_output_details()[0]['index']
    correct = 0
    preds   = []
    for img, lbl in zip(x, y_true):
        interp.set_tensor(inp_i, img[np.newaxis].astype(np.float32))
        interp.invoke()
        p = interp.get_tensor(out_i)[0].copy()
        preds.append(p)
        if np.argmax(p) == int(lbl):
            correct += 1
    acc = correct / len(y_true)
    print(f'  {name}: {acc*100:.2f}%  ({correct}/{len(y_true)})')
    return acc, np.array(preds)


def do_convert(converter):
    try:
        return converter.convert()
    except Exception as e:
        print(f'  Conversion error: {e}')
        print('  Retrying with SELECT_TF_OPS fallback…')
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        return converter.convert()


# ============================================================
# MAIN
# ============================================================
def run():
    t0 = time.time()
    banner('LEAF VALIDATOR — TFLite Export & Validation')
    setup_gpu()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Load Keras model ───────────────────────────────────────────────────
    section('1. Loading Keras Model')
    if not os.path.exists(LEAF_KERAS_PATH):
        print(f'  ERROR: {LEAF_KERAS_PATH} not found.')
        print('  Run leaf_stage1_train.py first.')
        sys.exit(1)
    model = keras.models.load_model(LEAF_KERAS_PATH)
    model.trainable = False
    for layer in model.layers:
        layer.trainable = False
    print(f'  Params      : {model.count_params():,}')
    print(f'  Input shape : {model.input_shape}')
    print(f'  Output shape: {model.output_shape}')

    # ── 2. BN diagnostics ─────────────────────────────────────────────────────
    debug_bn(model)

    # ── 3. Load test data ─────────────────────────────────────────────────────
    section('2. Loading Test Data')
    test_npz = os.path.join(RESULTS_DIR, 'test_data.npz')
    if not os.path.exists(test_npz):
        print('  WARNING: test_data.npz not found — using 20 random samples.')
        X_test = np.random.rand(20, IMAGE_SIZE, IMAGE_SIZE, 3).astype(np.float32)
        y_test = np.random.randint(0, NUM_CLASSES, 20)
    else:
        d      = np.load(test_npz)
        X_test = d['X_test'].astype(np.float32)
        y_test = d['y_test'].astype(np.int32)
    print(f'  Test set shape : {X_test.shape}')
    print(f'  Pixel range    : [{X_test.min():.4f}, {X_test.max():.4f}]')
    print(f'  not_leaf={np.sum(y_test==0)}, leaf={np.sum(y_test==1)}')

    # Balanced 10-sample probe
    probe_idx = []
    for cls in range(NUM_CLASSES):
        idx = np.where(y_test == cls)[0][:5]
        probe_idx.extend(idx.tolist())
    probe_idx = np.array(probe_idx[:10])
    x_probe   = X_test[probe_idx]
    y_probe   = y_test[probe_idx]

    # ── 4. Build inference-only graph ─────────────────────────────────────────
    section('3. Building Inference-Only Graph (training=False)')
    inp_layer   = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='leaf_input')
    out_layer   = model(inp_layer, training=False)
    inf_model   = keras.Model(inp_layer, out_layer, name='LeafValidator_Inference')
    print(f'  Inference model output: {inf_model.output_shape}')

    # ── 5. Keras sanity check ─────────────────────────────────────────────────
    section('4. Pre-Conversion Keras Sanity Check')
    kp = inf_model.predict(x_probe, batch_size=len(x_probe), verbose=0)
    print('  Keras predictions (first 10):')
    for i, (p, lbl) in enumerate(zip(kp, y_probe)):
        correct = '✓' if np.argmax(p) == lbl else '✗'
        print(f'    [{i}] {p}  → {CLASS_NAMES[np.argmax(p)]}  GT={CLASS_NAMES[lbl]}  {correct}')
    check_sanity(kp, 'Keras-Inference')

    # ── 6. FP32 TFLite conversion ─────────────────────────────────────────────
    section('5. FP32 TFLite Conversion')
    conv_fp32 = tf.lite.TFLiteConverter.from_keras_model(inf_model)
    conv_fp32.optimizations                = []
    conv_fp32.experimental_new_converter   = True
    conv_fp32.experimental_new_quantizer   = True
    buf_fp32 = do_convert(conv_fp32)

    with open(LEAF_FP32_PATH, 'wb') as f:
        f.write(buf_fp32)
    fp32_mb = os.path.getsize(LEAF_FP32_PATH) / 1024 / 1024
    print(f'  Saved : {LEAF_FP32_PATH}')
    print(f'  Size  : {fp32_mb:.2f} MB')

    # ── 7. FP32 verification (probe) ──────────────────────────────────────────
    section('6. FP32 Verification — 10 Probe Samples')
    tp_fp32 = tflite_predict_batch(LEAF_FP32_PATH, x_probe)
    print('  TFLite FP32 predictions (first 10):')
    for i, p in enumerate(tp_fp32):
        print(f'    [{i}] {p}  → {CLASS_NAMES[np.argmax(p)]}')
    diff_fp32 = float(np.max(np.abs(kp - tp_fp32)))
    print(f'\n  Max |Keras − TFLite FP32| = {diff_fp32:.8f}')
    try:
        check_sanity(tp_fp32, 'TFLite-FP32')
        fp32_ok = True
    except ValueError as e:
        print(f'\n  FP32 REJECTED: {e}')
        fp32_ok = False

    # ── 8. FP32 full test accuracy ────────────────────────────────────────────
    fp32_acc = None
    keras_acc = None
    if fp32_ok:
        section('7. FP32 Full Test-Set Accuracy')
        fp32_acc, _  = tflite_accuracy(LEAF_FP32_PATH, X_test, y_test, 'TFLite FP32')
        kp_full      = inf_model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
        keras_acc    = float(np.mean(np.argmax(kp_full, axis=1) == y_test))
        print(f'  Keras reference : {keras_acc*100:.2f}%')
        print(f'  Accuracy gap    : {abs(keras_acc - fp32_acc)*100:.2f}%')

    # ── 9. Float16 conversion ─────────────────────────────────────────────────
    f16_acc  = None
    f16_mb   = None
    f16_ok   = False

    if not fp32_ok:
        print('\n  Skipping Float16 — FP32 failed first. Fix FP32 issues.')
    else:
        section('8. Float16 TFLite Conversion')
        try:
            conv16 = tf.lite.TFLiteConverter.from_keras_model(inf_model)
            conv16.optimizations               = [tf.lite.Optimize.DEFAULT]
            conv16.target_spec.supported_types = [tf.float16]
            conv16.experimental_new_converter  = True
            conv16.experimental_new_quantizer  = True
            buf16 = do_convert(conv16)

            with open(LEAF_F16_PATH, 'wb') as f:
                f.write(buf16)
            f16_mb = os.path.getsize(LEAF_F16_PATH) / 1024 / 1024
            print(f'  Saved : {LEAF_F16_PATH}')
            print(f'  Size  : {f16_mb:.2f} MB')

            section('9. Float16 Verification — Probe Samples')
            tp_f16 = tflite_predict_batch(LEAF_F16_PATH, x_probe)
            diff_f16 = float(np.max(np.abs(kp - tp_f16)))
            print(f'  Max |Keras − TFLite Float16| = {diff_f16:.8f}')
            try:
                check_sanity(tp_f16, 'TFLite-Float16')
                f16_ok = True
                section('10. Float16 Full Test-Set Accuracy')
                f16_acc, _ = tflite_accuracy(LEAF_F16_PATH, X_test, y_test, 'TFLite Float16')
            except ValueError as e:
                print(f'  Float16 REJECTED: {e}')
                os.remove(LEAF_F16_PATH)
                print(f'  Broken file removed: {LEAF_F16_PATH}')
        except Exception as e:
            print(f'  Float16 conversion failed: {e}')

    # ── 10. Deployment report ─────────────────────────────────────────────────
    banner('DEPLOYMENT REPORT')
    elapsed = (time.time() - t0) / 60
    print(f'\n  {"Model":<38} {"Size":>8}   {"Accuracy":>10}   Status')
    print(f'  {"-"*75}')
    if keras_acc is not None:
        print(f'  {"Keras LeafValidator":<38} {"—":>8}   {keras_acc*100:>9.2f}%   reference')
    fp32_status = '✓ SAFE' if fp32_ok else '✗ BROKEN'
    fp32_acc_s  = f'{fp32_acc*100:.2f}%' if fp32_acc else '—'
    fp32_size_s = f'{fp32_mb:.2f} MB' if fp32_mb else '—'
    print(f'  {"FP32 TFLite":<38} {fp32_size_s:>8}   {fp32_acc_s:>10}   {fp32_status}')
    if f16_mb:
        f16_status = '✓ SAFE' if f16_ok else '✗ BROKEN'
        f16_acc_s  = f'{f16_acc*100:.2f}%' if f16_acc else '—'
        f16_size_s = f'{f16_mb:.2f} MB'
        mobile_ok  = '✓ < 5 MB' if f16_mb < 5.0 else f'✗ {f16_mb:.1f} MB > 5 MB'
        print(f'  {"Float16 TFLite":<38} {f16_size_s:>8}   {f16_acc_s:>10}   {f16_status}')
        print(f'  Mobile size target (<5 MB): {mobile_ok}')

    print(f'\n  Deployment safe : {"YES ✓" if fp32_ok else "NO ✗"}')
    print(f'  Elapsed         : {elapsed:.1f} min')

    # Save export metrics
    export_meta = {
        'keras_accuracy'    : keras_acc,
        'fp32_accuracy'     : fp32_acc,
        'fp32_size_mb'      : fp32_mb,
        'fp32_ok'           : fp32_ok,
        'f16_accuracy'      : f16_acc,
        'f16_size_mb'       : f16_mb,
        'f16_ok'            : f16_ok,
        'elapsed_min'       : round(elapsed, 2),
    }
    import json
    with open(os.path.join(RESULTS_DIR, 'export_metrics.json'), 'w') as jf:
        json.dump(export_meta, jf, indent=2)

    if fp32_ok:
        print('\n  ✓ FP32 TFLite validated and saved.')
        if f16_ok:
            print('  ✓ Float16 TFLite validated and saved.')
            print('  ✓ Ready for mobile deployment.')
        print('\n  Next: python leaf_stage3_evaluate.py')
    else:
        print('\n  ✗ Export FAILED. Do not deploy.')
        sys.exit(1)


if __name__ == '__main__':
    run()
