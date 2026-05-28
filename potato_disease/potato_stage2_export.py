"""
potato_stage2_export.py
=======================
TFLite export and validation for the Potato Disease Detector.

Run after potato_stage1_train.py:
    python potato_stage2_export.py

Saves:
    models/potato_disease/kd_mobilenetv2_potato_disease_fp32.tflite
    models/potato_disease/kd_mobilenetv2_potato_disease_float16.tflite
"""

import os, sys, time, json
os.environ['TF_GPU_ALLOCATOR']          = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL']      = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras

from potato_config import (
    setup_logging, setup_gpu, banner, section,
    POTATO_KERAS_PATH, POTATO_FP32_PATH, POTATO_F16_PATH,
    RESULTS_DIR, MODELS_DIR, IMAGE_SIZE, BATCH_SIZE,
    CLASS_NAMES, NUM_CLASSES
)

logger = setup_logging()


def check_sanity(preds, label):
    nan_n  = int(np.sum(np.isnan(preds)))
    inf_n  = int(np.sum(np.isinf(preds)))
    sums   = np.sum(preds, axis=1)
    sum_ok = bool(np.allclose(sums, 1.0, atol=1e-3))
    n_cls  = len(np.unique(np.argmax(preds, axis=1)))
    print(f'  [{label}] NaN={nan_n}  Inf={inf_n}  SumTo1={sum_ok}  ClassesHit={n_cls}/{NUM_CLASSES}')
    print(f'  [{label}] prob-sum  min={sums.min():.6f}  max={sums.max():.6f}')
    print(f'  [{label}] output    min={preds.min():.6f}  max={preds.max():.6f}')
    if nan_n > 0:
        raise ValueError(f'FAIL [{label}]: {nan_n} NaN(s) detected!')
    if inf_n > 0:
        raise ValueError(f'FAIL [{label}]: {inf_n} Inf(s) detected!')
    if not sum_ok:
        raise ValueError(f'FAIL [{label}]: probs do not sum to 1.0')
    if n_cls < NUM_CLASSES:
        raise ValueError(f'FAIL [{label}]: only {n_cls} class(es) predicted -- model collapsed!')
    print(f'  [{label}] Sanity PASSED.')


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
    correct, preds = 0, []
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
        print('  Retrying with SELECT_TF_OPS fallback...')
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        return converter.convert()


def run():
    t0 = time.time()
    banner('POTATO DISEASE DETECTOR -- TFLite Export & Validation')
    setup_gpu()
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load Keras model
    section('1. Loading Keras Model')
    if not os.path.exists(POTATO_KERAS_PATH):
        print(f'  ERROR: {POTATO_KERAS_PATH} not found.')
        print('  Run potato_stage1_train.py first.')
        sys.exit(1)
    model = keras.models.load_model(POTATO_KERAS_PATH)
    model.trainable = False
    for layer in model.layers:
        layer.trainable = False
    print(f'  Params      : {model.count_params():,}')
    print(f'  Input shape : {model.input_shape}')
    print(f'  Output shape: {model.output_shape}')

    # 2. Load test data
    section('2. Loading Test Data')
    test_npz = os.path.join(RESULTS_DIR, 'test_data.npz')
    if not os.path.exists(test_npz):
        print('  WARNING: test_data.npz not found -- using 30 random samples.')
        X_test = np.random.rand(30, IMAGE_SIZE, IMAGE_SIZE, 3).astype(np.float32)
        y_test = np.tile(np.arange(NUM_CLASSES), 10)
    else:
        d      = np.load(test_npz)
        X_test = d['X_test'].astype(np.float32)
        y_test = d['y_test'].astype(np.int32)
    print(f'  Test shape : {X_test.shape}')
    print(f'  ' + ' | '.join(f'{CLASS_NAMES[i]}={np.sum(y_test==i)}' for i in range(NUM_CLASSES)))

    # Balanced 15-sample probe (5 per class)
    probe_idx = []
    for cls in range(NUM_CLASSES):
        idx = np.where(y_test == cls)[0][:5]
        probe_idx.extend(idx.tolist())
    x_probe = X_test[probe_idx]
    y_probe = y_test[probe_idx]

    # 3. Build inference-only graph
    section('3. Building Inference-Only Graph')
    inp_layer = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='potato_input')
    out_layer = model(inp_layer, training=False)
    inf_model = keras.Model(inp_layer, out_layer, name='PotatoDetector_Inference')
    print(f'  Inference model output: {inf_model.output_shape}')

    # 4. Keras sanity check
    section('4. Pre-Conversion Keras Sanity Check')
    kp = inf_model.predict(x_probe, batch_size=len(x_probe), verbose=0)
    print('  Keras predictions (probe):')
    for i, (p, lbl) in enumerate(zip(kp, y_probe)):
        ok = 'OK' if np.argmax(p) == lbl else 'WRONG'
        print(f'    [{i}] {CLASS_NAMES[np.argmax(p)]:15s}  GT={CLASS_NAMES[lbl]:15s}  [{ok}]')
    check_sanity(kp, 'Keras-Inference')

    # 5. FP32 conversion
    section('5. FP32 TFLite Conversion')
    conv_fp32 = tf.lite.TFLiteConverter.from_keras_model(inf_model)
    conv_fp32.optimizations              = []
    conv_fp32.experimental_new_converter = True
    conv_fp32.experimental_new_quantizer = True
    buf_fp32 = do_convert(conv_fp32)
    with open(POTATO_FP32_PATH, 'wb') as f:
        f.write(buf_fp32)
    fp32_mb = os.path.getsize(POTATO_FP32_PATH) / 1024 / 1024
    print(f'  Saved: {POTATO_FP32_PATH}  ({fp32_mb:.2f} MB)')

    # 6. FP32 verification
    section('6. FP32 Verification')
    tp_fp32   = tflite_predict_batch(POTATO_FP32_PATH, x_probe)
    diff_fp32 = float(np.max(np.abs(kp - tp_fp32)))
    print(f'  Max |Keras - TFLite FP32| = {diff_fp32:.8f}')
    try:
        check_sanity(tp_fp32, 'TFLite-FP32')
        fp32_ok = True
    except ValueError as e:
        print(f'  FP32 REJECTED: {e}')
        fp32_ok = False

    # 7. FP32 full accuracy
    fp32_acc, keras_acc = None, None
    if fp32_ok:
        section('7. FP32 Full Test-Set Accuracy')
        fp32_acc, _ = tflite_accuracy(POTATO_FP32_PATH, X_test, y_test, 'TFLite FP32')
        kp_full     = inf_model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
        keras_acc   = float(np.mean(np.argmax(kp_full, axis=1) == y_test))
        print(f'  Keras reference : {keras_acc*100:.2f}%')
        print(f'  Accuracy gap    : {abs(keras_acc - fp32_acc)*100:.2f}%')

    # 8. Float16 conversion
    f16_acc, f16_mb, f16_ok = None, None, False
    if fp32_ok:
        section('8. Float16 TFLite Conversion')
        try:
            conv16 = tf.lite.TFLiteConverter.from_keras_model(inf_model)
            conv16.optimizations               = [tf.lite.Optimize.DEFAULT]
            conv16.target_spec.supported_types = [tf.float16]
            conv16.experimental_new_converter  = True
            conv16.experimental_new_quantizer  = True
            buf16 = do_convert(conv16)
            with open(POTATO_F16_PATH, 'wb') as f:
                f.write(buf16)
            f16_mb = os.path.getsize(POTATO_F16_PATH) / 1024 / 1024
            print(f'  Saved: {POTATO_F16_PATH}  ({f16_mb:.2f} MB)')

            section('9. Float16 Verification')
            tp_f16   = tflite_predict_batch(POTATO_F16_PATH, x_probe)
            diff_f16 = float(np.max(np.abs(kp - tp_f16)))
            print(f'  Max |Keras - TFLite Float16| = {diff_f16:.8f}')
            check_sanity(tp_f16, 'TFLite-Float16')
            f16_ok = True

            section('10. Float16 Full Test-Set Accuracy')
            f16_acc, _ = tflite_accuracy(POTATO_F16_PATH, X_test, y_test, 'TFLite Float16')
        except Exception as e:
            print(f'  Float16 failed: {e}')

    # Deployment report
    banner('DEPLOYMENT REPORT')
    elapsed = (time.time() - t0) / 60
    print(f'\n  {"Model":<40} {"Size":>8}   {"Accuracy":>10}   Status')
    print(f'  {"-"*70}')
    if keras_acc is not None:
        print(f'  {"Keras PotatoDiseaseDetector":<40} {"--":>8}   {keras_acc*100:>9.2f}%   reference')
    fp32_s = f'{fp32_mb:.2f} MB' if fp32_mb else '--'
    fp32_a = f'{fp32_acc*100:.2f}%' if fp32_acc else '--'
    print(f'  {"FP32 TFLite":<40} {fp32_s:>8}   {fp32_a:>10}   {"OK" if fp32_ok else "FAIL"}')
    if f16_mb:
        f16_s = f'{f16_mb:.2f} MB'
        f16_a = f'{f16_acc*100:.2f}%' if f16_acc else '--'
        mobile = 'OK < 5 MB' if f16_mb < 5.0 else f'LARGE {f16_mb:.1f} MB'
        print(f'  {"Float16 TFLite":<40} {f16_s:>8}   {f16_a:>10}   {"OK" if f16_ok else "FAIL"}')
        print(f'  Mobile size target (<5 MB): {mobile}')

    print(f'\n  Deployment safe : {"YES" if fp32_ok else "NO"}')
    print(f'  Elapsed         : {elapsed:.1f} min')

    export_meta = {
        'keras_accuracy': keras_acc, 'fp32_accuracy': fp32_acc,
        'fp32_size_mb': fp32_mb, 'fp32_ok': fp32_ok,
        'f16_accuracy': f16_acc, 'f16_size_mb': f16_mb, 'f16_ok': f16_ok,
        'elapsed_min': round(elapsed, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'export_metrics.json'), 'w') as jf:
        json.dump(export_meta, jf, indent=2)

    if fp32_ok:
        print('\n  FP32 TFLite validated and saved.')
        if f16_ok:
            print('  Float16 TFLite validated and saved.')
        print('\n  Next: python potato_stage3_evaluate.py')
    else:
        print('\n  Export FAILED.')
        sys.exit(1)


if __name__ == '__main__':
    run()
