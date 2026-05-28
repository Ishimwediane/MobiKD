"""
STAGE 6 EVALUATE - TFLite Model Evaluation
============================================
Run:    python3 stage6_evaluate.py
Needs:  - data/cifar100_tree_data.npz
        - models/literob_mobilenet.keras
        - models/literob_fp32.tflite        (from stage5_tflite_fix.py)
        - models/literob_float16_fixed.tflite  (optional)
Saves:  - results/tflite_evaluation_report.csv
"""
import os, time, sys
os.environ['TF_CPP_MIN_LOG_LEVEL']      = '2'
os.environ['TF_GPU_ALLOCATOR']          = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import tensorflow as tf
from tensorflow import keras

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'results')

KERAS_PATH = os.path.join(MODELS_DIR, 'literob_mobilenet.keras')
FP32_PATH  = os.path.join(MODELS_DIR, 'literob_fp32.tflite')
F16_PATH   = os.path.join(MODELS_DIR, 'literob_float16_fixed.tflite')
DATA_PATH  = os.path.join(DATA_DIR,   'cifar100_tree_data.npz')

BATCH_SIZE = 32

# ── Degradations (mirrors config.py, no import dependency) ───────────────────
import cv2

def apply_blur(img):
    u8 = (img * 255).astype(np.uint8)
    return cv2.GaussianBlur(u8, (5, 5), 0).astype(np.float32) / 255.0

def apply_noise(img):
    return np.clip(img + np.random.normal(0, 0.15, img.shape).astype(np.float32), 0, 1)

def apply_low_light(img):
    return np.power(img, 2.5).astype(np.float32)

def apply_combined(img):
    return apply_low_light(apply_noise(apply_blur(img)))

def degrade(x, fn, desc):
    print(f'  Building {desc}...', end=' ', flush=True)
    out = np.array([fn(i) for i in x])
    print('done')
    return out

# ── Utilities ─────────────────────────────────────────────────────────────────
def banner(t):
    print('\n' + '='*65)
    print(f'  {t}')
    print('='*65)

def section(t):
    print(f'\n── {t} ──')

def predict_keras(model, x):
    preds = model.predict(x, batch_size=BATCH_SIZE, verbose=0)
    return np.argmax(preds, axis=1).astype(np.int32)

def predict_tflite(path, x):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp_i = interp.get_input_details()[0]['index']
    out_i = interp.get_output_details()[0]['index']
    results = []
    nan_count = 0
    for img in x:
        interp.set_tensor(inp_i, img[np.newaxis].astype(np.float32))
        interp.invoke()
        p = interp.get_tensor(out_i)[0]
        if np.any(np.isnan(p)) or np.any(np.isinf(p)):
            nan_count += 1
            results.append(-1)
        else:
            results.append(int(np.argmax(p)))
    if nan_count > 0:
        print(f'  WARNING: {nan_count} NaN/Inf predictions out of {len(x)}!')
    return np.array(results, dtype=np.int32)

def compute_metrics(y_true, y_pred, n_classes=2):
    """Return accuracy, balanced accuracy, per-class recall."""
    valid   = (y_pred != -1)
    if valid.sum() == 0:
        return 0.0, 0.0, [0.0] * n_classes
    acc = float(np.mean(y_pred[valid] == y_true[valid]))
    recalls = []
    for c in range(n_classes):
        mask = (y_true == c)
        if mask.sum() == 0:
            recalls.append(0.0)
        else:
            recalls.append(float(np.mean(y_pred[mask] == c)))
    bal_acc = float(np.mean(recalls))
    return acc, bal_acc, recalls

def evaluate_all_conditions(name, predict_fn, x_clean, y_true, conditions):
    """Run predict_fn on every condition, return dict of results."""
    results = {}
    for cond, x in conditions.items():
        y_pred = predict_fn(x)
        acc, bal, recalls = compute_metrics(y_true, y_pred)
        nan_n = int(np.sum(y_pred == -1))
        cls_n = len(np.unique(y_pred[y_pred != -1]))
        results[cond] = {
            'acc': acc, 'bal_acc': bal,
            'recall_not_tree': recalls[0], 'recall_tree': recalls[1],
            'nan_preds': nan_n, 'classes_hit': cls_n
        }
        flag = ' ← NaN!' if nan_n > 0 else ''
        flag += ' ← COLLAPSED' if cls_n < 2 and nan_n == 0 else ''
        print(f'    {cond:<12}  acc={acc*100:6.2f}%  '
              f'bal={bal*100:6.2f}%  '
              f'tree-recall={recalls[1]*100:6.2f}%  '
              f'NaNs={nan_n}{flag}')
    return results

# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    banner('STAGE 6 EVALUATE — TFLite Model Evaluation')
    t0 = time.time()

    # GPU (not needed for CPU inference, but set up anyway)
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Dependency checks ─────────────────────────────────────────────────────
    section('Checking Dependencies')
    missing = []
    for label, path in [
        ('Data file',    DATA_PATH),
        ('Keras model',  KERAS_PATH),
        ('FP32 TFLite',  FP32_PATH),
    ]:
        exists = os.path.exists(path)
        size   = f'{os.path.getsize(path)/1024/1024:.2f} MB' if exists else '—'
        mark   = '✓' if exists else '✗ MISSING'
        print(f'  {mark}  {label:<20}  {size}  {path}')
        if not exists:
            missing.append(label)

    f16_available = os.path.exists(F16_PATH)
    if f16_available:
        sz = os.path.getsize(F16_PATH) / 1024 / 1024
        print(f'  ✓  {"Float16 TFLite":<20}  {sz:.2f} MB  {F16_PATH}')
    else:
        print(f'  –  Float16 TFLite        not found (optional)')

    if missing:
        print(f'\n  ERROR: Missing required files: {missing}')
        print('  Run stage5_tflite_fix.py first.')
        sys.exit(1)

    # ── Load data ─────────────────────────────────────────────────────────────
    section('Loading Test Data')
    data    = np.load(DATA_PATH)
    x_clean = data['x_test_clean']
    y_oh    = data['y_test']
    y_true  = np.argmax(y_oh, axis=1).astype(np.int32)

    print(f'  Test samples : {len(x_clean)}')
    print(f'  Not-tree (0) : {np.sum(y_true==0)}')
    print(f'  Tree     (1) : {np.sum(y_true==1)}')
    print(f'  Pixel range  : {x_clean.min():.4f} – {x_clean.max():.4f}')

    # Build degraded sets
    np.random.seed(42)
    section('Building Degraded Test Sets')
    conditions = {
        'Clean'    : x_clean,
        'Blur'     : degrade(x_clean, apply_blur,      'blur'),
        'Noise'    : degrade(x_clean, apply_noise,     'noise'),
        'Low Light': degrade(x_clean, apply_low_light, 'low-light'),
        'Combined' : degrade(x_clean, apply_combined,  'combined'),
    }

    # ── Load Keras model ──────────────────────────────────────────────────────
    section('Loading Keras Model')
    model = keras.models.load_model(KERAS_PATH)
    model.trainable = False
    for layer in model.layers:
        layer.trainable = False
    # Rebuild inference graph
    inp = keras.Input(shape=(32, 32, 3))
    out = model(inp, training=False)
    inf_model = keras.Model(inp, out)
    keras_fn = lambda x: predict_keras(inf_model, x)
    print(f'  Loaded.  Params: {model.count_params():,}')

    # ── Evaluate all models ───────────────────────────────────────────────────
    all_results = {}

    section('Keras LiteRob Results')
    all_results['Keras'] = evaluate_all_conditions(
        'Keras', keras_fn, x_clean, y_true, conditions)

    section('FP32 TFLite Results')
    fp32_fn = lambda x: predict_tflite(FP32_PATH, x)
    all_results['FP32'] = evaluate_all_conditions(
        'FP32', fp32_fn, x_clean, y_true, conditions)

    if f16_available:
        section('Float16 TFLite Results')
        f16_fn = lambda x: predict_tflite(F16_PATH, x)
        all_results['Float16'] = evaluate_all_conditions(
            'Float16', f16_fn, x_clean, y_true, conditions)

    # ── Summary table ─────────────────────────────────────────────────────────
    banner('ACCURACY SUMMARY TABLE')
    cond_list = list(conditions.keys())
    model_list = list(all_results.keys())

    header = f'  {"Condition":<12}' + ''.join(f'  {m:>12}' for m in model_list)
    print(header)
    print('  ' + '-' * (12 + 14 * len(model_list)))
    for cond in cond_list:
        row = f'  {cond:<12}'
        for m in model_list:
            row += f'  {all_results[m][cond]["acc"]*100:>11.2f}%'
        print(row)

    print('\n  Balanced Accuracy:')
    for cond in cond_list:
        row = f'  {cond:<12}'
        for m in model_list:
            row += f'  {all_results[m][cond]["bal_acc"]*100:>11.2f}%'
        print(row)

    # ── Accuracy drop ─────────────────────────────────────────────────────────
    banner('ROBUSTNESS DROP (Clean − Worst Degraded)')
    for m in model_list:
        clean = all_results[m]['Clean']['acc']
        worst = min(all_results[m][c]['acc'] for c in cond_list if c != 'Clean')
        drop  = clean - worst
        print(f'  {m:<12}  clean={clean*100:.2f}%  '
              f'worst={worst*100:.2f}%  drop={drop*100:.2f}%')

    # ── NaN / collapse check ──────────────────────────────────────────────────
    section('Numerical Health Check')
    all_ok = True
    for m in model_list:
        for cond in cond_list:
            r = all_results[m][cond]
            issues = []
            if r['nan_preds'] > 0:
                issues.append(f"NaN={r['nan_preds']}")
            if r['classes_hit'] < 2:
                issues.append('COLLAPSED')
            if issues:
                print(f'  ✗ {m} / {cond}: {", ".join(issues)}')
                all_ok = False
    if all_ok:
        print('  ✓ No NaN, no collapse in any model / condition.')

    # ── Save CSV ──────────────────────────────────────────────────────────────
    section('Saving Report')
    csv_path = os.path.join(RESULTS_DIR, 'tflite_evaluation_report.csv')
    with open(csv_path, 'w') as f:
        cols = ['Model', 'Condition', 'Accuracy', 'Balanced_Accuracy',
                'Tree_Recall', 'NotTree_Recall', 'NaN_Preds', 'Classes_Hit']
        f.write(','.join(cols) + '\n')
        for m in model_list:
            for cond in cond_list:
                r = all_results[m][cond]
                f.write(f'{m},{cond},'
                        f'{r["acc"]:.6f},{r["bal_acc"]:.6f},'
                        f'{r["recall_tree"]:.6f},{r["recall_not_tree"]:.6f},'
                        f'{r["nan_preds"]},{r["classes_hit"]}\n')
    print(f'  Saved: {csv_path}')

    # ── Model sizes ───────────────────────────────────────────────────────────
    section('Model Size Summary')
    size_map = {
        'Keras (.keras)' : KERAS_PATH,
        'FP32 TFLite'    : FP32_PATH,
        'Float16 TFLite' : F16_PATH,
    }
    for label, path in size_map.items():
        if os.path.exists(path):
            mb = os.path.getsize(path) / 1024 / 1024
            note = '  ← under 10MB ✓' if mb < 10 else ''
            print(f'  {label:<20}  {mb:>7.2f} MB{note}')

    # ── Final verdict ─────────────────────────────────────────────────────────
    banner('DEPLOYMENT VERDICT')
    elapsed = (time.time() - t0) / 60
    fp32_clean  = all_results['FP32']['Clean']['acc']
    keras_clean = all_results['Keras']['Clean']['acc']
    gap         = keras_clean - fp32_clean

    print(f'\n  Keras  clean accuracy : {keras_clean*100:.2f}%')
    print(f'  FP32   clean accuracy : {fp32_clean*100:.2f}%')
    print(f'  Accuracy gap          : {gap*100:.2f}%')
    print(f'  Numerical health      : {"✓ CLEAN" if all_ok else "✗ NaN/COLLAPSE DETECTED"}')
    print(f'  Deployment safe       : {"YES ✓" if all_ok and gap < 0.05 else "REVIEW NEEDED"}')
    print(f'\n  Evaluation time       : {elapsed:.1f} min')

    if not all_ok:
        print('\n  ACTION: Re-run stage5_tflite_fix.py — conversion is broken.')
        sys.exit(1)

if __name__ == '__main__':
    run()
