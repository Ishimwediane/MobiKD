"""
leaf_stage3_evaluate.py
=======================
Full robustness evaluation + comprehensive visualization suite
for the Leaf Validator model.

Run after leaf_stage2_export.py:
    python leaf_stage3_evaluate.py

Generates plots → plots/leaf/
Saves metrics  → results/leaf/
"""

import os, sys, json, warnings
os.environ['TF_GPU_ALLOCATOR']          = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL']      = '2'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, balanced_accuracy_score
)

import tensorflow as tf
from tensorflow import keras

from leaf_validator_config import (
    setup_logging, setup_gpu, banner, section,
    LEAF_KERAS_PATH, LEAF_FP32_PATH, LEAF_F16_PATH,
    RESULTS_DIR, PLOTS_DIR, MODELS_DIR,
    IMAGE_SIZE, BATCH_SIZE, CLASS_NAMES, NUM_CLASSES,
    apply_blur, apply_noise, apply_low_light, apply_combined, degrade_set
)

logger = setup_logging()

# ── Plot style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'  : 'DejaVu Sans',
    'font.size'    : 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi'   : 150,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
})

PALETTE = {
    'Clean'    : '#4CAF50',
    'Blur'     : '#2196F3',
    'Noise'    : '#FF9800',
    'Low Light': '#9C27B0',
    'Combined' : '#F44336',
}
TFLITE_PALETTE = {
    'Keras'   : '#1976D2',
    'FP32'    : '#388E3C',
    'Float16' : '#F57C00',
}

# ============================================================
# UTILITIES
# ============================================================
def tflite_predict_full(path, x):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp_i = interp.get_input_details()[0]['index']
    out_i = interp.get_output_details()[0]['index']
    preds = []
    for img in x:
        interp.set_tensor(inp_i, img[np.newaxis].astype(np.float32))
        interp.invoke()
        preds.append(interp.get_tensor(out_i)[0].copy())
    return np.array(preds, dtype=np.float32)


def get_accuracy(preds, y_true):
    return float(np.mean(np.argmax(preds, axis=1) == y_true))


def evaluate_robustness(model_fn, X_test, y_test, label):
    """Run model on all corruption variants, return dict of accuracies."""
    sets = {
        'Clean'     : X_test,
        'Blur'      : degrade_set(X_test, apply_blur,       'Blur'),
        'Noise'     : degrade_set(X_test, apply_noise,      'Noise'),
        'Low Light' : degrade_set(X_test, apply_low_light,  'Low-Light'),
        'Combined'  : degrade_set(X_test, apply_combined,   'Combined'),
    }
    results = {}
    for name, x in sets.items():
        preds = model_fn(x)
        acc   = get_accuracy(preds, y_test)
        results[name] = acc
        print(f'  [{label}] {name:12s}: {acc*100:.2f}%')
    return results


# ============================================================
# PLOT FUNCTIONS
# ============================================================
def save_fig(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  Saved: plots/leaf/{name}')


def plot_training_curves(p1_csv, p2_csv):
    """Plot loss and accuracy for both training phases."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Leaf Validator — Training Curves', fontsize=15, fontweight='bold')

    colors_p1 = ('#1976D2', '#FF5722')
    colors_p2 = ('#388E3C', '#9C27B0')

    for ax_idx, (csv_path, phase, colors) in enumerate([
        (p1_csv, 'Phase 1 (Head Only)', colors_p1),
        (p2_csv, 'Phase 2 (Fine-Tune)', colors_p2),
    ]):
        ax_loss = axes[ax_idx]
        if not os.path.exists(csv_path):
            ax_loss.text(0.5, 0.5, f'{phase}\nlog not found', ha='center', va='center')
            continue
        df   = pd.read_csv(csv_path)
        ep   = df['epoch'] if 'epoch' in df.columns else range(1, len(df)+1)
        loss_col = [c for c in df.columns if 'loss' in c and 'val' not in c][0]
        acc_col  = [c for c in df.columns if 'accuracy' in c and 'val' not in c][0]
        vl_col   = [c for c in df.columns if 'val_loss' in c]
        va_col   = [c for c in df.columns if 'val_accuracy' in c]

        ax2 = ax_loss.twinx()
        ax_loss.plot(ep, df[loss_col], color=colors[0], lw=2, label='Train Loss')
        if vl_col:
            ax_loss.plot(ep, df[vl_col[0]], '--', color=colors[0], alpha=0.6, lw=2, label='Val Loss')
        ax2.plot(ep, df[acc_col]*100, color=colors[1], lw=2, label='Train Acc')
        if va_col:
            ax2.plot(ep, df[va_col[0]]*100, '--', color=colors[1], alpha=0.6, lw=2, label='Val Acc')

        ax_loss.set_title(phase)
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss', color=colors[0])
        ax2.set_ylabel('Accuracy (%)', color=colors[1])
        lines1, labels1 = ax_loss.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax_loss.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
        ax_loss.tick_params(axis='y', labelcolor=colors[0])
        ax2.tick_params(axis='y', labelcolor=colors[1])

    plt.tight_layout()
    save_fig(fig, 'training_curves.png')


def plot_confusion_matrices(y_true, y_pred, title_suffix='Keras'):
    """Plot raw + normalized confusion matrices side by side."""
    cm     = confusion_matrix(y_true, y_pred)
    cm_n   = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Confusion Matrices — {title_suffix}', fontsize=14, fontweight='bold')

    for ax, matrix, title, fmt in [
        (axes[0], cm,   'Raw Counts',  'd'),
        (axes[1], cm_n, 'Normalized',  '.2f'),
    ]:
        cmap = sns.color_palette('Blues', as_cmap=True)
        sns.heatmap(
            matrix, annot=True, fmt=fmt, cmap=cmap,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, linecolor='white',
            ax=ax, cbar=True
        )
        ax.set_title(title, pad=10)
        ax.set_xlabel('Predicted', labelpad=8)
        ax.set_ylabel('True',      labelpad=8)
        ax.tick_params(axis='x', rotation=30)
        ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    save_fig(fig, f'confusion_matrix_{title_suffix.lower().replace(" ", "_")}.png')


def plot_roc_curve(y_true, y_score, model_name='Keras'):
    """ROC curve for binary classification."""
    fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#1976D2', lw=2.5,
            label=f'{model_name} (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.08, color='#1976D2')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve — Leaf Validator ({model_name})', fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, 'roc_curve.png')
    return roc_auc


def plot_precision_recall(y_true, y_score, model_name='Keras'):
    """Precision-Recall curve."""
    prec, rec, _ = precision_recall_curve(y_true, y_score[:, 1])
    ap           = average_precision_score(y_true, y_score[:, 1])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, color='#388E3C', lw=2.5,
            label=f'{model_name} (AP = {ap:.4f})')
    ax.fill_between(rec, prec, alpha=0.08, color='#388E3C')
    baseline = np.sum(y_true == 1) / len(y_true)
    ax.axhline(baseline, ls='--', color='gray', lw=1.2,
               label=f'Random baseline ({baseline:.2f})')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve — Leaf Validator ({model_name})', fontweight='bold')
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, 'precision_recall_curve.png')
    return ap


def plot_robustness_bars(results_dict):
    """Bar chart comparing robustness of Keras, FP32, Float16 across conditions."""
    conditions = ['Clean', 'Blur', 'Noise', 'Low Light', 'Combined']
    models     = list(results_dict.keys())
    x          = np.arange(len(conditions))
    width      = 0.25
    offsets    = np.linspace(-(len(models)-1)*width/2, (len(models)-1)*width/2, len(models))

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (model_name, accs) in enumerate(results_dict.items()):
        vals = [accs.get(c, 0) * 100 for c in conditions]
        bars = ax.bar(x + offsets[i], vals, width,
                      label=model_name, color=list(TFLITE_PALETTE.values())[i % 3],
                      alpha=0.88, edgecolor='white', linewidth=0.7)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_xlabel('Corruption Type')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Robustness Comparison — Keras vs FP32 vs Float16', fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend(framealpha=0.9)
    ax.set_ylim(0, 115)
    ax.axhline(95, ls='--', color='red', lw=1.2, alpha=0.6, label='95% Target')
    ax.grid(axis='y', alpha=0.3, zorder=0)
    plt.tight_layout()
    save_fig(fig, 'robustness_comparison.png')


def plot_accuracy_comparison(results_dict):
    """Grouped bar chart: Clean accuracy for each model."""
    clean_accs = {m: results_dict[m].get('Clean', 0)*100 for m in results_dict}
    names = list(clean_accs.keys())
    vals  = list(clean_accs.values())
    colors = [list(TFLITE_PALETTE.values())[i % 3] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, vals, color=colors, width=0.5,
                  edgecolor='white', linewidth=1.0, alpha=0.9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{v:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.axhline(95, ls='--', color='red', lw=1.5, alpha=0.7, label='95% Target')
    ax.set_ylabel('Clean Accuracy (%)')
    ax.set_title('Clean Accuracy — Model Comparison', fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis='y', alpha=0.3, zorder=0)
    plt.tight_layout()
    save_fig(fig, 'accuracy_comparison.png')


def plot_metrics_bar(y_true, y_pred, model_name='Keras'):
    """Bar chart of per-class and macro metrics."""
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    macro  = report['macro avg']

    metric_names = ['Precision', 'Recall', 'F1-Score']
    not_leaf_v   = [report['not_leaf'][k.lower()] for k in ['precision', 'recall', 'f1-score']]
    leaf_v       = [report['leaf'][k.lower()]     for k in ['precision', 'recall', 'f1-score']]
    macro_v      = [macro[k.lower()]              for k in ['precision', 'recall', 'f1-score']]

    x      = np.arange(len(metric_names))
    width  = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, not_leaf_v, width, label='not_leaf', color='#1976D2', alpha=0.88)
    ax.bar(x,         leaf_v,     width, label='leaf',     color='#388E3C', alpha=0.88)
    ax.bar(x + width, macro_v,    width, label='Macro Avg', color='#F57C00', alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel('Score')
    ax.set_title(f'Per-Class Metrics — {model_name}', fontweight='bold')
    ax.set_ylim(0, 1.12)
    ax.legend(framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    for rects in ax.containers:
        ax.bar_label(rects, fmt='%.2f', fontsize=8.5, padding=2)
    plt.tight_layout()
    save_fig(fig, 'metrics_bar.png')


def plot_robustness_heatmap(results_dict):
    """Heatmap: models × corruption conditions."""
    conditions = ['Clean', 'Blur', 'Noise', 'Low Light', 'Combined']
    models     = list(results_dict.keys())
    matrix     = np.array([
        [results_dict[m].get(c, 0) * 100 for c in conditions]
        for m in models
    ])

    fig, ax = plt.subplots(figsize=(10, 4))
    cmap = LinearSegmentedColormap.from_list(
        'leaf_heatmap', ['#f44336', '#ffeb3b', '#4caf50'], N=256
    )
    im = ax.imshow(matrix, cmap=cmap, vmin=50, vmax=100, aspect='auto')
    plt.colorbar(im, ax=ax, label='Accuracy (%)')

    ax.set_xticks(range(len(conditions)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels(conditions, rotation=30, ha='right')
    ax.set_yticklabels(models)
    ax.set_title('Robustness Heatmap — Leaf Validator', fontweight='bold', pad=10)

    for i in range(len(models)):
        for j in range(len(conditions)):
            v = matrix[i, j]
            ax.text(j, i, f'{v:.1f}%', ha='center', va='center',
                    color='white' if v < 75 else 'black', fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_fig(fig, 'robustness_heatmap.png')


def plot_degradation_grid(X_test, n=4):
    """Visual grid showing sample images under each corruption."""
    fig, axes = plt.subplots(n, 5, figsize=(14, 3.5 * n))
    titles = ['Clean', 'Blur', 'Noise', 'Low Light', 'Combined']
    funcs  = [
        lambda x: x,
        apply_blur, apply_noise, apply_low_light, apply_combined
    ]
    idxs = np.random.choice(len(X_test), n, replace=False)
    for row, idx in enumerate(idxs):
        img = X_test[idx]
        for col, (title, fn) in enumerate(zip(titles, funcs)):
            degraded = fn(img.copy())
            axes[row, col].imshow(np.clip(degraded, 0, 1))
            axes[row, col].axis('off')
            if row == 0:
                axes[row, col].set_title(title, fontsize=11, fontweight='bold')
    fig.suptitle('Sample Images Under Corruption Conditions', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'degradation_grid.png')


# ============================================================
# MAIN
# ============================================================
def run():
    banner('LEAF VALIDATOR — Evaluation & Visualization')
    setup_gpu()
    os.makedirs(PLOTS_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Load test data ─────────────────────────────────────────────────────
    section('1. Loading Test Data')
    test_npz = os.path.join(RESULTS_DIR, 'test_data.npz')
    if not os.path.exists(test_npz):
        print('  ERROR: test_data.npz not found. Run leaf_stage1_train.py first.')
        sys.exit(1)
    d      = np.load(test_npz)
    X_test = d['X_test'].astype(np.float32)
    y_test = d['y_test'].astype(np.int32)
    print(f'  Test shape : {X_test.shape}')
    print(f'  not_leaf={np.sum(y_test==0)}, leaf={np.sum(y_test==1)}')

    # ── 2. Load models ────────────────────────────────────────────────────────
    section('2. Loading Models')
    available = {}

    if os.path.exists(LEAF_KERAS_PATH):
        print(f'  Loading Keras model…')
        keras_model = keras.models.load_model(LEAF_KERAS_PATH)
        keras_model.trainable = False
        inp_l = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
        out_l = keras_model(inp_l, training=False)
        inf_m = keras.Model(inp_l, out_l)
        available['Keras'] = lambda x: inf_m.predict(x, batch_size=BATCH_SIZE, verbose=0)
    else:
        print(f'  WARNING: Keras model not found at {LEAF_KERAS_PATH}')

    if os.path.exists(LEAF_FP32_PATH):
        print(f'  FP32 TFLite found.')
        available['FP32'] = lambda x: tflite_predict_full(LEAF_FP32_PATH, x)
    if os.path.exists(LEAF_F16_PATH):
        print(f'  Float16 TFLite found.')
        available['Float16'] = lambda x: tflite_predict_full(LEAF_F16_PATH, x)

    if not available:
        print('  ERROR: No models found. Run training and export stages first.')
        sys.exit(1)

    # ── 3. Training curves ────────────────────────────────────────────────────
    section('3. Training Curves')
    p1_csv = os.path.join(RESULTS_DIR, 'phase1_log.csv')
    p2_csv = os.path.join(RESULTS_DIR, 'phase2_log.csv')
    plot_training_curves(p1_csv, p2_csv)

    # ── 4. Robustness evaluation ──────────────────────────────────────────────
    section('4. Robustness Evaluation')
    all_results = {}
    all_preds   = {}

    for model_name, model_fn in available.items():
        print(f'\n  Evaluating: {model_name}')
        preds  = model_fn(X_test)
        all_preds[model_name] = preds
        acc_clean = get_accuracy(preds, y_test)
        print(f'  Clean accuracy: {acc_clean*100:.2f}%')

        rob = {'Clean': acc_clean}
        for corruption, fn, label in [
            ('Blur',      apply_blur,      'Blur'),
            ('Noise',     apply_noise,     'Noise'),
            ('Low Light', apply_low_light, 'Low-Light'),
            ('Combined',  apply_combined,  'Combined'),
        ]:
            x_deg  = degrade_set(X_test, fn, label)
            p_deg  = model_fn(x_deg)
            rob[corruption] = get_accuracy(p_deg, y_test)
            print(f'  {corruption:12s}: {rob[corruption]*100:.2f}%')
        all_results[model_name] = rob

    # ── 5. Generate plots ─────────────────────────────────────────────────────
    section('5. Generating Plots')

    # Confusion matrices
    primary = 'Keras' if 'Keras' in all_preds else list(all_preds.keys())[0]
    y_pred_primary = np.argmax(all_preds[primary], axis=1)
    plot_confusion_matrices(y_test, y_pred_primary, title_suffix=primary)

    # ROC curve
    if all_preds[primary].shape[1] == 2:
        roc_auc = plot_roc_curve(y_test, all_preds[primary], primary)
        ap      = plot_precision_recall(y_test, all_preds[primary], primary)
        print(f'  ROC AUC = {roc_auc:.4f}  |  Average Precision = {ap:.4f}')

    # Per-class metrics bar
    plot_metrics_bar(y_test, y_pred_primary, primary)

    # Robustness plots
    plot_robustness_bars(all_results)
    plot_accuracy_comparison(all_results)
    plot_robustness_heatmap(all_results)

    # Degradation visual grid
    plot_degradation_grid(X_test, n=4)

    # ── 6. Save all results ───────────────────────────────────────────────────
    section('6. Saving Results')

    # JSON with all robustness scores
    with open(os.path.join(RESULTS_DIR, 'robustness_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # CSV summary
    rows = []
    for model_name, rob in all_results.items():
        row = {'model': model_name}
        row.update({k: round(v*100, 2) for k, v in rob.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'robustness_summary.csv'), index=False)
    print('\n  Robustness results:')
    print(df.to_string(index=False))

    # Full classification report
    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred_primary, target_names=CLASS_NAMES)
    report_path = os.path.join(RESULTS_DIR, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f'Leaf Validator — Classification Report ({primary})\n')
        f.write('=' * 60 + '\n')
        f.write(report)
    print(f'\n  Classification Report ({primary}):')
    print(report)

    # ── 7. Summary ────────────────────────────────────────────────────────────
    banner('EVALUATION COMPLETE')
    print(f'\n  Plots saved → plots/leaf/  ({len(os.listdir(PLOTS_DIR))} files)')
    print(f'  Results saved → results/leaf/')
    print('\n  Files generated:')
    for f in sorted(os.listdir(PLOTS_DIR)):
        print(f'    plots/leaf/{f}')
    print('\n  Next: test deployment with')
    print('    python deployment/inference_leaf_validator.py <image_path>')


if __name__ == '__main__':
    run()
