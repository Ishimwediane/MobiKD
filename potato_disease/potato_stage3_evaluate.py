"""
potato_stage3_evaluate.py
=========================
Comprehensive evaluation for the Potato Disease Detector.

Runs robustness tests (Blur, Noise, Low Light) and generates presentation plots.
Ensure potato_stage1_train.py and potato_stage2_export.py have run first.

Plots saved to: plots/potato_disease/
"""

import os, sys, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras

from potato_config import (
    setup_logging, setup_gpu, banner, section,
    POTATO_KERAS_PATH, RESULTS_DIR, PLOTS_DIR, IMAGE_SIZE, BATCH_SIZE,
    CLASS_NAMES, NUM_CLASSES, CLASS_LABELS,
    apply_blur, apply_noise, apply_low_light, apply_combined, degrade_set
)

logger = setup_logging('PotatoEval')


# ── Plotting Helpers ──────────────────────────────────────────────────────────
def save_plot(name):
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {path}')

def set_style():
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.2)


# ── 1. Load Data & Model ──────────────────────────────────────────────────────
def run():
    banner('POTATO DISEASE DETECTOR -- Evaluation & Visualization')
    setup_gpu()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    set_style()

    # Load Model
    section('Loading Model & Data')
    if not os.path.exists(POTATO_KERAS_PATH):
        print(f'  ERROR: {POTATO_KERAS_PATH} not found.')
        sys.exit(1)
    model = keras.models.load_model(POTATO_KERAS_PATH)
    print(f'  Loaded model: {POTATO_KERAS_PATH}')

    # Load Test Data
    test_npz = os.path.join(RESULTS_DIR, 'test_data.npz')
    if not os.path.exists(test_npz):
        print(f'  ERROR: {test_npz} not found.')
        sys.exit(1)
    
    data = np.load(test_npz)
    X_test = data['X_test']
    y_test = data['y_test']
    print(f'  Test set loaded: {X_test.shape} images')

    # ── 2. Clean Baseline Evaluation ──────────────────────────────────────────
    section('Evaluating Baseline (Clean)')
    preds_clean = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    y_pred_clean = np.argmax(preds_clean, axis=1)

    cm = confusion_matrix(y_test, y_pred_clean)
    acc = np.mean(y_pred_clean == y_test)
    print(f'\n  Clean Accuracy: {acc*100:.2f}%')

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    labels = [CLASS_LABELS[i] for i in range(NUM_CLASSES)]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix (Clean Test Set)', pad=15)
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    save_plot('confusion_matrix_clean.png')

    # ── 3. Robustness Testing ─────────────────────────────────────────────────
    section('Generating Degraded Datasets')
    x_blur = degrade_set(X_test, apply_blur, 'Blur')
    x_noise = degrade_set(X_test, apply_noise, 'Noise')
    x_dark = degrade_set(X_test, apply_low_light, 'Low Light')
    x_comb = degrade_set(X_test, apply_combined, 'Combined')

    section('Evaluating Robustness')
    def eval_set(x, name):
        p = model.predict(x, batch_size=BATCH_SIZE, verbose=0)
        a = np.mean(np.argmax(p, axis=1) == y_test)
        print(f'  {name:15s} Accuracy: {a*100:>6.2f}%')
        return a

    rob_acc = {
        'Clean': acc,
        'Blur': eval_set(x_blur, 'Blur'),
        'Noise': eval_set(x_noise, 'Noise'),
        'Low Light': eval_set(x_dark, 'Low Light'),
        'Combined': eval_set(x_comb, 'Combined'),
    }

    # Plot Robustness Bar Chart
    plt.figure(figsize=(10, 6))
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#f1c40f', '#e74c3c']
    bars = plt.bar(rob_acc.keys(), [v*100 for v in rob_acc.values()], color=colors)
    plt.ylim(0, 100)
    plt.ylabel('Accuracy (%)')
    plt.title('Model Robustness under Degradation', pad=15)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 1, f'{h:.1f}%', ha='center', va='bottom', fontweight='bold')
    save_plot('robustness_comparison.png')

    # ── 4. Training Curves ────────────────────────────────────────────────────
    section('Generating Training Curves')
    p1_csv = os.path.join(RESULTS_DIR, 'phase1_log.csv')
    p2_csv = os.path.join(RESULTS_DIR, 'phase2_log.csv')

    try:
        import pandas as pd
        df1 = pd.read_csv(p1_csv) if os.path.exists(p1_csv) else None
        df2 = pd.read_csv(p2_csv) if os.path.exists(p2_csv) else None

        if df1 is not None and df2 is not None:
            plt.figure(figsize=(12, 5))
            
            # Accuracy
            plt.subplot(1, 2, 1)
            plt.plot(df1['epoch'], df1['accuracy'], 'b--', label='P1 Train Acc')
            plt.plot(df1['epoch'], df1['val_accuracy'], 'b-', label='P1 Val Acc')
            offset = df1['epoch'].max() + 1
            plt.plot(df2['epoch'] + offset, df2['accuracy'], 'r--', label='P2 Train Acc')
            plt.plot(df2['epoch'] + offset, df2['val_accuracy'], 'r-', label='P2 Val Acc')
            plt.axvline(x=offset, color='k', linestyle=':', alpha=0.5)
            plt.title('Training Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Loss
            plt.subplot(1, 2, 2)
            plt.plot(df1['epoch'], df1['loss'], 'b--', label='P1 Train Loss')
            plt.plot(df1['epoch'], df1['val_loss'], 'b-', label='P1 Val Loss')
            plt.plot(df2['epoch'] + offset, df2['loss'], 'r--', label='P2 Train Loss')
            plt.plot(df2['epoch'] + offset, df2['val_loss'], 'r-', label='P2 Val Loss')
            plt.axvline(x=offset, color='k', linestyle=':', alpha=0.5)
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)

            save_plot('training_curves.png')
    except Exception as e:
        print(f'  Could not plot training curves: {e}')

    banner('EVALUATION COMPLETE')
    print('  All plots saved to: plots/potato_disease/')

if __name__ == '__main__':
    run()
