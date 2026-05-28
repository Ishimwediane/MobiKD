"""
STAGE 1 - DATA PREPARATION
Run: python stage1_data.py
Output: data/cifar100_tree_data.npz
"""
import os, time
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from config import (
    DATA_DIR, PLOTS_DIR, TREE_FINE_LABELS,
    apply_blur, apply_noise, apply_low_light, apply_combined,
    degrade_dataset, setup_logging, setup_gpu
)

def run():
    print('\n' + '='*60)
    print('STAGE 1: DATA PREPARATION')
    print('='*60)
    logger = setup_logging()
    setup_gpu(logger)

    os.makedirs(DATA_DIR,  exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    path = f'{DATA_DIR}/cifar100_tree_data.npz'
    if os.path.exists(path):
        print('   Data already exists. Loading to verify...')
        data = np.load(path)
        val_count = data["x_val_clean"].shape[0] if "x_val_clean" in data else 0
        print(f'  Train: {data["x_train"].shape[0]:,}  '
              f'Validation: {val_count:,}  '
              f'Test: {data["x_test_clean"].shape[0]:,}')
        if "x_val_clean" in data and "y_val" in data:
            logger.info('Stage 1: Already complete, skipping.')
            return
        data.close()
        print('   Existing data has no validation split. Rebuilding Stage 1...')

    logger.info('Stage 1: Starting')
    t0 = time.time()

    (x_tr, y_tr_f), (x_te, y_te_f) = cifar100.load_data(label_mode='fine')
    print(f'  Loaded: train={x_tr.shape}  test={x_te.shape}')

    y_tr = np.isin(y_tr_f.flatten(), TREE_FINE_LABELS).astype(np.int32)
    y_te = np.isin(y_te_f.flatten(), TREE_FINE_LABELS).astype(np.int32)

    # Keep the CIFAR-100 test set untouched for final evaluation only.
    x_train_raw, x_val_raw, y_train_raw, y_val_raw = train_test_split(
        x_tr, y_tr,
        test_size=0.2,
        random_state=42,
        stratify=y_tr
    )

    n_tree = int(np.sum(y_train_raw == 1))
    n_not  = int(np.sum(y_train_raw == 0))
    total  = len(y_train_raw)
    w0 = total / (2.0 * n_not)
    w1 = total / (2.0 * n_tree)
    print(f'  Train Tree: {n_tree:,}  Train Not-Tree: {n_not:,}  Ratio 1:{n_not//n_tree}')
    print(f'  Validation: {len(y_val_raw):,}  Test: {len(y_te):,} (held out)')
    print(f'  Weights -> Not-Tree: {w0:.4f}  Tree: {w1:.4f}')

    # Scale to [0,1] — degradation functions and models both expect this
    x_tr_n = x_train_raw.astype('float32') / 255.0
    x_val_n = x_val_raw.astype('float32') / 255.0
    x_te_n = x_te.astype('float32') / 255.0
    y_tr_c = to_categorical(y_train_raw, 2)
    y_val_c = to_categorical(y_val_raw, 2)
    y_te_c = to_categorical(y_te, 2)

    print('\n  Building degraded test sets...')
    x_te_blur = degrade_dataset(x_te_n, apply_blur,      'blur')
    x_te_nois = degrade_dataset(x_te_n, apply_noise,     'noise')
    x_te_lowl = degrade_dataset(x_te_n, apply_low_light, 'low-light')
    x_te_comb = degrade_dataset(x_te_n, apply_combined,  'combined')

    # Save degradation examples plot
    ti = np.where(y_train_raw == 1)[0][0]
    ni = np.where(y_train_raw == 0)[0][0]
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    fig.suptitle('LiteRob - Tree vs Not-Tree Degradations',
                 fontsize=13, fontweight='bold')
    for r, (s, rl) in enumerate(
        [(x_tr_n[ti], 'Tree'), (x_tr_n[ni], 'Not-Tree')]
    ):
        for c, (fn, cl) in enumerate(zip(
            [lambda x: x, apply_blur, apply_noise,
             apply_low_light, apply_combined],
            ['Clean', 'Blur', 'Noise', 'Low Light', 'Combined']
        )):
            axes[r, c].imshow(np.clip(fn(s), 0, 1))
            axes[r, c].axis('off')
            if r == 0:
                axes[r, c].set_title(cl, fontsize=10, fontweight='bold')
            if c == 0:
                axes[r, c].set_ylabel(rl, fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/degradation_examples.png', dpi=150,
                bbox_inches='tight')
    plt.close()

    np.savez_compressed(
        path,
        x_train=x_tr_n, y_train=y_tr_c,
        x_val_clean=x_val_n, y_val=y_val_c,
        x_test_clean=x_te_n, x_test_blur=x_te_blur,
        x_test_noise=x_te_nois, x_test_lowlight=x_te_lowl,
        x_test_combined=x_te_comb, y_test=y_te_c,
        class_weights=np.array([w0, w1])
    )
    elapsed = (time.time() - t0) / 60
    print(f'   Saved {path} ({os.path.getsize(path)/1024/1024:.1f} MB) '
          f'in {elapsed:.1f} min')
    logger.info(f'Stage 1 complete in {elapsed:.1f} min')

if __name__ == '__main__':
    run()
