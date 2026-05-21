"""
STAGE 3 - BASELINE STUDENT (Standard MobileNetV2)
Run: python stage3_baseline.py
Requires: data/cifar100_tree_data.npz  (run stage1_data.py first)
Output:   models/baseline_mobilenet.keras
"""
import os, time
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
)
from config import (
    DATA_DIR, MODELS_DIR, RESULTS_DIR,
    STUDENT_LR, BATCH_SIZE, BASELINE_EPOCHS, PATIENCE,
    build_student, setup_logging, setup_gpu, clear_gpu_memory
)

def run():
    print('\n' + '='*60)
    print('STAGE 3: BASELINE STUDENT (Standard MobileNetV2)')
    print('='*60)
    logger = setup_logging()
    setup_gpu(logger)
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    data_path = f'{DATA_DIR}/cifar100_tree_data.npz'
    if not os.path.exists(data_path):
        print('   Data not found. Run stage1_data.py first.')
        return
    data = np.load(data_path)
    if 'x_val_clean' not in data or 'y_val' not in data:
        print('   Data file has no validation split. Regenerate Stage 1 first.')
        return

    model_path = f'{MODELS_DIR}/baseline_mobilenet.keras'
    if os.path.exists(model_path):
        print('   Existing baseline found. Retraining to match the current data split...')

    logger.info('Stage 3: Starting')
    t0 = time.time()

    cw = {0: float(data['class_weights'][0]),
          1: float(data['class_weights'][1])}

    model = build_student('Baseline_MobileNetV2')
    print(f'  Parameters: {model.count_params():,}')

    model.compile(
        optimizer=keras.optimizers.Adam(STUDENT_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        data['x_train'], data['y_train'],
        validation_data=(data['x_val_clean'], data['y_val']),
        epochs=BASELINE_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=cw,
        callbacks=[
            EarlyStopping(patience=PATIENCE, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7, verbose=1),
            ModelCheckpoint(f'{MODELS_DIR}/baseline_ckpt.keras',
                            save_best_only=True, verbose=0),
            CSVLogger(f'{RESULTS_DIR}/baseline_log.csv')
        ],
        verbose=1
    )

    model.save(model_path)
    preds = model.predict(data['x_test_clean'], batch_size=BATCH_SIZE, verbose=0)
    acc = float(np.mean(
        np.argmax(preds, axis=1) == np.argmax(data['y_test'], axis=1)
    ))
    elapsed = (time.time() - t0) / 60
    print(f'\n   Accuracy: {acc*100:.2f}%  Time: {elapsed:.1f} min')
    print(f'  Saved: {model_path}')
    logger.info(f'Stage 3 complete. Acc: {acc*100:.2f}%. Time: {elapsed:.1f} min')
    clear_gpu_memory()

if __name__ == '__main__':
    run()
