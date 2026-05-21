"""
STAGE 2 - TEACHER TRAINING (ResNet50)
Run: python stage2_teacher.py
Requires: data/cifar100_tree_data.npz  (run stage1_data.py first)
Output:   models/teacher_resnet50.keras
"""
import os, time
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from config import (
    DATA_DIR, MODELS_DIR, RESULTS_DIR,
    TEACHER_LR, TEACHER_BATCH_SIZE, TEACHER_P1_EPOCHS,
    build_teacher_model, setup_logging, setup_gpu, clear_gpu_memory
)

def run():
    print('\n' + '='*60)
    print('STAGE 2: TEACHER TRAINING (ResNet50)')
    print('='*60)
    logger = setup_logging()
    setup_gpu(logger)
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- load data ----
    data_path = f'{DATA_DIR}/cifar100_tree_data.npz'
    if not os.path.exists(data_path):
        print('   Data not found. Run stage1_data.py first.')
        return
    data = np.load(data_path)
    if 'x_val_clean' not in data or 'y_val' not in data:
        print('   Data file has no validation split. Regenerate Stage 1 first.')
        return

    model_path = f'{MODELS_DIR}/teacher_resnet50.keras'
    if os.path.exists(model_path):
        print('  Existing teacher found. Retraining to match the current data split...')

    logger.info('Stage 2: Starting teacher training')
    t0 = time.time()

    teacher, base = build_teacher_model()
    print(f'  Parameters: {teacher.count_params():,}')
    cw = {0: float(data['class_weights'][0]),
          1: float(data['class_weights'][1])}

    # Fine-tune last 30 layers of ResNet50
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    teacher.compile(
        optimizer=keras.optimizers.Adam(learning_rate=TEACHER_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    teacher.fit(
        data['x_train'], data['y_train'],
        validation_data=(data['x_val_clean'], data['y_val']),
        epochs=TEACHER_P1_EPOCHS,
        batch_size=TEACHER_BATCH_SIZE,
        class_weight=cw,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=5, restore_best_weights=True,
                monitor='val_accuracy'
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=3, min_lr=1e-6, verbose=1
            ),
            keras.callbacks.CSVLogger(f'{RESULTS_DIR}/teacher_log.csv'),
            keras.callbacks.ModelCheckpoint(
                f'{MODELS_DIR}/teacher_ckpt.keras',
                save_best_only=True, verbose=0
            )
        ]
    )

    teacher.save(model_path)

    preds = teacher.predict(data['x_test_clean'],
                            batch_size=TEACHER_BATCH_SIZE, verbose=0)
    acc = float(np.mean(
        np.argmax(preds, axis=1) == np.argmax(data['y_test'], axis=1)
    ))

    elapsed = (time.time() - t0) / 60
    print(f'\n   Teacher Accuracy: {acc*100:.2f}%')
    print(f'  Training Time: {elapsed:.1f} min')
    print(f'  Saved: {model_path}')
    logger.info(f'Stage 2 complete. Acc: {acc*100:.2f}%. Time: {elapsed:.1f} min')
    clear_gpu_memory()

if __name__ == '__main__':
    run()
