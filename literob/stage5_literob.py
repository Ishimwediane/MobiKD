"""
STAGE 5 - LITEROB STUDENT (Knowledge Distillation)
Run: python stage5_literob.py
Requires:
  - data/cifar100_tree_data.npz   (run stage1_data.py first)
  - models/teacher_resnet50.keras (run stage2_teacher.py first)
Output: models/literob_mobilenet.keras
"""
import os, time
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
from tensorflow import keras
from config import (
    DATA_DIR, MODELS_DIR, RESULTS_DIR, BATCH_SIZE, LITEROB_EPOCHS,
    ALPHA, TEMPERATURE,
    build_student, load_teacher, custom_train_kd,
    apply_blur, apply_noise, apply_low_light, degrade_dataset,
    setup_logging, setup_gpu, clear_gpu_memory
)

def run():
    print('\n' + '='*60)
    print('STAGE 5: LITEROB STUDENT (Knowledge Distillation)')
    print('='*60)
    logger = setup_logging()
    setup_gpu(logger)
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    data_path = f'{DATA_DIR}/cifar100_tree_data.npz'
    if not os.path.exists(data_path):
        print('   Data not found. Run stage1_data.py first.')
        return

    teacher_path = f'{MODELS_DIR}/teacher_resnet50.keras'
    if not os.path.exists(teacher_path):
        print('   Teacher not found. Run stage2_teacher.py first.')
        return

    data = np.load(data_path)
    if 'x_val_clean' not in data or 'y_val' not in data:
        print('   Data file has no validation split. Regenerate Stage 1 first.')
        return

    model_path = f'{MODELS_DIR}/literob_mobilenet.keras'
    if os.path.exists(model_path):
        print('   Existing LiteRob model found. Retraining to match the current data split...')

    logger.info('Stage 5: Starting')
    t0 = time.time()

    print('  Loading teacher...')
    teacher = load_teacher()

    model = build_student('LiteRob_MobileNetV2')
    print(f'  Parameters: {model.count_params():,}  '
          f'Alpha: {ALPHA}  Temp: {TEMPERATURE}')
    print('  Building robustness-aware KD training mix...')
    x_clean = data['x_train']
    y_clean = data['y_train']
    x_blur = degrade_dataset(x_clean, apply_blur, 'train blur')
    x_noise = degrade_dataset(x_clean, apply_noise, 'train noise')
    x_lowlight = degrade_dataset(x_clean, apply_low_light, 'train low-light')
    robust_data = {
        'x_train': np.concatenate([x_clean, x_blur, x_noise, x_lowlight], axis=0),
        'y_train': np.concatenate([y_clean, y_clean, y_clean, y_clean], axis=0),
        'x_val_clean': data['x_val_clean'],
        'y_val': data['y_val'],
        'x_test_clean': data['x_test_clean'],
        'y_test': data['y_test'],
        'class_weights': data['class_weights'],
    }
    print(f'  LiteRob KD samples: {robust_data["x_train"].shape[0]:,} '
          f'(clean + blur + noise + low-light)')

    model = custom_train_kd(
        model, teacher, robust_data, LITEROB_EPOCHS,
        'LiteRob Student', f'{RESULTS_DIR}/literob_log.csv',
        checkpoint_path=f'{MODELS_DIR}/literob_ckpt.keras'
    )

    model.save(model_path)
    preds = model.predict(data['x_test_clean'], batch_size=BATCH_SIZE, verbose=0)
    acc = float(np.mean(
        np.argmax(preds, axis=1) == np.argmax(data['y_test'], axis=1)
    ))
    elapsed = (time.time() - t0) / 60
    print(f'\n   Accuracy: {acc*100:.2f}%  Time: {elapsed:.1f} min')
    print(f'  Saved: {model_path}')
    logger.info(f'Stage 5 complete. Acc: {acc*100:.2f}%. Time: {elapsed:.1f} min')

    del teacher
    clear_gpu_memory()

if __name__ == '__main__':
    run()
