"""
inference_pipeline.py
=====================
Combined 2-stage inference pipeline:
    Stage 1: Leaf Validator  -- is this a leaf?
    Stage 2: Potato Disease  -- Early Blight | Healthy | Late Blight

Usage:
    python inference_pipeline.py <image_path>
    python inference_pipeline.py <image_path> --verbose

Returns JSON result to stdout.
"""

import os, sys, json, argparse
import numpy as np
from PIL import Image
import tensorflow as tf

# ── Paths (relative to this file's location) ──────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__)) # deployment/potato_pipeline
_DEPLOY_DIR  = os.path.dirname(_SCRIPT_DIR)               # deployment/
_PROJECT     = os.path.dirname(_DEPLOY_DIR)               # MobiKD/

LEAF_MODEL_PATH   = os.path.join(
    _PROJECT, 'models', 'leaf_validator',
    'kd_mobilenetv2_leaf_validator_float16.tflite'
)
POTATO_MODEL_PATH = os.path.join(
    _PROJECT, 'models', 'potato_disease',
    'kd_mobilenetv2_potato_disease_float16.tflite'
)

LEAF_IMAGE_SIZE       = 32
POTATO_IMAGE_SIZE     = 96
LEAF_THRESHOLD        = 0.70   # min confidence to accept as leaf
DISEASE_THRESHOLD     = 0.65   # min confidence for disease prediction

LEAF_CLASSES    = ['not_leaf', 'leaf']
DISEASE_CLASSES = ['Early Blight', 'Healthy', 'Late Blight', 'Not a Potato Leaf']
DISEASE_ADVICE  = {
    'Early Blight': (
        'Potato Early Blight detected (Alternaria solani). '
        'Apply fungicide (chlorothalonil or mancozeb). '
        'Remove infected lower leaves. Improve air circulation.'
    ),
    'Late Blight': (
        'Potato Late Blight detected (Phytophthora infestans). '
        'Apply copper-based fungicide immediately. '
        'Remove and destroy infected plants to prevent spread.'
    ),
    'Healthy': (
        'Potato leaf appears healthy. '
        'Continue regular monitoring and maintenance.'
    ),
    'Not a Potato Leaf': (
        'This leaf does not appear to be a potato leaf. '
        'This classifier is optimised for potato disease detection only. '
        'Please provide a potato leaf image.'
    ),
}


# ── TFLite interpreter loader ─────────────────────────────────────────────────
def load_interpreter(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Model not found: {path}')
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp_i = interp.get_input_details()[0]['index']
    out_i = interp.get_output_details()[0]['index']
    return interp, inp_i, out_i


def predict(interp, inp_i, out_i, img_array):
    """Run single-image inference. img_array: float32 (32,32,3) in [0,1]."""
    interp.set_tensor(inp_i, img_array[np.newaxis].astype(np.float32))
    interp.invoke()
    return interp.get_tensor(out_i)[0].copy()


# ── Image loader ──────────────────────────────────────────────────────────────
def load_image(path, target_size):
    img = Image.open(path).convert('RGB').resize(
        (target_size, target_size), Image.BILINEAR
    )
    return np.array(img, dtype=np.float32) / 255.0


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(image_path, verbose=False):
    result = {
        'image'         : os.path.basename(image_path),
        'is_leaf'       : None,
        'leaf_confidence' : None,
        'disease'       : None,
        'disease_confidence' : None,
        'advice'        : None,
        'status'        : None,
    }

    # Load image
    if not os.path.exists(image_path):
        result['status'] = f'ERROR: image not found: {image_path}'
        return result

    # ── Stage 1: Leaf Validator ───────────────────────────────────────────────
    if verbose:
        print('\n[Stage 1] Leaf Validator...')
        
    img_leaf = load_image(image_path, LEAF_IMAGE_SIZE)
    
    leaf_interp, leaf_inp, leaf_out = load_interpreter(LEAF_MODEL_PATH)
    leaf_probs = predict(leaf_interp, leaf_inp, leaf_out, img_leaf)
    leaf_class = int(np.argmax(leaf_probs))
    leaf_conf  = float(leaf_probs[leaf_class])

    result['is_leaf']          = bool(leaf_class == 1)
    result['leaf_confidence']  = round(leaf_conf * 100, 2)

    if verbose:
        print(f'  Prediction : {LEAF_CLASSES[leaf_class]}')
        print(f'  Confidence : {leaf_conf*100:.2f}%')
        for i, c in enumerate(LEAF_CLASSES):
            print(f'  {c:10s} : {leaf_probs[i]*100:.2f}%')

    # Gate: reject if not a leaf or low confidence
    if leaf_class != 1 or leaf_conf < LEAF_THRESHOLD:
        result['status'] = 'REJECTED: not a leaf (or confidence too low)'
        result['disease'] = None
        if verbose:
            print(f'\n  REJECTED -- not a valid leaf image.')
        return result

    if verbose:
        print(f'\n  Confirmed as leaf. Proceeding to disease detection...')

    # ── Stage 2: Potato Disease Detector ─────────────────────────────────────
    if verbose:
        print('\n[Stage 2] Potato Disease Detector...')
        
    img_potato = load_image(image_path, POTATO_IMAGE_SIZE)
    
    disease_interp, dis_inp, dis_out = load_interpreter(POTATO_MODEL_PATH)
    disease_probs = predict(disease_interp, dis_inp, dis_out, img_potato)
    disease_idx   = int(np.argmax(disease_probs))
    disease_conf  = float(disease_probs[disease_idx])
    disease_name  = DISEASE_CLASSES[disease_idx]

    result['disease']             = disease_name
    result['disease_confidence']  = round(disease_conf * 100, 2)
    result['advice']              = DISEASE_ADVICE[disease_name]

    if disease_conf >= DISEASE_THRESHOLD:
        result['status'] = f'OK: {disease_name}'
    else:
        result['status'] = f'LOW CONFIDENCE: {disease_name} ({disease_conf*100:.1f}% < {DISEASE_THRESHOLD*100:.0f}%)'

    if verbose:
        print(f'  Scores:')
        for i, cls in enumerate(DISEASE_CLASSES):
            bar = '#' * int(disease_probs[i] * 30)
            print(f'  {cls:15s} : {disease_probs[i]*100:5.2f}%  {bar}')
        print(f'\n  Diagnosis  : {disease_name}')
        print(f'  Confidence : {disease_conf*100:.2f}%')
        print(f'  Advice     : {result["advice"]}')

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='2-Stage Potato Disease Pipeline: Leaf Validator -> Disease Detector'
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed per-stage output')
    parser.add_argument('--json', action='store_true',
                        help='Print only JSON result (for integration)')
    args = parser.parse_args()

    result = run_pipeline(args.image, verbose=args.verbose and not args.json)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print('\n' + '=' * 55)
        print('  DIAGNOSIS RESULT')
        print('=' * 55)
        print(f'  Image      : {result["image"]}')
        print(f'  Is Leaf    : {result["is_leaf"]}  ({result["leaf_confidence"]}%)')
        if result['disease']:
            print(f'  Disease    : {result["disease"]}  ({result["disease_confidence"]}%)')
            print(f'  Advice     : {result["advice"]}')
        print(f'  Status     : {result["status"]}')
        print('=' * 55)


if __name__ == '__main__':
    main()
