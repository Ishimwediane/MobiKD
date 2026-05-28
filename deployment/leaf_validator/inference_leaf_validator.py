"""
inference_leaf_validator.py
===========================
Production-ready inference script for the Leaf Validator model.
Stage 1 of the Farmer Assistant Pipeline:

    Image → Leaf Validator → Crop Detector → Disease Classifier

Usage:
    python deployment/inference_leaf_validator.py <image_path>
    python deployment/inference_leaf_validator.py <image_path> --threshold 0.80
    python deployment/inference_leaf_validator.py <image_path> --model fp32

Outputs:
    LEAF      → confident leaf prediction
    NOT_LEAF  → confident not-leaf prediction
    UNCERTAIN → model not confident enough (below threshold)
"""

import argparse
import os
import sys
import time
import numpy as np

# ── Optional rich output ───────────────────────────────────────────────────────
try:
    from PIL import Image
except ImportError:
    print('ERROR: Pillow not installed. Run: pip install Pillow')
    sys.exit(1)

try:
    import tensorflow as tf
except ImportError:
    print('ERROR: TensorFlow not installed. Run: pip install tensorflow')
    sys.exit(1)

# ============================================================
# CONSTANTS
# ============================================================
IMAGE_SIZE   = 32
CLASS_NAMES  = {0: 'NOT_LEAF', 1: 'LEAF'}

# Default model path relative to the project root
_HERE        = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.dirname(_HERE)
_MODELS_DIR  = os.path.join(_ROOT, 'models')

MODEL_PATHS  = {
    'float16': os.path.join(_MODELS_DIR, 'kd_mobilenetv2_leaf_validator_float16.tflite'),
    'fp32'   : os.path.join(_MODELS_DIR, 'kd_mobilenetv2_leaf_validator_fp32.tflite'),
    'keras'  : os.path.join(_MODELS_DIR, 'kd_mobilenetv2_leaf_validator.keras'),
}

DEFAULT_MODEL     = 'float16'
DEFAULT_THRESHOLD = 0.70


# ============================================================
# PREPROCESSING
# ============================================================
def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess an image for leaf validator inference.

    Preprocessing pipeline:
        1. Open image as RGB
        2. Resize to 32×32 pixels (bilinear interpolation)
        3. Normalize pixel values to [0.0, 1.0]
        4. Add batch dimension → shape (1, 32, 32, 3)

    Args:
        image_path: Path to an image file (JPG, PNG, BMP, WEBP, etc.)

    Returns:
        Float32 numpy array of shape (1, 32, 32, 3) in range [0, 1].
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f'Image not found: {image_path}')

    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0

    if arr.min() < 0 or arr.max() > 1.0 + 1e-6:
        raise ValueError(f'Unexpected pixel range after normalization: '
                         f'[{arr.min():.4f}, {arr.max():.4f}]')

    return np.expand_dims(arr, axis=0)  # (1, 32, 32, 3)


# ============================================================
# INFERENCE RUNNERS
# ============================================================
def run_tflite(model_path: str, img_tensor: np.ndarray) -> np.ndarray:
    """Run inference with a TFLite model. Returns softmax probabilities."""
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()

    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    # Validate input shape
    expected = tuple(inp_d['shape'])
    if img_tensor.shape != expected:
        raise ValueError(f'Input shape mismatch. '
                         f'Expected {expected}, got {img_tensor.shape}')

    interp.set_tensor(inp_d['index'], img_tensor.astype(np.float32))
    interp.invoke()
    probs = interp.get_tensor(out_d['index'])[0].copy()

    # Sanity checks
    if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
        raise RuntimeError('NaN/Inf detected in model output. Model may be broken.')

    return probs


def run_keras(model_path: str, img_tensor: np.ndarray) -> np.ndarray:
    """Run inference with a Keras model. Returns softmax probabilities."""
    model = tf.keras.models.load_model(model_path)
    probs = model.predict(img_tensor, verbose=0)[0]
    return probs


# ============================================================
# PREDICTION LOGIC
# ============================================================
def predict(image_path: str,
            model_variant: str = DEFAULT_MODEL,
            threshold: float = DEFAULT_THRESHOLD,
            verbose: bool = True) -> dict:
    """
    Full inference pipeline: load → preprocess → infer → threshold → return result.

    Args:
        image_path   : Path to the image file
        model_variant: 'float16' | 'fp32' | 'keras'
        threshold    : Minimum confidence for a firm prediction (default 0.70)
        verbose      : Print detailed output

    Returns:
        dict with keys:
            label        : 'LEAF' | 'NOT_LEAF' | 'UNCERTAIN'
            class_id     : 0 (not_leaf) | 1 (leaf) | -1 (uncertain)
            confidence   : float in [0, 1]
            probabilities: list [p_not_leaf, p_leaf]
            model_used   : str
            threshold    : float
            inference_ms : float
    """
    model_path = MODEL_PATHS.get(model_variant)
    if model_path is None:
        raise ValueError(f'Unknown model variant: {model_variant}. '
                         f'Choose from: {list(MODEL_PATHS.keys())}')

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f'Model not found: {model_path}\n'
            f'Run leaf_stage1_train.py and leaf_stage2_export.py first.'
        )

    # Preprocessing
    t0         = time.time()
    img_tensor = preprocess_image(image_path)

    # Inference
    if model_variant == 'keras':
        probs = run_keras(model_path, img_tensor)
    else:
        probs = run_tflite(model_path, img_tensor)

    inference_ms = (time.time() - t0) * 1000

    # Decision with confidence threshold
    class_id   = int(np.argmax(probs))
    confidence = float(probs[class_id])

    if confidence < threshold:
        label    = 'UNCERTAIN'
        class_id = -1
    else:
        label = CLASS_NAMES[class_id]

    result = {
        'label'        : label,
        'class_id'     : class_id,
        'confidence'   : round(confidence, 4),
        'probabilities': [round(float(p), 4) for p in probs],
        'model_used'   : model_variant,
        'threshold'    : threshold,
        'inference_ms' : round(inference_ms, 2),
    }

    if verbose:
        _print_result(image_path, result)

    return result


def _print_result(image_path: str, result: dict):
    label = result['label']
    conf  = result['confidence']
    probs = result['probabilities']
    ms    = result['inference_ms']

    emoji = {'LEAF': '🍃', 'NOT_LEAF': '🚫', 'UNCERTAIN': '❓'}.get(label, '')

    print('\n' + '─' * 55)
    print(f'  Image     : {os.path.basename(image_path)}')
    print(f'  Prediction: {emoji}  {label}')
    print(f'  Confidence: {conf*100:.2f}%')
    print(f'  P(not_leaf): {probs[0]*100:.2f}%   P(leaf): {probs[1]*100:.2f}%')
    print(f'  Threshold : {result["threshold"]*100:.0f}%')
    print(f'  Model     : {result["model_used"]}')
    print(f'  Latency   : {ms:.1f} ms')
    print('─' * 55)

    if label == 'UNCERTAIN':
        print('  ⚠  Confidence below threshold.')
        print('  ⚠  This image is REJECTED for downstream processing.')
        print('  ⚠  Recommendation: capture a clearer image of the leaf.')


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Leaf Validator — Stage 1 of Farmer Assistant Pipeline',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python deployment/inference_leaf_validator.py leaf.jpg
  python deployment/inference_leaf_validator.py leaf.jpg --threshold 0.80
  python deployment/inference_leaf_validator.py leaf.jpg --model fp32
  python deployment/inference_leaf_validator.py leaf.jpg --model keras
        """
    )
    parser.add_argument('image', type=str,
                        help='Path to input image (JPG, PNG, BMP, WEBP)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        choices=list(MODEL_PATHS.keys()),
                        help=f'Model variant to use (default: {DEFAULT_MODEL})')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help=f'Confidence threshold [0.5–1.0] (default: {DEFAULT_THRESHOLD})')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress detailed output')
    args = parser.parse_args()

    if not (0.5 <= args.threshold <= 1.0):
        print('ERROR: --threshold must be between 0.5 and 1.0')
        sys.exit(1)

    try:
        result = predict(
            image_path     = args.image,
            model_variant  = args.model,
            threshold      = args.threshold,
            verbose        = not args.quiet,
        )
        # Exit code 0 = LEAF, 1 = NOT_LEAF, 2 = UNCERTAIN
        exit_codes = {'LEAF': 0, 'NOT_LEAF': 1, 'UNCERTAIN': 2}
        sys.exit(exit_codes.get(result['label'], 2))

    except FileNotFoundError as e:
        print(f'\nERROR: {e}')
        sys.exit(3)
    except Exception as e:
        print(f'\nERROR during inference: {e}')
        sys.exit(4)


if __name__ == '__main__':
    main()
