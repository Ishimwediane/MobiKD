# Deployment Guide: Leaf Validator (Stage 1 — Farmer Assistant Pipeline)

> **Model**: `kd_mobilenetv2_leaf_validator_float16.tflite`  
> **Pipeline Role**: Stage 1 — Binary classifier that determines whether an image contains a leaf before passing it to downstream crop detection and disease classification.

---

## Pipeline Overview

```
  Camera / Image Input
         │
         ▼
  ┌──────────────────────┐
  │   Leaf Validator     │  ← You are here
  │  (This model)        │
  └──────┬───────────────┘
         │
    LEAF │  NOT_LEAF / UNCERTAIN
         │         │
         ▼         ▼
  Crop Detector   Reject image
         │
         ▼
  Disease Classifier
```

---

## Model Files

| File | Size | Use Case |
|------|------|----------|
| `kd_mobilenetv2_leaf_validator_float16.tflite` | ~4.7 MB | **Recommended for mobile** |
| `kd_mobilenetv2_leaf_validator_fp32.tflite` | ~9.5 MB | Server / high-accuracy reference |
| `kd_mobilenetv2_leaf_validator.keras` | ~10 MB | Research / fine-tuning only |

---

## Input Requirements

| Property | Value |
|----------|-------|
| **Image Size** | 32 × 32 pixels |
| **Channels** | 3 (RGB) |
| **Pixel Range** | `[0.0, 1.0]` (divide uint8 by 255) |
| **Dtype** | `float32` |
| **Batch Shape** | `(1, 32, 32, 3)` |

> ⚠ **Important**: Do NOT pass images as uint8. Always normalize to `[0, 1]` before inference.

---

## Preprocessing Pipeline

```python
from PIL import Image
import numpy as np

def preprocess(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((32, 32), Image.BILINEAR)   # always bilinear
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)            # shape: (1, 32, 32, 3)
```

---

## Output Format

The model outputs a softmax probability vector of shape `(1, 2)`:

```
[P(not_leaf), P(leaf)]
```

| Index | Class | Meaning |
|-------|-------|---------|
| `0` | `not_leaf` | Image does NOT contain a leaf |
| `1` | `leaf` | Image contains a leaf |

---

## Inference Example

### Python (TFLite)

```python
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = 'models/kd_mobilenetv2_leaf_validator_float16.tflite'
THRESHOLD  = 0.70

def run_inference(image_path):
    # Preprocess
    img = Image.open(image_path).convert('RGB').resize((32, 32), Image.BILINEAR)
    arr = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

    # Load and run model
    interp = tf.lite.Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    inp_i = interp.get_input_details()[0]['index']
    out_i = interp.get_output_details()[0]['index']

    interp.set_tensor(inp_i, arr)
    interp.invoke()
    probs = interp.get_tensor(out_i)[0]

    # Apply threshold
    class_id   = int(np.argmax(probs))
    confidence = float(probs[class_id])

    if confidence < THRESHOLD:
        return 'UNCERTAIN', -1, confidence, probs
    labels = ['not_leaf', 'leaf']
    return labels[class_id], class_id, confidence, probs
```

### Command Line

```bash
# Basic inference
python deployment/inference_leaf_validator.py path/to/leaf.jpg

# Custom threshold
python deployment/inference_leaf_validator.py path/to/leaf.jpg --threshold 0.80

# Use FP32 model
python deployment/inference_leaf_validator.py path/to/leaf.jpg --model fp32

# Quiet mode (machine-readable exit code only)
python deployment/inference_leaf_validator.py path/to/leaf.jpg --quiet
echo "Exit code: $?"   # 0=LEAF, 1=NOT_LEAF, 2=UNCERTAIN
```

---

## Confidence Threshold Recommendation

| Threshold | Behaviour | Recommended For |
|-----------|-----------|-----------------|
| `0.60` | More permissive — accepts uncertain cases | Lab / controlled environment |
| `0.70` | **Balanced default** ← recommended | Field use, mixed conditions |
| `0.80` | Strict — rejects borderline cases | High-stakes automated pipeline |
| `0.90` | Very strict — minimal false positives | Critical applications only |

> 💡 **For farmer field use**: Start with `0.70`. If you see many false positives (non-leaves passing through), increase to `0.75–0.80`.

---

## Mobile Deployment Guidance

### Android (Java/Kotlin via TFLite Android SDK)

```kotlin
val model = Interpreter(loadModelFile(context, "kd_mobilenetv2_leaf_validator_float16.tflite"))
val input  = Array(1) { Array(32) { Array(32) { FloatArray(3) } } }
val output = Array(1) { FloatArray(2) }

// Fill input with normalized pixels...
model.run(input, output)
val probs = output[0]
val leafProb = probs[1]
```

### iOS (Swift via Core ML / TFLite iOS)

```swift
// Convert .tflite to .mlmodel using coremltools, or use TFLite Swift SDK
// Input tensor: [1, 32, 32, 3] Float32
// Output tensor: [1, 2] Float32
```

### Model Size Check

```bash
# Float16 model should be < 5 MB
ls -lh models/kd_mobilenetv2_leaf_validator_float16.tflite
```

---

## Edge Deployment Notes

- **RAM requirement**: < 50 MB at runtime (MobileNetV2 is very lightweight)
- **Inference latency**: ~5–20 ms on modern Android/iOS (float16)
- **No internet required**: fully offline inference
- **Input resolution**: 32×32 is intentionally small for speed; bilinear downscaling handles most real-world images well
- **Robustness**: Model was evaluated on blur, noise, low-light, and combined corruptions

---

## UNCERTAIN Predictions

If the model returns `UNCERTAIN` (confidence < threshold):

1. **Do NOT pass the image to the disease classifier**
2. Ask the farmer to retake the photo:
   - Ensure the leaf is centered and fills the frame
   - Use adequate lighting (avoid deep shadows)
   - Avoid motion blur

---

## Class Definition

| Class | Description |
|-------|-------------|
| `leaf` | Any plant leaf (crop leaves, weed leaves, etc.) |
| `not_leaf` | Anything that is not a leaf: soil, sky, rocks, tools, hands, etc. |

---

## Files

```
deployment/
├── inference_leaf_validator.py   ← main inference script (this stage)
├── README_leaf_validator.md      ← this file
├── inference.py                  ← disease classifier inference (Stage 3)
└── README.md                     ← disease classifier deployment guide
```

---

*Generated by MobiKD Leaf Validator Pipeline — Transfer learning from distilled MobileNetV2*
