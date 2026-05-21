# Deployment Guide: KD-MobileNetV2

## Which Model to Deploy
**Use**: `kd_mobilenetv2_float16.tflite`
This model has been validated for numerical stability and provides the best trade-off between size (4.63 MB) and accuracy (98.54%).

## Input Specifications
- **Image Size**: 32x32 pixels (or 64x64 if resizing is handled internally, ensure aspect ratio is maintained)
- **Channels**: 3 (RGB)
- **Normalization**: The model internally handles scaling from `[0, 255]` or `[0, 1]`. Pass the normalized float32 array or uint8 array depending on your TFLite interpreter setup. Check the exact input details of the `.tflite` file.
  *Note*: The Keras model included an internal `Rescaling(scale=2.0, offset=-1.0)` to map `[0, 1]` to `[-1, 1]`.

## Expected Outputs
- **Output Tensor**: Shape `[1, 2]`
- **Format**: Softmax probabilities (summing to 1.0)
- Class 0: Not-Tree
- Class 1: Tree

## Confidence Threshold
**Recommendation**: Use `0.65` or `0.70` as the confidence threshold for Class 1 (Tree) to minimize false positives in noisy environments.
