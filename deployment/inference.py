import numpy as np
import tensorflow as tf
from PIL import Image

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((32, 32))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict(tflite_path, image_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    inp_idx = interpreter.get_input_details()[0]['index']
    out_idx = interpreter.get_output_details()[0]['index']
    
    img_tensor = load_image(image_path)
    interpreter.set_tensor(inp_idx, img_tensor)
    interpreter.invoke()
    
    probs = interpreter.get_tensor(out_idx)[0]
    class_id = np.argmax(probs)
    return class_id, probs

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python inference.py <model.tflite> <image.jpg>")
    else:
        cid, p = predict(sys.argv[1], sys.argv[2])
        labels = {0: "Not-Tree", 1: "Tree"}
        print(f"Prediction: {labels[cid]} (Probabilities: {p})")
