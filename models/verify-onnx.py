import onnxruntime as ort
import numpy as np
import json
from PIL import Image

def preprocess_for_xception(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((299, 299))
    x = np.array(img).astype(np.float32)
    x = np.expand_dims(x, axis=0)
    # Xception preprocess_input (mode='tf')
    x /= 127.5
    x -= 1.0
    return x

def run_prediction(image_path):
    # Load labels
    with open('labels.json', 'r') as f:
        labels = json.load(f)

    # Load ONNX session
    session = ort.InferenceSession("./xception_v4_final.onnx")
    input_name = session.get_inputs()[0].name

    # Preprocess and Predict
    image_data = preprocess_for_xception(image_path)
    preds = session.run(None, {input_name: image_data})[0]
    
    # Process results
    class_idx = str(np.argmax(preds, axis=1)[0])
    confidence = np.max(preds) * 100
    label = labels.get(class_idx, "Unknown")

    print(f"--- Result ---")
    print(f"Predicted: {label} ({confidence:.2f}%)")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_prediction(sys.argv[1])
    else:
        print("Usage: uv run verify_onnx.py <path_to_image>")