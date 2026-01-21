import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from typing import Dict
import uvicorn
import requests
import io
from PIL import Image

app = FastAPI(title="garbage-classifier")

session = ort.InferenceSession(
    "models/xception_v4_final.onnx", 
    providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

classes = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash"
]


class PredictRequest(BaseModel):
    url: HttpUrl


class PredictResponse(BaseModel):
    predictions: Dict[str, float]  # Use Dict for Python 3.8 compatibility
    top_class: str
    top_probability: float


def softmax(x):
    """Apply softmax to convert raw scores to probabilities"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def predict(url: str):
    # Download image with User-Agent
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status() 
    
    # Open and convert to RGB
    img = Image.open(io.BytesIO(response.content)).convert('RGB')
    
    # Resize to target size (299x299 for Xception)
    img = img.resize((299, 299), Image.LANCZOS)
    
    # Convert to numpy array and preprocess for Xception
    img_array = np.array(img, dtype=np.float32)
    
    # Xception preprocessing: scale to [-1, 1]
    img_array = img_array / 127.5 - 1.0
    
    # Add batch dimension
    X = np.expand_dims(img_array, axis=0)
    
    # Run inference
    result = session.run([output_name], {input_name: X})
    scores = result[0][0]
    
    # Apply softmax
    probs = softmax(scores)
    
    # Format predictions
    predictions_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}
    top_class = max(predictions_dict, key=predictions_dict.get)
    top_probability = float(predictions_dict[top_class])
    
    return predictions_dict, top_class, top_probability

@app.get("/")
def root():
    return {"message": "Garbage Classification Service"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    predictions, top_class, top_prob = predict(str(request.url))
    
    return PredictResponse(
        predictions=predictions,
        top_class=top_class,
        top_probability=top_prob
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)