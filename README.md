# Garbage Classification Model End-to-End Deployment (TensorFlow → ONNX → FastAPI → Docker → Kubernetes)
This is an end-to-end machine learning project that classifies garbage images into 10 categories using a deep learning model (Xception), deployed as a FastAPI microservice on Kubernetes with horizontal pod autoscaling.

## Problem Statement

Waste management is a global challenge, with improper sorting leading to increased landfill waste, contamination of recyclables, and environmental harm. Manual garbage classification is:

- Time-consuming and labor-intensive
- Prone to human error
- Difficult to scale
- Inconsistent across different facilities

**Solution:** An automated garbage classification system that uses deep learning to accurately categorize waste items into 10 classes, enabling:

- Automated waste sorting in recycling facilities
- Real-time classification via API
- Scalable deployment on Kubernetes
- Easy integration with existing waste management systems

## Model Details

- **Architecture:** Xception (transfer learning from ImageNet)
- **Framework:** TensorFlow/Keras
- **Input:** 299×299 RGB images
- **Output:** 10 garbage categories
- **Accuracy:** ~94%
- **Format:** ONNX (optimized for inference)
  
The original Keras model(before conversion to ONNX) can be found [here](https://huggingface.co/Folakunmi21/garbage-classifier/tree/main). The full training notebook can also be found in this repo in /models.

## Categories

- Battery
- Biological (organic waste)
- Cardboard
- Clothes
- Glass
- Metal
- Paper
- Plastic
- Shoes
- Trash (general waste)

## Dataset

- **Source:** [Garbage Classification v2 on Kaggle](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)
- **Size:** ~15,000 images
- **Split:** 70% train, 15% validation, 15% test

## Prerequisites

### System Requirements

- OS: Windows 10/11, macOS, or Linux
- RAM: 4GB minimum (8GB recommended)
- Python: 3.13.5
- Docker Desktop: Latest version
- kubectl: v1.28+
- kind: v0.20+

## Tools Installation

### 1. Install Chocolatey (Windows)

```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

### 2. Install kubectl

**Windows (Chocolatey):**

```bash
choco install kubernetes-cli
```

**macOS (Homebrew):**

```bash
brew install kubectl
```

**Linux:**

```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

**Verify:**

```bash
kubectl version --client
```

### 3. Install kind (Kubernetes in Docker)

**Windows (Chocolatey):**

```bash
choco install kind
```

**macOS (Homebrew):**

```bash
brew install kind
```

**Linux:**

```bash
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind
```

**Verify:**

```bash
kind version
```

### 4. Install uv (Python Package Manager)

**macOS/Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Verify:**

```bash
uv --version
```

### 5. Install Docker Desktop

Download and install from: https://www.docker.com/products/docker-desktop

**Verify:**

```bash
docker --version
```

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/folakunmi21/garbage-classification-kubernetes.git
cd garbage-classification-kubernetes
```

### 2. Create Virtual Environment with uv

```bash
uv venv

# Activate environment
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Git Bash)
source .venv/Scripts/activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install all dependencies
uv pip install fastapi uvicorn onnxruntime pillow requests pydantic numpy

# Or sync from pyproject.toml
uv sync
```

## Convert Keras to ONNX (inside Docker)

### 1. Build Converter Docker Image

```bash
cd models
docker build -f Dockerfile -t model-converter .
```

### 2. Run Docker Image

```bash
docker run --rm -v ${PWD}/../models:/app/models model-converter
```

This will:

- Download the Keras model from Hugging Face
- Convert it to ONNX format
- Save `xception_v4_final.onnx` in the `models/` directory

### 3. Verify ONNX Model (optional)

```bash
python verify-onnx.py
```

Example of expected output:

```
Result...
Predicted: plastic (confidence: 0.87)
```

## Building the FastAPI Service

The API is defined in `app.py`.

### 1. Build the Docker image

```bash
docker build -f Dockerfile.api -t garbage-classifier:v1 .
```

### 2. Test the container locally

```bash
docker run -it --rm -p 8080:8080 garbage-classifier:v1
```

### 3. Open FastAPI interactive docs

http://127.0.0.1:8080/docs

### 4. In another terminal, run the test script

```bash
uv run python test.py
```

`test.py` loads an online image URL. Check the prediction result on FastAPI interactive docs.

## Kubernetes Deployment

### 1. Navigate to k8s directory

```bash
cd k8s
```

### 2. Create a local Kubernetes cluster

```bash
kind create cluster --name garbage-cnn
```

### 3. Load Docker image to kind

```bash
kind load docker-image garbage-classifier:v1 --name garbage-cnn
```

### 4. Apply deployment + service

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

Check the service:

```bash
kubectl get services
kubectl describe service garbage-classifier
```

## Testing the Deployed Service

Check the health endpoint:

```bash
curl http://localhost:30080/health
```

Port forward the service:

```bash
kubectl port-forward service/garbage-classifier 30080:8080
```

Now it's accessible on port 30080.

## Horizontal Pod Autoscaling

### 1. Install metrics-server in kubectl

```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

### 2. Patch metrics-server to work without TLS

```bash
kubectl patch -n kube-system deployment metrics-server --type=json -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
```

### 3. Wait for metrics-server to be ready

```bash
kubectl get deployment metrics-server -n kube-system
```

### 4. Deploy HPA

```bash
kubectl apply -f hpa.yaml
```

### 5. Check HPA status

```bash
kubectl get hpa
kubectl describe hpa garbage-classifier-hpa
```

## Testing Autoscaling

### 1. Check that you can access the endpoint

```bash
curl http://localhost:30080/health
```

### 2. Run the test

```bash
uv run python test-hpa.py
```

### 3. Watch the HPA in another terminal

While running the load test:

```bash
kubectl get hpa -w
```

You should see the number of replicas increase as CPU usage rises.

### 4. Check Pods

```bash
kubectl get pods -w
```
