Garbage Classification with Deep Learning & Kubernetes
This is an end-to-end machine learning project that classifies garbage images into 10 categories using a deep learning model (Xception), deployed as a FastAPI microservice on Kubernetes with horizontal pod autoscaling.

Problem Statement
Waste management is a global challenge, with improper sorting leading to increased landfill waste, contamination of recyclables, and environmental harm. Manual garbage classification is:

- Time-consuming and labor-intensive
- Prone to human error
- Difficult to scale
- Inconsistent across different facilities

Solution: An automated garbage classification system that uses deep learning to accurately categorize waste items into 10 classes, enabling:

- Automated waste sorting in recycling facilities
- Real-time classification via API
- Scalable deployment on Kubernetes
- Easy integration with existing waste management systems

Model Details

Architecture: Xception (transfer learning from ImageNet)
Framework: TensorFlow/Keras
Input: 299×299 RGB images
Output: 10 garbage categories
Accuracy: ~94%
Format: ONNX (optimized for inference)

Categories

Battery
Biological (organic waste)
Cardboard
Clothes
Glass
Metal
Paper
Plastic
Shoes
Trash (general waste)

Dataset

Source: Garbage Classification v2 on Kaggle
Size: ~15,000 images
Split: 70% train, 15% validation, 15% test

### Prerequisites
System Requirements

OS: Windows 10/11, macOS, or Linux
RAM: 4GB minimum (8GB recommended)
Python: 3.13.5
Docker Desktop: Latest version
kubectl: v1.28+
kind: v0.20+


Tools Installation
1. Install Chocolatey (Windows)
powershell# Run PowerShell as Administrator
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
2. Install kubectl
Windows (Chocolatey):
bashchoco install kubernetes-cli
macOS (Homebrew):
bashbrew install kubectl
Linux:
bashcurl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
Verify:
bashkubectl version --client
3. Install kind (Kubernetes in Docker)
Windows (Chocolatey):
bashchoco install kind
macOS (Homebrew):
bashbrew install kind
Linux:
bashcurl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind
Verify:
bashkind version
4. Install uv (Python Package Manager)
macOS/Linux:
bashcurl -LsSf https://astral.sh/uv/install.sh | sh
Windows (PowerShell):
powershellpowershell -c "irm https://astral.sh/uv/install.ps1 | iex"
Verify:
bashuv --version
5. Install Docker Desktop
Download and install from: https://www.docker.com/products/docker-desktop
Verify:
bashdocker --version

Setup & Installation
1. Clone the Repository
bashgit clone https://github.com/yourusername/garbage-classification-kubernetes.git
cd garbage-classification-kubernetes
2. Create Virtual Environment with uv
bash# Create virtual environment
uv venv

# Activate environment
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Git Bash)
source .venv/Scripts/activate

# macOS/Linux
source .venv/bin/activate
3. Install Dependencies
bash# Install all dependencies
uv pip install fastapi uvicorn onnxruntime pillow requests pydantic numpy

# Or sync from pyproject.toml if available
uv sync