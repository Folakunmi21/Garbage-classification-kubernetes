FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for downloading
RUN apt-get update && apt-get install -y git-lfs && rm -rf /var/lib/apt/lists/*

# Install specific versions to avoid AttributeErrors
RUN pip install --no-cache-dir \
    tensorflow==2.16.1 \
    tf2onnx==1.16.1 \
    onnx==1.16.0 \
    huggingface_hub

# Step 1: Download script
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='Folakunmi21/garbage-classifier', \
    allow_patterns=['xception_v4_final/*'], local_dir='.')"

# Step 2: Convert using the CLI (more reliable than the internal Python API)
# This maps the input/output and handles the graph conversion automatically
RUN python3 -m tf2onnx.convert \
    --saved-model xception_v4_final \
    --output xception_v4_final.onnx \
    --opset 13

# Step 3: Cleanup
RUN rm -rf xception_v4_final

CMD ["ls", "-lh", "xception_v4_final.onnx"]