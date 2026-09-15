# Worker serverless RunPod — ComfyUI + FLUX + PuLID + ReActor + Wan 2.2
# Os modelos NAO vao na imagem: ficam no network volume, montado em /runpod-volume.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    COMFY=/comfyui \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3-pip git wget curl unzip ca-certificates \
      build-essential cmake \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python -m pip install --upgrade pip setuptools wheel

# 1) torch (cu126) — o ComfyUI atual (comfy_kitchen) exige torch >= 2.7
RUN pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
      --index-url https://download.pytorch.org/whl/cu126

# 2) ComfyUI
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git $COMFY \
    && grep -viE '^(torch|torchvision|torchaudio)([=<>~]|$)' $COMFY/requirements.txt > /tmp/req.txt \
    && pip install -r /tmp/req.txt

# 3) dependencias dos custom nodes
RUN pip install "numpy<2" cython \
 && pip install \
      insightface==0.7.3 onnx onnxruntime-gpu \
      facexlib torchsde timm ftfy \
      opencv-python-headless ultralytics segment_anything albumentations \
      "huggingface_hub>=0.25" accelerate einops

# 4) custom nodes
WORKDIR $COMFY/custom_nodes
RUN git clone --depth 1 https://github.com/balazik/ComfyUI-PuLID-Flux.git \
 && git clone --depth 1 https://github.com/Gourieff/ComfyUI-ReActor.git \
 && for d in */ ; do \
      if [ -f "$d/requirements.txt" ]; then \
        grep -viE '^(torch|torchvision|torchaudio|onnxruntime|insightface|numpy)([=<>~]|$)' "$d/requirements.txt" > /tmp/n.txt || true; \
        pip install -r /tmp/n.txt || true; \
      fi; \
    done

# 5) SDK do runpod + handler
RUN pip install runpod requests
WORKDIR /
COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
