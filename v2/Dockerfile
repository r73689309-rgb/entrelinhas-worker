# Worker serverless RunPod — v2: ComfyUI + Z-Image Base + detailers + ControlNet + Qwen-Image-Edit
#
# Diferente da v1, NENHUM modelo grande entra na imagem: o handler baixa o que falta
# para o network volume no primeiro uso ({"preparar": [...]}) e o ComfyUI enxerga o
# volume pelo extra_model_paths.yaml. A imagem fica pequena (so programa e bibliotecas),
# o build cabe no runner do GitHub e o endpoint PRECISA de volume (lab: EU-RO-1).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    COMFY=/comfyui \
    AITK=/ai-toolkit \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3.10-venv python3-pip git wget curl unzip ca-certificates \
      build-essential cmake \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python -m pip install --upgrade pip setuptools wheel

# 1) torch (cu128)
RUN pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
      --index-url https://download.pytorch.org/whl/cu128

# 2) ComfyUI (atual: Z-Image, Qwen-Image-Edit e o no QwenImageDiffsynthControlnet sao nativos)
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git $COMFY \
    && grep -viE '^(torch|torchvision|torchaudio)([=<>~]|$)' $COMFY/requirements.txt > /tmp/req.txt \
    && pip install -r /tmp/req.txt

# 3) dependencias dos custom nodes
RUN pip install "numpy<2" cython \
 && pip install \
      onnx onnxruntime-gpu \
      torchsde timm ftfy \
      opencv-python-headless ultralytics segment_anything albumentations \
      "huggingface_hub>=0.25" accelerate einops gguf

# 4) custom nodes: detailers (rosto/maos), preprocessadores (DWPose/depth), GGUF (Qwen-Edit)
WORKDIR $COMFY/custom_nodes
RUN git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Pack.git \
 && git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git \
 && git clone --depth 1 https://github.com/Fannovel16/comfyui_controlnet_aux.git \
 && git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git \
 && for d in */ ; do \
      if [ -f "$d/requirements.txt" ]; then \
        grep -viE '^(torch|torchvision|torchaudio|onnxruntime|numpy)([=<>~]|$)' "$d/requirements.txt" > /tmp/n.txt || true; \
        pip install -r /tmp/n.txt || true; \
      fi; \
    done

# 5) pastas de modelos (vazias: o conteudo vem do volume)
WORKDIR $COMFY/models
RUN mkdir -p unet diffusion_models text_encoders vae loras controlnet model_patches \
             ultralytics/bbox ultralytics/segm sams clip_vision checkpoints \
 && rm -rf $COMFY/models/clip && ln -s text_encoders $COMFY/models/clip

# 6) Treinador: ai-toolkit (ostris) — e o treinador padrao para Z-Image em 2026.
#    Ambiente proprio que REAPROVEITA o torch do sistema. As bibliotecas de video/audio
#    e a interface web ficam de fora: so treinamos LoRA de imagem.
RUN git clone --depth 1 https://github.com/ostris/ai-toolkit.git $AITK \
 && python -m venv --system-site-packages /aitk-venv \
 && /aitk-venv/bin/pip install --no-cache-dir --upgrade pip \
 && grep -vhiE '^\s*(#|-r|torch==|torch$|torchvision|torchaudio|torchcodec|av==|librosa|mutagen|gradio|fastapi|uvicorn)' \
      $AITK/requirements_base.txt $AITK/requirements.txt > /tmp/aitk.txt \
 && cat /tmp/aitk.txt \
 && ( /aitk-venv/bin/pip install --no-cache-dir -r /tmp/aitk.txt \
      || ( sed -i 's/^torchao==.*/torchao/' /tmp/aitk.txt && /aitk-venv/bin/pip install --no-cache-dir -r /tmp/aitk.txt ) ) \
 && ( /aitk-venv/bin/python -c "import torchao" || /aitk-venv/bin/pip install --no-cache-dir torchao==0.10.0 || /aitk-venv/bin/pip install --no-cache-dir torchao ) \
 && /aitk-venv/bin/python -c "import torch, diffusers, transformers, peft, torchao; print('ai-toolkit ok', torch.__version__, diffusers.__version__, transformers.__version__)" \
 && test -f $AITK/run.py \
 && test -f $AITK/extensions_built_in/diffusion_models/z_image/z_image.py

# 7) SDK do runpod + handler
RUN pip install runpod requests
WORKDIR /
COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
