# Worker serverless RunPod — ComfyUI + FLUX + PuLID + ReActor + Wan 2.2 + SDXL
# Os modelos ficam DENTRO da imagem: o endpoint nao depende de network volume
# e por isso pode rodar em qualquer datacenter.
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

# 1) torch (cu128) — o ComfyUI atual (comfy_kitchen) exige torch >= 2.7
RUN pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
      --index-url https://download.pytorch.org/whl/cu128

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

# 4b) Compatibilidade: o ComfyUI novo chama forward_orig com argumentos extras
#     (timestep_zero_index, transformer_options, attn_mask) que o no PuLID-Flux
#     nao conhece. Aceitamos e ignoramos — e o comportamento antigo.
RUN P=$COMFY/custom_nodes/ComfyUI-PuLID-Flux/pulidflux.py \
 && sed -i 's|^    control=None,$|    control=None,\n    timestep_zero_index=None,\n    transformer_options={},\n    attn_mask: Tensor = None,\n    **kwargs,|' $P \
 && grep -q "timestep_zero_index" $P \
 && python3 -c "import ast,sys; ast.parse(open('$P').read())" \
 && echo "pulidflux.py corrigido"

# ---------------------------------------------------------------- modelos
ARG HF=https://huggingface.co
WORKDIR $COMFY/models
RUN mkdir -p unet text_encoders vae loras pulid diffusion_models insightface facerestore_models checkpoints

# 5a) FLUX (foto)
RUN wget -q --show-progress -O unet/flux1-dev-fp8.safetensors \
      $HF/Kijai/flux-fp8/resolve/main/flux1-dev-fp8.safetensors \
 && wget -q -O text_encoders/t5xxl_fp8_e4m3fn.safetensors \
      $HF/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors \
 && wget -q -O text_encoders/clip_l.safetensors \
      $HF/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors \
 && wget -q -O vae/ae.safetensors \
      $HF/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors \
 && wget -q -O loras/boreal-v2.safetensors \
      $HF/kudzueye/boreal-flux-dev-v2/resolve/main/boreal-v2.safetensors

# 5b) PuLID (identidade do rosto) + EVA-CLIP no cache do HF
RUN wget -q -O pulid/pulid_flux_v0.9.1.safetensors \
      $HF/guozinan/PuLID/resolve/main/pulid_flux_v0.9.1.safetensors \
 && python -c "from huggingface_hub import hf_hub_download; hf_hub_download('QuanSun/EVA-CLIP','EVA02_CLIP_L_336_psz14_s6B.pt')"

# 5c) ReActor (troca de rosto) + antelopev2 (deteccao usada pelo PuLID)
RUN wget -q -O insightface/inswapper_128.onnx \
      $HF/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx \
 && wget -q -O facerestore_models/codeformer-v0.1.0.pth \
      $HF/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/codeformer-v0.1.0.pth \
 && mkdir -p insightface/models \
 && wget -q -O /tmp/antelopev2.zip $HF/MonsterMMORPG/tools/resolve/main/antelopev2.zip \
 && unzip -q /tmp/antelopev2.zip -d insightface/models/ && rm /tmp/antelopev2.zip

# 5d) Wan 2.2 (video)
RUN wget -q -O diffusion_models/wan2.2_ti2v_5B_fp16.safetensors \
      $HF/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors \
 && wget -q -O text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
      $HF/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
 && wget -q -O vae/wan2.2_vae.safetensors \
      $HF/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors

# 5e) SDXL (RealVisXL V5.0) — usado pelos geradores "Hibrido" e "RealVisXL" do app
RUN wget -q -O checkpoints/realvisxl5.safetensors \
      $HF/SG161222/RealVisXL_V5.0/resolve/main/RealVisXL_V5.0_fp16.safetensors \
 && test -s checkpoints/realvisxl5.safetensors

# 5f) LoRAs de realismo — varias opcoes para escolher no engrenagem e comparar
RUN wget -q -O loras/super-realism.safetensors \
      $HF/strangerzonehf/Flux-Super-Realism-LoRA/resolve/main/super-realism.safetensors \
 && wget -q -O loras/hdr-realism.safetensors \
      $HF/prithivMLmods/Flux.1-Dev-LoRA-HDR-Realism/resolve/main/HDR.safetensors \
 && wget -q -O loras/ultra-realism.safetensors \
      $HF/prithivMLmods/Canopus-LoRA-Flux-UltraRealism-2.0/resolve/main/Canopus-LoRA-Flux-UltraRealism.safetensors \
 && wget -q -O loras/face-realism.safetensors \
      $HF/prithivMLmods/Canopus-LoRA-Flux-FaceRealism/resolve/main/Canopus-LoRA-Flux-FaceRealism.safetensors \
 && wget -q -O loras/fine-detailed.safetensors \
      $HF/prithivMLmods/Flux-Realism-FineDetailed/resolve/main/Flux-Realism-FineDetailed.safetensors \
 && wget -q -O loras/koda-film.safetensors \
      $HF/alvdansen/flux-koda/resolve/main/araminta_k_flux_koda.safetensors \
 && wget -q -O loras/xlabs-realism.safetensors \
      $HF/XLabs-AI/flux-RealismLora/resolve/main/lora.safetensors \
 && for f in super-realism hdr-realism ultra-realism face-realism fine-detailed koda-film xlabs-realism; do \
      test -s loras/$f.safetensors || exit 1; \
    done

# compatibilidade: versoes antigas do ComfyUI procuram os text encoders em models/clip
RUN rm -rf $COMFY/models/clip && ln -s text_encoders $COMFY/models/clip

# 6) SDK do runpod + handler
RUN pip install runpod requests
WORKDIR /
COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
