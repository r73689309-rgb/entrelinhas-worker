# Worker de PRODUCAO (Z-Image): a mesma imagem do laboratorio (v2) + os modelos DENTRO.
# Sem network volume: o worker liga sem ler 20 GB do volume e pode rodar em
# qualquer datacenter (mais GPU disponivel). As LoRAs das personagens vem do
# Hugging Face privado do Rodrigo, baixadas pelo handler a cada pedido.
# Cada modelo numa camada propria: camadas menores sobem e descem com mais seguranca.
FROM ghcr.io/r73689309-rgb/entrelinhas-worker@sha256:d90172c0e45f165672d932c77758f41645c669056021a914145eacac6b7d8df0

ARG HF=https://huggingface.co
WORKDIR /comfyui/models

# gerador principal (Z-Image Base, bf16)
RUN wget -q --tries=5 -O diffusion_models/z_image_bf16.safetensors \
      $HF/Comfy-Org/z_image/resolve/main/split_files/diffusion_models/z_image_bf16.safetensors \
 && test $(stat -c%s diffusion_models/z_image_bf16.safetensors) -gt 1000000000

# codificador de texto
RUN wget -q --tries=5 -O text_encoders/qwen_3_4b.safetensors \
      $HF/Comfy-Org/z_image/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors \
 && test $(stat -c%s text_encoders/qwen_3_4b.safetensors) -gt 1000000000

# VAE + controle de pose
RUN wget -q --tries=5 -O vae/ae.safetensors \
      $HF/Comfy-Org/z_image/resolve/main/split_files/vae/ae.safetensors \
 && wget -q --tries=5 -O model_patches/Z-Image-Fun-Controlnet-Union-2.1.safetensors \
      $HF/alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1/resolve/main/Z-Image-Fun-Controlnet-Union-2.1.safetensors \
 && test $(stat -c%s vae/ae.safetensors) -gt 100000000 \
 && test $(stat -c%s model_patches/Z-Image-Fun-Controlnet-Union-2.1.safetensors) -gt 100000000

# detectores de rosto/mao + SAM (retoque das maos)
RUN mkdir -p ultralytics/bbox sams \
 && wget -q --tries=5 -O ultralytics/bbox/hand_yolov8s.pt $HF/Bingsu/adetailer/resolve/main/hand_yolov8s.pt \
 && wget -q --tries=5 -O ultralytics/bbox/face_yolov8m.pt $HF/Bingsu/adetailer/resolve/main/face_yolov8m.pt \
 && wget -q --tries=5 -O sams/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
 && ls -la diffusion_models text_encoders vae model_patches ultralytics/bbox sams

WORKDIR /
