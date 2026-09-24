"""
Handler serverless para ComfyUI (RunPod) — v2: Z-Image Base + ai-toolkit.

Entrada aceita:
  {"workflow": {...}}                      -> executa o workflow em formato API
  {"workflow": {...}, "images": [ {"name":"ref.png","image":"<base64>"} ]}
  {"workflow": {...}, "loras": [ {"url":"https://...","name":"tok.safetensors","token":"<hf>"} ]}
                                           -> garante o LoRA no MESMO worker antes de gerar
  {"get": "/object_info"}                  -> proxy GET para a API do ComfyUI
  {"ls": "loras"}                          -> lista os arquivos de modelo disponiveis
  {"download": {"url": "...", "dir": "loras", "name": "x.safetensors"}}
                                           -> baixa um modelo para a pasta de modelos
  {"put": {"path": "tok/img/10_tok/001.jpg", "b64": "..."}}
                                           -> grava um arquivo na pasta de treino do volume
  {"limpa_treino": "tok"}                  -> apaga o conjunto de treino daquela personagem
  {"poda": {"pasta":"tok_xl","manter":2}}  -> apaga os pontos antigos, guarda os N mais novos
  {"espaco_treino": 1}                     -> quanto cada pasta de treino ocupa e o que sobra
  {"treino_estado": "tok"}                 -> quantas fotos e quantos passos ja treinados
  {"preparar": ["zimage","controlnet","detailer","qwenedit"]}
                                           -> baixa para o volume os modelos que faltam (v2)
  {"treina": {"token":"tok","passos":400,"total":1600,"base":"zimage","dim":16,"alpha":16}}
                                           -> treina UM pedaco (ai-toolkit, Z-Image Base)
  {"publicar": {"token":"<hf>","repo":"user/loras","nome":"tok.safetensors","pasta":"tok_xl"}}
                                           -> sobe o LoRA para um repo privado do HF
Saida:
  {"images":[{"filename":..., "mime":..., "data":"<base64>"}], "seconds": 12.3}
"""
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

import runpod

COMFY = os.environ.get("COMFY", "/comfyui")
HOST = "127.0.0.1:8188"
OUT_DIR = "/tmp/comfy-out"
TMP_DIR = "/tmp/comfy-tmp"
BOOT_TIMEOUT = int(os.environ.get("BOOT_TIMEOUT", "600"))
JOB_TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "900"))

_proc = None


# ---------------------------------------------------------------- modelos
VOL_ROOTS = ("/runpod-volume/models_store",
             "/runpod-volume/ComfyUI/models",
             "/runpod-volume/models")
IMG_ROOT = os.path.join(COMFY, "models")

# pastas que o app usa; o ComfyUI aceita varios caminhos por chave
MODEL_DIRS = ("checkpoints", "unet", "diffusion_models", "loras", "vae",
              "text_encoders", "clip", "clip_vision", "controlnet",
              "upscale_models", "embeddings", "pulid", "insightface",
              "facerestore_models", "style_models", "gligen", "ultralytics",
              "model_patches", "sams")


def volume_root():
    """Raiz de modelos no network volume, se houver um montado."""
    for r in VOL_ROOTS:
        if os.path.isdir(r):
            return r
    # volume montado mas ainda vazio: cria a raiz padrao
    if os.path.isdir("/runpod-volume"):
        try:
            os.makedirs(VOL_ROOTS[0], exist_ok=True)
            return VOL_ROOTS[0]
        except Exception as e:
            print("[worker] nao consegui criar a raiz no volume:", e)
    return None


def link_models():
    """Faz o ComfyUI enxergar os modelos da imagem E os do volume ao mesmo tempo.

    Antes isso era feito com symlink, e uma pasta que ja existia cheia na imagem
    (loras, por exemplo) fazia a do volume ser ignorada. Agora escrevemos o
    extra_model_paths.yaml, que o ComfyUI le na inicializacao e que SOMA os
    caminhos em vez de substituir.
    """
    vol = volume_root()
    cfg = os.path.join(COMFY, "extra_model_paths.yaml")
    if not vol:
        print("[worker] sem network volume: usando so os modelos da imagem")
        try:
            if os.path.exists(cfg):
                os.remove(cfg)
        except Exception:
            pass
        return
    for name in MODEL_DIRS:
        try:
            os.makedirs(os.path.join(vol, name), exist_ok=True)
        except Exception:
            pass
    linhas = ["volume:", "  base_path: %s" % vol, "  is_default: false"]
    for name in MODEL_DIRS:
        linhas.append("  %s: %s" % (name, name))
    # o Impact Subpack lista os detectores por "ultralytics_bbox"/"ultralytics_segm",
    # e nao por "ultralytics": sem isso o detector de maos do volume nao aparece
    linhas.append("  ultralytics_bbox: ultralytics/bbox")
    linhas.append("  ultralytics_segm: ultralytics/segm")
    try:
        with open(cfg, "w") as f:
            f.write("\n".join(linhas) + "\n")
        print("[worker] volume em", vol, "- modelos da imagem e do volume somados")
    except Exception as e:
        print("[worker] nao consegui escrever", cfg, e)


SAFE = re.compile(r"^[A-Za-z0-9._-]+$")


def models_root():
    """Onde GRAVAR: o volume quando existir (persiste), senao a imagem."""
    return volume_root() or (IMG_ROOT if os.path.isdir(IMG_ROOT) else None)


def _scan(root, names):
    out = {}
    if not root or not os.path.isdir(root):
        return out
    for name in (names or sorted(os.listdir(root))):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        files = []
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if os.path.isfile(p):
                files.append({"name": f, "mb": round(os.path.getsize(p) / 1048576, 1)})
        out[name] = files
    return out


def list_models(which=None):
    """Lista a uniao dos modelos da imagem e do volume, sem repetir nomes."""
    vol = volume_root()
    roots = [r for r in (IMG_ROOT, vol) if r and os.path.isdir(r)]
    if not roots:
        return {"error": "nenhum diretorio de modelos encontrado"}
    names = [which] if isinstance(which, str) and which else None
    juntos = {}
    for r in roots:
        for pasta, arquivos in _scan(r, names).items():
            alvo = juntos.setdefault(pasta, [])
            ja = {x["name"] for x in alvo}
            alvo.extend(a for a in arquivos if a["name"] not in ja)
    for pasta in juntos:
        juntos[pasta].sort(key=lambda x: x["name"])
    livre = shutil.disk_usage(vol or IMG_ROOT)
    return {"root": vol or IMG_ROOT, "volume": bool(vol), "dirs": juntos,
            "free_gb": round(livre.free / 1073741824, 1),
            "total_gb": round(livre.total / 1073741824, 1)}


def download_model(spec):
    root = models_root()
    if not root:
        return {"error": "nenhum diretorio de modelos encontrado"}
    url = (spec.get("url") or "").strip()
    if not url.lower().startswith(("http://", "https://")):
        return {"error": "url invalida"}
    sub = (spec.get("dir") or "").strip()
    name = (spec.get("name") or "").strip() or os.path.basename(url.split("?")[0])
    if not SAFE_REL.match(sub or "x") or ".." in sub or not SAFE.match(name):
        return {"error": "nome de pasta ou arquivo invalido"}
    dest_dir = os.path.join(root, sub) if sub else root
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, name)
    if os.path.exists(dest) and not spec.get("overwrite"):
        return {"ok": True, "path": dest, "mb": round(os.path.getsize(dest) / 1048576, 1),
                "note": "ja existia; nada foi baixado"}
    headers = {"User-Agent": "entrelinhas-worker"}
    if spec.get("token"):
        headers["Authorization"] = "Bearer " + spec["token"]
    req = urllib.request.Request(url, headers=headers)
    # nome temporario UNICO: dois workers baixando o mesmo arquivo ao mesmo tempo
    # nao podem escrever no mesmo .part (foi assim que um download corrompeu o outro)
    tmp = dest + "." + uuid.uuid4().hex[:8] + ".part"
    t0 = time.time()
    esperado = None
    baixado = 0
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            try:
                esperado = int(r.headers.get("Content-Length") or 0) or None
            except Exception:
                esperado = None
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                baixado += len(chunk)
    except Exception as e:
        try:
            os.remove(tmp)
        except Exception:
            pass
        return {"error": f"falha ao baixar: {e}"}
    if esperado and baixado != esperado:
        try:
            os.remove(tmp)
        except Exception:
            pass
        return {"error": "download incompleto: %d de %d bytes" % (baixado, esperado)}
    if os.path.exists(dest) and not spec.get("overwrite"):
        # outro worker terminou antes: fica com o dele, joga o nosso fora
        try:
            os.remove(tmp)
        except Exception:
            pass
        return {"ok": True, "path": dest, "mb": round(os.path.getsize(dest) / 1048576, 1),
                "note": "outro worker baixou antes"}
    os.replace(tmp, dest)
    mb = round(os.path.getsize(dest) / 1048576, 1)
    persistente = bool(volume_root()) and dest.startswith("/runpod-volume")
    extracted = None
    if spec.get("unzip"):
        try:
            import zipfile
            out = os.path.join(dest_dir, "models") if sub == "insightface" else dest_dir
            os.makedirs(out, exist_ok=True)
            with zipfile.ZipFile(dest) as z:
                bad = [n for n in z.namelist() if n.startswith("/") or ".." in n]
                if bad:
                    return {"error": "zip com caminhos suspeitos", "detail": bad[:5]}
                z.extractall(out)
                extracted = z.namelist()[:20]
            os.remove(dest)
        except Exception as e:
            return {"error": f"baixou mas nao descompactou: {e}", "path": dest, "mb": mb}
    try:
        api_get("/object_info/CheckpointLoaderSimple")
    except Exception:
        pass
    return {"ok": True, "path": dest, "mb": mb, "extracted": extracted,
            "persistente": persistente,
            "aviso": ("guardado no volume — fica para sempre" if persistente
                      else "SEM volume: este arquivo some quando o worker hibernar"),
            "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- manifesto v2
# Tudo o que a v2 usa mora no volume. Cada grupo e baixado sob demanda.
HF = "https://huggingface.co"
MANIFESTO = {
    "zimage": [  # gerador principal (Z-Image Base) + codificador de texto + VAE
        ("diffusion_models", "z_image_bf16.safetensors",
         HF + "/Comfy-Org/z_image/resolve/main/split_files/diffusion_models/z_image_bf16.safetensors"),
        ("text_encoders", "qwen_3_4b.safetensors",
         HF + "/Comfy-Org/z_image/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors"),
        ("vae", "ae.safetensors",
         HF + "/Comfy-Org/z_image/resolve/main/split_files/vae/ae.safetensors"),
    ],
    "controlnet": [  # silhueta/pose para o Z-Image (Alibaba PAI, Apache 2.0) — no QwenImageDiffsynthControlnet
        ("model_patches", "Z-Image-Fun-Controlnet-Union-2.1.safetensors",
         HF + "/alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1/resolve/main/Z-Image-Fun-Controlnet-Union-2.1.safetensors"),
    ],
    "detailer": [  # detectores de rosto e mao (Impact Pack) + SAM
        ("ultralytics/bbox", "face_yolov8m.pt", HF + "/Bingsu/adetailer/resolve/main/face_yolov8m.pt"),
        ("ultralytics/bbox", "hand_yolov8s.pt", HF + "/Bingsu/adetailer/resolve/main/hand_yolov8s.pt"),
        ("sams", "sam_vit_b_01ec64.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"),
    ],
    "qwenedit": [  # "Juntos": edicao por referencia com varias pessoas (Qwen-Image-Edit-2511, GGUF Q6)
        ("unet", "Qwen-Image-Edit-2511-Q6_K.gguf",
         HF + "/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/Qwen-Image-Edit-2511-Q6_K.gguf"),
        ("text_encoders", "qwen_2.5_vl_7b_fp8_scaled.safetensors",
         HF + "/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
        ("vae", "qwen_image_vae.safetensors",
         HF + "/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors"),
        ("loras", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
         HF + "/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"),
    ],
    "sdxl": [  # so para comparar com o que tinhamos (RealVisXL 5)
        ("checkpoints", "realvisxl5.safetensors",
         HF + "/SG161222/RealVisXL_V5.0/resolve/main/RealVisXL_V5.0_fp16.safetensors"),
    ],
}


def _tem(sub, name):
    for raiz in [r for r in (IMG_ROOT, volume_root()) if r]:
        p = os.path.join(raiz, sub, name)
        if os.path.isfile(p) and os.path.getsize(p) > 1024:
            return p
    return None


def _tamanho_remoto(url, token=None):
    """Content-Length do arquivo no servidor (seguindo redirecionamentos)."""
    try:
        headers = {"User-Agent": "entrelinhas-worker"}
        if token:
            headers["Authorization"] = "Bearer " + token
        req = urllib.request.Request(url, headers=headers, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers.get("Content-Length") or 0) or None
    except Exception:
        return None


def preparar(grupos, token=None, verificar=True):
    """Garante no volume os arquivos dos grupos pedidos. Idempotente.
    Com verificar=True confere o tamanho de cada arquivo com o servidor e
    baixa de novo o que estiver truncado/corrompido."""
    if not volume_root():
        return {"error": "este endpoint nao tem network volume: a v2 precisa de um"}
    if isinstance(grupos, str):
        grupos = [grupos]
    grupos = [g for g in (grupos or []) if g in MANIFESTO] or ["zimage"]
    feito, baixado, erros, refeitos = [], [], [], []
    t0 = time.time()
    for g in grupos:
        for sub, name, url in MANIFESTO[g]:
            local = _tem(sub, name)
            if local:
                if verificar:
                    esperado = _tamanho_remoto(url, token)
                    real = os.path.getsize(local)
                    if esperado and real != esperado:
                        refeitos.append({"arquivo": sub + "/" + name, "local": real, "esperado": esperado})
                        try:
                            os.remove(local)
                        except Exception as e:
                            erros.append({"arquivo": sub + "/" + name, "erro": "nao consegui apagar o corrompido: %s" % e})
                            continue
                    else:
                        feito.append(sub + "/" + name)
                        continue
                else:
                    feito.append(sub + "/" + name)
                    continue
            # limpa restos de .part antigos desse arquivo
            try:
                d = os.path.join(volume_root(), sub)
                for f in os.listdir(d) if os.path.isdir(d) else []:
                    if f.startswith(name + ".") and f.endswith(".part"):
                        os.remove(os.path.join(d, f))
            except Exception:
                pass
            r = download_model({"url": url, "dir": sub, "name": name, "token": token})
            if r.get("error"):
                erros.append({"arquivo": sub + "/" + name, "erro": r["error"]})
            else:
                baixado.append({"arquivo": sub + "/" + name, "mb": r.get("mb")})
    return {"ok": not erros, "ja_tinha": feito, "baixado": baixado, "erros": erros,
            "refeitos": refeitos, "livre_gb": _livre_gb(), "segundos": round(time.time() - t0, 1)}


def falta(grupo):
    return [sub + "/" + name for sub, name, _ in MANIFESTO.get(grupo, []) if not _tem(sub, name)]


# ---------------------------------------------------------------- comfyui
def comfy_up():
    try:
        with urllib.request.urlopen(f"http://{HOST}/system_stats", timeout=3):
            return True
    except Exception:
        return False


def start_comfy():
    global _proc
    if _proc and _proc.poll() is None:
        while not comfy_up():
            time.sleep(1)
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    link_models()
    cmd = [
        sys.executable, "main.py",
        "--listen", "127.0.0.1", "--port", "8188",
        "--disable-auto-launch", "--disable-metadata",
        "--output-directory", OUT_DIR,
        "--temp-directory", TMP_DIR,
    ]
    print("[worker] iniciando ComfyUI...")
    _proc = subprocess.Popen(cmd, cwd=COMFY)
    t0 = time.time()
    while time.time() - t0 < BOOT_TIMEOUT:
        if comfy_up():
            print(f"[worker] ComfyUI pronto em {time.time()-t0:.1f}s")
            return
        if _proc.poll() is not None:
            raise RuntimeError("ComfyUI encerrou durante a inicializacao")
        time.sleep(1)
    raise RuntimeError("ComfyUI nao respondeu dentro do tempo limite")


def api_get(path):
    with urllib.request.urlopen(f"http://{HOST}{path}", timeout=60) as r:
        return json.loads(r.read().decode())


def api_post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://{HOST}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def upload_image(name, b64):
    """Envia uma imagem de referencia para o input do ComfyUI (multipart)."""
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    boundary = "----rp" + uuid.uuid4().hex
    parts = []
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
                 f'filename="{name}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode())
    parts.append(raw)
    parts.append(f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="overwrite"'
                 f'\r\n\r\ntrue\r\n--{boundary}--\r\n'.encode())
    body = b"".join(parts)
    req = urllib.request.Request(
        f"http://{HOST}/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".gif": "image/gif", ".mp4": "video/mp4"}


def collect(history_entry):
    out = []
    for node_out in history_entry.get("outputs", {}).values():
        for key in ("images", "gifs", "videos"):
            for item in node_out.get(key, []) or []:
                sub = item.get("subfolder") or ""
                folder = item.get("type") or "output"
                base = OUT_DIR if folder == "output" else TMP_DIR
                path = os.path.join(base, sub, item["filename"])
                if not os.path.exists(path):
                    continue
                ext = os.path.splitext(path)[1].lower()
                with open(path, "rb") as f:
                    out.append({
                        "filename": item["filename"],
                        "mime": MIME.get(ext, "application/octet-stream"),
                        "data": base64.b64encode(f.read()).decode(),
                    })
    return out


def delete_model(spec):
    """Remove um arquivo de modelo. So dentro das pastas conhecidas, so no volume
    (o que esta na imagem volta no proximo worker, entao apagar nao adianta)."""
    vol = volume_root()
    if not vol:
        return {"error": "sem network volume: nada para apagar aqui"}
    sub = (spec.get("dir") or "").strip()
    name = (spec.get("name") or "").strip()
    if sub not in MODEL_DIRS:
        return {"error": "pasta desconhecida: %s" % sub}
    if not SAFE.match(name or "x"):
        return {"error": "nome de arquivo invalido"}
    alvo = os.path.realpath(os.path.join(vol, sub, name))
    raiz = os.path.realpath(os.path.join(vol, sub))
    # trava: o caminho final tem que continuar dentro da pasta de modelos
    if not alvo.startswith(raiz + os.sep):
        return {"error": "caminho fora da pasta de modelos"}
    if not os.path.exists(alvo):
        return {"error": "nao encontrei %s em %s (talvez esteja na imagem, e nao no volume)" % (name, sub)}
    mb = 0
    try:
        if os.path.isdir(alvo):
            import shutil
            for r, _, fs in os.walk(alvo):
                for x in fs:
                    try: mb += os.path.getsize(os.path.join(r, x))
                    except Exception: pass
            shutil.rmtree(alvo)
        else:
            mb = os.path.getsize(alvo)
            os.remove(alvo)
    except Exception as e:
        return {"error": "nao consegui apagar: %s" % e}
    livre = 0
    try:
        st = os.statvfs(vol); livre = round(st.f_bavail * st.f_frsize / (1024 ** 3), 1)
    except Exception:
        pass
    return {"ok": True, "apagado": name, "dir": sub,
            "mb": round(mb / 1048576, 1), "free_gb": livre}


# ---------------------------------------------------------------- treino de LoRA
# O treino nao cabe em um job (30 min); entao ele roda em PEDACOS: cada job treina
# alguns passos partindo do LoRA do pedaco anterior e devolve o arquivo no volume.
# O app encadeia os pedacos e mostra o progresso.
AITK = os.environ.get("AITK", "/ai-toolkit")
AITK_PY = "/aitk-venv/bin/python" if os.path.isfile("/aitk-venv/bin/python") else sys.executable
SAFE_REL = re.compile(r"^[A-Za-z0-9._\-/]+$")
TOKEN_RE = re.compile(r"^[a-z0-9]{2,32}$")
# A pasta no volume pode diferir da palavra-chave: a mesma personagem pode ter
# um treino de FLUX e um de SDXL, e misturar os dois quebra a continuacao.
PASTA_RE = re.compile(r"^[a-z0-9]{2,32}(_xl|_zi)?$")


def _pasta(spec_ou_token, base=None):
    if isinstance(spec_ou_token, dict):
        p = (spec_ou_token.get("pasta") or spec_ou_token.get("token") or "").strip().lower()
    else:
        p = (spec_ou_token or "").strip().lower()
    if base == "sdxl" and not p.endswith("_xl"):
        p += "_xl"
    if base == "zimage" and not p.endswith("_zi"):
        p += "_zi"
    return p if PASTA_RE.match(p) else ""


def vol_base():
    return "/runpod-volume" if os.path.isdir("/runpod-volume") else None


def treino_raiz():
    b = vol_base()
    return os.path.join(b, "treino") if b else None


def _dentro(caminho, raiz):
    return os.path.abspath(caminho).startswith(os.path.abspath(raiz) + os.sep)


def put_file(spec):
    """Recebe um pedaco de arquivo do app e grava na pasta de treino do volume."""
    raiz = treino_raiz()
    if not raiz:
        return {"error": "este endpoint nao tem volume de rede: use o laboratorio"}
    rel = (spec.get("path") or "").strip().lstrip("/")
    if not rel or ".." in rel or not SAFE_REL.match(rel):
        return {"error": "caminho invalido: %s" % rel}
    dest = os.path.join(raiz, rel)
    if not _dentro(dest, raiz):
        return {"error": "caminho fora da pasta de treino"}
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        dados = base64.b64decode(spec.get("b64") or "")
        with open(dest, "ab" if spec.get("append") else "wb") as f:
            f.write(dados)
        return {"ok": True, "path": rel, "bytes": os.path.getsize(dest)}
    except Exception as e:
        return {"error": "nao consegui gravar: %s" % e}


def limpa_treino(token):
    """Apaga o conjunto anterior daquela personagem, para nao misturar fotos."""
    raiz = treino_raiz()
    if not raiz:
        return {"error": "este endpoint nao tem volume de rede: use o laboratorio"}
    token = _pasta(token)
    if not token:
        return {"error": "palavra-chave invalida"}
    alvo = os.path.join(raiz, token)
    if not _dentro(alvo, raiz):
        return {"error": "caminho fora da pasta de treino"}
    try:
        if os.path.isdir(alvo):
            shutil.rmtree(alvo)
        return {"ok": True, "apagado": token}
    except Exception as e:
        return {"error": "nao consegui apagar: %s" % e}


def _acha_ckpt(pref=""):
    """O checkpoint SDXL (RealVisXL) usado para treinar LoRA de SDXL."""
    pads = []
    if pref:
        pads.append(re.escape(pref))
    pads += [r"realvis", r"\.safetensors$"]
    for pasta in ("checkpoints", "Stable-diffusion"):
        a = _acha(pasta, pads)
        if a:
            return a
    return None


def _acha(pasta, padroes):
    """Procura um arquivo na imagem e no volume, na ordem dos padroes."""
    for raiz in [r for r in (IMG_ROOT, volume_root()) if r and os.path.isdir(r)]:
        d = os.path.join(raiz, pasta)
        if not os.path.isdir(d):
            continue
        arquivos = sorted(os.listdir(d))
        for p in padroes:
            for f in arquivos:
                if re.search(p, f, re.I):
                    return os.path.join(d, f)
    return None


def estado_treino(token):
    """Quantos passos ja foram treinados e qual e o ultimo arquivo."""
    raiz = treino_raiz()
    token = _pasta(token)
    if not raiz or not token:
        return {"error": "palavra-chave invalida"}
    base = os.path.join(raiz, token)
    imgs = os.path.join(base, "img", "10_" + token)
    saida = _saida(token)   # ai-toolkit: saida/<nome do job>/ (token aqui ja e a pasta)
    fotos = 0
    if os.path.isdir(imgs):
        fotos = len([f for f in os.listdir(imgs) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    partes = []
    quebrados = []
    if os.path.isdir(saida):
        quebrados = apaga_pontos_quebrados(token)
        partes = sorted(f for f in os.listdir(saida) if f.endswith(".safetensors"))
    feitos = 0
    for p in partes:
        m = re.search(r"_(\d{6,9})\.safetensors$", p)
        if m:
            feitos = max(feitos, int(m.group(1)))
    # o arquivo final (sem numero) tem o passo no metadado
    for p in partes:
        if not re.search(r"_\d{6,9}\.safetensors$", p):
            try:
                feitos = max(feitos, int(_passo_meta(os.path.join(saida, p)) or 0))
            except Exception:
                pass
    partes.sort(key=lambda f: (int((re.search(r"_(\d{6,9})\.safetensors$", f) or [0, "0"])[1]) if re.search(r"_(\d{6,9})\.safetensors$", f) else 10**9))
    return {"ok": True, "token": token, "fotos": fotos, "passos_feitos": feitos,
            "partes": partes, "ultimo": (partes[-1] if partes else None),
            "quebrados": quebrados, "livre_gb": _livre_gb()}


def _passo_meta(caminho):
    """Le training_info.step do cabecalho do safetensors (ai-toolkit grava la)."""
    import struct
    with open(caminho, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        cab = json.loads(f.read(min(n, 4_000_000)).decode("utf-8", "ignore"))
    meta = (cab.get("__metadata__") or {})
    ti = meta.get("training_info") or meta.get("ss_training_info") or "{}"
    if isinstance(ti, str):
        ti = json.loads(ti)
    return ti.get("step")


def _livre_gb():
    try:
        u = shutil.disk_usage(vol_base() or "/")
        return round(u.free / 1073741824, 2)
    except Exception:
        return None


def _lora_ok(caminho):
    """Um .safetensors comeca com 8 bytes (tamanho do cabecalho) + JSON.
    Um arquivo cortado no meio (disco cheio) nao passa nem por isso."""
    try:
        tam = os.path.getsize(caminho)
        with open(caminho, "rb") as f:
            cab = f.read(8)
            if len(cab) < 8:
                return False
            n = int.from_bytes(cab, "little")
            if n <= 0 or n > 100_000_000 or 8 + n > tam:
                return False
            js = f.read(n)
        json.loads(js.decode("utf-8"))
        # o corpo dos tensores tem que existir alem do cabecalho
        return tam > 8 + n + 1024
    except Exception:
        return False


def _saida(pasta):
    """Onde o ai-toolkit grava os pontos: <treino>/<pasta>/saida/<pasta>/."""
    raiz = treino_raiz()
    return os.path.join(raiz, pasta, "saida", pasta) if raiz and pasta else None


def _passo_de(f):
    m = re.search(r"_(\d{6,9})\.safetensors$", f)
    return int(m.group(1)) if m else None


def apaga_pontos_quebrados(pasta):
    """Remove pontos truncados: um deles seria carregado como 'ultimo' e derrubaria o treino."""
    raiz = treino_raiz()
    pasta = _pasta(pasta)
    if not raiz or not pasta:
        return []
    d = _saida(pasta)
    if not os.path.isdir(d):
        return []
    fora = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".safetensors") and not _lora_ok(os.path.join(d, f)):
            try:
                os.remove(os.path.join(d, f))
                fora.append(f)
            except Exception:
                pass
    return fora


def poda_pontos(pasta, manter=2):
    """Apaga os pontos antigos do treino, guardando so os N mais novos.

    Cada pedaco grava um .safetensors de ~170 MB. Sem poda, um treino de 3000
    passos em pedacos de 400 deixa 8 arquivos (1,4 GB) e enche o volume — foi
    exatamente o que aconteceu ("Disk quota exceeded" na hora de salvar).
    """
    raiz = treino_raiz()
    pasta = _pasta(pasta)
    if not raiz or not pasta:
        return {"error": "pasta invalida"}
    d = _saida(pasta)
    if not os.path.isdir(d):
        return {"ok": True, "apagados": [], "mb": 0, "livre_gb": _livre_gb()}
    quebrados = apaga_pontos_quebrados(pasta)
    # so os pontos numerados entram na poda; o final (sem numero) fica sempre
    arquivos = sorted((f for f in os.listdir(d) if f.endswith(".safetensors") and _passo_de(f) is not None),
                      key=_passo_de)
    manter = max(1, int(manter or 2))
    velhos = arquivos[:-manter] if len(arquivos) > manter else []
    mb = 0.0
    apagados = []
    for f in velhos:
        try:
            cam = os.path.join(d, f)
            mb += os.path.getsize(cam) / 1048576
            os.remove(cam)
            apagados.append(f)
        except Exception:
            pass
    return {"ok": True, "apagados": apagados, "quebrados": quebrados, "mb": round(mb, 1),
            "guardados": arquivos[-manter:], "livre_gb": _livre_gb()}


def _tamanho_dir(base):
    total = 0
    for r, _, fs in os.walk(base):
        for f in fs:
            try:
                total += os.path.getsize(os.path.join(r, f))
            except Exception:
                pass
    return total


def espaco_treino():
    """Quanto cada pasta de treino ocupa, e o que mais ocupa o volume.

    O disk_usage do Linux mostra o disco FISICO do servidor (centenas de TB),
    nao a cota do volume — a cota nao e visivel daqui. Entao medimos o que
    esta gravado: cada pasta de treino e cada pasta de modelo no volume.
    """
    raiz = treino_raiz()
    if not raiz:
        return {"error": "este endpoint nao tem volume de rede: use o laboratorio"}
    vb = vol_base()
    volume = []   # pastas de primeiro nivel do volume, com tamanho
    try:
        for nome in sorted(os.listdir(vb)):
            cam = os.path.join(vb, nome)
            if os.path.isdir(cam):
                volume.append({"pasta": nome, "mb": round(_tamanho_dir(cam) / 1048576, 1)})
        if os.path.isdir(os.path.join(vb, "models_store")):
            for nome in sorted(os.listdir(os.path.join(vb, "models_store"))):
                cam = os.path.join(vb, "models_store", nome)
                if os.path.isdir(cam):
                    volume.append({"pasta": "models_store/" + nome, "mb": round(_tamanho_dir(cam) / 1048576, 1)})
    except Exception as e:
        print("[worker] nao consegui medir o volume:", e)
    volume.sort(key=lambda x: -x["mb"])
    pastas = []
    if os.path.isdir(raiz):
        for nome in sorted(os.listdir(raiz)):
            base = os.path.join(raiz, nome)
            if not os.path.isdir(base):
                continue
            total = 0
            pontos = 0
            for r, _, fs in os.walk(base):
                for f in fs:
                    try:
                        total += os.path.getsize(os.path.join(r, f))
                    except Exception:
                        pass
                    if f.endswith(".safetensors"):
                        pontos += 1
            pastas.append({"pasta": nome, "mb": round(total / 1048576, 1), "pontos": pontos})
    pastas.sort(key=lambda x: -x["mb"])
    total_mb = round(sum(x["mb"] for x in volume if "/" not in x["pasta"]), 1)
    return {"ok": True, "pastas": pastas, "volume": volume, "total_mb": total_mb,
            "raiz": raiz, "livre_gb": None}


def treina_lora(spec):
    """Roda UM pedaco do treino no ai-toolkit (Z-Image Base). O app chama de novo
    ate chegar no total; o ai-toolkit retoma sozinho do ultimo ponto salvo."""
    raiz = treino_raiz()
    if not raiz:
        return {"error": "este endpoint nao tem volume de rede: use o laboratorio"}
    token = (spec.get("token") or "").strip().lower()
    if not TOKEN_RE.match(token):
        return {"error": "palavra-chave invalida (use so letras e numeros)"}
    if not os.path.isfile(os.path.join(AITK, "run.py")):
        return {"error": "o treinador (ai-toolkit) nao esta nesta imagem do worker"}
    base_treino = "zimage"
    pasta = _pasta(spec, base_treino)
    if not pasta:
        return {"error": "pasta de treino invalida"}
    base = os.path.join(raiz, pasta)
    dados = os.path.join(base, "img", "10_" + pasta)
    if not os.path.isdir(dados):
        # tolera o layout antigo (img/10_<token>)
        alt = os.path.join(base, "img", "10_" + token)
        if os.path.isdir(alt):
            dados = alt
    saida = os.path.join(base, "saida")
    os.makedirs(saida, exist_ok=True)
    if not os.path.isdir(dados):
        return {"error": "nao achei as fotos: mande o conjunto antes de treinar"}
    fotos = [f for f in os.listdir(dados) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if len(fotos) < 5:
        return {"error": "poucas fotos na pasta (%d): mande o conjunto antes de treinar" % len(fotos)}

    faltando = falta("zimage")
    if faltando:
        r = preparar(["zimage"], spec.get("hf"))
        if r.get("erros"):
            return {"error": "faltam arquivos do Z-Image e nao consegui baixar", "detail": r["erros"]}
    modelo = _tem("diffusion_models", "z_image_bf16.safetensors")

    est = estado_treino(pasta)
    feitos = est.get("passos_feitos", 0)
    passos = max(50, min(1500, int(spec.get("passos") or 500)))
    total = max(passos, min(6000, int(spec.get("total") or 2000)))
    if feitos >= total:
        return {"ok": True, "pronto": True, "passos_feitos": feitos, "arquivo": est.get("ultimo")}
    alvo = min(total, feitos + passos)

    try:
        poda_pontos(pasta, int(spec.get("manter") or 2))
    except Exception as e:
        print("[worker] nao consegui podar pontos antigos:", e)

    dim = max(4, min(64, int(spec.get("dim") or 16)))
    alpha = max(1, min(dim, int(spec.get("alpha") or dim)))
    lr = str(spec.get("lr") or "1e-4")
    salvar_cada = max(50, min(passos, int(spec.get("salvar_cada") or 250)))
    nome = pasta  # o ai-toolkit retoma pelo nome do job: precisa ser estavel
    hf_home = os.path.join(vol_base(), "hf")
    os.makedirs(hf_home, exist_ok=True)
    cfg = {
        "job": "extension",
        "config": {
            "name": nome,
            "process": [{
                "type": "sd_trainer",
                "training_folder": saida,
                "device": "cuda:0",
                "trigger_word": token,
                "network": {"type": "lora", "linear": dim, "linear_alpha": alpha},
                "save": {"dtype": "bf16", "save_every": salvar_cada, "max_step_saves_to_keep": 3},
                "datasets": [{
                    "folder_path": dados, "caption_ext": "txt",
                    "caption_dropout_rate": 0.05, "shuffle_tokens": False,
                    "cache_latents_to_disk": True,
                    "resolution": [768, 1024],
                }],
                "train": {
                    "batch_size": 1, "steps": alvo, "gradient_accumulation": 1,
                    "train_unet": True, "train_text_encoder": False,
                    "cache_text_embeddings": True,
                    "gradient_checkpointing": True, "noise_scheduler": "flowmatch",
                    "optimizer": "adamw8bit", "lr": float(lr), "dtype": "bf16",
                    "skip_first_sample": True, "disable_sampling": True,
                },
                "model": {
                    "name_or_path": modelo,
                    "extras_name_or_path": "Tongyi-MAI/Z-Image",
                    "arch": "zimage",
                    "quantize": True, "qtype": "qfloat8",
                    "quantize_te": True, "qtype_te": "qfloat8",
                    "low_vram": True,
                },
                "sample": {"sampler": "flowmatch", "sample_every": 100000, "width": 1024, "height": 1024, "prompts": []},
            }],
        },
        "meta": {"name": nome, "version": "1.0"},
    }
    cfg_path = os.path.join(base, "treino.yaml")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=1)  # YAML aceita JSON

    env = dict(os.environ)
    env.update({"HF_HOME": hf_home, "HF_HUB_ENABLE_HF_TRANSFER": "0", "DISABLE_TELEMETRY": "1"})
    if spec.get("hf"):
        env["HF_TOKEN"] = spec["hf"]
    cmd = [AITK_PY, os.path.join(AITK, "run.py"), cfg_path]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=AITK, capture_output=True, text=True, env=env,
                           timeout=max(120, int(spec.get("limite") or 1700)))
    except subprocess.TimeoutExpired:
        return {"error": "o pedaco passou do tempo do job: diminua os passos por pedaco"}
    cauda = ((p.stdout or "")[-1500:] + "\n" + (p.stderr or "")[-2500:]).strip()
    if spec.get("hf"):
        cauda = cauda.replace(spec["hf"], "<token>")
    novo = estado_treino(pasta)
    if p.returncode != 0 or novo.get("passos_feitos", 0) <= feitos:
        return {"error": "o treino falhou", "detail": cauda}

    # copia o ponto mais novo para loras/, onde o app ja sabe procurar
    copiado = None
    try:
        origem = os.path.join(_saida(pasta), novo["ultimo"]) if novo.get("ultimo") else None
        if origem and os.path.isfile(origem):
            destino_dir = os.path.join(models_root() or "", "loras")
            os.makedirs(destino_dir, exist_ok=True)
            copiado = token + "-zi.safetensors"
            shutil.copyfile(origem, os.path.join(destino_dir, copiado))
    except Exception:
        copiado = None
    return {"ok": True, "pronto": novo.get("passos_feitos", 0) >= total,
            "passos_feitos": novo.get("passos_feitos", 0), "total": total,
            "arquivo": novo.get("ultimo"), "lora": copiado, "base": base_treino,
            "livre_gb": _livre_gb(),
            "segundos": round(time.time() - t0, 1), "log": cauda[-600:]}


def publica_lora(spec):
    """Sobe um LoRA treinado para um repositorio PRIVADO do Hugging Face.

    E assim que o arquivo sai do volume do laboratorio e fica ao alcance da
    producao, que nao tem volume: ela baixa por HTTP quando precisa.
    """
    token = (spec.get("token") or "").strip()
    repo = (spec.get("repo") or "").strip()
    nome = (spec.get("nome") or "").strip()
    if not token:
        return {"error": "falta o token de escrita do Hugging Face"}
    if not re.match(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$", repo or ""):
        return {"error": "repositorio invalido: use usuario/nome"}
    if not SAFE.match(nome or "x"):
        return {"error": "nome de arquivo invalido"}

    # Onde procurar. O nome do arquivo NAO diz a pasta do treino: um ponto
    # intermediario chama-se "tok-000800.safetensors" e o LoRA de SDXL chama-se
    # "tok-xl.safetensors", mas os dois moram em "tok" ou "tok_xl". Entao o app
    # manda a pasta em "pasta"; sem ela, tentamos as derivacoes conhecidas.
    candidatos = []
    root = models_root()
    if root:
        candidatos.append(os.path.join(root, "loras", nome))
    raiz = treino_raiz()
    if raiz:
        base_nome = re.sub(r"\.safetensors$", "", nome, flags=re.I)
        tok = re.sub(r"(-\d{4,6}|-xl)$", "", base_nome)
        pastas = []
        for cand in ((spec.get("pasta") or "").strip().lower(), tok + "_zi", tok + "_xl", tok, base_nome):
            cand = _pasta(cand)
            if cand and cand not in pastas:
                pastas.append(cand)
        for pst in pastas:
            for d in (os.path.join(raiz, pst, "saida", pst), os.path.join(raiz, pst, "saida")):
                candidatos.append(os.path.join(d, nome))
                if os.path.isdir(d):
                    for f in sorted(os.listdir(d)):
                        if f.endswith(".safetensors"):
                            candidatos.append(os.path.join(d, f))
    arq = next((c for c in candidatos if os.path.isfile(c)), None)
    if not arq:
        return {"error": "nao achei o arquivo %s para publicar" % nome}

    py = AITK_PY
    script = (
        "import os\n"
        "from huggingface_hub import HfApi\n"
        "api=HfApi(token=os.environ['HF_TOKEN'])\n"
        "api.create_repo(repo_id=os.environ['REPO'], private=True, exist_ok=True)\n"
        "api.upload_file(path_or_fileobj=os.environ['ARQ'],"
        " path_in_repo=os.environ['NOME'], repo_id=os.environ['REPO'])\n"
        "print('ok')\n"
    )
    env = dict(os.environ)
    env.update({"HF_TOKEN": token, "REPO": repo, "NOME": nome, "ARQ": arq})
    try:
        p = subprocess.run([py, "-c", script], capture_output=True, text=True,
                           timeout=900, env=env)
    except subprocess.TimeoutExpired:
        return {"error": "a publicacao passou do tempo"}
    if p.returncode != 0:
        saida = ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
        # nunca devolver o token em mensagem de erro
        saida = saida.replace(token, "<token>")
        return {"error": "o Hugging Face recusou", "detail": saida[-1200:]}
    return {"ok": True, "repo": repo, "nome": nome, "origem": arq,
            "url": "https://huggingface.co/%s/resolve/main/%s" % (repo, nome),
            "mb": round(os.path.getsize(arq) / 1048576, 1)}


# ---------------------------------------------------------------- handler
def _log(res):
    """Escreve o resultado no log do worker (sem imagens): da para ler no console."""
    try:
        if isinstance(res, dict) and "images" not in res:
            print("[worker] resultado:", json.dumps(res, ensure_ascii=False)[:3000])
    except Exception:
        pass
    return res


def handler(job):
    inp = job.get("input") or {}
    try:
        print("[worker] pedido:", json.dumps({k: (v if k not in ("workflow", "prompt", "images", "put") else "...") for k, v in inp.items()}, ensure_ascii=False)[:500])
    except Exception:
        pass

    # estas nao precisam do ComfyUI no ar
    if inp.get("ls") is not None:
        return _log(list_models(inp.get("ls")))

    if inp.get("download"):
        return _log(download_model(inp["download"]))

    if inp.get("delete"):
        return delete_model(inp["delete"])

    if inp.get("put"):
        return put_file(inp["put"])

    if inp.get("limpa_treino"):
        return limpa_treino(inp["limpa_treino"])

    if inp.get("treino_estado"):
        return _log(estado_treino(inp["treino_estado"]))

    if inp.get("poda"):
        d = inp["poda"]
        if isinstance(d, str):
            d = {"pasta": d}
        return poda_pontos(d.get("pasta"), d.get("manter") or 2)

    if inp.get("espaco_treino"):
        return _log(espaco_treino())

    if inp.get("preparar") is not None:
        d = inp["preparar"]
        if isinstance(d, dict):
            return _log(preparar(d.get("grupos"), d.get("token"), d.get("verificar", True)))
        return _log(preparar(d))

    if inp.get("treina"):
        return _log(treina_lora(inp["treina"]))

    if inp.get("publicar"):
        return _log(publica_lora(inp["publicar"]))

    start_comfy()

    if inp.get("get"):
        return api_get(inp["get"])

    wf = inp.get("workflow") or inp.get("prompt")
    if not wf:
        return {"error": "faltou o campo 'workflow'"}

    # LoRAs da personagem: baixados AQUI, no mesmo worker que vai gerar.
    # (baixar num job separado nao serve: o proximo job pode cair em outro worker)
    try:
        txt = json.dumps(wf)
        grupos = [g for g, chave in (("zimage", "z_image_bf16"), ("controlnet", "Z-Image-Fun-Controlnet"),
                                     ("detailer", "yolov8"), ("qwenedit", "Qwen-Image-Edit-2511"),
                                     ("sdxl", "realvisxl5")) if chave in txt and falta(g)]
        if grupos:
            r = preparar(grupos, inp.get("hf"))
            if r.get("erros"):
                return {"error": "faltam modelos e nao consegui baixar", "detail": r["erros"]}
    except Exception as e:
        print("[worker] preparar:", e)

    for lr in inp.get("loras") or []:
        try:
            r = download_model({"url": lr.get("url"), "dir": "loras",
                                "name": lr.get("name"), "token": lr.get("token")})
            if r.get("error"):
                return {"error": "nao consegui trazer o LoRA %s: %s"
                                 % (lr.get("name"), r["error"])}
        except Exception as e:
            return {"error": "falha ao trazer o LoRA %s: %s" % (lr.get("name"), e)}

    for img in inp.get("images") or []:
        try:
            upload_image(img["name"], img["image"])
        except Exception as e:
            return {"error": f"falha ao enviar a imagem {img.get('name')}: {e}"}

    t0 = time.time()
    client_id = str(uuid.uuid4())
    try:
        res = api_post("/prompt", {"prompt": wf, "client_id": client_id})
    except urllib.error.HTTPError as e:
        return _log({"error": "workflow rejeitado pelo ComfyUI",
                     "detail": e.read().decode()[:4000]})
    pid = res.get("prompt_id")
    if not pid:
        return {"error": "ComfyUI nao devolveu prompt_id", "detail": res}

    while time.time() - t0 < JOB_TIMEOUT:
        time.sleep(0.6)
        try:
            hist = api_get(f"/history/{pid}")
        except Exception:
            continue
        entry = hist.get(pid)
        if not entry:
            continue
        status = (entry.get("status") or {})
        if status.get("completed") or entry.get("outputs"):
            imgs = collect(entry)
            if not imgs and status.get("status_str") == "error":
                return {"error": "erro na execucao", "detail": status.get("messages")}
            for f in os.listdir(OUT_DIR):
                try:
                    os.remove(os.path.join(OUT_DIR, f))
                except Exception:
                    pass
            return {"images": imgs, "seconds": round(time.time() - t0, 1)}
        if status.get("status_str") == "error":
            return _log({"error": "erro na execucao", "detail": status.get("messages")})

    return _log({"error": "tempo limite excedido"})


if __name__ == "__main__":
    # NAO subir o ComfyUI aqui: a RunPod espera o worker se registrar em poucos
    # segundos e mata o processo se ele demorar. O ComfyUI sobe no primeiro job.
    runpod.serverless.start({"handler": handler})
