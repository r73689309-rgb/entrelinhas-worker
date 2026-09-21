"""
Handler serverless para ComfyUI (RunPod).

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
  {"treino_estado": "tok"}                 -> quantas fotos e quantos passos ja treinados
  {"treina": {"token":"tok","passos":400,"total":1600,"base":"flux|sdxl"}}
                                           -> treina UM pedaco e devolve o LoRA parcial
  {"publicar": {"token":"<hf>","repo":"user/loras","nome":"tok.safetensors"}}
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
              "facerestore_models", "style_models", "gligen", "ultralytics")


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
    if not SAFE.match(sub or "x") or not SAFE.match(name):
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
    tmp = dest + ".part"
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        try:
            os.remove(tmp)
        except Exception:
            pass
        return {"error": f"falha ao baixar: {e}"}
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
SD_SCRIPTS = os.environ.get("SD_SCRIPTS", "/sd-scripts")
SAFE_REL = re.compile(r"^[A-Za-z0-9._\-/]+$")
TOKEN_RE = re.compile(r"^[a-z0-9]{2,32}$")
# A pasta no volume pode diferir da palavra-chave: a mesma personagem pode ter
# um treino de FLUX e um de SDXL, e misturar os dois quebra a continuacao.
PASTA_RE = re.compile(r"^[a-z0-9]{2,32}(_xl)?$")


def _pasta(spec_ou_token, base=None):
    if isinstance(spec_ou_token, dict):
        p = (spec_ou_token.get("pasta") or spec_ou_token.get("token") or "").strip().lower()
    else:
        p = (spec_ou_token or "").strip().lower()
    if base == "sdxl" and not p.endswith("_xl"):
        p += "_xl"
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
    saida = os.path.join(base, "saida")
    fotos = 0
    if os.path.isdir(imgs):
        fotos = len([f for f in os.listdir(imgs) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    partes = []
    if os.path.isdir(saida):
        partes = sorted(f for f in os.listdir(saida) if f.endswith(".safetensors"))
    feitos = 0
    for p in partes:
        m = re.search(r"-(\d+)\.safetensors$", p)
        if m:
            feitos = max(feitos, int(m.group(1)))
    return {"ok": True, "token": token, "fotos": fotos, "passos_feitos": feitos,
            "partes": partes, "ultimo": (partes[-1] if partes else None)}


def treina_lora(spec):
    """Roda UM pedaco do treino. O app chama de novo ate chegar no total."""
    raiz = treino_raiz()
    if not raiz:
        return {"error": "este endpoint nao tem volume de rede: use o laboratorio"}
    token = (spec.get("token") or "").strip().lower()
    if not TOKEN_RE.match(token):
        return {"error": "palavra-chave invalida (use so letras e numeros)"}
    if not os.path.isdir(SD_SCRIPTS):
        return {"error": "o treinador nao esta nesta imagem do worker"}

    base_treino = (spec.get("base") or "flux").strip().lower()
    if base_treino not in ("flux", "sdxl"):
        base_treino = "flux"
    pasta = _pasta(spec, base_treino)
    if not pasta:
        return {"error": "pasta de treino invalida"}
    base = os.path.join(raiz, pasta)
    dados = os.path.join(base, "img")
    saida = os.path.join(base, "saida")
    logs = os.path.join(base, "log")
    for d in (saida, logs):
        os.makedirs(d, exist_ok=True)
    if not os.path.isdir(dados):
        return {"error": "nao achei as fotos: mande o conjunto antes de treinar"}

    est = estado_treino(pasta)
    feitos = est.get("passos_feitos", 0)
    passos = max(50, min(1200, int(spec.get("passos") or 400)))
    total = max(passos, min(6000, int(spec.get("total") or 1600)))
    if feitos >= total:
        return {"ok": True, "pronto": True, "passos_feitos": feitos, "arquivo": est.get("ultimo")}
    passos = min(passos, total - feitos)

    # Duas bases possiveis. FLUX: 4 arquivos soltos (unet, clip_l, t5, ae) e
    # networks.lora_flux. SDXL: um unico checkpoint e networks.lora — mais
    # barato de treinar e com prompt negativo de verdade na hora de gerar.
    if base_treino == "sdxl":
        ckpt = _acha_ckpt((spec.get("ckpt") or "").strip())
        if not ckpt:
            return {"error": "nao achei um checkpoint SDXL (RealVisXL) neste worker: "
                             "instale o modelo antes de treinar"}
    else:
        unet = _acha("unet", [r"flux.*dev.*fp8", r"flux.*dev", r"flux"]) or _acha("diffusion_models", [r"flux"])
        clip_l = _acha("text_encoders", [r"^clip_l"]) or _acha("clip", [r"^clip_l"])
        t5 = _acha("text_encoders", [r"t5xxl.*fp8", r"t5xxl"]) or _acha("clip", [r"t5xxl"])
        ae = _acha("vae", [r"^ae\.", r"flux.*vae", r"ae"])
        faltam = [n for n, v in (("unet FLUX", unet), ("clip_l", clip_l), ("t5xxl", t5), ("vae ae", ae)) if not v]
        if faltam:
            return {"error": "faltam arquivos para treinar: " + ", ".join(faltam)}

    anterior = est.get("ultimo")
    nome = "%s-%06d" % (token, feitos + passos)
    # o treinador roda no ambiente proprio dele quando existir (venv /sd-venv)
    acc = "/sd-venv/bin/accelerate" if os.path.isfile("/sd-venv/bin/accelerate") else "accelerate"
    script = "sdxl_train_network.py" if base_treino == "sdxl" else "flux_train_network.py"
    cmd = [
        acc, "launch", "--num_cpu_threads_per_process", "2",
        "--num_processes", "1", "--num_machines", "1", "--mixed_precision", "bf16",
        "--dynamo_backend", "no",
        os.path.join(SD_SCRIPTS, script),
    ]
    if base_treino == "sdxl":
        cmd += ["--pretrained_model_name_or_path", ckpt]
    else:
        cmd += ["--pretrained_model_name_or_path", unet,
                "--clip_l", clip_l, "--t5xxl", t5, "--ae", ae]
    cmd += [
        "--train_data_dir", dados,
        "--output_dir", saida, "--output_name", nome, "--logging_dir", logs,
        "--save_model_as", "safetensors", "--save_precision", "bf16",
        "--mixed_precision", "bf16", "--sdpa", "--gradient_checkpointing",
        "--network_module", ("networks.lora" if base_treino == "sdxl" else "networks.lora_flux"),
        "--network_dim", str(max(4, min(64, int(spec.get("dim") or 16)))),
        "--optimizer_type", "adafactor",
        "--optimizer_args", "relative_step=False", "scale_parameter=False", "warmup_init=False",
        "--lr_scheduler", "constant_with_warmup", "--lr_warmup_steps", "10",
        "--learning_rate", str(spec.get("lr") or "1e-4"),
        "--max_train_steps", str(passos),
        "--train_batch_size", "1",
        "--resolution", str(spec.get("res") or "1024,1024"),
        "--enable_bucket", "--min_bucket_reso", "512", "--max_bucket_reso", "1536",
        "--cache_latents", "--cache_latents_to_disk",
        "--seed", "42",
        "--max_grad_norm", ("1.0" if base_treino == "sdxl" else "0.0"),
        "--save_every_n_steps", "100000",
    ]
    if base_treino == "sdxl":
        # No SDXL o codificador de texto treina junto: e o que faz a palavra-chave
        # grudar na personagem. Isso e incompativel com o cache do text encoder,
        # entao aqui ele NAO entra (no FLUX entra, e o T5 fica congelado).
        cmd += ["--no_half_vae", "--min_snr_gamma", "5",
                "--text_encoder_lr", str(spec.get("telr") or "5e-5"),
                "--unet_lr", str(spec.get("lr") or "1e-4"),
                "--noise_offset", "0.03",
                "--clip_skip", "1"]
    else:
        cmd += ["--cache_text_encoder_outputs", "--cache_text_encoder_outputs_to_disk",
                "--fp8_base", "--highvram",
                "--timestep_sampling", "shift", "--discrete_flow_shift", "3.1582",
                "--model_prediction_type", "raw", "--guidance_scale", "1.0"]
    if anterior:
        cmd += ["--network_weights", os.path.join(saida, anterior)]

    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=SD_SCRIPTS, capture_output=True, text=True,
                           timeout=max(120, int(spec.get("limite") or 1500)))
    except subprocess.TimeoutExpired:
        return {"error": "o pedaco passou do tempo do job: diminua os passos por pedaco"}
    cauda = ((p.stdout or "")[-1500:] + "\n" + (p.stderr or "")[-2500:]).strip()
    arq = os.path.join(saida, nome + ".safetensors")
    if p.returncode != 0 or not os.path.isfile(arq):
        return {"error": "o treino falhou", "detail": cauda}

    # o arquivo pronto vai para a pasta de LoRAs, onde o app ja sabe procurar
    copiado = None
    try:
        destino_dir = os.path.join(models_root() or "", "loras")
        os.makedirs(destino_dir, exist_ok=True)
        copiado = (token + "-xl.safetensors") if base_treino == "sdxl" else (token + ".safetensors")
        shutil.copyfile(arq, os.path.join(destino_dir, copiado))
    except Exception:
        copiado = None

    novo = estado_treino(pasta)
    return {"ok": True, "pronto": novo.get("passos_feitos", 0) >= total,
            "passos_feitos": novo.get("passos_feitos", 0), "total": total,
            "arquivo": nome + ".safetensors", "lora": copiado, "base": base_treino,
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

    # procura o arquivo: primeiro na pasta de loras, depois na saida do treino
    candidatos = []
    root = models_root()
    if root:
        candidatos.append(os.path.join(root, "loras", nome))
    raiz = treino_raiz()
    if raiz:
        tok = nome.split(".")[0]
        candidatos.append(os.path.join(raiz, tok, "saida", nome))
        d = os.path.join(raiz, tok, "saida")
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".safetensors"):
                    candidatos.append(os.path.join(d, f))
    arq = next((c for c in candidatos if os.path.isfile(c)), None)
    if not arq:
        return {"error": "nao achei o arquivo %s para publicar" % nome}

    py = "/sd-venv/bin/python" if os.path.isfile("/sd-venv/bin/python") else sys.executable
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
    return {"ok": True, "repo": repo, "nome": nome,
            "url": "https://huggingface.co/%s/resolve/main/%s" % (repo, nome),
            "mb": round(os.path.getsize(arq) / 1048576, 1)}


# ---------------------------------------------------------------- handler
def handler(job):
    inp = job.get("input") or {}

    # estas nao precisam do ComfyUI no ar
    if inp.get("ls") is not None:
        return list_models(inp.get("ls"))

    if inp.get("download"):
        return download_model(inp["download"])

    if inp.get("delete"):
        return delete_model(inp["delete"])

    if inp.get("put"):
        return put_file(inp["put"])

    if inp.get("limpa_treino"):
        return limpa_treino(inp["limpa_treino"])

    if inp.get("treino_estado"):
        return estado_treino(inp["treino_estado"])

    if inp.get("treina"):
        return treina_lora(inp["treina"])

    if inp.get("publicar"):
        return publica_lora(inp["publicar"])

    start_comfy()

    if inp.get("get"):
        return api_get(inp["get"])

    wf = inp.get("workflow") or inp.get("prompt")
    if not wf:
        return {"error": "faltou o campo 'workflow'"}

    # LoRAs da personagem: baixados AQUI, no mesmo worker que vai gerar.
    # (baixar num job separado nao serve: o proximo job pode cair em outro worker)
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
        return {"error": "workflow rejeitado pelo ComfyUI",
                "detail": e.read().decode()[:4000]}
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
            return {"error": "erro na execucao", "detail": status.get("messages")}

    return {"error": "tempo limite excedido"}


if __name__ == "__main__":
    # NAO subir o ComfyUI aqui: a RunPod espera o worker se registrar em poucos
    # segundos e mata o processo se ele demorar. O ComfyUI sobe no primeiro job.
    runpod.serverless.start({"handler": handler})
