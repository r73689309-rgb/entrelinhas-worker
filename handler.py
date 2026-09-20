"""
Handler serverless para ComfyUI (RunPod).

Entrada aceita:
  {"workflow": {...}}                      -> executa o workflow em formato API
  {"workflow": {...}, "images": [ {"name":"ref.png","image":"<base64>"} ]}
  {"get": "/object_info"}                  -> proxy GET para a API do ComfyUI
  {"ls": "loras"}                          -> lista os arquivos de modelo disponiveis
  {"download": {"url": "...", "dir": "loras", "name": "x.safetensors"}}
                                           -> baixa um modelo para a pasta de modelos
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

    start_comfy()

    if inp.get("get"):
        return api_get(inp["get"])

    wf = inp.get("workflow") or inp.get("prompt")
    if not wf:
        return {"error": "faltou o campo 'workflow'"}

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
