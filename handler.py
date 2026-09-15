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
def link_models():
    """Se houver network volume, liga os diretorios de modelos dele no ComfyUI."""
    roots = [
        "/runpod-volume/models_store",
        "/runpod-volume/ComfyUI/models",
        "/runpod-volume/models",
    ]
    root = next((r for r in roots if os.path.isdir(r)), None)
    if not root:
        print("[worker] sem network volume: usando os modelos da propria imagem")
        return
    print("[worker] usando modelos de", root)
    dest_root = os.path.join(COMFY, "models")
    os.makedirs(dest_root, exist_ok=True)
    for name in sorted(os.listdir(root)):
        src = os.path.join(root, name)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(dest_root, name)
        try:
            if os.path.islink(dst):
                continue
            if os.path.isdir(dst) and not os.listdir(dst):
                os.rmdir(dst)
            if not os.path.exists(dst):
                os.symlink(src, dst)
                print("[worker]   ->", name)
        except Exception as e:
            print("[worker]   falhou", name, e)


SAFE = re.compile(r"^[A-Za-z0-9._-]+$")


def models_root():
    for r in ("/runpod-volume/models_store",
              "/runpod-volume/ComfyUI/models",
              "/runpod-volume/models",
              os.path.join(COMFY, "models")):
        if os.path.isdir(r):
            return r
    return None


def list_models(which=None):
    root = models_root()
    if not root:
        return {"error": "nenhum diretorio de modelos encontrado"}
    out = {}
    names = [which] if isinstance(which, str) and which else sorted(os.listdir(root))
    for name in names:
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        files = []
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if os.path.isfile(p):
                files.append({"name": f, "mb": round(os.path.getsize(p) / 1048576, 1)})
        out[name] = files
    free = shutil.disk_usage(root)
    return {"root": root, "dirs": out,
            "free_gb": round(free.free / 1073741824, 1),
            "total_gb": round(free.total / 1073741824, 1)}


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


# ---------------------------------------------------------------- handler
def handler(job):
    inp = job.get("input") or {}

    # estas nao precisam do ComfyUI no ar
    if inp.get("ls") is not None:
        return list_models(inp.get("ls"))

    if inp.get("download"):
        return download_model(inp["download"])

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
