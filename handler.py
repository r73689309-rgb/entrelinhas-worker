"""
Handler serverless para ComfyUI (RunPod).

Entrada aceita:
  {"workflow": {...}}                      -> executa o workflow em formato API
  {"workflow": {...}, "images": [ {"name":"ref.png","image":"<base64>"} ]}
  {"get": "/object_info"}                  -> proxy GET para a API do ComfyUI
Saida:
  {"images":[{"filename":..., "mime":..., "data":"<base64>"}], "seconds": 12.3}
"""
import base64
import json
import os
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
BOOT_TIMEOUT = int(os.environ.get("BOOT_TIMEOUT", "300"))
JOB_TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "900"))

_proc = None


# ---------------------------------------------------------------- modelos
def link_models():
    """Liga os diretorios de modelos do network volume dentro do ComfyUI."""
    roots = [
        "/runpod-volume/models_store",
        "/runpod-volume/ComfyUI/models",
        "/runpod-volume/models",
    ]
    root = next((r for r in roots if os.path.isdir(r)), None)
    if not root:
        print("[worker] AVISO: nenhum diretorio de modelos encontrado em /runpod-volume")
        try:
            print("[worker] conteudo de /runpod-volume:", os.listdir("/runpod-volume"))
        except Exception as e:
            print("[worker] /runpod-volume inacessivel:", e)
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
    try:
        start_comfy()
    except Exception as e:
        print("[worker] falha ao subir o ComfyUI no boot:", e)
    runpod.serverless.start({"handler": handler})
