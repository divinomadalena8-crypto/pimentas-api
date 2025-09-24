# main.py — YOLOv8 DETECÇÃO otimizado para Render Free (CPU)
# Foco: baixar latência
import os
# Limita threads (evita overhead no CPU free)
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from ultralytics import YOLO
from PIL import Image
import io, time, requests, base64, json
from typing import List
from urllib.parse import urlparse
import numpy as np

app = FastAPI(title="API Pimentas YOLOv8 — Rápido")

# ====== CONFIG ======
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# COLE AQUI O LINK DIRETO DO SEU MODELO (.pt ou .onnx) ENTRE ASPAS
# Exemplo: "https://huggingface.co/SEU_USUARIO/pimentas-model/resolve/main/best.pt"
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://huggingface.co/bulipucca/pimentas-model/resolve/f704849d54bc90d866d4acceb663f6d11ed03a21/best.pt"  # <<< COLE AQUI O LINK DO SEU MODELO >>>
)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Nome local: pegamos do próprio URL (mantendo a extensão .pt ou .onnx)
MODEL_PATH = os.path.basename(urlparse(MODEL_URL).path) if MODEL_URL and not MODEL_URL.startswith("COLE_AQUI") else "best.pt"

# Presets: "RAPIDO" (padrão), "EQUILIBRADO", "PRECISO", "MAX_RECALL"
PRESET = os.getenv("PRESET", "RAPIDO")
PRESETS = {
    "RAPIDO":      dict(imgsz=384, conf=0.30, iou=0.50, max_det=4),
    "EQUILIBRADO": dict(imgsz=448, conf=0.30, iou=0.50, max_det=6),
    "PRECISO":     dict(imgsz=512, conf=0.40, iou=0.55, max_det=8),
    "MAX_RECALL":  dict(imgsz=640, conf=0.12, iou=0.45, max_det=10),
}
CFG = PRESETS.get(PRESET, PRESETS["RAPIDO"])

# Gera imagem anotada? (custa CPU/tempo). 0 = não (recomendado p/ app)
RETURN_IMAGE = os.getenv("RETURN_IMAGE", "0") == "1"

def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    if not MODEL_URL or MODEL_URL.startswith("COLE_AQUI"):
        raise RuntimeError("Defina MODEL_URL com link direto do modelo (.pt ou .onnx).")
    print(f"[init] Baixando modelo: {MODEL_URL}")
    with requests.get(MODEL_URL, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print("[init] Download concluído:", MODEL_PATH)

ensure_model()

# Carrega YOLO (aceita .pt e .onnx)
print(f"[init] Carregando modelo: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
try:
    model.fuse()  # pequeno ganho
except Exception:
    pass
labels = model.names

# ====== UTILS ======
def to_b64_png(np_bgr: np.ndarray) -> str | None:
    try:
        rgb = np_bgr[:, :, ::-1]
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None

@app.get("/")
def root():
    return {"status": "ok", "preset": PRESET, "cfg": CFG, "model": MODEL_PATH, "classes": list(labels.values())}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    t0 = time.time()
    try:
        im_bytes = await file.read()
        image = Image.open(io.BytesIO(im_bytes)).convert("RGB")

        res = model.predict(
            image,
            imgsz=CFG["imgsz"],
            conf=CFG["conf"],
            iou=CFG["iou"],
            max_det=CFG["max_det"],
            device="cpu",
            verbose=False,
        )
        r = res[0]
        preds: List[dict] = []

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy().tolist()
            cls  = r.boxes.cls.cpu().numpy().astype(int).tolist()
            conf = r.boxes.conf.cpu().numpy().tolist()
            for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, conf):
                preds.append({
                    "classe": labels.get(int(c), str(int(c))),
                    "conf": round(float(cf), 4),
                    "bbox_xyxy": [round(x1,2), round(y1,2), round(x2,2), round(y2,2)]
                })

            img_b64 = None
            if RETURN_IMAGE:
                annotated = r.plot()  # numpy BGR
                img_b64 = to_b64_png(annotated)

            top = max(preds, key=lambda p: p["conf"]) if preds else None
            return JSONResponse({
                "ok": True,
                "inference_time_s": round(time.time() - t0, 3),
                "num_dets": len(preds),
                "top_pred": top,
                "preds": preds,
                "image_b64": img_b64
            })

        return JSONResponse({
            "ok": True,
            "inference_time_s": round(time.time() - t0, 3),
            "num_dets": 0,
            "top_pred": None,
            "preds": [],
            "image_b64": None
        })
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e), "inference_time_s": round(time.time() - t0, 3)}, status_code=200)

@app.get("/ui")
def ui():
    # A UI comprime/redimensiona a imagem no navegador ANTES do upload
    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <title>Detecção de Pimentas (Rápido)</title>
      <style>
        body {{ font-family: Arial, sans-serif; padding:16px; max-width:800px; margin:auto; }}
        .row {{ display:flex; gap:16px; flex-wrap:wrap; }}
        .card {{ flex:1 1 360px; border:1px solid #ddd; border-radius:12px; padding:16px; }}
        img {{ max-width:100%; border-radius:12px; }}
        button {{ padding:10px 16px; border-radius:10px; border:1px solid #ccc; background:#f8f8f8; cursor:pointer; }}
        pre {{ background:#f4f4f4; padding:12px; border-radius:8px; overflow:auto; }}
        small {{ color:#666; }}
      </style>
    </head>
    <body>
      <h2>Detecção de Pimentas (YOLOv8n — Rápido)</h2>
      <small>Preset: <b>{PRESET}</b> | imgsz={{CFG['imgsz']}} | conf={{CFG['conf']}} | iou={{CFG['iou']}} | retorno_imagem={'sim' if RETURN_IMAGE else 'não'}</small>

      <div class="card">
        <input id="file" type="file" accept="image/*" capture="environment"/>
        <button onclick="doPredict()">Identificar</button>
        <p id="resumo"></p>
      </div>

      <div class="row">
        <div class="card">
          <h3>Prévia (compactada)</h3>
          <img id="preview"/>
        </div>
        <div class="card">
          <h3>Anotada (se habilitada)</h3>
          <img id="annotated"/>
        </div>
      </div>

      <div class="card">
        <h3>JSON</h3>
        <pre id="json"></pre>
      </div>

      <script>
        const MAX_DIM = 800;       // limite do lado maior antes do upload
        const JPEG_QUALITY = 0.82; // compactação

        function readFile(file) {{
          return new Promise((resolve) => {{
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.readAsDataURL(file);
          }});
        }}

        async function compressImage(file) {{
          const dataUrl = await readFile(file);
          const img = new Image();
          img.src = dataUrl;
          await img.decode();

          let w = img.width, h = img.height;
          const scale = Math.min(1, MAX_DIM / Math.max(w, h));
          w = Math.round(w * scale); h = Math.round(h * scale);

          const canvas = document.createElement('canvas');
          canvas.width = w; canvas.height = h;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0, w, h);

          return new Promise((resolve) => {{
            canvas.toBlob((blob) => resolve(blob), 'image/jpeg', JPEG_QUALITY);
          }});
        }}

        function showPreviewFromBlob(blob) {{
          const url = URL.createObjectURL(blob);
          document.getElementById('preview').src = url;
        }}

        async function doPredict() {{
          const f = document.getElementById('file').files[0];
          if (!f) {{ alert('Escolha uma imagem.'); return; }}
          document.getElementById('resumo').innerText = 'Compactando imagem...';
          const blob = await compressImage(f);
          showPreviewFromBlob(blob);

          const fd = new FormData();
          fd.append('file', blob, 'photo.jpg');

          document.getElementById('resumo').innerText = 'Analisando...';
          try {{
            const r = await fetch('/predict', {{ method: 'POST', body: fd }});
            const data = await r.json();
            document.getElementById('json').innerText = JSON.stringify(data, null, 2);
            if (data.image_b64) document.getElementById('annotated').src = data.image_b64;

            if (data.ok === false) {{
              document.getElementById('resumo').innerText = 'Erro: ' + (data.error || 'desconhecido');
              return;
            }}

            if (data.top_pred) {{
              const pct = Math.round(data.top_pred.conf * 100);
              document.getElementById('resumo').innerText = 'Detectou: ' + data.top_pred.classe + ' | ' + pct + '% | caixas: ' + data.num_dets;
            }} else {{
              document.getElementById('resumo').innerText = 'Nenhuma pimenta detectada.';
            }}
          }} catch (e) {{
            document.getElementById('resumo').innerText = 'Erro ao chamar a API.';
          }}
        }}
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
