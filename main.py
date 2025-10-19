# main.py — YOLOv8 (detecção) para Render Free
# - Uvicorn sobe rápido; modelo baixa/carrega em background (evita 502)
# - Sempre retorna imagem anotada
# - Salva PNG em /static/annotated e devolve image_url
# - Suporte a HF_TOKEN para repositório privado (Hugging Face)

import os, io, time, threading, base64, requests, uuid
from typing import List
from urllib.parse import urlparse

# Reduz overhead de threads BLAS no CPU free
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image
import numpy as np

app = FastAPI(title="API Pimentas YOLOv8")

# ===================== CONFIG =====================

# <<< COLE AQUI O LINK DO SEU MODELO (.pt ou .onnx) >>>
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://huggingface.co/bulipucca/pimentas-model/resolve/main/best.onnx"  # <<< COLE AQUI O LINK DO SEU MODELO >>>
)
MODEL_PATH = (
    os.path.basename(urlparse(MODEL_URL).path)
    if MODEL_URL and not MODEL_URL.startswith("COLE_AQUI")
    else "best.pt"
)

# Presets
PRESET = os.getenv("PRESET", "ULTRA")
PRESETS = {
    "ULTRA":       dict(imgsz=320, conf=0.35, iou=0.50, max_det=4),   # mais rápido
    "RAPIDO":      dict(imgsz=384, conf=0.30, iou=0.50, max_det=4),
    "EQUILIBRADO": dict(imgsz=448, conf=0.30, iou=0.50, max_det=6),
    "PRECISO":     dict(imgsz=512, conf=0.40, iou=0.55, max_det=8),
    "MAX_RECALL":  dict(imgsz=640, conf=0.12, iou=0.45, max_det=10),
}
CFG = PRESETS.get(PRESET, PRESETS["ULTRA"])

# Sempre devolver imagem anotada
RETURN_IMAGE = True

# Suporte a repo privado (opcional)
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
REQ_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# Pasta estática para salvar PNG anotado
STATIC_DIR = os.path.join(os.getcwd(), "static")
ANNOT_DIR = os.path.join(STATIC_DIR, "annotated")
os.makedirs(ANNOT_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ===================== ESTADO GLOBAL =====================

model = None
labels = {}
READY = False
LOAD_ERR = None

# ===================== FUNÇÕES AUX =====================

def ensure_model_file():
    """Baixa o arquivo do modelo se não existir localmente (stream)."""
    if os.path.exists(MODEL_PATH):
        return
    if not MODEL_URL or MODEL_URL.startswith("COLE_AQUI"):
        raise RuntimeError("Defina MODEL_URL com link direto do modelo (.pt ou .onnx).")
    print(f"[init] Baixando modelo: {MODEL_URL}")
    with requests.get(MODEL_URL, headers=REQ_HEADERS, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print("[init] Download concluído:", MODEL_PATH)


def background_load():
    """Baixa e carrega YOLO; faz uma inferência curta; marca READY ao final."""
    global model, labels, READY, LOAD_ERR
    try:
        t0 = time.time()
        ensure_model_file()
        m = YOLO(MODEL_PATH)
        try:
            m.fuse()
        except Exception:
            pass

        # WARM-UP: 1 inferência rapidinha em imagem 64x64
        try:
            img = Image.new("RGB", (64, 64), (255, 255, 255))
            _ = m.predict(img, imgsz=CFG["imgsz"], conf=CFG["conf"], iou=CFG["iou"],
                          max_det=1, device="cpu", verbose=False)
        except Exception:
            pass

        model = m
        labels = m.names
        READY = True
        print(f"[init] Modelo pronto em {time.time() - t0:.1f}s")
    except Exception as e:
        LOAD_ERR = str(e)
        READY = False
        print("[init] ERRO ao carregar modelo:", LOAD_ERR)


# ===================== LIFECYCLE =====================

@app.on_event("startup")
def on_startup():
    threading.Thread(target=background_load, daemon=True).start()

# ===================== ROTAS =====================

@app.get("/")
def health():
    return {
        "status": "ok" if READY else "warming",
        "ready": READY,
        "error": LOAD_ERR,
        "preset": PRESET,
        "cfg": CFG,
        "model": MODEL_PATH if MODEL_PATH else None,
        "classes": list(labels.values()) if READY else None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not READY:
        # serviço no ar, mas modelo ainda aquecendo
        return JSONResponse({"ok": False, "warming_up": True, "error": LOAD_ERR}, status_code=503)

    t0 = time.time()
    try:
        im_bytes = await file.read()
        image = Image.open(io.BytesIO(im_bytes)).convert("RGB")

        # ADICIONADO: reduz o maior lado para 1024 px (acelera em CPU, mantém proporção)
        image.thumbnail((1024, 1024))  # in-place; não retorna nada

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

            image_b64 = None
            image_url = None

            if RETURN_IMAGE:
                # annotated = np.ndarray (BGR)
                annotated = r.plot()

                # Salva PNG e devolve URL
                fname = f"{uuid.uuid4().hex}.png"
                fpath = os.path.join(ANNOT_DIR, fname)
                Image.fromarray(annotated[:, :, ::-1]).save(fpath)  # converte BGR->RGB
                image_url = f"/static/annotated/{fname}"

                # (opcional) também em base64
                image_b64 = to_b64_png(annotated)

            top = max(preds, key=lambda p: p["conf"]) if preds else None
            return JSONResponse({
                "ok": True,
                "inference_time_s": round(time.time() - t0, 3),
                "num_dets": len(preds),
                "top_pred": top,
                "preds": preds,
                "image_b64": image_b64,
                "image_url": image_url
            })

        # Sem detecções
        return JSONResponse({
            "ok": True,
            "inference_time_s": round(time.time() - t0, 3),
            "num_dets": 0,
            "top_pred": None,
            "preds": [],
            "image_b64": None,
            "image_url": None
        })

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e), "inference_time_s": round(time.time() - t0, 3)}, status_code=200)


@app.get("/warmup")
def warmup():
    """Bloqueia até READY ou 90s, e roda 1 inferência curtinha para compilar o caminho."""
    t0 = time.time()
    while not READY and time.time() - t0 < 90:
        time.sleep(0.5)

    if not READY:
        return {"ok": False, "warming_up": True}

    # imagem 64x64 branca só para aquecer
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    _ = model.predict(
        img, imgsz=CFG["imgsz"], conf=CFG["conf"], iou=CFG["iou"],
        max_det=1, device="cpu", verbose=False
    )
    return {"ok": True}


@app.get("/ui")
def ui():
    # UI aguarda "ready" e usa image_url para exibir a anotada
    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <title>Detecção de Pimentas</title>
      <style>
        body {{ font-family: Arial, sans-serif; padding:16px; max-width:820px; margin:auto; }}
        .row {{ display:flex; gap:16px; flex-wrap:wrap; }}
        .card {{ flex:1 1 380px; border:1px solid #ddd; border-radius:12px; padding:16px; }}
        img {{ max-width:100%; border-radius:12px; }}
        button {{ padding:10px 16px; border-radius:10px; border:1px solid #ccc; background:#f8f8f8; cursor:pointer; }}
        pre {{ background:#f4f4f4; padding:12px; border-radius:8px; overflow:auto; }}
        #status {{ margin:6px 0; }}
      </style>
    </head>
    <body>
      <h2>Detecção de Pimentas (YOLOv8)</h2>
      <div id="status">Carregando modelo...</div>

      <div class="card">
        <input id="file" type="file" accept="image/*" capture="environment" disabled/>
        <button id="btn" disabled onclick="doPredict()">Identificar</button>
        <p id="resumo"></p>
      </div>

      <div class="row">
        <div class="card">
          <h3>Prévia</h3>
          <img id="preview"/>
        </div>
        <div class="card">
          <h3>Anotada</h3>
          <img id="annotated"/>
        </div>
      </div>

      <div class="card">
        <h3>JSON</h3>
        <pre id="json"></pre>
      </div>

      <script>
        const MAX_DIM = 640, JPEG_QUALITY = 0.8;
        const BASE = window.location.origin;

        async function waitReady() {{
          while (true) {{
            try {{
              const r = await fetch('/', {{cache: 'no-store'}});
              const d = await r.json();
              if (d.ready) {{
                document.getElementById('status').innerText = 'Modelo pronto ✅';
                document.getElementById('file').disabled = false;
                document.getElementById('btn').disabled = false;
                break;
              }} else {{
                document.getElementById('status').innerText = 'Aquecendo modelo...';
              }}
            }} catch (e) {{
              document.getElementById('status').innerText = 'Conectando...';
            }}
            await new Promise(res => setTimeout(res, 1000));
          }}
        }}

        function readFile(file) {{
          return new Promise((resolve) => {{
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.readAsDataURL(file);
          }});
        }}

        async function compressImage(file) {{
          const dataUrl = await readFile(file);
          const img = new Image(); img.src = dataUrl; await img.decode();
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

        function showPreview(blob) {{
          document.getElementById('preview').src = URL.createObjectURL(blob);
        }}

        async function doPredict() {{
          const f = document.getElementById('file').files[0];
          if (!f) {{ alert('Escolha uma imagem.'); return; }}
          document.getElementById('resumo').innerText = 'Compactando...';
          const blob = await compressImage(f);
          showPreview(blob);

          const fd = new FormData();
          fd.append('file', blob, 'photo.jpg');

          document.getElementById('resumo').innerText = 'Analisando...';
          try {{
            const r = await fetch('/predict', {{ method: 'POST', body: fd }});
            const data = await r.json();
            document.getElementById('json').innerText = JSON.stringify(data, null, 2);

            if (data.ok === false && data.warming_up) {{
              document.getElementById('resumo').innerText = 'Aquecendo modelo... tente novamente em instantes.';
              return;
            }}
            if (data.ok === false) {{
              document.getElementById('resumo').innerText = 'Erro: ' + (data.error || 'desconhecido');
              return;
            }}

            // Imagem anotada por URL (preferida)
            if (data.image_url) {{
              document.getElementById('annotated').src = BASE + data.image_url;
            }} else if (data.image_b64) {{
              document.getElementById('annotated').src = data.image_b64;
            }}

            if (data.top_pred) {{
              const pct = Math.round(data.top_pred.conf * 100);
              document.getElementById('resumo').innerText = 'Detectou: ' + data.top_pred.classe +
                ' | ' + pct + '% | Caixas: ' + data.num_dets + ' | ' +
                data.inference_time_s + ' s';
            }} else {{
              document.getElementById('resumo').innerText = 'Nenhuma pimenta detectada. (' + data.inference_time_s + ' s)';
            }}
          }} catch (e) {{
            document.getElementById('resumo').innerText = 'Falha ao chamar a API.';
          }}
        }}

        waitReady();
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

