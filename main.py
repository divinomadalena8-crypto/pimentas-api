# main.py — YOLOv8 DETECÇÃO otimizado p/ Render Free (CPU)
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from ultralytics import YOLO
from PIL import Image
import io, json, time, os, requests, base64

app = FastAPI(title="API Pimentas YOLOv8 — Detecção")

# ========= CONFIG =========
MODEL_PATH = "best.pt"
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://huggingface.co/bulipucca/pimentas-model/resolve/main/best.pt"
)

# Escolha UM preset (troque a string abaixo): "RAPIDO", "EQUILIBRADO", "PRECISO"
PRESET = os.getenv("PRESET", "EQUILIBRADO")

PRESETS = {
    # mais rápido (menos acurácia)
    "RAPIDO":     dict(imgsz=416, conf=0.35, iou=0.50, max_det=5),
    # bom compromisso
    "EQUILIBRADO":dict(imgsz=448, conf=0.35, iou=0.50, max_det=5),
    # mais acurácia (um pouco mais lento)
    "PRECISO":    dict(imgsz=512, conf=0.40, iou=0.55, max_det=8),
}
CFG = PRESETS.get(PRESET, PRESETS["EQUILIBRADO"])

def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    if not MODEL_URL or MODEL_URL.startswith("COLE_AQUI"):
        raise RuntimeError("Defina MODEL_URL (ou cole o link do best.pt no código).")
    print(f"[init] Baixando modelo de: {MODEL_URL}")
    with requests.get(MODEL_URL, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print("[init] Download do modelo concluído.")

ensure_model()
model = YOLO(MODEL_PATH)
# pequenos ganhos de latência
try:
    model.fuse()
except Exception:
    pass

labels = model.names  # {id: nome}

# ========= ROTAS =========
@app.get("/")
def root():
    return {
        "status": "ok",
        "task": getattr(model, "task", None),
        "preset": PRESET,
        "cfg": CFG,
        "classes": list(labels.values())
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Detecção com imagem anotada (base64)"""
    im_bytes = await file.read()
    image = Image.open(io.BytesIO(im_bytes)).convert("RGB")

    t0 = time.time()
    res = model.predict(
        image,
        imgsz=CFG["imgsz"],
        conf=CFG["conf"],
        iou=CFG["iou"],
        max_det=CFG["max_det"],
        device="cpu",
        verbose=False
    )
    elapsed = round(time.time() - t0, 3)

    r = res[0]
    preds = []

    if r.boxes is not None and len(r.boxes) > 0:
        for b in r.boxes:
            cls_id = int(b.cls[0].item()) if getattr(b.cls, "ndim", 0) else int(b.cls.item())
            conf = float(b.conf[0].item()) if getattr(b.conf, "ndim", 0) else float(b.conf.item())
            xyxy = [round(v, 2) for v in r.boxes.xyxy[0].tolist()]
            classe = labels.get(cls_id, str(cls_id))
            preds.append({"classe": classe, "conf": round(conf, 4), "bbox_xyxy": xyxy})

        # imagem anotada
        annotated = r.plot()              # numpy (BGR)
        annotated_rgb = annotated[:, :, ::-1]
        buf = io.BytesIO()
        Image.fromarray(annotated_rgb).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        img_b64 = f"data:image/png;base64,{b64}"

        top = max(preds, key=lambda p: p["conf"])
        print(f"[infer] {len(preds)} detecções | top={top}")
        return JSONResponse({
            "inference_time_s": elapsed,
            "task": "detect",
            "num_dets": len(preds),
            "top_pred": top,
            "preds": preds,
            "image_b64": img_b64
        })

    print("[infer] 0 detecções")
    return JSONResponse({
        "inference_time_s": elapsed,
        "task": "detect",
        "num_dets": 0,
        "top_pred": None,
        "preds": [],
        "image_b64": None
    })

@app.get("/ui")
def ui():
    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <title>Detecção de Pimentas (YOLOv8)</title>
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
      <h2>Detecção de Pimentas (YOLOv8)</h2>
      <small>Preset atual: <b>{PRESET}</b> | imgsz={CFG['imgsz']} | conf={CFG['conf']} | iou={CFG['iou']}</small>

      <div class="card">
        <input id="file" type="file" accept="image/*" capture="environment"/>
        <button onclick="doPredict()">Identificar</button>
        <p id="resumo"></p>
      </div>

      <div class="row">
        <div class="card">
          <h3>Entrada</h3>
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
        function showPreview(file) {{
          const img = document.getElementById('preview');
          img.src = URL.createObjectURL(file);
        }}

        async function doPredict() {{
          const f = document.getElementById('file').files[0];
          if (!f) {{ alert('Escolha uma imagem.'); return; }}
          showPreview(f);
          const fd = new FormData();
          fd.append('file', f);
          document.getElementById('resumo').innerText = 'Analisando...';

          try {{
            const r = await fetch('/predict', {{ method: 'POST', body: fd }});
            const data = await r.json();
            document.getElementById('json').innerText = JSON.stringify(data, null, 2);
            if (data.image_b64) document.getElementById('annotated').src = data.image_b64;

            if (data.top_pred) {{
              const pct = Math.round(data.top_pred.conf * 100);
              document.getElementById('resumo').innerText =
                'Detectou: ' + data.top_pred.classe + ' | Confiança: ' + pct + '% | Caixas: ' + data.num_dets;
            }} else {{
              document.getElementById('resumo').innerText = 'Nenhuma pimenta detectada (tente aproximar/centralizar).';
            }}
          }} catch (e) {{
            document.getElementById('resumo').innerText = 'Erro ao chamar a API.';
          }}
        }}

        document.getElementById('file').addEventListener('change', (e) => {{
          if (e.target.files[0]) showPreview(e.target.files[0]);
        }});
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
