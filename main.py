# main.py — YOLOv8 DETECÇÃO (Render Free / CPU)
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from ultralytics import YOLO
from PIL import Image
import io, json, time, os, requests, base64
from typing import List

app = FastAPI(title="API Pimentas YOLOv8 — Detecção")

# ========= CONFIG BÁSICA =========
MODEL_PATH = "best.pt"

# COLE AQUI o link direto do seu best.pt no Hugging Face
# Ex.: "https://huggingface.co/SEU_USUARIO/pimentas-model/resolve/main/best.pt"
MODEL_URL = os.getenv("MODEL_URL", "COLE_AQUI_O_LINK_DIRETO_DO_BEST_PT")

# Escolha UM preset: "RAPIDO", "EQUILIBRADO", "PRECISO", "MAX_RECALL"
PRESET = os.getenv("PRESET", "EQUILIBRADO")

PRESETS = {
    # Mais rápido (menor acurácia)
    "RAPIDO":      dict(imgsz=416, conf=0.30, iou=0.50, max_det=5),
    # Bom compromisso (padrão)
    "EQUILIBRADO": dict(imgsz=448, conf=0.30, iou=0.50, max_det=6),
    # Mais preciso (um pouco mais lento)
    "PRECISO":     dict(imgsz=512, conf=0.40, iou=0.55, max_det=8),
    # Máximo recall (para não perder detecções difíceis)
    "MAX_RECALL":  dict(imgsz=640, conf=0.12, iou=0.45, max_det=10),
}
CFG = PRESETS.get(PRESET, PRESETS["EQUILIBRADO"])

# ========= DOWNLOAD DO MODELO (se faltar) =========
def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    if not MODEL_URL or MODEL_URL.startswith("COLE_AQUI"):
        raise RuntimeError(
            "https://huggingface.co/bulipucca/pimentas-model/resolve/f704849d54bc90d866d4acceb663f6d11ed03a21/best.pt"
        )
    print(f"[init] Baixando modelo de: {MODEL_URL}")
    with requests.get(MODEL_URL, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print("[init] Download do modelo concluído.")

ensure_model()

# ========= CARREGA YOLO =========
model = YOLO(MODEL_PATH)
try:
    model.fuse()  # pequeno ganho de velocidade
except Exception:
    pass
labels = model.names  # {id: nome}

# ========= UTILS =========
def _encode_image(np_bgr) -> str | None:
    """Converte numpy BGR -> PNG base64 data URI."""
    try:
        rgb = np_bgr[:, :, ::-1]
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None

# ========= ROTAS =========
@app.get("/")
def root():
    return {
        "status": "ok",
        "task": getattr(model, "task", None),
        "preset": PRESET,
        "cfg": CFG,
        "classes": list(labels.values()),
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Detecção com leitura robusta das caixas + imagem anotada (base64).
    Nunca quebra a UI: em erro, retorna {"ok": False, "error": "..."} com HTTP 200.
    """
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

        # Leitura robusta dos tensores (evita erro de shape/ndim)
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

            annotated = r.plot()  # numpy (BGR)
            img_b64 = _encode_image(annotated)
            top = max(preds, key=lambda p: p["conf"]) if preds else None

            return JSONResponse({
                "ok": True,
                "inference_time_s": round(time.time() - t0, 3),
                "task": "detect",
                "num_dets": len(preds),
                "top_pred": top,
                "preds": preds,
                "image_b64": img_b64
            })

        # Sem detecções
        return JSONResponse({
            "ok": True,
            "inference_time_s": round(time.time() - t0, 3),
            "task": "detect",
            "num_dets": 0,
            "top_pred": None,
            "preds": [],
            "image_b64": None
        })

    except Exception as e:
        # Nunca estourar erro na UI
        return JSONResponse({
            "ok": False,
            "error": str(e),
            "inference_time_s": round(time.time() - t0, 3)
        }, status_code=200)

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
      <small>Preset: <b>{PRESET}</b> | imgsz={{CFG['imgsz']}} | conf={{CFG['conf']}} | iou={{CFG['iou']}}</small>

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
          document.getElementById('preview').src = URL.createObjectURL(file);
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

            if (data.ok === false) {{
              document.getElementById('resumo').innerText = 'Erro: ' + (data.error || 'desconhecido');
              return;
            }}

            if (data.top_pred) {{
              const pct = Math.round(data.top_pred.conf * 100);
              document.getElementById('resumo').innerText =
                'Detectou: ' + data.top_pred.classe + ' | Confiança: ' + pct + '% | Caixas: ' + data.num_dets;
            }} else {{
              document.getElementById('resumo').innerText = 'Nenhuma pimenta detectada.';
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
