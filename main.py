# main.py — DETECÇÃO YOLOv8
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from ultralytics import YOLO
from PIL import Image
import io, json, time, os, requests, base64

app = FastAPI(title="API Pimentas YOLOv8 — Detecção")

# =========================
# MODELO
# =========================
MODEL_PATH = "best.pt"

# COLE AQUI o link direto do seu best.pt no Hugging Face (ou use a env var MODEL_URL no Render)
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://huggingface.co/bulipucca/pimentas-model/resolve/main/best.pt"
)

def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    if not MODEL_URL or MODEL_URL.startswith("COLE_AQUI"):
        raise RuntimeError("Defina MODEL_URL (ou cole o link no código).")
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
labels = model.names  # {id: nome}

# (opcional) infos extras
try:
    with open("pepper_info.json", "r", encoding="utf-8") as f:
        PEPPER_INFO = json.load(f)
except Exception:
    PEPPER_INFO = {}

# =========================
# ROTAS
# =========================
@app.get("/")
def root():
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "task": getattr(model, "task", None),
        "classes": list(labels.values())
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Detecção: retorna caixas + imagem anotada (base64).
    Thresholds mais sensíveis para não perder detecções.
    """
    im_bytes = await file.read()
    image = Image.open(io.BytesIO(im_bytes)).convert("RGB")

    t0 = time.time()
    # ajustes importantes para sensibilidade e estabilidade em CPU free
    res = model.predict(
        image,
        imgsz=640,     # maior que 480 melhora recall
        conf=0.10,     # mais sensível (pode subir p/ 0.25 depois)
        iou=0.45,      # NMS padrão estável
        device="cpu",
        verbose=False
    )
    elapsed = round(time.time() - t0, 3)

    r = res[0]
    preds = []

    if r.boxes is not None and len(r.boxes) > 0:
        for b in r.boxes:
            cls_id = int(b.cls[0].item()) if b.cls.ndim else int(b.cls.item())
            conf = float(b.conf[0].item()) if b.conf.ndim else float(b.conf.item())
            xyxy = [round(v, 2) for v in b.xyxy[0].tolist()]
            classe = labels.get(cls_id, str(cls_id))
            preds.append({"classe": classe, "conf": round(conf, 4), "bbox_xyxy": xyxy})

        # imagem anotada (PNG base64)
        annotated = r.plot()  # numpy (BGR)
        annotated_rgb = annotated[:, :, ::-1]  # BGR->RGB
        buf = io.BytesIO()
        Image.fromarray(annotated_rgb).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        img_b64 = f"data:image/png;base64,{b64}"

        top = max(preds, key=lambda p: p["conf"])
        print(f"[infer] {len(preds)} detecções. Top: {top}")
        return JSONResponse({
            "inference_time_s": elapsed,
            "task": "detect",
            "num_dets": len(preds),
            "top_pred": top,
            "preds": preds,
            "image_b64": img_b64
        })

    print("[infer] 0 detecções.")
    return JSONResponse({
        "inference_time_s": elapsed,
        "task": "detect",
        "num_dets": 0,
        "top_pred": None,
        "preds": [],
        "image_b64": None
    })

@app.get("/pepperinfo/{classe}")
def pepper_info(classe: str):
    data = PEPPER_INFO.get(classe)
    if not data:
        return JSONResponse({"erro": f"Sem informações para {classe}"}, status_code=404)
    return data

# =========================
# UI /ui (mostra imagem anotada)
# =========================
@app.get("/ui")
def ui():
    html = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <title>Detecção de Pimentas (YOLOv8)</title>
      <style>
        body { font-family: Arial, sans-serif; padding:16px; max-width:800px; margin:auto; }
        .row { display:flex; gap:16px; flex-wrap:wrap; }
        .card { flex:1 1 360px; border:1px solid #ddd; border-radius:12px; padding:16px; }
        img { max-width:100%; border-radius:12px; }
        button { padding:10px 16px; border-radius:10px; border:1px solid #ccc; background:#f8f8f8; cursor:pointer; }
        pre { background:#f4f4f4; padding:12px; border-radius:8px; overflow:auto; }
      </style>
    </head>
    <body>
      <h2>Detecção de Pimentas (YOLOv8)</h2>

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
        function showPreview(file) {
          const img = document.getElementById('preview');
          img.src = URL.createObjectURL(file);
        }

        async function doPredict() {
          const f = document.getElementById('file').files[0];
          if (!f) { alert('Escolha uma imagem.'); return; }
          showPreview(f);
          const fd = new FormData();
          fd.append('file', f);
          document.getElementById('resumo').innerText = 'Analisando...';

          try {
            const r = await fetch('/predict', { method: 'POST', body: fd });
            const data = await r.json();
            document.getElementById('json').innerText = JSON.stringify(data, null, 2);
            if (data.image_b64) document.getElementById('annotated').src = data.image_b64;

            if (data.top_pred) {
              const pct = Math.round(data.top_pred.conf * 100);
              document.getElementById('resumo').innerText =
                'Detectou: ' + data.top_pred.classe + ' | Confiança: ' + pct + '% | Caixas: ' + data.num_dets;
            } else {
              document.getElementById('resumo').innerText = 'Nenhuma pimenta detectada (tente aproximar/centralizar).';
            }
          } catch (e) {
            document.getElementById('resumo').innerText = 'Erro ao chamar a API.';
          }
        }

        document.getElementById('file').addEventListener('change', (e) => {
          if (e.target.files[0]) showPreview(e.target.files[0]);
        });
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
