# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from ultralytics import YOLO
from PIL import Image
import io, json, time, os, requests

app = FastAPI(title="API Pimentas YOLOv8")

# =========================
# CONFIG DO MODELO
# =========================
MODEL_PATH = "best.pt"

# COLE AQUI o link direto do seu best.pt no Hugging Face (ou defina a env var MODEL_URL no Render)
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://huggingface.co/bulipucca/pimentas-model/resolve/main/best.pt"
)

def ensure_model():
    """Baixa o best.pt se ainda não existir localmente."""
    if os.path.exists(MODEL_PATH):
        return
    if not MODEL_URL or MODEL_URL.startswith("COLE_AQUI"):
        raise RuntimeError(
            "MODEL_URL não definido. Cole o link do best.pt no código ou crie a variável de ambiente MODEL_URL."
        )
    print(f"[init] Baixando modelo de: {MODEL_URL}")
    with requests.get(MODEL_URL, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print("[init] Download do modelo concluído.")

# Baixa (se precisar) e carrega o YOLO
ensure_model()
model = YOLO(MODEL_PATH)
labels = model.names  # {id: nome}

# (opcional) infos extras por variedade
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
    # 'model.task' costuma ser "detect", "segment" ou "classify"
    task = getattr(model, "task", None)
    return {"status": "ok", "model": MODEL_PATH, "task": task, "classes": list(labels.values())}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Aceita imagem e retorna top_pred + lista completa (detecção OU classificação)."""
    im_bytes = await file.read()
    image = Image.open(io.BytesIO(im_bytes)).convert("RGB")

    t0 = time.time()
    # imgsz 640 dá mais chance de detectar; conf 0.10 fica mais sensível
    res = model.predict(image, imgsz=640, conf=0.10, device="cpu", verbose=False)
    elapsed = round(time.time() - t0, 3)

    out = {"inference_time_s": elapsed, "task": getattr(model, "task", None)}
    preds = []
    r = res[0]

    # --- CAMINHO 1: DETECÇÃO (tem caixas) ---
    if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
        for b in r.boxes:
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            xyxy = [round(v, 2) for v in b.xyxy[0].tolist()]
            classe = labels.get(cls_id, str(cls_id))
            preds.append({"classe": classe, "conf": round(conf, 4), "bbox_xyxy": xyxy})
        top = max(preds, key=lambda p: p["conf"]) if preds else None
        out.update({"top_pred": top, "preds": preds})
        return JSONResponse(out)

    # --- CAMINHO 2: CLASSIFICAÇÃO (tem probs) ---
    if hasattr(r, "probs") and r.probs is not None:
        # top1
        top1 = int(r.probs.top1)
        top1conf = float(r.probs.top1conf)
        top_pred = {"classe": labels.get(top1, str(top1)), "conf": round(top1conf, 4)}
        # top5
        idxs = [int(i) for i in r.probs.top5]
        confs = [float(c) for c in r.probs.top5conf]
        preds = [{"classe": labels.get(i, str(i)), "conf": round(c, 4)} for i, c in zip(idxs, confs)]
        out.update({"top_pred": top_pred, "preds": preds})
        return JSONResponse(out)

    # --- fallback (nada retornado) ---
    out.update({"top_pred": None, "preds": []})
    return JSONResponse(out)

@app.get("/pepperinfo/{classe}")
def pepper_info(classe: str):
    data = PEPPER_INFO.get(classe)
    if not data:
        return JSONResponse({"erro": f"Sem informações para {classe}"}, status_code=404)
    return data

# =========================
# UI mínima para teste (/ui)
# =========================
@app.get("/ui")
def ui():
    html = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <title>Classificador de Pimentas</title>
      <style>
        body { font-family: Arial, sans-serif; padding:16px; max-width:700px; margin:auto; }
        .card { border:1px solid #ddd; border-radius:12px; padding:16px; margin-top:12px; }
        button { padding:10px 16px; border-radius:10px; border:1px solid #ccc; background:#f8f8f8; cursor:pointer; }
        img { max-width:100%; border-radius:12px; margin-top:8px; }
        pre { background:#f4f4f4; padding:12px; border-radius:8px; overflow:auto; }
      </style>
    </head>
    <body>
      <h2>Classificador de Pimentas (YOLOv8)</h2>
      <div class="card">
        <input id="file" type="file" accept="image/*" capture="environment"/>
        <button onclick="doPredict()">Identificar</button>
        <div id="preview"></div>
      </div>
      <div class="card">
        <h3>Resultado</h3>
        <div id="resumo"></div>
        <pre id="json"></pre>
      </div>

      <script>
        function showPreview(file) {
          const img = document.createElement('img');
          img.src = URL.createObjectURL(file);
          const prev = document.getElementById('preview');
          prev.innerHTML = '';
          prev.appendChild(img);
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

            if (data.top_pred) {
              const pct = Math.round(data.top_pred.conf * 100);
              document.getElementById('resumo').innerText =
                'Pimenta: ' + data.top_pred.classe + '  |  Precisão: ' + pct + '%';
            } else {
              document.getElementById('resumo').innerText = 'Nenhuma pimenta identificada.';
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
