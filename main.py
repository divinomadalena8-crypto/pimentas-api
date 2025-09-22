# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from ultralytics import YOLO
from PIL import Image
import io, json, time

app = FastAPI(title="API Pimentas YOLOv8m")

# ====== CARREGAMENTO DO MODELO E METADADOS ======
MODEL_PATH = "best.pt"  # coloque seu best.pt na raiz do repo
model = YOLO(MODEL_PATH)
labels = model.names  # dict {id: nome}

try:
    with open("pepper_info.json", "r", encoding="utf-8") as f:
        PEPPER_INFO = json.load(f)
except:
    PEPPER_INFO = {}

# ====== ROTAS ======
@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_PATH, "classes": list(labels.values())}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Recebe imagem e retorna top_pred + lista completa."""
    im_bytes = await file.read()
    image = Image.open(io.BytesIO(im_bytes)).convert("RGB")

    t0 = time.time()
    res = model.predict(image, imgsz=640, conf=0.25, device="cpu", verbose=False)
    elapsed = round(time.time() - t0, 3)

    preds = []
    if res and len(res) > 0:
        r = res[0]
        for b in r.boxes:
            cls_id = int(b.cls.item())
            conf = float(b.conf.item())
            xyxy = [round(v, 2) for v in b.xyxy[0].tolist()]
            classe = labels.get(cls_id, str(cls_id))
            preds.append({"classe": classe, "conf": round(conf, 4), "bbox_xyxy": xyxy})

    top = max(preds, key=lambda p: p["conf"]) if preds else None
    return JSONResponse({"inference_time_s": elapsed, "top_pred": top, "preds": preds})

@app.get("/pepperinfo/{classe}")
def pepper_info(classe: str):
    data = PEPPER_INFO.get(classe)
    if not data:
        return JSONResponse({"erro": f"Sem informações para {classe}"}, status_code=404)
    return data

# ====== UI MINIMALISTA PARA TESTAR (funciona no WebViewer) ======
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
      <h2>Classificador de Pimentas (YOLOv8m)</h2>
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
              document.getElementById('resumo').innerText = 'Nenhuma pimenta detectada.';
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
