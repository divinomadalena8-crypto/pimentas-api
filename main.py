# main.py — YOLOv8 (detecção) para Render Free
# - Uvicorn sobe rápido; modelo baixa/carrega em background (evita 502)
# - Sempre retorna imagem anotada (opcional via RETURN_IMAGE)
# - Salva PNG em /static/annotated e devolve image_url (RELATIVA)
# - Suporte a HF_TOKEN para repositório privado (Hugging Face)

import os, io, time, threading, base64, requests, uuid
from typing import List
from urllib.parse import urlparse

# Reduz overhead de threads BLAS no CPU free
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image
import numpy as np

app = FastAPI(title="API Pimentas YOLOv8")

# ===================== CONFIG =====================

MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://huggingface.co/bulipucca/pimentas-model/resolve/main/best.onnx"
)
MODEL_PATH = (
    os.path.basename(urlparse(MODEL_URL).path)
    if MODEL_URL and not MODEL_URL.startswith("COLE_AQUI")
    else "best.pt"
)

PRESET = os.getenv("PRESET", "ULTRA")
PRESETS = {
    "ULTRA":       dict(imgsz=320, conf=0.35, iou=0.50, max_det=4),
    "RAPIDO":      dict(imgsz=384, conf=0.30, iou=0.50, max_det=4),
    "EQUILIBRADO": dict(imgsz=448, conf=0.30, iou=0.50, max_det=6),
    "PRECISO":     dict(imgsz=512, conf=0.40, iou=0.55, max_det=8),
    "MAX_RECALL":  dict(imgsz=640, conf=0.12, iou=0.45, max_det=10),
}
CFG = PRESETS.get(PRESET, PRESETS["ULTRA"])

RETURN_IMAGE = True

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
REQ_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

STATIC_DIR = os.path.join(os.getcwd(), "static")
ANNOT_DIR  = os.path.join(STATIC_DIR, "annotated")
os.makedirs(ANNOT_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ===================== ESTADO GLOBAL =====================

model = None
labels = {}
READY = False
LOAD_ERR = None

# ===================== AUX =====================

def ensure_model_file():
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

def to_b64_png(np_bgr: np.ndarray) -> str | None:
    try:
        rgb = np_bgr[:, :, ::-1]
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None

def background_load():
    """Carrega YOLO e marca READY (sem inferência aqui para não travar startup)."""
    global model, labels, READY, LOAD_ERR
    try:
        t0 = time.time()
        ensure_model_file()
        m = YOLO(MODEL_PATH)
        try:
            m.fuse()
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

@app.head("/")
def health_head():
    return Response(status_code=200)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not READY:
        return JSONResponse({"ok": False, "warming_up": True, "error": LOAD_ERR}, status_code=503)

    t0 = time.time()
    try:
        im_bytes = await file.read()
        image = Image.open(io.BytesIO(im_bytes)).convert("RGB")

        # ↓ acelera em CPU, mantém proporção
        image.thumbnail((1024, 1024))  # in-place

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
                annotated = r.plot()  # np.ndarray (BGR)
                # salva arquivo
                fname = f"{uuid.uuid4().hex}.png"
                fpath = os.path.join(ANNOT_DIR, fname)
                Image.fromarray(annotated[:, :, ::-1]).save(fpath)  # BGR->RGB
                # URL RELATIVA (evita mixed content)
                image_url = f"/static/annotated/{fname}"
                # também em base64 (fallback imediato)
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
    """Espera READY e roda 1 inferência curtinha (compila caminho)."""
    t0 = time.time()
    while not READY and time.time() - t0 < 90:
        time.sleep(0.5)
    if not READY:
        return {"ok": False, "warming_up": True}
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    _ = model.predict(img, imgsz=CFG["imgsz"], conf=CFG["conf"], iou=CFG["iou"],
                      max_det=1, device="cpu", verbose=False)
    return {"ok": True}

from fastapi.responses import HTMLResponse  # (já deve estar importado)

@app.get("/ui")
def ui():
    html = r"""
<!doctype html>
<html lang="pt-br">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Pimentas • Detector</title>
  <style>
    :root { --bg:#0f172a; --fg:#e2e8f0; --muted:#94a3b8; --card:#111827; --accent:#22c55e; }
    * { box-sizing:border-box; }
    html,body { margin:0; background:linear-gradient(180deg,#0b1220,#0f172a 40%); color:var(--fg); font:400 16px/1.4 system-ui,-apple-system,Segoe UI,Roboto; }
    .wrap { max-width:980px; margin:auto; padding:20px 16px 56px; }
    header { display:flex; align-items:center; gap:12px; }
    header h1 { font-size:22px; margin:0; }
    header small { color:var(--muted); }
    .grid { display:grid; grid-template-columns:1fr; gap:16px; margin-top:16px; }
    @media(min-width:900px){ .grid { grid-template-columns: 1.1fr .9fr; } }
    .card { background:var(--card); border:1px solid #1f2937; border-radius:16px; padding:16px; box-shadow:0 6px 30px rgba(0,0,0,.25); }
    .btn { appearance:none; border:1px solid #334155; background:#0b1220; color:var(--fg); padding:10px 14px; border-radius:12px; cursor:pointer; font-weight:600; }
    .btn[disabled] { opacity:.6; cursor:not-allowed; }
    .btn.accent { background:var(--accent); border-color:transparent; color:#04110a; }
    .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
    .tip { color:var(--muted); font-size:13px; }
    .drop { border:2px dashed #334155; border-radius:16px; padding:14px; text-align:center; color:var(--muted); }
    .drop.drag { background:#0b1326; border-color:#64748b; color:#cbd5e1; }
    .imgwrap { background:#0b1220; border:1px solid #1f2937; border-radius:12px; padding:8px; }
    img,video,canvas { max-width:100%; display:block; border-radius:10px; }
    .pill { display:inline-block; padding:6px 10px; border-radius:999px; background:#0b1220; border:1px solid #334155; color:#cbd5e1; font-size:13px; }
    .status { margin:8px 0 0; min-height:22px; }
    details { margin-top:8px; }
    pre { background:#0b1220; border:1px solid #1f2937; color:#d1d5db; padding:12px; border-radius:12px; overflow:auto; max-height:340px; }
    footer { margin-top:18px; color:#94a3b8; font-size:12px; }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <img src="https://pimentas-api.onrender.com/static/favicon.png" alt="" width="28" height="28" onerror="this.style.display='none'">
      <h1>Detecção de Pimentas</h1>
      <small id="tagPreset" class="pill"></small>
    </header>

    <div class="grid">
      <!-- COLUNA ESQUERDA -->
      <section class="card">
        <div class="row">
          <button id="btnPick" class="btn">Escolher imagem</button>
          <input id="file" type="file" accept="image/*" capture="environment" style="display:none"/>
          <button id="btnCam" class="btn">Abrir câmera</button>
          <button id="btnShot" class="btn" style="display:none">Capturar</button>
          <button id="btnSend" class="btn accent" disabled>Identificar</button>
          <span id="chip" class="pill">Conectando...</span>
        </div>
        <p class="tip">Dica: a qualidade ideal é ~1024px no maior lado. Enviaremos a imagem comprimida para acelerar.</p>

        <div id="drop" class="drop" style="margin-top:10px">Solte uma imagem aqui…</div>

        <div class="row" style="margin-top:10px">
          <div class="imgwrap" style="flex:1">
            <small class="tip">Original</small>
            <video id="video" autoplay playsinline style="display:none"></video>
            <img id="preview" alt="preview" style="display:none"/>
            <canvas id="canvas" style="display:none"></canvas>
          </div>
          <div class="imgwrap" style="flex:1">
            <small class="tip">Resultado</small>
            <img id="annotated" alt="resultado"/>
          </div>
        </div>

        <div id="resumo" class="status tip"></div>
      </section>

      <!-- COLUNA DIREITA -->
      <aside class="card">
        <div class="row" style="justify-content:space-between">
          <strong>Resposta</strong>
          <span id="badgeTime" class="pill">–</span>
        </div>
        <details>
          <summary>Ver JSON</summary>
          <pre id="json">{}</pre>
        </details>
        <footer>Powered by YOLOv8 (CPU). © Você 🙂</footer>
      </aside>
    </div>
  </div>

<script>
const API = window.location.origin;
let currentFile = null;   // File/Blob atual a enviar
let stream = null;        // MediaStream da câmera

// ---------- UTIL ----------
function sleep(ms){ return new Promise(r => setTimeout(r, ms)); }

async function compressImage(file, maxSide=1024, quality=0.8) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const w = img.width, h = img.height;
      const scale = Math.min(1, maxSide / Math.max(w,h));
      const cw = Math.round(w*scale), ch = Math.round(h*scale);
      const cv = document.getElementById('canvas');
      const ctx = cv.getContext('2d');
      cv.width = cw; cv.height = ch;
      ctx.drawImage(img, 0, 0, cw, ch);
      cv.toBlob(b => {
        if(!b) return reject(new Error("compress failed"));
        resolve(new File([b], file.name || "photo.jpg", { type:"image/jpeg" }));
      }, "image/jpeg", quality);
    };
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}

function setStatus(txt){ document.getElementById('chip').textContent = txt; }

// ---------- PING/WARMUP ----------
async function waitReady(){
  setStatus("Conectando…");
  try{
    const r = await fetch(API + "/", {cache:"no-store"});
    const d = await r.json();
    document.getElementById('tagPreset').textContent = d.preset ? ("preset: " + d.preset) : "";
    if(d.ready){
      setStatus("Pronto");
      document.getElementById('btnSend').disabled = !currentFile;
      return;
    }
    setStatus("Aquecendo…");
  }catch(e){
    setStatus("Sem conexão, tentando…");
  }
  await sleep(1200);
  waitReady();
}

// ---------- PICKER / DRAG & DROP ----------
const fileInput = document.getElementById('file');
document.getElementById('btnPick').onclick = () => fileInput.click();
fileInput.onchange = () => useLocalFile(fileInput.files?.[0]);

const drop = document.getElementById('drop');
["dragenter","dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("drag"); }));
["dragleave","drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("drag"); }));
drop.addEventListener("drop", e => {
  const f = e.dataTransfer?.files?.[0];
  if(f) useLocalFile(f);
});

async function useLocalFile(f){
  if(!f) return;
  // Comprime antes de enviar
  currentFile = await compressImage(f);
  document.getElementById('preview').src = URL.createObjectURL(currentFile);
  document.getElementById('preview').style.display = "block";
  document.getElementById('video').style.display = "none";
  document.getElementById('btnSend').disabled = false;
  document.getElementById('resumo').textContent = "";
}

// ---------- CÂMERA ----------
const btnCam  = document.getElementById('btnCam');
const btnShot = document.getElementById('btnShot');
const video   = document.getElementById('video');

btnCam.onclick = async () => {
  try{
    stream = await (navigator.mediaDevices && navigator.mediaDevices.getUserMedia)
      ? navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } })
      : null;
    if(!stream){ alert("Seu navegador/webview não permite câmera aqui. Use 'Escolher imagem' ou abra no navegador do aparelho."); return; }
    video.srcObject = stream;
    video.style.display = "block";
    document.getElementById('preview').style.display = "none";
    btnShot.style.display = "inline-block";
  }catch(e){
    alert("Não foi possível abrir a câmera. Permissão concedida?");
  }
};

btnShot.onclick = () => {
  // tira foto do vídeo e vira File
  const cv = document.getElementById('canvas');
  const ctx = cv.getContext('2d');
  cv.width = video.videoWidth; cv.height = video.videoHeight;
  ctx.drawImage(video, 0, 0);
  cv.toBlob(async b => {
    currentFile = await compressImage(new File([b], "camera.jpg", {type:"image/jpeg"}));
    document.getElementById('preview').src = URL.createObjectURL(currentFile);
    document.getElementById('preview').style.display = "block";
    video.style.display = "none";
    btnShot.style.display = "none";
    if(stream){ stream.getTracks().forEach(t => t.stop()); stream = null; }
    document.getElementById('btnSend').disabled = false;
  },"image/jpeg",0.9);
};

// ---------- PREDICT ----------
document.getElementById('btnSend').onclick = async () => {
  if(!currentFile){ return; }
  document.getElementById('btnSend').disabled = true;
  document.getElementById('resumo').textContent = "Enviando...";
  const t0 = performance.now();
  try{
    const fd = new FormData();
    fd.append("file", currentFile, currentFile.name || "image.jpg"); // campo 'file'
    const r = await fetch(API + "/predict", { method:"POST", body: fd });
    const d = await r.json();
    document.getElementById('json').textContent = JSON.stringify(d, null, 2);

    if(d.ok === false && d.warming_up){ 
      document.getElementById('resumo').textContent = "Aquecendo o modelo, tente novamente...";
      document.getElementById('btnSend').disabled = false;
      return;
    }
    if(d.ok === false){ 
      document.getElementById('resumo').textContent = "Erro: " + (d.error || "desconhecido");
      document.getElementById('btnSend').disabled = false;
      return;
    }
    // imagem anotada
    if(d.image_b64){
      document.getElementById('annotated').src = d.image_b64;
    }else if(d.image_url){
      const url = d.image_url.startsWith("http") ? d.image_url : (API + d.image_url);
      document.getElementById('annotated').src = url;
    }
    // resumo
    const ms = (performance.now()-t0)/1000;
    document.getElementById('badgeTime').textContent = (d.inference_time_s || ms).toFixed(2) + " s";
    if(d.top_pred){
      const pct = Math.round((d.top_pred.conf||0)*100);
      document.getElementById('resumo').textContent = `Detectou: ${d.top_pred.classe} · ${pct}% · Caixas: ${d.num_dets}`;
    }else{
      document.getElementById('resumo').textContent = `Nenhuma pimenta detectada.`;
    }
  }catch(e){
    document.getElementById('resumo').textContent = "Falha ao chamar a API.";
  }finally{
    document.getElementById('btnSend').disabled = false;
  }
};

// start
waitReady();
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)

