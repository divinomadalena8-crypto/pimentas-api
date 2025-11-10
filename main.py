# main.py — API + UI (YOLOv8 + Chat estilo WhatsApp com sinônimos e fallback)
# Render-friendly: baixa o modelo em background, serve /ui e /info (chat)

import os, io, time, threading, base64, uuid, requests
from urllib.parse import urlparse
from typing import List, Optional

# Menos threads BLAS em instâncias free
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from PIL import Image
import numpy as np

# YOLO (Ultralytics)
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# App e diretórios estáticos
# -----------------------------------------------------------------------------
app = FastAPI(title="Pimentas API/UI")

STATIC_DIR = os.path.join(os.getcwd(), "static")
ANNOT_DIR  = os.path.join(STATIC_DIR, "annotated")
os.makedirs(ANNOT_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# -----------------------------------------------------------------------------
# Config do modelo
# -----------------------------------------------------------------------------
MODEL_URL = os.getenv(
    "MODEL_URL",
    # Ex.: ONNX do HuggingFace (troque se necessário)
    "https://huggingface.co/bulipucca/pimentas-model/resolve/main/best.onnx"
)
MODEL_PATH = (
    os.path.basename(urlparse(MODEL_URL).path)
    if MODEL_URL and not MODEL_URL.startswith("COLE_AQUI")
    else "best.pt"
)

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
REQ_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# Presets (aqui usamos até 10 caixas)
PRESET = os.getenv("PRESET", "ULTRA")
PRESETS = {
    "ULTRA":       dict(imgsz=384, conf=0.30, iou=0.50, max_det=10),
    "RAPIDO":      dict(imgsz=352, conf=0.28, iou=0.50, max_det=8),
    "EQUILIBRADO": dict(imgsz=448, conf=0.30, iou=0.50, max_det=10),
    "PRECISO":     dict(imgsz=512, conf=0.40, iou=0.55, max_det=12),
    "MAX_RECALL":  dict(imgsz=640, conf=0.12, iou=0.45, max_det=16),
}
CFG = PRESETS.get(PRESET, PRESETS["ULTRA"])

RETURN_IMAGE = True  # salva e retorna anotada

# -----------------------------------------------------------------------------
# Estado global do modelo
# -----------------------------------------------------------------------------
model: Optional[YOLO] = None
labels: dict = {}
READY = False
LOAD_ERR: Optional[str] = None


# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
def ensure_model_file():
    """Baixa arquivo de modelo se não existir."""
    if os.path.exists(MODEL_PATH):
        return
    if not MODEL_URL or MODEL_URL.startswith("COLE_AQUI"):
        raise RuntimeError("Defina MODEL_URL com link direto (.pt ou .onnx).")
    print(f"[init] Baixando modelo: {MODEL_URL}")
    with requests.get(MODEL_URL, headers=REQ_HEADERS, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print("[init] Download concluído:", MODEL_PATH)


def to_b64_png(np_bgr: np.ndarray) -> Optional[str]:
    try:
        rgb = np_bgr[:, :, ::-1]
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def background_load():
    """Carrega YOLO em background para inicialização rápida."""
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


@app.on_event("startup")
def on_startup():
    threading.Thread(target=background_load, daemon=True).start()


# -----------------------------------------------------------------------------
# Endpoints API
# -----------------------------------------------------------------------------
@app.get("/")
def health():
    return {
        "status": "ok" if READY else "warming",
        "ready": READY,
        "error": LOAD_ERR,
        "model": MODEL_PATH if MODEL_PATH else None,
        "classes": list(labels.values()) if READY else None,
    }


@app.head("/")
def health_head():
    return Response(status_code=200)


@app.get("/warmup")
def warmup():
    """Compila caminho e marca pronto (chamada curta)."""
    t0 = time.time()
    while not READY and time.time() - t0 < 90:
        time.sleep(0.5)
    if not READY:
        return {"ok": False, "warming_up": True}
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    _ = model.predict(img, imgsz=CFG["imgsz"], conf=CFG["conf"], iou=CFG["iou"],
                      max_det=1, device="cpu", verbose=False)
    return {"ok": True}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Recebe imagem, roda YOLO e devolve JSON + imagem anotada (URL e base64)."""
    if not READY:
        return JSONResponse({"ok": False, "warming_up": True, "error": LOAD_ERR}, status_code=503)

    t0 = time.time()
    try:
        im_bytes = await file.read()
        image = Image.open(io.BytesIO(im_bytes)).convert("RGB")
        image.thumbnail((1024, 1024))  # acelera em CPU mantendo proporção

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
                annotated = r.plot()  # np.ndarray BGR
                fname = f"{uuid.uuid4().hex}.png"
                fpath = os.path.join(ANNOT_DIR, fname)
                Image.fromarray(annotated[:, :, ::-1]).save(fpath)  # BGR->RGB
                image_url = f"/static/annotated/{fname}"
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


# -----------------------------------------------------------------------------
# Páginas UI
# -----------------------------------------------------------------------------
@app.get("/ui")
def ui():
    html = r"""
<!doctype html>
<html lang="pt-br">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Identificação de Pimentas</title>
  <link rel="icon" href="/static/pimenta-logo.png" type="image/png" sizes="any">
  <style>
    :root{ --bg:#f7fafc; --card:#ffffff; --fg:#0f172a; --muted:#475569; --line:#e2e8f0; --accent:#16a34a; --accent2:#2563eb; }
    *{box-sizing:border-box}
    html,body{ margin:0;background:var(--bg);color:var(--fg);font:400 16px/1.45 system-ui,-apple-system,Segoe UI,Roboto }
    .wrap{max-width:980px;margin:auto;padding:20px 14px 20px}
    header{display:flex;align-items:center;gap:10px}
    header h1{font-size:22px;margin:0}
    .card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:0 4px 24px rgba(15,23,42,.06)}
    .btn{appearance:none;border:1px solid var(--line);background:#fff;color:var(--fg);padding:10px 14px;border-radius:12px;cursor:pointer;font-weight:600}
    .btn[disabled]{opacity:.6;cursor:not-allowed}
    .btn.accent{background:var(--accent);border-color:var(--accent);color:#fff}
    .btn.info{background:var(--accent2);border-color:var(--accent2);color:#fff}
    .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
    .tip{color:var(--muted);font-size:13px}
    .imgwrap{background:#fff;border:1px solid var(--line);border-radius:12px;padding:8px}
    img,video,canvas{max-width:100%;display:block;border-radius:10px}
    .status{margin-top:8px;min-height:22px}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <img src="/static/pimenta-logo.png" alt="Logo" width="28" height="28" onerror="this.style.display='none'">
      <h1>Identificação de Pimentas</h1>
    </header>

    <section class="card">
      <div class="row">
        <button id="btnPick" class="btn">Escolher imagem</button>
        <input id="fileGallery" type="file" accept="image/*" style="display:none"/>
        <input id="fileCamera"  type="file" accept="image/*" capture="environment" style="display:none"/>

        <button id="btnCam" class="btn">Abrir câmera</button>
        <button id="btnShot" class="btn" style="display:none">Capturar</button>

        <button id="btnSend" class="btn accent" disabled>Identificar</button>
        <button id="btnChat" class="btn info" style="display:none">Mais informações</button>
      </div>
      <p class="tip">A imagem é comprimida para ~1024px antes do envio para acelerar.</p>

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
  </div>

<script>
const API = window.location.origin;
let currentFile = null;
let stream = null;
let lastClass = null;

function sleep(ms){ return new Promise(r=>setTimeout(r,ms)); }
function setStatus(txt){ document.getElementById('resumo').textContent = txt; }

async function compressImage(file, maxSide=1024, quality=0.85){
  return new Promise((resolve,reject)=>{
    const img = new Image();
    img.onload=()=>{
      const scale=Math.min(1, maxSide/Math.max(img.width,img.height));
      const w=Math.round(img.width*scale), h=Math.round(img.height*scale);
      const cv=document.getElementById('canvas'), ctx=cv.getContext('2d');
      cv.width=w; cv.height=h; ctx.drawImage(img,0,0,w,h);
      cv.toBlob(b=>{
        if(!b) return reject(new Error("compress fail"));
        resolve(new File([b], file.name||"photo.jpg", {type:"image/jpeg"}));
      },"image/jpeg",quality);
    };
    img.onerror=reject;
    img.src=URL.createObjectURL(file);
  });
}

async function waitReady(){
  try{
    const r = await fetch(API + "/", {cache:"no-store"});
    const d = await r.json();
    if(!d.ready){
      setStatus("Iniciando o modelo…");
      await sleep(1200);
      return waitReady();
    }
    setStatus("");
  }catch(e){
    setStatus("Sem conexão, tentando novamente…");
    await sleep(1200);
    return waitReady();
  }
}
waitReady();

// -------- inputs
const inputGallery = document.getElementById('fileGallery');
const inputCamera  = document.getElementById('fileCamera');

document.getElementById('btnPick').onclick = () => {
  inputGallery.value = "";
  inputGallery.click();
};
inputGallery.onchange = () => useLocalFile(inputGallery.files?.[0]);
inputCamera.onchange  = () => useLocalFile(inputCamera.files?.[0]);

async function useLocalFile(f){
  if(!f) return;
  currentFile = await compressImage(f);
  document.getElementById('preview').src = URL.createObjectURL(currentFile);
  document.getElementById('preview').style.display = "block";
  document.getElementById('video').style.display   = "none";
  document.getElementById('btnSend').disabled = false;
  document.getElementById('btnChat').style.display = "none";
  lastClass = null;
  setStatus("Imagem pronta para envio.");
}

const btnCam  = document.getElementById('btnCam');
const btnShot = document.getElementById('btnShot');
const video   = document.getElementById('video');

btnCam.onclick = async () => {
  try{
    if(!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) throw new Error("no gum");
    stream = await navigator.mediaDevices.getUserMedia({ video:{ facingMode:{ideal:"environment"} } });
    video.srcObject = stream;
    video.style.display = "block";
    document.getElementById('preview').style.display = "none";
    btnShot.style.display = "inline-block";
    setStatus("Câmera aberta");
  }catch(e){
    inputCamera.value = "";
    inputCamera.click(); // fallback: seletor que abre câmera nativa no WebView
  }
};

btnShot.onclick = () => {
  const cv=document.getElementById('canvas'), ctx=cv.getContext('2d');
  cv.width=video.videoWidth; cv.height=video.videoHeight;
  ctx.drawImage(video,0,0);
  cv.toBlob(async b=>{
    currentFile = await compressImage(new File([b],"camera.jpg",{type:"image/jpeg"}));
    document.getElementById('preview').src = URL.createObjectURL(currentFile);
    document.getElementById('preview').style.display = "block";
    video.style.display = "none";
    btnShot.style.display = "none";
    if(stream){ stream.getTracks().forEach(t=>t.stop()); stream=null; }
    document.getElementById('btnSend').disabled = false;
    document.getElementById('btnChat').style.display = "none";
    lastClass = null;
    setStatus("Foto capturada");
  },"image/jpeg",0.92);
};

// -------- predição
document.getElementById('btnSend').onclick = async () => {
  if(!currentFile) return;
  document.getElementById('btnSend').disabled = true;
  setStatus("Processando…");
  try{
    const fd=new FormData(); fd.append("file", currentFile, currentFile.name||"image.jpg");
    const r=await fetch(API + "/predict", {method:"POST", body:fd});
    const d=await r.json();

    if(d.ok===false && d.warming_up){ setStatus("Aquecendo o modelo… toque novamente"); return; }
    if(d.ok===false){ setStatus("Erro: " + (d.error||"desconhecido")); return; }

    if(d.image_b64){ document.getElementById('annotated').src = d.image_b64; }
    else if(d.image_url){
      const url = d.image_url.startsWith("http")? d.image_url : (API + d.image_url);
      document.getElementById('annotated').src = url;
    }

    const resumo = d.top_pred ? `Detectado: ${d.top_pred.classe} · ${Math.round((d.top_pred.conf||0)*100)}% · Caixas: ${d.num_dets}`
                              : "Nenhuma pimenta detectada.";
    setStatus(resumo);

    const chatBtn = document.getElementById('btnChat');
    if(d.top_pred && d.top_pred.classe){
      lastClass = d.top_pred.classe;
      chatBtn.style.display = "inline-block";
      chatBtn.onclick = () => { location.href = "/info?pepper=" + encodeURIComponent(lastClass); };
    }else{
      chatBtn.style.display = "none";
      lastClass = null;
    }
  }catch(e){
    setStatus("Falha ao chamar a API.");
  }finally{
    document.getElementById('btnSend').disabled = false;
  }
};
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)


@app.get("/info")
def info():
    html = r"""
<!doctype html>
<html lang="pt-br">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no"/>
  <title>Chat de Pimentas</title>
  <style>
    :root{ --bg:#eef2f7; --chat:#fefefe; --mine:#dbeafe; --their:#f1f5f9; --fg:#0f172a; --muted:#475569; --line:#e2e8f0; --accent:#16a34a; }
    *{box-sizing:border-box}
    html,body{margin:0;background:var(--bg);color:var(--fg);font:400 16px/1.45 system-ui,-apple-system,Segoe UI,Roboto;height:100%}
    .wrap{max-width:880px;margin:auto;height:100%;display:flex;flex-direction:column}
    header{padding:10px 12px;display:flex;align-items:center;gap:8px}
    header .back{appearance:none;border:1px solid var(--line);background:#fff;border-radius:10px;padding:6px 10px;cursor:pointer}
    header h1{font-size:18px;margin:0}
    .chat{flex:1;display:flex;flex-direction:column;padding:10px}
    .board{flex:1;background:var(--chat);border:1px solid var(--line);border-radius:14px;padding:10px;overflow:auto}
    .row{display:flex;margin:6px 0}
    .me{justify-content:flex-end}
    .bubble{max-width:78%;padding:10px 12px;border-radius:12px;border:1px solid var(--line);white-space:pre-wrap}
    .bubble.me{background:var(--mine)}
    .bubble.other{background:var(--their)}
    .input{display:flex;gap:8px;margin-top:8px}
    .input input{flex:1;padding:12px 12px;border-radius:14px;border:1px solid var(--line);font-size:16px}
    .input button{padding:12px 14px;border-radius:14px;border:1px solid var(--line);background:var(--accent);color:#fff;font-weight:700;cursor:pointer}
    .hint{color:var(--muted);font-size:13px;margin-bottom:6px}
    .pill{display:inline-block;padding:4px 8px;border:1px solid var(--line);border-radius:999px;background:#fff;margin-right:6px;font-size:12px}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <button class="back" onclick="history.back()">← Voltar</button>
      <h1 id="title">Chat</h1>
    </header>

    <div class="chat">
      <div class="hint"><span class="pill">Digite 1–8</span> ou escreva o nome/sinônimo da pimenta (ex.: <em>chilli, jalapeño, biquinho…</em>)</div>
      <div id="board" class="board"></div>
      <div class="input">
        <input id="inp" placeholder="Digite a opção (1–8) ou o nome da pimenta…"/>
        <button id="send">Enviar</button>
      </div>
    </div>
  </div>

<script>
const qs = new URLSearchParams(location.search);
const pepperFromQuery = qs.get("pepper") || "";

let KB = null;    // base de conhecimento
let CUR = null;   // chave atual da pimenta

// ------------------ utilidades ------------------
function el(tag, cls, text){ const e=document.createElement(tag); if(cls) e.className=cls; if(text!=null) e.textContent=text; return e; }
function addMsg(txt, me=false){
  const row=el("div","row"+(me?" me":""));
  const b=el("div","bubble"+(me?" me":" other")); b.textContent=txt;
  row.appendChild(b); document.getElementById("board").appendChild(row);
  const bd=document.getElementById("board"); bd.scrollTop=bd.scrollHeight;
}
function normalize(s){
  return (s||"").toString().toLowerCase()
    .normalize("NFD").replace(/[\u0300-\u036f]/g,"") // tira acentos
    .replace(/[^a-z0-9\s-]/g," ").replace(/\s+/g," ").trim();
}

// fallback mínimo — usado se /static/pepper_info.json falhar
const FALLBACK_KB = {
 "Biquinho-Pepper": {nome:"Biquinho-Pepper", sinonimos:["biquinho","pimenta biquinho"] ,
   descricao:"Muito suave, aromática, ótima para conservas e petiscos.", scoville:"0–1.000"},
 "Bode-Pepper": {nome:"Bode-Pepper", sinonimos:["bode","pimenta bode"],
   descricao:"Picância média-baixa; comum no Centro-Oeste.", scoville:"5.000–15.000"},
 "Chili-Pepper": {nome:"Chili-Pepper", sinonimos:["chili","chilli","pimenta chili","pimenta chilli"],
   descricao:"Termo amplo (misturas/variedades). Picância variável.", scoville:"varia"},
 "Fidalga-Pepper": {nome:"Fidalga-Pepper", sinonimos:["fidalga","pimenta fidalga"],
   descricao:"Aroma marcante, usada regionalmente em MG/GO.", scoville:"média"},
 "Habanero-Pepper": {nome:"Habanero-Pepper", sinonimos:["habanero","pimenta habanero"],
   descricao:"Muito picante, perfume frutado/cítrico. Use com parcimônia.", scoville:"100.000–350.000"},
 "Jalapeño-Pepper": {nome:"Jalapeño-Pepper", sinonimos:["jalapeño","jalapeno","jalapenho","pimenta jalapeno","pimenta jalapeño"],
   descricao:"Picância moderada; popular em nachos e picles.", scoville:"2.500–8.000"},
 "ScotchBonnet-Pepper": {nome:"ScotchBonnet-Pepper", sinonimos:["scotch bonnet","scotchbonnet","pimenta scotch bonnet"],
   descricao:"Muito picante, típica do Caribe.", scoville:"100.000–350.000"},
 "Cambuci-Pepper": {nome:"Cambuci-Pepper", sinonimos:["cambuci","pimenta cambuci"],
   descricao:"Formato de sino/chapéu; muito suave; conhecida como chapéu-de-frade.", scoville:"0–1.000"}
};

// menu textual
function menuText(nome){
  const n = nome || "pimenta";
  return [
    `📌 *${n}* — escolha uma opção:`,
    `1️⃣  O que é`,
    `2️⃣  Ardência (SHU)`,
    `3️⃣  Usos/Receitas`,
    `4️⃣  Conservação`,
    `5️⃣  Substituições`,
    `6️⃣  Origem`,
    `7️⃣  Curiosidades/Extras`,
    `8️⃣  Trocar pimenta`,
    ``,
    `💡 Você também pode digitar o *nome/sinônimo* da pimenta (ex.: jalapeno/jalapeño, chilli, biquinho…).`
  ].join("\n");
}

// pega chave pelo nome/sinônimo
function keyFromAny(raw){
  if(!raw||!KB) return null;
  if(KB[raw]) return raw;

  const want = normalize(raw).replace(/-?pepper$/,'').replace(/-/g,' ').trim();

  // nome exato
  for (const k of Object.keys(KB)){
    const n = normalize(KB[k].nome||k).replace(/-?pepper$/,'').replace(/-/g,' ').trim();
    if(n===want) return k;
  }
  // sinônimos exatos: sinonimos / sinônimos / aliases
  for (const k of Object.keys(KB)){
    const synRaw = KB[k].sinonimos || KB[k]['sinônimos'] || KB[k].aliases || [];
    const syns = synRaw.map(s => normalize(s).replace(/-?pepper$/,'').replace(/-/g,' ').trim());
    if (syns.includes(want)) return k;
  }
  // contém/parecido
  for (const k of Object.keys(KB)){
    const n = normalize(KB[k].nome||k).replace(/-?pepper$/,'').replace(/-/g,' ').trim();
    if(n.includes(want) || want.includes(n)) return k;
    const synRaw = KB[k].sinonimos || KB[k]['sinônimos'] || KB[k].aliases || [];
    const syns = synRaw.map(s => normalize(s).replace(/-?pepper$/,'').replace(/-/g,' ').trim());
    if (syns.some(s => s.includes(want) || want.includes(s))) return k;
  }
  return null;
}

async function loadKB(){
  try{
    const r = await fetch("/static/pepper_info.json", {cache:"no-store"});
    if(!r.ok) throw new Error("KB HTTP "+r.status);
    KB = await r.json();
  }catch(e){
    KB = {};
  }
  if(!KB || !Object.keys(KB).length){
    KB = FALLBACK_KB;
    addMsg("⚠️ Não consegui carregar o arquivo local. Usei uma base mínima para continuar.");
  }
  if(KB.meta) delete KB.meta;
}

// mensagens por opção (usa dados do KB quando houver)
function msgByOption(k, opt){
  const d = KB[k] || {};
  const nm = d.nome || k;
  switch(opt){
    case "1": return d.descricao ? `Sobre ${nm}: ${d.descricao}` : `Sobre ${nm}: descrição indisponível.`;
    case "2": return d.scoville ? `Ardência (SHU) de ${nm}: ${d.scoville}.` : `Não tenho registro de SHU para ${nm}.`;
    case "3": return d.usos || d.receitas
      ? `Usos/Receitas:\n${[d.usos, d.receitas].filter(Boolean).join("\n")}`
      : `Sem usos/receitas registrados para ${nm}.`;
    case "4": return d.conservacao || d.conservação || `Sem orientações de conservação registradas para ${nm}.`;
    case "5": return d.substituicoes || d["substituições"] || d.substituicoes_sugeridas || `Sem substituições sugeridas para ${nm}.`;
    case "6": return d.origem || `Origem não registrada para ${nm}.`;
    case "7": return d.curiosidades || d.extras || `Sem curiosidades adicionais registradas para ${nm}.`;
    default:  return menuText(d.nome || "pimenta");
  }
}

function setPepperByKey(k){
  CUR = k;
  const name = (KB[k] && KB[k].nome) ? KB[k].nome : k;
  document.getElementById("title").textContent = "Chat: " + name;
  addMsg(menuText(name));
}

document.getElementById("send").onclick = () => {
  const inp = document.getElementById("inp");
  const txt = (inp.value||"").trim();
  if(!txt) return;
  inp.value = "";
  addMsg(txt, true);

  // se for número 1–8 e já temos pimenta, responde
  if(/^[1-8]$/.test(txt) && CUR){
    const m = msgByOption(CUR, txt);
    addMsg(m);
    return;
  }

  // tentar mudar/definir pimenta
  const k = keyFromAny(txt);
  if(k){
    addMsg("🔄 Trocando para: " + (KB[k].nome || k));
    setPepperByKey(k);
    return;
  }

  // se digitou "8" ou "trocar", mostra menu da atual mesmo
  if(txt === "8" || normalize(txt).startsWith("trocar")){
    addMsg(menuText(KB[CUR]?.nome || "pimenta"));
    return;
  }

  // fallback
  addMsg("Não entendi 🫠. Responda com *1–8* ou digite o *nome/sinônimo* da pimenta.\n\n" + menuText(KB[CUR]?.nome || "pimenta"));
};

(async function(){
  await loadKB();
  // tenta vir da tela anterior
  let k = keyFromAny(pepperFromQuery);
  if(!k){
    // default amigável
    k = keyFromAny("Habanero-Pepper") || Object.keys(KB)[0];
  }
  addMsg("Use os botões ou digite o nome/sinônimo.");
  setPepperByKey(k);
})();
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)
