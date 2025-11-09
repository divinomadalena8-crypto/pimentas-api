# main.py — API + UI (YOLOv8 + Chat estilo WhatsApp)
# - /ui  : tela de identificação (clara, imagens grandes, sem resumo/pronto)
# - /info: chat minimalista com menu (emojis) e sinônimos de pimenta
# - /predict: retorna imagem anotada e lista de caixas
# Obs: deixe o arquivo static/pepper_info.json (modelo no final da resposta)

import os, io, time, threading, base64, requests, uuid
from typing import List
from urllib.parse import urlparse

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

try:
    from ultralytics import YOLO
    from PIL import Image
    import numpy as np
except Exception as e:
    # Render instala em runtime; logs ajudam a depurar
    print("[init] Aviso: libs ainda não disponíveis:", e)

# -------------------------------------------------
# App + pastas estáticas
# -------------------------------------------------
app = FastAPI(title="Pimentas • YOLOv8")

ROOT = os.getcwd()
STATIC_DIR = os.path.join(ROOT, "static")
ANNOT_DIR  = os.path.join(STATIC_DIR, "annotated")
os.makedirs(ANNOT_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# -------------------------------------------------
# Config do modelo
# -------------------------------------------------
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://huggingface.co/bulipucca/pimentas-model/resolve/main/best.onnx"
)
MODEL_PATH = (os.path.basename(urlparse(MODEL_URL).path)
              if MODEL_URL and not MODEL_URL.startswith("COLE_AQUI")
              else "best.pt")

# Presets internos (não exibimos na UI)
PRESETS = {
    "ULTRA":       dict(imgsz=320, conf=0.35, iou=0.50, max_det=10),
    "RAPIDO":      dict(imgsz=384, conf=0.30, iou=0.50, max_det=10),
    "EQUILIBRADO": dict(imgsz=448, conf=0.30, iou=0.50, max_det=12),
    "PRECISO":     dict(imgsz=512, conf=0.40, iou=0.55, max_det=12),
}
CFG = PRESETS.get(os.getenv("PRESET", "ULTRA"), PRESETS["ULTRA"])

RETURN_IMAGE = True

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
REQ_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# Estado
model = None
labels = {}
READY = False
LOAD_ERR = None

# -------------------------------------------------
# Auxiliares
# -------------------------------------------------
def ensure_model_file():
    if os.path.exists(MODEL_PATH):
        return
    if not MODEL_URL or MODEL_URL.startswith("COLE_AQUI"):
        raise RuntimeError("Defina MODEL_URL (link direto .pt ou .onnx).")
    print(f"[init] Baixando modelo: {MODEL_URL}")
    with requests.get(MODEL_URL, headers=REQ_HEADERS, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk: f.write(chunk)
    print("[init] Download concluído:", MODEL_PATH)

def to_b64_png(np_bgr):
    try:
        from PIL import Image
        rgb = np_bgr[:, :, ::-1]
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None

def background_load():
    global model, labels, READY, LOAD_ERR
    try:
        from ultralytics import YOLO
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
        print(f"[init] Modelo pronto em {time.time()-t0:.1f}s")
    except Exception as e:
        LOAD_ERR = str(e)
        READY = False
        print("[init] ERRO ao carregar modelo:", LOAD_ERR)

@app.on_event("startup")
def on_startup():
    threading.Thread(target=background_load, daemon=True).start()

# -------------------------------------------------
# Rotas API
# -------------------------------------------------
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
    t0 = time.time()
    while not READY and time.time() - t0 < 90:
        time.sleep(0.5)
    if not READY:
        return {"ok": False, "warming_up": True}
    from PIL import Image
    _ = model.predict(Image.new("RGB",(64,64),(255,255,255)),
                      imgsz=CFG["imgsz"], conf=CFG["conf"], iou=CFG["iou"],
                      max_det=1, device="cpu", verbose=False)
    return {"ok": True}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not READY:
        return JSONResponse({"ok": False, "warming_up": True, "error": LOAD_ERR}, status_code=503)

    t0 = time.time()
    try:
        from PIL import Image
        import numpy as np

        im_bytes = await file.read()
        image = Image.open(io.BytesIO(im_bytes)).convert("RGB")
        image.thumbnail((1024, 1024))

        res = model.predict(
            image,
            imgsz=CFG["imgsz"], conf=CFG["conf"], iou=CFG["iou"],
            max_det=CFG["max_det"], device="cpu", verbose=False
        )
        r = res[0]
        preds: List[dict] = []

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy().tolist()
            cls  = r.boxes.cls.cpu().numpy().astype(int).tolist()
            conf = r.boxes.conf.cpu().numpy().tolist()
            for (x1,y1,x2,y2), c, cf in zip(xyxy, cls, conf):
                preds.append({
                    "classe": labels.get(int(c), str(int(c))),
                    "conf": round(float(cf), 4),
                    "bbox_xyxy": [round(x1,2), round(y1,2), round(x2,2), round(y2,2)]
                })

            image_b64 = None
            image_url = None
            if RETURN_IMAGE:
                annotated = r.plot()  # numpy BGR
                fname = f"{uuid.uuid4().hex}.png"
                fpath = os.path.join(ANNOT_DIR, fname)
                from PIL import Image as _Image
                _Image.fromarray(annotated[:, :, ::-1]).save(fpath)
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

# -------------------------------------------------
# Páginas
# -------------------------------------------------

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
  :root{ --bg:#f7fafc; --card:#ffffff; --fg:#0f172a; --muted:#475569; --line:#e2e8f0; --accent:#16a34a; --info:#f97316; }
  *{box-sizing:border-box}
  html,body{ margin:0;background:var(--bg);color:var(--fg);font:400 16px/1.45 system-ui,-apple-system,Segoe UI,Roboto }
  .wrap{max-width:980px;margin:auto;padding:20px 14px 72px}
  header{display:flex;align-items:center;gap:10px}
  header h1{font-size:22px;margin:0}
  .card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:0 4px 24px rgba(15,23,42,.06)}
  .btn{appearance:none;border:1px solid var(--line);background:#fff;color:var(--fg);padding:10px 14px;border-radius:12px;cursor:pointer;font-weight:600}
  .btn[disabled]{opacity:.6;cursor:not-allowed}
  .btn.accent{background:var(--accent);border-color:var(--accent);color:#fff}
  .btn.info{background:var(--info);border-color:var(--info);color:#fff}
  .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
  .tip{color:var(--muted);font-size:13px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  @media(max-width:860px){ .grid{grid-template-columns:1fr} }
  .imgwrap{background:#fff;border:1px solid var(--line);border-radius:12px;padding:10px}
  img,video,canvas{max-width:100%;display:block;border-radius:10px}
</style>
</head>
<body>
  <div class="wrap">
    <header>
      <img src="/static/pimenta-logo.png" alt="Logo" width="28" height="28" onerror="this.style.display='none'">
      <h1>Identificação de Pimentas</h1>
    </header>

    <section class="card" style="margin-top:12px">
      <div class="row" style="margin-bottom:8px">
        <button id="btnPick" class="btn">Escolher imagem</button>
        <input id="fileGallery" type="file" accept="image/*" style="display:none"/>
        <input id="fileCamera"  type="file" accept="image/*" capture="environment" style="display:none"/>
        <button id="btnCam" class="btn">Abrir câmera</button>
        <button id="btnShot" class="btn" style="display:none">Capturar</button>
        <button id="btnSend" class="btn accent" disabled>Identificar</button>
        <button id="btnChat" class="btn info" style="display:none">Mais informações</button>
      </div>
      <p class="tip">Comprimimos para ~1024px antes do envio para acelerar.</p>

      <div class="grid" style="margin-top:10px">
        <div class="imgwrap">
          <small class="tip">Original</small>
          <video id="video" autoplay playsinline style="display:none"></video>
          <img id="preview" alt="preview" style="display:none"/>
          <canvas id="canvas" style="display:none"></canvas>
        </div>
        <div class="imgwrap">
          <small class="tip">Resultado</small>
          <img id="annotated" alt="resultado"/>
        </div>
      </div>
    </section>
  </div>

<script>
const API = window.location.origin;
let currentFile = null;
let stream = null;
let lastClass = null;

function sleep(ms){ return new Promise(r=>setTimeout(r,ms)); }

async function compressImage(file, maxSide=1024, quality=0.8){
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
  document.getElementById('video').style.display = "none";
  document.getElementById('btnSend').disabled = false;
  document.getElementById('btnChat').style.display = "none";
  lastClass = null;
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
  }catch(e){
    inputCamera.value = "";
    inputCamera.click(); // fallback: abre seletor que chama câmera nativa em WebView
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
  },"image/jpeg",0.92);
};

document.getElementById('btnSend').onclick = async () => {
  if(!currentFile) return;
  document.getElementById('btnSend').disabled = true;
  try{
    const fd=new FormData(); fd.append("file", currentFile, currentFile.name||"image.jpg");
    const r=await fetch(API + "/predict", {method:"POST", body:fd});
    const d=await r.json();

    if(d.ok===false && d.warming_up){ alert("Aquecendo o modelo… tente novamente."); return; }
    if(d.ok===false){ alert("Erro: "+(d.error||"desconhecido")); return; }

    if(d.image_b64){ document.getElementById('annotated').src = d.image_b64; }
    else if(d.image_url){ const url = d.image_url.startsWith("http")? d.image_url : (API + d.image_url); document.getElementById('annotated').src = url; }

    const chatBtn = document.getElementById('btnChat');
    if(d.top_pred && d.top_pred.classe){
      lastClass = d.top_pred.classe;
      chatBtn.style.display = "inline-block";
      chatBtn.onclick = () => { location.href = "/info?pepper=" + encodeURIComponent(lastClass); };
    }else{
      chatBtn.style.display = "none";
      lastClass = null;
      alert("Nenhuma pimenta detectada.");
    }
  }catch(e){
    alert("Falha ao chamar a API.");
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
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"/>
<title>Chat de Pimentas</title>
<link rel="icon" href="/static/pimenta-logo.png" type="image/png" sizes="any">
<style>
  :root{ --bg:#eae6df; --bubble:#fff; --me:#dcf8c6; --text:#111; --muted:#6b7280; --top:#075E54; }
  *{box-sizing:border-box}
  html,body{margin:0;background:var(--bg);font:400 16px/1.45 system-ui,-apple-system,Segoe UI,Roboto;color:var(--text)}
  .app{max-width:740px;margin:auto;height:100dvh;display:flex;flex-direction:column}
  .top{padding:10px 14px;background:var(--top);color:#fff;display:flex;gap:10px;align-items:center}
  .title{font-weight:600}
  .msgs{flex:1;overflow:auto;padding:14px 12px 8px}
  .row{display:flex;margin:6px 0}
  .me{justify-content:flex-end}
  .bub{background:var(--bubble);padding:10px 12px;border-radius:12px;max-width:80%;box-shadow:0 1px 0 rgba(0,0,0,.05)}
  .bub.me{background:var(--me)}
  .input{display:flex;gap:8px;padding:10px;background:#f0f0f0}
  input[type=text]{flex:1;border-radius:18px;border:0;padding:10px 14px;font:inherit}
  button{border:0;border-radius:18px;background:#25D366;color:#fff;font-weight:600;padding:10px 14px}
  .menu{white-space:pre-wrap}
</style>
</head>
<body>
<div class="app">
  <div class="top">
    <div class="title">Chat de Pimentas</div>
    <div id="pep" style="margin-left:auto;font-size:14px;opacity:.9"></div>
    <button onclick="location.href='/ui'" style="margin-left:8px;border:0;border-radius:16px;padding:6px 10px;background:#128C7E;color:#fff">← Voltar</button>
  </div>

  <div id="msgs" class="msgs"></div>

  <div class="input">
    <input id="txt" type="text" placeholder="Digite a opção (1–8) ou o nome da pimenta..."/>
    <button id="send">Enviar</button>
  </div>
</div>

<script>
const qs = new URLSearchParams(location.search);
let currentPepper = (qs.get("pepper")||"").trim(); // já vem da detecção
let KB = null;
let waitingPepperChoice = false;

function $(id){ return document.getElementById(id); }
function addMsg(text, me=false){
  const line = document.createElement("div"); line.className = "row"+(me?" me":"");
  const b = document.createElement("div"); b.className = "bub"+(me?" me":"");
  b.innerHTML = String(text||"").replace(/\n/g,"<br>");
  line.appendChild(b); $("msgs").appendChild(line);
  $("msgs").scrollTop = $("msgs").scrollHeight;
}
function menuText(){
  const p = KB[currentPepper]?.nome || currentPepper || "pimenta";
  return (
`📌 *${p}* — escolha uma opção:
1️⃣  O que é
2️⃣  Ardência (SHU)
3️⃣  Usos/Receitas
4️⃣  Conservação
5️⃣  Substituições
6️⃣  Origem
7️⃣  Curiosidades/Extras
8️⃣  Trocar pimenta

💡 Você também pode digitar o *nome/sinônimo* da pimenta (ex.: jalapeno, chilli, biquinho...).`
  );
}
function listPeppers(){
  const keys = Object.keys(KB);
  return "🧭 *Trocar pimenta*\n" + keys.map((k,i)=>`${i+1}. ${KB[k].nome}`).join("\n") + "\n\nResponda com o número.";
}
function setHeader(){ $("pep").textContent = KB[currentPepper]?.nome || currentPepper || ""; }
function normalize(s){
  return (s||"").toLowerCase()
    .normalize('NFD').replace(/[\u0300-\u036f]/g,'')
    .replace(/[^a-z0-9\s-]/g,' ').trim();
}
function pickByName(name){
  const want = normalize(name).replace(/-?pepper$/,'').trim();
  const keys = Object.keys(KB);
  for(const k of keys){
    const nk = normalize(KB[k].nome).replace(/-?pepper$/,'').trim();
    if(nk===want) return k;
  }
  for(const k of keys){
    const syns = (KB[k].sinonimos||[]).map(normalize);
    if(syns.includes(want)) return k;
    if(want && syns.some(s => want.includes(s) || s.includes(want))) return k;
  }
  for(const k of keys){
    const nk = normalize(KB[k].nome).replace(/-?pepper$/,'').trim();
    if(want && (nk.includes(want) || want.includes(nk))) return k;
  }
  return null;
}
async function loadKB(){
  try{ const r = await fetch("/static/pepper_info.json", {cache:"no-store"}); KB = await r.json(); }
  catch{ KB = {}; }
  if(KB.meta) delete KB.meta;
}
function answer(opt){
  const d = KB[currentPepper];
  if(!d) return "Não encontrei dados desta pimenta. Digite 8 para trocar.";
  switch(opt){
    case "1": return d.descricao || "Sem descrição.";
    case "2": return d.scoville || "Sem SHU informado.";
    case "3": {
      const a = d.usos?("Usos: "+d.usos+"\n"): "";
      const b = d.receitas?("Receitas: "+d.receitas):"";
      return (a+b).trim() || "Sem dados.";
    }
    case "4": return d.conservacao || "Sem dados de conservação.";
    case "5": return d.substituicoes || "Sem substituições registradas.";
    case "6": return d.origem || "Sem dados de origem.";
    case "7": {
      const extras = [];
      if(d.perfil_sabor) extras.push("Perfil: "+d.perfil_sabor);
      if(d.aromas) extras.push("Aromas: "+d.aromas);
      if(d.textura) extras.push("Textura: "+d.textura);
      if(d.formas_preparo) extras.push("Preparo: "+d.formas_preparo);
      if(d.harmonizacao) extras.push("Harmonização: "+d.harmonizacao);
      if(d.nutrientes) extras.push("Nutrientes: "+d.nutrientes);
      if(d.seguranca) extras.push("Segurança: "+d.seguranca);
      if(d.curiosidades) extras.push("Curiosidades: "+d.curiosidades);
      return extras.join("<br>") || "Sem extras cadastrados.";
    }
    default:  return "Opção inválida.";
  }
}

$("send").onclick = async ()=>{
  const raw = $("txt").value.trim(); if(!raw) return;
  $("txt").value = "";
  addMsg(raw, true);

  if(waitingPepperChoice){
    const idx = Number(raw);
    const keys = Object.keys(KB);
    if(Number.isInteger(idx) && idx>=1 && idx<=keys.length){
      currentPepper = keys[idx-1];
      waitingPepperChoice = false;
      setHeader();
      addMsg(`✅ Agora falando sobre *${KB[currentPepper].nome}*.\n\n${menuText()}`);
    }else{
      addMsg("❌ Número inválido.\n\n"+listPeppers());
    }
    return;
  }

  const maybe = pickByName(raw);
  if(maybe){
    currentPepper = maybe;
    setHeader();
    addMsg(`🔁 Trocado para *${KB[currentPepper].nome}*.\n\n${menuText()}`);
    return;
  }

  if(/^\d+$/.test(raw)){
    if(raw==="8"){
      waitingPepperChoice = True = true; // hack to keep lints quiet in static context
      waitingPepperChoice = true;
      addMsg(listPeppers());
      return;
    }
    const out = answer(raw);
    if(out==="Opção inválida."){
      addMsg("❌ Opção inválida.\n\n"+menuText());
    }else{
      addMsg(out);
      addMsg("—\n"+menuText());
    }
    return;
  }

  addMsg("Não entendi 😅. Responda com *1–8* ou digite o *nome da pimenta*.\n\n"+menuText());
};

(async function init(){
  await loadKB();
  if(!currentPepper || !KB[currentPepper]) currentPepper = Object.keys(KB)[0];
  setHeader();
  addMsg("Olá! 👋 Bem-vindo ao *Chat de Pimentas*.");
  addMsg(menuText());
})();
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)
