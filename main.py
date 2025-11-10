# main.py — PWA + Splash + Detector com imagem anotada + Chat tipo WhatsApp
# --------------------------------------------------------------------------------
# Requisitos:
# fastapi==0.115.0
# uvicorn==0.30.0
# python-multipart==0.0.9
# pillow==10.4.0
# numpy==1.26.4
# ultralytics==8.3.34
# requests==2.32.3
#
# Variáveis (Render > Environment):
# USE_MODEL=1
# MODEL_URL=https://huggingface.co/.../best.onnx  (ou .pt)
# (opcional) PORT=10000
# --------------------------------------------------------------------------------

import os, io, json, uuid, time, threading, base64
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response, PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# --------------------------------- App & static ---------------------------------
app = FastAPI(title="Pimentas PWA")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

ROOT = Path(os.getcwd())
STATIC = ROOT / "static"
ANNOT = STATIC / "annotated"
(STATIC / "icons").mkdir(parents=True, exist_ok=True)
ANNOT.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

# --------------------------------- Modelo (opcional) ----------------------------
USE_MODEL = os.getenv("USE_MODEL", "1") == "1"
READY = not USE_MODEL
LOAD_ERR: Optional[str] = None
model = None
labels = {}

MODEL_URL = os.getenv("MODEL_URL", "").strip()
MODEL_PATH = Path(os.path.basename(urlparse(MODEL_URL).path)) if MODEL_URL else Path("best.pt")

def ensure_model_file():
    if MODEL_PATH.exists():
        return
    if not MODEL_URL:
        raise RuntimeError("Defina MODEL_URL (.pt/.onnx) nas variáveis de ambiente.")
    import requests
    with requests.get(MODEL_URL, stream=True, timeout=600) as r:
        r.raise_for_status()
        with MODEL_PATH.open("wb") as f:
            for chunk in r.iter_content(8192):
                if chunk: f.write(chunk)

def _b64_png(np_bgr) -> Optional[str]:
    try:
        from PIL import Image
        import numpy as np  # noqa
        rgb = np_bgr[:, :, ::-1]
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None

def background_load():
    global model, labels, READY, LOAD_ERR
    try:
        ensure_model_file()
        from ultralytics import YOLO
        m = YOLO(str(MODEL_PATH))
        try:
            m.fuse()
        except Exception:
            pass
        model = m
        labels = m.names or {}
        READY = True
        print("[init] Modelo carregado com sucesso.")
    except Exception as e:
        LOAD_ERR = str(e)
        READY = False
        print("[init] ERRO:", LOAD_ERR)

if USE_MODEL:
    threading.Thread(target=background_load, daemon=True).start()

# --------------------------------- KB (chat) ------------------------------------
KB_PATH = STATIC / "pepper_info.json"

FALLBACK_KB = {
    "Jalapeño": {
        "sinonimos": ["jalapeno", "jalapeños", "jalapenho", "jalapénio", "chili", "chilli"],
        "oque": "Pimenta de picância moderada, versátil e muito popular.",
        "shu": "≈ 2.500–8.000 SHU",
        "usos": "Nachos, tacos, hambúrgueres, picles e recheios (poppers).",
        "receitas": "Jalapeño em conserva; poppers recheados; chipotle caseiro.",
        "conservacao": "Geladeira (fresca), conserva em vinagre, defumada vira chipotle.",
        "substituicoes": "Serrano (mais picante) ou poblano (menos picante).",
        "origem": "México.",
        "extras": "Verde é imaturo; vermelho é maduro. Defumada vira chipotle."
    },
    "Biquinho": {
        "sinonimos": ["biquinho", "chupetinha", "sweet drop"],
        "oque": "Muito suave, adocicada, aroma característico.",
        "shu": "≈ 0–500 SHU",
        "usos": "Conservas, aperitivos, saladas.",
        "receitas": "Biquinho em conserva; geleia.",
        "conservacao": "Conserva em vinagre ou refrigerada.",
        "substituicoes": "Mini pimentas doces.",
        "origem": "Brasil.",
        "extras": "Muito usada em petiscos e conservas."
    },
    "Habanero": {
        "sinonimos": ["habanero", "habaneros"],
        "oque": "Muito picante, perfume frutado/cítrico.",
        "shu": "≈ 100.000–350.000 SHU",
        "usos": "Molhos encorpados, marinadas, salsas intensas.",
        "receitas": "Salsa de habanero; molho de habanero com manga.",
        "conservacao": "Secar, congelar, fermentar ou fazer molhos.",
        "substituicoes": "Scotch Bonnet (perfil semelhante).",
        "origem": "Caribe/México (Yucatán).",
        "extras": "Use luvas; manuseie com cuidado."
    }
}

def load_kb():
    if KB_PATH.exists():
        try:
            return json.loads(KB_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return FALLBACK_KB

def build_map(kb: dict) -> dict:
    m = {}
    for canon, data in kb.items():
        if canon == "__map__": continue
        m[canon.lower()] = canon
        for s in data.get("sinonimos", []) or []:
            m[str(s).lower()] = canon
    return m

# --------------------------------- PWA files ------------------------------------
MANIFEST_JSON = """
{
  "name": "Identificação de Pimentas",
  "short_name": "Pimentas",
  "start_url": "/ui",
  "display": "standalone",
  "background_color": "#ffffff",
  "theme_color": "#16a34a",
  "icons": [
    { "src": "/static/icons/pimenta-192.png", "sizes": "192x192", "type": "image/png" },
    { "src": "/static/icons/pimenta-512.png", "sizes": "512x512", "type": "image/png" }
  ]
}
""".strip()

SW_JS = """
const CACHE = 'pimentas-v2';
const APP_SHELL = ['/ui','/chat','/manifest.webmanifest','/static/splash.png','/static/pimenta-logo.png'];
self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(APP_SHELL)));
  self.skipWaiting();
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k=>k!==CACHE).map(k=>caches.delete(k)))));
  self.clients.claim();
});
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET') return;
  if (url.pathname.startsWith('/predict') || url.pathname.startsWith('/warmup')) {
    e.respondWith(fetch(e.request).catch(()=>new Response('Offline',{status:503})));
    return;
  }
  e.respondWith(caches.match(e.request).then(r => r || fetch(e.request).then(resp => {
    if (resp && resp.status===200 && url.origin===location.origin) {
      const clone = resp.clone();
      caches.open(CACHE).then(c => c.put(e.request, clone));
    }
    return resp;
  })));
});
""".strip()

# --------------------------------- UI (/ui) -------------------------------------
UI_HTML = r"""
<!doctype html><html lang="pt-BR"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Identificação de Pimentas</title>
<link rel="manifest" href="/manifest.webmanifest">
<meta name="theme-color" content="#16a34a">
<link rel="apple-touch-icon" href="/static/icons/pimenta-192.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<style>
:root{--bg:#f7fafc;--card:#fff;--fg:#0f172a;--muted:#475569;--line:#e2e8f0;--accent:#16a34a}
*{box-sizing:border-box}html,body{margin:0;background:var(--bg);color:var(--fg);font:400 16px/1.45 system-ui,-apple-system,Segoe UI,Roboto}
.wrap{max-width:980px;margin:auto;padding:20px 14px 72px}
header{display:flex;align-items:center;gap:10px}
h1{font-size:22px;margin:0}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:0 4px 24px rgba(15,23,42,.06)}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.btn{appearance:none;border:1px solid var(--line);background:#fff;color:var(--fg);padding:10px 14px;border-radius:12px;cursor:pointer;font-weight:600}
.btn.accent{background:var(--accent);border-color:var(--accent);color:#fff}
.imgwrap{background:#fff;border:1px solid var(--line);border-radius:12px;padding:8px}
img,video,canvas{max-width:100%;display:block;border-radius:10px}
footer{position:fixed;left:0;right:0;bottom:0;padding:10px 14px;background:#ffffffd9;border-top:1px solid var(--line);color:var(--muted);font-size:12px;text-align:center;backdrop-filter:saturate(140%) blur(6px)}
/* SPLASH */
#splash{position:fixed;inset:0;z-index:9999;background:#fff url('/static/splash.png') no-repeat center 34%;background-size:min(72vmin,520px);display:flex;align-items:flex-end;justify-content:center;padding:32px;transition:opacity .35s}
#splash.hide{opacity:0;pointer-events:none}
#splash .bar{width:min(520px,80vw);height:14px;border-radius:999px;background:#e6f4ea;border:1px solid #cce9d6;overflow:hidden}
#splash .bar>i{display:block;width:35%;height:100%;background:#16a34a;animation:load 1.6s infinite}
@keyframes load{0%{transform:translateX(-100%)}50%{transform:translateX(30%)}100%{transform:translateX(120%)}}
</style>
<script>if('serviceWorker'in navigator){navigator.serviceWorker.register('/sw.js').catch(()=>{});}</script>
</head><body>
<div id="splash"><div class="bar"><i></i></div></div>
<div class="wrap">
  <header>
    <img src="/static/pimenta-logo.png" alt="logo" width="28" height="28" onerror="this.style.display='none'">
    <h1>Identificação de Pimentas</h1>
    <!-- (removido "Pronto"/status) -->
  </header>

  <section class="card" style="margin-top:16px">
    <div class="row">
      <button id="btnPick" class="btn">Escolher imagem</button>
      <input id="fileGallery" type="file" accept="image/*" style="display:none"/>
      <input id="fileCamera"  type="file" accept="image/*" capture="environment" style="display:none"/>
      <button id="btnCam" class="btn">Abrir câmera</button>
      <button id="btnSend" class="btn accent" disabled>Identificar</button>
      <a id="btnInfo" class="btn" style="display:none;background:#0ea5e9;border-color:#0ea5e9;color:#fff">Mais informações ↗</a>
    </div>

    <div class="row" style="margin-top:10px">
      <div class="imgwrap" style="flex:1">
        <small style="color:#475569">Original</small>
        <img id="preview" alt="preview" style="display:none"/>
        <canvas id="canvas" style="display:none"></canvas>
      </div>
      <div class="imgwrap" style="flex:1">
        <small style="color:#475569">Resultado</small>
        <img id="annotated" alt="resultado"/>
      </div>
    </div>

    <div id="resumo" class="tip" style="color:#475569; margin-top:8px"></div>
  </section>
</div>
<footer>Desenvolvido por <strong>Madalena de Oliveira Barbosa</strong>, 2025</footer>

<script>
const API = window.location.origin;
let currentFile=null, lastClass=null;
const preview=document.getElementById('preview');

function hideSplash(){ const s=document.getElementById('splash'); if(!s) return; s.classList.add('hide'); setTimeout(()=>s.remove(), 400); }
function sleep(ms){ return new Promise(r=>setTimeout(r,ms)); }

async function waitReady(){
  try{
    const r = await fetch(API + "/", {cache:"no-store"});
    const d = await r.json();
    if(d.ready){ hideSplash(); document.getElementById('btnSend').disabled=!currentFile; return; }
  }catch(e){}
  await sleep(800); waitReady();
}

async function compressImage(file, maxSide=1024, quality=.8){
  return new Promise((resolve,reject)=>{
    const img=new Image();
    img.onload=()=>{
      const scale=Math.min(1, maxSide/Math.max(img.width,img.height));
      const w=Math.round(img.width*scale), h=Math.round(img.height*scale);
      const cv=document.getElementById('canvas'), ctx=cv.getContext('2d');
      cv.width=w; cv.height=h; ctx.drawImage(img,0,0,w,h);
      cv.toBlob(b=>{ if(!b) return reject(Error("compress")); resolve(new File([b], file.name||"photo.jpg",{type:"image/jpeg"})); },"image/jpeg",quality);
    };
    img.onerror=reject; img.src=URL.createObjectURL(file);
  });
}

const inputGallery=document.getElementById('fileGallery');
const inputCamera =document.getElementById('fileCamera');
document.getElementById('btnPick').onclick=()=>{inputGallery.value="";inputGallery.click();}
document.getElementById('btnCam' ).onclick=()=>{inputCamera.value=""; inputCamera.click();}
inputGallery.onchange=()=>useLocalFile(inputGallery.files?.[0]);
inputCamera .onchange=()=>useLocalFile(inputCamera .files?.[0]);

async function useLocalFile(f){
  if(!f) return;
  currentFile = await compressImage(f);
  preview.src = URL.createObjectURL(currentFile);
  preview.style.display = "block";
  document.getElementById('btnSend').disabled = false;
  document.getElementById('resumo').textContent = "Imagem pronta para envio.";
}

document.getElementById('btnSend').onclick = async () => {
  if(!currentFile) return;
  const btn=document.getElementById('btnSend');
  btn.disabled=true;
  document.getElementById('resumo').textContent="Enviando...";
  try{
    const fd=new FormData(); fd.append("file", currentFile, currentFile.name||"image.jpg");
    const r=await fetch(API + "/predict", {method:"POST", body:fd});
    const d=await r.json();

    if(d.ok===false){ document.getElementById('resumo').textContent="Erro: " + (d.error||"desconhecido"); return; }

    // mostra imagem ANOTADA (base64 ou URL)
    if(d.image_b64){ document.getElementById('annotated').src = d.image_b64; }
    else if(d.image_url){
      const url = d.image_url.startsWith("http") ? d.image_url : (API + d.image_url);
      document.getElementById('annotated').src = url;
    }

    lastClass = d.top_pred?.classe || null;
    document.getElementById('resumo').textContent = d.top_pred
      ? (`Pimenta: ${d.top_pred.classe} (${Math.round((d.top_pred.conf||0)*100)}%) · Caixas: ${d.num_dets||1}`)
      : "Nenhuma pimenta detectada.";

    const info=document.getElementById('btnInfo');
    if(lastClass){ info.style.display="inline-block"; info.href="/chat?pepper="+encodeURIComponent(lastClass); }
    else{ info.style.display="none"; }
  }catch(e){
    document.getElementById('resumo').textContent="Falha ao enviar.";
  }finally{
    btn.disabled=false;
  }
};

setTimeout(hideSplash, 4000);
waitReady();
</script>
</body></html>
"""

# --------------------------------- CHAT (/chat) ---------------------------------
CHAT_HTML = r"""
<!doctype html><html lang="pt-BR"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"/>
<title>Chat de Pimentas</title>
<link rel="manifest" href="/manifest.webmanifest"><meta name="theme-color" content="#16a34a">
<style>
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:#e5ddd5;font:400 16px/1.45 system-ui,-apple-system,Segoe UI,Roboto}
@supports (height: 100svh){ html,body{height:100svh} }
.app{display:flex;flex-direction:column;height:100%}
.top{background:#075e54;color:#fff;padding:12px 14px;display:flex;align-items:center;gap:10px}
.top a{color:#cce;text-decoration:none}
.title{font-weight:700}
.chat{flex:1;overflow:auto;background:#efeae2;padding:12px}
.bubble{max-width:80%; margin:6px 0; padding:10px 12px; border-radius:12px; box-shadow:0 2px 4px rgba(0,0,0,.05)}
.bot{background:#fff}
.me{background:#dcf8c6; margin-left:auto}
.menu{white-space:pre-wrap}
.inputbar{position:sticky;bottom:0;display:flex;gap:8px;background:#f0f0f0;padding:8px;padding-bottom:calc(8px + env(safe-area-inset-bottom))}
.inputbar input{flex:1;padding:12px 14px;border-radius:20px;border:none;outline:none}
.inputbar button{background:#25d366;color:#fff;border:none;border-radius:20px;padding:0 16px;font-weight:700}
</style>
</head><body>
<div class="app">
  <div class="top">
    <div class="title">Chat: <span id="pepname">—</span></div>
    <span style="margin-left:auto"><a href="/ui">← Voltar</a></span>
  </div>
  <div id="chat" class="chat"><div class="bubble bot" id="welcome">Carregando dados…</div></div>
  <div class="inputbar" id="ibar">
    <input id="msg" placeholder="Digite 1–8 ou o nome/sinônimo…">
    <button id="send">Enviar</button>
  </div>
</div>
<script>
const params=new URLSearchParams(location.search);
let current = params.get('pepper') || '';
let KB = {};
const chat = document.getElementById('chat');
const pepname = document.getElementById('pepname');

function bubble(text, who="bot"){
  const div=document.createElement('div');
  div.className='bubble ' + (who==="me"?"me":"bot");
  div.textContent=text;
  chat.appendChild(div); chat.scrollTop = chat.scrollHeight;
}
function menu(canonName){
  const m =
`📌 ${canonName} — escolha:
1️⃣ O que é
2️⃣ Ardência (SHU)
3️⃣ Usos/Receitas
4️⃣ Conservação
5️⃣ Substituições
6️⃣ Origem
7️⃣ Curiosidades
8️⃣ Trocar pimenta

💡 Também aceito nome/sinônimo (ex.: jalapeno/jalapeño, chilli, biquinho).`;
  const div=document.createElement('div');
  div.className='bubble bot menu'; div.innerText = m;
  chat.appendChild(div); chat.scrollTop = chat.scrollHeight;
}
function canonical(name){
  if(!name) return null;
  const key=String(name).trim().toLowerCase();
  if(KB.__map && KB.__map[key]) return KB.__map[key];
  return null;
}
function showPepper(name){
  const c = canonical(name) || canonical(current);
  if(!c){ bubble("Não encontrei essa pimenta. Digite um nome válido ou use 8️⃣ para trocar."); return; }
  current = c; pepname.textContent = c; chat.innerHTML = '';
  bubble("Use as opções (1–8) ou pergunte pelo nome/sinônimo.");
  menu(c);
}
function answer(opt){
  const c = canonical(current); if(!c) return;
  const d = KB[c] || {};
  const map = {"1":"oque","2":"shu","3":"usos","4":"conservacao","5":"substituicoes","6":"origem","7":"extras"};
  if(opt==="8"){ chat.innerHTML=""; bubble("Digite o nome/sinônimo da pimenta (ex.: jalapeño, biquinho, habanero…)."); return; }
  const k = map[opt];
  bubble(opt, "me");
  if(!k){ bubble("Opção inválida. Use 1–8."); return; }
  bubble(d[k] || "Ainda não tenho dados desta pimenta.");
  menu(c);
}
async function loadKB(){
  const r = await fetch('/kb'); const data = await r.json();
  const MAP = {};
  Object.keys(data).forEach(canon=>{
    if(canon==='__map__') return;
    MAP[canon.toLowerCase()] = canon;
    (data[canon].sinonimos||[]).forEach(s=> MAP[String(s||'').toLowerCase()] = canon);
  });
  data.__map = MAP; return data;
}
// manter botão visível com teclado aberto (Android/iOS)
if (window.visualViewport){
  const iv=window.visualViewport, bar=document.getElementById('ibar');
  function adjust(){ bar.style.paddingBottom = `calc(8px + env(safe-area-inset-bottom) + ${Math.max(0, (iv.height + iv.offsetTop) - window.innerHeight)}px)`; }
  iv.addEventListener('resize', adjust); iv.addEventListener('scroll', adjust);
}
document.getElementById('send').onclick = ()=>{
  const input = document.getElementById('msg'); const t = (input.value||'').trim(); if(!t) return;
  if(/^[1-8]$/.test(t)){ answer(t); input.value=''; return; }
  const c = canonical(t); bubble(t, "me");
  if(c){ current=c; showPepper(c); input.value=''; return; }
  bubble("Não entendi. Responda com 1–8 ou digite o nome/sinônimo da pimenta.");
  menu(canonical(current)||"Pimenta"); input.value='';
};
(async()=>{
  KB = await loadKB();
  const first = canonical(current) || Object.keys(KB).find(k => k!=="__map");
  current = first; document.getElementById('welcome').remove(); showPepper(first);
})();
</script>
</body></html>
"""

# --------------------------------- Rotas básicas --------------------------------
@app.get("/", response_class=JSONResponse)
def root():
    return {"status": "ok", "ready": READY, "error": LOAD_ERR}

@app.get("/ui", response_class=HTMLResponse)
def ui():
    return HTMLResponse(UI_HTML)

@app.get("/chat", response_class=HTMLResponse)
def chat(pepper: Optional[str] = Query(None)):
    return HTMLResponse(CHAT_HTML)

@app.get("/kb", response_class=JSONResponse)
def kb():
    data = load_kb()
    data["__map__"] = build_map(data)
    return data

@app.get("/manifest.webmanifest", response_class=Response)
def manifest():
    return Response(MANIFEST_JSON, media_type="application/manifest+json")

@app.get("/sw.js", response_class=PlainTextResponse)
def sw():
    return PlainTextResponse(SW_JS, media_type="application/javascript")

# --------------------------------- Predição (com ANOTADA!) ----------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Retorna:
      ok: bool
      num_dets: int
      top_pred: {classe, conf}
      image_b64: dataURL (fallback) ou None
      image_url: /static/annotated/<uuid>.png (preferível)
    """
    if not file or not file.filename:
        return {"ok": False, "error": "arquivo não enviado"}
    data = await file.read()
    if not data:
        return {"ok": False, "error": "arquivo vazio"}

    if not USE_MODEL or not READY or model is None:
        # Stub: sem modelo, só ecoa a imagem (sem anotação)
        return {"ok": True, "num_dets": 0, "top_pred": None, "image_b64": None, "image_url": None}

    from PIL import Image
    import numpy as np

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.thumbnail((1024, 1024))  # acelera CPU mantendo aspecto

        # Config "equilibrada" por padrão
        res = model.predict(img, imgsz=448, conf=0.30, iou=0.50, max_det=8, device="cpu", verbose=False)
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

            # imagem anotada
            annotated = r.plot()  # np.ndarray (BGR)
            fname = f"{uuid.uuid4().hex}.png"
            fpath = ANNOT / fname
            from PIL import Image as PILImage
            PILImage.fromarray(annotated[:, :, ::-1]).save(fpath)  # BGR->RGB

            top = max(preds, key=lambda p: p["conf"]) if preds else None
            return {
                "ok": True,
                "num_dets": len(preds),
                "top_pred": top,
                "preds": preds,
                "image_b64": _b64_png(annotated),
                "image_url": f"/static/annotated/{fname}"
            }

        # sem detecções
        return {"ok": True, "num_dets": 0, "top_pred": None, "preds": [], "image_b64": None, "image_url": None}

    except Exception as e:
        return {"ok": False, "error": str(e)}

# --------------------------------- Warmup (opcional) ----------------------------
@app.get("/warmup")
def warmup():
    t0 = time.time()
    while USE_MODEL and not READY and time.time() - t0 < 60:
        time.sleep(0.5)
    return {"ok": READY, "error": LOAD_ERR}

# --------------------------------- Run local ------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
