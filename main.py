# main.py — Pimentas App (YOLOv8 + UI + Chat) + PWA (manifest + sw + splash 1.8s)
import os, io, time, threading, base64, uuid, json, requests
from typing import List, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, UploadFile, File, Response, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from PIL import Image
import numpy as np

# ===================== CONFIG =====================
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

# Modelo: link direto (.pt ou .onnx). Mantém seu caminho atual.
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://huggingface.co/bulipucca/pimentas-model/resolve/main/best.onnx"
)
MODEL_PATH = (
    os.path.basename(urlparse(MODEL_URL).path)
    if MODEL_URL and not MODEL_URL.startswith("COLE_AQUI")
    else "best.pt"
)

# Presets (mantém baixos para rodar em CPU)
PRESET = os.getenv("PRESET", "ULTRA")
PRESETS = {
    "ULTRA":       dict(imgsz=320, conf=0.35, iou=0.50, max_det=12),
    "RAPIDO":      dict(imgsz=384, conf=0.30, iou=0.50, max_det=12),
    "EQUILIBRADO": dict(imgsz=448, conf=0.30, iou=0.50, max_det=12),
    "PRECISO":     dict(imgsz=512, conf=0.40, iou=0.55, max_det=12),
    "MAX_RECALL":  dict(imgsz=640, conf=0.15, iou=0.45, max_det=16),
}
CFG = PRESETS.get(PRESET, PRESETS["ULTRA"])

RETURN_IMAGE = True

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
REQ_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

app = FastAPI(title="Pimentas App")

# Static
STATIC_DIR = os.path.join(os.getcwd(), "static")
ANNOT_DIR  = os.path.join(STATIC_DIR, "annotated")
os.makedirs(ANNOT_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ===================== ESTADO GLOBAL =====================
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

model: Optional["YOLO"] = None
labels = {}
READY = False
LOAD_ERR = None

# ===================== HELPERS =====================
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

def to_b64_png(np_bgr: np.ndarray) -> Optional[str]:
    try:
        rgb = np_bgr[:, :, ::-1]
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None

def background_load():
    """Carrega YOLO e marca READY (sem inferência aqui)."""
    global model, labels, READY, LOAD_ERR
    try:
        t0 = time.time()
        ensure_model_file()
        if YOLO is None:
            raise RuntimeError("Pacote ultralytics não disponível.")
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

# ===================== HEALTH =====================
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

# ===================== PREDICT =====================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not READY:
        return JSONResponse({"ok": False, "warming_up": True, "error": LOAD_ERR}, status_code=503)

    t0 = time.time()
    try:
        im_bytes = await file.read()
        image = Image.open(io.BytesIO(im_bytes)).convert("RGB")

        # acelerar em CPU, mantendo proporção
        image.thumbnail((1024, 1024))

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
        image_b64 = None
        image_url = None

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

            if RETURN_IMAGE:
                annotated = r.plot()  # np.ndarray (BGR)
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
    t0 = time.time()
    while not READY and time.time() - t0 < 90:
        time.sleep(0.5)
    if not READY:
        return {"ok": False, "warming_up": True}
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    _ = model.predict(img, imgsz=CFG["imgsz"], conf=CFG["conf"], iou=CFG["iou"],
                      max_det=1, device="cpu", verbose=False)
    return {"ok": True}

# ===================== CHAT (usa JSON local) =====================
@app.get("/kb.json")
def kb_json():
    p = os.path.join(STATIC_DIR, "pepper_info.json")
    if not os.path.exists(p):
        return JSONResponse({"detail": "pepper_info.json não encontrado em /static"}, status_code=404)
    return FileResponse(p, media_type="application/json")

# ===================== UI: /ui (principal) =====================
@app.get("/ui")
def ui():
    html = r"""
<!doctype html>
<html lang="pt-br">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"/>
  <title>Identificação de Pimentas</title>
  <link rel="icon" href="/static/pimenta-logo.png" type="image/png" sizes="any">
  <style>
    :root{ --bg:#f7fafc; --card:#ffffff; --fg:#0f172a; --muted:#475569; --line:#e2e8f0; --accent:#16a34a; }
    *{box-sizing:border-box}
    html,body{ margin:0;background:var(--bg);color:var(--fg);font:400 16px/1.45 system-ui,-apple-system,Segoe UI,Roboto }
    .wrap{max-width:980px;margin:auto;padding:20px 14px 20px}
    header{display:flex;align-items:center;gap:10px}
    header h1{font-size:22px;margin:0}
    .card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:0 4px 24px rgba(15,23,42,.06)}
    .btn{appearance:none;border:1px solid var(--line);background:#fff;color:var(--fg);padding:10px 14px;border-radius:12px;cursor:pointer;font-weight:600}
    .btn[disabled]{opacity:.6;cursor:not-allowed}
    .btn.accent{background:var(--accent);border-color:var(--accent);color:#fff}
    .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
    .tip{color:var(--muted);font-size:13px}
    .imgwrap{background:#fff;border:1px solid var(--line);border-radius:12px;padding:8px}
    img,video,canvas{max-width:100%;display:block;border-radius:10px}
    footer{color:#64748b;font-size:12px;text-align:center;margin-top:12px}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <img src="/static/pimenta-logo.png" alt="Logo" width="28" height="28" onerror="this.style.display='none'">
      <h1>Identificação de Pimentas</h1>
    </header>

    <section class="card" style="margin-top:12px">
      <div class="row">
        <button id="btnPick" class="btn">Escolher imagem</button>
        <input id="fileGallery" type="file" accept="image/*" style="display:none"/>
        <input id="fileCamera"  type="file" accept="image/*" capture="environment" style="display:none"/>

        <button id="btnCam" class="btn">Abrir câmera</button>
        <button id="btnShot" class="btn" style="display:none">Capturar</button>

        <button id="btnSend" class="btn accent" disabled>Identificar</button>
        <button id="btnChat" class="btn" style="display:none">Mais informações</button>
      </div>
      <p class="tip" style="margin-top:6px">A imagem é comprimida (~1024px) antes do envio para acelerar.</p>

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
    </section>

    <footer>Desenvolvido por <strong>Madalena de Oliveira Barbosa</strong>, 2025</footer>
  </div>

<script>
const API = window.location.origin;
let currentFile = null;
let stream = null;
let lastClass = null;

function setStatus(_){ /* removido o "Pronto" */ }

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

const inputGallery = document.getElementById('fileGallery');
const inputCamera  = document.getElementById('fileCamera');

document.getElementById('btnPick').onclick = () => { inputGallery.value=""; inputGallery.click(); };
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
    inputCamera.click(); // fallback WebView
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
    if(d.ok===false){ alert("Erro: " + (d.error||"desconhecido")); return; }

    if(d.image_b64){ document.getElementById('annotated').src = d.image_b64; }
    else if(d.image_url){
      const url = d.image_url.startsWith("http")? d.image_url : (API + d.image_url);
      document.getElementById('annotated').src = url;
    }

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
    return HTMLResponse(apply_pwa(html))

# ===================== UI: /info (chat simples com JSON local) =====================
@app.get("/info")
def info(req: Request):
    html = r"""
<!doctype html>
<html lang="pt-br">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"/>
  <title>Chat: Pimenta</title>
  <link rel="icon" href="/static/pimenta-logo.png" type="image/png" sizes="any">
  <style>
    :root{ --bg:#f7fafc; --card:#ffffff; --fg:#0f172a; --muted:#475569; --line:#e2e8f0; --accent:#16a34a;}
    *{box-sizing:border-box}
    html,body{margin:0;background:var(--bg);color:var(--fg);font:400 16px/1.45 system-ui,-apple-system,Segoe UI,Roboto}
    .wrap{max-width:980px;margin:auto;padding:12px}
    header{display:flex;align-items:center;gap:10px}
    header h1{font-size:18px;margin:0}
    .btn{appearance:none;border:1px solid var(--line);background:#fff;color:var(--fg);padding:8px 12px;border-radius:10px;cursor:pointer;font-weight:600}
    .card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:12px;box-shadow:0 4px 24px rgba(15,23,42,.06)}
    .messages{border:1px solid var(--line);border-radius:12px;padding:10px;background:#fff;height:58vh;min-height:290px;overflow:auto}
    .msg{margin:6px 0;display:flex}
    .msg.me{justify-content:flex-end}
    .bubble{max-width:80%;padding:8px 10px;border-radius:12px;border:1px solid var(--line);white-space:pre-wrap}
    .bubble.me{background:#eef2ff;border-color:#c7d2fe}
    .chips{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}
    .chip{border:1px solid var(--line);border-radius:999px;padding:6px 10px;background:#fff;cursor:pointer;font-size:13px}
    .dock{position:sticky;bottom:0;left:0;right:0;background:#ffffffd9;border-top:1px solid var(--line);padding:10px;border-radius:12px;backdrop-filter:saturate(140%) blur(6px)}
    .row{display:flex;gap:8px;align-items:center}
    input[type=text]{flex:1;border:1px solid var(--line);border-radius:10px;padding:10px 12px;font:inherit}
    .btn.accent{background:var(--accent);border-color:var(--accent);color:#fff}
    a.back{ text-decoration:none; }
    footer{color:#64748b;font-size:12px;text-align:center;margin-top:12px}
  </style>
</head>
<body>
<div class="wrap">
  <header>
    <a class="back" href="/ui"><button class="btn">← Voltar</button></a>
    <h1 id="title">Chat</h1>
  </header>

  <div class="card" style="margin-top:8px">
    <div class="chips">
      <span class="chip" data-q="O que é?">O que é?</span>
      <span class="chip" data-q="Usos e receitas">Usos/receitas</span>
      <span class="chip" data-q="Conservação">Conservação</span>
      <span class="chip" data-q="Substituições">Substituições</span>
      <span class="chip" data-q="Origem">Origem</span>
    </div>
    <div id="messages" class="messages"></div>

    <div class="dock">
      <div class="row">
        <input id="inputMsg" type="text" placeholder="Digite 1–5 (atalhos) ou faça uma pergunta livre..."/>
        <button id="btnSend" class="btn accent">Enviar</button>
      </div>
    </div>
  </div>

  <footer>Desenvolvido por <strong>Madalena de Oliveira Barbosa</strong>, 2025</footer>
</div>

<script>
const qs = new URLSearchParams(location.search);
const API = window.location.origin;
const pepper = qs.get("pepper") || ""; // vem da tela principal

let KB = null;     // JSON completo
let DOC = null;    // doc da pimenta

function el(tag, cls, text){ const e=document.createElement(tag); if(cls) e.className=cls; if(text!=null) e.textContent=text; return e; }
function scrollEnd(){ const box=document.getElementById('messages'); box.scrollTop = box.scrollHeight; }
function putMsg(text, me=false){
  const wrap = el("div","msg"+(me?" me":""));
  const b = el("div","bubble"+(me?" me":""), null);
  b.innerHTML = String(text||"").replace(/\n/g,"<br>");
  if(me) b.classList.add("me");
  wrap.appendChild(b);
  document.getElementById('messages').appendChild(wrap);
  scrollEnd();
}

function norm(s){
  return (s||"")
    .toLowerCase()
    .replace(/-?pepper/g,"")
    .normalize('NFD').replace(/[\u0300-\u036f]/g,'')
    .replace(/[^a-z0-9]/g,'');
}

async function loadKB(){
  try{
    const r = await fetch("/static/pepper_info.json", {cache:"no-store"});
    KB = await r.json();
  }catch(e){ KB = {}; }
}

function pickDoc(name){
  if(!KB || typeof KB !== "object") return null;
  const keys = Object.keys(KB);
  const want = norm(name);
  const idx = {};
  for(const k of keys){ idx[norm(k)] = k; }
  if(idx[want]) return KB[idx[want]];
  for(const k of keys){
    const nk = norm(k);
    if(want && (want.includes(nk) || nk.includes(want))) return KB[k];
  }
  return null;
}

function answerLocal(q){
  if(!DOC){
    return { text:"Ainda não tenho dados desta pimenta.", weak:true };
  }
  const msg = (q||"").toLowerCase();
  const parts = [];
  const has = (k) => DOC[k]!=null && String(DOC[k]).trim()!=="";

  if(/(o que|o que é|\bo que e\b|\bdefini|sobre )/.test(msg)){
    if(has("descricao")) parts.push(String(DOC.descricao));
  }
  if(/uso|receita|culin|molho|chutney|prato|cozinhar/.test(msg)){
    if(has("usos")) parts.push("Usos: "+String(DOC.usos));
    if(has("receitas")) parts.push("Receitas: "+String(DOC.receitas));
  }
  if(/conserva|armazen|guardar|dur[aá]vel/.test(msg)){
    if(has("conservacao")) parts.push("Conservação: "+String(DOC.conservacao));
  }
  if(/substitui|alternativa|trocar/.test(msg)){
    const s = DOC.substituicoes || DOC.substituicoes_sugeridas || DOC.substitutos;
    if(s) parts.push("Substituições: "+String(s));
  }
  if(/origem|hist[oó]ria|cultivo|plantio/.test(msg)){
    if(has("origem")) parts.push("Origem: "+String(DOC.origem));
  }

  if(!parts.length){
    const nome = DOC.nome || pepper || "pimenta";
    const base = [];
    if(has("descricao")) base.push("Sobre "+nome+": "+String(DOC.descricao));
    if(has("usos"))      base.push("Usos: "+String(DOC.usos));
    if(has("receitas"))  base.push("Receitas: "+String(DOC.receitas));
    if(has("conservacao")) base.push("Conservação: "+String(DOC.conservacao));
    if(has("origem"))    base.push("Origem: "+String(DOC.origem));
    return { text: base.join("\\n\\n") || "Sem informações locais registradas.", weak:true };
  }
  return { text: parts.join("\\n\\n"), weak:false };
}

async function ask(q){
  let L = answerLocal(q);
  return L.text || "Não encontrei uma boa resposta agora.";
}

document.getElementById('btnSend').onclick = async () => {
  const input = document.getElementById('inputMsg');
  let q = (input.value || "").trim();
  if(!q) return;
  input.value = "";

  if(/^\s*[1-5]\s*$/.test(q)){
    const n = Number(q.trim());
    const map = {1:"O que é?",2:"Usos e receitas",3:"Conservação",4:"Substituições",5:"Origem"};
    q = map[n] || q;
  }

  putMsg(q,true);
  const a = await ask(q);
  putMsg(a,false);
};

document.querySelector(".chips").addEventListener("click",(e)=>{
  const t = e.target.closest(".chip"); if(!t) return;
  const q = t.getAttribute("data-q");
  putMsg(q,true);
  ask(q).then(a=>putMsg(a,false));
});

(async function(){
  await loadKB();
  DOC = pickDoc(pepper) || null;
  const title = document.getElementById('title');
  title.textContent = "Chat: " + (DOC?.nome || pepper || "Pimenta");
  putMsg("Use os botões ou faça sua pergunta. Respondo com base no arquivo local.", false);
})();
</script>
</body>
</html>
"""
    return HTMLResponse(apply_pwa(html))

# ===================== PWA: manifest, sw e injeção de splash =====================
@app.get("/manifest.webmanifest", include_in_schema=False)
def pwa_manifest():
    return FileResponse("static/manifest.webmanifest",
                        media_type="application/manifest+json")

@app.get("/sw.js", include_in_schema=False)
def pwa_sw():
    return FileResponse("static/sw.js", media_type="text/javascript")

# Cabeçalho + overlay (1.8s) + registro do SW injetados nas páginas
PWA_HEAD = """
<link rel="manifest" href="/manifest.webmanifest">
<meta name="theme-color" content="#16a34a">
<link rel="icon" type="image/png" sizes="192x192" href="/static/pimenta-logo.png">
<link rel="apple-touch-icon" href="/static/pimenta-512.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<style>
  #pwa-splash{position:fixed;inset:0;z-index:9999;background:#f7fafc url('/static/splash.png') center 30%/480px no-repeat;display:flex;align-items:flex-end;justify-content:center;transition:opacity .28s ease;opacity:1}
  #pwa-splash .bar{width:56%;height:12px;margin:28px auto 10%;border-radius:999px;background:#e2e8f0;overflow:hidden}
  #pwa-splash .bar::after{content:"";display:block;height:100%;width:0%;background:#16a34a;animation:fill 1.8s linear forwards}
  .hide-splash{opacity:0;pointer-events:none}
  @keyframes fill{to{width:100%}}
</style>
"""

PWA_BODY_START = """
<div id="pwa-splash" aria-hidden="true"><div class="bar"></div></div>
"""

PWA_FOOT = """
<script>
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js', { scope: '/' });
  }
  let _t0=performance.now();
  const MIN=1800; // 1.8s
  function hideSplash(){
    const el=document.getElementById('pwa-splash');
    if(el){ el.classList.add('hide-splash'); setTimeout(()=>el.remove(),300); }
  }
  window.addEventListener('load', ()=>{
    const dt=performance.now()-_t0;
    setTimeout(hideSplash, Math.max(0, MIN-dt));
  });
</script>
"""

def apply_pwa(html: str) -> str:
    # injeta <link rel="manifest"> + CSS do splash no <head>
    if "</head>" in html:
        html = html.replace("</head>", PWA_HEAD + "</head>", 1)
    # injeta o container do splash logo após <body>
    if "<body>" in html:
        html = html.replace("<body>", "<body>" + PWA_BODY_START, 1)
    # registra SW e esconde splash ao final do body
    if "</body>" in html:
        html = html.replace("</body>", PWA_FOOT + "</body>", 1)
    return html

# ===================== ROOT REDIRECT =====================
@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/ui")

# ===================== MAIN (local) =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
