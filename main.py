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

# Modelo: mantém seu caminho (se usar MODEL_URL, baixa automaticamente)
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
        image.thumbnail((1024, 1024))  # acelera mantendo proporção

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

# ===================== KB JSON =====================
@app.get("/kb.json")
def kb_json():
    p = os.path.join(STATIC_DIR, "pepper_info.json")
    if not os.path.exists(p):
        return JSONResponse({"detail": "pepper_info.json não encontrado em /static"}, status_code=404)
    return FileResponse(p, media_type="application/json")

# ===================== UI PRINCIPAL =====================
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

# ===================== CHAT — estilo WhatsApp com emojis 1–8 =====================
ALIASES = {
    "biquinho": ["biquinho", "pimenta biquinho", "sweet drop"],
    "bode": ["bode", "pimenta de bode", "bode vermelha", "bode amarela"],
    "cambuci": ["cambuci", "chapéu-de-frade", "chapeu de frade", "chapéu de frade"],
    "chilli": ["chilli", "chili", "pimenta chili", "pimenta chilli"],
    "fidalga": ["fidalga", "fidalga-pepper"],
    "habanero": ["habanero", "pimenta habanero"],
    "jalapeno": ["jalapeno", "jalapeño", "jalapenho", "pimenta jalapeno"],
    "scotch": ["scotch", "scotch bonnet", "scotch-bonnet", "scotchbonnet"]
}

@app.get("/info")
def info(req: Request, pepper: Optional[str] = None):
    # menu com emojis 1–8; aceita nome/sinônimo livre
    html = f"""
<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>Chat: Pimentas</title>
<link rel="icon" href="/static/pimenta-logo.png">
<style>
  :root {{ --bg:#F6F8F3; --card:#fff; --fg:#0b1726; --muted:#6b7280; --brand:#0EA35A; --line:#e2e8f0; }}
  html, body {{ margin:0; background:var(--bg); color:var(--fg); font-family:ui-sans-serif,system-ui; }}
  .wrap {{ max-width:900px; margin:12px auto; padding:8px 12px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:14px; box-shadow:0 2px 6px rgba(0,0,0,.06); }}
  .row {{ display:flex; gap:10px; align-items:center; justify-content:space-between; }}
  .msg {{ background:#f3f4f6; border-radius:14px; padding:12px; margin:10px 0; }}
  .me {{ background:#e0f2fe; margin-left:auto; max-width:70%; white-space:pre-wrap; }}
  .bot {{ background:#ecfccb; max-width:80%; white-space:pre-wrap; }}
  .title {{ font-size:18px; font-weight:700; margin:6px 0 2px; }}
  .muted {{ color:var(--muted); font-size:13px }}
  .btn {{ border:0; background:#eef2ff; color:#3730a3; padding:8px 12px; border-radius:12px; font-weight:600; cursor:pointer; text-decoration:none }}
  .btn.brand {{ background:var(--brand); color:#fff; }}
  #footer {{ position:sticky; bottom:0; background:var(--bg); padding:8px 0; }}
  input, button {{ font:inherit; }}
  #input {{ width:100%; padding:10px 12px; border-radius:12px; border:1px solid var(--line); }}
</style>
</head>
<body>
<div class="wrap">
  <div class="row">
    <div class="title">Chat: <span id="pep">{"".join(pepper or "")}</span></div>
    <a href="/ui" class="btn">← Voltar</a>
  </div>

  <div id="chat" class="card"></div>

  <div id="footer" class="row" style="gap:8px;">
    <input id="input" placeholder="Digite 1–8 ou o nome/sinônimo (ex.: chilli, jalapeño, cambuci)…">
    <button id="send" class="btn brand">Enviar</button>
  </div>
</div>

<script>
const KB_URL = '/static/pepper_info.json';
let KB = {{}};
let current = {json.dumps(pepper or "")};

const MENU = [
  "1️⃣ O que é",
  "2️⃣ Ardência (SHU)",
  "3️⃣ Usos/Receitas",
  "4️⃣ Conservação",
  "5️⃣ Substituições",
  "6️⃣ Origem",
  "7️⃣ Curiosidades/Extras",
  "8️⃣ Trocar pimenta"
].join("\\n");

const ALIASES = {json.dumps(ALIASES)};

function dom(tag, cls, text){ const e=document.createElement(tag); if(cls) e.className=cls; if(text!=null) e.textContent=text; return e; }
function push(text, who){
  const el = dom('div', 'msg ' + (who==='me'?'me':'bot'), text);
  document.getElementById('chat').appendChild(el);
  el.scrollIntoView({behavior:'smooth', block:'end'});
}
function me(t){ push(t,'me'); }
function bot(t){ push(t,'bot'); }

function norm(s){
  return (s||"").toLowerCase()
    .normalize('NFD').replace(/[\u0300-\u036f]/g,'')
    .replace(/-?pepper/g,'')
    .replace(/[^a-z0-9\\s-]/g,'').trim();
}

function matchPepper(q){
  const n = norm(q);
  if(!n) return null;
  // por chave direta
  for(const k of Object.keys(KB)){ if(norm(k)===n) return k; }
  // por alias
  for(const [k, arr] of Object.entries(ALIASES)){
    for(const a of arr){ if(norm(a)===n) return k; }
  }
  // substring aproximada
  for(const k of Object.keys(KB)){ if(norm(k).includes(n) || n.includes(norm(k))) return k; }
  return null;
}

function replyFor(pe, opt){
  const info = KB[pe] || {{}};
  switch(opt){
    case '1': return info.descricao || 'Sem descrição.';
    case '2': return info.shu ? ('Ardência (SHU): '+info.shu) : 'Sem SHU.';
    case '3': return (info.usos ? 'Usos: '+info.usos+'\\n' : '') + (info.receitas ? 'Receitas: '+info.receitas : (info.usos?'':'Sem sugestões.'));
    case '4': return info.conservacao || 'Sem orientações de conservação.';
    case '5': return (info.substituicoes || info.substituicoes_sugeridas || info.substitutos || 'Sem substituições.');
    case '6': return info.origem || 'Sem dados de origem.';
    case '7': return info.extras || 'Sem extras.';
    case '8': return 'Ok! Digite o *nome* da pimenta para trocar (ex.: jalapeño, chilli, cambuci).';
    default : return 'Opção inválida.\\n\\n'+MENU;
  }
}

async function boot(){
  try{ const r = await fetch(KB_URL, {{cache:'no-store'}}); KB = await r.json(); } catch(e){ KB = {{}}; }
  bot('Escolha pelo *menu 1–8* ou digite o *nome/sinônimo* da pimenta.'); 
  if(current){
    const m = matchPepper(current); 
    if(m && KB[m]){{ current = m; bot('Pimenta atual: *'+m+'*'); }}
  }
  bot(MENU);
}

function send(){
  const el = document.getElementById('input');
  const v = (el.value||'').trim();
  if(!v) return;
  el.value = '';
  me(v);

  // número do menu (1–8)
  if(/^\\s*[1-8]\\s*$/.test(v) && current){
    const ans = replyFor(current, v.trim());
    bot(ans);
    if(v.trim()==='8') current = ""; // trocar
    return;
  }
  if(/^\\s*8\\s*$/.test(v)){ bot('Digite o *nome* da pimenta.'); current=""; return; }

  // tentativa de setar pimenta pelo nome/sinônimo
  const m = matchPepper(v);
  if(m && KB[m]){{ current = m; bot('Ok! *'+m+'* selecionada.'); bot(MENU); return; }}

  // pergunta livre: tenta responder com base nos campos
  if(current && KB[current]){
    const q = v.toLowerCase();
    const info = KB[current];
    const parts = [];
    if(/(o que|defini|sobre)/.test(q) && info.descricao) parts.push(info.descricao);
    if(/(uso|receita|culin|molho)/.test(q)){ if(info.usos) parts.push('Usos: '+info.usos); if(info.receitas) parts.push('Receitas: '+info.receitas); }
    if(/(conserva|armazen|guardar)/.test(q) && info.conservacao) parts.push('Conservação: '+info.conservacao);
    if(/(substitu)/.test(q)){ const s = info.substituicoes||info.substituicoes_sugeridas||info.substitutos; if(s) parts.push('Substituições: '+s); }
    if(/(origem|hist)/.test(q) && info.origem) parts.push('Origem: '+info.origem);
    bot(parts.length ? parts.join('\\n\\n') : 'Não encontrei na base local. Use o menu 1–8 ou troque a pimenta (8️⃣).');
    return;
  }

  bot('Não entendi 🤷. Use *1–8* ou o *nome/sinônimo* da pimenta.\\n\\n'+MENU);
}

document.getElementById('send').addEventListener('click', send);
document.getElementById('input').addEventListener('keydown', (e)=>{{ if(e.key==='Enter') send(); }});
boot();
</script>
</body>
</html>
"""
    return HTMLResponse(apply_pwa(html))

# ===================== PWA: manifest, sw e splash injection =====================
@app.get("/manifest.webmanifest", include_in_schema=False)
def pwa_manifest():
    return FileResponse("static/manifest.webmanifest",
                        media_type="application/manifest+json")

@app.get("/sw.js", include_in_schema=False)
def pwa_sw():
    return FileResponse("static/sw.js", media_type="text/javascript")

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
PWA_BODY_START = """<div id="pwa-splash" aria-hidden="true"><div class="bar"></div></div>"""
PWA_FOOT = """
<script>
  if ('serviceWorker' in navigator) { navigator.serviceWorker.register('/sw.js', { scope: '/' }); }
  let _t0=performance.now(); const MIN=1800;
  function hideSplash(){ const el=document.getElementById('pwa-splash'); if(el){ el.classList.add('hide-splash'); setTimeout(()=>el.remove(),300); } }
  window.addEventListener('load', ()=>{ const dt=performance.now()-_t0; setTimeout(hideSplash, Math.max(0, MIN-dt)); });
</script>
"""
def apply_pwa(html: str) -> str:
    if "</head>" in html: html = html.replace("</head>", PWA_HEAD + "</head>", 1)
    if "<body>" in html:  html = html.replace("<body>", "<body>" + PWA_BODY_START, 1)
    if "</body>" in html: html = html.replace("</body>", PWA_FOOT + "</body>", 1)
    return html

# ===================== ROOT → /ui =====================
@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/ui")

# ===================== MAIN (local) =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
