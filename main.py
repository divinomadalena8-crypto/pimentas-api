# main.py — Pimentas App (YOLOv8 + Chat com IA/JSON/Web-RAG)
# UI clara, sem indicador "Pronto" e sem painel/linha de "Resumo".
# Chat robusto: IA (se houver) -> JSON local -> IA geral -> Web-RAG (Tavily).

import os, io, time, threading, base64, uuid, json
from typing import Optional
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# YOLO opcional (app sobe mesmo sem)
try:
    from ultralytics import YOLO
    _ULTRA = True
except Exception:
    YOLO = None
    _ULTRA = False

from PIL import Image
import numpy as np

app = FastAPI(title="Pimentas App")

# ----------------- Config -----------------
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://huggingface.co/bulipucca/pimentas-model/resolve/main/best.onnx"
)
MODEL_PATH = os.path.basename(urlparse(MODEL_URL).path) if MODEL_URL else "best.onnx"

PRESET = os.getenv("PRESET", "ULTRA")
PRESETS = {
    "ULTRA":       dict(imgsz=320, conf=0.35, iou=0.50, max_det=16),
    "RAPIDO":      dict(imgsz=384, conf=0.30, iou=0.50, max_det=12),
    "EQUILIBRADO": dict(imgsz=448, conf=0.30, iou=0.50, max_det=16),
    "PRECISO":     dict(imgsz=512, conf=0.40, iou=0.55, max_det=20),
    "MAX_RECALL":  dict(imgsz=640, conf=0.12, iou=0.45, max_det=24),
}
CFG = PRESETS.get(PRESET, PRESETS["ULTRA"])

RETURN_IMAGE = True
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
REQ_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

STATIC_DIR = os.path.join(os.getcwd(), "static")
ANNOT_DIR  = os.path.join(STATIC_DIR, "annotated")
os.makedirs(ANNOT_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "").strip()
ENABLE_GENERAL_AI = os.getenv("ENABLE_GENERAL_AI", "0") == "1"
ENABLE_WEB_RAG    = os.getenv("ENABLE_WEB_RAG", "0") == "1"
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY", "").strip()

# ----------------- Estado -----------------
model: Optional["YOLO"] = None
labels = {}
READY = False
LOAD_ERR = None

# ----------------- Helpers -----------------
def ensure_model_file():
    if os.path.exists(MODEL_PATH) or not _ULTRA:
        return
    with requests.get(MODEL_URL, headers=REQ_HEADERS, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk: f.write(chunk)

def _to_b64_png(np_bgr: np.ndarray) -> Optional[str]:
    try:
        rgb = np_bgr[:, :, ::-1]
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None

def background_load():
    global model, labels, READY, LOAD_ERR
    try:
        t0 = time.time()
        if _ULTRA:
            ensure_model_file()
            m = YOLO(MODEL_PATH)
            try: m.fuse()
            except: pass
            model = m
            labels = m.names
        READY = True
        print(f"[init] pronto em {time.time()-t0:.1f}s (ultra={_ULTRA})")
    except Exception as e:
        LOAD_ERR = str(e)
        READY = False
        print("[init] erro:", LOAD_ERR)

@app.on_event("startup")
def on_startup():
    threading.Thread(target=background_load, daemon=True).start()

# ----------------- Health -----------------
@app.get("/")
def health():
    return {
        "status": "ok" if READY else "warming",
        "ready": READY,
        "error": LOAD_ERR,
        "model": MODEL_PATH if MODEL_PATH else None,
        "classes": list(labels.values()) if READY and labels else None,
    }

@app.head("/")
def head():
    return Response(status_code=200)

# ----------------- KB (debug) -----------------
@app.get("/kb.json")
def kb_json():
    p = os.path.join(STATIC_DIR, "pepper_info.json")
    if not os.path.exists(p):
        return JSONResponse({"detail": "pepper_info.json não encontrado em /static"}, status_code=404)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(data)

# ----------------- IA helpers -----------------
def _openai_chat(messages, model="gpt-4o-mini", temperature=0.2, max_tokens=500) -> str:
    """Retorna string ou '' (nunca mensagem de erro para o usuário)."""
    if not OPENAI_API_KEY:
        return ""
    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type":"application/json"},
            json={"model": model, "temperature": temperature, "max_tokens": max_tokens, "messages": messages},
            timeout=30
        )
        j = r.json()
        if isinstance(j, dict) and j.get("choices"):
            text = j["choices"][0]["message"]["content"].strip()
            return text
        return ""
    except Exception:
        return ""

def _tavily_search(q: str, k: int = 5) -> str:
    if not (ENABLE_WEB_RAG and TAVILY_API_KEY): return ""
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_API_KEY, "query": q, "max_results": k, "include_answer": True},
            timeout=30
        )
        j = r.json()
        if j.get("answer"): return j["answer"].strip()
        if "results" in j and j["results"]:
            tops = []
            for it in j["results"][:k]:
                title = it.get("title","")
                url   = it.get("url","")
                snippet = it.get("content","")
                tops.append(f"- {title}: {snippet} ({url})")
            return "Resumo Web:\n" + "\n".join(tops)
        return ""
    except Exception:
        return ""

# ----------------- IA endpoints -----------------
@app.post("/ai")
def ai_local(payload: dict):
    pepper = payload.get("pepper","").strip()
    q      = payload.get("q","").strip()

    # Carrega KB local
    kb = {}
    p = os.path.join(STATIC_DIR, "pepper_info.json")
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                kb = json.load(f)
        except:
            kb = {}

    # Seleciona doc
    doc = None
    if isinstance(kb, dict):
        for k in kb.keys():
            if k.lower()==pepper.lower() or pepper.lower() in k.lower() or k.lower() in pepper.lower():
                doc = kb[k]; break

    # Tenta IA com contexto (silenciosa ao falhar)
    sys = ("Você é um assistente curto e direto. "
           "Responda em PT-BR com base no JSON fornecido (se compatível) e formate com parágrafos curtos.")
    user = f"Pergunta: {q}\n\nContexto: {json.dumps(doc or {}, ensure_ascii=False)}"
    text = _openai_chat(
        [{"role":"system","content":sys}, {"role":"user","content":user}]
    )
    ok = bool(text)
    return JSONResponse({"ok": ok, "text": text or ""})

@app.post("/ai_general")
def ai_general(payload: dict):
    if not ENABLE_GENERAL_AI:
        return JSONResponse({"ok": False, "text": ""})
    q = payload.get("q","")
    sys = "Responda em PT-BR, objetivo, até 4 linhas."
    text = _openai_chat(
        [{"role":"system","content":sys}, {"role":"user","content":q}],
        max_tokens=300
    )
    return JSONResponse({"ok": bool(text), "text": text or ""})

@app.post("/webqa")
def webqa(payload: dict):
    q = payload.get("q","")
    ans = _tavily_search(q)
    if not ans: return JSONResponse({"ok": False, "text": ""})
    if OPENAI_API_KEY:
        text = _openai_chat(
            [{"role":"system","content":"Resuma em até 4 linhas, cite fonte entre parênteses quando possível."},
             {"role":"user","content":ans}],
            max_tokens=300
        )
        return JSONResponse({"ok": True, "text": text or ans})
    return JSONResponse({"ok": True, "text": ans})

# ----------------- Predição -----------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not READY:
        return JSONResponse({"ok": False, "warming_up": True, "error": LOAD_ERR}, status_code=503)
    t0 = time.time()
    try:
        im_bytes = await file.read()
        image = Image.open(io.BytesIO(im_bytes)).convert("RGB")
        image.thumbnail((1024,1024))

        preds = []
        image_b64 = None
        image_url = None

        if _ULTRA and model is not None:
            r = model.predict(image, imgsz=CFG["imgsz"], conf=CFG["conf"], iou=CFG["iou"],
                              max_det=CFG["max_det"], device="cpu", verbose=False)[0]
            if r.boxes is not None and len(r.boxes)>0:
                xyxy = r.boxes.xyxy.cpu().numpy().tolist()
                cls  = r.boxes.cls.cpu().numpy().astype(int).tolist()
                conf = r.boxes.conf.cpu().numpy().tolist()
                for (x1,y1,x2,y2),c,cf in zip(xyxy,cls,conf):
                    preds.append({
                        "classe": labels.get(int(c), str(int(c))) if labels else str(int(c)),
                        "conf": round(float(cf),4),
                        "bbox_xyxy":[round(x1,2),round(y1,2),round(x2,2),round(y2,2)]
                    })
                if RETURN_IMAGE:
                    annotated = r.plot()
                    fname = f"{uuid.uuid4().hex}.png"
                    fpath = os.path.join(ANNOT_DIR, fname)
                    Image.fromarray(annotated[:, :, ::-1]).save(fpath)
                    image_url = f"/static/annotated/{fname}"
                    image_b64 = _to_b64_png(annotated)

        top = max(preds, key=lambda p:p["conf"]) if preds else None
        return JSONResponse({
            "ok": True,
            "inference_time_s": round(time.time()-t0,3),
            "num_dets": len(preds),
            "top_pred": top,
            "preds": preds,
            "image_b64": image_b64,
            "image_url": image_url
        })
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e), "inference_time_s": round(time.time()-t0,3)})

# ----------------- /info (CHAT) -----------------
@app.get("/info")
def info():
    html = r"""
<!doctype html>
<html lang="pt-br">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"/>
  <title>Chat: Pimentas</title>
  <link rel="icon" href="/static/pimenta-logo.png" type="image/png" sizes="any">
  <style>
    :root{ --bg:#f7fafc; --card:#ffffff; --fg:#0f172a; --muted:#475569; --line:#e2e8f0; --accent:#16a34a;}
    *{box-sizing:border-box}
    html,body{margin:0;background:var(--bg);color:var(--fg);font:400 16px/1.45 system-ui,-apple-system,Segoe UI,Roboto}
    .wrap{max-width:980px;margin:auto;padding:16px}
    header{display:flex;align-items:center;gap:10px}
    header h1{font-size:20px;margin:0}
    .card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:12px;box-shadow:0 4px 24px rgba(15,23,42,.06)}
    .chips{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}
    .chip{border:1px solid var(--line);border-radius:999px;padding:6px 10px;background:#fff;cursor:pointer;font-size:13px}
    .messages{border:1px solid var(--line);border-radius:12px;padding:10px;background:#fff;height:58vh;min-height:290px;overflow:auto}
    .msg{margin:6px 0;display:flex}
    .msg.me{justify-content:flex-end}
    .bubble{max-width:80%;padding:8px 10px;border-radius:12px;border:1px solid var(--line);white-space:pre-wrap}
    .bubble.me{background:#eef2ff;border-color:#c7d2fe}
    .composer{display:flex;gap:8px;margin-top:8px;position:sticky;bottom:0;background:var(--card);padding:8px;border-top:1px solid var(--line);border-radius:0 0 12px 12px}
    .input{flex:1;border:1px solid var(--line);border-radius:12px;padding:10px}
    .btn{appearance:none;border:1px solid var(--line);background:var(--accent);color:#fff;padding:10px 14px;border-radius:12px;cursor:pointer;font-weight:600}
    a.back{margin-left:auto;border:1px solid var(--line);padding:6px 10px;border-radius:10px;text-decoration:none;color:var(--fg);background:#fff}
  </style>
</head>
<body>
<div class="wrap">
  <header>
    <img src="/static/pimenta-logo.png" alt="Logo" width="28" height="28" onerror="this.style.display='none'">
    <h1 id="title">Chat de Pimentas</h1>
    <a class="back" href="/ui">← Voltar</a>
  </header>

  <section class="card">
    <div class="chips" id="chips">
      <span class="chip" data-q="O que é?">O que é?</span>
      <span class="chip" data-q="Usos/receitas">Usos/receitas</span>
      <span class="chip" data-q="Conservação">Conservação</span>
      <span class="chip" data-q="Substituições">Substituições</span>
      <span class="chip" data-q="Origem">Origem</span>
    </div>

    <div id="messages" class="messages"></div>

    <div class="composer">
      <input id="inputMsg" class="input" placeholder="Digite 1–5 (atalhos) ou faça uma pergunta livre..."/>
      <button id="btnSend" class="btn">Enviar</button>
    </div>
  </section>
</div>

<script>
const qs = new URLSearchParams(location.search);
const pepper = qs.get("pepper") || "";
let KB=null, DOC=null;

function el(t,c,txt){const e=document.createElement(t); if(c)e.className=c; if(txt!==undefined)e.textContent=txt; return e;}
function scrollEnd(){const b=document.getElementById('messages'); b.scrollTop=b.scrollHeight;}
function putMsg(text, me=false){const w=el("div","msg"+(me?" me":"")); const b=el("div","bubble"+(me?" me":""), text); w.appendChild(b); document.getElementById('messages').appendChild(w); scrollEnd();}

async function loadKB(){ try{ const r=await fetch("/kb.json",{cache:"no-store"}); KB=await r.json(); }catch{KB={};} }
function pickDoc(name){
  if(!KB||typeof KB!=="object") return null;
  const keys=Object.keys(KB);
  let k=keys.find(x=>x.toLowerCase()===(name||"").toLowerCase());
  if(!k&&name){ k=keys.find(x=> name.toLowerCase().includes(x.toLowerCase()) || x.toLowerCase().includes(name.toLowerCase())); }
  return k?KB[k]:null;
}
function normalizeQuestion(q){
  const t=q.trim();
  if(t==="1")return "O que é?"; if(t==="2")return "Usos/receitas"; if(t==="3")return "Conservação";
  if(t==="4")return "Substituições"; if(t==="5")return "Origem"; return q;
}

// local -> {text, weak}
function answerLocal(q){
  if(!DOC) return {text:"Ainda não tenho dados desta pimenta.", weak:true};
  const msg=q.toLowerCase();
  const parts=[]; let hits=0;

  if(/o que|o que é|\?$/.test(msg) && DOC.descricao){ parts.push(DOC.descricao); hits++; }
  if(/uso|receita|culin[aá]ria|prato/.test(msg)){
    if(DOC.usos){ parts.push("Usos: "+DOC.usos); hits++; }
    if(DOC.receitas){ parts.push("Receitas: "+DOC.receitas); hits++; }
    if(!DOC.usos && !DOC.receitas){ parts.push("Sem usos/receitas registrados."); }
  }
  if(/conserva[cç][aã]o|armazen/.test(msg)){ parts.push(DOC.conservacao || "Sem orientações de conservação registradas."); hits++; }
  if(/substitu/i.test(msg)){ const s=DOC.substituicoes||DOC.substituicoes_sugeridas; parts.push(s?("Substituições: "+s):"Sem substituições sugeridas."); hits++; }
  if(/origem|hist[oó]ria|cultivo|plantio/.test(msg)){ parts.push(DOC.origem || "Sem dados de origem registrados."); hits++; }

  if(!parts.length){
    const base = DOC.descricao ? ("Sobre "+(DOC.nome||pepper||"pimenta")+": "+DOC.descricao) :
                                  "Posso falar sobre usos/receitas, conservação, substituições ou origem.";
    return {text: base, weak: true};
  }
  return {text: parts.join("\\n\\n"), weak: hits<=1};
}

async function ask(q){
  // 1) IA com contexto (silenciosa ao falhar)
  let a=""; let ok=false;
  try{
    const r=await fetch("/ai",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({pepper:(DOC?.nome||pepper||""),q})});
    if(r.ok){ const j=await r.json(); if(j && j.ok && j.text) { a=j.text; ok=true; } }
  }catch(e){}

  // 2) local
  let weak=false;
  if(!ok){ const L=answerLocal(q); a=L.text; weak=L.weak; }

  // 3) IA geral se fraco
  if(weak){
    try{
      const g=await fetch("/ai_general",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({q})});
      if(g.ok){ const gj=await g.json(); if(gj && gj.ok && gj.text){ a=gj.text; weak=false; } }
    }catch(e){}
  }

  // 4) Web-RAG se ainda fraco
  if(weak){
    try{
      const w=await fetch("/webqa",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({q})});
      if(w.ok){ const wj=await w.json(); if(wj && wj.ok && wj.text){ a=wj.text; } }
    }catch(e){}
  }
  return a || "Não encontrei uma boa resposta agora.";
}

document.getElementById('chips').addEventListener('click', async (e)=>{
  const t=e.target.closest('.chip'); if(!t) return;
  const q=t.getAttribute('data-q'); putMsg(q,true); putMsg(await ask(q),false);
});

document.getElementById('btnSend').onclick = async ()=>{
  const input=document.getElementById('inputMsg'); let q=(input.value||"").trim(); if(!q) return; input.value="";
  const mapped=normalizeQuestion(q); if(mapped!==q) q=mapped;
  putMsg(q,true); putMsg(await ask(q),false);
  input.scrollIntoView({block:"nearest"});
};

(async function(){
  await loadKB();
  DOC=pickDoc(pepper)||null;
  document.getElementById('title').textContent="Chat: "+(DOC?.nome||pepper||"Pimenta");
  putMsg("Use os botões ou faça sua pergunta. Respondo com base no arquivo local e, se precisar, amplio com IA/Web.",false);
})();
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)

# ----------------- /ui (principal) -----------------
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
    .wrap{max-width:1100px;margin:auto;padding:16px 14px 24px}
    header{display:flex;align-items:center;gap:10px}
    header h1{font-size:22px;margin:0}
    .card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:0 4px 24px rgba(15,23,42,.06)}
    .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
    .btn{appearance:none;border:1px solid var(--line);background:#fff;color:var(--fg);padding:10px 14px;border-radius:12px;cursor:pointer;font-weight:600}
    .btn[disabled]{opacity:.6;cursor:not-allowed}
    .btn.accent{background:var(--accent);border-color:var(--accent);color:#fff}
    .tip{color:var(--muted);font-size:13px}
    .imggrid{display:grid;grid-template-columns:1fr;gap:12px}
    @media(min-width:900px){.imggrid{grid-template-columns:1fr 1fr}}
    .imgwrap{background:#fff;border:1px solid var(--line);border-radius:12px;padding:8px}
    img,video,canvas{width:100%;height:auto;display:block;border-radius:10px}
    footer{margin-top:16px;padding:10px 14px;background:#ffffff;border-top:1px solid var(--line);color:var(--muted);font-size:12px;text-align:center}
    .pill{display:inline-block;padding:6px 10px;border-radius:999px;background:#eef2ff;border:1px solid #c7d2fe;color:#3730a3;font-size:12px}
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
        <button id="btnChat" class="btn" style="display:none">Conversar sobre a pimenta</button>

        <span id="badgeTime" class="pill">–</span>
      </div>
      <p class="tip">Comprimimos para ~1024px antes do envio para acelerar.</p>

      <div class="imggrid" style="margin-top:10px">
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

    <footer>Desenvolvido por <strong>Madalena de Oliveira Barbosa</strong>, 2025</footer>
  </div>

<script>
const API = window.location.origin;
let currentFile=null, stream=null, lastClass=null;

function sleep(ms){return new Promise(r=>setTimeout(r,ms));}

async function compressImage(file,maxSide=1024,quality=0.85){
  return new Promise((resolve,reject)=>{
    const img=new Image();
    img.onload=()=>{
      const scale=Math.min(1,maxSide/Math.max(img.width,img.height));
      const w=Math.round(img.width*scale),h=Math.round(img.height*scale);
      const cv=document.getElementById('canvas'),ctx=cv.getContext('2d');
      cv.width=w; cv.height=h; ctx.drawImage(img,0,0,w,h);
      cv.toBlob(b=>{ if(!b) return reject(new Error("compress fail"));
        resolve(new File([b],file.name||"photo.jpg",{type:"image/jpeg"})); },"image/jpeg",quality);
    };
    img.onerror=reject; img.src=URL.createObjectURL(file);
  });
}

// Habilita o botão quando a API estiver pronta (sem mostrar status visual)
async function waitReady(){
  try{
    const r=await fetch(API+"/",{cache:"no-store"}); const d=await r.json();
    if(d.ready){ document.getElementById('btnSend').disabled=!currentFile; return; }
  }catch{}
  await sleep(1200); waitReady();
}

const inputGallery=document.getElementById('fileGallery');
const inputCamera=document.getElementById('fileCamera');
document.getElementById('btnPick').onclick=()=>{ inputGallery.value=""; inputGallery.click(); };
inputGallery.onchange=()=>useLocalFile(inputGallery.files?.[0]);
inputCamera.onchange =()=>useLocalFile(inputCamera.files?.[0]);

async function useLocalFile(f){
  if(!f) return;
  currentFile=await compressImage(f);
  document.getElementById('preview').src=URL.createObjectURL(currentFile);
  document.getElementById('preview').style.display="block";
  document.getElementById('video').style.display="none";
  document.getElementById('btnSend').disabled=false;
  document.getElementById('btnChat').style.display="none";
  lastClass=null;
}

const btnCam=document.getElementById('btnCam');
const btnShot=document.getElementById('btnShot');
const video=document.getElementById('video');

btnCam.onclick=async()=>{
  try{
    if(!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) throw new Error("no gum");
    stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:"environment"}}});
    video.srcObject=stream; video.style.display="block";
    document.getElementById('preview').style.display="none";
    btnShot.style.display="inline-block";
  }catch(e){ inputCamera.value=""; inputCamera.click(); }
};

btnShot.onclick=()=>{
  const cv=document.getElementById('canvas'),ctx=cv.getContext('2d');
  cv.width=video.videoWidth; cv.height=video.videoHeight; ctx.drawImage(video,0,0);
  cv.toBlob(async b=>{
    currentFile=await compressImage(new File([b],"camera.jpg",{type:"image/jpeg"}));
    document.getElementById('preview').src=URL.createObjectURL(currentFile);
    document.getElementById('preview').style.display="block"; video.style.display="none"; btnShot.style.display="none";
    if(stream){ stream.getTracks().forEach(t=>t.stop()); stream=null; }
    document.getElementById('btnSend').disabled=false; document.getElementById('btnChat').style.display="none"; lastClass=null;
  },"image/jpeg",0.92);
};

document.getElementById('btnSend').onclick=async()=>{
  if(!currentFile) return;
  document.getElementById('btnSend').disabled=true;
  const t0=performance.now();
  try{
    const fd=new FormData(); fd.append("file",currentFile,currentFile.name||"image.jpg");
    const r=await fetch(API+"/predict",{method:"POST",body:fd}); const d=await r.json();

    if(d.ok===false && d.warming_up){ return; }
    if(d.ok===false){ return; }

    if(d.image_b64){ document.getElementById('annotated').src=d.image_b64; }
    else if(d.image_url){ const url=d.image_url.startsWith("http")?d.image_url:(API+d.image_url); document.getElementById('annotated').src=url; }

    const ms=(performance.now()-t0)/1000;
    document.getElementById('badgeTime').textContent=(d.inference_time_s||ms).toFixed(2)+" s";

    const chatBtn=document.getElementById('btnChat');
    if(d.top_pred && d.top_pred.classe){
      lastClass=d.top_pred.classe; chatBtn.style.display="inline-block";
      chatBtn.onclick=()=>{ location.href="/info?pepper="+encodeURIComponent(lastClass); };
    }else{
      chatBtn.style.display="none"; lastClass=null;
    }
  }catch(e){ }
  finally{ document.getElementById('btnSend').disabled=false; }
};

waitReady();
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)
