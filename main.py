# main.py — Pimentas App (YOLOv8 + Chat com IA/JSON/Web-RAG)
# UI clara, sem "Pronto" e sem painel de "Resumo".
# Chat: JSON local -> IA com contexto (se DOC existir) -> IA geral (opcional) -> Web-RAG (opcional).

import os, io, time, threading, base64, uuid, json
from typing import Optional
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# YOLO opcional: o app sobe mesmo sem o pacote
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
ENABLE_GENERAL_AI = os.getenv("ENABLE_GENERAL_AI", "0") == "1"   # /ai_general
ENABLE_WEB_RAG    = os.getenv("ENABLE_WEB_RAG", "0") == "1"     # /webqa
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
    """Retorna string ou '' (silenciosa em falha)."""
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
            return (j["choices"][0]["message"]["content"] or "").strip()
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

    # Seleciona doc (match simples; no front há normalização extra)
    doc = None
    if isinstance(kb, dict):
        for k in kb.keys():
            if k.lower()==pepper.lower() or pepper.lower() in k.lower() or k.lower() in pepper.lower():
                doc = kb[k]; break

    # IA com contexto (se houver)
    if doc:
        sys = ("Você é um assistente curto e direto. "
               "Responda em PT-BR com base no JSON fornecido (se compatível) e use parágrafos curtos.")
        user = f"Pergunta: {q}\n\nContexto: {json.dumps(doc, ensure_ascii=False)}"
        text = _openai_chat(
            [{"role":"system","content":sys}, {"role":"user","content":user}]
        )
        return JSONResponse({"ok": bool(text), "text": text or ""})

    return JSONResponse({"ok": False, "text": ""})

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
      <button id="
