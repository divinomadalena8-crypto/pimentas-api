# main.py — API e UI de Identificação de Pimentas (YOLOv8) para Render
# - Carrega o modelo em background (startup rápido)
# - /ui: interface leve (galeria, câmera, compressão, resultado anotado)
# - /info: “chat” leve com informações da pimenta (usa /kb.json)
# - /kb.json: serve pepper_info.json local OU baixa do GitHub (KB_URL)
# - /predict: retorna boxes + imagem anotada (base64 e URL relativa)

import os, io, time, threading, base64, requests, uuid
from typing import List
from urllib.parse import urlparse

from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO
from PIL import Image
import numpy as np


# ===================== APP & STATIC =====================

app = FastAPI(title="API Pimentas YOLOv8")

STATIC_DIR = os.path.join(os.getcwd(), "static")
ANNOT_DIR  = os.path.join(STATIC_DIR, "annotated")
os.makedirs(ANNOT_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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

# Presets (não exibimos no UI; só usamos para inferência)
PRESET = os.getenv("PRESET", "ULTRA")
PRESETS = {
    "ULTRA":       dict(imgsz=320, conf=0.35, iou=0.50, max_det=4),
    "RAPIDO":      dict(imgsz=384, conf=0.30, iou=0.50, max_det=4),
    "EQUILIBRADO": dict(imgsz=448, conf=0.30, iou=0.50, max_det=6),
    "PRECISO":     dict(imgsz=512, conf=0.40, iou=0.55, max_det=8),
    "MAX_RECALL":  dict(imgsz=640, conf=0.12, iou=0.45, max_det=10),
}
CFG = PRESETS.get(PRESET, PRESETS["ULTRA"])

# Baixar modelo privado (opcional)
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
REQ_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# Base externa para /info (opcional). Se não existir local, carrega daqui:
KB_URL = os.getenv(
    "KB_URL",
    "https://raw.githubusercontent.com/divinomadalena8-crypto/pimentas-assets/main/pepper_info.json"
).strip()

# Sempre retornar imagem anotada
RETURN_IMAGE = True


# ===================== ESTADO GLOBAL =====================

model = None
labels = {}
READY = False
LOAD_ERR = None


# ===================== UTILS =====================

def ensure_model_file():
    if os.path.exists(MODEL_PATH):
        return
    if not MODEL_URL or MODEL_URL.startswith("COLE_AQUI"):
        raise RuntimeError("Defina MODEL_URL com link direto do modelo (.pt ou .onnx).")
    print(f"[init] Baixando modelo: {MODEL_URL}")
    with requests.get(MODEL_URL, headers=REQ_HEADERS, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_iterable(8192) if hasattr(r, "iter_iterable") else r.iter_content(8192):
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
    """Carrega YOLO e marca READY."""
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


# ===================== ROTAS DE STATUS =====================

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
    """Espera READY e roda 1 inferência curtinha (compila pipeline)."""
    t0 = time.time()
    while not READY and time.time() - t0 < 90:
        time.sleep(0.5)
    if not READY:
        return {"ok": False, "warming_up": True}
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    _ = model.predict(img, imgsz=CFG["imgsz"], conf=CFG["conf"], iou=CFG["iou"],
                      max_det=1, device="cpu", verbose=False)
    return {"ok": True}


# ===================== PREDICT =====================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not READY:
        return JSONResponse({"ok": False, "warming_up": True, "error": LOAD_ERR}, status_code=503)

    t0 = time.time()
    try:
        im_bytes = await file.read()
        image = Image.open(io.BytesIO(im_bytes)).convert("RGB")

        # acelera em CPU, mantendo proporção
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
                # base64 (mostra instantâneo)
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

        # sem detecções
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


# ===================== KB (JSON de pimentas) =====================

@app.get("/kb.json")
def kb_json():
    """
    Tenta servir pepper_info.json local (static/pepper_info.json).
    Se não existir, baixa do KB_URL (GitHub raw, por exemplo).
    """
    local_path = os.path.join(STATIC_DIR, "pepper_info.json")
    try:
        if os.path.exists(local_path):
            with open(local_path, "rb") as f:
                return Response(content=f.read(), media_type="application/json")
        if KB_URL:
            r = requests.get(KB_URL, timeout=20)
            r.raise_for_status()
            return Response(content=r.content, media_type="application/json")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse({"error": "pepper_info.json não encontrado"}, status_code=404)


# ===================== UI: INFO (chat leve) =====================

@app.get("/info")
def info():
    html = r"""
<!doctype html>
<html lang="pt-br">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Mais sobre a pimenta</title>
  <link rel="icon" href="/static/pimenta-logo.png" type="image/png" sizes="any">
  <style>
    :root{ --bg:#f7fafc; --card:#ffffff; --fg:#0f172a; --muted:#475569; --line:#e2e8f0; --accent:#16a34a; }
    *{box-sizing:border-box}
    html,body{ margin:0;background:var(--bg);color:var(--fg);font:400 16px/1.45 system-ui,-apple-system,Segoe UI,Roboto }
    .wrap{max-width:980px;margin:auto;padding:20px 14px 72px}
    header{display:flex;align-items:center;gap:10px}
    header h1{font-size:22px;margin:0}
    .grid{display:grid;grid-template-columns:1fr;gap:16px;margin-top:16px}
    @media(min-width:900px){.grid{grid-template-columns:1.1fr .9fr}}
    .card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:0 4px 24px rgba(15,23,42,.06)}
    .btn{appearance:none;border:1px solid var(--line);background:#fff;color:var(--fg);padding:10px 14px;border-radius:12px;cursor:pointer;font-weight:600}
    .btn.accent{background:var(--accent);border-color:var(--accent);color:#fff}
    .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
    .tip{color:var(--muted);font-size:13px}
    .pill{display:inline-block;padding:6px 10px;border-radius:999px;background:#eef2ff;border:1px solid #c7d2fe;color:#3730a3;font-size:12px}
    .facts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
    @media(min-width:700px){.facts{grid-template-columns:repeat(3,minmax(0,1fr))}}
    .fact{border:1px solid var(--line);border-radius:12px;padding:10px;background:#fff}
    .messages{border:1px solid var(--line);border-radius:12px;padding:10px;background:#fff;height:300px;overflow:auto}
    .msg{margin:6px 0;display:flex}
    .msg.me{justify-content:flex-end}
    .bubble{max-width:80%;padding:8px 10px;border-radius:12px;border:1px solid var(--line)}
    .bubble.me{background:#eef2ff;border-color:#c7d2fe}
    .chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
    .chip{border:1px solid var(--line);border-radius:999px;padding:6px 10px;background:#fff;cursor:pointer;font-size:13px}
    .hero{width:100%;border-radius:12px;border:1px solid var(--line);display:none}
    footer{position:fixed;left:0;right:0;bottom:0;padding:10px 14px;background:#ffffffd9;border-top:1px solid var(--line);color:var(--muted);font-size:12px;text-align:center;backdrop-filter:saturate(140%) blur(6px)}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <img src="/static/pimenta-logo.png" alt="Logo" width="28" height="28" onerror="this.style.display='none'">
      <h1 id="title">Mais sobre a pimenta</h1>
      <div style="margin-left:auto" class="row">
        <button class="btn" onclick="location.href='/ui'">← Voltar</button>
      </div>
    </header>

    <div class="grid">
      <!-- ESQUERDA -->
      <section class="card">
        <img id="hero" class="hero" alt="Foto ilustrativa da pimenta"/>
        <div class="row" style="justify-content:space-between;align-items:center; margin-top:8px">
          <strong id="pepperName">—</strong>
          <span id="scovillePill" class="pill" title="Unidades Scoville (SHU)">SHU: —</span>
        </div>
        <p id="descricao" class="tip" style="margin-top:8px">Carregando informações…</p>
        <div id="facts" class="facts" style="margin-top:8px"></div>
      </section>

      <!-- DIREITA (chat leve) -->
      <aside class="card">
        <strong>Assistente</strong>
        <div class="chips" id="chips">
          <span class="chip" data-q="O que é essa pimenta?">O que é?</span>
          <span class="chip" data-q="Qual a ardência?">Ardência (SHU)</span>
          <span class="chip" data-q="Como usar em receitas?">Usos/receitas</span>
          <span class="chip" data-q="Como armazenar/conservar?">Conservação</span>
          <span class="chip" data-q="Existe substituição?">Substituições</span>
        </div>
        <div id="messages" class="messages" style="margin-top:8px"></div>
        <div class="row" style="margin-top:8px">
          <input id="inputMsg" class="btn" style="flex:1;text-align:left;font-weight:400" placeholder="Pergunte algo (ex.: Como armazenar?)"/>
          <button id="btnSend" class="btn accent">Enviar</button>
        </div>
      </aside>
    </div>
  </div>

  <footer>Desenvolvido por <strong>Madalena de Oliveira Barbosa</strong>, 2025</footer>

<script>
const qs = new URLSearchParams(location.search);
const pepper = qs.get("pepper") || "";

let KB = null;     // base completa (JSON)
let DOC = null;    // doc da pimenta atual

function el(tag, cls, text){ const e=document.createElement(tag); if(cls) e.className=cls; if(text) e.textContent=text; return e; }
function scrollToEnd(){ const box=document.getElementById('messages'); box.scrollTop = box.scrollHeight; }

async function loadKB(){
  try{ const r = await fetch("/kb.json", {cache:"no-store"}); KB = await r.json(); }
  catch{ KB = {}; }
}

function pickDoc(name){
  if(!KB) return null;
  const keys = Object.keys(KB);
  let k = keys.find(x => x.toLowerCase() === (name||"").toLowerCase());
  if(!k && name){
    k = keys.find(x => (name.toLowerCase().includes(x.toLowerCase()) || x.toLowerCase().includes(name.toLowerCase())));
  }
  let doc = k ? KB[k] : null;
  if(doc && doc.alias_de){ doc = KB[doc.alias_de] || doc; } // suporta alias
  return doc;
}

function rangeLabel(shu){
  if(!shu) return "desconhecida";
  const n = Number(String(shu).replace(/[^0-9]/g,""));
  if(!isFinite(n)) return String(shu);
  if(n < 2500) return "suave";
  if(n < 10000) return "baixa";
  if(n < 50000) return "média";
  if(n < 200000) return "alta";
  return "muito alta";
}

function renderDoc(){
  const title = document.getElementById('title');
  const pName = document.getElementById('pepperName');
  const desc  = document.getElementById('descricao');
  const pill  = document.getElementById('scovillePill');
  const facts = document.getElementById('facts');
  const hero  = document.getElementById('hero');

  if(!DOC){
    title.textContent = "Mais sobre a pimenta";
    pName.textContent = pepper || "Pimenta (não identificada)";
    desc.textContent  = "Não encontrei informações detalhadas desta pimenta no arquivo.";
    pill.textContent  = "SHU: —";
    hero.style.display = "none";
    facts.innerHTML   = "";
    return;
  }
  const nome = DOC.nome || pepper || "Pimenta";
  title.textContent = "Mais sobre: " + nome;
  pName.textContent = nome;
  desc.textContent  = DOC.descricao || "—";
  if(DOC.scoville){
    pill.textContent = "SHU: " + DOC.scoville + " (" + rangeLabel(DOC.scoville) + ")";
  }else{
    pill.textContent = "SHU: —";
  }

  if(DOC.imagem){
    hero.src = DOC.imagem;
    hero.style.display = "block";
  }else{
    hero.style.display = "none";
  }

  const items = [];
  if(DOC.usos) items.push(["Usos", DOC.usos]);
  if(DOC.receitas) items.push(["Receitas", DOC.receitas]);
  if(DOC.conservacao) items.push(["Conservação", DOC.conservacao]);
  if(DOC.substituicoes || DOC.substituicoes_sugeridas) items.push(["Substituições", DOC.substituicoes || DOC.substituicoes_sugeridas]);
  if(DOC.origem) items.push(["Origem", DOC.origem]);

  facts.innerHTML = "";
  items.forEach(([k,v])=>{
    const card = el("div","fact");
    const h = el("div",null,k); h.style.fontWeight="600";
    const p = el("div","tip", typeof v==="string" ? v : JSON.stringify(v,null,2));
    card.appendChild(h); card.appendChild(p); facts.appendChild(card);
  });
}

function putMsg(text, me=false){
  const wrap = el("div","msg"+(me?" me":""));
  const b = el("div","bubble"+(me?" me":""), text);
  wrap.appendChild(b);
  document.getElementById('messages').appendChild(wrap);
  scrollToEnd();
}

function answer(q){
  if(!DOC){ return "Ainda não tenho dados desta pimenta."; }
  const msg = q.toLowerCase();
  const parts = [];
  if(/ard[eê]ncia|pic[aâ]ncia|shu/.test(msg)){
    if(DOC.scoville){ parts.push(`Ardência (SHU): ${DOC.scoville} — ${rangeLabel(DOC.scoville)}.`); }
    else parts.push("Não há registro de SHU nesta base.");
  }
  if(/usar|receita|molho|chutney|salsa|prato|culin[aá]ria/.test(msg)){
    if(DOC.usos || DOC.receitas){
      if(DOC.usos) parts.push(`Usos: ${DOC.usos}`);
      if(DOC.receitas) parts.push(`Receitas: ${DOC.receitas}`);
    } else parts.push("Sem sugestões de uso/receitas registradas.");
  }
  if(/armazenar|conservar|conserva[cç][aã]o|guardar|dur[aá]vel/.test(msg)){
    if(DOC.conservacao) parts.push(`Conservação: ${DOC.conservacao}`);
    else parts.push("Sem orientações de conservação registradas.");
  }
  if(/substitu/i.test(msg)){
    const s = DOC.substituicoes || DOC.substituicoes_sugeridas;
    parts.push(s ? `Substituições: ${s}` : "Sem substituições sugeridas.");
  }
  if(/origem|cultivo|plantio|hist[oó]ria/.test(msg)){
    parts.push(DOC.origem ? `Origem: ${DOC.origem}` : "Sem dados de origem registrados.");
  }
  if(!parts.length){
    const nome = DOC.nome || pepper || "pimenta";
    parts.push(`Sobre ${nome}: ${DOC.descricao || "sem descrição."}`);
    if(DOC.scoville) parts.push(`Ardência (SHU): ${DOC.scoville} (${rangeLabel(DOC.scoville)}).`);
    parts.push("Você pode perguntar sobre ardência, usos/receitas, conservação, substituições ou origem.");
  }
  return parts.join("\\n\\n");
}

document.getElementById('btnSend').onclick = () => {
  const input = document.getElementById('inputMsg');
  const q = (input.value || "").trim();
  if(!q) return;
  input.value = "";
  putMsg(q, true);
  putMsg(answer(q), false);
};
document.getElementById('chips').addEventListener('click', (e)=>{
  const t = e.target.closest('.chip'); if(!t) return;
  const q = t.getAttribute('data-q');
  putMsg(q, true);
  putMsg(answer(q), false);
});

(async function(){
  await loadKB();
  DOC = pickDoc(pepper) || null;
  renderDoc();
  if(pepper){
    putMsg(`Você selecionou: ${pepper}`, true);
    putMsg("O que você quer saber? (Ex.: ardência, usos/receitas, conservação, substituições, origem)");
  }else{
    putMsg("Qual pimenta você deseja saber mais? Volte e identifique uma imagem, ou informe o nome na sua pergunta.");
  }
})();
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)


# ===================== UI: PRINCIPAL =====================

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
    :root{ --bg:#f7fafc; --card:#ffffff; --fg:#0f172a; --muted:#475569; --line:#e2e8f0; --accent:#16a34a; }
    *{box-sizing:border-box}
    html,body{ margin:0;background:var(--bg);color:var(--fg);font:400 16px/1.45 system-ui,-apple-system,Segoe UI,Roboto }
    .wrap{max-width:980px;margin:auto;padding:20px 14px 72px}
    header{display:flex;align-items:center;gap:10px}
    header h1{font-size:22px;margin:0}
    .grid{display:grid;grid-template-columns:1fr;gap:16px;margin-top:16px}
    @media(min-width:900px){.grid{grid-template-columns:1.1fr .9fr}}
    .card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:0 4px 24px rgba(15,23,42,.06)}
    .btn{appearance:none;border:1px solid var(--line);background:#fff;color:var(--fg);padding:10px 14px;border-radius:12px;cursor:pointer;font-weight:600}
    .btn[disabled]{opacity:.6;cursor:not-allowed}
    .btn.accent{background:var(--accent);border-color:var(--accent);color:#fff}
    .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
    .tip{color:var(--muted);font-size:13px}
    .imgwrap{background:#fff;border:1px solid var(--line);border-radius:12px;padding:8px}
    img,video,canvas{max-width:100%;display:block;border-radius:10px}
    .pill{display:inline-block;padding:6px 10px;border-radius:999px;background:#eef2ff;border:1px solid #c7d2fe;color:#3730a3;font-size:12px}
    .status{margin-top:8px;min-height:22px}
    footer{position:fixed;left:0;right:0;bottom:0;padding:10px 14px;background:#ffffffd9;border-top:1px solid var(--line);color:var(--muted);font-size:12px;text-align:center;backdrop-filter:saturate(140%) blur(6px)}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <img src="/static/baixados.png" alt="Logo" width="28" height="28" onerror="this.style.display='none'">
      <h1><center>Identificação de Pimentas</center></h1>
    </header>

    <div class="grid">
      <section class="card">
        <div class="row">
          <button id="btnPick" class="btn">Escolher imagem</button>
          <input id="fileGallery" type="file" accept="image/*" style="display:none"/>
          <input id="fileCamera"  type="file" accept="image/*" capture="environment" style="display:none"/>

          <button id="btnCam" class="btn">Abrir câmera</button>
          <button id="btnShot" class="btn" style="display:none">Capturar</button>

          <button id="btnSend" class="btn accent" disabled>Identificar</button>
          <button id="btnChat" class="btn" style="display:none">Mais informações</button>

          <span id="chip" class="pill">Conectando…</span>
        </div>
        <p class="tip">Comprimimos para ~1024px antes do envio para acelerar.</p>

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

      <aside class="card">
        <div class="row" style="justify-content:space-between">
          <strong>Resumo</strong>
          <span id="badgeTime" class="pill">–</span>
        </div>
        <div id="textoResumo" class="tip" style="margin-top:8px">Envie uma imagem para começar.</div>
      </aside>
    </div>
  </div>

  <footer>Desenvolvido por <strong>Madalena de Oliveira Barbosa</strong>, 2025</footer>

<script>
const API = window.location.origin;
let currentFile = null;
let stream = null;
let lastClass = null;  // classe detectada para o /info

function sleep(ms){ return new Promise(r=>setTimeout(r,ms)); }
function setStatus(txt){ document.getElementById('chip').textContent = txt; }

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

async function waitReady(){
  setStatus("Conectando…");
  try{
    const r = await fetch(API + "/", {cache:"no-store"});
    const d = await r.json();
    if(d.ready){ setStatus("Pronto"); document.getElementById('btnSend').disabled=!currentFile; return; }
    setStatus("Aquecendo…");
  }catch(e){ setStatus("Sem conexão, tentando…"); }
  await sleep(1200); waitReady();
}

// Entradas (separadas)
const inputGallery = document.getElementById('fileGallery');
const inputCamera  = document.getElementById('fileCamera');

document.getElementById('btnPick').onclick = () => {
  inputGallery.value = "";
  inputGallery.click(); // galeria/arquivos
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
  document.getElementById('resumo').textContent = "";
  document.getElementById('textoResumo').textContent = "Imagem pronta para envio.";
}

// Câmera (getUserMedia + fallback input capture)
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
    inputCamera.value = "";   // fallback confiável para WebView
    inputCamera.click();      // abre câmera nativa via seletor
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

// Predição + /info
document.getElementById('btnSend').onclick = async () => {
  if(!currentFile) return;
  document.getElementById('btnSend').disabled = true;
  document.getElementById('resumo').textContent = "Enviando...";
  const t0=performance.now();
  try{
    const fd=new FormData(); fd.append("file", currentFile, currentFile.name||"image.jpg");
    const r=await fetch(API + "/predict", {method:"POST", body:fd});
    const d=await r.json();

    if(d.ok===false && d.warming_up){ document.getElementById('resumo').textContent="Aquecendo o modelo… tente novamente"; return; }
    if(d.ok===false){ document.getElementById('resumo').textContent="Erro: " + (d.error||"desconhecido"); return; }

    if(d.image_b64){ document.getElementById('annotated').src = d.image_b64; }
    else if(d.image_url){ const url = d.image_url.startsWith("http")? d.image_url : (API + d.image_url); document.getElementById('annotated').src = url; }

    const ms=(performance.now()-t0)/1000;
    document.getElementById('badgeTime').textContent = (d.inference_time_s||ms).toFixed(2) + " s";
    const resumo = d.top_pred ? `Pimenta: ${d.top_pred.classe} · ${Math.round((d.top_pred.conf||0)*100)}% · Caixas: ${d.num_dets}`
                              : "Nenhuma pimenta detectada.";
    document.getElementById('resumo').textContent = resumo;
    document.getElementById('textoResumo').textContent = "Resultado exibido ao lado.";

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
    document.getElementById('resumo').textContent = "Falha ao chamar a API.";
  }finally{
    document.getElementById('btnSend').disabled = false;
  }
};

waitReady();
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)
