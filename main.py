# main.py — API e UI de Identificação de Pimentas (YOLOv8) — Render
# Recursos:
# - Modelo carrega em background
# - /ui: câmera/galeria, imagens grandes, chooser quando há múltiplas classes, chips de pimentas no TOPO (cores por nome)
# - /info: chat simples, chips para alternar entre pimentas sem voltar, composer “grudado” (não some com teclado)
# - /kb.json: serve static/pepper_info.json ou baixa do KB_URL
# - Normalização de nomes (acentos, hífens, chili↔chilli) + fallback Levenshtein no /info
# - Rodapé estático (não sobrepõe conteúdo)
# - max_det configurável por env MAX_DET (opcional)

import os, io, time, threading, base64, requests, uuid
from typing import List
from urllib.parse import urlparse

from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO
from PIL import Image
import numpy as np

app = FastAPI(title="API Pimentas YOLOv8")

# ---------------------- STATIC ----------------------
STATIC_DIR = os.path.join(os.getcwd(), "static")
ANNOT_DIR  = os.path.join(STATIC_DIR, "annotated")
os.makedirs(ANNOT_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------------------- CONFIG ----------------------
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
# (opcional) sobrescrever por env:
try:
    _max_det_env = os.getenv("MAX_DET")
    if _max_det_env:
        CFG["max_det"] = int(_max_det_env)
except:  # noqa
    pass

RETURN_IMAGE = True
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
REQ_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

KB_URL = os.getenv(
    "KB_URL",
    "https://raw.githubusercontent.com/divinomadalena8-crypto/pimentas-assets/main/pepper_info.json"
).strip()

# ---------------------- ESTADO ----------------------
model = None
labels = {}
READY = False
LOAD_ERR = None

# ---------------------- UTILS ----------------------
def ensure_model_file():
    if os.path.exists(MODEL_PATH):
        return
    if not MODEL_URL or MODEL_URL.startswith("COLE_AQUI"):
        raise RuntimeError("Defina MODEL_URL (.pt/.onnx).")
    print(f"[init] Baixando modelo: {MODEL_URL}")
    with requests.get(MODEL_URL, headers=REQ_HEADERS, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(8192):
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

# ---------------------- LIFECYCLE ----------------------
@app.on_event("startup")
def on_startup():
    threading.Thread(target=background_load, daemon=True).start()

# ---------------------- STATUS ----------------------
@app.get("/")
def health():
    return {
        "status": "ok" if READY else "warming",
        "ready": READY,
        "error": LOAD_ERR,
        "model": MODEL_PATH if MODEL_PATH else None,
        "classes": list(labels.values()) if READY else None,
        "preset": PRESET,
        "cfg": CFG,
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
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    _ = model.predict(img, imgsz=CFG["imgsz"], conf=CFG["conf"], iou=CFG["iou"],
                      max_det=1, device="cpu", verbose=False)
    return {"ok": True}

# ---------------------- PREDICT ----------------------
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

# ---------------------- KB (JSON) ----------------------
@app.get("/kb.json")
def kb_json():
    """Serve pepper_info.json (local ou remoto)."""
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

# ---------------------- UI: CHAT (/info) ----------------------
@app.get("/info")
def info():
    html = r"""
<!doctype html>
<html lang="pt-br">
<head>
  <meta charset="utf-8"/>
  <!-- viewport-fit=cover ajuda o layout quando o teclado abre (iOS/WebView) -->
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
  <title>Chat de Pimentas</title>
  <link rel="icon" href="/static/pimenta-logo.png" type="image/png" sizes="any">
  <style>
    :root{ --bg:#f7fafc; --card:#ffffff; --fg:#0f172a; --muted:#475569; --line:#e2e8f0; --accent:#16a34a; }
    *{box-sizing:border-box}
    html,body{ margin:0;background:var(--bg);color:var(--fg);font:400 16px/1.5 system-ui,-apple-system,Segoe UI,Roboto; overflow-x:hidden }
    .wrap{max-width:980px; width:min(980px,100%); margin:auto; padding:16px 12px 24px}
    header{display:flex;align-items:center;gap:10px}
    header h1{font-size:20px;margin:0}
    .card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:0 4px 24px rgba(15,23,42,.06)}
    .btn{appearance:none;border:1px solid var(--line);background:#fff;color:#0f172a;padding:10px 14px;border-radius:12px;cursor:pointer;font-weight:600}
    .tip{color:var(--muted);font-size:13px}

    /* Área de mensagens com altura dinâmica (dvh) que respeita teclado */
    .messages{
      border:1px solid var(--line);
      border-radius:12px;
      padding:12px;
      background:#fff;
      height:50dvh;
      min-height:300px;
      overflow:auto;
    }
    .msg{margin:8px 0;display:flex}
    .msg.me{justify-content:flex-end}
    .bubble{max-width:80%;padding:10px 12px;border-radius:12px;border:1px solid var(--line); white-space:pre-wrap}
    .bubble.me{background:#eef2ff;border-color:#c7d2fe}
    .chips{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 12px}
    .chip{border:1px solid var(--line);border-radius:999px;padding:6px 10px;background:#fff;cursor:pointer;font-size:13px}

    /* Barra do compositor fica colada no final do card */
    .composer{
      position: sticky;
      bottom: 0;
      background: #fff;
      padding-top: 10px;
      margin-top: 10px;
      border-top: 1px dashed var(--line);
    }
    @supports (padding: env(safe-area-inset-bottom)){
      .composer{ padding-bottom: max(10px, env(safe-area-inset-bottom)); }
    }

    footer{
      position: static;
      padding: 12px 14px;
      background: #fff;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      text-align: center;
      margin-top: 16px;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <img src="/static/pimenta-logo.png" alt="Logo" width="24" height="24" onerror="this.style.display='none'">
      <h1 id="title">Chat de Pimentas</h1>
      <div style="margin-left:auto">
        <button class="btn" onclick="location.href='/ui'">← Voltar</button>
      </div>
    </header>

    <section class="card">
      <div id="altsWrap" style="display:none; margin-bottom:10px">
        <div class="tip" style="margin-bottom:6px">Outras pimentas detectadas:</div>
        <div id="altsChips" class="chips"></div>
      </div>

      <div class="tip" id="subtitle" style="margin-bottom:8px"></div>

      <div class="chips" id="chips">
        <span class="chip" data-q="O que é essa pimenta?">O que é?</span>
        <span class="chip" data-q="Como usar em receitas?">Usos/receitas</span>
        <span class="chip" data-q="Como armazenar/conservar?">Conservação</span>
        <span class="chip" data-q="Existe substituição?">Substituições</span>
        <span class="chip" data-q="Qual a origem dessa pimenta?">Origem</span>
      </div>

      <div id="messages" class="messages"></div>

      <!-- barra do input “colada” no fim do card -->
      <div class="composer">
        <div style="display:flex;gap:10px;">
          <input id="inputMsg" class="btn" style="flex:1;text-align:left;font-weight:400" placeholder="Pergunte algo (ex.: como usar?)"/>
          <button id="btnSend" class="btn" style="background:var(--accent);color:#fff;border-color:var(--accent)">Enviar</button>
        </div>
      </div>
    </section>
  </div>

  <footer>Desenvolvido por <strong>Madalena de Oliveira Barbosa</strong>, 2025</footer>

<script>
const qs = new URLSearchParams(location.search);
const pepper = qs.get("pepper") || "";
const altsParam = qs.get("alts") || "";
let alts = altsParam.split("|").map(s=>s.trim()).filter(Boolean);
alts = Array.from(new Set([pepper, ...alts].filter(Boolean)));

let KB = null;
let DOC = null;

function el(tag, cls, text){ const e=document.createElement(tag); if(cls) e.className=cls; if(text) e.textContent=text; return e; }
function putMsg(text, me=false){ const wrap=el("div","msg"+(me?" me":"")); wrap.appendChild(el("div","bubble"+(me?" me":""), text)); document.getElementById('messages').appendChild(wrap); const m=document.getElementById('messages'); m.scrollTop=m.scrollHeight; }
function norm(s){ return (s||"").normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase().replace(/[^a-z0-9]/g,''); }

// --- variações e Levenshtein para robustez do "Chili/Chilli", hífens etc. ---
function variants(n){
  const vs = new Set([n]);
  vs.add(n.replace(/chilli/g, "chili"));
  vs.add(n.replace(/chili/g, "chilli"));
  if(!n.endsWith("pepper")) vs.add(n + "pepper");
  if(n.endsWith("pepper"))  vs.add(n.replace(/pepper$/, ""));
  return Array.from(vs);
}
function lev(a,b){
  const m=a.length, n=b.length;
  if(m===0) return n; if(n===0) return m;
  const dp = Array.from({length:m+1}, (_,i)=> Array(n+1).fill(0));
  for(let i=0;i<=m;i++) dp[i][0]=i;
  for(let j=0;j<=n;j++) dp[0][j]=j;
  for(let i=1;i<=m;i++){
    for(let j=1;j<=n;j++){
      const cost = a[i-1]===b[j-1] ? 0 : 1;
      dp[i][j] = Math.min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost);
    }
  }
  return dp[m][n];
}
function pickDoc(name){
  if(!KB) return null;
  const keys = Object.keys(KB);
  if(!keys.length) return null;

  const normMap = Object.fromEntries(keys.map(k => [norm(k), k]));
  const n0 = norm(name||"");

  for(const v of variants(n0)){ if(normMap[v]) return KB[normMap[v]]; }
  for(const v of variants(n0)){
    for(const nk in normMap){
      if(nk.includes(v) || v.includes(nk)) return KB[normMap[nk]];
    }
  }
  let bestKey = None, best = 1e9;
  bestKey = None
  best = 1e9
  for(const nk in normMap){
    const d = lev(n0, nk);
    if(d < best){ best = d; bestKey = nk; }
  }
  if(best <= 3) return KB[normMap[bestKey]];
  return null;
}
// ---------------------------------------------------------------------------

async function loadKB(){
  try{ const r = await fetch("/kb.json", {cache:"no-store"}); KB = await r.json(); }
  catch{ KB = {}; }
}
function drawAlts(){
  const wrap = document.getElementById('altsWrap');
  const chips = document.getElementById('altsChips');
  if(alts.length <= 1){ wrap.style.display="none"; return; }
  wrap.style.display = "block";
  chips.innerHTML = "";
  alts.forEach(name => {
    const chip = el("span","chip", name);
    chip.onclick = () => {
      const url = "/info?pepper=" + encodeURIComponent(name) +
                  "&alts=" + encodeURIComponent(alts.join("|"));
      location.href = url;
    };
    chips.appendChild(chip);
  });
}

function answer(q){
  if(!DOC){ return "Ainda não tenho dados desta pimenta."; }
  const msg = q.toLowerCase();
  const parts = [];
  if(/usar|receita|molho|chutney|salsa|prato|culin[aá]ria/.test(msg)){
    if(DOC.usos || DOC.receitas){
      if(DOC.usos) parts.push(`Usos: ${DOC.usos}`);
      if(DOC.receitas) parts.push(`Receitas: ${DOC.receitas}`);
    } else parts.push("Sem sugestões de uso/receitas registradas.");
  }
  if(/armazenar|conservar|conserva[cç][aã]o|guardar|dur[aá]vel/.test(msg)){
    parts.push(DOC.conservacao ? `Conservação: ${DOC.conservacao}` : "Sem orientações de conservação registradas.");
  }
  if(/substitu/i.test(msg)){
    const s = DOC.substituicoes || DOC.substituicoes_sugeridas;
    parts.push(s ? `Substituições: ${s}` : "Sem substituições sugeridas.");
  }
  if(/origem|hist[oó]ria/.test(msg)){
    parts.push(DOC.origem ? `Origem: ${DOC.origem}` : "Sem dados de origem registrados.");
  }
  if(!parts.length){
    const nome = DOC.nome || pepper || "pimenta";
    parts.push(`Sobre ${nome}: ${DOC.descricao || "sem descrição disponível nesta base."}`);
    parts.push("Você pode perguntar sobre usos/receitas, conservação, substituições ou origem.");
  }
  return parts.join("\n\n");
}

/* --- manter o botão Enviar visível quando o teclado abre --- */
const messages = document.getElementById('messages');
const composer  = document.querySelector('.composer');
const input     = document.getElementById('inputMsg');
function keepComposerVisible(){
  try{ composer.scrollIntoView({block:'end'}); }catch(e){}
  messages.scrollTop = messages.scrollHeight;
}
input.addEventListener('focus', ()=> setTimeout(keepComposerVisible, 100));
window.addEventListener('resize', ()=> setTimeout(keepComposerVisible, 100));
/* ----------------------------------------------------------- */

document.getElementById('btnSend').onclick = () => {
  const q = (input.value || "").trim();
  if(!q) return;
  input.value = "";
  putMsg(q, true);
  putMsg(answer(q), false);
  keepComposerVisible();
};
document.getElementById('chips').addEventListener('click', (e)=>{
  const t = e.target.closest('.chip'); if(!t) return;
  const q = t.getAttribute('data-q');
  putMsg(q, true);
  putMsg(answer(q), false);
  keepComposerVisible();
});

(async function(){
  await loadKB();
  DOC = pickDoc(pepper) || null;
  drawAlts();
  const title = document.getElementById('title');
  const subtitle = document.getElementById('subtitle');
  if(DOC){
    title.textContent = "Chat: " + (DOC.nome || pepper || "Pimenta");
    subtitle.textContent = "Pergunte sobre usos, conservação, substituições e origem.";
  }else{
    title.textContent = "Chat de Pimentas";
    subtitle.textContent = pepper ? ("Não encontrei dados para: " + pepper) : "Informe uma pimenta via a tela inicial.";
    putMsg("Qual pimenta você deseja saber mais? Volte e identifique uma imagem, ou informe o nome na sua pergunta.");
  }
})();
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)

# ---------------------- UI: INICIAL (/ui) ----------------------
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
    html,body{ margin:0;background:var(--bg);color:var(--fg);font:400 16px/1.45 system-ui,-apple-system,Segoe UI,Roboto; overflow-x:hidden }
    .wrap{max-width:1040px; width:min(1040px,100%); margin:auto; padding:16px 12px 24px}
    header{display:flex;align-items:center;gap:10px}
    header h1{font-size:22px;margin:0}
    .card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:0 4px 24px rgba(15,23,42,.06)}
    .btn{appearance:none;border:1px solid var(--line);background:#fff;color:#0f172a;padding:10px 14px;border-radius:12px;cursor:pointer;font-weight:600}
    .btn[disabled]{opacity:.6;cursor:not-allowed}
    .btn.accent{background:var(--accent);border-color:var(--accent);color:#fff}
    .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center; width:100%}
    .tip{color:var(--muted);font-size:13px}
    .imgwrap{background:#fff;border:1px solid var(--line);border-radius:12px;padding:8px;flex:1 1 48%;min-width:260px}
    img,video,canvas{width:100%;height:auto;display:block;border-radius:10px}
    .pill{display:inline-block;padding:6px 10px;border-radius:999px;background:#eef2ff;border:1px solid #c7d2fe;color:#3730a3;font-size:12px}
    .status{margin-top:8px;min-height:22px}
    .chips{display:flex;gap:8px;flex-wrap:wrap}
    .chip{border:1px solid var(--line);border-radius:999px;padding:6px 10px;background:#fff;cursor:pointer;font-size:13px}

    @media(max-width:700px){
      .images{flex-direction:column}
      .imgwrap{flex:1 1 100%;min-width:0}
    }
    footer{
      position: static;
      padding: 12px 14px;
      background: #fff;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      text-align: center;
      margin-top: 16px;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <img src="/static/pimenta-logo.png" alt="Logo" width="28" height="28" onerror="this.style.display='none'">
      <h1>Identificação de Pimentas</h1>
    </header>

    <!-- NOVO: seção de chips de pimentas no topo, com cores por nome -->
    <section class="card">
      <strong style="display:block;margin-bottom:8px">Conheça as pimentas</strong>
      <div id="pepperTopChips" class="chips"></div>
      <p class="tip" style="margin-top:8px">Toque para abrir o chat dessa pimenta — sem precisar identificar uma imagem.</p>
    </section>

    <section class="card" style="margin-top:16px">
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

      <div class="row images" style="margin-top:10px">
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

      <div id="chooser" class="card" style="display:none; margin-top:10px">
        <strong style="display:block; margin-bottom:8px">Várias pimentas detectadas — qual você quer explorar?</strong>
        <div id="chipsClasses" class="row"></div>
      </div>

      <div id="resumo" class="status tip"></div>
    </section>
  </div>

  <footer>Desenvolvido por <strong>Madalena de Oliveira Barbosa</strong>, 2025</footer>

<script>
const API = window.location.origin;
let currentFile = null;
let stream = null;
let KB = null;

function sleep(ms){ return new Promise(r=>setTimeout(r,ms)); }
function setStatus(txt){ document.getElementById('chip').textContent = txt; }

/* ---- Cores por nome (estáveis): gera HSL a partir do texto ---- */
function hueFor(name){
  let h = 0;
  for (let i=0;i<name.length;i++) h = (h*31 + name.charCodeAt(i)) % 360;
  return h;
}
function styleForChip(name){
  const h = hueFor(name);
  const bg = `hsl(${h}, 85%, 93%)`;
  const bd = `hsl(${h}, 60%, 75%)`;
  return `background:${bg}; border-color:${bd}`;
}

/* ---- Chips do topo: carrega KB e desenha ---- */
async function loadKB(){
  try{
    const r = await fetch("/kb.json", {cache:"no-store"});
    KB = await r.json();
  }catch(e){ KB = {}; }
}
function drawTopChips(){
  const box = document.getElementById('pepperTopChips');
  box.innerHTML = "";
  const names = Object.keys(KB || {});
  names.forEach(n=>{
    const b = document.createElement('button');
    b.className = 'chip';
    b.textContent = n;
    b.setAttribute('style', styleForChip(n));
    b.onclick = () => { location.href = "/info?pepper=" + encodeURIComponent(n); };
    box.appendChild(b);
  });
}

/* ---- Compressão de imagem ---- */
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

// Entradas
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
  document.getElementById('chooser').style.display = "none";
  document.getElementById('resumo').textContent = "";
}

// Câmera
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
    // Fallback nativo (WebView abre seletor/câmera do SO)
    inputCamera.value = "";
    inputCamera.click();
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
    document.getElementById('chooser').style.display = "none";
    setStatus("Foto capturada");
  },"image/jpeg",0.92);
};

// Predição + chooser + chat
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
    const resumo = d.top_pred ? `Pimenta: ${d.top_pred.classe} · ${Math.round((d.top_pred.conf||0)*100)}% · Caixas: ${d.num_dets} · ${((d.inference_time_s||ms)).toFixed(2)} s`
                              : "Nenhuma pimenta detectada.";
    document.getElementById('resumo').textContent = resumo;

    // --- classes detectadas (únicas) ---
    const classes = [...new Set((d.preds || []).map(p => p.classe).filter(Boolean))];

    const chooser = document.getElementById('chooser');
    const chipsClasses = document.getElementById('chipsClasses');
    const chatBtn = document.getElementById('btnChat');

    if (classes.length > 1) {
      chooser.style.display = "block";
      chipsClasses.innerHTML = "";
      classes.forEach(c => {
        const b = document.createElement('button');
        b.className = 'btn';
        b.textContent = c;
        b.setAttribute('style', styleForChip(c));
        b.onclick = () => {
          const link = "/info?pepper=" + encodeURIComponent(c) +
                       "&alts=" + encodeURIComponent(classes.join("|"));
          location.href = link;
        };
        chipsClasses.appendChild(b);
      });
      chatBtn.style.display = "none";
    } else if (classes.length === 1) {
      chooser.style.display = "none";
      const c = classes[0];
      chatBtn.style.display = "inline-block";
      chatBtn.onclick = () => {
        const link = "/info?pepper=" + encodeURIComponent(c) +
                     "&alts=" + encodeURIComponent(classes.join("|"));
        location.href = link;
      };
    } else {
      chooser.style.display = "none";
      chatBtn.style.display = "none";
    }
  }catch(e){
    document.getElementById('resumo').textContent = "Falha ao chamar a API.";
  }finally{
    document.getElementById('btnSend').disabled = false;
  }
};

(async function(){
  await loadKB();
  drawTopChips();
  waitReady();
})();
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)
