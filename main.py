# main.py — PWA com Splash + Tela de Identificação + Chat estilo WhatsApp
# -----------------------------------------------------------------------
# requirements.txt:
# fastapi==0.115.0
# uvicorn==0.30.0
# python-multipart==0.0.9
# -----------------------------------------------------------------------

import os, json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Pimentas PWA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- KB (pepper_info) ---------
KB_PATH = Path("static/pepper_info.json")

FALLBACK_KB = {
    # exemplo mínimo; seu arquivo em static/pepper_info.json substitui isso automaticamente
    "Jalapeño": {
        "sinonimos": ["jalapeno", "jalapeños", "jalapenho", "jalapenho", "jalapeño", "chilli", "chili"],
        "oque": "Pimenta de picância moderada (muitas vezes entre 2.500–8.000 SHU). Versátil.",
        "shu": "≈ 2.500 – 8.000 SHU",
        "usos": "Nachos, tacos, hambúrgueres, picles e recheios (poppers).",
        "receitas": "Jalapeño em conserva; poppers recheados; chipotle caseiro.",
        "conservacao": "Na geladeira, inteira ou em conserva; secar para chipotle defumado.",
        "substituicoes": "Serrano (mais picante) ou poblano (menos picante).",
        "origem": "México, amplamente cultivada nas Américas.",
        "extras": "Quando madura fica vermelha; defumada vira 'chipotle'."
    },
    "Biquinho": {
        "sinonimos": ["biquinho", "chupetinha", "sweet drop"],
        "oque": "Pimenta muito suave, sabor adocicado e aroma característico.",
        "shu": "≈ 500 SHU (ou menos)",
        "usos": "Conservas, aperitivos, saladas.",
        "receitas": "Biquinho em conserva; geleia suave.",
        "conservacao": "Conserva em vinagre; refrigerada dura mais.",
        "substituicoes": "Pimenta doce/mini-pimenta.",
        "origem": "Brasil.",
        "extras": "Muito popular em conservas de bares brasileiros."
    },
    "Habanero": {
        "sinonimos": ["habanero", "habaneros"],
        "oque": "Muito picante; perfume frutado/cítrico.",
        "shu": "≈ 100.000 – 350.000 SHU",
        "usos": "Molhos encorpados, marinadas e salsas intensas.",
        "receitas": "Salsa de habanero; molho de habanero com manga.",
        "conservacao": "Secar, congelar ou fazer molho/fermentado.",
        "substituicoes": "Scotch bonnet, bhut jolokia (mais forte).",
        "origem": "Caribe/México (Yucatán).",
        "extras": "Use com parcimônia; use luvas ao manusear."
    }
}

def load_kb():
    if KB_PATH.exists():
        try:
            with KB_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return data
        except Exception:
            pass
    return FALLBACK_KB

KB = load_kb()

# Mapa rápido de sinônimos -> canônico
NAME_MAP = {}
for k, v in KB.items():
    NAME_MAP[k.lower()] = k
    for s in v.get("sinonimos", []):
        NAME_MAP[s.lower()] = k

def resolve_pepper(name: str | None) -> str | None:
    if not name: return None
    key = name.strip().lower()
    return NAME_MAP.get(key)

# --------- Static/PWA ---------
os.makedirs("static/icons", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

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
self.addEventListener('install', e => {
  e.waitUntil(
    caches.open('pimentas-v1').then(c => c.addAll([
      '/ui', '/chat', '/manifest.webmanifest',
      '/static/splash.png', '/static/pimenta-logo.png'
    ]))
  );
});
self.addEventListener('fetch', e => {
  e.respondWith(caches.match(e.request).then(r => r || fetch(e.request)));
});
""".strip()

# --------- UI: /ui (detector) + Splash ----------
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
.pill{display:inline-block;padding:6px 10px;border-radius:999px;background:#eef2ff;border:1px solid #c7d2fe;color:#3730a3;font-size:12px}
.imgwrap{background:#fff;border:1px solid var(--line);border-radius:12px;padding:8px}
img,video,canvas{max-width:100%;display:block;border-radius:10px}
footer{position:fixed;left:0;right:0;bottom:0;padding:10px 14px;background:#ffffffd9;border-top:1px solid var(--line);color:var(--muted);font-size:12px;text-align:center;backdrop-filter:saturate(140%) blur(6px)}
#splash{position:fixed;inset:0;z-index:9999;background:#fff url('/static/splash.png') no-repeat center 34%;background-size:min(72vmin,520px);display:flex;align-items:flex-end;justify-content:center;padding:32px;transition:opacity .35s}
#splash.hide{opacity:0;pointer-events:none}
#splash .bar{width:min(520px,80vw);height:14px;border-radius:999px;background:#e6f4ea;border:1px solid #cce9d6;overflow:hidden}
#splash .bar>i{display:block;width:35%;height:100%;background:#16a34a;animation:load 1.6s infinite}
@keyframes load{0%{transform:translateX(-100%)}50%{transform:translateX(30%)}100%{transform:translateX(120%)}}
</style>
<script>
if('serviceWorker'in navigator){navigator.serviceWorker.register('/sw.js').catch(()=>{});}
</script>
</head><body>
<div id="splash"><div class="bar"><i></i></div></div>
<div class="wrap">
  <header>
    <img src="/static/pimenta-logo.png" alt="logo" width="28" height="28" onerror="this.style.display='none'">
    <h1>Identificação de Pimentas</h1>
    <span style="margin-left:auto" class="pill" id="chip">Conectando…</span>
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

    <p class="tip" style="color:#475569">As imagens são comprimidas antes do envio.</p>

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
let currentFile = null, lastClass = null;
const chip=document.getElementById('chip'), preview=document.getElementById('preview');
function setStatus(t){ chip.textContent=t; }
function hideSplash(){ const s=document.getElementById('splash'); if(!s) return; s.classList.add('hide'); setTimeout(()=>s.remove(), 400); }
function sleep(ms){ return new Promise(r=>setTimeout(r,ms)); }
async function waitReady(){
  setStatus("Conectando…");
  try{
    const r=await fetch(API+"/",{cache:"no-store"}); const d=await r.json();
    if(d.ready){ setStatus("Pronto"); hideSplash(); document.getElementById('btnSend').disabled=!currentFile; return; }
    setStatus("Aquecendo…");
  }catch(e){ setStatus("Sem conexão"); }
  await sleep(900); waitReady();
}
async function compressImage(file,maxSide=1024,quality=.8){
  return new Promise((resolve,reject)=>{
    const img=new Image();
    img.onload=()=>{
      const scale=Math.min(1,maxSide/Math.max(img.width,img.height));
      const w=Math.round(img.width*scale), h=Math.round(img.height*scale);
      const cv=document.getElementById('canvas'), ctx=cv.getContext('2d');
      cv.width=w; cv.height=h; ctx.drawImage(img,0,0,w,h);
      cv.toBlob(b=>{ if(!b) return reject(Error("compress")); resolve(new File([b], file.name||"photo.jpg",{type:"image/jpeg"})); },"image/jpeg",quality);
    };
    img.onerror=reject; img.src=URL.createObjectURL(file);
  });
}
const inputGallery=document.getElementById('fileGallery');
const inputCamera=document.getElementById('fileCamera');
document.getElementById('btnPick').onclick=()=>{inputGallery.value="";inputGallery.click();}
document.getElementById('btnCam' ).onclick=()=>{inputCamera.value=""; inputCamera.click();}
inputGallery.onchange=()=>useLocalFile(inputGallery.files?.[0]);
inputCamera .onchange=()=>useLocalFile(inputCamera .files?.[0]);
async function useLocalFile(f){
  if(!f) return;
  currentFile=await compressImage(f);
  preview.src=URL.createObjectURL(currentFile);
  preview.style.display="block";
  document.getElementById('btnSend').disabled=false;
  document.getElementById('resumo').textContent="Imagem pronta para envio.";
}
document.getElementById('btnSend').onclick=async()=>{
  if(!currentFile) return;
  document.getElementById('btnSend').disabled=true;
  document.getElementById('resumo').textContent="Enviando...";
  try{
    const fd=new FormData(); fd.append("file",currentFile,currentFile.name||"image.jpg");
    const r=await fetch(API+"/predict",{method:"POST",body:fd}); const d=await r.json();
    if(d.ok===false){ document.getElementById('resumo').textContent="Erro: "+(d.error||"desconhecido"); return; }
    lastClass=d.top_pred?.classe||null;
    document.getElementById('resumo').textContent=d.top_pred?(`Pimenta: ${d.top_pred.classe} (${Math.round((d.top_pred.conf||0)*100)}%)`):"Nenhuma pimenta detectada.";
    if(d.image_url) document.getElementById('annotated').src=d.image_url;
    const info=document.getElementById('btnInfo');
    if(lastClass){ info.style.display="inline-block"; info.href="/chat?pepper="+encodeURIComponent(lastClass); info.textContent="Mais informações ↗"; }
    else{ info.style.display="none"; }
  }catch(e){ document.getElementById('resumo').textContent="Falha ao enviar."; }
  finally{ document.getElementById('btnSend').disabled=false; }
};
setTimeout(hideSplash,4000); waitReady();
</script>
</body></html>
"""

# --------- CHAT: /chat estilo WhatsApp ----------
CHAT_HTML = r"""
<!doctype html><html lang="pt-BR"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Chat de Pimentas</title>
<link rel="manifest" href="/manifest.webmanifest">
<meta name="theme-color" content="#16a34a">
<style>
*{box-sizing:border-box}html,body{margin:0;height:100%;background:#e5ddd5;font:400 16px/1.45 system-ui,-apple-system,Segoe UI,Roboto}
.app{display:flex;flex-direction:column;height:100%}
.top{background:#075e54;color:#fff;padding:12px 14px;display:flex;align-items:center;gap:10px}
.top a{color:#cce; text-decoration:none}
.title{font-weight:700}
.chat{flex:1;overflow:auto;background:#efeae2;padding:12px}
.bubble{max-width:80%; margin:6px 0; padding:10px 12px; border-radius:12px; box-shadow:0 2px 4px rgba(0,0,0,.05)}
.bot{background:#fff}
.me{background:#dcf8c6; margin-left:auto}
.menu{white-space:pre-wrap}
.inputbar{display:flex;gap:8px; background:#f0f0f0; padding:8px}
.inputbar input{flex:1; padding:10px 12px; border-radius:20px; border:none; outline:none}
.inputbar button{background:#25d366;color:#fff;border:none;border-radius:20px;padding:0 16px;font-weight:700}
.pill{display:inline-block;background:#e2f7ea;color:#075e54;border:1px solid #bfead0;border-radius:12px;padding:4px 8px;font-size:12px}
</style>
</head><body>
<div class="app">
  <div class="top">
    <div class="title">Chat: <span id="pepname">—</span></div>
    <span style="margin-left:auto"><a href="/ui">← Voltar</a></span>
  </div>

  <div id="chat" class="chat">
    <div class="bubble bot" id="welcome">Carregando dados…</div>
  </div>

  <div class="inputbar">
    <input id="msg" placeholder="Digite a opção (1–8) ou o nome da pimenta…">
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
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}
function menu(canonName){
  const m =
`📌 *${canonName}* — escolha uma opção:
1️⃣ O que é
2️⃣ Ardência (SHU)
3️⃣ Usos/Receitas
4️⃣ Conservação
5️⃣ Substituições
6️⃣ Origem
7️⃣ Curiosidades/Extras
8️⃣ Trocar pimenta

💡 Você também pode digitar o *nome/sinônimo* da pimenta (ex.: jalapeno/jalapeño, chilli, biquinho...).`;
  const div=document.createElement('div');
  div.className='bubble bot menu';
  div.innerText = m;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function canonical(name){
  if(!name) return null;
  const key=name.trim().toLowerCase();
  if(KB.__map && KB.__map[key]) return KB.__map[key];
  return null;
}
function showPepper(name){
  const c = canonical(name) || canonical(current);
  if(!c){ bubble("Não encontrei essa pimenta. Digite um nome válido ou escolha 8️⃣ para trocar."); return; }
  current = c; pepname.textContent = c;
  chat.innerHTML = ''; // reinicia
  bubble("Use os botões ou faça sua pergunta. Respondo com base no arquivo local.");
  menu(c);
}
function answer(opt){
  const c = canonical(current);
  if(!c) return;
  const d = KB[c] || {};
  const map = {
    "1":"oque","2":"shu","3":"usos","4":"conservacao","5":"substituicoes","6":"origem","7":"extras"
  };
  if(opt==="8"){ chat.innerHTML=""; bubble("Digite o nome/sinônimo da pimenta (ex.: jalapeño, biquinho, habanero…)."); return; }
  const k = map[opt];
  bubble(opt, "me");
  if(!k){ bubble("Opção inválida. Use 1–8."); return; }
  if(!d[k]){ bubble("Ainda não tenho dados desta pimenta."); return; }
  bubble(d[k]);
  // reexibe menu compacto
  menu(c);
}

async function loadKB(){
  const r = await fetch('/kb'); const data = await r.json();
  // constrói o mapa de sinônimos
  const MAP = {};
  Object.keys(data).forEach(canon=>{
    MAP[canon.toLowerCase()] = canon;
    (data[canon].sinonimos||[]).forEach(s=>MAP[(s||'').toLowerCase()] = canon);
  });
  data.__map = MAP;
  return data;
}

document.getElementById('send').onclick = ()=>{
  const input = document.getElementById('msg');
  const t = (input.value||'').trim();
  if(!t) return;
  // número?
  if(/^[1-8]$/.test(t)){ answer(t); input.value=''; return; }
  // nome de pimenta?
  const c = canonical(t);
  bubble(t, "me");
  if(c){ current=c; showPepper(c); input.value=''; return; }
  bubble("Não entendi 🥹. Responda com *1–8* ou digite o *nome/sinônimo* da pimenta.");
  menu(canonical(current)||"Pimenta");
  input.value='';
};

(async()=>{
  KB = await loadKB();
  const first = canonical(current) || Object.keys(KB).find(k => k!=="__map");
  current = first;
  pepname.textContent = first;
  document.getElementById('welcome').remove();
  showPepper(first);
})();
</script>
</body></html>
"""

# --------- ROUTES ---------
@app.get("/", response_class=JSONResponse)
async def root():
    return {"status": "ok", "ready": True}

@app.get("/ui", response_class=HTMLResponse)
async def ui():
    return HTMLResponse(UI_HTML)

@app.get("/chat", response_class=HTMLResponse)
async def chat(pepper: str | None = Query(None)):
    # só entrega o HTML; o JS decide a pimenta via query
    return HTMLResponse(CHAT_HTML)

@app.get("/kb", response_class=JSONResponse)
async def kb():
    # recarrega em runtime para permitir editar estático sem reiniciar
    global KB, NAME_MAP
    KB = load_kb()
    NAME_MAP = {}
    for k, v in KB.items():
        NAME_MAP[k.lower()] = k
        for s in v.get("sinonimos", []):
            NAME_MAP[(s or "").lower()] = k
    return KB

@app.get("/manifest.webmanifest", response_class=Response)
async def manifest():
    return Response(MANIFEST_JSON, media_type="application/manifest+json")

@app.get("/sw.js", response_class=PlainTextResponse)
async def sw():
    return PlainTextResponse(SW_JS, media_type="application/javascript")

@app.get("/healthz")
async def healthz():
    return {"ok": True}

# --------- /predict (stub de exemplo) ---------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file or not file.filename:
        return {"ok": False, "error": "arquivo não enviado"}
    data = await file.read()
    if not data:
        return {"ok": False, "error": "arquivo vazio"}
    # Exemplo simples: retorna sempre Jalapeño (troque pelo seu pipeline real)
    return {
        "ok": True,
        "top_pred": {"classe": "Jalapeño", "conf": 0.92},
        # "image_url": "/static/annotated/ultima.png"
    }

# --------- run local ---------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
