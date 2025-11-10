# main.py — PWA + Splash + API simples
# ------------------------------------------------------------
# Requisitos (requirements.txt):
# fastapi==0.115.0
# uvicorn==0.30.0
# python-multipart==0.0.9
# (opcional para CORS/produção)
# ------------------------------------------------------------

import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, Response, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Pimentas PWA")

# CORS (deixe mais estrito se quiser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- UI (HTML) com Splash ----------
UI_HTML = r"""
<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Identificação de Pimentas</title>

  <!-- PWA -->
  <link rel="manifest" href="/manifest.webmanifest">
  <meta name="theme-color" content="#16a34a">
  <link rel="apple-touch-icon" href="/static/icons/pimenta-192.png">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">

  <style>
    :root{ --bg:#f7fafc; --card:#fff; --fg:#0f172a; --muted:#475569; --line:#e2e8f0; --accent:#16a34a; }
    *{box-sizing:border-box}
    html,body{ margin:0; background:var(--bg); color:var(--fg); font:400 16px/1.45 system-ui, -apple-system, Segoe UI, Roboto }
    .wrap{ max-width:980px; margin:auto; padding:20px 14px 72px }
    header{ display:flex; align-items:center; gap:10px }
    h1{ font-size:22px; margin:0 }
    .card{ background:var(--card); border:1px solid var(--line); border-radius:16px; padding:16px; box-shadow:0 4px 24px rgba(15,23,42,.06) }
    .row{ display:flex; gap:10px; flex-wrap:wrap; align-items:center }
    .btn{ appearance:none; border:1px solid var(--line); background:#fff; color:var(--fg); padding:10px 14px; border-radius:12px; cursor:pointer; font-weight:600 }
    .btn.accent{ background:var(--accent); border-color:var(--accent); color:#fff }
    .pill{ display:inline-block; padding:6px 10px; border-radius:999px; background:#eef2ff; border:1px solid #c7d2fe; color:#3730a3; font-size:12px }
    .imgwrap{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:8px }
    img,video,canvas{ max-width:100%; display:block; border-radius:10px }
    footer{ position:fixed; left:0; right:0; bottom:0; padding:10px 14px; background:#ffffffd9; border-top:1px solid var(--line); color:var(--muted); font-size:12px; text-align:center; backdrop-filter:saturate(140%) blur(6px) }

    /* SPLASH overlay usando static/splash.png */
    #splash{
      position:fixed; inset:0; z-index:9999;
      background: #fff url('/static/splash.png') no-repeat center 34%;
      background-size: min(72vmin, 520px);
      display:flex; align-items:flex-end; justify-content:center; padding:32px;
      transition: opacity .35s ease;
    }
    #splash.hide{ opacity:0; pointer-events:none; }
    #splash .bar{
      width:min(520px,80vw); height:14px; border-radius:999px;
      background:#e6f4ea; border:1px solid #cce9d6; overflow:hidden;
    }
    #splash .bar>i{
      display:block; width:35%; height:100%; background:#16a34a;
      animation:load 1.6s infinite;
    }
    @keyframes load{
      0%{ transform:translateX(-100%) }
      50%{ transform:translateX(30%) }
      100%{ transform:translateX(120%) }
    }
  </style>

  <script>
    // Service Worker
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('/sw.js').catch(()=>{});
    }
  </script>
</head>
<body>
  <!-- SPLASH -->
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
      </div>

      <p class="tip" style="color:#475569">As imagens são comprimidas antes do envio.</p>

      <div class="row" style="margin-top:10px">
        <div class="imgwrap" style="flex:1">
          <small style="color:#475569">Original</small>
          <video id="video" autoplay playsinline style="display:none"></video>
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
    let currentFile = null;
    const chip = document.getElementById('chip');
    const preview = document.getElementById('preview');
    function setStatus(t){ chip.textContent=t; }
    function hideSplash(){
      const s=document.getElementById('splash'); if(!s) return;
      s.classList.add('hide'); setTimeout(()=>s.remove(), 400);
    }
    function sleep(ms){ return new Promise(r=>setTimeout(r,ms)); }

    async function waitReady(){
      setStatus("Conectando…");
      try{
        const r = await fetch(API + "/", {cache:"no-store"});
        const d = await r.json();
        if(d.ready){ setStatus("Pronto"); hideSplash(); document.getElementById('btnSend').disabled = !currentFile; return; }
        setStatus("Aquecendo…");
      }catch(e){ setStatus("Sem conexão"); }
      await sleep(900); waitReady();
    }

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

    const inputGallery=document.getElementById('fileGallery');
    const inputCamera =document.getElementById('fileCamera');
    document.getElementById('btnPick').onclick = ()=>{ inputGallery.value=""; inputGallery.click(); };
    document.getElementById('btnCam').onclick  = ()=>{ inputCamera.value="";  inputCamera.click(); };
    inputGallery.onchange = ()=> useLocalFile(inputGallery.files?.[0]);
    inputCamera.onchange  = ()=> useLocalFile(inputCamera.files?.[0]);

    async function useLocalFile(f){
      if(!f) return;
      currentFile = await compressImage(f);
      preview.src = URL.createObjectURL(currentFile);
      preview.style.display="block";
      document.getElementById('video').style.display="none";
      document.getElementById('btnSend').disabled=false;
      document.getElementById('resumo').textContent="Imagem pronta para envio.";
    }

    document.getElementById('btnSend').onclick = async () => {
      if(!currentFile) return;
      document.getElementById('btnSend').disabled = true;
      document.getElementById('resumo').textContent = "Enviando...";
      try{
        const fd=new FormData(); fd.append("file", currentFile, currentFile.name||"image.jpg");
        const r=await fetch(API + "/predict", {method:"POST", body:fd});
        const d=await r.json();
        if(d.ok===false && d.warming_up){ document.getElementById('resumo').textContent="Aquecendo o modelo…"; return; }
        if(d.ok===false){ document.getElementById('resumo').textContent="Erro: " + (d.error||"desconhecido"); return; }
        document.getElementById('resumo').textContent = d.top_pred
          ? (`Pimenta: ${d.top_pred.classe} (${Math.round((d.top_pred.conf||0)*100)}%)`)
          : "Nenhuma pimenta detectada.";
        if(d.image_url){ document.getElementById('annotated').src = d.image_url; }
      }catch(e){
        document.getElementById('resumo').textContent = "Falha ao enviar.";
      }finally{
        document.getElementById('btnSend').disabled = false;
      }
    };

    // fallback
    setTimeout(hideSplash, 4000);
    waitReady();
  </script>
</body>
</html>
"""

# ---------- Manifest (PWA) ----------
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

# ---------- Service Worker ----------
SW_JS = """
self.addEventListener('install', e => {
  e.waitUntil(
    caches.open('pimentas-v1').then(c => c.addAll([
      '/ui',
      '/manifest.webmanifest',
      '/static/splash.png',
      '/static/pimenta-logo.png'
    ]))
  );
});
self.addEventListener('fetch', e => {
  e.respondWith(
    caches.match(e.request).then(r => r || fetch(e.request))
  );
});
""".strip()

# ---------- Static (garanta a pasta 'static' no projeto) ----------
os.makedirs("static/icons", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------- Rotas ----------
@app.get("/", response_class=JSONResponse)
async def root():
    # Endpoint de "pronto/vivo" para o splash decidir quando sumir
    return {"status": "ok", "ready": True}

@app.get("/ui", response_class=HTMLResponse)
async def ui():
    return HTMLResponse(UI_HTML)

@app.get("/manifest.webmanifest", response_class=Response)
async def manifest():
    return Response(MANIFEST_JSON, media_type="application/manifest+json")

@app.get("/sw.js", response_class=PlainTextResponse)
async def sw():
    return PlainTextResponse(SW_JS, media_type="application/javascript")

@app.get("/healthz")
async def healthz():
    return {"ok": True}

# ---------- /predict (stub amigável) ----------
# Troque este stub pelo seu pipeline real de detecção se quiser.
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file or not file.filename:
        return {"ok": False, "error": "arquivo não enviado"}

    # Apenas um stub para funcionamento imediato
    # Você pode salvar/processar a imagem e retornar a anotada via /static.
    data = await file.read()
    if len(data) == 0:
        return {"ok": False, "error": "arquivo vazio"}

    # Exemplos de resposta esperada pelo front
    fake = {
        "ok": True,
        "top_pred": {"classe": "Habanero", "conf": 0.91},
        # se gerar imagem anotada, devolva a URL:
        # "image_url": "/static/annotated/ultima.png"
    }
    return fake

# ---------- Exec local ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
