// Versão do cache (troque ao publicar para forçar atualização)
const CACHE_NAME = 'pimentas-v1';

// “App shell”: páginas e assets estáticos
const APP_SHELL = [
  '/ui/',                          // sua UI
  '/static/pepper_info.json',      // base local do chat (se existir)
  '/static/pimenta-logo.png'       // seu logo (se existir)
];

// Instala e prepara cache
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL))
  );
  self.skipWaiting();
});

// Assume controle imediatamente
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Estratégia:
//  - POSTs/PUTs/PATCH/DELETE → passam direto (não cacheia)
//  - Chamadas de API (ex.: /predict) → sempre rede (network-first)
//  - Demais GETs → cache-first com fallback à rede
self.addEventListener('fetch', (event) => {
  const req = event.request;
  const url = new URL(req.url);

  // Nunca intercepta métodos não-GET
  if (req.method !== 'GET') return;

  // Evita interferir em endpoints dinâmicos
  if (url.pathname.startsWith('/predict') || url.pathname.startsWith('/warmup')) {
    event.respondWith(fetch(req).catch(() => new Response('Offline', {status: 503})));
    return;
  }

  // Cache-first para estáticos e /ui
  event.respondWith(
    caches.match(req).then(cached => {
      if (cached) return cached;
      return fetch(req).then(resp => {
        // Só cacheia respostas 200 e do mesmo host
        if (resp && resp.status === 200 && url.origin === location.origin) {
          const clone = resp.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(req, clone));
        }
        return resp;
      });
    })
  );
});
