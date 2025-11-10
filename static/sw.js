/* Pimentas PWA SW — cache estático + app shell */
const VERSION = 'v4.0';
const STATIC_CACHE = `static-${VERSION}`;
const APP_SHELL = [
  '/', '/ui', '/info',
  '/static/manifest.webmanifest',
  '/static/pimenta-logo.png',
  '/static/pimenta-512.png',
  '/static/splash.png',
  '/static/pepper_info.json'
];

// Instala: pré-cache do shell
self.addEventListener('install', (evt) => {
  self.skipWaiting();
  evt.waitUntil(caches.open(STATIC_CACHE).then(c => c.addAll(APP_SHELL)));
});

// Ativa: limpa caches antigos
self.addEventListener('activate', (evt) => {
  evt.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== STATIC_CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

// Busca: cache-first para estáticos; network-first para páginas
self.addEventListener('fetch', (evt) => {
  const url = new URL(evt.request.url);

  // Nunca cacheie a inferência
  if (url.pathname.startsWith('/predict')) return;

  // Estáticos
  if (url.pathname.startsWith('/static/')) {
    evt.respondWith(
      caches.open(STATIC_CACHE).then(cache =>
        cache.match(evt.request).then(hit => hit || fetch(evt.request).then(r => {
          cache.put(evt.request, r.clone()); return r;
        }))
      )
    );
    return;
  }

  // Páginas do app (ui/info): network-first com fallback ao cache
  if (url.pathname === '/' || url.pathname === '/ui' || url.pathname === '/info') {
    evt.respondWith(
      fetch(evt.request).then(r => {
        const clone = r.clone();
        caches.open(STATIC_CACHE).then(c => c.put(evt.request, clone));
        return r;
      }).catch(() => caches.match(evt.request))
    );
  }
});
