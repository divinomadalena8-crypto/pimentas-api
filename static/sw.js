// sw.js — cache básico do PWA
const CACHE = "pimentas-v1";
const ASSETS = [
  "/",
  "/ui",
  "/manifest.webmanifest",
  "/static/pimenta-logo.png",
  "/static/pimenta-512.png",
  "/static/splash.png",
  "/static/pepper_info.json"
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  // Nunca interceptar /predict
  if (url.pathname.startsWith("/predict")) return;

  // network-first para páginas
  if (url.pathname === "/ui" || url.pathname.startsWith("/info")) {
    event.respondWith((async () => {
      try {
        const fresh = await fetch(event.request);
        const cache = await caches.open(CACHE);
        cache.put(event.request, fresh.clone());
        return fresh;
      } catch (e) {
        const cached = await caches.match(event.request);
        return cached || new Response("Offline", { status: 503 });
      }
    })());
    return;
  }

  // cache-first para assets
  event.respondWith((async () => {
    const cached = await caches.match(event.request);
    if (cached) return cached;
    try {
      const fresh = await fetch(event.request);
      const cache = await caches.open(CACHE);
      cache.put(event.request, fresh.clone());
      return fresh;
    } catch (e) {
      return new Response("Offline", { status: 503 });
    }
  })());
});
