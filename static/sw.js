/* Pimentas PWA SW - v1.0 (Corrigido) */
const CACHE_NAME = 'pimentas-cache-v1';

// O "App Shell" - tudo que o app precisa para carregar
// Verifique se todos os caminhos estão corretos
const APP_SHELL = [
  '/ui',
  '/info',
  '/static/manifest.webmanifest',    // O manifesto (na pasta /static/)
  '/static/icons/pimenta-192.png', // Ícone 1 (na pasta /static/icons/)
  '/static/icons/pimenta-512.png', // Ícone 2 (na pasta /static/icons/)
  '/static/icons/splash.png',      // Splash (na pasta /static/icons/)
  '/static/pepper_info.json'       // O arquivo de dados (na pasta /static/)
];

// 1. Instala o Service Worker
self.addEventListener('install', (event) => {
  console.log('SW: Instalando...');
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('SW: Cache aberto. Adicionando o App Shell.');
        // Adiciona todos os arquivos do APP_SHELL ao cache
        return cache.addAll(APP_SHELL);
      })
      .then(() => {
        // Força o novo service worker a se tornar ativo
        return self.skipWaiting();
      })
  );
});

// 2. Ativa o Service Worker (limpa caches antigos)
self.addEventListener('activate', (event) => {
  console.log('SW: Ativando...');
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        // Deleta todos os caches que não sejam o CACHE_NAME atual
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log('SW: Limpando cache antigo:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      // Toma controle da página imediatamente
      return self.clients.claim();
    })
  );
});

// 3. Intercepta os pedidos (Fetch)
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // *** A REGRA MAIS IMPORTANTE ***
  // Se for um pedido de API (POST) ou para o /predict,
  // NÃO use o cache. Vá direto para a rede.
  if (event.request.method === 'POST' || url.pathname.startsWith('/predict')) {
    // Apenas busca na rede (comportamento normal do navegador)
    return fetch(event.request);
  }

  // Para páginas (como /ui ou /info), tente a rede primeiro (Network-First).
  // Se falhar (offline), pegue do cache.
  if (url.pathname === '/ui' || url.pathname === '/info' || url.pathname === '/') {
    event.respondWith(
      fetch(event.request)
        .then((networkResponse) => {
          // Se funcionou, salve uma cópia no cache e retorne
          const responseClone = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, responseClone);
          });
          return networkResponse;
        })
        .catch(() => {
          // Se a rede falhou, tente pegar a versão do cache
          return caches.match(event.request);
        })
    );
    return; // Importante para parar a execução aqui
  }

  // Para assets estáticos (imagens, JSON, etc.),
  // tente o cache primeiro (Cache-First). Se não achar, vá para a rede.
  event.respondWith(
    caches.match(event.request)
      .then((cachedResponse) => {
        if (cachedResponse) {
          // Se achou no cache, retorna
          return cachedResponse;
        }
        // Se não achou, vá para a rede
        return fetch(event.request).then((networkResponse) => {
          // E salve uma cópia no cache para a próxima vez
          const responseClone = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, responseClone);
          });
          return networkResponse;
        });
      })
  );
});
