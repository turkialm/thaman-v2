// THAMAN service worker — minimal, installability only
// No aggressive caching: ML API responses are dynamic and must not be stale
const CACHE = 'thaman-v2';
const STATIC = [
  '/ui', '/ui/style.css', '/ui/app.js',
  '/ui/charts.html', '/ui/batch.html', '/ui/embed.html',
  '/',
];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(STATIC).catch(() => {})));
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => clients.claim())
  );
});

// Network-first for API calls, cache-first for static assets
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (url.pathname.startsWith('/predict') || url.pathname.startsWith('/layers') ||
      url.pathname.startsWith('/nearby')  || url.pathname.startsWith('/health')) {
    return; // always network for API — no stale ML responses
  }
  e.respondWith(
    caches.match(e.request).then(cached => cached || fetch(e.request))
  );
});
