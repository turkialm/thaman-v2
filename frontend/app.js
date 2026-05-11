/**
 * THAMAN — Frontend Application
 * ==============================
 * Handles the interactive map, property form, and API integration.
 *
 * Flow:
 *   1. User clicks map → lat/lng captured, marker placed, borough auto-guessed
 *   2. User fills form and clicks "Estimate Price"
 *   3. POST /predict → API returns price + SHAP drivers + spatial features
 *   4. Results rendered: price card, SHAP bars, spatial grid
 */

'use strict';

// ── API Base URL (same origin since served by FastAPI) ────────────────
const API_BASE = '';   // e.g. '' means same-origin; change to 'http://localhost:8000' for dev

// ── Language (declared early — used before i18n block) ────────────────
let currentLang = localStorage.getItem('thamanLang') || 'en';

// ── Last prediction (used to compare against nearby comps) ────────────
let _lastPrediction = null;  // { price, sqft }

// ── NTA choropleth layers ─────────────────────────────────────────────
let _ntaGeoJSON    = null;   // raw GeoJSON loaded once
let _activeLayer   = null;   // current Leaflet GeoJSON layer on map
let _activeMetric  = 'none'; // which layer is showing

// ── Layer metadata ────────────────────────────────────────────────────
const LAYER_META = {
  income:  { key: 'median_income_nta',       label: 'Median Income',   unit: '/yr',    palette: 'green',  fmt: v => `$${(v/1000).toFixed(0)}k` },
  crime:   { key: 'crime_rate_nta',          label: 'Crime Rate',      unit: '/1k res',palette: 'red',    fmt: v => v.toFixed(1) },
  noise:   { key: 'noise_density_nta',       label: 'Noise Level',     unit: '/1k res',palette: 'orange', fmt: v => v.toFixed(1) },
  air:     { key: 'pm25_mean',               label: 'PM2.5 Air Quality',unit: 'µg/m³', palette: 'purple', fmt: v => v.toFixed(2) },
  trees:   { key: 'tree_count_200m',         label: 'Tree Cover',      unit: 'trees/200m',palette:'green',fmt: v => `~${Math.round(v)}` },
  hotness: { key: 'price_appreciation',      label: 'Market Hotness',  unit: '% gain', palette: 'blue',   fmt: v => `${v > 0 ? '+' : ''}${(v*100).toFixed(0)}%` },
};

// ── Map init ──────────────────────────────────────────────────────────
const map = L.map('map', {
  center:      [40.7128, -74.0060],   // NYC
  zoom:        11,
  zoomControl: true,
});

// CartoDB Voyager — cleaner, more professional than raw OSM (free, no API key)
L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
  attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors © <a href="https://carto.com/attributions">CARTO</a>',
  subdomains: 'abcd',
  maxZoom: 19,
}).addTo(map);

// ── NYC Borough boundary polygons ─────────────────────────────────────
// Simplified polygons used for (1) visual border on map, (2) borough auto-detection
const BOROUGHS = {
  '1': { name:'Manhattan',    nameAr:'مانهاتن',      color:'#3b82f6', center:[40.790,-73.965],
    coords:[[40.7001,-74.0168],[40.7551,-74.0080],[40.8007,-73.9579],
            [40.8691,-73.9220],[40.8799,-73.9072],[40.8610,-73.9108],
            [40.7960,-73.9314],[40.7487,-73.9714],[40.7050,-73.9764],
            [40.6997,-74.0136]] },
  '2': { name:'Bronx',        nameAr:'برونكس',        color:'#10b981', center:[40.855,-73.866],
    coords:[[40.7960,-73.9314],[40.8180,-73.9265],[40.8556,-73.9137],
            [40.8799,-73.9072],[40.9001,-73.9100],[40.9179,-73.9093],
            [40.9161,-73.8455],[40.9095,-73.8160],[40.8730,-73.7810],
            [40.8054,-73.8280],[40.7962,-73.8615]] },
  '3': { name:'Brooklyn',     nameAr:'بروكلين',       color:'#f59e0b', center:[40.638,-73.944],
    coords:[[40.7025,-73.9720],[40.7150,-73.9200],[40.6977,-73.8786],
            [40.6756,-73.8613],[40.6420,-73.8555],[40.5990,-73.8080],
            [40.5770,-73.9490],[40.5730,-74.0167],[40.6197,-74.0338],
            [40.6456,-74.0319],[40.6920,-74.0207]] },
  '4': { name:'Queens',       nameAr:'كوينز',         color:'#8b5cf6', center:[40.700,-73.820],
    coords:[[40.7776,-73.9179],[40.8073,-73.8300],[40.7661,-73.7098],
            [40.7273,-73.7009],[40.6620,-73.7329],[40.5880,-73.7485],
            [40.5430,-73.7600],[40.5800,-73.8170],[40.5990,-73.8080],
            [40.6420,-73.8555],[40.6756,-73.8613],[40.6977,-73.8786],
            [40.7150,-73.9200],[40.7280,-73.9480]] },
  '5': { name:'Staten Island', nameAr:'ستاتن آيلاند', color:'#ef4444', center:[40.579,-74.151],
    coords:[[40.6456,-74.0319],[40.6456,-74.1320],[40.5775,-74.1899],
            [40.5100,-74.2498],[40.4780,-74.2190],[40.4965,-74.1380],
            [40.5540,-74.0580],[40.6197,-74.0338]] },
};

// Ray-casting point-in-polygon check
function pointInPolygon(lat, lng, coords) {
  let inside = false;
  for (let i = 0, j = coords.length - 1; i < coords.length; j = i++) {
    const [xi, yi] = coords[i];
    const [xj, yj] = coords[j];
    if (((yi > lng) !== (yj > lng)) &&
        (lat < (xj - xi) * (lng - yi) / (yj - yi) + xi)) inside = !inside;
  }
  return inside;
}

// Returns borough code ('1'–'5') or null if outside NYC
function getBoroughCode(lat, lng) {
  for (const [code, b] of Object.entries(BOROUGHS)) {
    if (pointInPolygon(lat, lng, b.coords)) return code;
  }
  return null;
}

// Show a brief red error popup on the map
function showMapError(lat, lng, msg) {
  const p = L.popup({ closeButton:false, className:'oob-popup', autoPan:false })
    .setLatLng([lat, lng])
    .setContent(`<span>⚠️ ${msg}</span>`)
    .openOn(map);
  setTimeout(() => map.closePopup(p), 2600);
}

// Borough polygon data kept for point-in-polygon detection only (no visual border)
let _boroughLabelEls = {};

// ── Real NYC boundary — loaded once, used for both mask and validation ──
// coords: MultiPolygon [ [ [outerRing], [hole?], ... ], ... ] in GeoJSON [lng,lat] order
let nycBoundaryCoords = null;

// Ray-cast inside a single GeoJSON ring ([lng,lat] pairs)
function pointInGeoRing(lng, lat, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i]; // xi=lng, yi=lat
    const [xj, yj] = ring[j];
    if (((yi > lat) !== (yj > lat)) &&
        (lng < (xj - xi) * (lat - yi) / (yj - yi) + xi)) inside = !inside;
  }
  return inside;
}

// Returns true if (lat,lng) is inside the real NYC boundary (handles holes).
// Falls back to rough BOROUGHS check while GeoJSON is still loading.
function isInNYC(lat, lng) {
  if (!nycBoundaryCoords) return getBoroughCode(lat, lng) !== null;
  for (const polygon of nycBoundaryCoords) {
    if (pointInGeoRing(lng, lat, polygon[0])) {          // inside outer ring
      let inHole = false;
      for (let h = 1; h < polygon.length; h++) {
        if (pointInGeoRing(lng, lat, polygon[h])) { inHole = true; break; }
      }
      if (!inHole) return true;
    }
  }
  return false;
}

// ── Inverse mask: red overlay over whole world EXCEPT real NYC boundary ──
// Fetches nyc_boundary.geojson (MultiPolygon dissolved from NTA data).
// Each NYC land-mass polygon becomes a "hole" in the world rectangle.
fetch('/ui/nyc_boundary.geojson')
  .then(r => r.json())
  .then(data => {
    // GeoJSON uses [lng, lat]; Leaflet uses [lat, lng]
    const toLflt = ring => ring.map(([lng, lat]) => [lat, lng]);

    const geom = data.geometry || (data.type === 'MultiPolygon' ? data : null);
    nycBoundaryCoords = geom ? geom.coordinates : null;   // store for isInNYC()
    const coords = nycBoundaryCoords || [];

    // Build rings: world outer rect + one hole per NYC land-mass polygon
    const rings = [
      [[ 90, -180], [ 90,  180], [-90,  180], [-90, -180]], // world
      ...coords.map(poly => toLflt(poly[0])),                // NYC holes
    ];

    L.polygon(rings, {
      stroke:      false,
      fillColor:   '#ef4444',
      fillOpacity: 0.25,
      interactive: false,
    }).addTo(map);
  })
  .catch(() => {
    // Fallback: simple bounding-box hole if GeoJSON fails to load
    L.polygon([
      [[ 90, -180], [ 90,  180], [-90,  180], [-90, -180]],
      [[40.477399, -74.25909], [40.477399, -73.700272],
       [40.917577, -73.700272], [40.917577, -74.25909]],
    ], { stroke:false, fillColor:'#ef4444', fillOpacity:0.25, interactive:false }).addTo(map);
  });

// ── NTA choropleth layer system ───────────────────────────────────────
fetch(`${API_BASE}/layers/nta`)
  .then(r => r.ok ? r.json() : null)
  .then(data => {
    if (data) {
      _ntaGeoJSON = data;
      document.getElementById('layerBar').style.display = 'flex';
    }
  })
  .catch(() => {});

// Colour palettes: each is [light, dark] for interpolation
const _PALETTES = {
  green:  ['#d1fae5', '#065f46'],
  red:    ['#fee2e2', '#991b1b'],
  orange: ['#ffedd5', '#9a3412'],
  purple: ['#ede9fe', '#4c1d95'],
  blue:   ['#dbeafe', '#1e3a8a'],
};

function _hexToRgb(hex) {
  const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
  return [r, g, b];
}
function _rgbToHex(r, g, b) {
  return '#' + [r,g,b].map(v => Math.round(v).toString(16).padStart(2,'0')).join('');
}
function colorScale(val, min, max, palette) {
  if (val === null || val === undefined || isNaN(val)) return '#e5e7eb';
  const t = max === min ? 0.5 : Math.max(0, Math.min(1, (val - min) / (max - min)));
  const [c0, c1] = (_PALETTES[palette] || _PALETTES.blue).map(_hexToRgb);
  return _rgbToHex(c0[0] + t*(c1[0]-c0[0]), c0[1] + t*(c1[1]-c0[1]), c0[2] + t*(c1[2]-c0[2]));
}

function showLayer(metricId) {
  if (_activeLayer) { map.removeLayer(_activeLayer); _activeLayer = null; }
  _activeMetric = metricId;

  const legend = document.getElementById('layerLegend');
  if (metricId === 'none' || !_ntaGeoJSON) { legend.style.display = 'none'; return; }

  const meta   = LAYER_META[metricId];
  if (!meta) return;

  const values = _ntaGeoJSON.features
    .map(f => f.properties[meta.key])
    .filter(v => v !== null && v !== undefined && !isNaN(v));
  if (!values.length) return;

  const min = Math.min(...values), max = Math.max(...values);

  _activeLayer = L.geoJSON(_ntaGeoJSON, {
    style: feat => {
      const v = feat.properties[meta.key];
      return {
        fillColor:   colorScale(v, min, max, meta.palette),
        fillOpacity: 0.60,
        weight:      0.8,
        color:       '#ffffff',
        opacity:     0.5,
      };
    },
    onEachFeature: (feat, layer) => {
      const v = feat.properties[meta.key];
      const fmtVal = (v !== null && v !== undefined) ? meta.fmt(v) : 'N/A';
      layer.bindTooltip(
        `<b>${feat.properties.ntaname || ''}</b><br>${meta.label}: ${fmtVal} ${meta.unit}`,
        { sticky: true, opacity: 0.92 }
      );
      layer.on({
        mouseover: e => { e.target.setStyle({ weight: 2, color: '#1d4ed8', fillOpacity: 0.82 }); e.target.bringToFront(); },
        mouseout:  e => { if (_activeLayer) _activeLayer.resetStyle(e.target); },
      });
    },
  }).addTo(map);

  // Update legend
  const [c0, c1] = (_PALETTES[meta.palette] || _PALETTES.blue);
  document.getElementById('legendMin').textContent     = meta.fmt(min);
  document.getElementById('legendMax').textContent     = meta.fmt(max);
  document.getElementById('legendLabel').textContent   = meta.label;
  document.getElementById('legendGradient').style.background =
    `linear-gradient(to right, ${c0}, ${c1})`;
  legend.style.display = 'flex';
}

// Layer bar click handler
document.getElementById('layerBar').addEventListener('click', e => {
  const btn = e.target.closest('.layer-btn');
  if (!btn) return;
  document.querySelectorAll('.layer-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  showLayer(btn.dataset.layer);
});

// ── Pin marker (emoji-based, no image dependency) ──────────────────────
let lastValidPos = null;   // last accepted pin position (for snap-back on drag)

// Pin marker (emoji-based, no image dependency)
const pinIcon = L.divIcon({
  html:      '<div class="pin-icon pin-drop">📍</div>',
  iconSize:  [32, 32],
  iconAnchor:[16, 28],
  className: '',
});

let marker = null;

// ── DOM References ────────────────────────────────────────────────────
const locationText   = document.getElementById('locationText');
const latInput       = document.getElementById('latitude');
const lonInput       = document.getElementById('longitude');
const boroughSel     = document.getElementById('borough');
const bldgHidden     = document.getElementById('bldgclass');        // hidden value
const bldgSearchInput= document.getElementById('bldgSearchInput');  // visible text
const bldgDropdown   = document.getElementById('bldgDropdown');
const submitBtn      = document.getElementById('submitBtn');
const btnText        = document.getElementById('btnText');
const spinner        = document.getElementById('spinner');
const mapHint        = document.getElementById('mapHint');
const advancedToggle = document.getElementById('advancedToggle');
const advancedPanel  = document.getElementById('advancedPanel');
const advancedArrow  = document.getElementById('advancedArrow');
const predictForm    = document.getElementById('predictForm');

// Result elements
const resultCard     = document.getElementById('resultCard');
const shapCard       = document.getElementById('shapCard');
const spatialCard    = document.getElementById('spatialCard');
const priceMain      = document.getElementById('priceMain');
const priceRange     = document.getElementById('priceRange');
const priceContext   = document.getElementById('priceContext');
const tierBadge      = document.getElementById('tierBadge');
const confBarWrap    = document.getElementById('confBarWrap');
const confLow        = document.getElementById('confLow');
const confHigh       = document.getElementById('confHigh');
const confFill       = document.getElementById('confFill');
const confMarker     = document.getElementById('confMarker');
const shapBars       = document.getElementById('shapBars');
const spatialGrid    = document.getElementById('spatialGrid');

// ── Address search (Nominatim geocoding — free, no API key) ───────────
const addrInput = document.getElementById('addrInput');
const addrBtn   = document.getElementById('addrBtn');
const addrError = document.getElementById('addrError');

function showAddrError(msg) {
  addrError.textContent = msg;
  addrError.style.display = 'block';
  setTimeout(() => { addrError.style.display = 'none'; }, 4000);
}

async function geocodeAddress() {
  const q = addrInput.value.trim();
  if (!q) return;

  addrBtn.classList.add('loading');
  addrError.style.display = 'none';

  try {
    // Restrict to NYC bounding box to avoid false matches
    const url = `https://nominatim.openstreetmap.org/search?` +
      `q=${encodeURIComponent(q + ', New York City')}&format=json&limit=1` +
      `&countrycodes=us&bounded=1&viewbox=-74.26,40.47,-73.70,40.92`;

    const res  = await fetch(url, { headers: { 'Accept-Language': 'en' } });
    const data = await res.json();

    if (!data || data.length === 0) {
      showAddrError(TR[currentLang].addrNotFound);
      return;
    }

    const lat = parseFloat(data[0].lat);
    const lng = parseFloat(data[0].lon);

    if (!isInNYC(lat, lng)) {
      showAddrError(TR[currentLang].addrOutOfNYC);
      return;
    }

    // Place pin — reuse same logic as map click
    latInput.value = lat.toFixed(6);
    lonInput.value = lng.toFixed(6);
    lastValidPos   = [lat, lng];
    locationText.textContent = `${lat.toFixed(5)}, ${lng.toFixed(5)}`;
    locationText.classList.add('selected');

    if (marker) {
      marker.setLatLng([lat, lng]);
    } else {
      marker = L.marker([lat, lng], { icon: pinIcon, draggable: true }).addTo(map);
      marker.on('dragend', (ev) => {
        const pos = ev.target.getLatLng();
        if (!isInNYC(pos.lat, pos.lng)) {
          if (lastValidPos) marker.setLatLng(lastValidPos);
          showMapError(pos.lat, pos.lng, TR[currentLang].outOfNYC);
          return;
        }
        lastValidPos = [pos.lat, pos.lng];
        latInput.value = pos.lat.toFixed(6);
        lonInput.value = pos.lng.toFixed(6);
        locationText.textContent = `${pos.lat.toFixed(5)}, ${pos.lng.toFixed(5)}`;
        const bc = getBoroughCode(pos.lat, pos.lng);
        if (bc) boroughSel.value = bc;
      });
    }

    const bc = getBoroughCode(lat, lng);
    if (bc) boroughSel.value = bc;

    submitBtn.disabled = false;
    btnText.textContent = '🔍  ' + TR[currentLang].estimateBtn;
    mapHint.classList.add('hidden');

    map.flyTo([lat, lng], 15, { duration: 1.2, easeLinearity: 0.4 });

    // Show brief tooltip with found address
    if (marker) {
      marker.bindPopup(
        `<div class="map-popup"><small>📍 ${data[0].display_name.split(',').slice(0,3).join(',')}</small></div>`,
        { maxWidth: 220 }
      ).openPopup();
      setTimeout(() => marker.closePopup(), 3000);
    }

  } catch (err) {
    showAddrError(TR[currentLang].addrError);
    console.error('Geocoding error:', err);
  } finally {
    addrBtn.classList.remove('loading');
  }
}

addrBtn.addEventListener('click', geocodeAddress);
addrInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); geocodeAddress(); }
});

// ── Building class searchable dropdown ────────────────────────────────
let _bldgClasses = [];   // [{code, desc}]
let _bldgDescs   = {};   // code → description (from /bldgclasses common_examples)

async function loadBldgClasses() {
  try {
    const res  = await fetch(`${API_BASE}/bldgclasses`);
    const data = await res.json();
    _bldgDescs   = data.common_examples || {};
    _bldgClasses = data.bldgclasses.map(code => ({
      code,
      desc: _bldgDescs[code] || code,
    }));
  } catch (e) {
    // Fallback: common codes inline
    const common = {
      A1:'Two-story detached', A2:'One-story detached', A5:'Attached rowhouse',
      A7:'Mansion / townhouse', B1:'Two family brick', B2:'Two family frame',
      C0:'Three families', C2:'Five to six families', C4:'Old law tenement',
      C6:'Cooperative walk-up', D1:'Elevator apt (semi-fireproof)',
      D4:'Elevator apt building', R1:'Condo unit (elevator)', R4:'Condo unit (walk-up)',
      S1:'1-family + commercial', S2:'2-family + commercial',
    };
    _bldgDescs   = common;
    _bldgClasses = Object.entries(common).map(([code, desc]) => ({ code, desc }));
  }
}

function showBldgDropdown(query) {
  const q = query.toLowerCase().trim();
  const matches = q
    ? _bldgClasses.filter(b =>
        b.code.toLowerCase().startsWith(q) ||
        b.desc.toLowerCase().includes(q)
      ).slice(0, 12)
    : _bldgClasses.slice(0, 12);

  if (!matches.length) { bldgDropdown.style.display = 'none'; return; }

  bldgDropdown.innerHTML = matches.map(b =>
    `<div class="dropdown-item" data-code="${b.code}">
       <span class="dd-code">${b.code}</span>
       <span class="dd-desc">${b.desc}</span>
     </div>`
  ).join('');
  bldgDropdown.style.display = 'block';

  bldgDropdown.querySelectorAll('.dropdown-item').forEach(el => {
    el.addEventListener('mousedown', (ev) => {
      ev.preventDefault();
      const code = el.dataset.code;
      bldgHidden.value    = code;
      bldgSearchInput.value = `${code} — ${_bldgDescs[code] || code}`;
      bldgSearchInput.classList.remove('invalid');
      bldgDropdown.style.display = 'none';
      // Clear card selection since advanced search takes precedence
      document.querySelectorAll('.bldgtype-card').forEach(c => c.classList.remove('selected'));
    });
  });
}

bldgSearchInput.addEventListener('input', () => {
  bldgHidden.value = '';   // clear until user picks
  showBldgDropdown(bldgSearchInput.value);
});

bldgSearchInput.addEventListener('focus', () => {
  showBldgDropdown(bldgSearchInput.value);
});

bldgSearchInput.addEventListener('blur', () => {
  setTimeout(() => { bldgDropdown.style.display = 'none'; }, 150);
});

// Close on outside click
document.addEventListener('click', (e) => {
  if (!e.target.closest('#bldgSearchWrap')) {
    bldgDropdown.style.display = 'none';
  }
});

loadBldgClasses();

// ── Building type card selector ───────────────────────────────────────
function initBldgTypeCards() {
  const cards = document.querySelectorAll('.bldgtype-card');
  const grid  = document.getElementById('bldgTypeGrid');

  cards.forEach(card => {
    card.addEventListener('click', () => {
      cards.forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');
      bldgHidden.value = card.dataset.code;
      bldgSearchInput.value = '';
      grid.classList.remove('invalid');
    });
  });

  document.getElementById('bldgAdvToggle').addEventListener('click', () => {
    const panel = document.getElementById('bldgAdvPanel');
    panel.classList.toggle('open');
    // Update the arrow character only (first text node), preserve inner span
    document.getElementById('bldgAdvToggle').childNodes[0].textContent =
      panel.classList.contains('open') ? '▼ ' : '▶ ';
  });
}

initBldgTypeCards();

// ── Map click ─────────────────────────────────────────────────────────
map.on('click', (e) => {
  const { lat, lng } = e.latlng;

  // Reject clicks outside the real NYC boundary (uses GeoJSON, falls back to polygons)
  if (!isInNYC(lat, lng)) {
    showMapError(lat, lng, TR[currentLang].outOfNYC);
    return;
  }

  // Update hidden inputs
  latInput.value = lat.toFixed(6);
  lonInput.value = lng.toFixed(6);
  lastValidPos   = [lat, lng];

  // Update location display
  locationText.textContent = `${lat.toFixed(5)}, ${lng.toFixed(5)}`;
  locationText.classList.add('selected');

  // Place/move marker
  if (marker) {
    marker.setLatLng([lat, lng]);
  } else {
    marker = L.marker([lat, lng], { icon: pinIcon, draggable: true }).addTo(map);

    // Draggable marker: snap back if dragged outside NYC
    marker.on('dragend', (ev) => {
      const pos = ev.target.getLatLng();
      if (!isInNYC(pos.lat, pos.lng)) {
        if (lastValidPos) marker.setLatLng(lastValidPos);
        showMapError(pos.lat, pos.lng, TR[currentLang].outOfNYC);
        return;
      }
      lastValidPos = [pos.lat, pos.lng];
      latInput.value = pos.lat.toFixed(6);
      lonInput.value = pos.lng.toFixed(6);
      locationText.textContent = `${pos.lat.toFixed(5)}, ${pos.lng.toFixed(5)}`;
      const bc = getBoroughCode(pos.lat, pos.lng);
      if (bc) boroughSel.value = bc;   // auto-update borough on drag
    });
  }

  // Auto-set borough from rough polygons (best-effort; user can correct manually)
  const boroughCode = getBoroughCode(lat, lng);
  if (boroughCode) boroughSel.value = boroughCode;

  // Enable button, hide map hint
  submitBtn.disabled = false;
  btnText.textContent = '🔍  ' + TR[currentLang].estimateBtn;
  mapHint.classList.add('hidden');
});

// ── Map moveend: refresh sale bubbles (debounced) ─────────────────────
map.on('moveend', () => {
  clearTimeout(_salesTimer);
  _salesTimer = setTimeout(fetchSalesForView, 600);
});

// Initial bubble load
fetchSalesForView();

// ── Geolocation button ────────────────────────────────────────────────
document.getElementById('geolocBtn').addEventListener('click', () => {
  if (!navigator.geolocation) return;
  navigator.geolocation.getCurrentPosition(
    p => map.flyTo([p.coords.latitude, p.coords.longitude], 15, { duration: 1.0 }),
    () => {},
    { timeout: 6000 }
  );
});

// ── Advanced toggle ───────────────────────────────────────────────────
advancedToggle.addEventListener('click', () => {
  advancedPanel.classList.toggle('open');
  advancedArrow.classList.toggle('open');
});

// ── Form validation helpers ───────────────────────────────────────────
function getNum(id) {
  const v = document.getElementById(id).value.trim();
  return v === '' ? null : parseFloat(v);
}

function getString(id) {
  const v = document.getElementById(id).value.trim();
  return v === '' ? null : v;
}

function markInvalid(id) {
  const el = document.getElementById(id);
  el.classList.add('invalid');
  el.addEventListener('input', () => el.classList.remove('invalid'), { once: true });
}

// ── Format helpers ────────────────────────────────────────────────────
function fmt$(n) {
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000)     return `$${(n / 1_000).toFixed(0)}K`;
  return `$${n.toLocaleString()}`;
}

function fmtDist(m) {
  return m >= 1000 ? `${(m/1000).toFixed(1)} km` : `${Math.round(m)} m`;
}

// ── Bubble price label ($850k / $1.2M) ───────────────────────────────
function formatBubblePrice(n) {
  if (n >= 1e6) return '$' + (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e5) return '$' + Math.round(n / 1e3) + 'k';
  return '$' + n.toLocaleString();
}

// ── Bubble colour: neutral, or green/red relative to prediction ───────
function bubbleColor(salePrice) {
  if (!_lastPrediction) return '#4b6bfb';
  const r = salePrice / _lastPrediction.price;
  if (r >= 1.05) return '#dc2626';
  if (r <= 0.95) return '#059669';
  return '#6366f1';
}

function buildSaleIcon(price) {
  return L.divIcon({
    html:       `<div class="sale-bubble" style="background:${bubbleColor(price)}">${formatBubblePrice(price)}</div>`,
    className:  '',
    iconAnchor: [20, 10],
  });
}

function radiusForZoom(z) {
  if (z <= 11) return 2000;
  if (z <= 13) return 900;
  if (z >= 15) return 400;
  return 600;
}

// ── Render sale bubbles on map ────────────────────────────────────────
function renderSalesBubbles(sales) {
  if (_salesCluster) map.removeLayer(_salesCluster);
  _salesCluster = L.markerClusterGroup({
    maxClusterRadius:     40,
    showCoverageOnHover:  false,
    iconCreateFunction: c => L.divIcon({
      html:       `<div class="cluster-bubble">${c.getChildCount()}</div>`,
      className:  '',
      iconAnchor: [18, 18],
    }),
  });
  (sales || []).forEach(s => {
    if (!s.latitude || !s.longitude) return;
    const psf = s.gross_square_feet > 0
      ? `$${Math.round(s.sale_price / s.gross_square_feet).toLocaleString()}/sqft` : '';
    const m = L.marker([s.latitude, s.longitude], { icon: buildSaleIcon(s.sale_price) });
    let vsBadge = '';
    if (_lastPrediction && _lastPrediction.price) {
      const pct = ((s.sale_price - _lastPrediction.price) / _lastPrediction.price * 100);
      const sign = pct >= 0 ? '+' : '';
      const cls  = pct >= 5 ? 'delta-above' : pct <= -5 ? 'delta-below' : 'delta-neutral';
      vsBadge = `<span class="delta-badge ${cls}" style="margin-left:4px">${sign}${pct.toFixed(0)}% vs est.</span>`;
    }
    m.bindPopup(
      `<div class="sale-popup">
        <strong>${formatBubblePrice(s.sale_price)}${vsBadge}</strong>
        <div class="sale-popup-addr">${s.address || ''}</div>
        <div class="sale-popup-meta">${s.bldgclass || ''} · ${(s.gross_square_feet || 0).toLocaleString()} sqft${psf ? ' · ' + psf : ''}</div>
        <div class="sale-popup-date">Sold ${s.sale_date || ''}</div>
      </div>`,
      { maxWidth: 220 }
    );
    _salesCluster.addLayer(m);
  });
  map.addLayer(_salesCluster);
}

// ── Fetch recent sales for current map view ───────────────────────────
async function fetchSalesForView() {
  const c = map.getCenter(), z = map.getZoom();
  try {
    const r = await fetch(`${API_BASE}/nearby?lat=${c.lat}&lon=${c.lng}&radius_m=${radiusForZoom(z)}&limit=25`);
    if (!r.ok) return;
    const d = await r.json();
    renderSalesBubbles(d.nearby || []);
  } catch (_) {}
}

// ── Submit ─────────────────────────────────────────────────────────────
predictForm.addEventListener('submit', async (e) => {
  e.preventDefault();

  // Gather + validate
  const lat      = parseFloat(latInput.value);
  const lon      = parseFloat(lonInput.value);
  const borough  = getNum('borough');
  const bldg     = bldgHidden.value.trim();
  const sqft     = getNum('gross_square_feet');
  const age      = getNum('building_age');
  const floors   = getNum('numfloors');
  const units    = getNum('residential_units');

  let valid = true;
  if (!lat || !lon)    { valid = false; }
  if (!borough)        { markInvalid('borough'); valid = false; }
  if (!bldg)           { bldgSearchInput.classList.add('invalid'); document.getElementById('bldgTypeGrid').classList.add('invalid'); valid = false; }
  if (!sqft || sqft <= 0) { markInvalid('gross_square_feet'); valid = false; }
  if (age == null || age < 0) { markInvalid('building_age'); valid = false; }
  if (!floors || floors <= 0) { markInvalid('numfloors'); valid = false; }
  if (units == null || units < 0) { markInvalid('residential_units'); valid = false; }

  if (!valid) return;

  // Build payload
  const payload = {
    latitude:          lat,
    longitude:         lon,
    borough:           Math.round(borough),
    bldgclass:         bldg,
    gross_square_feet: sqft,
    building_age:      Math.round(age),
    numfloors:         floors,
    residential_units: Math.round(units),
  };

  // Optional fields
  const landSqft   = getNum('land_square_feet');
  const priorPrice = getNum('prior_sale_price');
  const renov      = getNum('renovated_since_2018');
  const saleYear   = getNum('sale_year');
  const saleMonth  = getNum('sale_month');
  if (landSqft   != null)  payload.land_square_feet     = landSqft;
  if (priorPrice != null)  payload.prior_sale_price      = priorPrice;
  if (renov      != null)  payload.renovated_since_2018  = Math.round(renov);
  if (saleYear   != null)  payload.sale_year             = Math.round(saleYear);
  if (saleMonth  != null)  payload.sale_month            = Math.round(saleMonth);
  if (priorPrice != null)  payload.has_prior_sale        = 1;

  // Loading state
  setLoading(true);
  hideResults();

  try {
    const res  = await fetch(`${API_BASE}/predict`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `API error ${res.status}`);
    }

    const data = await res.json();
    renderResults(data);
    fetchNearby(lat, lon, sqft);

  } catch (err) {
    console.error(err);
    alert(`Prediction failed: ${err.message}\n\nMake sure the API server is running at http://localhost:8000`);
  } finally {
    setLoading(false);
  }
});

// ── Loading state (with skeleton cards) ───────────────────────────────
function setLoading(on) {
  submitBtn.disabled = on;
  btnText.textContent = on ? TR[currentLang].estimating : '🔍 ' + TR[currentLang].estimateBtn;
  spinner.style.display = on ? 'inline-block' : 'none';
  ['skeletonResult','skeletonShap','skeletonSpatial'].forEach(id => {
    document.getElementById(id).style.display = on ? 'block' : 'none';
  });
}

function hideResults() {
  resultCard.style.display  = 'none';
  shapCard.style.display    = 'none';
  spatialCard.style.display = 'none';
  document.getElementById('marketCard').style.display  = 'none';
  document.getElementById('avmQcRow').style.display    = 'none';
  _compsLoaded = false;
}

// ── Render results ────────────────────────────────────────────────────
function animatePrice(el, target) {
  const start = target * 0.72;
  const dur   = 600;
  const t0    = performance.now();
  function tick(now) {
    const p    = Math.min((now - t0) / dur, 1);
    const ease = 1 - Math.pow(1 - p, 3);
    el.textContent = '$' + Math.round(start + (target - start) * ease).toLocaleString();
    if (p < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function renderResults(data) {
  _lastPrediction = { price: data.predicted_price, sqft: null };  // sqft filled below

  // Fly to pin
  const lat = parseFloat(latInput.value), lng = parseFloat(lonInput.value);
  if (lat && lng) map.flyTo([lat, lng], 15, { duration: 1.2, easeLinearity: 0.4 });

  // ── Price card ────────────────────────────────────────────────────
  // Price reveal + count-up animation
  priceMain.classList.remove('price-reveal');
  void priceMain.offsetWidth;
  priceMain.classList.add('price-reveal');
  animatePrice(priceMain, data.predicted_price);
  priceRange.textContent   = `Range: ${fmt$(data.confidence_low)} – ${fmt$(data.confidence_high)}`;
  priceContext.textContent = `🏠 ${data.borough_name} · ${data.bldgclass_description}`;

  // ── Price tier badge ──────────────────────────────────────────────
  const p = data.predicted_price;
  const TIERS = [
    { max: 500_000,      label: '< $500K',    cls: 'tier-entry'  },
    { max: 1_000_000,    label: '$500K–1M',   cls: 'tier-mid'    },
    { max: 3_000_000,    label: '$1M–3M',     cls: 'tier-upper'  },
    { max: 10_000_000,   label: '$3M–10M',    cls: 'tier-luxury' },
    { max: Infinity,     label: '$10M+',      cls: 'tier-ultra'  },
  ];
  const tier = TIERS.find(t => p < t.max) || TIERS[TIERS.length - 1];
  tierBadge.textContent  = tier.label;
  tierBadge.className    = `tier-badge ${tier.cls}`;
  tierBadge.style.display = 'inline-flex';

  resultCard.style.display = 'block';

  // ── Confidence bar ────────────────────────────────────────────────
  const low  = data.confidence_low;
  const high = data.confidence_high;
  const pred = data.predicted_price;
  const span = high - low;

  confLow.textContent  = fmt$(low);
  confHigh.textContent = fmt$(high);
  confFill.style.left  = '0%';
  const markerPct = Math.round(((pred - low) / span) * 100);
  confMarker.style.left = `${markerPct}%`;
  confBarWrap.style.display = 'block';

  // ── AVM QC block ─────────────────────────────────────────────────
  const qc = data.avm_qc;
  if (qc) {
    const T = TR[currentLang];
    // Grade badge
    const badgeEl = document.getElementById('confGradeBadge');
    badgeEl.textContent = qc.confidence_grade;
    badgeEl.className   = `conf-grade-badge conf-grade-${qc.confidence_grade}`;
    // Comparable count
    document.getElementById('avmCompsLine').textContent =
      `${qc.comparables_found} ${T.comparablesLabel}`;
    // Confidence bar fill reflects score strength (not always 100%)
    confFill.style.width = `${qc.confidence_score}%`;
    // QC flags
    const flagsEl = document.getElementById('avmFlagsRow');
    flagsEl.innerHTML = '';
    const FLAG_MAP = {
      'SPARSE_MARKET':    ['avm-flag-sparse',  'flagSparse'],
      'LUXURY_SEGMENT':   ['avm-flag-luxury',  'flagLuxury'],
      'HIGH_UNCERTAINTY': ['avm-flag-highunc', 'flagHighUnc'],
      'METRO_CORE':       ['avm-flag-metro',   'flagMetro'],
    };
    (qc.qc_flags || []).forEach(flag => {
      const [cls, key] = FLAG_MAP[flag] || ['', flag];
      const span = document.createElement('span');
      span.className   = `avm-qc-flag ${cls}`;
      span.textContent = T[key] || flag;
      flagsEl.appendChild(span);
    });
    document.getElementById('avmQcRow').style.display = 'flex';
  } else {
    confFill.style.width = '100%';
  }

  // ── Update map popup ──────────────────────────────────────────────
  if (marker) {
    marker.bindPopup(
      `<div class="map-popup">
        <strong>${fmt$(pred)}</strong>
        ${data.borough_name} · ${data.bldgclass_description}
        <br><small>${data.confidence_note}</small>
      </div>`,
      { maxWidth: 220 }
    ).openPopup();
  }

  // ── SHAP Drivers ──────────────────────────────────────────────────
  shapBars.innerHTML = '';
  if (data.top_drivers && data.top_drivers.length > 0) {
    const maxImpact = Math.max(...data.top_drivers.map(d => Math.abs(d.impact)));

    data.top_drivers.forEach(drv => {
      const isPos  = drv.direction === 'positive';
      const pct    = Math.round((Math.abs(drv.impact) / maxImpact) * 100);
      const arrow  = isPos ? '↑' : '↓';
      const cls    = isPos ? 'positive' : 'negative';
      const label  = drv.description || drv.feature;

      const row = document.createElement('div');
      row.className = 'shap-row';
      row.title = `${drv.feature}: value = ${drv.value.toFixed(2)}, impact = ${drv.impact > 0 ? '+' : ''}${drv.impact.toFixed(4)}`;
      row.innerHTML = `
        <span class="shap-arrow ${cls}">${arrow}</span>
        <span class="shap-label">${label}</span>
        <div class="shap-bar-wrap">
          <div class="shap-bar-fill ${cls}" style="width:${pct}%"></div>
        </div>
        <span class="shap-impact ${cls}">${drv.impact > 0 ? '+' : ''}${drv.impact.toFixed(3)}</span>
      `;
      shapBars.appendChild(row);
    });
    shapCard.style.display = 'block';
  }

  // ── Spatial features ──────────────────────────────────────────────
  const sf = data.spatial_features;
  const spatialItems = [
    { icon: '🚇', label: 'Subway',      value: fmtDist(sf.dist_subway_m),    unit: sf.nearest_station_is_express ? '⚡ express nearby' : '' },
    { icon: '🚌', label: 'Bus stop',    value: fmtDist(sf.dist_bus_m),        unit: '' },
    { icon: '🌳', label: 'Park',        value: fmtDist(sf.dist_park_m),       unit: '' },
    { icon: '🏥', label: 'Hospital',    value: fmtDist(sf.dist_hospital_m),   unit: '' },
    { icon: '🏫', label: 'School dist.', value: `#${Math.round(sf.school_district)}`, unit: `score ${sf.district_avg_score}` },
    { icon: '💰', label: 'Med. Income', value: `$${(sf.median_income_nta/1000).toFixed(0)}K`, unit: 'per household' },
    { icon: '🏠', label: 'Airbnb 500m', value: sf.airbnb_count_500m,          unit: 'listings' },
    { icon: '🔴', label: 'Crime rate',  value: sf.crime_rate_nta.toFixed(1),  unit: 'per 1k res.' },
    { icon: '🔊', label: 'Noise',       value: sf.noise_density_nta.toFixed(1), unit: 'per 1k res.' },
    { icon: '📈', label: 'Mortgage',    value: `${sf.mortgage_rate_30yr}%`,    unit: '30-yr rate' },
  ];

  const crimeMax = 80, noiseMax = 60;
  const crimePct = Math.min(100, (sf.crime_rate_nta / crimeMax) * 100).toFixed(0);
  const noisePct = Math.min(100, (sf.noise_density_nta / noiseMax) * 100).toFixed(0);
  spatialGrid.innerHTML = spatialItems.map(item => {
    let barAttr = '';
    if (item.label === 'Crime rate') barAttr = `data-bar="${crimePct}" style="--bar:${crimePct}%"`;
    if (item.label === 'Noise')      barAttr = `data-bar="${noisePct}" style="--bar:${noisePct}%"`;
    return `
    <div class="spatial-item">
      <div class="spatial-item-icon">${item.icon}</div>
      <div class="spatial-item-label">${item.label}</div>
      <div class="spatial-item-value" ${barAttr}>${item.value}
        ${item.unit ? `<span class="spatial-item-unit">${item.unit}</span>` : ''}
      </div>
    </div>`;
  }).join('');
  spatialCard.style.display = 'block';

  // Scroll sidebar to results
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Keyboard shortcut: Enter on form ─────────────────────────────────
document.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && e.target.form === predictForm && !submitBtn.disabled) {
    predictForm.requestSubmit();
  }
});

// ── Sales bubble cluster layer (Zillow-style price bubbles) ──────────
let _salesCluster = null;   // L.markerClusterGroup
let _salesTimer   = null;   // debounce handle

// ── Nearby sales markers (green circles on map) ───────────────────────
let _nearbyMarkers = [];

async function fetchNearby(lat, lon, sqft) {
  if (_lastPrediction) _lastPrediction.sqft = sqft || null;
  try {
    const res  = await fetch(`${API_BASE}/nearby?lat=${lat}&lon=${lon}&limit=8`);
    if (!res.ok) return;
    const data = await res.json();
    renderNearby(data.nearby || []);
    renderSalesBubbles(data.nearby || []);  // refresh colours with prediction context
  } catch (_) { /* silently skip if endpoint unavailable */ }
}

function renderNearby(sales) {
  // Map markers are now handled by renderSalesBubbles (price bubbles)
  // This function only updates the sidebar list
  if (!sales || sales.length === 0) return;

  const T = TR[currentLang];
  const nearbyCard    = document.getElementById('marketCard');
  const nearbyList    = document.getElementById('nearbyList');
  const compareSummary = document.getElementById('nearbyCompareSummary');

  // ── Comparison summary ─────────────────────────────────────────────
  const predicted = _lastPrediction ? _lastPrediction.price : null;
  const predSqft  = _lastPrediction ? _lastPrediction.sqft  : null;

  if (predicted && sales.length > 0) {
    const prices  = sales.map(s => s.sale_price).sort((a, b) => a - b);
    const minP    = prices[0];
    const maxP    = prices[prices.length - 1];
    const medP    = prices[Math.floor(prices.length / 2)];
    const delta   = predicted - medP;
    const deltaPct = ((delta / medP) * 100).toFixed(1);
    const aboveBelow = delta > 0
      ? `<span class="delta-badge delta-above">+${deltaPct}% ${T.aboveMedian || 'above median'}</span>`
      : `<span class="delta-badge delta-below">${deltaPct}% ${T.belowMedian || 'below median'}</span>`;

    // Position of predicted price on the min→max range bar (0–100%)
    const range = maxP - minP || 1;
    const predPct = Math.min(100, Math.max(0, Math.round(((predicted - minP) / range) * 100)));
    const medPct  = Math.min(100, Math.max(0, Math.round(((medP   - minP) / range) * 100)));

    compareSummary.innerHTML = `
      <div class="comps-summary">
        <div class="comps-stat">
          <div class="comps-stat-label">${T.compMedian || 'Comp. median'}</div>
          <div class="comps-stat-value">${fmt$(medP)}</div>
        </div>
        <div class="comps-stat comps-stat-est">
          <div class="comps-stat-label">${T.yourEstimate || 'Your estimate'}</div>
          <div class="comps-stat-value">${fmt$(predicted)}</div>
        </div>
        <div class="comps-stat">
          <div class="comps-stat-label">${T.compRange || 'Sales range'}</div>
          <div class="comps-stat-value comps-range-text">${fmt$(minP)} – ${fmt$(maxP)}</div>
        </div>
      </div>
      <div class="comps-bar-wrap">
        <div class="comps-bar-track">
          <div class="comps-bar-median" style="left:${medPct}%" title="${T.compMedian || 'Median'}: ${fmt$(medP)}"></div>
          <div class="comps-bar-pred"   style="left:${predPct}%" title="${T.yourEstimate || 'Estimate'}: ${fmt$(predicted)}"></div>
        </div>
        <div class="comps-bar-labels">
          <span>${fmt$(minP)}</span>
          <span>${aboveBelow}</span>
          <span>${fmt$(maxP)}</span>
        </div>
      </div>
    `;
    compareSummary.style.display = 'block';
  } else {
    compareSummary.style.display = 'none';
  }

  // ── Sale rows ──────────────────────────────────────────────────────
  nearbyList.innerHTML = sales.map((s, i) => {
    const psf = (s.gross_square_feet > 0)
      ? `<span class="nearby-psf">$${Math.round(s.sale_price / s.gross_square_feet).toLocaleString()}/sqft</span>`
      : '';

    let deltaBadge = '';
    if (predicted) {
      const pct = ((s.sale_price - predicted) / predicted * 100);
      const abs = Math.abs(pct);
      if (abs >= 3) {
        const cls  = pct > 0 ? 'delta-above' : 'delta-below';
        const sign = pct > 0 ? '+' : '';
        deltaBadge = `<span class="delta-badge ${cls}">${sign}${pct.toFixed(0)}%</span>`;
      } else {
        deltaBadge = `<span class="delta-badge delta-neutral">~est.</span>`;
      }
    }

    return `
      <div class="nearby-item">
        <div class="nearby-dot">${i + 1}</div>
        <div class="nearby-info">
          <div class="nearby-address">${s.address || T.unknownAddr}</div>
          <div class="nearby-meta">${s.bldgclass} · ${s.gross_square_feet.toLocaleString()} sq ft · ${s.sale_date || ''}</div>
        </div>
        <div class="nearby-right">
          <div class="nearby-price-row">
            <div class="nearby-price">${fmt$(s.sale_price)}</div>
            ${deltaBadge}
          </div>
          <div class="nearby-dist-row">
            <div class="nearby-dist">${fmtDist(s.distance_m)}</div>
            ${psf}
          </div>
        </div>
      </div>
    `;
  }).join('');

  nearbyCard.style.display = 'block';
}

// ── Market Intelligence tab state ─────────────────────────────────────
let _compsLoaded = false;
let _trendChart  = null;

// ── Market tab switching ──────────────────────────────────────────────
document.getElementById('marketTabs').addEventListener('click', e => {
  const btn = e.target.closest('.market-tab');
  if (!btn) return;
  const tab = btn.dataset.tab;
  document.querySelectorAll('.market-tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('marketNearby').style.display = tab === 'nearby' ? '' : 'none';
  document.getElementById('marketComps').style.display  = tab === 'comps'  ? '' : 'none';
  if (tab === 'comps' && !_compsLoaded) fetchAreaComps();
});

// ── NYC DOF area comps + trend fetch ─────────────────────────────────
async function fetchAreaComps() {
  const inner = document.getElementById('compsInner');
  const T     = TR[currentLang];
  const lat   = parseFloat(document.getElementById('latitude').value);
  const lon   = parseFloat(document.getElementById('longitude').value);
  if (!lat || !lon) return;

  inner.innerHTML = `<div class="comps-loading"><span class="comps-spinner"></span> ${T.compsChecking || 'Fetching NYC records…'}</div>`;

  try {
    const res  = await fetch(`${API_BASE}/market/comps?lat=${lat}&lon=${lon}`);
    const data = await res.json();

    if (!data.available) {
      inner.innerHTML = `<div class="comps-unavail">ℹ️ ${data.reason || 'No recent sales found.'}</div>`;
    } else {
      const { summary, comps, trend } = data;
      const price = _lastPrediction ? _lastPrediction.price : null;
      let deltaHtml = '';
      if (summary.median_price && price) {
        const pct  = ((price - summary.median_price) / summary.median_price * 100).toFixed(1);
        const cls  = pct >= 0 ? 'delta-above' : 'delta-below';
        const sign = pct >= 0 ? '+' : '';
        deltaHtml  = `<span class="delta-badge ${cls}">${sign}${pct}% vs median</span>`;
      }

      const compsHtml = comps.map(c => `
        <div class="comp-row">
          <div class="comp-addr">${c.address}${c.neighborhood ? ' · ' + c.neighborhood : ''}</div>
          <div class="comp-meta">
            <span class="comp-price">${fmt$(c.sale_price)}</span>
            ${c.psf ? `<span class="comp-psf">$${c.psf.toLocaleString()}/sqft</span>` : ''}
            <span class="comp-date">${c.sale_date}</span>
          </div>
        </div>`).join('');

      inner.innerHTML = `
        <div class="comps-header">
          <div class="comps-summary-row">
            <div class="comps-stat">
              <div class="comps-stat-label">${T.medianSale || 'Median Sale'}</div>
              <div class="comps-stat-val">${summary.median_price ? fmt$(summary.median_price) : '—'}</div>
              ${deltaHtml}
            </div>
            ${summary.median_psf ? `<div class="comps-stat">
              <div class="comps-stat-label">${T.medianPsf || 'Median $/sqft'}</div>
              <div class="comps-stat-val">$${summary.median_psf.toLocaleString()}</div>
            </div>` : ''}
          </div>
          <div class="comps-period">${summary.period}</div>
        </div>
        <div class="comps-list">${compsHtml}</div>
        <div class="comps-credit">
          <a href="${summary.source_url}" target="_blank">🏛 ${summary.source}</a>
        </div>`;

      // Render trend chart if data available
      if (trend && trend.length >= 3) {
        renderTrendChart(trend);
      }
    }
    _compsLoaded = true;
  } catch (_) {
    inner.innerHTML = `<div class="comps-unavail">⚠️ ${T.compsError || 'Could not reach NYC data service.'}</div>`;
    _compsLoaded = true;
  }
}

// ── Chart.js price trend ──────────────────────────────────────────────
function renderTrendChart(trend) {
  const wrap = document.getElementById('trendChart');
  wrap.style.display = 'block';

  const labels  = trend.map(t => t.month);
  const medians = trend.map(t => t.median);

  if (_trendChart) { _trendChart.destroy(); _trendChart = null; }

  const ctx = document.getElementById('trendCanvas').getContext('2d');
  _trendChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Median Sale Price',
        data:  medians,
        borderColor:     '#6366f1',
        backgroundColor: 'rgba(99,102,241,0.08)',
        pointRadius:     3,
        pointHoverRadius:5,
        tension:         0.35,
        fill:            true,
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => `$${ctx.parsed.y.toLocaleString()}`,
          }
        }
      },
      scales: {
        x: { ticks: { maxTicksLimit: 8, font: { size: 10 } }, grid: { display: false } },
        y: {
          ticks: {
            callback: v => `$${(v/1000).toFixed(0)}k`,
            font: { size: 10 },
            maxTicksLimit: 5,
          },
          grid: { color: 'rgba(0,0,0,0.06)' }
        }
      }
    }
  });
}

// ── i18n — Translations ───────────────────────────────────────────────
const TR = {
  en: {
    estimateBtn:    'Estimate Price',
    estimating:     'Estimating…',
    selectFirst:    '📍 Select a location first',
    howToUse:       'How to use',
    step1:          'Click the map to set the property location',
    step2:          'Fill in the property details below',
    step3Btn:       'Estimate Price',
    formTitle:      'Property Details',
    noLocation:     'No location selected — click the map',
    lblBorough:     'Borough',
    lblBldgType:    'Building Type',
    lblSqft:        'Building Size (sq ft)',
    lblAge:         'Building Age (years)',
    lblFloors:      'Floors',
    lblUnits:       'Residential Units',
    advancedLabel:  'Advanced options (optional)',
    lblLand:        'Land Size (sq ft)',
    lblRenov:       'Renovated since 2018?',
    lblYear:        'Valuation Year',
    lblMonth:       'Sale Month',
    lblPrior:       'Prior Sale Price ($)',
    disclaimer:     'Predictions are based on NYC property sales 2022–2026. Model accuracy: median error ±20.29%.',
    resultTitle:    'Estimated Value',
    confMid:        'Predicted',
    shapTitle:      'Price Drivers',
    shapSub:        'What factors most influenced this estimate',
    spatialTitle:   'Location Details',
    spatialSub:     'Auto-computed from coordinates',
    nearbyTitle:    'Nearby Sales',
    nearbySub:      'Recent sales within 800m',
    unknownAddr:    'Unknown address',
    compMedian:     'Comp. median',
    yourEstimate:   'Your estimate',
    compRange:      'Sales range',
    aboveMedian:    'above median',
    belowMedian:    'below median',
    analytics:      'Analytics',
    mapHint:        'Click anywhere in New York City to place a pin',
    tagline:        'NYC Property Valuation · AI-Powered',
    outOfNYC:       'Click within New York City boundaries',
    addrPlaceholder:'Search address, e.g. 350 5th Ave, Manhattan…',
    addrNotFound:   'Address not found in NYC. Try a more specific address.',
    addrOutOfNYC:   'Address is outside New York City boundaries.',
    addrError:      'Geocoding failed. Check your connection and try again.',
    compsChecking:  'Fetching NYC records…',
    compsError:     'Could not reach NYC data service.',
    medianSale:     'Median Sale',
    medianPsf:      'Median $/sqft',
    marketTitle:    'Market Intelligence',
    marketSub:      'Nearby sales & area comparables',
    tabNearby:      '📍 Nearby Sales',
    tabComps:       '🏛 Area Comps',
    trendHeader:    '📈 Price Trend (last 24 mo)',
    bldgAdvLabel:      'Advanced: search all codes',
    comparablesLabel:  'comparables found within 800m',
    flagSparse:        '⚠ Sparse data area — estimate less reliable',
    flagLuxury:        '🏆 Luxury tier — wider confidence range',
    flagHighUnc:       '⚠ High uncertainty segment',
    flagMetro:         '🏙 Manhattan core premium applies',
  },
  ar: {
    estimateBtn:    'تقدير السعر',
    estimating:     'جارٍ التقدير…',
    selectFirst:    '📍 حدد موقعاً أولاً',
    howToUse:       'كيفية الاستخدام',
    step1:          'انقر على الخريطة لتحديد موقع العقار',
    step2:          'أدخل تفاصيل العقار أدناه',
    step3Btn:       'تقدير السعر',
    formTitle:      'تفاصيل العقار',
    noLocation:     'لم يتم تحديد موقع — انقر على الخريطة',
    lblBorough:     'الحي الإداري',
    lblBldgType:    'نوع المبنى',
    lblSqft:        'مساحة المبنى (قدم مربع)',
    lblAge:         'عمر المبنى (سنة)',
    lblFloors:      'عدد الطوابق',
    lblUnits:       'الوحدات السكنية',
    advancedLabel:  'خيارات متقدمة (اختياري)',
    lblLand:        'مساحة الأرض (قدم مربع)',
    lblRenov:       'جُدِّد منذ 2018؟',
    lblYear:        'سنة التقييم',
    lblMonth:       'شهر البيع',
    lblPrior:       'سعر البيع السابق ($)',
    disclaimer:     'التوقعات مبنية على مبيعات العقارات في مدينة نيويورك 2022–2026. دقة النموذج: متوسط الخطأ ±20.29٪.',
    resultTitle:    'القيمة التقديرية',
    confMid:        'التقدير',
    shapTitle:      'محركات السعر',
    shapSub:        'العوامل الأكثر تأثيراً في هذا التقدير',
    spatialTitle:   'تفاصيل الموقع',
    spatialSub:     'محسوبة تلقائياً من الإحداثيات',
    nearbyTitle:    'المبيعات القريبة',
    nearbySub:      'مبيعات حديثة في نطاق 800 متر',
    unknownAddr:    'عنوان غير معروف',
    compMedian:     'الوسيط القريب',
    yourEstimate:   'تقديرك',
    compRange:      'نطاق المبيعات',
    aboveMedian:    'فوق الوسيط',
    belowMedian:    'تحت الوسيط',
    analytics:      'التحليلات',
    mapHint:        'انقر في أي مكان بمدينة نيويورك لوضع الدبوس',
    tagline:        'تقييم عقارات نيويورك · مدعوم بالذكاء الاصطناعي',
    outOfNYC:       'انقر داخل حدود مدينة نيويورك',
    addrPlaceholder:'ابحث عن عنوان، مثل: 350 5th Ave, Manhattan…',
    addrNotFound:   'لم يُعثر على العنوان في نيويورك. حاول بعنوان أكثر تفصيلاً.',
    addrOutOfNYC:   'العنوان خارج حدود مدينة نيويورك.',
    addrError:      'فشل البحث عن العنوان. تحقق من اتصالك وحاول مجدداً.',
    compsChecking:  'جارٍ جلب سجلات نيويورك…',
    compsError:     'تعذّر الوصول إلى خدمة بيانات نيويورك.',
    medianSale:     'وسيط المبيعات',
    medianPsf:      'وسيط $/قدم مربع',
    marketTitle:    'ذكاء السوق',
    marketSub:      'المبيعات القريبة ومقارنات المنطقة',
    tabNearby:      '📍 المبيعات القريبة',
    tabComps:       '🏛 مقارنات المنطقة',
    trendHeader:    '📈 اتجاه السعر (آخر 24 شهرًا)',
    bldgAdvLabel:      'متقدم: البحث في جميع الرموز',
    comparablesLabel:  'معاملة مقارنة في نطاق 800 متر',
    flagSparse:        '⚠ منطقة بيانات نادرة — التقدير أقل موثوقية',
    flagLuxury:        '🏆 فئة الفاخرة — نطاق ثقة أوسع',
    flagHighUnc:       '⚠ قطاع عدم يقين عالٍ',
    flagMetro:         '🏙 علاوة وسط مانهاتن مطبّقة',
  },
};

function setLang(lang) {
  currentLang = lang;
  localStorage.setItem('thamanLang', lang);
  const T = TR[lang];
  const isAr = lang === 'ar';

  // RTL / LTR
  document.documentElement.dir  = isAr ? 'rtl' : 'ltr';
  document.documentElement.lang = lang;

  // Toggle button styles
  document.getElementById('btnEN').classList.toggle('active', lang === 'en');
  document.getElementById('btnAR').classList.toggle('active', lang === 'ar');

  // Header
  document.getElementById('headerTagline').textContent = T.tagline;
  document.getElementById('analyticsLabel').textContent = T.analytics;
  document.getElementById('mapHintText').textContent   = T.mapHint;

  // How-to card
  document.getElementById('howToTitle').textContent = T.howToUse;
  document.getElementById('step1').textContent      = T.step1;
  document.getElementById('step2').textContent      = T.step2;
  // step3 — rebuild innerHTML so step3Btn id always exists for this call
  const s3 = document.getElementById('step3');
  if (s3) s3.innerHTML = isAr
    ? 'انقر على <strong id="step3Btn">تقدير السعر</strong>'
    : 'Click <strong id="step3Btn">Estimate Price</strong>';

  // Form card
  document.getElementById('formTitle').textContent    = T.formTitle;
  document.getElementById('lblBorough').textContent   = T.lblBorough;
  document.getElementById('lblBldgType').textContent  = T.lblBldgType;
  document.getElementById('lblSqft').textContent      = T.lblSqft;
  document.getElementById('lblAge').textContent       = T.lblAge;
  document.getElementById('lblFloors').textContent    = T.lblFloors;
  document.getElementById('lblUnits').textContent     = T.lblUnits;
  document.getElementById('advancedLabel').textContent = T.advancedLabel;
  const bldgAdvLabelEl = document.getElementById('bldgAdvLabel');
  if (bldgAdvLabelEl) bldgAdvLabelEl.textContent = T.bldgAdvLabel || 'Advanced: search all codes';
  document.getElementById('lblLand').textContent      = T.lblLand;
  document.getElementById('lblRenov').textContent     = T.lblRenov;
  document.getElementById('lblYear').textContent      = T.lblYear;
  document.getElementById('lblMonth').textContent     = T.lblMonth;
  document.getElementById('lblPrior').textContent     = T.lblPrior;
  document.getElementById('disclaimer').textContent   = T.disclaimer;

  // Address search placeholder
  document.getElementById('addrInput').placeholder = T.addrPlaceholder;

  // Location text (only if not selected)
  if (!latInput.value) {
    document.getElementById('locationText').textContent = T.noLocation;
  }

  // Submit button text
  if (!submitBtn.disabled) {
    btnText.textContent = '🔍 ' + T.estimateBtn;
  } else if (btnText.textContent.includes('Select') || btnText.textContent.includes('حدد')) {
    btnText.textContent = T.selectFirst;
  }

  // Results (if visible)
  document.getElementById('resultTitle').textContent   = T.resultTitle;
  document.getElementById('confMidLabel').textContent  = T.confMid;
  document.getElementById('shapTitle').textContent     = T.shapTitle;
  document.getElementById('shapSubtitle').textContent  = T.shapSub;
  document.getElementById('spatialTitle').textContent  = T.spatialTitle;
  document.getElementById('spatialSubtitle').textContent = T.spatialSub;
  if (document.getElementById('marketTitle'))
    document.getElementById('marketTitle').textContent   = T.marketTitle;
  if (document.getElementById('marketSubtitle'))
    document.getElementById('marketSubtitle').textContent = T.marketSub;
  if (document.getElementById('tabNearby'))
    document.getElementById('tabNearby').textContent = T.tabNearby;
  if (document.getElementById('tabComps'))
    document.getElementById('tabComps').textContent  = T.tabComps;
  if (document.getElementById('trendHeader'))
    document.getElementById('trendHeader').textContent = T.trendHeader;

  // AVM comps line — re-render label suffix if card is visible
  const compsEl = document.getElementById('avmCompsLine');
  if (compsEl && compsEl.textContent && !compsEl.textContent.startsWith('—')) {
    const count = compsEl.textContent.split(' ')[0];
    compsEl.textContent = `${count} ${T.comparablesLabel}`;
  }

  // Borough select options
  const boroughNames = {
    en: ['Select…', 'Manhattan', 'Bronx', 'Brooklyn', 'Queens', 'Staten Island'],
    ar: ['اختر…',   'مانهاتن',  'برونكس', 'بروكلين', 'كوينز', 'ستاتن آيلاند'],
  };
  Array.from(boroughSel.options).forEach((opt, i) => {
    opt.textContent = (boroughNames[lang] || boroughNames.en)[i] ?? opt.textContent;
  });


}

// Restore saved language on page load (called after setLang is defined)
if (currentLang === 'ar') setLang('ar');
