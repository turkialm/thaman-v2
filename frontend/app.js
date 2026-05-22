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
let _lastPrediction  = null;  // { price, sqft }
let _waterfallChart  = null;  // Chart.js waterfall instance
let _shapView        = 'bars'; // 'bars' | 'waterfall'
let _lastDrivers     = [];    // cache top_drivers for view toggle

// ── Fetch abort controllers — cancel in-flight requests on re-submit ──
let _geocodeAbort      = null;  // geocode (Nominatim)
let _predictAbort      = null;  // NYC /predict
let _riyadhPredictAbort= null;  // Riyadh /predict/riyadh

// ── City mode ─────────────────────────────────────────────────────────
let _cityMode = 'nyc';  // 'nyc' | 'riyadh'

const RIYADH_CENTER = [24.7136, 46.6753];
const NYC_CENTER    = [40.7128, -74.0060];
const RIYADH_BBOX   = { minLat: 24.35, maxLat: 25.10, minLon: 46.30, maxLon: 47.20 };

function isInRiyadh(lat, lng) {
  return lat >= RIYADH_BBOX.minLat && lat <= RIYADH_BBOX.maxLat
      && lng >= RIYADH_BBOX.minLon && lng <= RIYADH_BBOX.maxLon;
}

// ── NTA choropleth layers ─────────────────────────────────────────────
let _ntaGeoJSON      = null;   // raw NTA GeoJSON (NYC)
let _districtGeoJSON = null;   // raw district GeoJSON (Riyadh)
let _activeLayer     = null;   // current Leaflet GeoJSON layer on map
let _riyadhBorderLayer = null; // permanent district border outline (Riyadh)
let _activeMetric    = 'none'; // which layer is showing

// ── Layer metadata ────────────────────────────────────────────────────
const LAYER_META = {
  income:     { key: 'median_income_nta',        label: 'Median Income',      unit: '/yr',       palette: 'green',  fmt: v => `$${(v/1000).toFixed(0)}k` },
  crime:      { key: 'crime_rate_nta',           label: 'Crime Rate',         unit: '/1k res',   palette: 'red',    fmt: v => v.toFixed(1) },
  noise:      { key: 'noise_density_nta',        label: 'Noise Level',        unit: '/1k res',   palette: 'orange', fmt: v => v.toFixed(1) },
  air:        { key: 'pm25_mean',                label: 'PM2.5 Air Quality',  unit: 'µg/m³',     palette: 'purple', fmt: v => v.toFixed(2) },
  trees:      { key: 'tree_count_200m',          label: 'Tree Cover',         unit: 'trees/200m',palette: 'green',  fmt: v => `~${Math.round(v)}` },
  hotness:    { key: 'price_appreciation',       label: 'Market Hotness',     unit: '% gain',    palette: 'blue',   fmt: v => `${v > 0 ? '+' : ''}${(v*100).toFixed(0)}%` },
  rats:       { key: 'rat_density_nta',          label: 'Rodent Activity',    unit: '/1k res',   palette: 'red',    fmt: v => v.toFixed(2) },
  heat311:    { key: 'heat_density_nta',         label: 'Heat Complaints',    unit: '/1k res',   palette: 'orange', fmt: v => v.toFixed(2) },
  hpd:        { key: 'hpd_viol_rate_nta',        label: 'HPD Violations',     unit: 'rate',      palette: 'red',    fmt: v => v.toFixed(2) },
  livability: { key: 'livability_complaint_rate',label: '311 Livability',     unit: '/1k res',   palette: 'purple', fmt: v => v.toFixed(1) },
  transit:    { key: 'dist_subway_m',            label: 'Transit Access',     unit: 'm to subway',palette:'blue',   fmt: v => `${Math.round(v)}m`, invert: true },
  no2:        { key: 'no2_mean',                 label: 'NO₂ Pollution',      unit: 'µg/m³',     palette: 'purple', fmt: v => v.toFixed(3) },
  population: { key: 'population_2020',          label: 'Population',         unit: 'residents', palette: 'blue',   fmt: v => `${Math.round(v/1000).toFixed(0)}k` },
  bldgage:    { key: 'building_age',             label: 'Avg Building Age',   unit: 'years',     palette: 'orange', fmt: v => `${Math.round(v)}yr` },
  airbnb:     { key: 'airbnb_count_500m',        label: 'Airbnb Density',     unit: '/500m',     palette: 'purple', fmt: v => Math.round(v) },
  psf:        { key: 'price_psf',                label: 'Price / sq ft',      unit: '$/sqft',    palette: 'green',  fmt: v => `$${Math.round(v)}` },
  restaurant: { key: 'poi_restaurant_500m',      label: 'Restaurants',        unit: '/500m',     palette: 'green',  fmt: v => Math.round(v) },
  nightlife:  { key: 'poi_nightlife_500m',       label: 'Cafes & Bars',       unit: '/500m',     palette: 'orange', fmt: v => Math.round(v) },
  grocery:    { key: 'poi_grocery_500m',         label: 'Grocery Access',     unit: '/500m',     palette: 'green',  fmt: v => Math.round(v) },
  fitness:    { key: 'poi_gym_500m',             label: 'Gyms & Fitness',     unit: '/500m',     palette: 'blue',   fmt: v => Math.round(v) },
};

// ── Riyadh layer metadata ─────────────────────────────────────────────
const LAYER_META_RIYADH = {
  metro_access:  { key: 'dist_metro_m',                label: 'Metro Access',        unit: 'm to station', palette: 'blue',   fmt: v => `${Math.round(v)}m`,                 invert: true  },
  metro_density: { key: 'metro_stations_1km',          label: 'Metro Density',       unit: 'stations/1km', palette: 'blue',   fmt: v => Math.round(v)                                      },
  bus_access:    { key: 'bus_stops_500m',              label: 'Bus Stop Density',    unit: '/500m',        palette: 'blue',   fmt: v => Math.round(v)                                      },
  commercial:    { key: 'commercial_count_1km',        label: 'Commercial Services', unit: '/1km',         palette: 'green',  fmt: v => Math.round(v)                                      },
  air_no2:       { key: 'no2_nearest_mean',            label: 'NO₂ Level',           unit: 'ppb avg',      palette: 'red',    fmt: v => v.toFixed(1),                        invert: true  },
  air_quality:   { key: 'air_quality_score',           label: 'Air Quality Score',   unit: '0–100',        palette: 'green',  fmt: v => v.toFixed(0)                                       },
  price_index:   { key: 'rei_residential_qtr_idx',     label: 'Price Index',         unit: '2023=100',     palette: 'orange', fmt: v => v.toFixed(1)                                       },
  apt_price:     { key: 'district_median_price_sqm',   label: 'Median Price/sqm',    unit: 'SAR/sqm',      palette: 'green',  fmt: v => `﷼${Math.round(v).toLocaleString()}`               },
  price_trend:   { key: 'district_price_trend_slope',  label: 'Price Trend',         unit: 'SAR/qtr',      palette: 'blue',   fmt: v => `${v > 0 ? '+' : ''}${Math.round(v)}`              },
  connectivity:  { key: 'riyadh_connectivity_score',   label: 'Connectivity Score',  unit: '0–100',        palette: 'blue',   fmt: v => v.toFixed(0)                                       },
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

// ── Inverse mask: red overlay outside NYC (NYC mode only) ─────────────
// Fetches nyc_boundary.geojson (MultiPolygon dissolved from NTA data).
// Hidden in Riyadh mode — otherwise the whole city appears red-tinted.
let _nycOutOfBoundsMask    = null;
let _riyadhOutOfBoundsMask = null;

const _MASK_STYLE = { stroke: false, fillColor: '#ef4444', fillOpacity: 0.25, interactive: false };

// Build Riyadh out-of-bounds mask once (world minus Riyadh bbox hole)
function _ensureRiyadhMask() {
  if (_riyadhOutOfBoundsMask) return;
  _riyadhOutOfBoundsMask = L.polygon([
    [[ 90, -180], [ 90,  180], [-90,  180], [-90, -180]],          // world
    [[24.35, 46.30], [25.10, 46.30], [25.10, 47.20], [24.35, 47.20]], // Riyadh hole
  ], _MASK_STYLE);
}

function _setNycOutOfBoundsMask(visible) {
  if (!_nycOutOfBoundsMask) return;
  if (visible) {
    if (!map.hasLayer(_nycOutOfBoundsMask)) _nycOutOfBoundsMask.addTo(map);
  } else {
    if (map.hasLayer(_nycOutOfBoundsMask)) map.removeLayer(_nycOutOfBoundsMask);
  }
}

fetch('/ui/nyc_boundary.geojson')
  .then(r => r.json())
  .then(data => {
    const toLflt = ring => ring.map(([lng, lat]) => [lat, lng]);
    const geom = data.geometry || (data.type === 'MultiPolygon' ? data : null);
    nycBoundaryCoords = geom ? geom.coordinates : null;
    const coords = nycBoundaryCoords || [];
    const rings = [
      [[ 90, -180], [ 90,  180], [-90,  180], [-90, -180]],
      ...coords.map(poly => toLflt(poly[0])),
    ];
    _nycOutOfBoundsMask = L.polygon(rings, _MASK_STYLE);
    if (_cityMode === 'nyc') _nycOutOfBoundsMask.addTo(map);
  })
  .catch(() => {
    _nycOutOfBoundsMask = L.polygon([
      [[ 90, -180], [ 90,  180], [-90,  180], [-90, -180]],
      [[40.477399, -74.25909], [40.477399, -73.700272],
       [40.917577, -73.700272], [40.917577, -74.25909]],
    ], _MASK_STYLE);
    if (_cityMode === 'nyc') _nycOutOfBoundsMask.addTo(map);
  });

// ── NTA choropleth — lazy load (only fetched when first layer btn clicked) ──
let _ntaFetchStarted = false;
function ensureNtaLoaded() {
  if (_ntaFetchStarted || _ntaGeoJSON) return;
  _ntaFetchStarted = true;
  fetch(`${API_BASE}/layers/nta`)
    .then(r => r.ok ? r.json() : null)
    .then(data => {
      if (data) {
        _ntaGeoJSON = data;
        document.getElementById('layerBar').style.display = 'flex';
      }
    })
    .catch(() => {});
}

// ── District layer — lazy load (only when Riyadh layer btn clicked) ─────────
let _districtFetchStarted = false;
function ensureDistrictLoaded(cb) {
  if (_districtGeoJSON) { if (cb) cb(); return; }
  if (_districtFetchStarted) { if (cb) setTimeout(() => { if (_districtGeoJSON && cb) cb(); }, 800); return; }
  _districtFetchStarted = true;
  fetch(`${API_BASE}/layers/district`)
    .then(r => r.ok ? r.json() : null)
    .then(data => { if (data) { _districtGeoJSON = data; if (cb) cb(); } })
    .catch(() => {});
}

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
function colorScale(val, min, max, palette, invert=false) {
  if (val === null || val === undefined || isNaN(val)) return '#e5e7eb';
  let t = max === min ? 0.5 : Math.max(0, Math.min(1, (val - min) / (max - min)));
  if (invert) t = 1 - t;
  const [c0, c1] = (_PALETTES[palette] || _PALETTES.blue).map(_hexToRgb);
  return _rgbToHex(c0[0] + t*(c1[0]-c0[0]), c0[1] + t*(c1[1]-c0[1]), c0[2] + t*(c1[2]-c0[2]));
}

function showLayer(metricId) {
  if (_activeLayer) { map.removeLayer(_activeLayer); _activeLayer = null; }
  _activeMetric = metricId;

  const legend = document.getElementById('layerLegend');

  // Choose correct GeoJSON source and metadata table based on city mode
  const isRiyadh   = _cityMode === 'riyadh';
  const metaTable  = isRiyadh ? LAYER_META_RIYADH : LAYER_META;

  // Lazy-load GeoJSON on first layer click
  if (isRiyadh && !_districtGeoJSON) {
    ensureDistrictLoaded(() => showLayer(metricId));
    return;
  }
  if (!isRiyadh && !_ntaGeoJSON) {
    ensureNtaLoaded();
    return;
  }

  const activeGJ = isRiyadh ? _districtGeoJSON : _ntaGeoJSON;
  if (metricId === 'none' || !activeGJ) { legend.style.display = 'none'; return; }

  const meta = metaTable[metricId];
  if (!meta) return;

  const values = activeGJ.features
    .map(f => f.properties[meta.key])
    .filter(v => v !== null && v !== undefined && !isNaN(v));
  if (!values.length) return;

  const min = Math.min(...values), max = Math.max(...values);

  // Detect geometry type from first feature to support both polygon and point GeoJSON
  const firstGeomType = activeGJ.features.length ? activeGJ.features[0].geometry.type : 'Point';
  const riyadhIsPolygon = isRiyadh && firstGeomType !== 'Point';

  if (isRiyadh && !riyadhIsPolygon) {
    // Legacy centroid-point fallback
    _activeLayer = L.geoJSON(activeGJ, {
      pointToLayer: (feat, latlng) => L.circleMarker(latlng, {
        radius: 10,
        fillColor:   colorScale(feat.properties[meta.key], min, max, meta.palette, meta.invert || false),
        fillOpacity: 0.75,
        weight: 1, color: '#ffffff', opacity: 0.8,
      }),
      onEachFeature: (feat, layer) => {
        const v = feat.properties[meta.key];
        const fmtVal = (v !== null && v !== undefined) ? meta.fmt(v) : 'N/A';
        layer.bindTooltip(
          `<b>${feat.properties.name_ar || feat.properties.district_ar || ''}</b><br>${meta.label}: ${fmtVal} ${meta.unit}`,
          { sticky: true, className: 'layer-tooltip' }
        );
      },
    }).addTo(map);
  } else if (riyadhIsPolygon) {
    // Polygon choropleth for Riyadh districts
    _activeLayer = L.geoJSON(activeGJ, {
      style: feat => {
        const v = feat.properties[meta.key];
        return {
          fillColor:   colorScale(v, min, max, meta.palette, meta.invert || false),
          fillOpacity: v != null ? 0.65 : 0.10,
          weight: 1,
          color: '#ffffff',
          opacity: 0.6,
        };
      },
      onEachFeature: (feat, layer) => {
        const v = feat.properties[meta.key];
        const fmtVal = (v !== null && v !== undefined) ? meta.fmt(v) : 'N/A';
        layer.bindTooltip(
          `<b>${feat.properties.name_ar || feat.properties.district_ar || ''}</b><br>${meta.label}: ${fmtVal} ${meta.unit}`,
          { sticky: true, className: 'layer-tooltip' }
        );
        layer.on({ mouseover: e => e.target.setStyle({ fillOpacity: 0.85, weight: 2 }),
                   mouseout:  e => _activeLayer.resetStyle(e.target) });
      },
    }).addTo(map);
  } else {
    _activeLayer = L.geoJSON(activeGJ, {
    style: feat => {
      const v = feat.properties[meta.key];
      return {
        fillColor:   colorScale(v, min, max, meta.palette, meta.invert || false),
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
  } // end else (NYC polygon branch)

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

// ── Market Listings Layer ──────────────────────────────────────────────
let _listingsLayer = null;

async function toggleListingsLayer(btn) {
  if (_listingsLayer && map.hasLayer(_listingsLayer)) {
    map.removeLayer(_listingsLayer);
    btn.classList.remove('active');
    return;
  }
  if (_listingsLayer) {
    map.addLayer(_listingsLayer);
    btn.classList.add('active');
    return;
  }
  // First load
  btn.textContent = '⏳';
  try {
    const geojson = await fetch(`${API_BASE}/layers/listings`).then(r => r.json());
    const typeColors = {
      apartment: '#3b82f6',
      villa:     '#10b981',
      plot:      '#f59e0b',
      building:  '#8b5cf6',
      other:     '#6b7280'
    };
    _listingsLayer = L.geoJSON(geojson, {
      pointToLayer: (feat, latlng) => {
        const color = typeColors[feat.properties.type_en] || '#6b7280';
        return L.circleMarker(latlng, {
          radius: 6, fillColor: color, color: '#fff',
          weight: 1, opacity: 1, fillOpacity: 0.8
        });
      },
      onEachFeature: (feat, layer) => {
        const p = feat.properties;
        layer.bindPopup(`
          <div style="min-width:200px;font-family:inherit">
            <div style="font-weight:700;font-size:1rem;margin-bottom:6px">${p.district || '—'}</div>
            <div style="color:#64748b;font-size:0.8rem;margin-bottom:8px">${p.type_ar || p.type_en}</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:0.85rem">
              <span style="color:#64748b">Asking/sqm</span>
              <span style="font-weight:600;color:#0f172a">﷼${p.price_per_sqm.toLocaleString()}</span>
              <span style="color:#64748b">Total</span>
              <span style="font-weight:600">${fmtSAR(p.price_sar)}</span>
              <span style="color:#64748b">Area</span>
              <span>${p.area_sqm} sqm</span>
              ${p.bedrooms ? `<span style="color:#64748b">Beds</span><span>${p.bedrooms}</span>` : ''}
            </div>
            ${p.url ? `<a href="${p.url}" target="_blank" rel="noopener" style="display:block;margin-top:8px;font-size:0.8rem;color:#3b82f6">View on Haraj ↗</a>` : ''}
          </div>
        `);
      }
    }).addTo(map);
    btn.classList.add('active');
  } catch(e) {
    console.error('Listings layer failed:', e);
  } finally {
    btn.textContent = '🏠 Listings';
  }
}

// ── City mode toggle ──────────────────────────────────────────────────
function setCityMode(mode) {
  _cityMode = mode;
  const isRiyadh = mode === 'riyadh';
  // Kick off background prefetch so GeoJSON is ready when first layer clicked
  if (isRiyadh) ensureDistrictLoaded();
  else           ensureNtaLoaded();

  // Update toggle button styles
  document.getElementById('cityBtnNYC').classList.toggle('active', !isRiyadh);
  document.getElementById('cityBtnRiyadh').classList.toggle('active', isRiyadh);

  // Swap layer groups
  document.getElementById('nycLayerGroup').style.display    = isRiyadh ? 'none' : 'contents';
  document.getElementById('riyadhLayerGroup').style.display = isRiyadh ? 'contents' : 'none';

  // Reset active layer state
  document.querySelectorAll('.layer-btn').forEach(b => b.classList.remove('active'));
  const defaultBtn = document.querySelector(`#${isRiyadh ? 'riyadhLayerGroup' : 'nycLayerGroup'} [data-layer="none"]`);
  if (defaultBtn) defaultBtn.classList.add('active');
  showLayer('none');

  // Show/hide city-specific predict forms and clear Riyadh results
  document.getElementById('addrSearchWrap').style.display  = isRiyadh ? 'none' : '';
  document.getElementById('locationDisplay').style.display = isRiyadh ? 'none' : '';
  document.getElementById('predictForm').style.display     = isRiyadh ? 'none' : '';
  document.getElementById('riyadhForm').style.display      = isRiyadh ? ''     : 'none';
  document.getElementById('riyadhResults').style.display   = 'none';

  // Swap header badges, tagline, and page title
  document.getElementById('nycBadgeGroup').style.display    = isRiyadh ? 'none'    : '';
  document.getElementById('riyadhBadgeGroup').style.display = isRiyadh ? ''        : 'none';
  document.getElementById('headerTagline').textContent      = isRiyadh
    ? 'Riyadh Property Valuation · AI-Powered'
    : 'NYC Property Valuation · AI-Powered';
  document.title = isRiyadh
    ? 'THAMAN — Riyadh Property Valuation'
    : 'THAMAN — NYC Property Valuation';

  // Red out-of-bounds mask — each city masks areas outside its limits
  _setNycOutOfBoundsMask(!isRiyadh);
  _ensureRiyadhMask();
  if (isRiyadh) {
    if (!map.hasLayer(_riyadhOutOfBoundsMask)) _riyadhOutOfBoundsMask.addTo(map);
  } else {
    if (_riyadhOutOfBoundsMask && map.hasLayer(_riyadhOutOfBoundsMask)) map.removeLayer(_riyadhOutOfBoundsMask);
  }

  // Show/hide the Listings overlay button (Riyadh only)
  const listingsBtn = document.getElementById('listingsLayerBtn');
  if (listingsBtn) listingsBtn.style.display = isRiyadh ? '' : 'none';
  // Remove listings layer when switching away from Riyadh
  if (!isRiyadh && _listingsLayer && map.hasLayer(_listingsLayer)) {
    map.removeLayer(_listingsLayer);
  }

  // Update map hint text for correct city
  const mapHintTextEl = document.getElementById('mapHintText');
  if (mapHintTextEl) {
    mapHintTextEl.textContent = isRiyadh
      ? TR[currentLang].mapHintRiyadh
      : TR[currentLang].mapHint;
  }

  // Auto-load district price layer when entering Riyadh
  if (isRiyadh) {
    ensureDistrictLoaded(() => {
      if (_cityMode !== 'riyadh') return;  // guard: user may have switched back before GeoJSON loaded
      document.querySelectorAll('#riyadhLayerGroup .layer-btn').forEach(b => b.classList.remove('active'));
      const aptBtn = document.querySelector('#riyadhLayerGroup [data-layer="apt_price"]');
      if (aptBtn) aptBtn.classList.add('active');
      showLayer('apt_price');
    });
    ensureDistrictLoaded(() => {
      if (_cityMode !== 'riyadh') return;  // guard
      if (_riyadhBorderLayer) { map.removeLayer(_riyadhBorderLayer); _riyadhBorderLayer = null; }
      _riyadhBorderLayer = L.geoJSON(_districtGeoJSON, { style: { color: '#374151', weight: 1, fillOpacity: 0, opacity: 0.5 } }).addTo(map);
    });
  } else {
    if (_riyadhBorderLayer) { map.removeLayer(_riyadhBorderLayer); _riyadhBorderLayer = null; }
    showLayer('none');  // ensure legend cleared when returning to NYC
  }

  // Fly map to the selected city
  map.flyTo(isRiyadh ? RIYADH_CENTER : NYC_CENTER, 11, { duration: 1.5 });
}

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
  const errEl = (_cityMode === 'riyadh')
    ? (document.getElementById('riyadhAddrError') || addrError)
    : addrError;
  errEl.textContent = msg;
  errEl.style.display = 'block';
  setTimeout(() => { errEl.style.display = 'none'; }, 4000);
}

async function geocodeAddress() {
  const riyadhInput = document.getElementById('riyadhAddrInput');
  const riyadhBtn   = document.getElementById('riyadhAddrBtn');
  const activeInput = (_cityMode === 'riyadh' && riyadhInput) ? riyadhInput : addrInput;
  const activeBtn   = (_cityMode === 'riyadh' && riyadhBtn)   ? riyadhBtn   : addrBtn;

  const q = activeInput.value.trim();
  if (!q) return;

  activeBtn.classList.add('loading');
  addrError.style.display = 'none';
  const riyadhErrEl = document.getElementById('riyadhAddrError');
  if (riyadhErrEl) riyadhErrEl.style.display = 'none';

  const isRiyadhMode = _cityMode === 'riyadh';

  try {
    const url = isRiyadhMode
      ? `https://nominatim.openstreetmap.org/search?` +
        `q=${encodeURIComponent(q + ', Riyadh, Saudi Arabia')}&format=json&limit=1` +
        `&countrycodes=sa&bounded=1&viewbox=46.30,24.35,47.20,25.10`
      : `https://nominatim.openstreetmap.org/search?` +
        `q=${encodeURIComponent(q + ', New York City')}&format=json&limit=1` +
        `&countrycodes=us&bounded=1&viewbox=-74.26,40.47,-73.70,40.92`;

    if (_geocodeAbort) _geocodeAbort.abort();
    _geocodeAbort = new AbortController();
    const res  = await fetch(url, {
      headers: { 'Accept-Language': 'en' },
      signal:  _geocodeAbort.signal,
    });
    const data = await res.json();

    if (!data || data.length === 0) {
      showAddrError(TR[currentLang].addrNotFound);
      return;
    }

    const lat = parseFloat(data[0].lat);
    const lng = parseFloat(data[0].lon);

    if (isRiyadhMode) {
      if (!isInRiyadh(lat, lng)) {
        showAddrError('Location not found in Riyadh');
        return;
      }
      document.getElementById('riyadhLat').value = lat.toFixed(6);
      document.getElementById('riyadhLon').value = lng.toFixed(6);
      lastValidPos = [lat, lng];

      if (marker) {
        marker.setLatLng([lat, lng]);
      } else {
        marker = L.marker([lat, lng], { icon: pinIcon, draggable: true }).addTo(map);
        marker.on('dragend', (ev) => {
          const pos = ev.target.getLatLng();
          if (!isInRiyadh(pos.lat, pos.lng)) {
            if (lastValidPos) marker.setLatLng(lastValidPos);
            return;
          }
          lastValidPos = [pos.lat, pos.lng];
          document.getElementById('riyadhLat').value = pos.lat.toFixed(6);
          document.getElementById('riyadhLon').value = pos.lng.toFixed(6);
        });
      }

      document.getElementById('riyadhSubmitBtn').disabled = false;
      const riyadhBtnText = document.getElementById('riyadhBtnText');
      if (riyadhBtnText) riyadhBtnText.textContent = '🔍  تقدير السعر / Estimate';
      map.flyTo([lat, lng], 14, { duration: 1.2, easeLinearity: 0.4 });

      if (marker) {
        marker.bindPopup(
          `<div class="map-popup"><small>📍 ${data[0].display_name.split(',').slice(0,3).join(',')}</small></div>`,
          { maxWidth: 220 }
        ).openPopup();
        setTimeout(() => marker.closePopup(), 3000);
      }
      return;
    }

    // ── NYC geocode path ──────────────────────────────────────────────
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
    if (err.name === 'AbortError') return;  // superseded by newer request
    showAddrError(TR[currentLang].addrError);
    console.error('Geocoding error:', err);
  } finally {
    const rBtn = document.getElementById('riyadhAddrBtn');
    const aBtn = document.getElementById('addrBtn');
    if (rBtn) rBtn.classList.remove('loading');
    if (aBtn) aBtn.classList.remove('loading');
  }
}

addrBtn.addEventListener('click', geocodeAddress);
addrInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); geocodeAddress(); }
});

// Riyadh address search listeners (elements added to #riyadhForm)
document.getElementById('riyadhAddrBtn').addEventListener('click', geocodeAddress);
document.getElementById('riyadhAddrInput').addEventListener('keydown', (e) => {
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

loadBldgClasses().then(() => parseURLParams());

// ── Building type info data ────────────────────────────────────────────
const BLDG_INFO = {
  A1: {
    icon: '🏠', title: 'Single-Family Detached',
    desc: 'A standalone house on its own lot — no shared walls with neighbours. Common in Staten Island, Queens, and outer Brooklyn. Typically 1–3 floors with a private backyard.',
    descAr: 'منزل مستقل على قطعة أرض منفردة — لا جدران مشتركة مع الجيران. شائع في ستاتن آيلاند وكوينز وأجزاء من بروكلين. عادةً 1-3 طوابق مع حديقة خاصة.',
    median: '$769K', count: '15K sales/yr', accuracy: '±13%',
  },
  B2: {
    icon: '🏡', title: '2-Family Home (Frame)',
    desc: 'A two-unit house sharing a foundation — typically one unit per floor. Very common in Brooklyn and Queens. Owner often lives in one unit and rents the other.',
    descAr: 'منزل بوحدتين يشتركان في الأساس — عادةً وحدة واحدة في كل طابق. شائع جداً في بروكلين وكوينز. غالباً يسكن المالك في وحدة واحدة ويؤجر الأخرى.',
    median: '$925K', count: '9.6K sales/yr', accuracy: '±20%',
  },
  C0: {
    icon: '🏘', title: '3-Unit Walk-up',
    desc: 'A small apartment building with 3 units and no elevator — residents walk up the stairs. Typically 3–4 floors. Widespread across all boroughs except Manhattan.',
    descAr: 'مبنى سكني صغير بـ3 وحدات وبدون مصعد — يصعد السكان السلم. عادةً 3-4 طوابق. منتشر في جميع الأحياء.',
    median: '$1.2M', count: '7.4K sales/yr', accuracy: '±21%',
  },
  A5: {
    icon: '🏙', title: 'Row House (Attached Single-Family)',
    desc: 'A single-family home attached to neighbouring homes on one or both sides — like a terrace house. Very common in Brooklyn brownstone neighbourhoods and the Bronx.',
    descAr: 'منزل عائلة واحدة ملاصق للمنازل المجاورة من جانب أو جانبين — مثل المنازل المتصلة. شائع جداً في أحياء بروكلين التاريخية والبرونكس.',
    median: '$679K', count: '13K sales/yr', accuracy: '±20%',
  },
  R1: {
    icon: '🏢', title: 'Condo Unit (Walk-up)',
    desc: 'An individually owned apartment unit in a building without an elevator. The owner holds the deed to that specific unit plus a share of common areas. Popular in Brooklyn and Queens.',
    descAr: 'وحدة شقة مملوكة بشكل فردي في مبنى بدون مصعد. يملك المالك صك تلك الوحدة بالإضافة إلى حصة في المساحات المشتركة.',
    median: '$978K', count: '2.1K sales/yr', accuracy: '±21%',
  },
  D4: {
    icon: '🏗', title: 'Elevator Condo Apartment',
    desc: 'An individually owned unit in a high-rise or mid-rise building with an elevator. The most common property type in NYC (41K+ annual sales). Dominant in Manhattan and Long Island City.',
    descAr: 'وحدة مملوكة بشكل فردي في مبنى شاهق أو متوسط الارتفاع مع مصعد. أكثر أنواع العقارات شيوعاً في نيويورك (أكثر من 41 ألف صفقة سنوياً). الأكثر انتشاراً في مانهاتن وشواطئ لونغ آيلاند.',
    median: '$495K', count: '41K sales/yr', accuracy: '±20%',
  },
  C1: {
    icon: '🏠', title: 'Walk-up Apartment (4–6 Units)',
    desc: 'A small apartment building with 4 to 6 units and no elevator. Usually 4–5 floors, with a hallway and staircase shared by all residents. Common investment property in Brooklyn and the Bronx.',
    descAr: 'مبنى سكني صغير بـ4 إلى 6 وحدات وبدون مصعد. عادةً 4-5 طوابق. عقار استثماري شائع في بروكلين والبرونكس.',
    median: '$2.05M', count: '1.8K sales/yr', accuracy: '±21%',
  },
  C6: {
    icon: '🤝', title: 'Co-op Apartment',
    desc: 'A form of shared ownership — instead of owning your unit outright, you own shares in a corporation that owns the building and receive a lease for your apartment. Common in Manhattan and older Queens buildings.',
    descAr: 'شكل من أشكال الملكية المشتركة — بدلاً من امتلاك وحدتك مباشرة، تمتلك أسهماً في شركة تملك المبنى وتحصل على عقد إيجار لشقتك.',
    median: '$370K', count: '8.4K sales/yr', accuracy: '±21%',
  },
  S1: {
    icon: '🏪', title: 'Mixed-Use (1-Family + Store)',
    desc: 'A building that combines a ground-floor commercial space (shop, office, or restaurant) with a residential unit above. Common in neighbourhood main streets across all boroughs.',
    descAr: 'مبنى يجمع بين مساحة تجارية في الطابق الأرضي (محل أو مكتب أو مطعم) ووحدة سكنية بالأعلى. شائع في الشوارع الرئيسية للأحياء السكنية.',
    median: '$925K', count: '692 sales/yr', accuracy: '±20%',
  },
};

// ── Building type info popup (all types, single ! button by label) ─────
function initBldgInfoPopup() {
  const popup    = document.getElementById('bldgInfoPopup');
  const overlay  = document.getElementById('bldgInfoOverlay');
  const closeBtn = document.getElementById('bldgInfoClose');
  const listEl   = document.getElementById('bldgInfoList');
  const titleEl  = document.getElementById('bldgInfoPopupTitle');

  function openPopup() {
    const isAr = currentLang === 'ar';
    titleEl.textContent = isAr ? 'دليل أنواع المباني' : 'Building Types Guide';
    listEl.innerHTML = Object.entries(BLDG_INFO).map(([code, info]) => `
      <div class="bldg-info-row">
        <div class="bldg-info-row-icon">${info.icon}</div>
        <div class="bldg-info-row-body">
          <div class="bldg-info-row-head">
            <span class="bldg-info-row-title">${info.title}</span>
            <span class="bldg-info-row-code">${code}</span>
            <span class="bldg-info-row-median">${info.median}</span>
          </div>
          <div class="bldg-info-row-desc">${isAr ? info.descAr : info.desc}</div>
        </div>
      </div>
    `).join('');
    popup.style.display   = 'flex';
    overlay.style.display = 'block';
  }

  function closePopup() {
    popup.style.display   = 'none';
    overlay.style.display = 'none';
  }

  document.getElementById('bldgTypesInfoBtn').addEventListener('click', openPopup);
  closeBtn.addEventListener('click', closePopup);
  overlay.addEventListener('click', closePopup);
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closePopup(); });
}

initBldgInfoPopup();

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
    const panel  = document.getElementById('bldgAdvPanel');
    const toggle = document.getElementById('bldgAdvToggle');
    panel.classList.toggle('open');
    toggle.classList.toggle('open');
  });
}

initBldgTypeCards();

// ── Hover-prefetch GeoJSON on city button hover ───────────────────────
document.getElementById('cityBtnRiyadh').addEventListener('mouseenter', () => ensureDistrictLoaded(), { once: true });
document.getElementById('cityBtnNYC').addEventListener('mouseenter',    () => ensureNtaLoaded(),      { once: true });

// ── Riyadh property type card selector ───────────────────────────────
(function initRiyadhTypeCards() {
  const cards  = document.querySelectorAll('.riyadh-type-card');
  const hidden = document.getElementById('riyadhType');
  cards.forEach(card => {
    card.addEventListener('click', () => {
      cards.forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');
      hidden.value = card.dataset.value;
      card.closest('#riyadhTypeGrid')?.classList.remove('invalid');
    });
  });
})();

// ── Map click ─────────────────────────────────────────────────────────
map.on('click', (e) => {
  const { lat, lng } = e.latlng;

  // ── Riyadh mode ───────────────────────────────────────────────────
  if (_cityMode === 'riyadh') {
    if (!isInRiyadh(lat, lng)) {
      showMapError(lat, lng, 'Location is outside Riyadh bounds');
      return;
    }
    // Update hidden Riyadh inputs
    document.getElementById('riyadhLat').value = lat.toFixed(6);
    document.getElementById('riyadhLon').value = lng.toFixed(6);
    lastValidPos = [lat, lng];

    // Place/move marker
    if (marker) {
      marker.setLatLng([lat, lng]);
    } else {
      marker = L.marker([lat, lng], { icon: pinIcon, draggable: true }).addTo(map);
      marker.on('dragend', (ev) => {
        const pos = ev.target.getLatLng();
        if (!isInRiyadh(pos.lat, pos.lng)) {
          if (lastValidPos) marker.setLatLng(lastValidPos);
          showMapError(pos.lat, pos.lng, 'Location is outside Riyadh bounds');
          return;
        }
        lastValidPos = [pos.lat, pos.lng];
        document.getElementById('riyadhLat').value = pos.lat.toFixed(6);
        document.getElementById('riyadhLon').value = pos.lng.toFixed(6);
      });
    }

    // Enable Riyadh submit button and update location text
    document.getElementById('riyadhSubmitBtn').disabled = false;
    const riyadhBtnText = document.getElementById('riyadhBtnText');
    if (riyadhBtnText) riyadhBtnText.textContent = '🔍  تقدير السعر / Estimate';
    return;
  }

  // ── NYC mode ──────────────────────────────────────────────────────
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

function fmtSAR(n) {
  if (n >= 1_000_000) return `﷼${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 100_000)   return `﷼${Math.round(n / 1_000)}K`;
  return `﷼${Math.round(n).toLocaleString()}`;
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
  // Keep cluster group alive — only clear and batch-add markers
  if (!_salesCluster) {
    _salesCluster = L.markerClusterGroup({
      maxClusterRadius:    40,
      showCoverageOnHover: false,
      chunkedLoading:      true,
      iconCreateFunction: c => L.divIcon({
        html:      `<div class="cluster-bubble">${c.getChildCount()}</div>`,
        className: '',
        iconAnchor:[18, 18],
      }),
    });
    map.addLayer(_salesCluster);
  } else {
    _salesCluster.clearLayers();
  }

  const markers = (sales || []).filter(s => s.latitude && s.longitude).map(s => {
    const psf = s.gross_square_feet > 0
      ? `$${Math.round(s.sale_price / s.gross_square_feet).toLocaleString()}/sqft` : '';
    const m = L.marker([s.latitude, s.longitude], { icon: buildSaleIcon(s.sale_price) });
    let vsBadge = '';
    if (_lastPrediction && _lastPrediction.price) {
      const pct  = (s.sale_price - _lastPrediction.price) / _lastPrediction.price * 100;
      const sign = pct >= 0 ? '+' : '';
      const cls  = pct >= 5 ? 'delta-above' : pct <= -5 ? 'delta-below' : 'delta-neutral';
      vsBadge = `<span class="delta-badge ${cls}" style="margin-left:4px">${sign}${pct.toFixed(0)}% vs est.</span>`;
    }
    m.bindPopup(
      `<div class="sale-popup">
        <strong>${formatBubblePrice(s.sale_price)}${vsBadge}</strong>
        <div class="sale-popup-addr">${s.address || ''}</div>
        <div class="sale-popup-meta">${s.bldgclass || ''} · ${(s.gross_square_feet||0).toLocaleString()} sqft${psf ? ' · '+psf : ''}</div>
        <div class="sale-popup-date">Sold ${s.sale_date || ''}</div>
      </div>`,
      { maxWidth: 220 }
    );
    return m;
  });

  _salesCluster.addLayers(markers);   // single batch — much faster than addLayer() loop
}

// ── SHAP view toggle ─────────────────────────────────────────────────
function setShapView(view) {
  _shapView = view;
  document.getElementById('shapViewBars').classList.toggle('active', view === 'bars');
  document.getElementById('shapViewWaterfall').classList.toggle('active', view === 'waterfall');
  document.getElementById('shapBars').style.display       = view === 'bars' ? '' : 'none';
  document.getElementById('waterfallCard').style.display  = view === 'waterfall' ? '' : 'none';
  if (view === 'waterfall' && _lastDrivers.length) renderWaterfall(_lastDrivers, _lastPrediction?.price);
}

// ── SHAP waterfall chart ─────────────────────────────────────────────
function renderWaterfall(drivers, predictedPrice) {
  if (!drivers || !drivers.length) return;
  const ctx = document.getElementById('waterfallCanvas').getContext('2d');
  if (_waterfallChart) { _waterfallChart.destroy(); _waterfallChart = null; }

  // Build running total from baseline
  // Baseline = predicted - sum(top impacts)  (approximation; remaining = residual)
  const totalImpact = drivers.reduce((s, d) => s + d.impact, 0);
  const baseline    = predictedPrice ? predictedPrice - Math.round(Math.expm1(Math.abs(totalImpact)) * Math.sign(totalImpact) * predictedPrice * 0.05) : 0;

  const labels = ['Baseline', ...drivers.map(d => shortLabel(d.description || d.feature)), 'Estimate'];
  const colors = [];
  const floatData = []; // [bottom, top] for each bar
  let running = baseline || (predictedPrice * 0.80);

  // Baseline bar
  floatData.push([0, running]);
  colors.push('#6366f1');

  drivers.forEach(d => {
    const dollarImpact = (predictedPrice || 1000000) * (Math.expm1(Math.abs(d.impact))) * Math.sign(d.impact) * 0.12;
    const bottom = d.direction === 'positive' ? running : running + dollarImpact;
    const top    = d.direction === 'positive' ? running + dollarImpact : running;
    floatData.push([bottom, top]);
    colors.push(d.direction === 'positive' ? '#059669' : '#dc2626');
    running += dollarImpact;
  });

  // Final estimate bar
  floatData.push([0, predictedPrice || running]);
  colors.push('#2563eb');

  _waterfallChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data:            floatData,
        backgroundColor: colors,
        borderRadius:    4,
        borderWidth:     0,
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (item) => {
              const [lo, hi] = item.raw;
              const delta = hi - lo;
              return delta >= 0
                ? `+$${Math.abs(delta).toLocaleString()}`
                : `-$${Math.abs(delta).toLocaleString()}`;
            }
          }
        }
      },
      scales: {
        y: {
          ticks: { callback: v => '$' + (v/1e6).toFixed(1) + 'M' },
          grid: { color: 'rgba(0,0,0,.05)' }
        },
        x: { grid: { display: false }, ticks: { font: { size: 10 } } }
      }
    }
  });
}

function shortLabel(label) {
  const map = {
    'Building class encoding':   'Bldg Class',
    'Neighbourhood encoding':    'NTA',
    'Building size':             'Size',
    'Distance to Downtown':      '→ Downtown',
    'Distance to Midtown':       '→ Midtown',
    'NTA × Building class':      'NTA×Class',
    'Borough × Building class':  'Boro×Class',
    'Building age':              'Age',
    'Neighbourhood income':      'Income',
    'Distance to subway':        '→ Subway',
    'Crime rate':                'Crime',
    'Prior sale price/sqft':     'Prior PSF',
  };
  return map[label] || label.slice(0, 10);
}

// ── Natural language SHAP tooltip ────────────────────────────────────
function shapNlTooltip(drv) {
  const T    = TR[currentLang];
  const dir  = drv.direction === 'positive';
  const sign = dir ? (T.nlUp || 'pushed the price UP') : (T.nlDown || 'pushed the price DOWN');
  const feat = drv.description || drv.feature;
  const val  = drv.value;

  // Feature-specific human context
  const ctx = shapContext(drv.feature, val, dir);
  return `${feat} ${sign}${ctx ? ' — ' + ctx : ''}.`;
}

function shapContext(feature, value, isPositive) {
  if (feature.includes('dist_subway')) {
    const m = Math.round(value);
    return m < 300 ? `only ${m}m away (excellent transit)` : m > 1200 ? `${m}m — limited subway access` : `${m}m from nearest station`;
  }
  if (feature.includes('crime_rate')) {
    return value < 20 ? `low crime area (${value.toFixed(0)}/1k)` : value > 60 ? `high crime area (${value.toFixed(0)}/1k)` : `moderate crime (${value.toFixed(0)}/1k)`;
  }
  if (feature.includes('median_income')) {
    return `$${(value/1000).toFixed(0)}K median household income`;
  }
  if (feature.includes('building_age')) {
    return value > 80 ? `${Math.round(value)}-year-old building` : `built ${new Date().getFullYear() - Math.round(value)}`;
  }
  if (feature.includes('gross_square_feet') || feature.includes('building_size')) {
    return `${Math.round(value).toLocaleString()} sq ft`;
  }
  if (feature.includes('dist_downtown') || feature.includes('dist_midtown')) {
    return `${(value/1000).toFixed(1)} km from city centre`;
  }
  return '';
}

// ── Fetch sales for the current map viewport (bbox) ──────────────────
// ── Tile-based sales layer — O(1) server lookups, client-side cache ───
const _TILE_DEG    = 0.05;
const _NYC_MIN_LAT = 40.45, _NYC_MIN_LON = -74.30;
const _tileCache   = new Map();   // "tx_ty" → sales[]
const _tileInflight= new Set();   // tiles currently fetching

function _viewportTiles(bounds) {
  const minTx = Math.floor((bounds.getWest()  - _NYC_MIN_LON) / _TILE_DEG);
  const maxTx = Math.floor((bounds.getEast()  - _NYC_MIN_LON) / _TILE_DEG);
  const minTy = Math.floor((bounds.getSouth() - _NYC_MIN_LAT) / _TILE_DEG);
  const maxTy = Math.floor((bounds.getNorth() - _NYC_MIN_LAT) / _TILE_DEG);
  const tiles = [];
  for (let tx = Math.max(0, minTx); tx <= maxTx; tx++)
    for (let ty = Math.max(0, minTy); ty <= maxTy; ty++)
      tiles.push({ tx, ty, key: `${tx}_${ty}` });
  return tiles;
}

async function fetchSalesForView() {
  if (_cityMode === 'riyadh') return;  // no NYC sales in Riyadh mode
  const tiles   = _viewportTiles(map.getBounds());
  const missing = tiles.filter(t => !_tileCache.has(t.key) && !_tileInflight.has(t.key));

  if (missing.length) {
    missing.forEach(t => _tileInflight.add(t.key));
    await Promise.all(missing.map(async ({ tx, ty, key }) => {
      try {
        const r = await fetch(`${API_BASE}/sales/tile?tx=${tx}&ty=${ty}`);
        _tileCache.set(key, r.ok ? (await r.json()).sales || [] : []);
        // LRU eviction: keep at most 300 tiles (~8 MB) to prevent memory growth
        if (_tileCache.size > 300) _tileCache.delete(_tileCache.keys().next().value);
      } catch (_) { _tileCache.set(key, []); }
      finally     { _tileInflight.delete(key); }
    }));
  }

  const all = tiles.flatMap(t => _tileCache.get(t.key) || []);
  renderSalesBubbles(all);
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
    if (_predictAbort) _predictAbort.abort();
    _predictAbort = new AbortController();

    const res  = await fetch(`${API_BASE}/predict`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
      signal:  _predictAbort.signal,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `API error ${res.status}`);
    }

    const data = await res.json();
    renderResults(data);
    fetchNearby(lat, lon, sqft);

  } catch (err) {
    if (err.name === 'AbortError') return;
    console.error(err);
    alert(`Prediction failed: ${err.message}\n\nMake sure the API server is running at http://localhost:8000`);
  } finally {
    setLoading(false);
  }
});

// ── Riyadh submit ──────────────────────────────────────────────────────
document.getElementById('riyadhForm').addEventListener('submit', async (e) => {
  e.preventDefault();

  const lat  = parseFloat(document.getElementById('riyadhLat').value);
  const lon  = parseFloat(document.getElementById('riyadhLon').value);
  const type = document.getElementById('riyadhType').value;
  const area = parseFloat(document.getElementById('riyadhArea').value);

  if (!lat || !lon) {
    return;
  }
  if (!type) {
    document.getElementById('riyadhTypeGrid').classList.add('invalid');
    return;
  }
  if (!area || area <= 0) {
    document.getElementById('riyadhArea').classList.add('invalid');
    return;
  }

  const submitBtn2 = document.getElementById('riyadhSubmitBtn');
  const btnText2   = document.getElementById('riyadhBtnText');
  const spinner2   = document.getElementById('riyadhSpinner');
  submitBtn2.disabled = true;
  if (btnText2) btnText2.textContent = 'جارٍ التقدير…';
  if (spinner2) spinner2.style.display = 'inline-block';
  document.getElementById('riyadhResults').style.display = 'none';

  try {
    if (_riyadhPredictAbort) _riyadhPredictAbort.abort();
    _riyadhPredictAbort = new AbortController();

    const res = await fetch(`${API_BASE}/predict/riyadh`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ latitude: lat, longitude: lon, property_type: type, area_sqm: area }),
      signal:  _riyadhPredictAbort.signal,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `API error ${res.status}`);
    }
    const data = await res.json();
    renderRiyadhResults(data);
  } catch (err) {
    if (err.name === 'AbortError') return;
    console.error(err);
    alert(`Riyadh prediction failed: ${err.message}\n\nMake sure the API server is running at http://localhost:8000`);
  } finally {
    submitBtn2.disabled = false;
    if (btnText2) btnText2.textContent = '🔍  تقدير السعر / Estimate';
    if (spinner2) spinner2.style.display = 'none';
  }
});

// ── Riyadh result renderer ─────────────────────────────────────────────
function renderRiyadhResults(data) {
  const psqm  = data.predicted_price_sqm;
  const total = data.predicted_total_sar;
  const low   = data.confidence_low_sar;
  const high  = data.confidence_high_sar;
  const sf    = data.spatial_features || {};

  // Price hero — animated
  const sqmEl = document.getElementById('riyadhPriceSqm');
  if (sqmEl) animateSAR(sqmEl, psqm, fmtSAR);

  const totalEl = document.getElementById('riyadhPriceTotal');
  if (totalEl) animateSAR(totalEl, total, fmtSAR);

  const confEl = document.getElementById('riyadhConfidence');
  if (confEl) confEl.textContent = `Range: ${fmtSAR(low)} – ${fmtSAR(high)}`;

  const distEl = document.getElementById('riyadhDistrictLabel');
  if (distEl) {
    const district = data.district_ar || '';
    const medape   = data.medape_pct  || 23.4;
    distEl.textContent = `${district ? district + ' · ' : ''}MedAPE ${medape.toFixed(1)}%`;
  }

  // Spatial grid
  const grid = document.getElementById('riyadhSpatialGrid');
  if (grid && sf) {
    const items = [
      { icon: '🚇', label: 'Metro',        value: sf.dist_metro_m      != null ? fmtDist(sf.dist_metro_m)      : '—' },
      { icon: '🚌', label: 'Bus stop',     value: sf.dist_bus_m        != null ? fmtDist(sf.dist_bus_m)        : '—' },
      { icon: '🏪', label: 'Commercial',   value: sf.commercial_count_1km != null ? `${Math.round(sf.commercial_count_1km)} /1km` : '—' },
      { icon: '🕌', label: 'Mosque',       value: sf.dist_mosque_m     != null ? fmtDist(sf.dist_mosque_m)     : '—' },
      { icon: '🛍️', label: 'Mall',         value: sf.dist_mall_m       != null ? fmtDist(sf.dist_mall_m)       : '—' },
      { icon: '🌿', label: 'Air quality',  value: sf.air_quality_score != null ? `${Math.round(sf.air_quality_score)}/100` : '—' },
      { icon: '🏫', label: 'School',       value: sf.dist_school_m     != null ? fmtDist(sf.dist_school_m)     : '—' },
      { icon: '🏥', label: 'Hospital',     value: sf.dist_hospital_m   != null ? fmtDist(sf.dist_hospital_m)   : '—' },
    ];
    grid.innerHTML = items.map(item => `
      <div class="riyadh-spatial-item">
        <span class="rs-label">${item.icon} ${item.label}</span>
        <span class="rs-val">${item.value}</span>
      </div>`).join('');
  }

  // Map marker popup
  if (marker) {
    marker.bindPopup(
      `<div class="map-popup"><strong>${fmtSAR(psqm)}/sqm</strong><br>${data.property_type || ''}</div>`,
      { maxWidth: 200 }
    ).openPopup();
  }

  // SHAP drivers
  const driversSection = document.getElementById('riyadhDriversSection');
  const driversBars    = document.getElementById('riyadhDriversBars');
  const drivers = data.top_drivers || [];
  if (driversSection && driversBars) {
    if (drivers.length) {
      const maxImpact = Math.max(...drivers.map(d => Math.abs(d.impact)));
      driversBars.innerHTML = drivers.map(drv => {
        const isPos = drv.direction === 'positive';
        const pct   = maxImpact > 0 ? Math.round(Math.abs(drv.impact) / maxImpact * 100) : 0;
        const cls   = isPos ? 'positive' : 'negative';
        const arrow = isPos ? '↑' : '↓';
        const label = drv.description || drv.feature;
        return `<div class="shap-row">
          <span class="shap-arrow ${cls}">${arrow}</span>
          <div class="shap-label-wrap"><span class="shap-label">${label}</span></div>
          <div class="shap-bar-wrap"><div class="shap-bar-fill ${cls}" style="width:${pct}%"></div></div>
          <span class="shap-impact ${cls}">${drv.impact > 0 ? '+' : ''}${drv.impact.toFixed(3)}</span>
        </div>`;
      }).join('');
      driversSection.style.display = '';
    } else {
      driversSection.style.display = 'none';
    }
  }

  showCard(document.getElementById('riyadhResults'), 0);
  document.getElementById('riyadhResults').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Loading state (with skeleton cards) ───────────────────────────────
function setLoading(on) {
  submitBtn.disabled = on;
  btnText.textContent = on ? TR[currentLang].estimating : '🔍 ' + TR[currentLang].estimateBtn;
  spinner.style.display = on ? 'inline-block' : 'none';
  ['skeletonResult','skeletonShap','skeletonSpatial'].forEach(id => {
    document.getElementById(id).style.display = on ? 'block' : 'none';
  });
}

// ── Smooth card show / hide helpers ──────────────────────────────────
function showCard(el, delayMs) {
  el.style.display = 'block';
  el.classList.add('card-animated');
  el.classList.remove('card-in');
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      el.style.transitionDelay = delayMs ? `${delayMs}ms` : '';
      el.classList.add('card-in');
    });
  });
}
function hideCard(el) {
  el.classList.remove('card-in');
  el.style.transitionDelay = '';
  let done = false;
  const finish = () => { if (!done) { done = true; el.style.display = 'none'; } };
  el.addEventListener('transitionend', finish, { once: true });
  setTimeout(finish, 300); // safety fallback
}

// ── SAR price animation (mirrors animatePrice for USD) ────────────────
function animateSAR(el, target, formatter) {
  const start = target * 0.72;
  const dur   = 600;
  const t0    = performance.now();
  function tick(now) {
    const p    = Math.min((now - t0) / dur, 1);
    const ease = 1 - Math.pow(1 - p, 3);
    el.textContent = formatter(Math.round(start + (target - start) * ease));
    if (p < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function hideResults() {
  hideCard(resultCard);
  hideCard(shapCard);
  hideCard(spatialCard);
  hideCard(document.getElementById('marketCard'));
  document.getElementById('avmQcRow').style.display    = 'none';
  const _psfRow = document.getElementById('pricePsfRow');
  if (_psfRow) _psfRow.style.display = 'none';
  // (nycDriversSection removed — shapBars handles top_drivers directly)
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
  // Update model tag if available
  const modelTagEl = document.getElementById('modelTag');
  if (modelTagEl && data.model) {
    modelTagEl.textContent = data.model.includes('v11') ? 'Stack v11 · 4-Model' : data.model.includes('v10') ? 'Stack v10 · 4-Model' : data.model.includes('v9') ? 'Stack v9 · 4-Model' : data.model.includes('v8') ? 'Stack v8 · 4-Model' : data.model.includes('v7') ? 'Stack v7 · 4-Model' : data.model.includes('v6') ? 'Stack v6 · 4-Model' : data.model.includes('v5') ? 'Stack v5 · 4-Model' : data.model.includes('Stack') ? 'Stack v4 · XGB+LGB+CAT' : data.model;
  }
  // Price reveal + count-up animation
  priceMain.classList.remove('price-reveal');
  void priceMain.offsetWidth;
  priceMain.classList.add('price-reveal');
  animatePrice(priceMain, data.predicted_price);

  // Per-sqft / per-sqm secondary row
  const _psfRow = document.getElementById('pricePsfRow');
  if (_psfRow) {
    if (data.price_per_sqft) {
      document.getElementById('pricePsfVal').textContent = `$${Math.round(data.price_per_sqft).toLocaleString()}`;
      document.getElementById('pricePsmVal').textContent = `$${Math.round(data.price_per_sqm).toLocaleString()}`;
      _psfRow.style.display = '';
    } else {
      _psfRow.style.display = 'none';
    }
  }

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

  showCard(resultCard, 0);

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
    _lastDrivers = data.top_drivers;
    const maxImpact = Math.max(...data.top_drivers.map(d => Math.abs(d.impact)));

    data.top_drivers.forEach(drv => {
      const isPos  = drv.direction === 'positive';
      const pct    = Math.round((Math.abs(drv.impact) / maxImpact) * 100);
      const arrow  = isPos ? '↑' : '↓';
      const cls    = isPos ? 'positive' : 'negative';
      const label  = drv.description || drv.feature;
      const nlTip  = shapNlTooltip(drv);

      const row = document.createElement('div');
      row.className = 'shap-row';
      row.title = nlTip;
      row.innerHTML = `
        <span class="shap-arrow ${cls}">${arrow}</span>
        <div class="shap-label-wrap">
          <span class="shap-label">${label}</span>
          <span class="shap-nl-tip">${nlTip}</span>
        </div>
        <div class="shap-bar-wrap">
          <div class="shap-bar-fill ${cls}" style="width:${pct}%"></div>
        </div>
        <span class="shap-impact ${cls}">${drv.impact > 0 ? '+' : ''}${drv.impact.toFixed(3)}</span>
      `;
      shapBars.appendChild(row);
    });
    showCard(shapCard, 80);
    // Reset to bar view (waterfall renders on demand)
    if (_shapView === 'waterfall') setShapView('bars');
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
  showCard(spatialCard, 160);

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

  showCard(nearbyCard, 240);
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

// ── Share / Export ────────────────────────────────────────────────────

/** Toast helper — auto-hides after `ms` milliseconds */
function showToast(msg, ms = 2600) {
  const el = document.getElementById('toastNotif');
  el.textContent = msg;
  el.classList.add('visible');
  clearTimeout(el._timer);
  el._timer = setTimeout(() => el.classList.remove('visible'), ms);
}

/** Build a shareable URL that pre-fills the form when opened */
function _buildShareURL() {
  const lat    = latInput.value;
  const lon    = lonInput.value;
  const borough = document.getElementById('borough').value;
  const bldg    = bldgHidden.value;
  const sqft    = document.getElementById('gross_square_feet').value;
  const age     = document.getElementById('building_age').value;
  const floors  = document.getElementById('numfloors').value;
  const units   = document.getElementById('residential_units').value;

  const base = window.location.origin + window.location.pathname;
  const params = new URLSearchParams();
  if (lat)     params.set('lat',     parseFloat(lat).toFixed(6));
  if (lon)     params.set('lon',     parseFloat(lon).toFixed(6));
  if (borough) params.set('borough', borough);
  if (bldg)    params.set('bldg',    bldg);
  if (sqft)    params.set('sqft',    sqft);
  if (age)     params.set('age',     age);
  if (floors)  params.set('floors',  floors);
  if (units)   params.set('units',   units);
  params.set('autorun', '1');
  return `${base}?${params.toString()}`;
}

function shareEstimate() {
  const url = _buildShareURL();
  navigator.clipboard.writeText(url)
    .then(() => showToast('🔗 Share link copied!'))
    .catch(() => {
      // Fallback for older browsers
      const ta = document.createElement('textarea');
      ta.value = url; ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta); ta.select();
      document.execCommand('copy'); document.body.removeChild(ta);
      showToast('🔗 Share link copied!');
    });
}

function copyResults() {
  if (!_lastPrediction) { showToast('No estimate yet — run a prediction first.', 3000); return; }

  const priceEl   = document.getElementById('priceMain');
  const rangeEl   = document.getElementById('priceRange');
  const contextEl = document.getElementById('priceContext');
  const gradeEl   = document.getElementById('confGradeBadge');
  const compsEl   = document.getElementById('avmCompsLine');
  const flagsEl   = document.getElementById('avmFlagsRow');

  const price    = priceEl.textContent.trim();
  const range    = rangeEl.textContent.replace('Range: ', '').trim();
  const context  = contextEl.textContent.trim();
  const grade    = gradeEl.textContent.trim();
  const comps    = compsEl.textContent.trim();
  const flags    = Array.from(flagsEl.querySelectorAll('.avm-qc-flag'))
                        .map(s => s.textContent.trim()).join('  ');

  // Top drivers
  const driverLines = Array.from(document.querySelectorAll('#shapBars .shap-row'))
    .slice(0, 5)
    .map(row => {
      const arrow  = row.querySelector('.shap-arrow')?.textContent.trim() || '';
      const label  = row.querySelector('.shap-label')?.textContent.trim() || '';
      const impact = row.querySelector('.shap-impact')?.textContent.trim() || '';
      return `  ${arrow} ${label}: ${impact}`;
    }).join('\n');

  const shareURL = _buildShareURL();

  const text = [
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    '  THAMAN NYC Property Valuation',
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
    `📍 ${context}`,
    `💰 Estimated Value:  ${price}`,
    `   Confidence Range: ${range}`,
    grade ? `🏅 Confidence Grade: ${grade}` : '',
    comps ? `   ${comps}` : '',
    flags ? `   Flags: ${flags}` : '',
    '',
    driverLines ? `Top Price Drivers:\n${driverLines}` : '',
    '',
    `🔗 ${shareURL}`,
    '',
    `Generated by THAMAN · AI-Powered Property Valuation`,
    `R² 0.647 · MedAPE 20.23% · 185K NYC sales 2022–2026`,
  ].filter(l => l !== null && l !== undefined).join('\n').replace(/\n{3,}/g, '\n\n');

  navigator.clipboard.writeText(text)
    .then(() => showToast('📋 Results copied to clipboard!'))
    .catch(() => {
      const ta = document.createElement('textarea');
      ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta); ta.select();
      document.execCommand('copy'); document.body.removeChild(ta);
      showToast('📋 Results copied to clipboard!');
    });
}

/** Parse URL params on page load — pre-fills form and optionally auto-submits */
function parseURLParams() {
  const p = new URLSearchParams(window.location.search);
  const lat  = parseFloat(p.get('lat'));
  const lon  = parseFloat(p.get('lon'));
  if (!lat || !lon || !isInNYC(lat, lon)) return;

  // Set coordinates
  latInput.value = lat.toFixed(6);
  lonInput.value = lon.toFixed(6);
  lastValidPos   = [lat, lon];
  locationText.textContent = `${lat.toFixed(5)}, ${lon.toFixed(5)}`;
  locationText.classList.add('selected');

  // Place map marker
  if (!marker) {
    marker = L.marker([lat, lon], { icon: pinIcon, draggable: true }).addTo(map);
    marker.on('dragend', (ev) => {
      const pos = ev.target.getLatLng();
      if (!isInNYC(pos.lat, pos.lng)) { if (lastValidPos) marker.setLatLng(lastValidPos); return; }
      lastValidPos = [pos.lat, pos.lng];
      latInput.value = pos.lat.toFixed(6); lonInput.value = pos.lng.toFixed(6);
      locationText.textContent = `${pos.lat.toFixed(5)}, ${pos.lng.toFixed(5)}`;
      const bc = getBoroughCode(pos.lat, pos.lng); if (bc) boroughSel.value = bc;
    });
  } else {
    marker.setLatLng([lat, lon]);
  }
  map.setView([lat, lon], 15);

  // Set form fields
  const setField = (id, val) => { if (val && document.getElementById(id)) document.getElementById(id).value = val; };
  setField('borough',            p.get('borough'));
  setField('gross_square_feet',  p.get('sqft'));
  setField('building_age',       p.get('age'));
  setField('numfloors',          p.get('floors'));
  setField('residential_units',  p.get('units'));

  const bldg = p.get('bldg');
  if (bldg && _bldgDescs) {
    bldgHidden.value     = bldg;
    bldgSearchInput.value = `${bldg} — ${_bldgDescs[bldg] || bldg}`;
    document.querySelectorAll('.bldgtype-card').forEach(c => c.classList.remove('selected'));
  }

  // Enable submit
  submitBtn.disabled  = false;
  btnText.textContent = '🔍 ' + TR[currentLang].estimateBtn;
  mapHint.classList.add('hidden');

  const bc = getBoroughCode(lat, lon); if (bc) boroughSel.value = bc;

  // Auto-run if requested
  if (p.get('autorun') === '1') {
    // Small delay to let bldgClasses load
    setTimeout(() => { if (!submitBtn.disabled) predictForm.requestSubmit(); }, 800);
  }
}

// ── City Comparison Modal ─────────────────────────────────────────────
function openCityCompare() {
  document.getElementById('cityCompareOverlay').style.display = 'flex';
}
function closeCityCompare(e) {
  if (e && e.target.id !== 'cityCompareOverlay') return;
  document.getElementById('cityCompareOverlay').style.display = 'none';
}

// ── Property Comparison Tool ──────────────────────────────────────────
let _compareResults = { A: null, B: null };

function openCompare() {
  // Reset state
  _compareResults = { A: null, B: null };
  document.getElementById('compareDeltaBanner').style.display = 'none';
  ['A', 'B'].forEach(side => {
    document.getElementById(`compareResult${side}`).classList.remove('visible');
    document.getElementById(`compareErr${side}`).style.display = 'none';
  });

  // Pre-fill Property A from the most recent prediction
  if (_lastPrediction && latInput.value && lonInput.value) {
    document.getElementById('compareLatA').value = latInput.value;
    document.getElementById('compareLonA').value = lonInput.value;
    const copyNum = (srcId, dstId) => {
      const v = document.getElementById(srcId)?.value;
      if (v) document.getElementById(dstId).value = v;
    };
    copyNum('borough',           'compareBoroughA');
    copyNum('gross_square_feet', 'compareSqftA');
    copyNum('building_age',      'compareAgeA');
    copyNum('numfloors',         'compareFloorsA');
    copyNum('residential_units', 'compareUnitsA');
    if (bldgHidden.value) document.getElementById('compareBldgA').value = bldgHidden.value;
    if (addrInput.value)  document.getElementById('compareAddrA').value  = addrInput.value;
  }

  _populateCompareBldgSelects();
  document.getElementById('compareOverlay').style.display = 'flex';
  document.body.style.overflow = 'hidden';
}

function closeCompare(e) {
  // If triggered by overlay click, only close when clicking the backdrop itself
  if (e && e.target.id !== 'compareOverlay') return;
  document.getElementById('compareOverlay').style.display = 'none';
  document.body.style.overflow = '';
}

function _populateCompareBldgSelects() {
  const codes = _bldgClasses.length > 0 ? _bldgClasses : [
    { code:'A1', desc:'Two-story detached' }, { code:'A2', desc:'One-story detached' },
    { code:'A5', desc:'Attached rowhouse' },  { code:'A7', desc:'Mansion / townhouse' },
    { code:'B1', desc:'Two family brick' },   { code:'B2', desc:'Two family frame' },
    { code:'C0', desc:'Three families' },     { code:'C2', desc:'Five–six families' },
    { code:'C4', desc:'Old law tenement' },   { code:'C6', desc:'Cooperative walk-up' },
    { code:'D1', desc:'Elevator apt (semi-fireproof)' }, { code:'D4', desc:'Elevator apt building' },
    { code:'R1', desc:'Condo unit (elevator)' }, { code:'R4', desc:'Condo unit (walk-up)' },
    { code:'S1', desc:'1-family + commercial' }, { code:'S2', desc:'2-family + commercial' },
  ];
  ['A', 'B'].forEach(side => {
    const sel = document.getElementById(`compareBldg${side}`);
    if (sel.options.length > 1) return;   // already populated
    codes.slice(0, 50).forEach(b => {
      const opt = document.createElement('option');
      opt.value = b.code;
      opt.textContent = `${b.code} — ${b.desc}`;
      sel.appendChild(opt);
    });
  });
}

async function geocodeForCompare(side) {
  const addrEl  = document.getElementById(`compareAddr${side}`);
  const errEl   = document.getElementById(`compareErr${side}`);
  const sideBtn = document.querySelector(`#compareCol${side} .compare-addr-btn`);
  const q = addrEl.value.trim();
  if (!q) {
    errEl.style.color = 'var(--red)';
    errEl.textContent = 'Enter an address first.';
    errEl.style.display = 'block';
    return;
  }
  errEl.style.display = 'none';
  sideBtn.textContent = '…';
  sideBtn.disabled    = true;

  try {
    const url = `https://nominatim.openstreetmap.org/search?` +
      `q=${encodeURIComponent(q + ', New York City')}&format=json&limit=1` +
      `&countrycodes=us&bounded=1&viewbox=-74.26,40.47,-73.70,40.92`;
    const res  = await fetch(url, { headers: { 'Accept-Language': 'en' } });
    const data = await res.json();

    if (!data || data.length === 0) {
      errEl.style.color = 'var(--red)';
      errEl.textContent = TR[currentLang].addrNotFound;
      errEl.style.display = 'block';
      return;
    }
    const lat = parseFloat(data[0].lat);
    const lng = parseFloat(data[0].lon);
    if (!isInNYC(lat, lng)) {
      errEl.style.color = 'var(--red)';
      errEl.textContent = TR[currentLang].addrOutOfNYC;
      errEl.style.display = 'block';
      return;
    }

    document.getElementById(`compareLat${side}`).value = lat.toFixed(6);
    document.getElementById(`compareLon${side}`).value = lng.toFixed(6);

    const bc = getBoroughCode(lat, lng);
    if (bc) document.getElementById(`compareBorough${side}`).value = bc;

    errEl.style.color = 'var(--green)';
    errEl.textContent = `✓ ${data[0].display_name.split(',').slice(0, 3).join(',')}`;
    errEl.style.display = 'block';
    setTimeout(() => { errEl.style.display = 'none'; errEl.style.color = ''; }, 3500);

  } catch (err) {
    errEl.style.color = 'var(--red)';
    errEl.textContent = TR[currentLang].addrError;
    errEl.style.display = 'block';
  } finally {
    sideBtn.textContent = '📍';
    sideBtn.disabled    = false;
  }
}

async function runCompareEstimate(side) {
  const lat     = parseFloat(document.getElementById(`compareLat${side}`).value);
  const lon     = parseFloat(document.getElementById(`compareLon${side}`).value);
  const borough = parseInt(document.getElementById(`compareBorough${side}`).value);
  const bldg    = document.getElementById(`compareBldg${side}`).value;
  const sqft    = parseFloat(document.getElementById(`compareSqft${side}`).value);
  const age     = parseFloat(document.getElementById(`compareAge${side}`).value);
  const floors  = parseFloat(document.getElementById(`compareFloors${side}`).value);
  const units   = parseFloat(document.getElementById(`compareUnits${side}`).value) || 1;
  const errEl   = document.getElementById(`compareErr${side}`);
  const btn     = document.getElementById(`compareEstBtn${side}`);

  // Validate
  const missing = [];
  if (!lat || !lon)         missing.push('location');
  if (!borough)             missing.push('borough');
  if (!bldg)                missing.push('building class');
  if (!sqft || sqft <= 0)   missing.push('sq ft');
  if (isNaN(age) || age < 0) missing.push('age');
  if (!floors || floors < 1) missing.push('floors');

  if (missing.length) {
    errEl.style.color   = 'var(--red)';
    errEl.textContent   = `Missing: ${missing.join(', ')}.`;
    errEl.style.display = 'block';
    return;
  }
  errEl.style.display = 'none';

  btn.disabled     = true;
  btn.innerHTML    = '<span>⏳ Estimating…</span>';

  try {
    const payload = {
      latitude: lat, longitude: lon,
      borough:  Math.round(borough),
      bldgclass: bldg,
      gross_square_feet: sqft,
      building_age:      Math.round(age),
      numfloors:         floors,
      residential_units: Math.round(units),
    };

    const res = await fetch(`${API_BASE}/predict`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `API error ${res.status}`);
    }

    const data = await res.json();
    _compareResults[side] = { ...data, _sqft: sqft };
    _renderCompareResult(side, data, sqft);

    if (_compareResults.A && _compareResults.B) _renderCompareDelta();

  } catch (err) {
    errEl.style.color   = 'var(--red)';
    errEl.textContent   = `Error: ${err.message}`;
    errEl.style.display = 'block';
  } finally {
    btn.disabled  = false;
    btn.innerHTML = `<span>🔍 Estimate ${side}</span>`;
  }
}

function _renderCompareResult(side, data, sqft) {
  const el = document.getElementById(`compareResult${side}`);
  el.classList.add('visible');

  document.getElementById(`comparePrice${side}`).textContent = fmt$(data.predicted_price);
  document.getElementById(`compareRange${side}`).textContent =
    `Range: ${fmt$(data.confidence_low)} – ${fmt$(data.confidence_high)}`;

  // Meta row: grade badge + borough name + $/sqft badge
  const qc = data.avm_qc;
  const gradeHtml = qc
    ? `<span class="conf-grade-badge conf-grade-${qc.confidence_grade}" style="width:22px;height:22px;font-size:.73rem">${qc.confidence_grade}</span>`
    : '';
  const psfHtml = sqft > 0
    ? `<span class="compare-psf-badge">$${Math.round(data.predicted_price / sqft).toLocaleString()}/sqft</span>`
    : '';
  document.getElementById(`compareMeta${side}`).innerHTML =
    `${gradeHtml}<span style="font-size:.72rem;color:var(--gray-700)">${data.borough_name || ''}</span>${psfHtml}`;

  // Top 3 SHAP drivers
  const driversEl = document.getElementById(`compareDrivers${side}`);
  driversEl.innerHTML = '';
  const top3 = (data.top_drivers || []).slice(0, 3);
  if (top3.length > 0) {
    const maxImp = Math.max(...top3.map(d => Math.abs(d.impact)));
    top3.forEach(drv => {
      const isPos = drv.direction === 'positive';
      const pct   = Math.round((Math.abs(drv.impact) / maxImp) * 100);
      const row   = document.createElement('div');
      row.className = 'compare-driver-row';
      row.innerHTML = `
        <span class="compare-driver-arrow" style="color:${isPos ? 'var(--green)' : 'var(--red)'}">${isPos ? '↑' : '↓'}</span>
        <span class="compare-driver-label">${drv.description || drv.feature}</span>
        <div class="compare-driver-bar-wrap"><div class="compare-driver-bar ${isPos ? 'positive' : 'negative'}" style="width:${pct}%"></div></div>
        <span class="compare-driver-impact ${isPos ? 'positive' : 'negative'}">${drv.impact > 0 ? '+' : ''}${drv.impact.toFixed(3)}</span>
      `;
      driversEl.appendChild(row);
    });
  }
}

function _renderCompareDelta() {
  const a      = _compareResults.A.predicted_price;
  const b      = _compareResults.B.predicted_price;
  const banner = document.getElementById('compareDeltaBanner');
  const diff   = Math.abs(a - b);
  const pct    = ((diff / Math.min(a, b)) * 100).toFixed(1);

  if (diff < a * 0.02) {
    banner.className   = 'compare-delta-banner compare-delta-equal';
    banner.textContent = `≈ Roughly equal in value (within 2%)`;
  } else if (a > b) {
    banner.className   = 'compare-delta-banner compare-delta-higher';
    banner.textContent = `Property A is ${fmt$(diff)} (+${pct}%) more expensive than Property B`;
  } else {
    banner.className   = 'compare-delta-banner compare-delta-lower';
    banner.textContent = `Property B is ${fmt$(diff)} (+${pct}%) more expensive than Property A`;
  }
  banner.style.display = 'block';
}

// Close comparison modal on Escape key
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    const ov = document.getElementById('compareOverlay');
    if (ov && ov.style.display !== 'none') {
      ov.style.display   = 'none';
      document.body.style.overflow = '';
    }
  }
});

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
    disclaimer:     'Predictions are based on NYC property sales 2022–2026. Model accuracy: median error ±20.23%.',
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
    mapHintRiyadh:  'Click anywhere in Riyadh to place a pin',
    tagline:        'NYC Property Valuation · AI-Powered',
    taglineRiyadh:  'Riyadh Property Valuation · AI-Powered',
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
    nlUp:              'pushed the price UP',
    nlDown:            'pushed the price DOWN',
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
    disclaimer:     'التوقعات مبنية على مبيعات العقارات في مدينة نيويورك 2022–2026. دقة النموذج: متوسط الخطأ ±20.16٪.',
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
    mapHintRiyadh:  'انقر في أي مكان في الرياض لوضع الدبوس',
    tagline:        'تقييم عقارات نيويورك · مدعوم بالذكاء الاصطناعي',
    taglineRiyadh:  'تقييم عقارات الرياض · مدعوم بالذكاء الاصطناعي',
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
    nlUp:              'رفع السعر',
    nlDown:            'خفض السعر',
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
  document.getElementById('headerTagline').textContent = _cityMode === 'riyadh' ? T.taglineRiyadh : T.tagline;
  document.getElementById('analyticsLabel').textContent = T.analytics;
  document.getElementById('mapHintText').textContent   = _cityMode === 'riyadh' ? T.mapHintRiyadh : T.mapHint;

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

// (parseURLParams is now chained onto loadBldgClasses() above)
