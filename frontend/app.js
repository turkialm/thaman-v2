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

// ── Map init ──────────────────────────────────────────────────────────
const map = L.map('map', {
  center:      [40.7128, -74.0060],   // NYC
  zoom:        11,
  zoomControl: true,
});

// OpenStreetMap tile layer (free, no API key)
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  maxZoom: 19,
}).addTo(map);

// Pin marker (emoji-based, no image dependency)
const pinIcon = L.divIcon({
  html:      '<div class="pin-icon">📍</div>',
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

// ── Borough auto-detection from lat/lng (rough bounding boxes) ────────
function guessBoroughFromCoords(lat, lon) {
  // Very rough bounding boxes — just to pre-fill the dropdown helpfully
  if (lat > 40.700 && lat < 40.880 && lon > -74.020 && lon < -73.907) {
    // Manhattan (narrow island)
    if (lon > -74.020 && lon < -73.907 && lat > 40.700 && lat < 40.880) return '1';
  }
  if (lat > 40.785 && lat < 40.915 && lon > -73.934 && lon < -73.766) return '2'; // Bronx
  if (lat > 40.570 && lat < 40.739 && lon > -74.042 && lon < -73.834) return '3'; // Brooklyn
  if (lat > 40.541 && lat < 40.800 && lon > -73.962 && lon < -73.700) return '4'; // Queens
  if (lat > 40.477 && lat < 40.651 && lon > -74.259 && lon < -74.034) return '5'; // Staten Island
  return '';
}

// ── Map click ─────────────────────────────────────────────────────────
map.on('click', (e) => {
  const { lat, lng } = e.latlng;

  // Update hidden inputs
  latInput.value = lat.toFixed(6);
  lonInput.value = lng.toFixed(6);

  // Update location display
  locationText.textContent = `${lat.toFixed(5)}, ${lng.toFixed(5)}`;
  locationText.classList.add('selected');

  // Place/move marker
  if (marker) {
    marker.setLatLng([lat, lng]);
  } else {
    marker = L.marker([lat, lng], { icon: pinIcon, draggable: true }).addTo(map);

    // Draggable marker: update inputs when dragged
    marker.on('dragend', (ev) => {
      const pos = ev.target.getLatLng();
      latInput.value = pos.lat.toFixed(6);
      lonInput.value = pos.lng.toFixed(6);
      locationText.textContent = `${pos.lat.toFixed(5)}, ${pos.lng.toFixed(5)}`;
      boroughSel.value = guessBoroughFromCoords(pos.lat, pos.lng) || boroughSel.value;
    });
  }

  // Auto-guess borough
  const guessed = guessBoroughFromCoords(lat, lng);
  if (guessed && !boroughSel.value) boroughSel.value = guessed;

  // Enable button, hide map hint
  submitBtn.disabled = false;
  btnText.textContent = '🔍  Estimate Price';
  mapHint.classList.add('hidden');
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
  if (!bldg)           { bldgSearchInput.classList.add('invalid'); valid = false; }
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

  } catch (err) {
    console.error(err);
    alert(`Prediction failed: ${err.message}\n\nMake sure the API server is running at http://localhost:8000`);
  } finally {
    setLoading(false);
  }
});

// ── Loading state ─────────────────────────────────────────────────────
function setLoading(on) {
  submitBtn.disabled = on;
  btnText.textContent = on ? 'Estimating…' : '🔍  Estimate Price';
  spinner.style.display = on ? 'inline-block' : 'none';
}

function hideResults() {
  resultCard.style.display  = 'none';
  shapCard.style.display    = 'none';
  spatialCard.style.display = 'none';
}

// ── Render results ────────────────────────────────────────────────────
function renderResults(data) {
  // ── Price card ────────────────────────────────────────────────────
  priceMain.textContent    = `$${data.predicted_price.toLocaleString()}`;
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
  confFill.style.width = '100%';
  const markerPct = Math.round(((pred - low) / span) * 100);
  confMarker.style.left = `${markerPct}%`;
  confBarWrap.style.display = 'block';

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

  spatialGrid.innerHTML = spatialItems.map(item => `
    <div class="spatial-item">
      <div class="spatial-item-icon">${item.icon}</div>
      <div class="spatial-item-label">${item.label}</div>
      <div class="spatial-item-value">${item.value}
        ${item.unit ? `<span class="spatial-item-unit">${item.unit}</span>` : ''}
      </div>
    </div>
  `).join('');
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
