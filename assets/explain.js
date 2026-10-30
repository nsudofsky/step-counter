// "How it decides" page. Same socket messages as app.js (the old page), plus "dataset":
//   board -> page: "live" (twice a second), "status", "dataset", "record", "train"
//   page -> board: "hello", "dataset", "record", "stop", "train", "delete", "matrix_model"
const socket = io(window.location.origin);

const EMOJI = { standing: '🧍', walking: '🚶', running: '🏃', biking: '🚴' };
const ACTS = ['standing', 'walking', 'running', 'biking'];
const HISTORY = 120;                                   // 60 s of predictions at 2 per second
const $ = (id) => document.getElementById(id);
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(`--${n}`).trim();
const color = (a) => cssVar(a) || '#888';

let status = null, live = {}, dataset = null;
const hist = { peak: [], tree: [] };
let activity = 'standing';

// ---------------------------------------------------------------- plain-English feature names
const NICE = {
  acc_jerk: 'impact sharpness', cadence: 'rhythm (per min)',
  acc_mag_mean: 'average total acceleration', acc_mag_std: 'how much total accel wobbles',
  acc_mag_max: 'biggest total acceleration', acc_mag_p90: 'big total acceleration',
  gyro_mag_mean: 'average spin speed', gyro_mag_std: 'how much spin wobbles',
  gyro_mag_max: 'fastest spin', gyro_mag_p90: 'fast spin',
};
const UNIT = { acc_jerk: 'g/s', cadence: '/min' };
function nice(f) {
  if (NICE[f]) return NICE[f];
  const [ch, stat] = f.split('_');
  const what = ch[0] === 'a' ? `acceleration along ${ch[1]}` : `spin around ${ch[1]}`;
  return { mean: 'average', std: 'wobble of', max: 'highest', min: 'lowest', range: 'range of' }[stat] + ' ' + what;
}
function unit(f) {
  if (UNIT[f]) return UNIT[f];
  return f.startsWith('g') ? '°/s' : 'g';
}
const fmt = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));

// ---------------------------------------------------------------- tabs
function showTab(name) {
  for (const b of document.querySelectorAll('.tabs button')) b.classList.toggle('on', b.dataset.tab === name);
  $('tab-live').hidden = name !== 'live';
  $('tab-data').hidden = name !== 'data';
  try { localStorage.setItem('gait-tab', name); } catch (e) { /* private mode */ }
  if (name === 'data') { drawD1(); drawD2(); }
}
for (const b of document.querySelectorAll('.tabs button')) b.onclick = () => showTab(b.dataset.tab);
let firstTab = location.hash === '#data' ? 'data' : location.hash === '#live' ? 'live' : null;
try { firstTab = firstTab || localStorage.getItem('gait-tab'); } catch (e) { /* private mode */ }
showTab(firstTab || 'live');

// ---------------------------------------------------------------- sound for recording
let audio = null;
function beep(freq = 880, ms = 120) {
  try {
    audio = audio || new (window.AudioContext || window.webkitAudioContext)();
    const o = audio.createOscillator(), g = audio.createGain();
    o.frequency.value = freq; g.gain.value = 0.2;
    o.connect(g).connect(audio.destination);
    o.start(); o.stop(audio.currentTime + ms / 1000);
  } catch (e) { /* no audio */ }
}

// ---------------------------------------------------------------- socket
socket.on('connect', () => { socket.emit('hello', {}); socket.emit('dataset', {}); });
socket.on('disconnect', () => setPill(false, 'disconnected'));

let lastModelKey = null;
socket.on('status', (s) => {
  status = s;
  const key = JSON.stringify(s.models && s.models.tree) + s.recordings.map((r) => r.file).join();
  if (key !== lastModelKey && lastModelKey !== null) socket.emit('dataset', {});   // retrained / new data
  lastModelKey = key;
  for (const b of document.querySelectorAll('.matrix-pick button'))
    b.classList.toggle('on', b.dataset.model === s.matrix_model);
  renderMiniTree();
  renderCounts();
  renderScores();
  renderActs();
  renderRecs();
  lastPath = null;
});

socket.on('dataset', (d) => {
  dataset = d;
  renderExamples();
  fillSelects();
  drawD1();
  drawD2();
  lastPath = null;
});

socket.on('live', (m) => {
  live = m;
  setPill(m.imu, m.imu ? `IMU ${m.rate} Hz` : m.dead ? 'sensor sending zeros' : 'no IMU');
  if (m.peak_label) {
    pushHist('peak', m.peak_label);
    pushHist('tree', m.tree_label);
  }
  renderHero();
  drawTimeline('peak');
  drawTimeline('tree');
  drawSensor();
  renderModel1();
  renderModel2();
  if (!$('tab-data').hidden) { drawD1(); drawD2(); }
});

socket.on('record', (r) => {
  const el = $('rec-status');
  if (r.phase === 'countdown') { el.textContent = `Starting in ${r.left}…`; beep(660, 100); }
  else if (r.phase === 'recording') { el.textContent = `● Recording ${activity}: ${r.left} s left`; }
  else { el.textContent = r.file ? `Saved ✓ ${r.file}` : r.error || 'Cancelled'; beep(r.file ? 990 : 330, 400); }
});
socket.on('train', (t) => {
  $('rec-status').textContent = { training: 'Training…', done: 'Trained ✓', error: `Error: ${t.error}` }[t.phase];
});

for (const b of document.querySelectorAll('.matrix-pick button'))
  b.onclick = () => socket.emit('matrix_model', { model: b.dataset.model });

function setPill(ok, text) {
  $('imu').textContent = text;
  $('imu').className = `pill ${ok ? 'good' : 'bad'}`;
}
function pushHist(model, label) {
  hist[model].push(label);
  if (hist[model].length > HISTORY) hist[model].shift();
}

// ---------------------------------------------------------------- canvas helper (sharp on retina)
function canvas(id) {
  const c = $(id);
  const dpr = window.devicePixelRatio || 1;
  c._h = c._h || c.height;
  const w = c.clientWidth;
  if (!w) return [null, 0, 0];
  if (c.width !== Math.round(w * dpr)) {
    c.width = Math.round(w * dpr);
    c.height = Math.round(c._h * dpr);
    c.style.height = `${c._h}px`;
  }
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, c._h);
  ctx.font = '13px -apple-system, BlinkMacSystemFont, sans-serif';
  return [ctx, w, c._h];
}

// ---------------------------------------------------------------- hero + timeline
function renderHero() {
  const model = status ? status.matrix_model : 'tree';
  const label = model === 'peak' ? live.peak_label : live.tree_label;
  const ans = document.querySelector('.answer');
  if (!live.imu) {
    $('hero-emoji').textContent = '🔌';
    $('hero-word').textContent = live.dead ? 'sensor sending zeros' : 'no sensor data';
    $('hero-word').style.color = '';
    ans.style.background = '';
    $('agree').textContent = '';
    return;
  }
  $('hero-kicker').textContent = `The LED matrix says (Model ${model === 'peak' ? 1 : 2})`;
  $('hero-emoji').textContent = EMOJI[label] || '❔';
  $('hero-word').textContent = label || 'not trained yet';
  $('hero-word').style.color = label ? color(label) : '';
  ans.style.background = label ? `color-mix(in srgb, ${color(label)} 13%, var(--card))` : '';
  ans.style.borderColor = label ? color(label) : '';
  if (live.peak_label && live.tree_label) {
    $('agree').innerHTML = live.peak_label === live.tree_label
      ? '✓ Both models agree'
      : `Models disagree: Model 1 says <b>${live.peak_label}</b>, Model 2 says <b>${live.tree_label}</b>`;
  }
}

function drawTimeline(model) {
  const [ctx, w, h] = canvas(`tl-${model}`);
  if (!ctx) return;
  const hs = hist[model];
  const bw = w / HISTORY;
  hs.forEach((l, i) => {
    ctx.fillStyle = color(l);
    ctx.fillRect(w - (hs.length - i) * bw, 0, bw + 0.5, h);
  });
}

// ---------------------------------------------------------------- 1. sensor
function drawSensor() {
  const [ctx, w, h] = canvas('sensor');
  if (!ctx || !live.trace_mag) return;
  const p = status && status.models ? status.models.peak : null;
  const tr = live.trace || [];
  const mag = live.trace_mag;
  const all = [...tr, ...mag, ...(p ? p.thresholds : []), 0];
  let lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo) * 0.1 + 0.1; lo -= pad; hi += pad;
  const L = 44, R = 8, T = 8, B = 22;
  const X = (i, n) => L + (w - L - R) * i / (n - 1);
  const Y = (v) => T + (h - T - B) * (1 - (v - lo) / (hi - lo));
  // axis
  ctx.fillStyle = cssVar('muted'); ctx.textAlign = 'right';
  for (const v of [lo + pad, 0, hi - pad]) ctx.fillText(`${v.toFixed(1)} g`, L - 6, Y(v) + 4);
  ctx.strokeStyle = cssVar('line'); ctx.beginPath(); ctx.moveTo(L, Y(0)); ctx.lineTo(w - R, Y(0)); ctx.stroke();
  // model 1 thresholds
  if (p) p.thresholds.forEach((t, i) => {
    ctx.strokeStyle = color(p.order[i + 1]); ctx.setLineDash([5, 5]);
    ctx.beginPath(); ctx.moveTo(L, Y(t)); ctx.lineTo(w - R, Y(t)); ctx.stroke();
  });
  ctx.setLineDash([]);
  // total accel (thin) + model-1 direction (bold)
  const line = (arr, col, lw) => {
    ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.beginPath();
    arr.forEach((v, i) => (i ? ctx.lineTo(X(i, arr.length), Y(v)) : ctx.moveTo(X(i, arr.length), Y(v))));
    ctx.stroke(); ctx.lineWidth = 1;
  };
  line(mag, cssVar('muted'), 1.2);
  if (tr.length) {
    line(tr, cssVar('ink'), 2.4);
    const im = tr.indexOf(Math.max(...tr));
    ctx.fillStyle = color(live.peak_label || 'running');
    ctx.beginPath(); ctx.arc(X(im, tr.length), Y(tr[im]), 7, 0, 7); ctx.fill();
    ctx.fillStyle = cssVar('ink'); ctx.textAlign = 'center';
    ctx.fillText('peak', X(im, tr.length), Math.max(Y(tr[im]) - 12, 14));
  }
  ctx.textAlign = 'left'; ctx.fillStyle = cssVar('muted');
  ctx.fillText(p ? `━ acceleration along ${p.direction} (what Model 1 watches)    ─ total acceleration    - - Model 1 thresholds` : '─ total acceleration', L, h - 4);
}

// ---------------------------------------------------------------- 2. model 1
let shownPeak = null;
function m1Range(p) {
  if (dataset && dataset.windows.length) {
    const v = dataset.windows.map((w) => w.peak);
    const lo = Math.min(...v), hi = Math.max(...v), pad = (hi - lo) * 0.06;
    return [lo - pad, hi + pad];
  }
  const t = p.thresholds, span = Math.max(t[t.length - 1] - t[0], 0.5);
  return [t[0] - span * 0.5, t[t.length - 1] + span * 0.5];
}

function renderModel1() {
  if (!status || !status.models || live.peak === undefined) return;
  const p = status.models.peak;
  $('m1-sub').textContent = `Only watches the biggest push along ${p.direction} in the window. Three thresholds split it into four activities.`;
  $('m1-value').textContent = live.peak.toFixed(2);
  let i = p.thresholds.findIndex((t) => live.peak <= t);   // where this window lands (before smoothing)
  if (i < 0) i = p.order.length - 1;
  const raw = p.order[i];
  const below = p.thresholds[i - 1], above = p.thresholds[i];
  let why = `${live.peak.toFixed(2)} g is `;
  if (below === undefined) why += `below ${above.toFixed(2)} g`;
  else if (above === undefined) why += `above ${below.toFixed(2)} g`;
  else why += `between ${below.toFixed(2)} and ${above.toFixed(2)} g`;
  $('m1-because').innerHTML = `${why} → <b style="color:${color(raw)}">${EMOJI[raw]} ${raw}</b>`;
  animateLine();
}

function animateLine() {
  const target = live.peak;
  if (shownPeak === null) shownPeak = target;
  const step = () => {
    shownPeak += (target - shownPeak) * 0.25;
    drawLine(shownPeak);
    if (Math.abs(target - shownPeak) > 0.005 && live.peak === target) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function drawLine(value) {
  const [ctx, w, h] = canvas('m1-line');
  if (!ctx || !status || !status.models) return;
  const p = status.models.peak;
  const [lo, hi] = m1Range(p);
  const X = (v) => 8 + (w - 16) * (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo);
  const edges = [lo, ...p.thresholds, hi];
  const current = p.order[edges.findIndex((e, k) => k > 0 && value <= e) - 1] || p.order[p.order.length - 1];
  p.order.forEach((a, k) => {
    const x0 = X(edges[k]), x1 = X(edges[k + 1]);
    ctx.globalAlpha = a === current ? 1 : 0.28;
    ctx.fillStyle = color(a);
    ctx.beginPath(); ctx.roundRect(x0 + 1, 34, x1 - x0 - 2, 30, 6); ctx.fill();
    ctx.globalAlpha = 1;
    if (x1 - x0 > 46) {
      ctx.fillStyle = a === current ? '#fff' : cssVar('ink');
      ctx.textAlign = 'center'; ctx.font = '600 13px -apple-system, sans-serif';
      ctx.fillText(`${EMOJI[a]} ${x1 - x0 > 100 ? a : ''}`, (x0 + x1) / 2, 54);
    }
  });
  ctx.font = '12px -apple-system, sans-serif'; ctx.fillStyle = cssVar('muted'); ctx.textAlign = 'center';
  p.thresholds.forEach((t) => {
    ctx.fillStyle = cssVar('ink'); ctx.fillRect(X(t) - 1, 30, 2, 38);
    ctx.fillStyle = cssVar('muted'); ctx.fillText(`${t.toFixed(2)}`, X(t), 84);
  });
  const x = X(value);
  ctx.fillStyle = cssVar('ink');
  ctx.beginPath(); ctx.moveTo(x, 30); ctx.lineTo(x - 9, 12); ctx.lineTo(x + 9, 12); ctx.fill();
  ctx.font = '700 13px -apple-system, sans-serif';
  ctx.fillText('you', Math.min(Math.max(x, 20), w - 20), 9);
  ctx.font = '12px -apple-system, sans-serif'; ctx.fillStyle = cssVar('muted');
  ctx.fillText(`peak acceleration along ${p.direction} (g)`, w / 2, h - 6);
}

// ---------------------------------------------------------------- 3. model 2
let lastPath = null;
function featRange(f, thr) {
  if (dataset && dataset.windows.length && dataset.windows[0].f[f] !== undefined) {
    const v = dataset.windows.map((w) => w.f[f]);
    const lo = Math.min(...v), hi = Math.max(...v), pad = (hi - lo) * 0.05;
    return [lo - pad, hi + pad];
  }
  return thr > 0 ? [0, thr * 2] : [thr * 2, 0];
}

function renderModel2() {
  if (!status || !status.models || !live.tree_path) return;
  const key = live.tree_path.join();
  const ol = $('m2-steps');
  let node = status.models.tree.root;
  if (key !== lastPath) {                      // new route through the tree -> rebuild the steps
    ol.innerHTML = '';
    live.tree_path.forEach((ans, k) => {
      const li = document.createElement('li');
      li.innerHTML = `<div class="q" data-k="${k}">
          <div class="text">Is the <u>${nice(node.feature)}</u> ≤ ${fmt(node.threshold)} ${unit(node.feature)}?
            <span class="chip ${ans}">${ans.toUpperCase()}</span></div>
          <div class="gauge"><div class="fill"></div><div class="thr"></div><div class="dot"></div></div>
          <div class="nums"><span>now <b class="val"></b></span><span>cut point ${fmt(node.threshold)}</span></div>
        </div><div class="arrow">↓</div>`;
      ol.appendChild(li);
      node = node[ans];
    });
    const leaf = document.createElement('li');
    leaf.innerHTML = `<div class="leafcard" style="background:${color(node.leaf)}">${EMOJI[node.leaf] || ''} ${node.leaf}</div>`;
    ol.appendChild(leaf);
    lastPath = key;
    node = status.models.tree.root;
  }
  // every tick: slide the dots to the newest values
  live.tree_path.forEach((ans, k) => {
    const q = ol.querySelector(`.q[data-k="${k}"]`);
    if (!q) return;
    const v = live.tree_values[node.feature];
    const [lo, hi] = featRange(node.feature, node.threshold);
    const P = (x) => `${100 * (Math.max(lo, Math.min(hi, x)) - lo) / (hi - lo)}%`;
    q.querySelector('.thr').style.left = P(node.threshold);
    const fill = q.querySelector('.fill');          // shade the side the answer is on
    fill.style.background = `var(--${ans})`;
    fill.style.left = ans === 'yes' ? '0' : P(node.threshold);
    fill.style.right = ans === 'yes' ? `calc(100% - ${P(node.threshold)})` : '0';
    if (v !== undefined) {
      q.querySelector('.dot').style.left = P(v);
      q.querySelector('.val').textContent = `${fmt(v)} ${unit(node.feature)}`;
    }
    node = node[ans];
  });
  highlightMiniTree();
}

function renderMiniTree() {
  const el = $('m2-tree');
  if (!status.models) { el.innerHTML = '<p class="sub">No model yet.</p>'; $('m2-steps').innerHTML = ''; return; }
  el.innerHTML = '';
  el.appendChild(treeNode(status.models.tree.root, ''));
}
function treeNode(n, path) {
  const d = document.createElement('div');
  d.className = 'node'; d.dataset.path = path;
  if (n.leaf) {
    d.innerHTML = `<div class="tleaf" style="background:${color(n.leaf)}">${EMOJI[n.leaf] || ''} ${n.leaf}</div>`;
    return d;
  }
  d.innerHTML = `<div class="tq">${nice(n.feature)} ≤ ${fmt(n.threshold)}?</div>`;
  const kids = document.createElement('div'); kids.className = 'kids';
  for (const ans of ['yes', 'no']) {
    const b = document.createElement('div'); b.className = 'branch';
    b.innerHTML = `<span class="yn">${ans}</span>`;
    b.appendChild(treeNode(n[ans], path + (ans === 'yes' ? 'Y' : 'N')));
    kids.appendChild(b);
  }
  d.appendChild(kids);
  return d;
}
function highlightMiniTree() {
  const p = live.tree_path.map((a) => (a === 'yes' ? 'Y' : 'N')).join('');
  for (const n of document.querySelectorAll('#m2-tree .node')) {
    const on = p.startsWith(n.dataset.path);
    n.classList.toggle('active', on);
    n.firstElementChild.classList.toggle('dim', !on);
  }
}

// ---------------------------------------------------------------- data tab
function renderCounts() {
  const el = $('counts');
  el.innerHTML = ACTS.map((a) => {
    const recs = status.recordings.filter((r) => r.activity === a && !r.dead);
    const secs = recs.reduce((s, r) => s + r.seconds, 0);
    const people = [...new Set(recs.map((r) => r.person))];
    return `<div class="count" style="background:${color(a)}">
      <div class="e">${EMOJI[a]}</div><div class="n">${recs.length} rec${recs.length === 1 ? '' : 's'}</div>
      <div class="t">${a} · ${Math.round(secs / 6) / 10} min · ${people.join(', ') || 'nobody yet'}</div></div>`;
  }).join('');
}

function renderExamples() {
  const el = $('examples');
  const ex = dataset.examples;
  const acts = ACTS.filter((a) => ex[a]);
  if (!acts.length) { el.innerHTML = '<p class="sub">No recordings yet.</p>'; return; }
  const all = acts.flatMap((a) => [...ex[a].mag, ...ex[a].axis]);
  const lo = Math.min(...all), hi = Math.max(...all);
  el.innerHTML = acts.map((a) => `<div class="example"><div class="h" style="color:${color(a)}">${EMOJI[a]} ${a}</div>
    <canvas id="ex-${a}" height="110"></canvas></div>`).join('');
  for (const a of acts) {
    const [ctx, w, h] = canvas(`ex-${a}`);
    if (!ctx) continue;
    const Y = (v) => 4 + (h - 8) * (1 - (v - lo) / (hi - lo));
    const draw = (arr, col, lw) => {
      ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.beginPath();
      arr.forEach((v, i) => { const x = w * i / (arr.length - 1); i ? ctx.lineTo(x, Y(v)) : ctx.moveTo(x, Y(v)); });
      ctx.stroke();
    };
    ctx.strokeStyle = cssVar('line'); ctx.beginPath(); ctx.moveTo(0, Y(0)); ctx.lineTo(w, Y(0)); ctx.stroke();
    draw(ex[a].mag, cssVar('muted'), 1);
    draw(ex[a].axis, color(a), 2);
  }
  const dir = status && status.models ? status.models.peak.direction : '';
  $('ex-sub').textContent = `2.5 s from the middle of one recording of each. Colored = acceleration along ${dir} (Model 1's direction), gray = total acceleration. Same scale on all four.`;
}

function drawD1() {
  const [ctx, w, h] = canvas('d1');
  if (!ctx || !dataset || !status || !status.models) return;
  const p = status.models.peak;
  const [lo, hi] = m1Range(p);
  const L = 100, R = 16, T = 26, B = 34;
  const X = (v) => L + (w - L - R) * (v - lo) / (hi - lo);
  const rowH = (h - T - B) / p.order.length;
  const RowY = (k) => T + rowH * (p.order.length - 1 - k) + rowH / 2;
  p.order.forEach((a, k) => {
    ctx.fillStyle = color(a); ctx.textAlign = 'right'; ctx.font = '600 14px -apple-system, sans-serif';
    ctx.fillText(`${EMOJI[a]} ${a}`, L - 10, RowY(k) + 5);
  });
  ctx.globalAlpha = 0.55;
  dataset.windows.forEach((d, i) => {
    const k = p.order.indexOf(d.a);
    if (k < 0) return;
    const jitter = (((i * 7919) % 1000) / 1000 - 0.5) * rowH * 0.6;
    ctx.fillStyle = color(d.a);
    ctx.beginPath(); ctx.arc(X(d.peak), RowY(k) + jitter, 3.2, 0, 7); ctx.fill();
  });
  ctx.globalAlpha = 1;
  ctx.font = '12px -apple-system, sans-serif'; ctx.textAlign = 'center';
  p.thresholds.forEach((t) => {
    ctx.strokeStyle = cssVar('ink'); ctx.setLineDash([5, 5]);
    ctx.beginPath(); ctx.moveTo(X(t), T - 4); ctx.lineTo(X(t), h - B + 4); ctx.stroke();
    ctx.fillStyle = cssVar('ink'); ctx.fillText(`${t.toFixed(2)} g`, X(t), h - B + 18);
  });
  ctx.setLineDash([]);
  ctx.fillStyle = cssVar('muted'); ctx.fillText(`peak acceleration along ${p.direction} in each 2.5 s window (g)`, (L + w) / 2, h - 2);
  if (live.peak !== undefined) {
    const x = X(Math.max(lo, Math.min(hi, live.peak)));
    ctx.fillStyle = cssVar('ink');
    ctx.beginPath(); ctx.moveTo(x, T - 2); ctx.lineTo(x - 8, T - 16); ctx.lineTo(x + 8, T - 16); ctx.fill();
    ctx.strokeStyle = cssVar('ink'); ctx.globalAlpha = 0.4;
    ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, h - B); ctx.stroke(); ctx.globalAlpha = 1;
    ctx.font = '700 12px -apple-system, sans-serif'; ctx.fillText('you', x, T - 19);
  }
}

function fillSelects() {
  if (!dataset.windows.length || !status || !status.models) return;
  const names = Object.keys(dataset.windows[0].f);
  // default axes: the tree's first two different questions
  const used = [];
  const queue = [status.models.tree.root];
  while (queue.length) {
    const n = queue.shift();
    if (n.leaf) continue;
    if (!used.includes(n.feature)) used.push(n.feature);
    queue.push(n.yes, n.no);
  }
  for (const [id, def] of [['d2-x', used[0]], ['d2-y', used[1] || names.find((n) => n !== used[0])]]) {
    const s = $(id);
    const keep = s.value;
    s.innerHTML = names.map((n) => `<option value="${n}">${nice(n)}${used.includes(n) ? ' ★' : ''}</option>`).join('');
    s.value = names.includes(keep) ? keep : def;
    s.onchange = drawD2;
  }
  $('legend').innerHTML = ACTS.map((a) => `<span><i style="background:${color(a)}"></i>${a}</span>`).join('') +
    '<span class="sub">★ = a question the tree actually asks</span>';
}

function treeCuts(node, f, out = []) {
  if (node.leaf) return out;
  if (node.feature === f) out.push(node.threshold);
  treeCuts(node.yes, f, out); treeCuts(node.no, f, out);
  return out;
}

function drawD2() {
  const [ctx, w, h] = canvas('d2');
  if (!ctx || !dataset || !dataset.windows.length || !status || !status.models) return;
  const fx = $('d2-x').value, fy = $('d2-y').value;
  if (!fx || !fy) return;
  const xs = dataset.windows.map((d) => d.f[fx]), ys = dataset.windows.map((d) => d.f[fy]);
  const rng = (v) => {                       // 1st-99th percentile so a few wild windows don't squash the rest
    const s = [...v].sort((a, b) => a - b);
    const lo = s[Math.floor(s.length * 0.01)], hi = s[Math.ceil(s.length * 0.99) - 1], p = (hi - lo) * 0.06 || 1;
    return [lo - p, hi + p];
  };
  const [x0, x1] = rng(xs), [y0, y1] = rng(ys);
  const L = 56, R = 14, T = 12, B = 40;
  const X = (v) => L + (w - L - R) * (Math.max(x0, Math.min(x1, v)) - x0) / (x1 - x0);
  const Y = (v) => T + (h - T - B) * (1 - (Math.max(y0, Math.min(y1, v)) - y0) / (y1 - y0));
  ctx.strokeStyle = cssVar('line');
  ctx.strokeRect(L, T, w - L - R, h - T - B);
  ctx.fillStyle = cssVar('muted'); ctx.textAlign = 'center';
  ctx.fillText(`${nice(fx)} (${unit(fx)}) →`, (L + w) / 2, h - 8);
  ctx.save(); ctx.translate(14, (T + h - B) / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText(`${nice(fy)} (${unit(fy)}) →`, 0, 0); ctx.restore();
  ctx.textAlign = 'right';
  ctx.fillText(fmt(y0), L - 6, h - B); ctx.fillText(fmt(y1), L - 6, T + 10);
  ctx.textAlign = 'center';
  ctx.fillText(fmt(x0), L, h - B + 14); ctx.fillText(fmt(x1), w - R, h - B + 14);
  ctx.globalAlpha = 0.6;
  dataset.windows.forEach((d) => {
    ctx.fillStyle = color(d.a);
    ctx.beginPath(); ctx.arc(X(d.f[fx]), Y(d.f[fy]), 3.4, 0, 7); ctx.fill();
  });
  ctx.globalAlpha = 1;
  ctx.strokeStyle = cssVar('ink'); ctx.setLineDash([6, 5]); ctx.lineWidth = 1.5;
  ctx.fillStyle = cssVar('ink'); ctx.font = '12px -apple-system, sans-serif';
  for (const t of treeCuts(status.models.tree.root, fx)) {
    ctx.beginPath(); ctx.moveTo(X(t), T); ctx.lineTo(X(t), h - B); ctx.stroke();
    ctx.fillText(`cut ${fmt(t)}`, X(t), T + 12);
  }
  ctx.textAlign = 'left';
  for (const t of treeCuts(status.models.tree.root, fy)) {
    ctx.beginPath(); ctx.moveTo(L, Y(t)); ctx.lineTo(w - R, Y(t)); ctx.stroke();
    ctx.fillText(`cut ${fmt(t)}`, L + 4, Y(t) - 4);
  }
  ctx.setLineDash([]); ctx.lineWidth = 1;
  const v = live.tree_values || {};
  if (v[fx] !== undefined && v[fy] !== undefined) {
    const x = X(v[fx]), y = Y(v[fy]);
    const pulse = 10 + 3 * Math.sin(Date.now() / 200);
    ctx.fillStyle = color(live.tree_label || 'standing');
    ctx.beginPath(); ctx.arc(x, y, 8, 0, 7); ctx.fill();
    ctx.strokeStyle = cssVar('ink'); ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(x, y, pulse, 0, 7); ctx.stroke(); ctx.lineWidth = 1;
    ctx.fillStyle = cssVar('ink'); ctx.font = '700 13px -apple-system, sans-serif'; ctx.textAlign = 'left';
    ctx.fillText('you', x + 16, y + 4);
  }
}

function renderScores() {
  const cv = status.models && status.models.cv;
  if (!cv) { $('scores').innerHTML = '<p class="sub">Train to see scores.</p>'; $('cv-sub').textContent = ''; return; }
  $('cv-sub').textContent = cv.scheme === 'leave-one-person-out'
    ? `Each person was left out in turn and tested on a model trained on everyone else (${cv.people.join(', ')}). That's the same test your peers will give it.`
    : 'Each recording was cut into fifths, and each fifth was tested on a model trained on the rest.';
  const table = (cm) => '<table class="cm"><tr><th>really ↓ / said →</th>' +
    cv.labels.map((l) => `<th>${EMOJI[l]}</th>`).join('') + '</tr>' +
    cm.map((row, i) => `<tr><th>${EMOJI[cv.labels[i]]} ${cv.labels[i]}</th>` + row.map((v, j) => {
      const tot = row.reduce((s, x) => s + x, 0) || 1;
      const bg = v ? `color-mix(in srgb, ${i === j ? color(cv.labels[i]) : 'var(--no)'} ${Math.round(15 + 70 * v / tot)}%, transparent)` : 'transparent';
      return `<td style="background:${bg}">${v}</td>`;
    }).join('') + '</tr>').join('') + '</table>';
  $('scores').innerHTML = `
    <div class="score"><div class="lbl">Model 1 · peak along ${status.models.peak.direction}</div>
      <div class="acc">${Math.round(cv.peak.accuracy * 100)}%</div>${table(cv.peak.confusion)}</div>
    <div class="score"><div class="lbl">Model 2 · decision tree</div>
      <div class="acc">${Math.round(cv.tree.accuracy * 100)}%</div>${table(cv.tree.confusion)}</div>`;
}

function renderActs() {
  $('acts').innerHTML = '';
  for (const a of ACTS) {
    const b = document.createElement('button');
    b.textContent = `${EMOJI[a]} ${a}`;
    if (a === activity) { b.classList.add('sel'); b.style.background = color(a); }
    b.onclick = () => { activity = a; renderActs(); };
    $('acts').appendChild(b);
  }
}
$('rec').onclick = () => {
  beep(1, 1);
  socket.emit('record', { activity, person: $('person').value, seconds: Number($('seconds').value) });
};
$('stop').onclick = () => socket.emit('stop', {});
$('train').onclick = () => socket.emit('train', {});

function renderRecs() {
  const recs = status.recordings;
  $('recs').innerHTML = recs.length ? '<tr><th>activity</th><th>who</th><th>seconds</th><th></th></tr>' + recs.map((r) =>
    `<tr><td>${EMOJI[r.activity] || ''} ${r.activity}${r.dead ? ' <b style="color:var(--no)">⚠ sensor sent zeros, delete me</b>' : ''}</td><td>${r.person}</td><td>${r.seconds}</td>
     <td><button data-f="${r.file}">delete</button></td></tr>`).join('') : '';
  for (const b of $('recs').querySelectorAll('button'))
    b.onclick = () => { if (confirm(`Delete ${b.dataset.f}?`)) socket.emit('delete', { file: b.dataset.f }); };
}

window.addEventListener('resize', () => {
  for (const c of document.querySelectorAll('canvas')) c.width = 0;   // force re-size
  drawSensor(); if (live.peak !== undefined) drawLine(live.peak);
  drawTimeline('peak'); drawTimeline('tree');
  if (dataset) { renderExamples(); drawD1(); drawD2(); }
});
setInterval(() => { if (!$('tab-data').hidden) drawD2(); }, 80);   // keeps "you" pulsing
