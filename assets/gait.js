// Gait trainer web page. Talks to python/main.py over socket.io:
//   board -> page: "live" (twice a second), "status", "record", "train"
//   page -> board: "hello", "record", "stop", "train", "delete", "matrix_model"
const socket = io(window.location.origin);

const EMOJI = { standing: '🧍', walking: '🚶', running: '🏃', biking: '🚴' };
const css = getComputedStyle(document.documentElement);
const color = (a) => css.getPropertyValue(`--${a}`).trim() || '#888';

// Plain-English names for the tree's questions
const NICE = {
  acc_jerk: 'impact sharpness (g/s)', cadence: 'rhythm (per min)',
  acc_mag_mean: 'avg total accel (g)', acc_mag_std: 'total accel wobble (g)',
  acc_mag_max: 'max total accel (g)', acc_mag_p90: 'big total accel (g)',
  gyro_mag_mean: 'avg spin (°/s)', gyro_mag_std: 'spin wobble (°/s)',
  gyro_mag_max: 'max spin (°/s)', gyro_mag_p90: 'big spin (°/s)',
};
function nice(f) {
  if (NICE[f]) return NICE[f];
  const [ch, stat] = f.split('_');
  const what = ch[0] === 'a' ? `accel ${ch[1]}` : `spin ${ch[1]}`;
  const unit = ch[0] === 'a' ? 'g' : '°/s';
  const s = { mean: 'avg', std: 'wobble', max: 'max', min: 'min', range: 'range' }[stat] || stat;
  return `${s} ${what} (${unit})${ch[0] === 'a' && stat === 'mean' ? ' = tilt' : ''}`;
}

let status = null;      // last "status" message
let live = {};          // last "live" message
let activity = 'standing';

// ---------------------------------------------------------------- sound (phone is in your pocket)
let audio = null;
function beep(freq = 880, ms = 120) {
  try {
    audio = audio || new (window.AudioContext || window.webkitAudioContext)();
    const o = audio.createOscillator(), g = audio.createGain();
    o.frequency.value = freq;
    g.gain.value = 0.2;
    o.connect(g).connect(audio.destination);
    o.start();
    o.stop(audio.currentTime + ms / 1000);
  } catch (e) { /* no audio, fine */ }
}

// ---------------------------------------------------------------- socket
socket.on('connect', () => socket.emit('hello', {}));
socket.on('disconnect', () => setImu(false, 'disconnected'));

socket.on('status', (s) => {
  status = s;
  renderVerdictFrames();
  renderActs();
  renderTree();
  renderResults();
  renderRecs();
  document.getElementById('dir').textContent = s.models ? s.models.peak.direction : '?';
});

socket.on('live', (m) => {
  live = m;
  setImu(m.imu, m.imu ? `IMU ${m.rate} Hz` : m.dead ? 'sensor sending zeros' : 'no IMU');
  if (m.cadence !== undefined) document.getElementById('cadence').textContent = `${m.cadence} strides/min`;
  setVerdict('peak', m.peak_label);
  setVerdict('tree', m.tree_label);
  drawLine();
  drawTrace();
  highlightTree();
});

let lastLeft = null;
socket.on('record', (r) => {
  const el = document.getElementById('rec-status');
  if (r.phase === 'countdown') {
    el.textContent = `Starting in ${r.left}…`;
    beep(660, 100);
  } else if (r.phase === 'recording') {
    if (lastLeft === null) beep(1320, 300);
    lastLeft = r.left;
    el.textContent = `● Recording ${activity}: ${r.left} s left`;
  } else {
    lastLeft = null;
    el.textContent = r.file ? `Saved ✓ ${r.file}` : r.error || 'Cancelled';
    beep(r.file ? 990 : 330, 400);
  }
});

socket.on('train', (t) => {
  const el = document.getElementById('train-status');
  el.textContent = { training: 'Training… (a few seconds)', done: 'Done ✓', error: `Error: ${t.error}` }[t.phase];
});

// ---------------------------------------------------------------- top verdicts
function setImu(ok, text) {
  const el = document.getElementById('imu');
  el.textContent = text;
  el.className = `pill ${ok ? 'good' : 'bad'}`;
}

function setVerdict(model, label) {
  document.getElementById(`e-${model}`).textContent = EMOJI[label] || '❔';
  const l = document.getElementById(`l-${model}`);
  l.textContent = label || (status && !status.models ? 'not trained' : '–');
  l.style.color = label ? color(label) : '';
}

function renderVerdictFrames() {
  for (const m of ['peak', 'tree'])
    document.getElementById(`v-${m}`).classList.toggle('on-matrix', status.matrix_model === m);
}
for (const b of document.querySelectorAll('.verdict'))
  b.addEventListener('click', () => socket.emit('matrix_model', { model: b.dataset.model }));

// ---------------------------------------------------------------- model 1 pictures
// Sharp canvas on phones: draw in CSS pixels, back it with devicePixelRatio pixels
function fitCanvas(c) {
  const dpr = window.devicePixelRatio || 1;
  c._h = c._h || c.height;             // height from the HTML, in CSS pixels
  const w = c.clientWidth;
  if (c.width !== Math.round(w * dpr)) {
    c.width = Math.round(w * dpr);
    c.height = Math.round(c._h * dpr);
    c.style.height = `${c._h}px`;
  }
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [ctx, w, c._h];
}

function lineRange(p) {
  const t = p.thresholds;
  const span = Math.max(t[t.length - 1] - t[0], 0.5);
  return [Math.min(0, t[0] - span * 0.5), t[t.length - 1] + span * 0.7];
}

// Number line: colored zones for each activity, the 3 thresholds, and a dot for right now
function drawLine() {
  const c = document.getElementById('line');
  const [ctx, w, h] = fitCanvas(c);
  ctx.clearRect(0, 0, w, h);
  if (!status || !status.models) return;
  const p = status.models.peak;
  const [lo, hi] = lineRange(p);
  const X = (v) => 10 + (w - 20) * (v - lo) / (hi - lo);
  const edges = [lo, ...p.thresholds, hi];
  ctx.font = '12px -apple-system, sans-serif';
  ctx.textAlign = 'center';
  p.order.forEach((a, i) => {
    const x0 = X(edges[i]), x1 = X(edges[i + 1]);
    ctx.fillStyle = color(a);
    ctx.globalAlpha = live.peak_label === a ? 0.9 : 0.35;
    ctx.fillRect(x0, 30, x1 - x0, 22);
    ctx.globalAlpha = 1;
    ctx.fillStyle = css.getPropertyValue('--ink');
    ctx.fillText(`${EMOJI[a]} ${a}`, (x0 + x1) / 2, 22);
  });
  ctx.fillStyle = css.getPropertyValue('--muted');
  p.thresholds.forEach((t) => {
    ctx.fillRect(X(t) - 1, 28, 2, 26);
    ctx.fillText(`${t.toFixed(2)} g`, X(t), 70);
  });
  if (live.peak !== undefined) {
    const x = X(Math.max(lo, Math.min(hi, live.peak)));
    ctx.fillStyle = css.getPropertyValue('--ink');
    ctx.beginPath();
    ctx.moveTo(x, 56); ctx.lineTo(x - 7, 66); ctx.lineTo(x + 7, 66); ctx.fill();
    ctx.fillText(`now ${live.peak.toFixed(2)} g`, Math.min(Math.max(x, 40), w - 40), 86);
  }
}

// Live signal along model 1's direction, with the thresholds as horizontal lines
function drawTrace() {
  const c = document.getElementById('trace');
  const [ctx, w, h] = fitCanvas(c);
  ctx.clearRect(0, 0, w, h);
  const tr = live.trace || live.trace_mag;
  if (!tr) return;
  const p = status && status.models ? status.models.peak : null;
  let [lo, hi] = p ? lineRange(p) : [0, 3];
  lo = Math.min(lo, ...tr); hi = Math.max(hi, ...tr);
  const Y = (v) => h - 14 - (h - 24) * (v - lo) / (hi - lo);
  const X = (i) => (w - 4) * i / (tr.length - 1) + 2;
  if (p) p.thresholds.forEach((t, i) => {
    ctx.strokeStyle = color(p.order[i + 1]);
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(0, Y(t)); ctx.lineTo(w, Y(t)); ctx.stroke();
  });
  ctx.setLineDash([]);
  ctx.strokeStyle = css.getPropertyValue('--ink');
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  tr.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v))));
  ctx.stroke();
  const imax = tr.indexOf(Math.max(...tr));
  ctx.fillStyle = color(live.peak_label || 'running');
  ctx.beginPath(); ctx.arc(X(imax), Y(tr[imax]), 4, 0, 7); ctx.fill();
  ctx.fillStyle = css.getPropertyValue('--muted');
  ctx.font = '11px -apple-system, sans-serif';
  ctx.fillText(p ? `accel along ${p.direction} (g), last 2.5 s · dot = peak` : 'total accel (g), last 2.5 s', 4, h - 2);
}
window.addEventListener('resize', () => { drawLine(); drawTrace(); });

// ---------------------------------------------------------------- model 2 picture
function renderTree() {
  const el = document.getElementById('tree');
  if (!status.models) { el.innerHTML = '<p class="sub">No model yet. Record each activity, then Train.</p>'; return; }
  el.innerHTML = '';
  el.appendChild(treeNode(status.models.tree.root, ''));
  highlightTree();
}

function treeNode(n, path) {
  const d = document.createElement('div');
  d.className = 'node';
  d.dataset.path = path;
  if (n.leaf) {
    const l = document.createElement('div');
    l.className = 'leaf';
    l.style.background = color(n.leaf);
    l.textContent = `${EMOJI[n.leaf] || ''} ${n.leaf}`;
    d.appendChild(l);
    return d;
  }
  const q = document.createElement('div');
  q.className = 'q';
  q.innerHTML = `${nice(n.feature)} ≤ ${n.threshold.toFixed(2)}?<span class="v" data-f="${n.feature}"></span>`;
  d.appendChild(q);
  const kids = document.createElement('div');
  kids.className = 'kids';
  for (const ans of ['yes', 'no']) {
    const b = document.createElement('div');
    b.className = 'branch';
    b.innerHTML = `<span class="yn">${ans}</span>`;
    b.appendChild(treeNode(n[ans], path + (ans === 'yes' ? 'Y' : 'N')));
    kids.appendChild(b);
  }
  d.appendChild(kids);
  return d;
}

function highlightTree() {
  if (!live.tree_path) return;
  const p = live.tree_path.map((a) => (a === 'yes' ? 'Y' : 'N')).join('');
  for (const n of document.querySelectorAll('#tree .node')) {
    const on = p.startsWith(n.dataset.path);
    n.classList.toggle('active', on);
    const box = n.firstElementChild;
    box.classList.toggle('dim', !on);
  }
  for (const v of document.querySelectorAll('#tree .v')) {
    const val = live.tree_values ? live.tree_values[v.dataset.f] : undefined;
    v.textContent = val === undefined ? '' : `now ${val}`;
  }
}

// ---------------------------------------------------------------- record / train / list
function renderActs() {
  const el = document.getElementById('acts');
  el.innerHTML = '';
  for (const a of status.activities) {
    const b = document.createElement('button');
    b.textContent = `${EMOJI[a]} ${a}`;
    if (a === activity) { b.classList.add('sel'); b.style.background = color(a); }
    b.onclick = () => { activity = a; renderActs(); };
    el.appendChild(b);
  }
}

document.getElementById('rec').onclick = () => {
  beep(1, 1);     // unlock audio on iPhone (needs a tap)
  socket.emit('record', {
    activity,
    person: document.getElementById('person').value,
    seconds: Number(document.getElementById('seconds').value),
  });
};
document.getElementById('stop').onclick = () => socket.emit('stop', {});
document.getElementById('train').onclick = () =>
  socket.emit('train', { direction: document.getElementById('direction').value });

function confusion(labels, cm) {
  let h = '<table class="cm"><tr><th>true ↓ / guess →</th>';
  h += labels.map((l) => `<th>${EMOJI[l]}</th>`).join('') + '</tr>';
  cm.forEach((row, i) => {
    h += `<tr><th>${EMOJI[labels[i]]}</th>` +
      row.map((v, j) => `<td class="${i === j ? 'hit' : ''}">${v}</td>`).join('') + '</tr>';
  });
  return h + '</table>';
}

function renderResults() {
  const el = document.getElementById('results');
  const cv = status.models && status.models.cv;
  if (!cv) { el.innerHTML = ''; return; }
  el.innerHTML = `<p class="sub">Tested on data each model did NOT train on (${cv.scheme}, ${cv.windows} windows${
    cv.people.length > 1 ? `, people: ${cv.people.join(', ')}` : ''}).</p>
    <div class="results">
      <div><b>Model 1 · peak ${status.models.peak.direction}</b><div class="acc">${Math.round(cv.peak.accuracy * 100)}%</div>
        ${confusion(cv.labels, cv.peak.confusion)}</div>
      <div><b>Model 2 · tree</b><div class="acc">${Math.round(cv.tree.accuracy * 100)}%</div>
        ${confusion(cv.labels, cv.tree.confusion)}</div>
    </div>`;
}

function renderRecs() {
  const t = document.getElementById('recs');
  const recs = status.recordings;
  if (!recs.length) { t.innerHTML = '<tr><td class="sub">Nothing recorded yet.</td></tr>'; return; }
  t.innerHTML = '<tr><th>activity</th><th>who</th><th>s</th><th></th></tr>' + recs.map((r) =>
    `<tr><td>${EMOJI[r.activity] || ''} ${r.activity}${r.dead ? ' ⚠ zeros' : ''}</td><td>${r.person}</td><td>${r.seconds}</td>
     <td><button data-f="${r.file}">delete</button></td></tr>`).join('');
  for (const b of t.querySelectorAll('button'))
    b.onclick = () => { if (confirm(`Delete ${b.dataset.f}?`)) socket.emit('delete', { file: b.dataset.f }); };
}
