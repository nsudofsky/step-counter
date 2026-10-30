// Step counter page. Talks to python/main.py over socket.io:
//   board -> page: "live" (twice a second, with .steps), "status", "record"
//   page -> board: "hello", "reset", "record", "stop", "label", "delete", "matrix_model"
const socket = io(window.location.origin);
const $ = (id) => document.getElementById(id);
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(`--${n}`).trim();
const EMOJI = { standing: '🧍', walking: '🚶', running: '🏃', biking: '🚴' };
const ACTS = ['walking', 'running', 'standing', 'biking'];

let status = null;
let activity = 'walking';
const history = [];                 // last 60 s of {mean, lo, hi}
let lastFile = null;

// ---------------------------------------------------------------- socket
socket.on('connect', () => socket.emit('hello', {}));
socket.on('disconnect', () => setPill(false, 'disconnected'));

socket.on('status', (s) => {
  status = s;
  for (const b of document.querySelectorAll('.matrix-pick button'))
    b.classList.toggle('on', b.dataset.model === s.matrix_model);
  renderQuality();
  renderActs();
  renderRecs();
});

socket.on('live', (m) => {
  setPill(m.imu, m.imu ? `IMU ${m.rate} Hz` : m.dead ? 'sensor sending zeros' : 'no IMU');
  $('activity').textContent = m.tree_label ? `${EMOJI[m.tree_label]} ${m.tree_label}` : '–';
  const st = m.steps;
  if (!st) return;
  const n = Math.round(st.count);
  if (shownCount !== null && n > shownCount) celebrate(Math.min(n - shownCount, 3));
  shownCount = n;
  $('count').textContent = n;
  $('count-unc').textContent = `± ${Math.max(1, Math.round(st.unc))}`;
  $('meters').textContent = st.meters.toFixed(1);
  $('meters-unc').textContent = `± ${st.meters_unc.toFixed(1)}${st.calibrated ? '' : ' ?'}`;
  $('cadence').textContent = st.cadence;
  const mins = Math.floor((Date.now() / 1000 - st.since) / 60);
  $('since').textContent = `counting for ${mins} min`;
  history.push({ mean: st.rate, lo: Math.min(...st.per_net), hi: Math.max(...st.per_net) });
  if (history.length > 120) history.shift();
  renderNets(st.per_net);
  drawRate();
});

socket.on('record', (r) => {
  const el = $('rec-status');
  if (r.phase === 'countdown') { el.textContent = `Starting in ${r.left}… (start counting at the beep)`; beep(660, 100); }
  else if (r.phase === 'recording') {
    if (r.elapsed === 0) beep(1320, 300);
    el.textContent = `● Recording ${activity}: ${r.elapsed} s, count your steps!`;
  } else {
    beep(r.file ? 990 : 330, 400);
    el.textContent = r.file ? `Saved ✓ ${r.file}` : r.error || 'Cancelled';
    if (r.file) showLabelForm(r.file);
  }
});

function setPill(ok, text) { $('imu').textContent = text; $('imu').className = `pill ${ok ? 'good' : 'bad'}`; }

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

// ---------------------------------------------------------------- every step: pop, footprint, click
let shownCount = null, leftFoot = true, clicks = false;
function celebrate(k) {
  for (let i = 0; i < k; i++) {
    setTimeout(() => {
      const c = $('count');
      c.classList.add('pop');
      setTimeout(() => c.classList.remove('pop'), 120);
      const f = document.createElement('span');
      f.className = 'foot';
      f.textContent = '👣';
      f.style.left = leftFoot ? '58%' : '74%';
      f.style.top = leftFoot ? '10%' : '30%';
      f.style.transform = `scaleX(${leftFoot ? -1 : 1})`;
      leftFoot = !leftFoot;
      $('feet').appendChild(f);
      setTimeout(() => f.remove(), 900);
      if (clicks) beep(leftFoot ? 1500 : 1200, 25);
    }, i * 160);          // several steps in one update -> spread them out a little
  }
}
$('clicks').onclick = () => {
  clicks = !clicks;
  beep(1, 1);             // iPhone only plays sound after a tap
  $('clicks').textContent = clicks ? '🔊 Click per step' : '🔇 Click per step';
};
const toggleBig = () => document.body.classList.toggle('bigmode');
$('bigview').onclick = toggleBig;
$('big').onclick = toggleBig;

// ---------------------------------------------------------------- live pictures
function renderNets(per) {
  const el = $('per-net');
  const top = Math.max(4, ...per);
  el.innerHTML = per.map((r, i) => `<div class="netbar"><span>net ${i + 1}</span>
    <div class="track"><div class="fill" style="width:${100 * r / top}%"></div></div><b>${r.toFixed(2)}/s</b></div>`).join('');
}

function drawRate() {
  const c = $('rate');
  const dpr = window.devicePixelRatio || 1;
  c._h = c._h || c.height;
  const w = c.clientWidth, h = c._h;
  if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); c.style.height = `${h}px`; }
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const top = Math.max(3.5, ...history.map((p) => p.hi));
  const X = (i) => w - (history.length - 1 - i) * (w / 119);
  const Y = (v) => h - 16 - (h - 24) * v / top;
  ctx.fillStyle = cssVar('muted'); ctx.font = '11px -apple-system, sans-serif';
  for (const v of [0, 1, 2, 3]) {
    if (v > top) continue;
    ctx.fillText(`${v}/s`, 2, Y(v) - 2);
    ctx.strokeStyle = cssVar('line'); ctx.beginPath(); ctx.moveTo(30, Y(v)); ctx.lineTo(w, Y(v)); ctx.stroke();
  }
  if (history.length < 2) return;
  ctx.fillStyle = cssVar('walking'); ctx.globalAlpha = 0.25;
  ctx.beginPath();
  history.forEach((p, i) => (i ? ctx.lineTo(X(i), Y(p.hi)) : ctx.moveTo(X(i), Y(p.hi))));
  for (let i = history.length - 1; i >= 0; i--) ctx.lineTo(X(i), Y(history[i].lo));
  ctx.fill(); ctx.globalAlpha = 1;
  ctx.strokeStyle = cssVar('ink'); ctx.lineWidth = 2; ctx.beginPath();
  history.forEach((p, i) => (i ? ctx.lineTo(X(i), Y(p.mean)) : ctx.moveTo(X(i), Y(p.mean))));
  ctx.stroke(); ctx.lineWidth = 1;
}

// ---------------------------------------------------------------- how good is it
function renderQuality() {
  const n = status.step_net;
  if (!n || !n.test) { $('quality').innerHTML = '<p class="sub">No step net yet. Record, then train on the laptop.</p>'; return; }
  const t = n.test, sl = n.stride;
  const distTxt = sl.n ? `${sl.err ? `±${sl.err.mean_abs_pct.toFixed(0)}%` : 'calibrated'}` : 'not calibrated';
  $('quality').innerHTML = `
    <div class="q2"><div class="v">±${t.mean_abs_pct_error.toFixed(1)}%</div>
      <div class="d">typical step-count error walking/running, on ${t.recordings} recordings the net never saw (${t.hand_counted} hand-counted)</div></div>
    <div class="q2"><div class="v">${t.mean_pct_error >= 0 ? '+' : ''}${t.mean_pct_error.toFixed(1)}%</div>
      <div class="d">bias: does it count too many (+) or too few (−) on average</div></div>
    <div class="q2"><div class="v">${t.false_steps_per_min_still.toFixed(1)}</div>
      <div class="d">phantom steps per minute while standing or biking</div></div>
    <div class="q2"><div class="v">${distTxt}</div>
      <div class="d">distance: ${sl.n ? `${sl.n} measured walks, meters per step = ${sl.a.toFixed(2)}${sl.b ? ` + ${sl.b.toFixed(4)} × steps/min` : ''}` : 'walk a measured path and type the meters to calibrate (0.70 m per step until then)'}</div></div>`;
}

// ---------------------------------------------------------------- record + label
function renderActs() {
  $('acts').innerHTML = '';
  for (const a of ACTS) {
    const b = document.createElement('button');
    b.textContent = `${EMOJI[a]} ${a}`;
    if (a === activity) { b.classList.add('sel'); b.style.background = cssVar(a); }
    b.onclick = () => { activity = a; renderActs(); };
    $('acts').appendChild(b);
  }
}

function showLabelForm(file) {
  lastFile = file;
  $('label-form').hidden = false;
  $('lab-file').textContent = file;
  $('lab-steps').value = ''; $('lab-meters').value = '';
  $('lab-steps').focus();
}
$('label-form').onsubmit = (e) => {
  e.preventDefault();
  socket.emit('label', { file: lastFile, steps: $('lab-steps').value, meters: $('lab-meters').value });
  $('label-form').hidden = true;
  $('rec-status').textContent = `Labeled ${lastFile} ✓`;
};

$('rec').onclick = () => {
  beep(1, 1);
  socket.emit('record', { activity, person: $('person').value, seconds: Number($('seconds').value) });
};
$('stop').onclick = () => socket.emit('stop', {});
$('reset').onclick = () => { history.length = 0; shownCount = null; socket.emit('reset', {}); };
for (const b of document.querySelectorAll('.matrix-pick button'))
  b.onclick = () => socket.emit('matrix_model', { model: b.dataset.model });

function renderRecs() {
  const recs = status.recordings;
  $('recs').innerHTML = recs.length ? '<tr><th>activity</th><th>who</th><th>s</th><th>steps counted</th><th>meters</th><th></th></tr>' +
    recs.map((r) => `<tr><td>${EMOJI[r.activity] || ''} ${r.activity}${r.dead ? ' <b style="color:var(--no)">⚠ zeros</b>' : ''}</td>
      <td>${r.person}</td><td>${r.seconds}</td><td>${r.steps ?? '–'}</td><td>${r.meters ?? '–'}</td>
      <td><button data-label="${r.file}">label</button> <button data-del="${r.file}">delete</button></td></tr>`).join('') : '';
  for (const b of $('recs').querySelectorAll('[data-label]')) b.onclick = () => showLabelForm(b.dataset.label);
  for (const b of $('recs').querySelectorAll('[data-del]'))
    b.onclick = () => { if (confirm(`Delete ${b.dataset.del}?`)) socket.emit('delete', { file: b.dataset.del }); };
}
window.addEventListener('resize', () => { $('rate').width = 0; drawRate(); });
