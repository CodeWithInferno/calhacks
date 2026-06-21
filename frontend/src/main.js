import { Vector3 } from 'three';
import { createViewer } from './viewer.js';
import { loadRobot, TrajectoryPlayer } from './player.js';
import { Terrain } from './terrain.js';
import { Carton } from './carton.js';
import { generateFallbackRollout } from './fallback.js';

const canvas = document.getElementById('scene');
const el = {
  status: document.getElementById('status'),
  play: document.getElementById('play'),
  scrub: document.getElementById('scrub'),
  time: document.getElementById('time'),
  speed: document.getElementById('speed'),
  regen: document.getElementById('regen'),
  src: document.getElementById('src-note'),
  incline: document.getElementById('f-incline'),
  payload: document.getElementById('f-payload'),
  friction: document.getElementById('f-friction'),
  slopes: document.getElementById('f-slopes'),
  speedF: document.getElementById('f-speed'),
  vIncline: document.getElementById('v-incline'),
  vPayload: document.getElementById('v-payload'),
  vFriction: document.getElementById('v-friction'),
  vSlopes: document.getElementById('v-slopes'),
  vSpeed: document.getElementById('v-speed'),
};

function setStatus(msg, kind = '') {
  el.status.textContent = msg;
  el.status.className = kind;
}

// ---- scene + subsystems ----
const viewer = createViewer(canvas);
viewer.ground.visible = false; // terrain ribbon replaces the flat ground
viewer.grid.visible = false;
const terrain = new Terrain(viewer.frame);

const SCRUB_MAX = Number(el.scrub.max);
const state = { playing: false, playback: 1, time: 0, duration: 0, player: null, robot: null, carton: null };

// ---- camera follow ----
const camOffset = new Vector3();      // camera position relative to its target
const robotWorld = new Vector3();
const followTarget = new Vector3();
let followInit = false;

function followRobot(dt) {
  if (!state.robot) return;
  state.robot.getWorldPosition(robotWorld);
  robotWorld.y += 0.4; // aim a bit above the pelvis, toward the torso
  if (!followInit) {
    followTarget.copy(robotWorld);
    camOffset.copy(viewer.camera.position).sub(viewer.controls.target);
    followInit = true;
  }
  // Preserve the user's current orbit offset, just track the moving robot.
  camOffset.copy(viewer.camera.position).sub(viewer.controls.target);
  followTarget.lerp(robotWorld, Math.min(1, dt * 4));
  viewer.controls.target.copy(followTarget);
  viewer.camera.position.copy(followTarget).add(camOffset);
}

// ---- factor reads ----
function factors() {
  return {
    incline_deg: Number(el.incline.value),
    payload_kg: Number(el.payload.value),
    friction: Number(el.friction.value),
    num_slopes: Number(el.slopes.value),
    speed_mps: Number(el.speedF.value),
    seconds: 8,
  };
}

function refreshFactorLabels() {
  el.vIncline.textContent = `${el.incline.value}°`;
  el.vPayload.textContent = `${el.payload.value} kg`;
  el.vFriction.textContent = Number(el.friction.value).toFixed(2);
  el.vSlopes.textContent = el.slopes.value;
  el.vSpeed.textContent = `${Number(el.speedF.value).toFixed(1)} m/s`;
}

// ---- timeline ----
const fmt = (t) => t.toFixed(2);
function syncTimeUI() {
  el.time.textContent = `${fmt(state.time)} / ${fmt(state.duration)}s`;
  el.scrub.value = String(Math.round((state.duration ? state.time / state.duration : 0) * SCRUB_MAX));
}
function setPlaying(p) {
  state.playing = p;
  el.play.textContent = p ? '❚❚' : '▶';
}

// The deployed site is standalone: rollouts are generated in-browser. In local
// dev (`npm run dev`) we additionally try the FastAPI service, so you can test
// the real Isaac integration. Flip with VITE_USE_BACKEND=1 in a prod build.
const USE_BACKEND = import.meta.env.DEV || import.meta.env.VITE_USE_BACKEND === '1';

// ---- request a rollout (backend if available, else in-browser) ----
async function requestRollout() {
  const params = factors();
  el.regen.disabled = true;
  setStatus('running rollout…');
  let data = null;
  if (USE_BACKEND) {
    try {
      const res = await fetch('/api/rollout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      if (res.ok) data = await res.json();
    } catch (err) {
      console.warn('rollout backend unavailable, generating in-browser:', err.message);
    }
  }
  if (!data) data = generateFallbackRollout(params);
  applyRollout(data);
  el.regen.disabled = false;
}

function applyRollout(data) {
  terrain.update(data.terrain);
  state.player.setTerrain(data.terrain); // so feet plant on the surface
  state.carton.update(data.params.payload_kg);
  state.player.setFrames(data.frames);
  state.duration = state.player.duration;
  state.time = 0;
  state.player.seek(0);
  followInit = false; // re-anchor the camera on the new path
  syncTimeUI();
  setPlaying(true);

  const src = data.source || 'isaac';
  const labels = {
    isaac: 'source: Isaac policy',
    stub: 'source: backend stub',
    fallback: 'source: in-browser simulation',
  };
  el.src.textContent = labels[src] || `source: ${src}`;
  el.src.className = src === 'isaac' ? 'src' : 'src stub';
  setStatus('ready');
}

// ---- input wiring ----
let debounce;
function onFactorInput() {
  refreshFactorLabels();
  clearTimeout(debounce);
  debounce = setTimeout(requestRollout, 250);
}
for (const s of [el.incline, el.payload, el.friction, el.slopes, el.speedF]) {
  s.addEventListener('input', onFactorInput);
}
el.regen.addEventListener('click', requestRollout);
el.play.addEventListener('click', () => setPlaying(!state.playing));
el.scrub.addEventListener('input', () => {
  setPlaying(false);
  state.time = (Number(el.scrub.value) / SCRUB_MAX) * state.duration;
  state.player?.seek(state.time);
  syncTimeUI();
});
el.speed.addEventListener('change', () => { state.playback = Number(el.speed.value); });
window.addEventListener('keydown', (e) => {
  if (e.code === 'Space') { e.preventDefault(); setPlaying(!state.playing); }
  if (e.code === 'Escape') closeOverlay();
});

// ---- intro / about overlay ----
const overlay = document.getElementById('overlay');
const SEEN_KEY = 'g1viewer.introSeen';
function openOverlay() { overlay.classList.remove('hidden'); }
function closeOverlay() {
  overlay.classList.add('hidden');
  try { localStorage.setItem(SEEN_KEY, '1'); } catch {}
}
document.getElementById('info-btn').addEventListener('click', openOverlay);
document.getElementById('overlay-close').addEventListener('click', closeOverlay);
overlay.addEventListener('click', (e) => { if (e.target === overlay) closeOverlay(); });
let seen = false;
try { seen = localStorage.getItem(SEEN_KEY) === '1'; } catch {}
if (!seen) openOverlay();

// ---- animation loop ----
let prev = performance.now();
function tick(now) {
  const dt = Math.min((now - prev) / 1000, 0.05);
  prev = now;
  if (state.player && state.playing) {
    state.time += dt * state.playback;
    if (state.time >= state.duration) { state.time = state.duration; setPlaying(false); }
    state.player.seek(state.time);
    syncTimeUI();
  }
  followRobot(dt);
  viewer.render();
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

// ---- boot ----
refreshFactorLabels();
setStatus('loading robot…');
loadRobot('/robot/robot.urdf', '/robot', viewer.frame)
  .then((robot) => {
    state.robot = robot;
    state.player = new TrajectoryPlayer(robot);
    state.carton = new Carton(robot);
    return requestRollout();
  })
  .catch((err) => {
    console.error(err);
    setStatus(err.message || String(err), 'error');
  });
