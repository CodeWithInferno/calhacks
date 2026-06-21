# Robot Rollout Viewer

Browser viewer for the **Unitree G1** (23-dof humanoid, from
[unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)) trained on
**Nebius** (Isaac Lab). Five training factors are exposed as UI sliders; changing
any of them POSTs to a backend inference service that runs your **trained policy
in Isaac Sim** and returns a fresh rollout, which the viewer plays back on terrain
matching those factors.

### The five factors

| Slider | Param | Effect on the rollout |
|---|---|---|
| Incline | `incline_deg` | tilts the ground the robot climbs |
| Payload | `payload_kg` | mass carried — shown as a brown ~1 ft carton cradled in the arms (grows slightly when heavier) and switches the arms to a carry pose |
| Friction | `friction` | ground friction — also tints the terrain and makes loose crates slide/stick |
| Slopes | `num_slopes` | number of slope/bump features along the path |
| Speed | `speed_mps` | commanded locomotion speed (baked into the rollout) |

A separate **playback** control (0.25×–4×) fast-forwards/slows the replay without
changing the physics.

## Architecture — backend on-demand

```
 Browser sliders ──POST /api/rollout {5 factors}──►  FastAPI (backend/)
        ▲                                                  │
        │  {params, terrain, frames}                       │  run_rollout()  ◄── YOUR Isaac Lab policy
        └──────────────────────────────────────────────────┘     (integration seam)

 Robot URDF + meshes: fetched once from Nebius (scripts/fetch-run.mjs) → public/robot/
```

- **Credentials never reach the browser.** The browser only ever receives rollout
  JSON. Nebius creds are read at runtime by the Node fetch script (to pull the
  robot once) and, if your policy needs them, by the Python backend
  (`backend/nebius.py`) — from env / AWS profile only.
- The robot is **pure replay**; cannon-es steps only the loose crates (so incline
  + friction are visible on them too). It does not re-simulate the robot.

## Verified versions (June 2026)

three 0.184 · urdf-loader 0.12.7 · cannon-es 0.20.0 · FastAPI 0.128 · pydantic 2.x
· @aws-sdk/client-s3 3.x

## Quick start

Two processes: the FastAPI backend and the Vite frontend.

```bash
# 1. Backend (inference service)
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --port 8000 --reload

# 2. Frontend (in another terminal)
npm install
npm run dev            # http://localhost:5173  (Vite proxies /api -> :8000)
```

Move any slider → a new rollout is requested and played. The panel shows the
rollout **source**: `Isaac policy` (real), `backend stub` (placeholder), or
`offline fallback` (backend unreachable — an in-browser mirror so the UI still
works without Python).

## Wiring in your trained Isaac policy

Everything already speaks one rollout schema, so only **one function** changes:
`run_rollout(params)` in [`backend/inference.py`](backend/inference.py). Replace
the stub with real inference:

1. Build the Isaac Lab env with these domain params (terrain incline + slopes,
   added payload mass, contact friction, velocity command = `speed_mps`).
2. Load your policy checkpoint (from the Nebius bucket — `backend/nebius.py` has a
   credential-safe S3 client) and step the env.
3. Record per-frame joint positions + base pose and return the same dict:

```jsonc
{
  "params": { "incline_deg": 20, "payload_kg": 10, "friction": 0.2, "num_slopes": 3, "speed_mps": 1.5 },
  "terrain": { "incline_deg": 20, "friction": 0.2, "width": 1.5, "profile": [[x, z], ...] },
  "frames": [
    { "t": 0.0,
      "joints": { "joint1": 0.0, "joint2": 0.0 },
      "root":   { "pos": [x, y, z], "quat": [x, y, z, w] },   // Z-up
      "objects": [] }
  ],
  "source": "isaac"
}
```

`terrain.profile` is a dense height polyline along the travel axis so the viewer
draws ground that exactly matches the robot's recorded path.

> **Heavy / async inference:** Isaac Sim is slow to spin up. If a rollout takes
> many seconds, switch `/api/rollout` to a job pattern (`POST` returns a `job_id`;
> the UI polls `GET /api/rollout/:id`). The frontend's `requestRollout()` is the
> single place to adapt.

## Robot assets (Unitree G1)

The viewer loads [`public/robot/robot.urdf`](public/robot/robot.urdf) (Unitree G1
23-dof) + its STL meshes. The **URDF is committed; the 52 MB of meshes are
gitignored**. Repopulate them from the Unitree repo:

```bash
git clone --depth 1 https://github.com/unitreerobotics/unitree_rl_gym.git /tmp/unitree
cp -r /tmp/unitree/resources/robots/g1_description/meshes public/robot/meshes
# robot.urdf is g1_description/g1_23dof.urdf (already committed)
```

The gait in `inference.py` is keyed to G1's joint names. To use a **different
robot**, drop its URDF + meshes into `public/robot/` and update the joint names in
the gait (or, once trained, return real joint trajectories from Isaac — then the
joint set is whatever your policy outputs).

To pull a trained robot's artifacts from **Nebius** instead, set up `.env` (see
`.env.example` for creating the IAM access key) and:

```bash
npm run fetch -- <run_id>   # downloads <prefix>/<run_id>/* into public/robot/
```

> **Feet on the surface (all slopes/inclines).** Two parts: (1) the gait adds the
> local terrain slope to each ankle so the sole stays flat on the ground
> (`inference.py` `_leg`), and (2) the viewer **grounds the robot per frame** —
> it shifts the body vertically so the lowest planted foot rests on the terrain
> height under it (`player.js` `_ground()`). Together the legs plant like a human
> walk across ramps and bumps. This is self-correcting for real Isaac Lab rollouts
> too — if the feet are already at terrain height, the shift is ~0. Stance crouch
> is tunable via `STANDING_HEIGHT` / `k_base` in `backend/inference.py`.

## Deploy

```bash
npm run build             # static frontend in dist/
```

Host `dist/` on any static host and run the FastAPI backend wherever your Isaac
inference lives (it holds the Nebius creds). Point the frontend at it by proxying
`/api` or setting the fetch base URL. **No credentials ship to the browser.**

## Layout

```
backend/app.py          FastAPI: POST /api/rollout, GET /api/health
backend/inference.py    run_rollout() — INTEGRATION SEAM + parametric stub
backend/nebius.py       credential-safe boto3 client (checkpoints / caching)
scripts/fetch-run.mjs   one-time pull of the robot URDF/meshes from Nebius
src/viewer.js           scene, lights, OrbitControls, Z-up content frame
src/player.js           URDF load (once) + trajectory playback (setFrames per rollout)
src/terrain.js          ground ribbon from terrain.profile, tinted by friction
src/carton.js           brown carton carried on the chest, sized by payload weight
src/fallback.js         in-browser mirror of the gait (used if backend is down)
src/main.js             5 sliders -> backend -> viewer/terrain + timeline + camera follow
public/robot/           Unitree G1 URDF (committed) + meshes (gitignored, fetch separately)
```
