import { LoadingManager, Quaternion, Vector3, Box3, Mesh, MeshStandardMaterial } from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import URDFLoader from 'urdf-loader';

/**
 * Load a URDF robot ONCE (resolving glTF / STL meshes relative to baseUrl) and
 * add it to `frame`. Returns the THREE object for the robot.
 */
export async function loadRobot(urdfUrl, baseUrl, frame) {
  const robot = await new Promise((resolve, reject) => {
    const manager = new LoadingManager();
    const loader = new URDFLoader(manager);
    // urdf-loader concatenates workingPath + the URDF's relative mesh path with
    // no separator, so the base MUST end in a slash (else /robot + meshes/x.STL
    // -> /robotmeshes/x.STL, a 404 that dev servers answer with HTML).
    const base = baseUrl.endsWith('/') ? baseUrl : baseUrl + '/';
    loader.packages = base;
    loader.workingPath = base;

    loader.loadMeshCb = (path, mgr, onComplete) => {
      const ext = path.split(/[?#]/)[0].split('.').pop().toLowerCase();
      try {
        if (ext === 'glb' || ext === 'gltf') {
          new GLTFLoader(mgr).load(
            path, (g) => onComplete(g.scene), undefined, (e) => onComplete(null, e),
          );
        } else if (ext === 'stl') {
          new STLLoader(mgr).load(
            path,
            (geom) => onComplete(new Mesh(geom, new MeshStandardMaterial({ color: 0xbfc4cc }))),
            undefined, (e) => onComplete(null, e),
          );
        } else {
          onComplete(null, new Error(`Unsupported mesh type: ${path}`));
        }
      } catch (e) {
        onComplete(null, e);
      }
    };

    loader.load(
      urdfUrl,
      (r) => resolve(r),
      undefined,
      (e) => reject(new Error(`Failed to load URDF (${urdfUrl}): ${e?.message || e}`)),
    );
  });

  robot.traverse((o) => {
    if (o.isMesh) {
      o.castShadow = true;
      o.receiveShadow = true;
    }
  });
  frame.add(robot);
  return robot;
}

/**
 * Drives a loaded robot's joints + root pose from a set of trajectory frames.
 * Frames can be swapped at any time via setFrames() (used when a new rollout
 * comes back from the backend).
 */
export class TrajectoryPlayer {
  constructor(robot) {
    this.robot = robot;
    this.frames = [];
    this.duration = 0;
    this.t0 = 0;
    this.hasObjects = false;
    this._pos = new Vector3();
    this._qa = new Quaternion();
    this._qb = new Quaternion();
    // Foot links used to plant the robot on the terrain (per-foot grounding).
    this.feet = ['left_ankle_roll_link', 'right_ankle_roll_link']
      .map((n) => robot.links?.[n])
      .filter(Boolean);
    this._footBox = new Box3();
    this._profile = null; // [[x, z], ...] sorted by x; set via setTerrain()
  }

  /** Supply the terrain so the feet can be grounded to its height. */
  setTerrain(terrain) {
    this._profile = terrain?.profile ?? null;
  }

  /** Terrain height (local Z / world Y, up) at travel coordinate x. */
  _groundAt(x) {
    const p = this._profile;
    if (!p || p.length === 0) return 0;
    if (x <= p[0][0]) return p[0][1];
    if (x >= p[p.length - 1][0]) return p[p.length - 1][1];
    for (let i = 0; i < p.length - 1; i++) {
      const [x0, z0] = p[i];
      const [x1, z1] = p[i + 1];
      if (x >= x0 && x <= x1) {
        const k = (x - x0) / (x1 - x0 || 1);
        return z0 + (z1 - z0) * k;
      }
    }
    return 0;
  }

  /**
   * Shift the robot vertically so its lowest planted foot rests on the terrain.
   * The content frame maps local Z (up) -> world Y, so foot world-min Y compares
   * directly to the terrain profile, and a robot.position.z delta moves world Y
   * 1:1. Works for any pose/crouch and follows slopes (each foot sees the
   * terrain height under its own x). No-op if feet or terrain are missing.
   */
  _ground() {
    if (this.feet.length === 0 || !this._profile) return;
    this.robot.updateMatrixWorld(true);
    let shift = Infinity;
    for (const foot of this.feet) {
      this._footBox.setFromObject(foot);
      const footX = (this._footBox.min.x + this._footBox.max.x) / 2;
      const clearance = this._footBox.min.y - this._groundAt(footX);
      if (clearance < shift) shift = clearance;
    }
    if (Number.isFinite(shift)) this.robot.position.z -= shift;
  }

  setFrames(frames) {
    if (!Array.isArray(frames) || frames.length === 0) {
      throw new Error('rollout must be a non-empty array of frames');
    }
    this.frames = frames.slice().sort((a, b) => a.t - b.t);
    this.t0 = this.frames[0].t;
    this.duration = this.frames[this.frames.length - 1].t - this.t0;
    this.hasObjects = this.frames.some((f) => (f.objects?.length ?? 0) > 0);
    this.seek(0);
    return this;
  }

  _at(local) {
    return this.t0 + Math.max(0, Math.min(this.duration, local));
  }

  /** Pose the robot at `local` seconds (0..duration), interpolating frames. */
  seek(local) {
    const { frames } = this;
    if (frames.length === 0) return null;
    const t = this._at(local);

    let a = frames[0];
    let b = frames[frames.length - 1];
    for (let i = 0; i < frames.length - 1; i++) {
      if (t >= frames[i].t && t <= frames[i + 1].t) {
        a = frames[i];
        b = frames[i + 1];
        break;
      }
    }
    const span = b.t - a.t || 1;
    const k = Math.max(0, Math.min(1, (t - a.t) / span));

    for (const name in a.joints) {
      const va = a.joints[name];
      const vb = b.joints[name] ?? va;
      this.robot.setJointValue(name, va + (vb - va) * k);
    }

    if (a.root) {
      const pa = a.root.pos;
      const pb = (b.root || a.root).pos;
      this._pos.set(
        pa[0] + (pb[0] - pa[0]) * k,
        pa[1] + (pb[1] - pa[1]) * k,
        pa[2] + (pb[2] - pa[2]) * k,
      );
      this.robot.position.copy(this._pos);

      const qa = a.root.quat;
      const qb = (b.root || a.root).quat;
      this._qa.set(qa[0], qa[1], qa[2], qa[3]);
      this._qb.set(qb[0], qb[1], qb[2], qb[3]);
      this._qa.slerp(this._qb, k);
      this.robot.quaternion.copy(this._qa);
    }

    this._ground();
    return { a, b, k, objects: a.objects ?? [] };
  }
}
