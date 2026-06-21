import {
  Mesh,
  BufferGeometry,
  Float32BufferAttribute,
  MeshStandardMaterial,
  Color,
  DoubleSide,
  Group,
  LineSegments,
  EdgesGeometry,
  LineBasicMaterial,
} from 'three';

/**
 * Renders the ground the backend says the robot walked on: a ribbon following
 * the 1D height profile (along +X), in the Z-up content frame so it lines up
 * exactly with the robot's recorded root path. Tinted by friction (low = icy
 * blue / slippery, high = warm grey / grippy).
 */
export class Terrain {
  constructor(frame) {
    this.group = new Group();
    frame.add(this.group);
    this.mesh = null;
    this.edges = null;
  }

  _frictionColor(friction) {
    // friction 0.05 -> icy blue, 0.6 -> neutral, 1.5 -> warm grey
    const t = Math.max(0, Math.min(1, (friction - 0.05) / (1.5 - 0.05)));
    const icy = new Color(0x3a6ea5);
    const grippy = new Color(0x6b6258);
    return icy.clone().lerp(grippy, t);
  }

  /** terrain = { profile: [[x,z],...], width, friction } */
  update(terrain) {
    this.dispose();

    const profile = terrain.profile;
    const halfW = (terrain.width ?? 1.5) / 2;

    const positions = [];
    const indices = [];
    // Two vertices (±Y) per profile sample; Z is height (up, after frame rotation).
    for (let i = 0; i < profile.length; i++) {
      const [x, z] = profile[i];
      positions.push(x, -halfW, z);
      positions.push(x, halfW, z);
    }
    for (let i = 0; i < profile.length - 1; i++) {
      const a = i * 2, b = i * 2 + 1, c = i * 2 + 2, d = i * 2 + 3;
      indices.push(a, b, c, c, b, d);
    }

    const geo = new BufferGeometry();
    geo.setAttribute('position', new Float32BufferAttribute(positions, 3));
    geo.setIndex(indices);
    geo.computeVertexNormals();

    const mat = new MeshStandardMaterial({
      color: this._frictionColor(terrain.friction ?? 0.6),
      roughness: 0.95,
      metalness: 0,
      side: DoubleSide,
    });
    this.mesh = new Mesh(geo, mat);
    this.mesh.receiveShadow = true;
    this.group.add(this.mesh);

    // Faint wireframe so slope features read clearly.
    this.edges = new LineSegments(
      new EdgesGeometry(geo, 1),
      new LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.12 }),
    );
    this.group.add(this.edges);
  }

  dispose() {
    for (const obj of [this.mesh, this.edges]) {
      if (!obj) continue;
      this.group.remove(obj);
      obj.geometry.dispose();
      obj.material.dispose();
    }
    this.mesh = this.edges = null;
  }
}
