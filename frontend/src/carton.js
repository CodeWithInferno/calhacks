import {
  Group,
  Mesh,
  BoxGeometry,
  MeshStandardMaterial,
  EdgesGeometry,
  LineSegments,
  LineBasicMaterial,
  Color,
} from 'three';

const FT = 0.3048; // metres per foot

/**
 * A brown cardboard carton the robot carries, held against its chest. Nominal
 * size is ~2 ft square; it grows with the payload weight so a heavier carton
 * reads as visibly bigger/heavier. Parented to the torso so it moves and tilts
 * with the robot.
 */
export class Carton {
  constructor(robot) {
    // Hold it on the torso if available, else the robot root.
    const parent = robot.links?.torso_link ?? robot;

    this.group = new Group();
    parent.add(this.group);

    const box = new Mesh(
      new BoxGeometry(1, 1, 1),
      new MeshStandardMaterial({ color: new Color(0x9c6b3f), roughness: 0.92, metalness: 0 }),
    );
    box.castShadow = true;
    box.receiveShadow = true;
    this.group.add(box);

    // Cardboard seams.
    this.edges = new LineSegments(
      new EdgesGeometry(box.geometry),
      new LineBasicMaterial({ color: 0x5c3c1e }),
    );
    box.add(this.edges);

    // Packing-tape strip across the top.
    this.tape = new Mesh(
      new BoxGeometry(0.18, 1.02, 1.02),
      new MeshStandardMaterial({ color: 0xc9a36a, roughness: 0.8 }),
    );
    box.add(this.tape);

    this.box = box;
    this.group.visible = false;
  }

  /** Size + position the carton for the carried weight (kg). Hidden at 0. */
  update(kg) {
    if (kg <= 0) {
      this.group.visible = false;
      return;
    }
    this.group.visible = true;

    // ~1 ft box, growing only modestly with weight (heavier = a bit bigger).
    const sizeFt = 1.0 + 0.02 * kg; // ft
    const s = Math.min(1.8, Math.max(0.9, sizeFt)) * FT; // metres, clamped
    this.box.scale.setScalar(s);

    // Cradled in the hands in front of the waist (torso-local: +X forward,
    // +Z up). A heavier box rides a touch lower.
    this.group.position.set(0.18 + s / 2, 0, -0.06 - 0.003 * kg);
  }
}
