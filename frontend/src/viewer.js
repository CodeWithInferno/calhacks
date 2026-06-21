import {
  Scene,
  Color,
  PerspectiveCamera,
  WebGLRenderer,
  AmbientLight,
  DirectionalLight,
  HemisphereLight,
  Group,
  Mesh,
  PlaneGeometry,
  MeshStandardMaterial,
  GridHelper,
  PCFShadowMap,
  MathUtils,
} from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

/**
 * Builds the scene (ground, lights, OrbitControls) and a Z-up `frame` group.
 *
 * URDF + the trajectory use a Z-up convention; three.js is Y-up. Everything
 * recorded in robot frame (the robot itself and any loose objects) is parented
 * under `frame`, which is rotated -90° about X. So a point at z=0 in robot
 * space lands on the y=0 floor plane, and all trajectory math stays in native
 * Z-up coordinates.
 */
export function createViewer(canvas) {
  const renderer = new WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = PCFShadowMap;

  const scene = new Scene();
  scene.background = new Color(0x15171c);

  const camera = new PerspectiveCamera(55, 1, 0.01, 100);
  camera.position.set(2.2, 1.8, 2.4);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 0.6, 0);

  // --- lighting ---
  scene.add(new AmbientLight(0xffffff, 0.35));
  scene.add(new HemisphereLight(0xbcd3ff, 0x202024, 0.5));
  const key = new DirectionalLight(0xffffff, 1.4);
  key.position.set(3, 5, 2);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.camera.near = 0.5;
  key.shadow.camera.far = 20;
  const s = 5;
  Object.assign(key.shadow.camera, { left: -s, right: s, top: s, bottom: -s });
  scene.add(key);

  // --- ground (three.js Y-up world) ---
  const ground = new Mesh(
    new PlaneGeometry(20, 20),
    new MeshStandardMaterial({ color: 0x202531, roughness: 0.95, metalness: 0 }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);

  const grid = new GridHelper(20, 40, 0x3a3f4b, 0x2a2e38);
  grid.position.y = 0.001;
  scene.add(grid);

  // --- Z-up content frame (robot + loose objects live here) ---
  const frame = new Group();
  frame.rotation.x = -Math.PI / 2;
  scene.add(frame);

  function resize() {
    const w = canvas.clientWidth || window.innerWidth;
    const h = canvas.clientHeight || window.innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resize);
  resize();

  function render() {
    controls.update();
    renderer.render(scene, camera);
  }

  return { renderer, scene, camera, controls, frame, ground, grid, render, resize, MathUtils };
}
