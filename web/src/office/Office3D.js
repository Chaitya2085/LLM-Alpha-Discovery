// The 3D office: a cut-away building with one room per desk, the agents at their desks, and wall screens
// with live numbers. Plain three.js; the React wrapper (Office.jsx) feeds it the team's data.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/examples/jsm/renderers/CSS2DRenderer.js";

// ---------------------------------------------------------------- layout (metres; x right, z towards you)
const CORRIDOR = 1.6;          // half-width of the corridor along z = 0
const TOP = [-9, -CORRIDOR], BOTTOM = [CORRIDOR, 9];
export const ROOMS = [
  { id: "Head office", x: [-16, -8], z: TOP, rug: "#6b2f2a", label: "Head office" },
  { id: "Market data", x: [-8, 0], z: TOP, rug: "#2d4a5e", label: "Market data" },
  { id: "Model room", x: [0, 8], z: TOP, rug: "#3d3a63", label: "Model room" },
  { id: "Research lab", x: [8, 16], z: TOP, rug: "#2f5a45", label: "Research lab" },
  { id: "Reception", x: [-16, -8], z: BOTTOM, rug: "#5a4630", label: "Reception & lounge" },
  { id: "Meeting room", x: [-8, 0], z: BOTTOM, rug: "#4a3a2a", label: "Meeting room" },
  { id: "Risk & compliance", x: [0, 8], z: BOTTOM, rug: "#5e3434", label: "Risk & compliance" },
  { id: "Communications", x: [8, 16], z: BOTTOM, rug: "#5b4b25", label: "Communications" },
];
const DESK_SPOTS = {       // where each agent's desk sits inside its room (offset from the room centre)
  "Chief Strategist": [0, -1.4],
  "Data Collector": [-1.8, -1.2], "Data Checker": [1.8, -1.2],
  "Regime Watcher": [-1.8, -1.2], "Model Trainer": [1.8, -1.2],
  "Research lead": [-1.2, -1.2],
  "Risk Officer": [-1.8, 1.2], "Compliance Officer": [1.8, 1.2],
  "Explainer": [-1.8, 1.2], "Reporter": [1.8, 1.2],
};
const SHIRT = {
  "Head office": "#c0504d", "Market data": "#4a86b8", "Model room": "#7b6fd0", "Research lab": "#3fa877",
  "Risk & compliance": "#d07a3a", "Communications": "#c9a227",
};
const STATUS_COLOR = { ok: "#3fcf8e", warning: "#f2b544", stop: "#ef6a5b", idle: "#5b7a99" };
const WALL_H = 1.25, BACK_H = 2.9, T = 0.14;

const room = id => ROOMS.find(r => r.id === id);
const centre = r => [(r.x[0] + r.x[1]) / 2, (r.z[0] + r.z[1]) / 2];
const isTop = r => r.z[1] <= 0;
const door = r => { const [cx] = centre(r); return [cx, isTop(r) ? -CORRIDOR : CORRIDOR]; };

// ---------------------------------------------------------------- textures
function canvasTex(w, h, draw) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  draw(c.getContext("2d"), w, h);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}
function herringbone(a = "#8a5a35", b = "#7a4d2c", c = "#6c4426") {
  const t = canvasTex(512, 512, (g, w) => {
    const L = 64, Wd = 16;
    g.fillStyle = c; g.fillRect(0, 0, w, w);
    for (let y = -L; y < w + L; y += Wd * 2) {
      for (let x = -L; x < w + L; x += L) {
        [[0, a], [1, b]].forEach(([k, col]) => {
          g.save(); g.translate(x + k * L / 2, y + k * Wd); g.rotate(k ? -Math.PI / 4 : Math.PI / 4);
          g.fillStyle = col; g.fillRect(0, 0, L * 0.7, Wd - 1.5);
          g.fillStyle = "rgba(0,0,0,0.08)"; g.fillRect(0, Wd - 3, L * 0.7, 1.5); g.restore();
        });
      }
    }
  });
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}
function signTex(text, sub) {
  return canvasTex(1024, 256, (g, w, h) => {
    g.fillStyle = "#2a1c14"; g.fillRect(0, 0, w, h);
    g.fillStyle = "#f3e4c8"; g.font = "700 110px Georgia, serif"; g.textAlign = "center"; g.textBaseline = "middle";
    g.fillText(text, w / 2, h / 2 - 18);
    g.font = "500 34px Menlo, monospace"; g.fillStyle = "#c9b28c"; g.fillText(sub, w / 2, h / 2 + 70);
  });
}

// ---------------------------------------------------------------- the scene

export default class Office3D {
  constructor(el, { onSelect, dark = true, reducedMotion = false } = {}) {
    this.el = el;
    this.onSelect = onSelect || (() => {});
    this.reduced = reducedMotion;
    this.agents = new Map();
    this.screens = {};
    this.clickable = [];
    this.clock = new THREE.Clock();

    const r = this.renderer = new THREE.WebGLRenderer({ antialias: true });
    r.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 1.25;
    el.appendChild(r.domElement);
    const labels = this.labels = new CSS2DRenderer();
    labels.domElement.className = "office-labels";
    el.appendChild(labels.domElement);

    const s = this.scene = new THREE.Scene();
    s.background = new THREE.Color(dark ? "#0d1210" : "#dfe4df");
    s.fog = new THREE.Fog(s.background, 55, 95);
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.1, 200);
    this.home = { pos: new THREE.Vector3(0, 23, 25), target: new THREE.Vector3(0, 0, 0.8) };
    this.camera.position.copy(this.home.pos);
    const c = this.controls = new OrbitControls(this.camera, labels.domElement);
    c.target.copy(this.home.target);
    c.enableDamping = true; c.dampingFactor = 0.08;
    c.minDistance = 6; c.maxDistance = 55;
    c.minPolarAngle = 0.25; c.maxPolarAngle = 1.2;
    c.screenSpacePanning = false;

    s.add(new THREE.HemisphereLight("#ffe9cf", "#3a2c20", 1.35));
    const sun = new THREE.DirectionalLight("#fff1dc", 2.2);
    sun.position.set(-14, 26, 12);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -22, right: 22, top: 16, bottom: -16, near: 1, far: 70 });
    sun.shadow.bias = -0.0005;
    s.add(sun);

    this.mats = {
      wood: new THREE.MeshStandardMaterial({ map: herringbone(), roughness: 0.75 }),
      corridor: new THREE.MeshStandardMaterial({ map: herringbone("#9a6a42", "#8a5c37", "#7a5030"), roughness: 0.7 }),
      wall: new THREE.MeshStandardMaterial({ color: "#3b2b22", roughness: 0.9 }),
      wallTop: new THREE.MeshStandardMaterial({ color: "#d8c6aa", roughness: 0.8 }),
      desk: new THREE.MeshStandardMaterial({ color: "#5b3a24", roughness: 0.6 }),
      deskTop: new THREE.MeshStandardMaterial({ color: "#7a5236", roughness: 0.45 }),
      metal: new THREE.MeshStandardMaterial({ color: "#2a2a2a", roughness: 0.4, metalness: 0.6 }),
      chair: new THREE.MeshStandardMaterial({ color: "#5a1f1c", roughness: 0.55 }),
      black: new THREE.MeshStandardMaterial({ color: "#151515", roughness: 0.5 }),
      skin: new THREE.MeshStandardMaterial({ color: "#d9a57c", roughness: 0.7 }),
      hair: new THREE.MeshStandardMaterial({ color: "#2b1d16", roughness: 0.8 }),
      leaf: new THREE.MeshStandardMaterial({ color: "#3f7d3a", roughness: 0.8 }),
      pot: new THREE.MeshStandardMaterial({ color: "#c9b79c", roughness: 0.8 }),
      sofa: new THREE.MeshStandardMaterial({ color: "#6d6158", roughness: 0.9 }),
      felt: new THREE.MeshStandardMaterial({ color: "#2f6b4f", roughness: 0.9 }),
      glass: new THREE.MeshStandardMaterial({ color: "#a8c8e0", roughness: 0.1, metalness: 0.1, transparent: true, opacity: 0.35 }),
    };
    this.build();
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.downAt = null;
    labels.domElement.addEventListener("pointerdown", e => { this.downAt = [e.clientX, e.clientY]; });
    labels.domElement.addEventListener("pointerup", e => this.click(e));
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(el);
    this.resize();
    this.fly = null;
    this.nextStroll = 8 + Math.random() * 10;
    this.running = true;
    const loop = () => { if (!this.running) return; this.frame = requestAnimationFrame(loop); this.tick(); };
    loop();
  }

  // ------------------------------------------------------------ helpers
  box(w, h, d, mat, x, y, z, parent = this.scene, cast = true) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
    m.position.set(x, y, z); m.castShadow = cast; m.receiveShadow = true;
    parent.add(m); return m;
  }
  cyl(rt, rb, h, mat, x, y, z, parent = this.scene, seg = 16) {
    const m = new THREE.Mesh(new THREE.CylinderGeometry(rt, rb, h, seg), mat);
    m.position.set(x, y, z); m.castShadow = true; m.receiveShadow = true;
    parent.add(m); return m;
  }
  wall(x0, z0, x1, z1, h = WALL_H, gap = null) {
    // a straight wall from (x0,z0) to (x1,z1), optionally with a door gap [from, to] along its length
    const len = Math.hypot(x1 - x0, z1 - z0), along = x1 !== x0;
    const parts = gap ? [[0, gap[0]], [gap[1], len]] : [[0, len]];
    parts.forEach(([a, b]) => {
      if (b - a < 0.05) return;
      const mid = (a + b) / 2;
      const x = along ? x0 + Math.sign(x1 - x0) * mid : x0, z = along ? z0 : z0 + Math.sign(z1 - z0) * mid;
      const w = along ? b - a : T, d = along ? T : b - a;
      this.box(w, h, d, this.mats.wall, x, h / 2, z);
      this.box(w + 0.02, 0.04, d + 0.02, this.mats.wallTop, x, h + 0.02, z, this.scene, false);
    });
  }
  plant(x, z, s = 1) {
    const g = new THREE.Group(); g.position.set(x, 0, z); g.scale.setScalar(s);
    this.cyl(0.22, 0.17, 0.42, this.mats.pot, 0, 0.21, 0, g);
    [[0, 0.75, 0, 0.32], [0.12, 1.0, 0.05, 0.24], [-0.1, 0.95, -0.08, 0.22]].forEach(([a, b, c, r]) => {
      const m = new THREE.Mesh(new THREE.IcosahedronGeometry(r, 1), this.mats.leaf);
      m.position.set(a, b, c); m.castShadow = true; g.add(m);
    });
    this.scene.add(g);
  }
  label(text, cls, x, y, z, parent = this.scene) {
    const d = document.createElement("div"); d.className = cls; d.textContent = text;
    const o = new CSS2DObject(d); o.position.set(x, y, z); parent.add(o); return o;
  }

  // ------------------------------------------------------------ building
  build() {
    const s = this.scene, m = this.mats;
    // ground and floors
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(140, 140), new THREE.MeshStandardMaterial({ color: s.background.clone().multiplyScalar(1.15), roughness: 1 }));
    ground.rotation.x = -Math.PI / 2; ground.position.y = -0.02; ground.receiveShadow = true; s.add(ground);
    const slab = this.box(33, 0.3, 19, new THREE.MeshStandardMaterial({ color: "#2a1f19", roughness: 0.9 }), 0, -0.16, 0, s, false);
    slab.receiveShadow = true;
    const corr = new THREE.Mesh(new THREE.PlaneGeometry(32, CORRIDOR * 2), m.corridor.clone());
    corr.material.map = m.corridor.map.clone(); corr.material.map.repeat.set(16, 1.6); corr.material.map.needsUpdate = true;
    corr.rotation.x = -Math.PI / 2; corr.position.y = 0.001; corr.receiveShadow = true; s.add(corr);

    ROOMS.forEach(r => {
      const [cx, cz] = centre(r), w = r.x[1] - r.x[0], d = r.z[1] - r.z[0];
      const fl = new THREE.Mesh(new THREE.PlaneGeometry(w, d), m.wood.clone());
      fl.material.map = m.wood.map.clone(); fl.material.map.repeat.set(w / 2, d / 2); fl.material.map.needsUpdate = true;
      fl.rotation.x = -Math.PI / 2; fl.position.set(cx, 0.002, cz); fl.receiveShadow = true;
      fl.userData.room = r.id; s.add(fl); this.clickable.push(fl);
      const rug = new THREE.Mesh(new THREE.PlaneGeometry(w * 0.62, d * 0.55), new THREE.MeshStandardMaterial({ color: r.rug, roughness: 1 }));
      rug.rotation.x = -Math.PI / 2; rug.position.set(cx, 0.006, cz); rug.receiveShadow = true; s.add(rug);
      // walls: side walls between rooms, the corridor wall with a door, outer walls
      const corridorZ = isTop(r) ? r.z[1] : r.z[0], outerZ = isTop(r) ? r.z[0] : r.z[1];
      this.wall(r.x[0], corridorZ, r.x[1], corridorZ, WALL_H, [w / 2 - 0.85, w / 2 + 0.85]);
      this.wall(r.x[0], r.z[0], r.x[0], r.z[1]);
      if (isTop(r)) {                                   // tall back wall with windows
        this.wall(r.x[0], outerZ, r.x[1], outerZ, BACK_H);
        for (let i = 0; i < 3; i++) this.box(1.5, 1.1, 0.03, m.glass, r.x[0] + w * (i + 0.5) / 3, 1.75, outerZ + T / 2 + 0.02, s, false);
      } else this.wall(r.x[0], outerZ, r.x[1], outerZ);
      this.label(r.label, "office-room", cx, isTop(r) ? BACK_H + 0.5 : WALL_H + 0.4, outerZ);
    });
    this.wall(16, -9, 16, 9);
    this.wall(-16, -CORRIDOR, -16, CORRIDOR); this.wall(16, -CORRIDOR, 16, CORRIDOR);

    this.furnish();
  }

  desk(x, z, faceZ = 1, name = null) {
    const g = new THREE.Group(); g.position.set(x, 0, z); g.rotation.y = faceZ > 0 ? 0 : Math.PI;
    const m = this.mats;
    this.box(1.6, 0.06, 0.75, m.deskTop, 0, 0.76, 0, g);
    [[-0.74, -0.32], [0.74, -0.32], [-0.74, 0.32], [0.74, 0.32]].forEach(([a, b]) => this.box(0.05, 0.74, 0.05, m.metal, a, 0.37, b, g));
    this.box(1.5, 0.5, 0.04, m.desk, 0, 0.5, -0.34, g);
    this.box(0.08, 0.28, 0.08, m.metal, 0, 0.93, -0.2, g);
    const screenMat = new THREE.MeshStandardMaterial({ color: "#0b0f14", emissive: STATUS_COLOR.idle, emissiveIntensity: 0.6 });
    this.box(0.78, 0.46, 0.04, m.black, 0, 1.26, -0.22, g);
    const scr = this.box(0.72, 0.4, 0.01, screenMat, 0, 1.26, -0.197, g, false);
    this.box(0.5, 0.02, 0.16, m.black, 0, 0.8, 0.1, g);          // keyboard
    // chair
    const ch = new THREE.Group(); ch.position.set(0, 0, 0.75); g.add(ch);
    this.cyl(0.04, 0.04, 0.42, m.metal, 0, 0.21, 0, ch);
    this.box(0.55, 0.1, 0.52, m.chair, 0, 0.47, 0, ch);
    this.box(0.55, 0.62, 0.09, m.chair, 0, 0.85, 0.25, ch);
    this.scene.add(g);
    return { group: g, screen: scr, seat: new THREE.Vector3(x, 0, z + 0.75 * faceZ), faceZ };
  }

  furnish() {
    const m = this.mats;
    // head office: big desk, bookshelf, sofa
    const ho = room("Head office"), [hx, hz] = centre(ho);
    this.box(2.4, 2.0, 0.4, m.desk, hx - 2.4, 1.0, ho.z[0] + 0.35);
    for (let i = 0; i < 4; i++) this.box(2.3, 0.04, 0.36, m.deskTop, hx - 2.4, 0.4 + i * 0.45, ho.z[0] + 0.4, this.scene, false);
    this.box(2.0, 0.45, 0.8, m.sofa, hx + 2.3, 0.23, hz + 1.6); this.box(2.0, 0.5, 0.2, m.sofa, hx + 2.3, 0.6, hz + 1.95);
    this.plant(ho.x[0] + 0.6, ho.z[1] - 0.6);
    // market data: server rack with blinking lights
    const md = room("Market data"), [mx] = centre(md);
    this.rack = this.box(0.8, 2.0, 0.7, m.black, mx, 1.0, md.z[0] + 0.6);
    this.leds = [];
    for (let i = 0; i < 8; i++) {
      const led = this.box(0.05, 0.05, 0.02, new THREE.MeshBasicMaterial({ color: "#3fcf8e" }), mx - 0.25 + (i % 4) * 0.16, 0.6 + Math.floor(i / 4) * 0.9, md.z[0] + 0.96, this.scene, false);
      this.leds.push(led);
    }
    this.plant(md.x[1] - 0.6, md.z[1] - 0.6, 0.9);
    // model room: plants; research lab: whiteboard and a plant wall
    const mr = room("Model room");
    this.plant(mr.x[0] + 0.6, mr.z[1] - 0.6, 0.9);
    const rl = room("Research lab"), [rx, rz] = centre(rl);
    this.box(2.6, 1.3, 0.05, new THREE.MeshStandardMaterial({ color: "#f4f1ea", roughness: 0.4 }), rx + 1.2, 1.55, rl.z[0] + 0.12);
    this.box(1.2, 0.75, 1.2, m.deskTop, rx + 1.8, 0.38, rz + 1.3);
    this.plant(rl.x[1] - 0.6, rl.z[1] - 0.6); this.plant(rl.x[1] - 0.6, rl.z[0] + 0.6, 1.2);
    // reception & lounge: reception desk + sign, sofas, pool table, coffee machine
    const rc = room("Reception"), [cx, cz] = centre(rc);
    this.box(3.2, 1.05, 0.7, m.desk, cx, 0.52, cz - 1.5); this.box(3.3, 0.06, 0.8, m.deskTop, cx, 1.07, cz - 1.5);
    const sign = new THREE.Mesh(new THREE.PlaneGeometry(3.4, 0.85), new THREE.MeshBasicMaterial({ map: signTex("AGENT OFFICE", "LLM ALPHA DISCOVERY · NSE"), toneMapped: false }));
    sign.position.set(cx, 1.65, cz - 1.12); sign.rotation.y = 0; this.scene.add(sign);
    this.box(2.6, 0.12, 1.4, m.felt, cx - 1.2, 0.85, cz + 1.8); this.box(2.8, 0.75, 1.6, m.desk, cx - 1.2, 0.4, cz + 1.8);
    this.box(0.8, 0.45, 2.0, m.sofa, rc.x[1] - 0.8, 0.23, cz + 1.3); this.box(0.2, 0.5, 2.0, m.sofa, rc.x[1] - 0.35, 0.6, cz + 1.3);
    this.coffee = new THREE.Vector3(rc.x[0] + 0.9, 0, cz - 1.2);
    this.box(0.6, 1.2, 0.5, m.metal, rc.x[0] + 0.45, 0.6, cz - 1.2);
    this.plant(rc.x[0] + 0.6, rc.z[1] - 0.6, 1.2);
    // meeting room: long table, chairs, big live board
    const mt = room("Meeting room"), [tx, tz] = centre(mt);
    this.box(4.6, 0.08, 1.6, m.deskTop, tx, 0.78, tz + 0.4); this.box(4.2, 0.74, 0.3, m.desk, tx, 0.38, tz + 0.4);
    this.meetingSpots = [];
    for (let i = 0; i < 5; i++) {
      [-1, 1].forEach(side => {
        const x = tx - 1.8 + i * 0.9, z = tz + 0.4 + side * 1.15;
        this.box(0.45, 0.1, 0.45, m.chair, x, 0.47, z); this.box(0.45, 0.5, 0.08, m.chair, x, 0.78, z + side * 0.2);
        this.meetingSpots.push(new THREE.Vector3(x, 0, z));
      });
    }
    // the live board (meeting room) and one screen per team room
    // screens face the camera (+z): on the back wall of top rooms, on the corridor wall of bottom rooms
    this.screens.board = this.screen(2.9, 1.5, mt.x[0] + 1.75, 1.95, mt.z[0] + 0.12, 1024, 530);
    ROOMS.filter(r => SHIRT[r.id]).forEach(r => {
      const [x] = centre(r);
      if (isTop(r)) this.screens[r.id] = this.screen(2.0, 1.05, x + (r.id === "Research lab" ? -2.2 : 2.3), 1.85, r.z[0] + 0.12, 640, 336);
      else this.screens[r.id] = this.screen(2.0, 1.05, x + 2.3, 1.5, r.z[0] + 0.14, 640, 336, true);
    });
    // desks and people
    const DESK_NAMES = Object.keys(DESK_SPOTS);
    DESK_NAMES.forEach(name => {
      const desk = this.deskOf(name);
      const r = room(desk), [x, z] = centre(r), [ox, oz] = DESK_SPOTS[name];
      const faceZ = oz < 0 ? 1 : -1;                  // desks face into the room
      const d = this.desk(x + ox, z + oz, faceZ);
      this.addAgent(name, desk, d);
    });
  }

  deskOf(name) {
    return { "Chief Strategist": "Head office", "Data Collector": "Market data", "Data Checker": "Market data",
      "Regime Watcher": "Model room", "Model Trainer": "Model room", "Risk Officer": "Risk & compliance",
      "Compliance Officer": "Risk & compliance", "Explainer": "Communications", "Reporter": "Communications",
      "Research lead": "Research lab" }[name];
  }

  screen(w, h, x, y, z, cw, chh, standing = false) {
    if (standing) this.box(0.1, y - h / 2, 0.1, this.mats.metal, x, (y - h / 2) / 2, z);
    const c = document.createElement("canvas"); c.width = cw; c.height = chh;
    const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
    this.box(w + 0.1, h + 0.1, 0.06, this.mats.black, x, y, z);
    const p = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ map: tex, toneMapped: false }));
    p.position.set(x, y, z + 0.035); this.scene.add(p);
    return { c, g: c.getContext("2d"), tex };
  }

  addAgent(name, desk, d) {
    const g = new THREE.Group();
    const shirt = new THREE.MeshStandardMaterial({ color: SHIRT[desk], roughness: 0.7 });
    const pants = new THREE.MeshStandardMaterial({ color: "#2b2f3a", roughness: 0.8 });
    const body = new THREE.Group(); g.add(body);
    const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.2, 0.42, 4, 10), shirt); torso.position.y = 1.02; torso.castShadow = true; body.add(torso);
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.17, 16, 12), this.mats.skin); head.position.y = 1.52; head.castShadow = true; body.add(head);
    const hair = new THREE.Mesh(new THREE.SphereGeometry(0.175, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2.1), this.mats.hair); hair.position.y = 1.54; body.add(hair);
    const legs = [-0.09, 0.09].map(x => { const l = new THREE.Mesh(new THREE.CapsuleGeometry(0.075, 0.42, 4, 8), pants); l.position.set(x, 0.42, 0); l.castShadow = true; body.add(l); return l; });
    const arms = [-0.27, 0.27].map(x => { const a = new THREE.Mesh(new THREE.CapsuleGeometry(0.06, 0.38, 4, 8), shirt); a.position.set(x, 1.0, 0); a.castShadow = true; body.add(a); return a; });
    [torso, head, hair].forEach(o => { o.userData.agent = name; this.clickable.push(o); });
    // name tag
    const tag = document.createElement("div"); tag.className = "office-tag";
    tag.innerHTML = `<i></i><span></span>`; tag.querySelector("span").textContent = name;
    // stop the camera controls from grabbing the pointer, so a click on the tag selects the agent
    tag.addEventListener("pointerdown", e => e.stopPropagation());
    tag.addEventListener("click", e => { e.stopPropagation(); this.onSelect(name); });
    const tagObj = new CSS2DObject(tag); tagObj.position.set(0, 1.85, 0); g.add(tagObj);
    const bubble = document.createElement("div"); bubble.className = "office-bubble"; bubble.hidden = true;
    const bubbleObj = new CSS2DObject(bubble); bubbleObj.position.set(0, 2.3, 0); g.add(bubbleObj);
    this.scene.add(g);
    const a = { name, desk, d, g, body, legs, arms, tag, bubble, path: [], state: "sit", status: "idle", lastTime: null, phase: Math.random() * 6, wait: 0, queue: [] };
    this.sit(a);
    this.agents.set(name, a);
  }

  sit(a) {
    a.state = "sit";
    a.g.position.copy(a.d.seat);
    a.g.rotation.y = a.d.faceZ > 0 ? Math.PI : 0;
    a.body.position.y = -0.3;
    a.legs.forEach(l => { l.rotation.x = -Math.PI / 2.4; l.position.z = a.d.faceZ > 0 ? 0 : 0; });
  }
  stand(a) { a.body.position.y = 0; a.legs.forEach(l => { l.rotation.x = 0; }); }

  route(a, to) {
    // desk -> own door -> corridor -> target door -> target spot
    const pts = [];
    const from = a.g.position.clone();
    const [fx, fz] = door(room(a.desk));
    pts.push(new THREE.Vector3(from.x, 0, fz + (fz < 0 ? -0.8 : 0.8)));
    pts.push(new THREE.Vector3(fx, 0, fz + (fz < 0 ? -0.8 : 0.8)));
    pts.push(new THREE.Vector3(fx, 0, 0));
    const tr = ROOMS.find(r => to.x >= r.x[0] && to.x <= r.x[1] && to.z >= r.z[0] && to.z <= r.z[1]);
    if (tr) {
      const [dx, dz] = door(tr);
      pts.push(new THREE.Vector3(dx, 0, 0));
      pts.push(new THREE.Vector3(dx, 0, dz + (dz < 0 ? -0.8 : 0.8)));
      pts.push(new THREE.Vector3(to.x, 0, dz + (dz < 0 ? -0.8 : 0.8)));
    }
    pts.push(to.clone());
    return pts;
  }

  // a real event: walk to the meeting room board, say it, walk back
  report(name, message, status) {
    const a = this.agents.get(name);
    if (!a) return;
    if (this.reduced) { this.say(a, message, status, 7); return; }
    if (a.state !== "sit") { a.queue.push([message, status]); return; }
    const spot = this.meetingSpots[[...this.agents.keys()].indexOf(name) % this.meetingSpots.length];
    this.stand(a);
    a.state = "walk"; a.path = this.route(a, spot); a.after = () => { a.state = "talk"; a.wait = 4.5; this.say(a, message, status, 6); };
    a.back = true;
  }
  say(a, message, status, secs) {
    a.bubble.textContent = message.length > 110 ? message.slice(0, 107) + "…" : message;
    a.bubble.dataset.status = status; a.bubble.hidden = false;
    clearTimeout(a.bubbleTimer);
    a.bubbleTimer = setTimeout(() => { a.bubble.hidden = true; }, secs * 1000);
  }

  // ------------------------------------------------------------ data in
  update({ agents = [], overview = null, exam = null }) {
    agents.forEach(info => {
      const a = this.agents.get(info.name);
      if (!a) return;
      a.status = info.status;
      a.tag.dataset.status = info.status;
      a.d.screen.material.emissive.set(STATUS_COLOR[info.status] || STATUS_COLOR.idle);
      if (a.lastTime && info.time && info.time !== a.lastTime && info.message) this.report(info.name, info.message, info.status);
      a.lastTime = info.time;
      a.message = info.message;
    });
    this.drawBoard(overview, exam);
    ROOMS.filter(r => this.screens[r.id] && r.id !== "board").forEach(r => this.drawRoomScreen(r.id, agents.filter(x => x.desk === r.id)));
  }

  drawBoard(o, exam) {
    const s = this.screens.board; if (!s) return;
    const { g, c } = s, W = c.width, H = c.height;
    g.fillStyle = "#07110d"; g.fillRect(0, 0, W, H);
    g.fillStyle = "#3fcf8e"; g.font = "600 40px Menlo, monospace"; g.fillText("LIVE BOARD · MODEL vs NIFTY 200", 40, 70);
    g.fillStyle = "#1d3a2e"; g.fillRect(40, 92, W - 80, 3);
    const rows = [
      ["Data through", o?.as_of || "–"], ["Market mood", o?.regime || "–"],
      ["Last run", o?.running ? "running…" : (o?.last_run?.status || "–")], ["Ledger", `${o?.ledger_weeks ?? 0} week(s) sealed`],
    ];
    if (exam) rows.push(["Exam 2021+", `${(exam.model * 100).toFixed(1)}% vs ${(exam.index * 100).toFixed(1)}% a year`], ["Verdict", exam.verdict]);
    g.font = "400 34px Menlo, monospace";
    rows.forEach(([k, v], i) => {
      g.fillStyle = "#8fb3a3"; g.fillText(k, 40, 150 + i * 58);
      g.fillStyle = "#eaf5ef"; g.fillText(v, 380, 150 + i * 58);
    });
    s.tex.needsUpdate = true;
  }
  drawRoomScreen(id, list) {
    const s = this.screens[id]; if (!s) return;
    const { g, c } = s, W = c.width, H = c.height;
    g.fillStyle = "#0a0f14"; g.fillRect(0, 0, W, H);
    g.fillStyle = SHIRT[id]; g.fillRect(0, 0, W, 10);
    g.fillStyle = "#e8eef3"; g.font = "600 34px Menlo, monospace"; g.fillText(id.toUpperCase(), 28, 62);
    g.font = "400 22px Menlo, monospace";
    let y = 110;
    list.forEach(a => {
      g.fillStyle = STATUS_COLOR[a.status] || STATUS_COLOR.idle; g.beginPath(); g.arc(36, y - 8, 9, 0, 7); g.fill();
      g.fillStyle = "#e8eef3"; g.fillText(a.name, 56, y);
      g.fillStyle = "#93a4b3";
      wrap(g, a.message || a.job, 56, y + 30, W - 80, 26, 3);
      y += 120;
    });
    s.tex.needsUpdate = true;
  }

  // ------------------------------------------------------------ camera & input
  flyTo(id) {
    let pos, target;
    if (!id || id === "overview") { pos = this.home.pos.clone(); target = this.home.target.clone(); }
    else {
      const r = room(id) || room(this.deskOf(id) || "");
      const a = this.agents.get(id);
      if (a) { target = a.g.position.clone().setY(1); pos = target.clone().add(new THREE.Vector3(2.5, 4.5, 6)); }
      else if (r) { const [x, z] = centre(r); target = new THREE.Vector3(x, 0.6, z); pos = new THREE.Vector3(x + 2, 9, z + (isTop(r) ? 9 : 10)); }
      else return;
    }
    this.fly = { t: 0, p0: this.camera.position.clone(), t0: this.controls.target.clone(), p1: pos, t1: target };
  }
  click(e) {
    if (!this.downAt || Math.hypot(e.clientX - this.downAt[0], e.clientY - this.downAt[1]) > 6) return;
    const r = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hit = this.raycaster.intersectObjects(this.clickable, false)[0];
    if (!hit) return;
    if (hit.object.userData.agent) this.onSelect(hit.object.userData.agent);
    else if (hit.object.userData.room) this.flyTo(hit.object.userData.room);
  }
  resize() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h); this.labels.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.fov = w < 600 ? 52 : 38;
    this.camera.updateProjectionMatrix();
  }

  // ------------------------------------------------------------ animation
  tick() {
    const dt = Math.min(this.clock.getDelta(), 0.05), t = this.clock.elapsedTime;
    if (this.fly) {
      this.fly.t = Math.min(1, this.fly.t + dt / 0.9);
      const k = 1 - Math.pow(1 - this.fly.t, 3);
      this.camera.position.lerpVectors(this.fly.p0, this.fly.p1, k);
      this.controls.target.lerpVectors(this.fly.t0, this.fly.t1, k);
      if (this.fly.t >= 1) this.fly = null;
    }
    this.controls.update();
    this.leds?.forEach((l, i) => { l.visible = Math.sin(t * (2 + i * 0.7) + i) > -0.3; });
    if (!this.reduced) {
      this.nextStroll -= dt;
      if (this.nextStroll <= 0) {                     // someone fetches a coffee now and then
        this.nextStroll = 18 + Math.random() * 25;
        const idle = [...this.agents.values()].filter(a => a.state === "sit");
        const a = idle[Math.floor(Math.random() * idle.length)];
        if (a) { this.stand(a); a.state = "walk"; a.path = this.route(a, this.coffee.clone().add(new THREE.Vector3(0.4, 0, 0.3 * Math.random()))); a.after = () => { a.state = "talk"; a.wait = 2.5; }; a.back = true; }
      }
    }
    this.agents.forEach(a => this.step(a, dt, t));
    this.renderer.render(this.scene, this.camera);
    this.labels.render(this.scene, this.camera);
  }
  step(a, dt, t) {
    if (a.state === "sit") {
      a.arms.forEach((arm, i) => { arm.rotation.x = -1.1 + Math.sin(t * 9 + i * 1.7 + a.phase) * (a.status === "idle" ? 0.02 : 0.12); });
      a.body.rotation.y = Math.sin(t * 0.5 + a.phase) * 0.06;
      return;
    }
    if (a.state === "talk") {
      a.arms.forEach((arm, i) => { arm.rotation.x = Math.sin(t * 3 + i) * 0.25; });
      a.wait -= dt;
      if (a.wait <= 0) {
        if (a.back) { a.back = false; a.state = "walk"; a.path = this.route(a, a.d.seat); a.after = () => { this.sit(a); const q = a.queue.shift(); if (q) setTimeout(() => this.report(a.name, ...q), 400); }; }
      }
      return;
    }
    // walking along the path
    const target = a.path[0];
    if (!target) { a.after?.(); return; }
    const pos = a.g.position, dir = target.clone().sub(pos); dir.y = 0;
    const dist = dir.length(), speed = 2.2;
    if (dist < 0.05) { a.path.shift(); return; }
    dir.normalize();
    pos.addScaledVector(dir, Math.min(dist, speed * dt));
    const want = Math.atan2(dir.x, dir.z);
    let diff = want - a.g.rotation.y; diff = Math.atan2(Math.sin(diff), Math.cos(diff));
    a.g.rotation.y += diff * Math.min(1, dt * 10);
    const sw = Math.sin(t * 10 + a.phase);
    a.legs[0].rotation.x = sw * 0.6; a.legs[1].rotation.x = -sw * 0.6;
    a.arms[0].rotation.x = -sw * 0.5; a.arms[1].rotation.x = sw * 0.5;
    a.body.position.y = Math.abs(sw) * 0.04;
  }

  dispose() {
    this.running = false;
    cancelAnimationFrame(this.frame);
    this.ro.disconnect();
    this.agents.forEach(a => clearTimeout(a.bubbleTimer));
    this.scene.traverse(o => { o.geometry?.dispose(); const m = o.material; (Array.isArray(m) ? m : m ? [m] : []).forEach(x => { x.map?.dispose(); x.dispose(); }); });
    this.renderer.dispose();
    this.el.replaceChildren();
  }
}

function wrap(g, text, x, y, maxW, lh, maxLines) {
  const words = String(text || "").split(" ");
  let line = "", n = 0;
  for (const w of words) {
    const test = line ? line + " " + w : w;
    if (g.measureText(test).width > maxW && line) {
      if (++n === maxLines) { g.fillText(line + "…", x, y); return; }
      g.fillText(line, x, y); line = w; y += lh;
    } else line = test;
  }
  if (line) g.fillText(line, x, y);
}
