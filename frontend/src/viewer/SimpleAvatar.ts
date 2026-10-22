import * as THREE from 'three';
import { CONNECTIONS, reliable, type Landmark, type PoseFrame } from '../motion';

export interface PoseAvatar { object: THREE.Group; setPose(frame: PoseFrame | undefined, names: string[]): void; dispose(): void; }

// Coordinates enter through this adapter, so a GLB rig can replace the geometry later.
export class SimpleAvatar implements PoseAvatar {
  object = new THREE.Group();
  private ball = new THREE.SphereGeometry(1, 16, 12);
  private rod = new THREE.CylinderGeometry(1, 1, 1, 14);
  private green = new THREE.MeshStandardMaterial({ color: '#a4f27d', roughness: .65 });
  private gray = new THREE.MeshStandardMaterial({ color: '#6b7770', roughness: .85 });
  private jointMaterial = new THREE.MeshStandardMaterial({ color: '#edf4e8', roughness: .5 });
  private limbs = CONNECTIONS.map(() => new THREE.Mesh(this.rod, this.green));
  private joints = Array.from({ length: 33 }, () => new THREE.Mesh(this.ball, this.jointMaterial));
  private torso = new THREE.Mesh(this.ball, this.green);
  private head = new THREE.Mesh(this.ball, this.green);
  private neck = new THREE.Mesh(this.rod, this.green);

  constructor() {
    this.object.add(...this.limbs, ...this.joints, this.torso, this.head, this.neck);
    this.object.visible = false;
  }

  setPose(frame: PoseFrame | undefined, names: string[]) {
    const points = frame?.world_landmarks ?? [];
    this.object.visible = points.some(Boolean);
    if (!this.object.visible) return;
    const ids = new Map(names.map((n, i) => [n, i]));
    const get = (name: string) => points[ids.get(name) ?? -1];
    // Hip-centered estimated meters -> Three.js. Do not mirror the person's left/right.
    const vector = (p: Landmark) => new THREE.Vector3(p.x, -p.y, -p.z);
    const feet = ['left_heel','right_heel','left_ankle','right_ankle','left_foot_index','right_foot_index'].map(get).filter(reliable);
    const ground = feet.length ? Math.min(...feet.map(p => -p.y)) - .04 : 0;
    const position = (p: Landmark) => vector(p).add(new THREE.Vector3(0, -ground, 0));
    const setBone = (mesh: THREE.Mesh, a: Landmark | null | undefined, b: Landmark | null | undefined, radius = .035) => {
      mesh.visible = !!a && !!b;
      if (!a || !b) return;
      const start = position(a), end = position(b), delta = end.clone().sub(start);
      mesh.position.copy(start.add(end).multiplyScalar(.5));
      mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), delta.clone().normalize());
      mesh.scale.set(radius, Math.max(.001, delta.length()), radius);
      mesh.material = reliable(a) && reliable(b) ? this.green : this.gray;
    };
    CONNECTIONS.forEach(([a,b], i) => setBone(this.limbs[i], get(a), get(b)));
    this.joints.forEach((mesh,i) => {
      const p = points[i]; mesh.visible = !!p && i >= 11;
      if (!p) return;
      mesh.position.copy(position(p)); mesh.scale.setScalar(.045);
      mesh.material = reliable(p) ? this.jointMaterial : this.gray;
    });
    const midpoint = (a: Landmark | null | undefined, b: Landmark | null | undefined): Landmark | null =>
      a && b ? { x:(a.x+b.x)/2, y:(a.y+b.y)/2, z:(a.z+b.z)/2, visibility:Math.min(a.visibility,b.visibility), presence:Math.min(a.presence,b.presence) } : null;
    const shoulders = midpoint(get('left_shoulder'),get('right_shoulder'));
    const hips = midpoint(get('left_hip'),get('right_hip'));
    const ears = midpoint(get('left_ear'),get('right_ear'));
    setBone(this.torso, hips, shoulders);
    if (hips && shoulders) {
      const shoulderL = get('left_shoulder'), shoulderR = get('right_shoulder');
      const width = shoulderL && shoulderR ? vector(shoulderL).distanceTo(vector(shoulderR)) : .25;
      this.torso.scale.x = Math.max(.08, width * .43);
      this.torso.scale.y *= .57; this.torso.scale.z = .11;
    }
    this.head.visible = !!ears;
    if (ears) { this.head.position.copy(position(ears)); this.head.scale.set(.10,.125,.10); this.head.material = reliable(ears) ? this.green : this.gray; }
    setBone(this.neck, shoulders, ears, .04);
  }

  dispose() { this.ball.dispose(); this.rod.dispose(); this.green.dispose(); this.gray.dispose(); this.jointMaterial.dispose(); }
}
