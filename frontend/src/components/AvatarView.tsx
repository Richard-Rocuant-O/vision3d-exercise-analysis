import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SimpleAvatar } from '../viewer/SimpleAvatar';
import type { Motion, PoseFrame } from '../motion';

export default function AvatarView({ frame, motion }: { frame?: PoseFrame; motion: Motion | null }) {
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<{ avatar: SimpleAvatar; camera: THREE.PerspectiveCamera; controls: OrbitControls } | null>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    const container = host.current;
    if (!container) return;
    let renderer: THREE.WebGLRenderer;
    try { renderer = new THREE.WebGLRenderer({ antialias: true }); }
    catch { setError('El navegador no pudo abrir el visor 3D. Prueba con aceleración gráfica habilitada.'); return; }
    renderer.setPixelRatio(Math.min(devicePixelRatio,2));
    renderer.setClearColor('#131c17'); container.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, 1, .01, 100);
    camera.position.set(0, .95, 3.3);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0,.8,0); controls.enableDamping = true;
    controls.minDistance = .7; controls.maxDistance = 7;
    const avatar = new SimpleAvatar(); scene.add(avatar.object);
    scene.add(new THREE.HemisphereLight('#f3ffe9','#25382b',2.5));
    const light = new THREE.DirectionalLight('#ffffff',2); light.position.set(2,4,3); scene.add(light);
    const grid = new THREE.GridHelper(5, 25, '#3f5746', '#24372b'); grid.position.y = -.02; scene.add(grid);
    const resize = () => { const w = container.clientWidth, h = container.clientHeight; if(!w||!h)return; renderer.setSize(w,h); camera.aspect=w/h; camera.updateProjectionMatrix(); };
    const observer = new ResizeObserver(resize); observer.observe(container); resize();
    let id = 0;
    const loop = () => { controls.update(); renderer.render(scene,camera); id=requestAnimationFrame(loop); };
    view.current = {avatar,camera,controls}; loop();
    return () => {
      cancelAnimationFrame(id); observer.disconnect(); controls.dispose(); avatar.dispose();
      grid.geometry.dispose(); (grid.material as THREE.Material).dispose(); renderer.dispose();
      renderer.domElement.remove(); view.current = null;
    };
  }, []);
  useEffect(() => { view.current?.avatar.setPose(frame, motion?.landmark_names ?? []); }, [frame, motion]);
  const reset = (side: boolean) => {
    const v=view.current;if(!v)return;
    v.camera.position.set(side?3.3:0,.95,side?0:3.3);
    v.controls.target.set(0,.8,0);v.controls.update();
  };
  return <div className="avatar-stage">
    <div className="rig" ref={host} role="img" aria-label="Muñeco articulado de la postura estimada en 3D" />
    {!frame?.world_landmarks.some(Boolean) && <div className="avatar-empty">{motion?'No hay una pose 3D disponible en este instante.':'Carga un archivo y analiza su postura.'}</div>}
    {error && <div className="avatar-empty">{error}</div>}
    <div className="view-buttons"><button onClick={()=>reset(false)}>Frontal</button><button onClick={()=>reset(true)}>Lateral</button></div>
    <span className="orbit-hint">Arrastra para girar · rueda para acercar</span>
  </div>;
}
