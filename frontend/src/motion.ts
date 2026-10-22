export type Landmark = { x: number; y: number; z: number; visibility: number; presence: number };
export type PoseFrame = { timestamp_ms: number; frame_index: number; landmarks: (Landmark | null)[]; world_landmarks: (Landmark | null)[] };
export type Motion = {
  schema_version: string; input_type: 'image' | 'video'; filename: string;
  duration_ms: number; width: number; height: number; detected_frames: number;
  landmark_names: string[]; frames: PoseFrame[];
};
export const QUALITY_THRESHOLD = 0.5; // Experimental display gate, not calibrated accuracy.
export function reliable(p: Landmark | null | undefined): p is Landmark {
  return !!p && Math.min(p.visibility, p.presence) >= QUALITY_THRESHOLD;
}
export const CONNECTIONS: [string, string][] = [
  ['left_shoulder','right_shoulder'], ['left_shoulder','left_elbow'], ['left_elbow','left_wrist'],
  ['right_shoulder','right_elbow'], ['right_elbow','right_wrist'], ['left_shoulder','left_hip'],
  ['right_shoulder','right_hip'], ['left_hip','right_hip'], ['left_hip','left_knee'],
  ['left_knee','left_ankle'], ['right_hip','right_knee'], ['right_knee','right_ankle'],
  ['left_ankle','left_heel'], ['right_ankle','right_heel'],
  ['left_heel','left_foot_index'], ['right_heel','right_foot_index'],
];
export function frameAt(motion: Motion | null, timeMs: number): PoseFrame | undefined {
  if (!motion?.frames.length) return;
  let lo = 0, hi = motion.frames.length - 1;
  while (lo < hi) {
    const mid = Math.floor((lo + hi + 1) / 2);
    if (motion.frames[mid].timestamp_ms <= timeMs) lo = mid; else hi = mid - 1;
  }
  // No interpolation over missing detections.
  return motion.frames[lo];
}
