export type Landmark = { x: number; y: number; z: number; visibility: number; presence: number };
export type ExerciseEstimate = {
  status: 'identified' | 'uncertain' | 'not_evaluable' | 'model_missing' | 'model_unavailable';
  label: string | null; name: string | null; candidate_name?: string | null;
};
export type Measurement = { value: number; visibility: number } | null;
export type DifferenceAlert = { metric: string; kind: 'reference_difference'; message: string; delta_deg: number; joints: string[]; segments: [string, string][] };
export type Comparison = {
  status: 'differences' | 'within_display_tolerance' | 'not_evaluable'; reason?: string;
  phase?: 'top' | 'bottom' | 'descending' | 'ascending'; alerts: DifferenceAlert[];
  reference_timestamp_ms?: number;
};
export type PoseFrame = {
  timestamp_ms: number; frame_index: number; landmarks: (Landmark | null)[]; world_landmarks: (Landmark | null)[];
  exercise?: ExerciseEstimate;
  measurements?: { left_knee_flexion_deg: Measurement; right_knee_flexion_deg: Measurement; trunk_inclination_deg: Measurement };
  technique?: { status: 'pending_review' | 'not_evaluable' | 'unsupported'; reason: string };
  comparison?: Comparison;
};
export type Motion = {
  schema_version: string; input_type: 'image' | 'video'; filename: string;
  duration_ms: number; width: number; height: number; detected_frames: number;
  landmark_names: string[]; frames: PoseFrame[];
  analysis_window?: { start_ms: number; end_ms: number; requested_start_ms: number; requested_duration_ms: number; source_duration_ms: number };
  reference_comparison?: { kind: string; reference_video: string; reference_reviewed: boolean; visual_tolerance_deg: number; red_means: string };
  exercise_classifier?: { available: boolean; name?: string; error?: string; sample_interval_ms?: number };
  exercise_summary?: ExerciseEstimate;
  feedback?: { technique_evaluation_available: boolean; reference: { reference_video: string; review: {status: string}; version: string } };
  dataset_source?: { label: string; source_row: string; image_hash: string | null; validation_fold: string | null; version: string };
};
export const QUALITY_THRESHOLD = 0.5; // Experimental display gate, not calibrated accuracy.
export function reliable(p: Landmark | null | undefined): p is Landmark {
  return !!p && [p.x,p.y,p.visibility,p.presence].every(Number.isFinite) && Math.min(p.visibility, p.presence) >= QUALITY_THRESHOLD;
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
  if (motion.analysis_window && (timeMs < motion.analysis_window.start_ms || timeMs > motion.analysis_window.end_ms)) return;
  let lo = 0, hi = motion.frames.length - 1;
  while (lo < hi) {
    const mid = Math.floor((lo + hi + 1) / 2);
    if (motion.frames[mid].timestamp_ms <= timeMs) lo = mid; else hi = mid - 1;
  }
  // No interpolation over missing detections.
  return motion.frames[lo];
}
