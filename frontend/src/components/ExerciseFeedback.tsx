import type { Measurement, Motion, PoseFrame } from '../motion';

function angle(value: Measurement | undefined) {
  return value && Number.isFinite(value.value) ? `${value.value.toFixed(0)}°` : 'No evaluable';
}

export default function ExerciseFeedback({ frame, motion, time, outside }: { frame?: PoseFrame; motion: Motion; time: number; outside: boolean }) {
  const exercise = motion.exercise_summary ?? frame?.exercise;
  const identified = exercise?.status === 'identified';
  const unavailable = exercise?.status === 'model_missing' || exercise?.status === 'model_unavailable' || motion.exercise_classifier?.available === false;
  const exerciseName = outside ? 'Fuera del tramo analizado' : unavailable ? 'Reconocedor no disponible' : identified ? exercise.name : 'Ejercicio no detectado';
  const comparison = outside ? undefined : frame?.comparison;
  const alerts = comparison?.alerts ?? [];
  const different = comparison?.status === 'differences' && alerts.length > 0;
  const compared = comparison?.status === 'within_display_tolerance';
  const comparisonName = outside ? 'Fuera del tramo analizado' : different ? `${alerts.length} zona${alerts.length===1?'':'s'} con diferencia` : compared ? 'Sin diferencias destacadas' : 'Comparación no disponible';
  const reasons: Record<string,string> = {
    disabled:'La comparación con el ejemplo está desactivada.',video_required:'Selecciona un video para comparar la misma fase del movimiento.',
    exercise_uncertain:identified?'La comparación actual solo está preparada para sentadillas.':'No se compara la técnica porque no se ha identificado el ejercicio.',
    reference_missing:'Falta preparar el video de ejemplo. Consulta las instrucciones locales.',
    reference_phase_unavailable:'El ejemplo no ofrece un ciclo completo con una vista lateral suficientemente clara.',
    phase_or_view_unavailable:'No hay una repetición completa, articulaciones suficientes o una vista lateral clara para alinear este instante.',
    phase_match_unavailable:'No se encontró la misma fase en el ejemplo. Este instante no se compara.',
  };
  const explanation = outside ? 'Vuelve al tramo analizado.' : different ? 'Rojo significa diferencia respecto al ejemplo. Revisa la fase y la toma antes de interpretarlo como algo que corregir.' : compared ? 'No hay diferencias persistentes por encima de la tolerancia visual. Esto no certifica una ejecución correcta.' : reasons[comparison?.reason??'']??'Analiza un video para ver la comparación experimental.';
  const values = outside ? undefined : frame?.measurements;
  const reference = motion.feedback?.reference.reference_video.split(/[\\/]/).pop();

  return <section className="exercise-feedback" aria-label="Análisis del ejercicio durante la reproducción">
    <div className="feedback-heading"><span className="eyebrow">ANÁLISIS DEL EJERCICIO</span><span>{motion.input_type === 'video' ? `${time.toFixed(1)} s · sincronizado con el video` : 'Imagen estática'}</span></div>
    <div className="feedback-cards">
      <div className="feedback-card"><small>EJERCICIO ESTIMADO</small><strong aria-live="polite">{exerciseName}</strong><p>{unavailable ? 'Las articulaciones siguen disponibles. Revisa la preparación del reconocedor.' : identified ? 'Identificación experimental del tramo; se supone un solo ejercicio. Puede confundirse con otros similares.' : 'No se pudo identificar el ejercicio con suficiente confianza. Puede no estar entre los ejercicios del modelo o la toma puede no aportar información suficiente.'}</p></div>
      <div className={`feedback-card ${different ? 'differences' : ''}`}><small>DIFERENCIAS CON EL EJEMPLO</small><strong aria-live="polite">{comparisonName}</strong><p>{explanation}</p></div>
    </div>
    {different&&<ul className="comparison-alerts" aria-label="Zonas con diferencias respecto al ejemplo">{alerts.map(alert=><li key={alert.metric}>{alert.message}</li>)}</ul>}
    <div className="feedback-measurements">
      <div><span>Flexión de rodilla izquierda</span><b>{angle(values?.left_knee_flexion_deg)}</b></div>
      <div><span>Flexión de rodilla derecha</span><b>{angle(values?.right_knee_flexion_deg)}</b></div>
      <div><span>Inclinación del tronco</span><b>{angle(values?.trunk_inclination_deg)}</b></div>
    </div>
    <p className="feedback-note">Ángulos aproximados proyectados en la imagen; requieren una toma lateral. Rodilla: 0° corresponde aproximadamente a extensión. Tronco: respecto a la vertical de la imagen.</p>
    {reference && identified && exercise.label==='Squat' && !outside && <p className="feedback-note">Ejemplo: <b>{reference}</b> · {motion.reference_comparison?.reference_reviewed?'revisión registrada':'sin revisión de técnica'}. Tolerancia visual usada: {motion.reference_comparison?.visual_tolerance_deg??15}°. El resaltado no sustituye criterios revisados ni determina un ejercicio «perfecto».</p>}
    {motion.input_type === 'video' && <p className="feedback-note">El análisis se prepara antes de reproducir. Los resultados se muestran según el instante del video.</p>}
  </section>;
}
