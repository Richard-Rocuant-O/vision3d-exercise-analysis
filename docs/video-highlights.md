# Resaltado sobre el video

El visor actual muestra los puntos y segmentos directamente sobre el video, sin muñeco 3D. Rojo significa **diferencia persistente con el ejemplo de sentadilla**, no un error de técnica confirmado. Verde indica detección suficiente, no ejecución correcta. Gris indica detección poco confiable.

## Probar localmente

1. Inicia `start-local.ps1` desde la raíz y abre http://127.0.0.1:5173.
2. Elige un video de sentadilla lateral, con una persona y el cuerpo entero visible.
3. Selecciona un tramo con al menos una repetición completa: posición alta, bajada, posición baja, subida y posición alta. Activa **Comparar con el ejemplo de sentadilla** y pulsa **Analizar**.
4. Durante la reproducción, las rodillas y el segmento del tronco con diferencias evaluables se pintan en rojo. Una etiqueta junto a la articulación indica la diferencia angular. Pausa o mueve la línea de tiempo para inspeccionarla.

No se dibuja una posición objetivo inventada. Las etiquetas señalan las coordenadas detectadas en el instante actual. Un video de fondos, zancadas u otro ejercicio no recibe reglas de sentadilla.

## Preparar el ejemplo en otro equipo

Con el entorno Python y el modelo MediaPipe preparados, desde `backend`:

```powershell
.venv/Scripts/python.exe scripts/prepare_reference.py
```

Se lee `backend/config/squat-reference.json` y se procesa el video local `Ejercicios/squat/squat_01.mp4`. El resultado se guarda en `data/references/squat_01.json`, excluido de Git. La revisión de técnica sigue pendiente: preparar los puntos no valida el movimiento.

## Cómo se decide el rojo

Se comparan la flexión de rodillas y la inclinación del tronco, proyectadas en la imagen. Se busca la misma fase y un progreso relativo similar dentro de ciclos completos, de modo que los videos pueden durar distinto. Se rechazan detecciones insuficientes, saltos temporales y proyecciones claramente frontales; esa comprobación geométrica no verifica por sí sola el ángulo de cámara.

La tolerancia inicial de 15° es un ajuste visual experimental, modificable entre 5° y 45°. **No es un límite de técnica aceptable.** La diferencia debe persistir al menos 240 ms antes de resaltarse; una detección perdida interrumpe el aviso. Si falta la referencia, la fase, la confianza o la identificación de sentadilla, se muestra «no evaluable» y se evita el rojo.

Las proporciones corporales, la cámara, ropa, luz y oclusiones pueden producir diferencias sin que exista un error. La ausencia de rojo no certifica una ejecución perfecta. Para dar indicaciones de corrección faltan criterios de técnica revisados por ejercicio y fase, además de validación con personas y tomas distintas.

## Verificación

```powershell
# Desde backend, con dependencias preparadas:
.venv/Scripts/python.exe scripts/test_reference_compare.py
# Con la aplicación iniciada:
.venv/Scripts/python.exe scripts/check_feedback.py ../Ejercicios/squat/squat_01.mp4 --start 0 --duration 15

# Desde frontend:
node scripts/check-overlay.mjs
node scripts/check-feedback-ui.mjs
npm run build
```

Las pruebas sintéticas comprueban fases, persistencia, supresión por baja confianza y posiciones de etiquetas. No miden precisión del sistema. La prueba API procesa un video local real; el panel se comprueba con React sin captura de navegador.
