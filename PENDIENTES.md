# Pendientes

- [ ] Evaluar compresión local de intensidades para sustituir el clip conservando el factor 2.
  - Trabajar con `raw = 2 * mixed_input * mask_source1` antes de recortar; no se recuperan diferencias ya eliminadas por el clip.
  - Comparar normalización independiente de zonas saturadas y no saturadas contra una transformación continua con transición suave. Normalizar por separado puede crear saltos en los bordes y alterar la escala relativa.
  - Estimar un rango objetivo con vecinos no saturados del mismo objeto: comparar media y desviación estándar con mediana y percentiles. Excluir fondo y otros objetos usando una máscara aproximada; considerar que la máscara del modelo puede equivocarse.
  - Comprimir las intensidades altas al intervalo local estimado, preservando su orden dentro de cada región. Una media sola no define un intervalo.
  - Para una compresión creciente y continua que deje intacto todo valor hasta un umbral t, el rango de salida de la rama alta debe comenzar en t. Si el rango local está debajo de t, ampliar la corrección a una banda de transición y evaluar inversión de contraste o halos.
  - Definir un respaldo cuando no haya vecinos válidos (ampliar vecindario o usar la reconstrucción del CVAE).
  - Concepto de referencia: mapeo tonal local (local tone mapping), relacionado con el control local de exposición en fotografía (dodging and burning). Referencia: https://www-old.cs.utah.edu/docs/techreports/2002/pdf/UUCS-02-001.pdf
  - Optimizar umbral, tamaño de vecindario e intensidad de compresión en validación; comparar precisión, SSIM, PSNR y error de consistencia de mezcla en evaluación reservada.
  - Estado: hipótesis pendiente; sin cambios al modelo ni resultados experimentales.

- [ ] Evaluar la corrección de estimación solo en zonas saturadas de `project/CROP/models/crop3.py`.
  - Detectar la zona saturada antes del `clip`: `saturated = raw > 1.0`, con `raw = 2 * mixed_input * mask_source1`.
  - Obtener `filtrado = tf.clip_by_value(raw, 0, 1)`.
  - En esa zona, guardar la ponderación `(1 - delta) * filtrado + delta * cvae_output`, usando `tf.where(saturated, corrected, filtrado)`.
  - Mantener lo filtrado fuera de la zona saturada. Usar `delta` entre 0 y 1 para una media ponderada.
  - Optimizar previamente `gamma` y `delta` por separado para la variante global y la localizada, con el mismo presupuesto de búsqueda.
  - Comparar ambas en mezclas reservadas para evaluación, con las mismas imágenes y semillas; registrar SSIM, PSNR y precisión (`acc_both`, `acc_at_least_one`).
  - Revisar el signo de la corrección en las variantes del notebook: `x - (x - cvae_output) * delta` combina con el CVAE; `x - (cvae_output - x) * delta` extrapola en sentido contrario.
  - Estado: propuesta pendiente de evaluación; el cambio localizado fue revertido y no está aplicado al modelo. No hay resultados que demuestren una mejora.
