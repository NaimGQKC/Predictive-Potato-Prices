# Deja aquí el fichero de precios reales

Ahora mismo el proyecto **no usa ningún dato real**. Funciona con una serie
inventada (`patata/loaders/falso.py`). Para que los resultados signifiquen algo
hay que sustituirla por el histórico de lonja de verdad.

## Qué hay que dejar aquí

Un fichero con el precio semanal de lonja, ~1.100 filas desde 2004.
CSV o Excel, da igual. El nombre da igual.

**No hace falta que tenga este formato exacto.** Déjalo tal cual lo tengas
—aunque esté sucio, con celdas combinadas o columnas raras— y yo escribo el
loader que lo lea. Lo importante es que no falte información, no que esté
ordenada.

`PLANTILLA.csv` es solo un ejemplo de a dónde queremos llegar.

## Lo mínimo imprescindible

| columna | qué es | ¿imprescindible? |
|---|---|---|
| `fecha_dato` | a qué semana se refiere el precio | **Sí** |
| `precio_eur_kg` | el precio | **Sí** |
| `fecha_publicacion` | qué día se publicó ese precio | **Sí, y es el problema** |
| `variedad` | agria, monalisa… | No, pero ayuda |

## La columna que suele faltar

`fecha_publicacion` es cuándo ese precio se hizo público. Casi ningún histórico
la trae, porque a nadie le importa... salvo a nosotros.

Nos importa por esto: si la lonja publica el precio de una semana el jueves de
esa semana, entonces el lunes **todavía no lo sabes**. Un modelo que use ese
precio para decidir el lunes está usando información que no tenías. Funciona
de maravilla en el ordenador y pierde dinero en la vida real.

Todo el proyecto está construido alrededor de esa regla.

### Si tu fichero no la trae

No es un bloqueo. Se sustituye por un supuesto —"la lonja publica siempre N
días después"— y ese supuesto queda escrito en el código y en los resultados,
para que nadie lo olvide más adelante.

Pero necesito que alguien que conozca la lonja conteste a esto:

1. ¿Qué día de la semana publica el precio?
2. ¿Ese precio se refiere a esa misma semana o a la anterior?
3. ¿Corrige precios ya publicados alguna vez?

Con esas tres respuestas basta.
