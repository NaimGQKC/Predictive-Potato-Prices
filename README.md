# Predictive-Potato-Prices — Fase 0: esqueleto de evaluación

Sistema para predecir el precio de la patata en España a 4 semanas vista.
Cliente: envasador que compra ~40.000 t/año en mercado libre.

**Esta fase no contiene ningún modelo.** Contiene la infraestructura que
decidirá si un modelo futuro merece la pena. La pregunta que responde es una
sola:

> ¿es batible la regla tonta "el precio dentro de 4 semanas = el de hoy"?

Y, sobre todo, deja montado el aparato para que esa respuesta no sea mentira.

---

## La regla innegociable: doble fecha

Cada dato se guarda con dos fechas:

| campo | significado |
|---|---|
| `fecha_dato` | a qué semana se refiere el valor (lunes de esa semana) |
| `fecha_publicacion` | cuándo ese valor estuvo disponible públicamente |

Para predecir la semana **T** solo se pueden usar filas con
`fecha_publicacion <= lunes de la semana T`.

Esto tiene dos consecuencias que es fácil pasar por alto:

1. **El lunes de la semana T no conoces el precio de la semana T.** La lonja
   publica el jueves. Tu dato más fresco es el de la semana anterior, y cuando
   hay retraso, el de hace dos.
2. **Las etiquetas también tienen fecha de publicación.** Para entrenar un
   modelo que predice a 4 semanas, el ejemplo más reciente que puedes usar es
   de hace ~5 semanas, no de hace 4: el precio objetivo del ejemplo de hace 4
   semanas todavía no se ha publicado. Ignorar esto infla los resultados de
   forma silenciosa.

Ambas están cubiertas por tests. `tests/test_doble_fecha.py` es el fichero más
importante del repositorio. Incluye el test adversario: se inserta una revisión
absurda (99 €/kg) publicada con dos años de retraso y se comprueba que **no
altera ni una sola fila** de features anterior a su fecha de publicación, y que
sí altera las posteriores (si no, el test no probaría nada).

Si alguno de esos tests falla, los resultados del backtest no se publican.

---

## Estructura

```
patata/
  config.py          parámetros: horizonte, coste de almacenaje, inicio del backtest
  schema.sql         esquema DuckDB (raw por fuente + features + resultados)
  db.py              conexión al fichero .db único
  loaders/falso.py   1.100 semanas sintéticas con estacionalidad, ciclo y ruido
  features.py        constructor de features con la regla de doble fecha
  predictores.py     baseline naive (y el contrato para modelos futuros)
  backtest.py        motor walk-forward + auditoría anti-fuga
  metricas.py        MAPE, RMSE, MAE, dirección, clasificación, skill score
  evaluacion.py      de predicciones crudas a tabla de resultados
  pipeline.py        orquestación
scripts/run_eval.py  ejecuta todo y saca la tabla
tests/               52 tests
```

## Cómo se ejecuta

```bash
pip install -r requirements.txt
python scripts/run_eval.py --csv data/salidas
python -m pytest tests/ -q
```

Genera `data/patata.db` (un solo fichero DuckDB) con las tablas `raw_lonja`,
`features_semanal`, `backtest_predicciones` y `backtest_metricas`.

---

## Qué hace el motor de backtest

Ventana expansiva, un origen por semana, empezando en enero de 2015:

1. corta el entrenamiento a lo que era conocido **ese lunes** (features y
   etiquetas, con sus dos filtros distintos)
2. reentrena
3. predice la semana `lunes + 4`
4. avanza una semana

En cada paso vuelve a comprobar las invariantes point-in-time y aborta con
`FugaTemporalError` si algo no cuadra. Además de comprobar que se aplicó el
filtro, comprueba que las fechas son **coherentes**: una etiqueta publicada
antes de la semana que describe es imposible, y es justo lo que se vería si
alguien "arreglase" el calendario para conseguir más ejemplos de entrenamiento.

## Los dos objetivos, ambos evaluados

- **(a) regresión**: el precio de lonja dentro de 4 semanas.
- **(b) clasificación**: si ese precio superará el de hoy más 0,01 €/kg de
  coste de almacenaje. Es la decisión real (`etiqueta_comprar_ahora = 1`
  significa *compra ahora, esperar sale caro*).

En (b) la referencia "el precio de hoy" es `precio_ref`, el último precio
**publicado** en el momento de decidir — no el precio verdadero de la semana en
curso, que todavía no conoces. Es la única definición que se puede calcular en
tiempo real.

---

## Resultado: el listón a batir

Sobre **datos sintéticos** (526 orígenes semanales, 2015–2025, 522 puntuables):

| métrica | naive |
|---|---|
| MAPE | 15,52 % |
| RMSE | 0,0409 €/kg |
| MAE | 0,0326 €/kg |
| accuracy de la decisión (b) | 59,6 % |
| recall de la decisión (b) | 0,0 % |
| tasa base ("comprar ahora") | 40,4 % |

Tres lecturas que condicionan la fase siguiente:

**La naive no opina sobre la dirección.** Predice exactamente el precio de
referencia, así que su variación prevista es 0 y nunca dice ni "sube" ni "baja".
Reportar un 0 % de acierto de dirección sería mentir sobre lo que hace; el
código devuelve `NaN` y reporta aparte la cobertura (0 %) y el listón real:
apostar siempre a la dirección más frecuente, **50,2 %**.

**En la decisión (b) la naive es estructuralmente inútil.** Como su variación
prevista es 0 y 0 < 0,01 €/kg, *siempre* dice "no compres ahora, espera". Su
59,6 % de accuracy es puro artefacto de la clase mayoritaria y su recall es
cero: no detecta ni una sola de las subidas que justificarían comprar. Por eso
el listón de (b) no es la naive, es la clase mayoritaria (59,6 %), y la métrica
que hay que mirar es la **accuracy balanceada por encima de 0,5**.

**El MAPE varía entre 10,9 % y 21,3 % según el año.** Cualquier mejora que no
aguante año a año es ruido. Por eso `tabla_por_anyo` está en la salida estándar
y no como extra.

> Estos números son de la serie sintética y sirven para validar el esqueleto.
> Con datos reales cambiarán. Lo que no cambia es el procedimiento.

## ¿Es batible entonces?

El esqueleto no lo decide: lo mide. Lo que sí deja fijado es qué contaría como
respuesta afirmativa, y son cuatro condiciones a la vez:

1. MAPE y RMSE por debajo de los de la naive (`skill_mape`, `skill_rmse` > 0),
2. accuracy balanceada de (b) por encima de 0,5, con recall no trivial,
3. estable año a año, no ganando en dos años y perdiendo en ocho,
4. con la diferencia fuera del ruido (`dm_t`, test pareado sobre los errores al
   cuadrado, con corrección de Newey-West por el solape de horizontes).

Sobre la serie sintética hay margen para (1) y (2): la variación a 4 semanas
tiene componente estacional y reversión a la media, que la naive no captura.
Sobre datos reales está por ver, y ese es exactamente el punto de haber montado
esto antes que el modelo.

---

## Decisiones tomadas (y dónde revisarlas)

- **Etiqueta de objetivo = primera publicación**, no la última revisión. Es el
  valor que habrías tenido entrenando en tiempo real. Si algún día la lonja
  revisa precios de forma sistemática, hay que revisitarlo (`features.py`,
  `_tabla_objetivos`).
- **Los lags se alinean por calendario**, no por posición: si falta una semana
  el lag es `NaN`, no se cuela el vecino.
- **1.100 filas es poco.** El backtest sacrifica los primeros 11 años como
  entrenamiento mínimo y deja 522 puntos evaluables. Con esa n, diferencias de
  MAPE menores de ~1 punto no son distinguibles del ruido; de ahí el test
  pareado.
- **El horizonte y el coste de almacenaje son parámetros**, no constantes
  escritas por ahí (`config.py`). El coste de 0,01 €/kg conviene contrastarlo
  con el cliente: mueve la tasa base de la clase positiva y por tanto todo (b).

## Qué NO hay aquí, a propósito

- Ningún modelo (ni LightGBM ni nada). Siguiente fase.
- Ninguna fuente de datos real. Solo el loader falso.
- Ninguna interfaz gráfica.

## Cómo se enchufa un modelo en la fase siguiente

Sin tocar el motor. Basta con un objeto que cumpla el contrato de
`patata/predictores.py`:

```python
class MiModelo:
    nombre = "lgbm"
    def entrenar(self, train): ...        # solo recibe filas ya publicadas
    def predecir(self, X): ...            # -> precio_pred [, prob_comprar_ahora]
```

y añadirlo a la lista en `pipeline.ejecutar`. El filtrado point-in-time, la
alineación y las métricas ya están hechos. Un modelo que devuelva
`prob_comprar_ahora` decide (b) con esa probabilidad; si no la devuelve, la
decisión se deriva de `precio_pred` contra `precio_ref + coste`.
