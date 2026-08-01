# Predictive-Potato-Prices

Sistema para predecir el precio de la patata en España a 4 semanas vista.
Cliente: envasador que compra ~40.000 t/año en mercado libre.

Dos fases entregadas:

- **Fase 0 — esqueleto de evaluación.** Almacén point-in-time, features con
  doble fecha, backtest walk-forward, baseline naive y métricas. Sin modelos.
- **Fase 1 — retadores.** Baseline estacional, lineal regularizado y LightGBM,
  todos por el mismo motor. Responde a la pregunta original:

> ¿es batible la regla tonta "el precio dentro de 4 semanas = el de hoy"?

**Sobre la serie sintética, sí, y por mucho.** Con el detalle importante de que
eso no dice nada todavía sobre el mercado real — ver *Qué significa y qué no*
más abajo.

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
  predictores.py     naive, naive estacional, lineal regularizado, LightGBM
  backtest.py        motor walk-forward + auditoría anti-fuga
  metricas.py        MAPE, RMSE, MAE, dirección, clasificación, skill score
  evaluacion.py      de predicciones crudas a tabla de resultados
  economia.py        de decisiones a euros por kg
  placebo.py         control negativo: destruir la señal y comprobar que se nota
  pipeline.py        orquestación
scripts/run_eval.py  ejecuta todo y saca la tabla
tests/               67 tests
```

## Cómo se ejecuta

```bash
pip install -r requirements.txt
python scripts/run_eval.py --csv data/salidas   # ~3,5 min
python scripts/run_eval.py --placebo            # control negativo
python -m pytest tests/ -q                      # ~2 min
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

## Resultados

522 orígenes semanales evaluados (2015–2025), ventana expansiva, reentrenando
cada semana. Todo sobre la serie sintética.

### (a) Regresión — precio a 4 semanas

| predictor | MAPE | RMSE €/kg | MAE €/kg | acierto dirección | skill MAPE | dm_t |
|---|---|---|---|---|---|---|
| **lineal_regularizado** | **10,12 %** | **0,0294** | **0,0219** | **75,7 %** | **+0,348** | **−5,69** |
| lgbm | 11,17 % | 0,0313 | 0,0239 | 72,8 % | +0,280 | −4,60 |
| naive_estacional | 15,16 % | 0,0450 | 0,0330 | 66,7 % | +0,023 | +1,67 |
| naive_ultimo_precio | 15,52 % | 0,0409 | 0,0326 | — (no opina) | 0 | — |

El listón de dirección es 50,2 % (apostar siempre a la más frecuente).

### (b) Decisión — comprar ahora vs esperar

| predictor | accuracy | precisión | recall | acc. balanceada |
|---|---|---|---|---|
| lgbm | **74,7 %** | **69,3 %** | 67,3 % | **73,5 %** |
| lineal_regularizado | 71,5 % | 62,0 % | **75,8 %** | 72,2 % |
| naive_estacional | 64,0 % | 55,7 % | 53,1 % | 62,2 % |
| naive_ultimo_precio | 59,6 % | — | 0,0 % | 50,0 % |

Clase mayoritaria 59,6 %, tasa base 40,4 %.

### En euros

Coste medio de compra, decisiones semanales, 40.000 t/año:

| estrategia | coste €/kg | ahorro €/kg | ahorro anual | % del máximo |
|---|---|---|---|---|
| oráculo perfecto | 0,2164 | 0,0113 | 453.226 € | 100 % |
| **lgbm** | **0,2218** | **0,0060** | **238.452 €** | **52,6 %** |
| lineal_regularizado | 0,2227 | 0,0051 | 202.421 € | 44,7 % |
| naive_estacional | 0,2271 | 0,0006 | 25.992 € | 5,7 % |
| naive / siempre esperar | 0,2277 | 0 | 0 € | 0 % |
| siempre comprar ya | 0,2386 | −0,0108 | −433.931 € | −95,7 % |

---

## Qué significa y qué no

**Sí, la naive es batible — sobre datos sintéticos.** La diferencia es grande
(35 % menos de MAPE) y estadísticamente sólida (`dm_t` = −5,7, muy fuera del
ruido). Pero la serie la generé yo con estacionalidad y reversión a la media
explícitas. Que un modelo las encuentre demuestra que **el aparato detecta señal
cuando la hay**; no demuestra que haya señal en el mercado real. Ese número solo
sale con datos de lonja.

Lo que sí se traslada al mundo real son estas cuatro lecturas:

**El modelo lineal gana al LightGBM en regresión.** Con ~600 filas de
entrenamiento y ~40 columnas, el gradient boosting tiene de sobra para memorizar
y no le llega para generalizar mejor que un ridge. Es un argumento fuerte para
**no** empezar por LightGBM cuando lleguen los datos reales: el orden correcto
es naive → estacional → lineal → boosting, y parar en cuanto deje de mejorar.

**El mejor modelo de regresión no es el que más dinero ahorra.** LightGBM pierde
en MAPE y gana en euros (238k € vs 202k €), porque el dinero depende de la
*precisión* de la decisión binaria (69,3 % vs 62,0 %), no del error de
predicción. Equivocarse recomendando comprar cuesta caro; equivocarse por 2
céntimos en un precio que no se usa para decidir, no cuesta nada. **La métrica
que manda es la de la decisión, no la del precio.**

**El baseline estacional casi no aporta.** Mejora el MAPE un 2 % y *empeora* el
RMSE un 10 %, con `dm_t` = +1,67: dentro del ruido. Copiar la variación del año
pasado no basta. Está en la tabla precisamente porque un retador que no gana es
información útil.

**El techo es finito.** Ni con predicción perfecta se ahorra más de 0,0113 €/kg
(453k €/año sobre 40.000 t). Eso acota cuánto tiene sentido invertir en el
sistema, y hay que decirlo antes de prometer nada.

## El control negativo (lo más importante de esta fase)

Un backtest con buenos números puede estar midiendo dos cosas muy distintas:
que el modelo predice, o que se coló información del futuro. Los tests de doble
fecha atacan el problema por construcción; el placebo lo ataca por el resultado.

`python scripts/run_eval.py --placebo` baraja la variación objetivo entre
orígenes dejando las features intactas. Después de barajar no queda nada que
predecir, así que **todos los skill deben caer a ≤ 0 y las accuracy balanceadas
a ~0,5**. Es exactamente lo que pasa, sobre los mismos 522 orígenes:

| predictor | skill MAPE | skill RMSE | Δ acc. balanceada | dm_t |
|---|---|---|---|---|
| lineal_regularizado | −0,018 | −0,017 | +0,014 | +2,21 |
| lgbm | −0,091 | −0,067 | −0,004 | +4,69 |
| naive_estacional | −0,334 | −0,357 | +0,006 | +7,10 |

Compárese con el +0,348 de skill y el −5,69 de `dm_t` que saca el lineal con la
señal intacta. El signo de `dm_t` se invierte: sin nada que predecir, los
modelos son *peores* que la naive, que es justo lo que debe pasar. Si algún día
un modelo gana en modo placebo, hay fuga.

Merece la pena contar cómo salió, porque es instructivo: la primera versión del
placebo **no colapsó** — los modelos seguían sacando skill 0,29 y accuracy
balanceada 0,81. No era fuga: era el placebo mal diseñado. Barajaba el *nivel*
del precio objetivo, y eso crea una señal nueva y perfectamente aprendible, ya
que `variación = precio_al_azar − precio_ref` se predice muy bien conociendo
`precio_ref`. Hay que barajar **la variación**, que es lo que los modelos
predicen. Está documentado en `patata/placebo.py` para que nadie lo revierta.

---

## Decisiones tomadas (y dónde revisarlas)

- **Los modelos predicen la variación, no el nivel** (`predictores.py`). El
  precio de la patata no es estacionario: un modelo entrenado sobre el nivel
  aprende la media histórica y la arrastra. Prediciendo la variación tiene que
  ganarse cada euro contra la naive.
- **Imputación y estandarizado se ajustan dentro de cada ventana**, nunca sobre
  la tabla entera. Ajustarlos globalmente mete medias del futuro en el pasado;
  es una fuga sutil y muy común.
- **Hiperparámetros fijos, sin tunear** (`config.PARAMS_LGBM`). Con 522 puntos
  de evaluación, ajustarlos mirando el backtest es sobreajustar el backtest. Si
  algún día se tunean, tiene que ser con validación interna dentro de cada
  ventana de entrenamiento.
- **Umbral de decisión en 0,5**, sin optimizar. Mover el umbral mirando el
  resultado sería la misma trampa. Ver abiertos.
- **Etiqueta de objetivo = primera publicación**, no la última revisión. Es el
  valor que habrías tenido entrenando en tiempo real. Si la lonja revisa precios
  de forma sistemática, hay que revisitarlo (`features.py`, `_tabla_objetivos`).
- **Los lags se alinean por calendario**, no por posición: si falta una semana
  el lag es `NaN`, no se cuela el vecino.
- **1.100 filas es poco.** Diferencias de MAPE menores de ~1 punto no son
  distinguibles del ruido con esta n; de ahí el test pareado.
- **El horizonte y el coste de almacenaje son parámetros** (`config.py`), no
  constantes escritas por ahí.

## Abiertos

Por orden de impacto:

1. **Datos reales.** Todo lo anterior es sobre una serie que me inventé. Hace
   falta decidir la fuente (MAPA, Mercabarna, lonja de Salamanca…) y, sobre
   todo, si su histórico conserva la **fecha de publicación** real. Si solo hay
   fecha de dato, la regla de doble fecha se sostiene con un supuesto sobre el
   retardo, y eso hay que declararlo.
2. **Validar el coste de almacenaje con el cliente.** 0,01 €/kg es una
   suposición mía. Mueve la tasa base de la clase positiva y con ella toda la
   evaluación de (b) y la cifra en euros.
3. **Umbral de decisión.** El 0,5 no tiene por qué ser óptimo: comprar de más y
   comprar de menos no cuestan lo mismo. Se puede optimizar **dentro de cada
   ventana de entrenamiento** (nunca sobre el backtest) y evaluar si aporta.
4. **El modelo económico es de juguete.** Decisiones semanales independientes,
   sin inventario, sin capacidad de almacén, sin contratos, y sin considerar que
   un comprador de 40.000 t/año mueve el mercado. Sirve como orden de magnitud;
   para prometer ahorro hay que modelar las restricciones reales.
5. **Sin intervalos de confianza en la predicción.** El envasador se beneficiaría
   de saber cuándo el modelo no tiene ni idea, para no actuar esas semanas.
6. **Reentreno semanal completo.** Con datos reales puede no ser necesario;
   `reentrenar_cada` ya está parametrizado, falta medir si degrada.

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
