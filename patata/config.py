"""Parametros globales del proyecto.

Todo lo que un dia habra que discutir con el cliente vive aqui, no
esparcido por el codigo.
"""

from __future__ import annotations

from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DIR_DATOS = RAIZ / "data"
RUTA_DB = DIR_DATOS / "patata.db"

# --- Horizonte de decision -------------------------------------------------
# El envasador decide con ~1 mes de margen: compro ahora o espero.
HORIZONTE_SEMANAS = 4

# Coste de almacenaje por kg durante el horizonte. Si el precio dentro de
# 4 semanas no supera el de hoy MAS este coste, esperar no compensa.
COSTE_ALMACENAJE_EUR_KG = 0.01

# --- Backtest --------------------------------------------------------------
# Primer lunes que se predice de verdad. Todo lo anterior es solo entrenamiento.
INICIO_BACKTEST = "2015-01-05"  # lunes
# Minimo de ejemplos etiquetados antes de aceptar un origen como evaluable.
MIN_EJEMPLOS_ENTRENO = 104  # ~2 anos

# --- Features --------------------------------------------------------------
# Retardos (en semanas, contados sobre la serie observada) que entran como
# features. Se calculan siempre sobre el snapshot point-in-time.
# El 48 y el 52 van juntos a proposito: su diferencia es la variacion a 4
# semanas observada en el mismo tramo del calendario el anyo pasado, que es lo
# que usa el baseline estacional.
LAGS = (0, 1, 2, 3, 4, 8, 12, 26, 48, 52)
VENTANAS_MEDIA = (4, 8, 13, 26, 52)

# --- Loader falso ----------------------------------------------------------
SEMANAS_SINTETICAS = 1100
PRIMER_LUNES_SINTETICO = "2004-01-05"
SEMILLA = 20240105

# --- Modelos (fase 1) ------------------------------------------------------
# Hiperparametros FIJOS, deliberadamente conservadores. Con ~500 puntos de
# evaluacion, ajustarlos mirando el backtest es sobreajustar el backtest: la
# mejora que saldria no existiria en produccion. Si algun dia se tunean, tiene
# que ser con validacion interna dentro de cada ventana de entrenamiento.
PARAMS_LGBM = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 7,
    "min_child_samples": 25,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_lambda": 1.0,
    "verbosity": -1,
    "n_jobs": 1,
}
ALPHA_RIDGE = 10.0
C_LOGISTICA = 0.1
