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
LAGS = (0, 1, 2, 3, 4, 8, 12, 26, 52)
VENTANAS_MEDIA = (4, 8, 13, 26, 52)

# --- Loader falso ----------------------------------------------------------
SEMANAS_SINTETICAS = 1100
PRIMER_LUNES_SINTETICO = "2004-01-05"
SEMILLA = 20240105
