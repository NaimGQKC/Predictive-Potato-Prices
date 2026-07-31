"""Loader falso: 1.100 semanas de precio de patata sinteticas.

No pretende ser realista en el nivel de precio, sino tener la MISMA FORMA que
los datos reales para que el esqueleto se pueda validar:

  - frecuencia semanal, indexada al lunes
  - estacionalidad anual marcada (minimo en cosecha, maximo en primavera)
  - ciclo plurianual tipo telarana (sobreoferta -> hundimiento -> siembra baja)
  - ruido autocorrelado, no ruido blanco (si fuera blanco la naive seria
    imbatible por construccion y el backtest no diria nada)
  - algun pico esporadico de escasez
  - y sobre todo: RETARDO DE PUBLICACION, a veces con retraso extra

Ese ultimo punto es el que importa. Si la lonja publica el precio de la semana
T el jueves de la semana T, el lunes de la semana T todavia no lo conoces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from patata import config, db

FUENTE = "lonja_sintetica"
VARIEDAD = "agria_lavada"

# La lonja publica el jueves de la misma semana: 3 dias despues del lunes.
RETARDO_BASE_DIAS = 3
# Y un 8% de las semanas se retrasa una semana entera (festivos, revisiones).
PROB_RETRASO_EXTRA = 0.08
RETRASO_EXTRA_DIAS = 7


def generar_precios(
    n_semanas: int = config.SEMANAS_SINTETICAS,
    primer_lunes: str = config.PRIMER_LUNES_SINTETICO,
    semilla: int = config.SEMILLA,
) -> pd.DataFrame:
    """Devuelve un DataFrame con las dos fechas obligatorias y el precio."""
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range(start=pd.Timestamp(primer_lunes), periods=n_semanas, freq="W-MON")

    t = np.arange(n_semanas)
    semana_anyo = fechas.isocalendar().week.to_numpy().astype(float)
    fase = 2 * np.pi * semana_anyo / 52.0

    nivel = 0.19 + 0.00004 * t  # deriva suave, inflacion de fondo
    estacional = 0.055 * np.sin(fase - 1.1) + 0.018 * np.sin(2 * fase + 0.4)
    ciclo = 0.035 * np.sin(2 * np.pi * t / (3.2 * 52) + 0.7)  # telarana ~3,2 anos

    # ruido AR(1): la memoria corta es lo que hace competitiva a la naive
    ruido = np.zeros(n_semanas)
    phi = 0.86
    for i in range(1, n_semanas):
        ruido[i] = phi * ruido[i - 1] + rng.normal(0.0, 0.011)

    picos = np.zeros(n_semanas)
    for i in np.flatnonzero(rng.random(n_semanas) < 0.012):
        amp = rng.uniform(0.05, 0.16)
        largo = int(rng.integers(3, 10))
        decaimiento = amp * np.exp(-np.arange(largo) / 3.0)
        fin = min(n_semanas, i + largo)
        picos[i:fin] += decaimiento[: fin - i]

    precio = nivel + estacional + ciclo + ruido + picos
    precio = np.maximum(precio, 0.045)  # la patata no cotiza a cero
    precio = np.round(precio, 4)

    retraso_extra = np.where(rng.random(n_semanas) < PROB_RETRASO_EXTRA, RETRASO_EXTRA_DIAS, 0)
    fecha_publicacion = fechas + pd.to_timedelta(RETARDO_BASE_DIAS + retraso_extra, unit="D")

    return pd.DataFrame(
        {
            "fuente": FUENTE,
            "fecha_dato": fechas.date,
            "fecha_publicacion": fecha_publicacion.date,
            "variedad": VARIEDAD,
            "precio_eur_kg": precio,
            "unidad": "EUR/kg",
        }
    )


def cargar(con, **kwargs) -> int:
    """Escribe la serie sintetica en `raw_lonja`. Devuelve el numero de filas."""
    df = generar_precios(**kwargs)
    con.execute("DELETE FROM fuentes WHERE fuente = ?", [FUENTE])
    con.execute(
        "INSERT INTO fuentes VALUES (?, ?, ?, ?)",
        [FUENTE, "Serie semanal sintetica para validar el esqueleto", RETARDO_BASE_DIAS, True],
    )
    con.execute("DELETE FROM raw_lonja WHERE fuente = ?", [FUENTE])
    db.guardar_df(con, df, "raw_lonja", reemplazar=False)
    return len(df)
