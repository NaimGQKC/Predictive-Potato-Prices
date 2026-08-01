"""Traduccion de las decisiones a euros.

El cliente no compra MAPE, compra patata. Este modulo convierte la decision
binaria semanal en un coste medio por kg y lo compara con lo que habria pasado
sin modelo.

SIMPLIFICACIONES (importantes, hay que decirlas antes de ensenyar el numero):
  - cada semana se decide de forma independiente sobre el tramo de esa semana;
    no hay inventario, ni capacidad de almacen, ni limites de compra
  - se compra al precio de lonja, sin contratos ni descuentos por volumen
  - se ignora que un comprador de 40.000 t/anyo mueve el mercado al operar
  - el coste de almacenaje es lineal y unico (0,01 EUR/kg por las 4 semanas)

Con esas cuatro, el numero de euros es un ORDEN DE MAGNITUD del valor
potencial, no una promesa de ahorro.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from patata import config

VOLUMEN_ANUAL_KG = 40_000_000.0  # ~40.000 t/anyo


def _coste(decision, precio_ref, precio_objetivo, coste_almacenaje):
    """Coste por kg del tramo de esa semana.

    decision=True  -> compro hoy y almaceno 4 semanas: precio_ref + almacenaje
    decision=False -> espero y compro dentro de 4 semanas: precio_objetivo
    """
    return np.where(decision, precio_ref + coste_almacenaje, precio_objetivo)


def evaluar_decisiones(
    predicciones: pd.DataFrame,
    coste_almacenaje: float = config.COSTE_ALMACENAJE_EUR_KG,
    volumen_anual_kg: float = VOLUMEN_ANUAL_KG,
) -> pd.DataFrame:
    """Coste medio por kg de cada estrategia, y ahorro anual implicado."""
    df = predicciones.dropna(subset=["precio_ref", "precio_real"]).copy()
    if df.empty:
        return pd.DataFrame()

    ref = df["precio_ref"].to_numpy(dtype=float)
    obj = df["precio_real"].to_numpy(dtype=float)

    estrategias: dict[str, np.ndarray] = {}
    # referencias que no dependen de ningun modelo
    una = df["predictor"].iloc[0]
    base = df["predictor"] == una
    estrategias["[ref] siempre esperar"] = np.zeros(base.sum(), dtype=bool)
    estrategias["[ref] siempre comprar ya"] = np.ones(base.sum(), dtype=bool)
    estrategias["[ref] oraculo perfecto"] = (
        obj[base.to_numpy()] > ref[base.to_numpy()] + coste_almacenaje
    )

    filas = []
    for nombre, dec in estrategias.items():
        r, o = ref[base.to_numpy()], obj[base.to_numpy()]
        filas.append({"estrategia": nombre, "n": len(dec),
                      "coste_medio_eur_kg": float(_coste(dec, r, o, coste_almacenaje).mean()),
                      "pct_semanas_compra_ya": float(dec.mean())})

    for nombre, g in df.groupby("predictor", sort=True):
        dec = g["pred_comprar_ahora"].to_numpy(dtype=bool)
        r = g["precio_ref"].to_numpy(dtype=float)
        o = g["precio_real"].to_numpy(dtype=float)
        filas.append({"estrategia": nombre, "n": len(g),
                      "coste_medio_eur_kg": float(_coste(dec, r, o, coste_almacenaje).mean()),
                      "pct_semanas_compra_ya": float(dec.mean())})

    tabla = pd.DataFrame(filas)

    esperar = tabla.loc[tabla["estrategia"] == "[ref] siempre esperar",
                        "coste_medio_eur_kg"].iloc[0]
    oraculo = tabla.loc[tabla["estrategia"] == "[ref] oraculo perfecto",
                        "coste_medio_eur_kg"].iloc[0]
    margen = esperar - oraculo

    tabla["ahorro_eur_kg"] = esperar - tabla["coste_medio_eur_kg"]
    tabla["ahorro_anual_eur"] = tabla["ahorro_eur_kg"] * volumen_anual_kg
    tabla["pct_del_ahorro_maximo"] = (
        tabla["ahorro_eur_kg"] / margen if margen > 0 else np.nan
    )
    return tabla.sort_values("coste_medio_eur_kg").reset_index(drop=True)
