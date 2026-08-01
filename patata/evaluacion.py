"""De predicciones crudas a tabla de resultados."""

from __future__ import annotations

import numpy as np
import pandas as pd

from patata import metricas as m
from patata.backtest import ResultadoBacktest


def _fila_metricas(g: pd.DataFrame) -> dict:
    reg = m.metricas_regresion(g["precio_real"], g["precio_pred"], g["precio_ref"])
    cls = m.metricas_clasificacion(g["real_comprar_ahora"], g["pred_comprar_ahora"])
    return {
        "n": reg["n"],
        "mape_pct": reg["mape_pct"],
        "rmse_eur_kg": reg["rmse_eur_kg"],
        "mae_eur_kg": reg["mae_eur_kg"],
        "acierto_direccion": reg["acierto_direccion"],
        "cobertura_direccion": reg["cobertura_direccion"],
        "direccion_mayoritaria": reg["direccion_mayoritaria"],
        "clf_accuracy": cls["accuracy"],
        "clf_precision": cls["precision"],
        "clf_recall": cls["recall"],
        "clf_f1": cls["f1"],
        "clf_acc_balanceada": cls["accuracy_balanceada"],
        "tasa_base_comprar": cls["tasa_base"],
        "acierto_clase_mayoritaria": cls["acierto_clase_mayoritaria"],
    }


def tabla_resultados(res: ResultadoBacktest) -> pd.DataFrame:
    """Una fila por predictor con todas las metricas."""
    if res.predicciones.empty:
        return pd.DataFrame()
    filas = {
        nombre: _fila_metricas(g)
        for nombre, g in res.predicciones.groupby("predictor", sort=True)
    }
    return pd.DataFrame(filas).T.rename_axis("predictor").reset_index()


def tabla_por_anyo(res: ResultadoBacktest, predictor: str | None = None) -> pd.DataFrame:
    """Las mismas metricas partidas por anyo: un modelo que solo gana en 2016
    y pierde en los otros nueve anyos no gana."""
    if res.predicciones.empty:
        return pd.DataFrame()
    df = res.predicciones
    if predictor:
        df = df.loc[df["predictor"] == predictor]
    filas = []
    for (nombre, anyo), g in df.groupby(["predictor", df["fecha_origen"].dt.year]):
        # los ultimos origenes todavia no tienen objetivo observado: son
        # predicciones vivas, no resultados. No generan fila de metricas.
        if g["precio_real"].notna().sum() == 0:
            continue
        filas.append({"predictor": nombre, "anyo": int(anyo), **_fila_metricas(g)})
    return pd.DataFrame(filas).sort_values(["predictor", "anyo"]).reset_index(drop=True)


def comparar_con_baseline(res: ResultadoBacktest, baseline: str) -> pd.DataFrame:
    """Skill score de cada predictor contra el baseline, mas el test pareado."""
    tabla = tabla_resultados(res).set_index("predictor")
    if baseline not in tabla.index:
        raise ValueError(f"baseline '{baseline}' no esta en el backtest")
    base = tabla.loc[baseline]

    preds = res.predicciones

    def _errores(nombre: str) -> pd.Series:
        g = preds.loc[preds["predictor"] == nombre].set_index("fecha_origen")
        return g["precio_real"] - g["precio_pred"]

    err_base = _errores(baseline)

    filas = []
    for nombre in tabla.index:
        fila = tabla.loc[nombre]
        err = _errores(nombre)
        comun = err.index.intersection(err_base.index)
        dm = m.diebold_mariano_simple(err.loc[comun], err_base.loc[comun])
        filas.append(
            {
                "predictor": nombre,
                "skill_mape": m.skill_score(fila["mape_pct"], base["mape_pct"]),
                "skill_rmse": m.skill_score(fila["rmse_eur_kg"], base["rmse_eur_kg"]),
                # la naive no opina sobre la direccion, asi que su acierto es
                # NaN y restarlo no dice nada. El liston util es apostar
                # siempre a la direccion mas frecuente.
                "direccion_vs_mayoritaria": (
                    fila["acierto_direccion"] - fila["direccion_mayoritaria"]
                ),
                "delta_clf_acc_balanceada": (
                    fila["clf_acc_balanceada"] - base["clf_acc_balanceada"]
                ),
                "dm_t": np.nan if nombre == baseline else dm["t"],
            }
        )
    return pd.DataFrame(filas)


def a_formato_largo(tabla: pd.DataFrame) -> pd.DataFrame:
    """Pasa la tabla ancha a (predictor, metrica, valor) para persistirla."""
    return tabla.melt(id_vars="predictor", var_name="metrica", value_name="valor")
