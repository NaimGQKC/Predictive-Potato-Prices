"""Orquestacion: db -> loader -> features -> backtest -> metricas."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from patata import backtest, config, db, evaluacion, features, predictores
from patata.loaders import falso


def ejecutar(
    ruta_db: str | Path | None = None,
    recargar: bool = True,
    inicio: str = config.INICIO_BACKTEST,
    min_entreno: int = config.MIN_EJEMPLOS_ENTRENO,
    reentrenar_cada: int = 1,
) -> dict:
    """Ejecuta la fase completa y devuelve todo lo producido."""
    con = db.conectar(ruta_db)

    if recargar or con.execute("SELECT count(*) FROM raw_lonja").fetchone()[0] == 0:
        n_filas = falso.cargar(con)
    else:
        n_filas = con.execute("SELECT count(*) FROM raw_lonja").fetchone()[0]

    feats = features.construir_features(con, fuente=falso.FUENTE)
    backtest.auditar_pit(feats)

    lista = [predictores.NaiveUltimoPrecio()]
    res = backtest.walk_forward(
        feats,
        lista,
        inicio=inicio,
        min_entreno=min_entreno,
        reentrenar_cada=reentrenar_cada,
    )

    resultados = evaluacion.tabla_resultados(res)
    por_anyo = evaluacion.tabla_por_anyo(res)
    comparacion = evaluacion.comparar_con_baseline(res, predictores.NaiveUltimoPrecio.nombre)

    _persistir(con, res, resultados)

    return {
        "con": con,
        "n_filas_raw": n_filas,
        "features": feats,
        "resultado": res,
        "resultados": resultados,
        "por_anyo": por_anyo,
        "comparacion": comparacion,
    }


def _persistir(con, res: backtest.ResultadoBacktest, resultados: pd.DataFrame) -> None:
    if res.predicciones.empty:
        return
    preds = res.predicciones.copy()
    preds["real_comprar_ahora"] = preds["real_comprar_ahora"].astype("object").where(
        preds["real_comprar_ahora"].notna(), None
    )
    db.guardar_df(con, preds, "backtest_predicciones", reemplazar=True)
    db.guardar_df(
        con, evaluacion.a_formato_largo(resultados), "backtest_metricas", reemplazar=True
    )
