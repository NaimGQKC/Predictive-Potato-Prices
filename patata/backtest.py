"""Motor de backtest walk-forward con ventana expansiva.

Una iteracion por lunes:
  1. corta el conjunto de entrenamiento a lo que era conocido ESE lunes
  2. reentrena
  3. predice la semana lunes + horizonte
  4. avanza una semana

Dos cortes distintos, y confundirlos es la fuga clasica:
  - las FEATURES de un origen solo usan datos publicados <= ese lunes
    (lo garantiza features.construir_features)
  - las ETIQUETAS de entrenamiento solo valen si el precio objetivo ya estaba
    publicado ese lunes: para predecir a 4 semanas, el ejemplo mas reciente
    utilizable tiene ~5 semanas de antiguedad, no 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from patata import config


class FugaTemporalError(AssertionError):
    """Se ha colado informacion del futuro. El backtest se para en seco."""


@dataclass
class ResultadoBacktest:
    predicciones: pd.DataFrame
    origenes_evaluados: int
    origenes_saltados: int
    motivos_salto: dict = field(default_factory=dict)
    horizonte_semanas: int = config.HORIZONTE_SEMANAS
    coste_almacenaje: float = config.COSTE_ALMACENAJE_EUR_KG


def auditar_pit(feats: pd.DataFrame, horizonte_semanas: int = config.HORIZONTE_SEMANAS) -> None:
    """Comprueba las invariantes de doble fecha sobre la tabla de features.

    No basta con mirar si el filtro de publicacion se aplico: hay que comprobar
    que las propias fechas son coherentes. Una `fecha_pub_objetivo` anterior a
    la semana que describe es imposible en el mundo real, y es justo lo que
    veriamos si alguien "arreglase" el calendario para tener mas ejemplos.
    """
    if (feats["fecha_asof"] != feats["fecha_origen"]).any():
        raise FugaTemporalError("fecha_asof no coincide con fecha_origen")

    usada = feats["max_fecha_pub_usada"]
    mal = feats.loc[usada.notna() & (usada > feats["fecha_origen"])]
    if len(mal):
        raise FugaTemporalError(
            f"{len(mal)} filas usan datos publicados despues de su origen; "
            f"primera: {mal['fecha_origen'].iloc[0]}"
        )

    ult = feats["fecha_dato_ultimo"]
    mal = feats.loc[ult.notna() & (ult >= feats["fecha_origen"])]
    if len(mal):
        raise FugaTemporalError(
            "hay origenes que ya conocen el precio de su propia semana o posterior; "
            f"primero: {mal['fecha_origen'].iloc[0]}"
        )

    esperado = feats["fecha_origen"] + pd.Timedelta(weeks=horizonte_semanas)
    obj = feats["fecha_objetivo"]
    mal = feats.loc[obj.notna() & (obj != esperado)]
    if len(mal):
        raise FugaTemporalError(
            f"fecha_objetivo no esta a {horizonte_semanas} semanas del origen; "
            f"primera: {mal['fecha_origen'].iloc[0]}"
        )

    pub_obj = feats["fecha_pub_objetivo"]
    mal = feats.loc[pub_obj.notna() & obj.notna() & (pub_obj < obj)]
    if len(mal):
        raise FugaTemporalError(
            "hay etiquetas publicadas antes de la semana que describen: imposible; "
            f"primera: {mal['fecha_origen'].iloc[0]}"
        )


def entrenamiento_disponible(feats: pd.DataFrame, origen: pd.Timestamp) -> pd.DataFrame:
    """Filas utilizables como entrenamiento el lunes `origen`."""
    return feats.loc[
        (feats["fecha_origen"] < origen)
        & feats["fecha_pub_objetivo"].notna()
        & (feats["fecha_pub_objetivo"] <= origen)
        & feats["precio_objetivo"].notna()
        & feats["precio_ref"].notna()
    ]


def walk_forward(
    feats: pd.DataFrame,
    predictores,
    inicio: str = config.INICIO_BACKTEST,
    fin: str | None = None,
    horizonte_semanas: int = config.HORIZONTE_SEMANAS,
    coste_almacenaje: float = config.COSTE_ALMACENAJE_EUR_KG,
    min_entreno: int = config.MIN_EJEMPLOS_ENTRENO,
    reentrenar_cada: int = 1,
    auditar: bool = True,
) -> ResultadoBacktest:
    """Ejecuta el walk-forward y devuelve predicciones alineadas con lo real."""
    if auditar:
        auditar_pit(feats, horizonte_semanas=horizonte_semanas)

    feats = feats.sort_values("fecha_origen").reset_index(drop=True)
    inicio = pd.Timestamp(inicio)
    fin = pd.Timestamp(fin) if fin else feats["fecha_origen"].max()

    origenes = feats.loc[
        feats["fecha_origen"].between(inicio, fin), "fecha_origen"
    ].tolist()

    registros: list[dict] = []
    motivos: dict[str, int] = {}
    saltados = 0
    evaluados = 0
    paso = 0

    for origen in origenes:
        fila_test = feats.loc[feats["fecha_origen"] == origen]
        if not np.isfinite(fila_test["precio_ref"].iloc[0]):
            motivos["sin_precio_ref"] = motivos.get("sin_precio_ref", 0) + 1
            saltados += 1
            continue

        train = entrenamiento_disponible(feats, origen)
        if len(train) < min_entreno:
            motivos["entreno_insuficiente"] = motivos.get("entreno_insuficiente", 0) + 1
            saltados += 1
            continue

        # Cinturon y tirantes: el motor vuelve a comprobar lo que ya garantiza
        # el constructor de features. Si esto salta, no publicamos resultados.
        if auditar:
            if (train["fecha_pub_objetivo"] > origen).any():
                raise FugaTemporalError(f"etiqueta no publicada aun en el origen {origen.date()}")
            if (train["max_fecha_pub_usada"] > train["fecha_origen"]).any():
                raise FugaTemporalError(f"features del futuro en el entreno de {origen.date()}")
            if (fila_test["max_fecha_pub_usada"] > origen).any():
                raise FugaTemporalError(f"features del futuro en el test de {origen.date()}")

        toca_reentrenar = (paso % reentrenar_cada) == 0
        paso += 1

        for pred in predictores:
            if toca_reentrenar:
                pred.entrenar(train)
            salida = pred.predecir(fila_test)
            precio_pred = float(salida["precio_pred"].iloc[0])
            prob = (
                float(salida["prob_comprar_ahora"].iloc[0])
                if "prob_comprar_ahora" in salida.columns
                else np.nan
            )
            precio_ref = float(fila_test["precio_ref"].iloc[0])
            if np.isfinite(prob):
                decision = bool(prob >= 0.5)
            else:
                decision = bool(precio_pred > precio_ref + coste_almacenaje)

            real = fila_test["precio_objetivo"].iloc[0]
            etiqueta = fila_test["etiqueta_comprar_ahora"].iloc[0]
            registros.append(
                {
                    "predictor": pred.nombre,
                    "fecha_origen": origen,
                    "fecha_objetivo": fila_test["fecha_objetivo"].iloc[0],
                    "n_entreno": int(len(train)),
                    "precio_ref": precio_ref,
                    "precio_pred": precio_pred,
                    "precio_real": float(real) if pd.notna(real) else np.nan,
                    "prob_comprar_ahora": prob,
                    "pred_comprar_ahora": decision,
                    "real_comprar_ahora": bool(etiqueta) if pd.notna(etiqueta) else None,
                }
            )
        evaluados += 1

    preds = pd.DataFrame(registros)
    if not preds.empty:
        preds["real_comprar_ahora"] = preds["real_comprar_ahora"].astype("boolean")
        preds = preds.sort_values(["predictor", "fecha_origen"]).reset_index(drop=True)

    return ResultadoBacktest(
        predicciones=preds,
        origenes_evaluados=evaluados,
        origenes_saltados=saltados,
        motivos_salto=motivos,
        horizonte_semanas=horizonte_semanas,
        coste_almacenaje=coste_almacenaje,
    )
