"""Predictores. De momento SOLO baselines: ningun modelo entra en esta fase.

El contrato es minimo a proposito, para que el dia que entre un LightGBM se
enchufe aqui sin tocar el motor de backtest:

    class MiModelo:
        nombre = "lgbm"
        def entrenar(self, train: pd.DataFrame) -> None: ...
        def predecir(self, X: pd.DataFrame) -> pd.DataFrame: ...

`entrenar` recibe SOLO filas cuyo objetivo ya era publico en el origen; el
motor se encarga de eso. `predecir` devuelve un DataFrame con:
    precio_pred          -> precio estimado a T+horizonte (regresion)
    prob_comprar_ahora   -> opcional; si falta, la decision binaria se deriva
                            de precio_pred contra precio_ref + coste
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Predictor(Protocol):
    nombre: str

    def entrenar(self, train: pd.DataFrame) -> None: ...

    def predecir(self, X: pd.DataFrame) -> pd.DataFrame: ...


class NaiveUltimoPrecio:
    """La regla tonta: el precio dentro de 4 semanas sera el ultimo conocido.

    Es el rival a batir. No entrena nada. Como su prediccion de variacion es
    exactamente 0, y 0 < coste de almacenaje, en la decision binaria siempre
    dice "no compres ahora, espera": no ve nunca una subida que compense
    almacenar. Ese es justo el limite que hay que superar.
    """

    nombre = "naive_ultimo_precio"

    def entrenar(self, train: pd.DataFrame) -> None:  # noqa: D102 - no hay nada que entrenar
        return None

    def predecir(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "precio_pred": X["precio_ref"].to_numpy(dtype=float),
                "prob_comprar_ahora": np.full(len(X), np.nan),
            },
            index=X.index,
        )
