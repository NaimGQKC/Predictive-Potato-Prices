"""Predictores: el baseline naive y sus retadores.

El contrato es minimo a proposito, para que el motor de backtest no sepa nada
de modelos:

    class MiModelo:
        nombre = "lo_que_sea"
        def entrenar(self, train: pd.DataFrame) -> None: ...
        def predecir(self, X: pd.DataFrame) -> pd.DataFrame: ...

`entrenar` recibe SOLO filas cuyo objetivo ya era publico en el origen; de eso
se encarga el motor. `predecir` devuelve un DataFrame con:
    precio_pred          -> precio estimado a T+horizonte (regresion)
    prob_comprar_ahora   -> opcional; si falta, la decision binaria se deriva
                            de precio_pred contra precio_ref + coste

DECISION DE MODELADO: todos los retadores predicen la VARIACION
(precio_objetivo - precio_ref), no el nivel. El precio de la patata no es
estacionario: un modelo entrenado sobre el nivel aprende la media historica y
la arrastra, que es la forma mas comun de parecer bueno en train y ser inutil
en produccion. Prediciendo la variacion, el modelo tiene que ganarse cada
euro contra la naive, que es exactamente la comparacion que nos interesa.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from patata import config
from patata.features import columnas_predictoras


@runtime_checkable
class Predictor(Protocol):
    nombre: str

    def entrenar(self, train: pd.DataFrame) -> None: ...

    def predecir(self, X: pd.DataFrame) -> pd.DataFrame: ...


def _salida(X: pd.DataFrame, precio_pred, prob=None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "precio_pred": np.asarray(precio_pred, dtype=float),
            "prob_comprar_ahora": (
                np.full(len(X), np.nan) if prob is None else np.asarray(prob, dtype=float)
            ),
        },
        index=X.index,
    )


# --- baselines -------------------------------------------------------------

class NaiveUltimoPrecio:
    """La regla tonta: el precio dentro de 4 semanas sera el ultimo conocido.

    Es el rival a batir. No entrena nada. Como su variacion prevista es
    exactamente 0, y 0 < coste de almacenaje, en la decision binaria siempre
    dice "no compres ahora, espera": no ve nunca una subida que compense
    almacenar. Ese es justo el limite que hay que superar.
    """

    nombre = "naive_ultimo_precio"

    def entrenar(self, train: pd.DataFrame) -> None:
        return None

    def predecir(self, X: pd.DataFrame) -> pd.DataFrame:
        return _salida(X, X["precio_ref"].to_numpy(dtype=float))


class NaiveEstacional:
    """La variacion sera la misma que hizo en estas fechas el anyo pasado.

    precio_pred = precio_ref + (lag_48 - lag_52)

    Es el baseline honesto para una serie con estacionalidad fuerte, y mucho
    mas duro de batir que copiar el nivel del anyo pasado: aprovecha la forma
    del calendario sin arrastrar el nivel de hace un anyo, que ya no aplica.
    Cuando no hay un anyo de historico, se cae a la naive.
    """

    nombre = "naive_estacional"

    def entrenar(self, train: pd.DataFrame) -> None:
        return None

    def predecir(self, X: pd.DataFrame) -> pd.DataFrame:
        ref = X["precio_ref"].to_numpy(dtype=float)
        delta = (X["lag_48"] - X["lag_52"]).to_numpy(dtype=float)
        return _salida(X, ref + np.nan_to_num(delta, nan=0.0))


# --- modelos ---------------------------------------------------------------

class _ModeloBase:
    """Fontaneria comun: matriz de features, objetivo en variacion, fallbacks.

    Los fallbacks importan mas de lo que parece. En los primeros origenes hay
    poco historico y una sola clase en la etiqueta; sin fallback el backtest
    reventaria o, peor, devolveria basura silenciosamente.
    """

    nombre = "base"

    def __init__(self):
        self.cols: list[str] = []
        self.reg = None
        self.clf = None
        self.prob_constante: float | None = None

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.cols].to_numpy(dtype=float)

    @staticmethod
    def _y_delta(df: pd.DataFrame) -> np.ndarray:
        return (df["precio_objetivo"] - df["precio_ref"]).to_numpy(dtype=float)

    def entrenar(self, train: pd.DataFrame) -> None:
        self.cols = columnas_predictoras(train)
        X = self._X(train)
        y = self._y_delta(train)
        etiqueta = train["etiqueta_comprar_ahora"].astype("boolean")

        self._ajustar_regresion(X, y)

        # con una sola clase no hay clasificador posible: se devuelve la tasa
        # base observada y se deja constancia
        clases = etiqueta.dropna().unique()
        if len(clases) < 2:
            self.clf = None
            self.prob_constante = float(etiqueta.dropna().mean()) if len(clases) else 0.0
        else:
            self.prob_constante = None
            self._ajustar_clasificacion(X, etiqueta.to_numpy(dtype=bool))

    def _ajustar_regresion(self, X, y) -> None:
        raise NotImplementedError

    def _ajustar_clasificacion(self, X, y) -> None:
        raise NotImplementedError

    def predecir(self, X: pd.DataFrame) -> pd.DataFrame:
        M = self._X(X)
        ref = X["precio_ref"].to_numpy(dtype=float)
        delta = np.asarray(self.reg.predict(M), dtype=float)
        if self.clf is not None:
            prob = self.clf.predict_proba(M)[:, 1]
        else:
            prob = np.full(len(X), self.prob_constante)
        return _salida(X, ref + delta, prob)


class LinealRegularizado(_ModeloBase):
    """Ridge para la variacion + logistica para la decision.

    Imputacion por mediana y estandarizado, ambos ajustados SOLO con el train
    de cada ventana: si se ajustasen sobre la tabla entera, las medias
    llevarian informacion del futuro dentro. Es una fuga sutil y muy comun.
    """

    nombre = "lineal_regularizado"

    def _pipeline(self, final):
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
            ("modelo", final),
        ])

    def _ajustar_regresion(self, X, y) -> None:
        from sklearn.linear_model import Ridge

        self.reg = self._pipeline(Ridge(alpha=config.ALPHA_RIDGE))
        self.reg.fit(X, y)

    def _ajustar_clasificacion(self, X, y) -> None:
        from sklearn.linear_model import LogisticRegression

        self.clf = self._pipeline(
            LogisticRegression(C=config.C_LOGISTICA, max_iter=2000, class_weight="balanced")
        )
        self.clf.fit(X, y)


class GradientBoosting(_ModeloBase):
    """LightGBM: regresor sobre la variacion + clasificador sobre la decision.

    Hiperparametros fijos y conservadores (`config.PARAMS_LGBM`). Con ~600
    filas de entrenamiento y ~40 columnas, un LightGBM por defecto memoriza;
    de ahi las hojas cortas y el min_child_samples alto. No se tunean mirando
    el backtest, por lo mismo de siempre.
    """

    nombre = "lgbm"

    def _ajustar_regresion(self, X, y) -> None:
        from lightgbm import LGBMRegressor

        self.reg = LGBMRegressor(**config.PARAMS_LGBM, random_state=0)
        self.reg.fit(X, y)

    def _ajustar_clasificacion(self, X, y) -> None:
        from lightgbm import LGBMClassifier

        self.clf = LGBMClassifier(
            **config.PARAMS_LGBM, random_state=0, class_weight="balanced"
        )
        self.clf.fit(X, y)


def todos() -> list:
    """Los predictores de la fase 1, en orden de complejidad creciente."""
    return [
        NaiveUltimoPrecio(),
        NaiveEstacional(),
        LinealRegularizado(),
        GradientBoosting(),
    ]
