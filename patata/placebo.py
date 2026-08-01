"""Prueba placebo: destruir la senyal y comprobar que el aparato lo nota.

Un backtest que da buenos numeros puede estar midiendo dos cosas muy
distintas: que el modelo predice, o que se ha colado informacion del futuro.
Los tests de doble fecha atacan el problema por construccion; esto lo ataca por
el resultado.

Se baraja el objetivo entre origenes, dejando las features intactas. Despues de
barajar NO QUEDA NADA que predecir: cualquier modelo tiene que caer a
rendimiento nulo (skill ~ 0, accuracy balanceada ~ 0,5). Si sigue ganando,
esta leyendo el futuro por algun sitio y hay que buscar por donde.

OJO CON COMO SE BARAJA. La primera version de este modulo permutaba el NIVEL
del precio objetivo, y los modelos seguian sacando skill 0,29 y accuracy
balanceada 0,81. No era fuga: era el placebo mal disenyado. Si el objetivo pasa
a ser un precio al azar del historico, entonces

    variacion = precio_al_azar - precio_ref

es perfectamente predecible a partir de precio_ref (basta con estimar el precio
medio), y la etiqueta "precio_al_azar > precio_ref + coste" tambien lo es. El
placebo habia creado una senyal nueva en vez de destruir la que habia.

Lo que hay que permutar es la VARIACION, que es lo que los modelos predicen.
Asi la variacion queda independiente de las features y el objetivo se
reconstruye como precio_ref + variacion_barajada, conservando el nivel de cada
semana. Entonces si: no queda nada que aprender.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from patata import config


def barajar_objetivos(
    feats: pd.DataFrame,
    semilla: int = 0,
    coste_almacenaje: float = config.COSTE_ALMACENAJE_EUR_KG,
) -> pd.DataFrame:
    """Permuta la VARIACION a 4 semanas entre origenes y recalcula la etiqueta.

    Se conservan las fechas, el nivel de precio de cada semana y todas las
    features: lo unico que se rompe es la relacion entre lo que sabemos en el
    origen y como se movera el precio. La etiqueta se RECALCULA a partir del
    objetivo reconstruido, no se baraja aparte: si no, dejaria de ser coherente
    con la regresion.
    """
    rng = np.random.default_rng(semilla)
    barajado = feats.copy()

    observados = barajado["precio_objetivo"].notna() & barajado["precio_ref"].notna()
    variacion = (
        barajado.loc[observados, "precio_objetivo"] - barajado.loc[observados, "precio_ref"]
    ).to_numpy()
    barajado.loc[observados, "precio_objetivo"] = (
        barajado.loc[observados, "precio_ref"].to_numpy() + rng.permutation(variacion)
    )

    umbral = barajado["precio_ref"] + coste_almacenaje
    etiqueta = barajado["precio_objetivo"] > umbral
    barajado["etiqueta_comprar_ahora"] = etiqueta.where(
        barajado["precio_objetivo"].notna() & barajado["precio_ref"].notna()
    )
    return barajado
