"""El motor: alineacion, ventana expansiva y reentreno."""

from __future__ import annotations

import pandas as pd

from patata import backtest, evaluacion
from patata.predictores import NaiveUltimoPrecio

INICIO = "2007-01-01"


def _correr(feats, **kw):
    return backtest.walk_forward(
        feats, [NaiveUltimoPrecio()], inicio=INICIO, min_entreno=52, **kw
    )


def test_las_predicciones_estan_alineadas_con_lo_real(feats, con):
    res = _correr(feats)
    preds = res.predicciones

    # el objetivo esta exactamente a 4 semanas del origen
    assert (preds["fecha_objetivo"] - preds["fecha_origen"] == pd.Timedelta(weeks=4)).all()

    # y `precio_real` es de verdad el precio de esa semana en la tabla raw
    raw = con.execute(
        "SELECT fecha_dato, min(precio_eur_kg) AS p FROM raw_lonja GROUP BY fecha_dato"
    ).df()
    raw["fecha_dato"] = pd.to_datetime(raw["fecha_dato"])
    unido = preds.merge(raw, left_on="fecha_objetivo", right_on="fecha_dato", how="inner")
    assert len(unido) > 100
    assert (unido["precio_real"] - unido["p"]).abs().max() < 1e-9


def test_la_ventana_es_expansiva(feats):
    res = _correr(feats)
    n = res.predicciones["n_entreno"]
    assert (n.diff().dropna() >= 0).all(), "el entrenamiento debe crecer, no encogerse"
    assert n.iloc[-1] > n.iloc[0]


def test_avanza_una_semana_cada_vez(feats):
    res = _correr(feats)
    saltos = res.predicciones["fecha_origen"].diff().dropna().unique()
    assert list(saltos) == [pd.Timedelta(weeks=1)]


def test_no_evalua_antes_del_inicio(feats):
    res = _correr(feats)
    assert res.predicciones["fecha_origen"].min() >= pd.Timestamp(INICIO)


def test_salta_los_origenes_sin_entrenamiento_suficiente(feats):
    res = backtest.walk_forward(
        feats, [NaiveUltimoPrecio()], inicio="2004-01-05", min_entreno=200
    )
    assert res.origenes_saltados > 0
    assert res.motivos_salto.get("entreno_insuficiente", 0) > 0


def test_el_reentreno_periodico_se_respeta(feats):
    """Cuenta las llamadas a entrenar: con reentrenar_cada=4 tiene que ser
    aproximadamente una cuarta parte de los origenes."""

    class Espia(NaiveUltimoPrecio):
        nombre = "espia"

        def __init__(self):
            self.entrenos = 0

        def entrenar(self, train):
            self.entrenos += 1

    cada1, cada4 = Espia(), Espia()
    n1 = backtest.walk_forward(feats, [cada1], inicio=INICIO, min_entreno=52)
    n4 = backtest.walk_forward(feats, [cada4], inicio=INICIO, min_entreno=52,
                               reentrenar_cada=4)
    assert cada1.entrenos == n1.origenes_evaluados
    assert cada4.entrenos == -(-n4.origenes_evaluados // 4)


def test_las_predicciones_vivas_no_puntuan(feats):
    """Los ultimos origenes no tienen objetivo publicado: aparecen en la tabla
    de predicciones pero no cuentan como aciertos ni fallos."""
    res = _correr(feats)
    vivas = res.predicciones["precio_real"].isna().sum()
    assert vivas > 0
    tabla = evaluacion.tabla_resultados(res)
    assert tabla["n"].iloc[0] == len(res.predicciones) - vivas


def test_la_naive_predice_exactamente_el_ultimo_precio(feats):
    res = _correr(feats)
    p = res.predicciones
    assert (p["precio_pred"] == p["precio_ref"]).all()
    # y por tanto nunca recomienda comprar ahora: 0 < coste de almacenaje
    assert not p["pred_comprar_ahora"].any()
