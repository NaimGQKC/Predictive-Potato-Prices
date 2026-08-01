"""Los retadores: contrato, objetivo en variacion, fallbacks y control placebo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from patata import backtest, evaluacion, features, placebo, predictores

INICIO = "2007-01-01"


# --- contrato --------------------------------------------------------------

def test_todos_los_predictores_tienen_nombre_unico():
    lista = predictores.todos()
    nombres = [p.nombre for p in lista]
    assert len(nombres) == len(set(nombres))
    assert predictores.NaiveUltimoPrecio.nombre in nombres


@pytest.mark.parametrize("clase", [
    predictores.NaiveUltimoPrecio,
    predictores.NaiveEstacional,
    predictores.LinealRegularizado,
    predictores.GradientBoosting,
])
def test_cada_predictor_cumple_el_contrato(clase, feats):
    p = clase()
    assert isinstance(p, predictores.Predictor)

    train = feats.dropna(subset=["precio_objetivo", "precio_ref", "etiqueta_comprar_ahora"])
    train = train.iloc[:150]
    test = feats.iloc[[200]]

    p.entrenar(train)
    salida = p.predecir(test)
    assert list(salida.columns) == ["precio_pred", "prob_comprar_ahora"]
    assert salida.index.equals(test.index)
    assert np.isfinite(salida["precio_pred"].iloc[0])


def test_ningun_modelo_ve_columnas_de_objetivo(feats):
    """La lista de columnas que usa el modelo se calcula una sola vez, aqui.

    Si alguien anyade una feature derivada del objetivo, este test es el que
    tiene que saltar.
    """
    train = feats.dropna(subset=["precio_objetivo", "precio_ref"]).iloc[:150]
    m = predictores.LinealRegularizado()
    m.entrenar(train)
    prohibidas = set(features.COLS_NO_PREDICTORAS)
    assert prohibidas.isdisjoint(m.cols)
    assert "precio_ref" in m.cols


# --- el baseline estacional ------------------------------------------------

def test_el_naive_estacional_usa_la_variacion_del_anyo_pasado():
    X = pd.DataFrame({"precio_ref": [0.20], "lag_48": [0.30], "lag_52": [0.25]})
    salida = predictores.NaiveEstacional().predecir(X)
    # el anyo pasado subio 0,05 en ese mismo tramo -> 0,20 + 0,05
    assert salida["precio_pred"].iloc[0] == pytest.approx(0.25)


def test_el_naive_estacional_se_cae_a_la_naive_sin_historico():
    X = pd.DataFrame({"precio_ref": [0.20], "lag_48": [np.nan], "lag_52": [np.nan]})
    salida = predictores.NaiveEstacional().predecir(X)
    assert salida["precio_pred"].iloc[0] == pytest.approx(0.20)


# --- objetivo en variacion y fallbacks -------------------------------------

def test_los_modelos_entrenan_sobre_la_variacion_no_sobre_el_nivel(feats):
    train = feats.dropna(subset=["precio_objetivo", "precio_ref"]).iloc[:150]
    y = predictores._ModeloBase._y_delta(train)
    esperado = (train["precio_objetivo"] - train["precio_ref"]).to_numpy()
    assert np.allclose(y, esperado)
    # y la variacion esta centrada cerca de cero, a diferencia del nivel
    assert abs(np.mean(y)) < 0.02
    assert train["precio_objetivo"].mean() > 0.1


def test_con_una_sola_clase_no_revienta_y_devuelve_la_tasa_base(feats):
    """Los primeros origenes pueden no tener ni una semana de 'comprar ahora'."""
    train = feats.dropna(subset=["precio_objetivo", "precio_ref"]).iloc[:150].copy()
    train["etiqueta_comprar_ahora"] = False

    m = predictores.LinealRegularizado()
    m.entrenar(train)
    assert m.clf is None
    assert m.prob_constante == pytest.approx(0.0)

    salida = m.predecir(feats.iloc[[200]])
    assert salida["prob_comprar_ahora"].iloc[0] == pytest.approx(0.0)


# --- control placebo -------------------------------------------------------

def test_el_placebo_conserva_features_y_nivel_pero_rompe_la_relacion(feats):
    barajado = placebo.barajar_objetivos(feats, semilla=3)
    cols_feat = features.columnas_predictoras(feats)
    pd.testing.assert_frame_equal(feats[cols_feat], barajado[cols_feat])
    assert not feats["precio_objetivo"].equals(barajado["precio_objetivo"])

    # el conjunto de variaciones es el mismo, solo cambia a que origen va cada una
    v1 = (feats["precio_objetivo"] - feats["precio_ref"]).dropna().sort_values()
    v2 = (barajado["precio_objetivo"] - barajado["precio_ref"]).dropna().sort_values()
    assert np.allclose(v1.to_numpy(), v2.to_numpy())


def test_sin_senyal_ningun_modelo_bate_a_la_naive(feats):
    """EL CONTROL NEGATIVO.

    Con la variacion objetivo barajada no queda nada que predecir. Si un modelo
    sigue sacando skill positivo, esta leyendo el futuro por algun sitio.

    Se usa solo el lineal por tiempo de ejecucion; es ademas el que mas skill
    saca con senyal, asi que es el canario adecuado.
    """
    barajado = placebo.barajar_objetivos(feats, semilla=1)
    res = backtest.walk_forward(
        barajado,
        [predictores.NaiveUltimoPrecio(), predictores.LinealRegularizado()],
        inicio=INICIO,
        min_entreno=104,
    )
    comp = evaluacion.comparar_con_baseline(res, predictores.NaiveUltimoPrecio.nombre)
    lineal = comp.loc[comp["predictor"] == "lineal_regularizado"].iloc[0]
    assert lineal["skill_mape"] < 0.10, f"skill sospechoso sin senyal: {lineal['skill_mape']}"

    tabla = evaluacion.tabla_resultados(res).set_index("predictor")
    assert abs(tabla.loc["lineal_regularizado", "clf_acc_balanceada"] - 0.5) < 0.10


def test_con_senyal_el_lineal_si_bate_a_la_naive(feats):
    """La otra mitad del control: el mismo montaje, sin barajar, SI gana.

    Sin este test, el anterior pasaria tambien con un motor roto que devolviese
    siempre lo mismo.
    """
    res = backtest.walk_forward(
        feats,
        [predictores.NaiveUltimoPrecio(), predictores.LinealRegularizado()],
        inicio=INICIO,
        min_entreno=104,
    )
    comp = evaluacion.comparar_con_baseline(res, predictores.NaiveUltimoPrecio.nombre)
    lineal = comp.loc[comp["predictor"] == "lineal_regularizado"].iloc[0]
    assert lineal["skill_mape"] > 0.10
    assert lineal["dm_t"] < 0

    tabla = evaluacion.tabla_resultados(res).set_index("predictor")
    assert tabla.loc["lineal_regularizado", "clf_acc_balanceada"] > 0.60

    # Nota: aqui NO se exige |dm_t| > 2. El fixture son 300 semanas y deja ~90
    # origenes evaluables, con los que la diferencia no es concluyente aunque
    # exista (dm_t ~ -0,8). Con las 1.100 semanas completas sale ~ -5,7. Que el
    # test pareado sea prudente con muestras pequenyas es la gracia de tenerlo.
