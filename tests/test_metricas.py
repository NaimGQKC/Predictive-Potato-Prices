"""Metricas. Valores conocidos calculados a mano."""

from __future__ import annotations

import math

import numpy as np
import pytest

from patata import metricas as m


def test_mape_con_valores_conocidos():
    assert m.mape([100, 200], [110, 180]) == pytest.approx(10.0)


def test_rmse_y_mae_con_valores_conocidos():
    assert m.rmse([1.0, 2.0, 3.0], [1.0, 2.0, 5.0]) == pytest.approx(math.sqrt(4 / 3))
    assert m.mae([1.0, 2.0, 3.0], [1.0, 2.0, 5.0]) == pytest.approx(2 / 3)


def test_las_metricas_ignoran_los_nan():
    assert m.rmse([1.0, np.nan, 3.0], [1.0, 5.0, 3.0]) == pytest.approx(0.0)
    assert m.mape([0.0, 100.0], [50.0, 110.0]) == pytest.approx(10.0)


def test_un_predictor_perfecto_acierta_toda_la_direccion():
    real = [0.22, 0.18, 0.25]
    ref = [0.20, 0.20, 0.20]
    d = m.direccion(real, real, ref)
    assert d["acierto_direccion"] == pytest.approx(1.0)
    assert d["cobertura_direccion"] == pytest.approx(1.0)


def test_un_predictor_plano_no_opina_sobre_la_direccion():
    """La naive predice el precio de referencia: no dice ni sube ni baja.

    Su acierto de direccion es NaN, no 0. Contarlo como 0% seria mentir sobre
    lo que hace, y contarlo como acierto cuando el precio no se mueve tambien.
    """
    ref = [0.20, 0.20, 0.20]
    real = [0.22, 0.18, 0.25]
    d = m.direccion(real, ref, ref)
    assert math.isnan(d["acierto_direccion"])
    assert d["cobertura_direccion"] == pytest.approx(0.0)


def test_la_direccion_mayoritaria_es_el_liston_real():
    ref = [0.20] * 4
    real = [0.22, 0.23, 0.24, 0.18]  # 3 de 4 suben
    assert m.direccion(real, real, ref)["direccion_mayoritaria"] == pytest.approx(0.75)


def test_matriz_de_confusion_de_la_clasificacion():
    real = [True, True, False, False, False]
    pred = [True, False, True, False, False]
    c = m.metricas_clasificacion(real, pred)
    assert (c["vp"], c["fn"], c["fp"], c["vn"]) == (1.0, 1.0, 1.0, 2.0)
    assert c["accuracy"] == pytest.approx(0.6)
    assert c["precision"] == pytest.approx(0.5)
    assert c["recall"] == pytest.approx(0.5)
    assert c["f1"] == pytest.approx(0.5)
    assert c["tasa_base"] == pytest.approx(0.4)
    assert c["acierto_clase_mayoritaria"] == pytest.approx(0.6)


def test_la_clasificacion_ignora_las_etiquetas_desconocidas():
    c = m.metricas_clasificacion([True, None, False], [True, True, False])
    assert c["n"] == 2.0
    assert c["accuracy"] == pytest.approx(1.0)


def test_skill_score_positivo_solo_si_se_bate_al_baseline():
    assert m.skill_score(8.0, 10.0) == pytest.approx(0.2)      # error menor: mejora
    assert m.skill_score(12.0, 10.0) == pytest.approx(-0.2)    # error mayor: peor
    assert m.skill_score(0.6, 0.5, menor_es_mejor=False) == pytest.approx(0.2)


def test_el_test_pareado_no_ve_diferencia_entre_iguales():
    err = np.linspace(-0.05, 0.05, 200)
    dm = m.diebold_mariano_simple(err, err)
    assert dm["dif_media"] == pytest.approx(0.0)


def test_el_test_pareado_detecta_una_mejora_clara():
    rng = np.random.default_rng(0)
    peor = rng.normal(0, 0.05, 400)
    mejor = peor * 0.3
    dm = m.diebold_mariano_simple(mejor, peor)
    assert dm["dif_media"] < 0          # el primero tiene menos error cuadratico
    assert dm["t"] < -2                 # y la diferencia no cabe en el ruido
