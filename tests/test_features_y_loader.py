"""Loader falso, objetivos y columnas predictoras."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from patata import config, features
from patata.loaders import falso


# --- loader falso ----------------------------------------------------------

def test_el_loader_genera_las_semanas_pedidas():
    df = falso.generar_precios(n_semanas=1100)
    assert len(df) == 1100
    assert df["fecha_dato"].is_unique


def test_todas_las_fechas_dato_son_lunes():
    df = falso.generar_precios(n_semanas=200)
    assert all(pd.Timestamp(f).dayofweek == 0 for f in df["fecha_dato"])


def test_la_publicacion_siempre_es_posterior_al_dato():
    df = falso.generar_precios(n_semanas=400)
    delta = pd.to_datetime(df["fecha_publicacion"]) - pd.to_datetime(df["fecha_dato"])
    assert (delta >= pd.Timedelta(days=1)).all()
    assert set(delta.dt.days.unique()) == {
        falso.RETARDO_BASE_DIAS,
        falso.RETARDO_BASE_DIAS + falso.RETRASO_EXTRA_DIAS,
    }


def test_los_precios_son_positivos_y_plausibles():
    df = falso.generar_precios(n_semanas=1100)
    assert df["precio_eur_kg"].min() > 0
    assert df["precio_eur_kg"].max() < 1.0


def test_la_serie_es_reproducible():
    a = falso.generar_precios(n_semanas=120, semilla=7)
    b = falso.generar_precios(n_semanas=120, semilla=7)
    c = falso.generar_precios(n_semanas=120, semilla=8)
    pd.testing.assert_frame_equal(a, b)
    assert not a["precio_eur_kg"].equals(c["precio_eur_kg"])


def test_la_serie_tiene_estacionalidad_detectable():
    """Si el loader no tuviera estacionalidad, el esqueleto no probaria nada."""
    df = falso.generar_precios(n_semanas=1100)
    df["mes"] = pd.to_datetime(df["fecha_dato"]).dt.month
    por_mes = df.groupby("mes")["precio_eur_kg"].mean()
    assert por_mes.max() - por_mes.min() > 0.03


def test_el_nivel_esta_muy_autocorrelado():
    """Por eso la naive es un rival serio: la semana que viene se parece mucho
    a esta. Si no fuera asi, batirla no tendria merito."""
    p = falso.generar_precios(n_semanas=1100)["precio_eur_kg"].to_numpy()
    assert np.corrcoef(p[:-1], p[1:])[0, 1] > 0.8


def test_la_variacion_a_4_semanas_tiene_senyal_estacional():
    """El otro lado: la serie sintetica NO es un paseo aleatorio puro.

    Si lo fuera, la naive seria optima por construccion y el esqueleto no
    podria distinguir "no hay senyal" de "el modelo es malo". Aqui se comprueba
    que la variacion a 4 semanas depende de la epoca del anyo.
    """
    df = falso.generar_precios(n_semanas=1100)
    p = df["precio_eur_kg"].to_numpy()
    d4 = p[4:] - p[:-4]
    semana = pd.to_datetime(df["fecha_dato"]).dt.isocalendar().week.to_numpy()[:-4]
    media_por_semana = pd.DataFrame({"s": semana, "d": d4}).groupby("s")["d"].mean()
    assert media_por_semana.std() > 0.01


def test_cargar_escribe_en_la_base(con):
    n = con.execute("SELECT count(*) FROM raw_lonja").fetchone()[0]
    assert n == 300
    assert con.execute("SELECT count(*) FROM fuentes").fetchone()[0] == 1


# --- objetivos -------------------------------------------------------------

def test_el_objetivo_de_regresion_es_el_precio_a_4_semanas(feats, con):
    raw = con.execute(
        "SELECT fecha_dato, min(precio_eur_kg) AS p FROM raw_lonja GROUP BY fecha_dato"
    ).df()
    raw["fecha_dato"] = pd.to_datetime(raw["fecha_dato"])
    unido = feats.merge(raw, left_on="fecha_objetivo", right_on="fecha_dato", how="inner")
    assert (unido["precio_objetivo"] - unido["p"]).abs().max() < 1e-9


def test_la_etiqueta_binaria_incluye_el_coste_de_almacenaje(feats):
    v = feats.dropna(subset=["precio_objetivo", "precio_ref", "etiqueta_comprar_ahora"])
    esperado = v["precio_objetivo"] > v["precio_ref"] + config.COSTE_ALMACENAJE_EUR_KG
    assert (v["etiqueta_comprar_ahora"].astype(bool) == esperado).all()


def test_el_coste_de_almacenaje_cambia_la_etiqueta(con):
    """Con coste 0 hay estrictamente mas semanas de 'comprar ahora' que con
    coste 0,01: el umbral no es decorativo."""
    origenes = pd.date_range("2005-01-03", "2008-01-07", freq="W-MON")
    sin_coste = features.construir_features(
        con, fuente=falso.FUENTE, coste_almacenaje=0.0, origenes=origenes, persistir=False
    )
    con_coste = features.construir_features(
        con, fuente=falso.FUENTE, coste_almacenaje=0.01, origenes=origenes, persistir=False
    )
    assert sin_coste["etiqueta_comprar_ahora"].sum() > con_coste["etiqueta_comprar_ahora"].sum()


def test_el_horizonte_es_configurable(con):
    origenes = pd.date_range("2005-01-03", "2007-01-01", freq="W-MON")
    f = features.construir_features(
        con, fuente=falso.FUENTE, horizonte_semanas=8, origenes=origenes, persistir=False
    )
    assert (f["fecha_objetivo"] - f["fecha_origen"] == pd.Timedelta(weeks=8)).all()


# --- columnas --------------------------------------------------------------

def test_las_columnas_objetivo_no_son_predictoras(feats):
    cols = features.columnas_predictoras(feats)
    for prohibida in ("precio_objetivo", "etiqueta_comprar_ahora", "fecha_objetivo",
                      "fecha_pub_objetivo", "max_fecha_pub_usada"):
        assert prohibida not in cols
    assert "precio_ref" in cols
    assert "lag_4" in cols


def test_la_tabla_de_features_se_persiste_en_duckdb(con, feats):
    n = con.execute("SELECT count(*) FROM features_semanal").fetchone()[0]
    assert n == len(feats)
    cols = {r[0] for r in con.execute("DESCRIBE features_semanal").fetchall()}
    assert {"lag_0", "media_52", "retorno_4"} <= cols


def test_los_lags_salen_del_snapshot_no_del_calendario_del_origen(feats):
    """lag_0 tiene que ser el ultimo precio publicado, que es precio_ref."""
    v = feats.dropna(subset=["precio_ref", "lag_0"])
    assert (v["lag_0"] == v["precio_ref"]).all()


def test_falla_con_la_tabla_raw_vacia():
    from patata import db

    vacia = db.conectar(":memory:")
    with pytest.raises(ValueError, match="vacia"):
        features.construir_features(vacia)
    vacia.close()
