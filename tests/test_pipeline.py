"""Test de integracion: la fase entera, de la base vacia a la tabla final."""

from __future__ import annotations

from patata import pipeline, predictores


def test_la_fase_entera_se_ejecuta_y_persiste(tmp_path):
    salida = pipeline.ejecutar(ruta_db=tmp_path / "patata.db")
    con = salida["con"]

    assert salida["n_filas_raw"] == 1100
    assert len(salida["features"]) == 1100

    res = salida["resultado"]
    assert res.origenes_evaluados > 500
    assert res.origenes_saltados == 0

    tabla = salida["resultados"]
    assert list(tabla["predictor"]) == [predictores.NaiveUltimoPrecio.nombre]
    fila = tabla.iloc[0]
    assert 0 < fila["mape_pct"] < 100
    assert fila["rmse_eur_kg"] > 0
    assert 0 <= fila["tasa_base_comprar"] <= 1

    # la comparacion del baseline contra si mismo tiene que dar empate exacto
    comp = salida["comparacion"].iloc[0]
    assert comp["skill_mape"] == 0.0
    assert comp["skill_rmse"] == 0.0

    # y todo esto tiene que haber quedado en el fichero .db
    assert con.execute("SELECT count(*) FROM backtest_predicciones").fetchone()[0] > 500
    assert con.execute("SELECT count(*) FROM backtest_metricas").fetchone()[0] > 0
    con.close()


def test_estabilidad_por_anyo_cubre_todo_el_backtest(tmp_path):
    salida = pipeline.ejecutar(ruta_db=tmp_path / "patata.db")
    por_anyo = salida["por_anyo"]
    assert por_anyo["anyo"].min() == 2015
    assert len(por_anyo) >= 10
    # ningun anyo se cuela vacio
    assert (por_anyo["n"] > 0).all()
    salida["con"].close()
