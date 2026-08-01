"""Test de integracion: la fase entera, de la base vacia a la tabla final.

Se ejecuta con los dos predictores baratos. Lo que se prueba aqui es la
fontaneria (que todo se encadena y se persiste), no la calidad de los modelos:
de eso se encarga test_predictores.py.
"""

from __future__ import annotations

from patata import pipeline, predictores


def _rapido():
    return [predictores.NaiveUltimoPrecio(), predictores.NaiveEstacional()]


def test_la_fase_entera_se_ejecuta_y_persiste(tmp_path):
    salida = pipeline.ejecutar(ruta_db=tmp_path / "patata.db", predictores_usados=_rapido())
    con = salida["con"]

    assert salida["n_filas_raw"] == 1100
    assert len(salida["features"]) == 1100

    res = salida["resultado"]
    assert res.origenes_evaluados > 500
    assert res.origenes_saltados == 0

    tabla = salida["resultados"]
    assert set(tabla["predictor"]) == {"naive_ultimo_precio", "naive_estacional"}
    fila = tabla.set_index("predictor").loc["naive_ultimo_precio"]
    assert 0 < fila["mape_pct"] < 100
    assert fila["rmse_eur_kg"] > 0
    assert 0 <= fila["tasa_base_comprar"] <= 1

    # la comparacion del baseline contra si mismo tiene que dar empate exacto
    comp = salida["comparacion"].set_index("predictor").loc["naive_ultimo_precio"]
    assert comp["skill_mape"] == 0.0
    assert comp["skill_rmse"] == 0.0

    # y todo esto tiene que haber quedado en el fichero .db
    assert con.execute("SELECT count(*) FROM backtest_predicciones").fetchone()[0] > 1000
    assert con.execute("SELECT count(*) FROM backtest_metricas").fetchone()[0] > 0
    con.close()


def test_la_tabla_en_euros_es_coherente(tmp_path):
    salida = pipeline.ejecutar(ruta_db=tmp_path / "patata.db", predictores_usados=_rapido())
    eco = salida["economia"].set_index("estrategia")

    # el oraculo no puede ser batido por nadie
    assert eco["coste_medio_eur_kg"].min() == eco.loc["[ref] oraculo perfecto",
                                                      "coste_medio_eur_kg"]
    # la naive nunca compra por adelantado, asi que es exactamente "siempre esperar"
    assert eco.loc["naive_ultimo_precio", "coste_medio_eur_kg"] == (
        eco.loc["[ref] siempre esperar", "coste_medio_eur_kg"]
    )
    assert eco.loc["naive_ultimo_precio", "ahorro_eur_kg"] == 0.0
    assert eco.loc["[ref] oraculo perfecto", "pct_del_ahorro_maximo"] == 1.0
    salida["con"].close()


def test_el_modo_placebo_destruye_la_relacion(tmp_path):
    """El pipeline en modo placebo tiene que dar metricas distintas: si diera
    lo mismo, el flag no estaria haciendo nada."""
    normal = pipeline.ejecutar(ruta_db=tmp_path / "a.db", predictores_usados=_rapido())
    barajado = pipeline.ejecutar(
        ruta_db=tmp_path / "b.db",
        predictores_usados=_rapido(),
        placebo=True,
    )
    m1 = normal["resultados"].set_index("predictor").loc["naive_estacional", "mape_pct"]
    m2 = barajado["resultados"].set_index("predictor").loc["naive_estacional", "mape_pct"]
    assert m1 != m2
    normal["con"].close()
    barajado["con"].close()


def test_estabilidad_por_anyo_cubre_todo_el_backtest(tmp_path):
    salida = pipeline.ejecutar(ruta_db=tmp_path / "patata.db", predictores_usados=_rapido())
    por_anyo = salida["por_anyo"]
    assert por_anyo["anyo"].min() == 2015
    assert len(por_anyo) >= 20  # dos predictores x 10 anyos
    # ningun anyo se cuela vacio
    assert (por_anyo["n"] > 0).all()
    salida["con"].close()
