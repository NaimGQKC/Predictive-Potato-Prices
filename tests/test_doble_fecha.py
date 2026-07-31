"""LA REGLA INNEGOCIABLE.

Si cualquiera de estos tests falla, los resultados del backtest son mentira y
no se publican. Todo lo demas del proyecto es negociable; esto no.
"""

from __future__ import annotations

import pandas as pd
import pytest

from patata import backtest, db, features
from patata.loaders import falso


# --- 1. el snapshot no ve el futuro ---------------------------------------

def test_snapshot_nunca_incluye_publicaciones_posteriores(con):
    for fecha_asof in pd.date_range("2005-01-03", "2009-01-01", freq="13W-MON"):
        snap = features.snapshot(con, fecha_asof, fuente=falso.FUENTE)
        assert (snap["fecha_publicacion"] <= fecha_asof).all(), (
            f"snapshot del {fecha_asof.date()} contiene datos publicados despues"
        )


def test_snapshot_crece_de_forma_monotona(con):
    anterior = -1
    for fecha_asof in pd.date_range("2005-01-03", "2009-01-01", freq="4W-MON"):
        n = len(features.snapshot(con, fecha_asof, fuente=falso.FUENTE))
        assert n >= anterior
        anterior = n


# --- 2. la tabla de features respeta la regla ------------------------------

def test_ninguna_fila_de_features_usa_datos_publicados_despues_de_su_origen(feats):
    usada = feats["max_fecha_pub_usada"]
    infractoras = feats.loc[usada.notna() & (usada > feats["fecha_origen"])]
    assert infractoras.empty, (
        f"{len(infractoras)} filas usan publicaciones futuras: "
        f"{infractoras['fecha_origen'].head().tolist()}"
    )


def test_el_ultimo_dato_conocido_es_siempre_anterior_al_origen(feats):
    """El lunes de la semana T todavia no se conoce el precio de la semana T.

    Si esto falla es la fuga clasica: un shift() mal puesto.
    """
    ult = feats["fecha_dato_ultimo"]
    malas = feats.loc[ult.notna() & (ult >= feats["fecha_origen"])]
    assert malas.empty


def test_antiguedad_refleja_el_retardo_real_de_publicacion(feats):
    """Con retardo de 3 dias y algun retraso extra, el ultimo dato tiene
    1 semana de antiguedad casi siempre y 2 cuando la lonja se retrasa."""
    validas = feats.loc[feats["antiguedad_semanas"].notna(), "antiguedad_semanas"]
    assert validas.min() >= 1, "algun origen conoce el precio de su propia semana"
    assert set(validas.unique()) <= {1.0, 2.0}


def test_auditoria_pit_pasa_sobre_la_tabla_construida(feats):
    backtest.auditar_pit(feats)  # no debe lanzar


# --- 3. el test adversario: envenenar el futuro ----------------------------

def test_un_dato_publicado_tarde_no_altera_el_pasado(con):
    """Publicamos una revision absurda y comprobamos que solo afecta a los
    origenes POSTERIORES a su fecha de publicacion.

    Este es el test que de verdad demuestra la propiedad point-in-time: si el
    constructor mirase la serie entera, este precio de 99 EUR/kg contaminaria
    filas de hace anyos.
    """
    antes = features.construir_features(con, fuente=falso.FUENTE, persistir=False)

    fecha_dato = pd.Timestamp("2006-01-02")
    fecha_pub_tardia = pd.Timestamp("2008-06-02")
    con.execute(
        "INSERT INTO raw_lonja (fuente, fecha_dato, fecha_publicacion, variedad, "
        "precio_eur_kg, unidad) VALUES (?, ?, ?, ?, ?, ?)",
        [falso.FUENTE, fecha_dato.date(), fecha_pub_tardia.date(),
         falso.VARIEDAD, 99.0, "EUR/kg"],
    )
    despues = features.construir_features(con, fuente=falso.FUENTE, persistir=False)

    # ojo: el origen que cae EXACTAMENTE en la fecha de publicacion si debe
    # ver el dato nuevo (la regla es fecha_publicacion <= lunes del origen)
    pasado = antes["fecha_origen"] < fecha_pub_tardia
    pd.testing.assert_frame_equal(
        antes.loc[pasado].reset_index(drop=True),
        despues.loc[pasado].reset_index(drop=True),
        check_exact=True,
        obj="filas anteriores a la publicacion tardia",
    )

    # ...y despues de esa fecha si tiene que notarse, si no el test no probaria nada
    futuro = antes["fecha_origen"] >= fecha_pub_tardia
    assert not antes.loc[futuro].equals(despues.loc[futuro])


def test_recortar_la_base_por_fecha_de_publicacion_no_cambia_el_pasado(con):
    """Reconstruir con una base que solo contiene lo publicado hasta X debe dar
    exactamente las mismas features para los origenes <= X.

    Es la formulacion fuerte de la regla: el pasado no depende del futuro.
    """
    corte = pd.Timestamp("2007-01-01")
    origenes = pd.date_range("2005-06-06", corte, freq="4W-MON")
    completa = features.construir_features(
        con, fuente=falso.FUENTE, origenes=origenes, persistir=False
    )

    recortada_con = db.conectar(":memory:")
    filas = con.execute(
        "SELECT fuente, fecha_dato, fecha_publicacion, variedad, precio_eur_kg, unidad "
        "FROM raw_lonja WHERE fecha_publicacion <= ?", [corte.date()]
    ).df()
    db.guardar_df(recortada_con, filas, "raw_lonja", reemplazar=False)
    recortada = features.construir_features(
        recortada_con, fuente=falso.FUENTE, origenes=origenes, persistir=False
    )

    cols = [c for c in completa.columns if not c.startswith(("fecha_objetivo",
                                                             "precio_objetivo",
                                                             "fecha_pub_objetivo",
                                                             "etiqueta_"))]
    pd.testing.assert_frame_equal(completa[cols], recortada[cols], check_exact=True)
    recortada_con.close()


# --- 4. las etiquetas de entrenamiento tambien tienen fecha ----------------

def test_el_entrenamiento_solo_usa_etiquetas_ya_publicadas(feats):
    for origen in feats["fecha_origen"].iloc[120::20]:
        train = backtest.entrenamiento_disponible(feats, origen)
        assert (train["fecha_pub_objetivo"] <= origen).all()
        assert (train["fecha_origen"] < origen).all()


def test_el_ejemplo_mas_reciente_respeta_horizonte_mas_retardo(feats):
    """Para predecir a 4 semanas, el ultimo ejemplo etiquetado NO puede ser de
    hace 4 semanas: su precio objetivo aun no se ha publicado."""
    for origen in feats["fecha_origen"].iloc[150::30]:
        train = backtest.entrenamiento_disponible(feats, origen)
        hueco_semanas = (origen - train["fecha_origen"].max()).days / 7
        assert hueco_semanas >= 5, (
            f"origen {origen.date()}: hueco de solo {hueco_semanas} semanas"
        )


# --- 5. el motor se planta si algo huele mal -------------------------------

def test_el_motor_aborta_si_las_features_miran_al_futuro(feats):
    corrupto = feats.copy()
    corrupto.loc[10, "max_fecha_pub_usada"] += pd.Timedelta(days=14)
    with pytest.raises(backtest.FugaTemporalError):
        backtest.auditar_pit(corrupto)


def test_el_motor_aborta_si_se_conoce_el_precio_de_la_semana_actual(feats):
    """Simula el bug mas comun: alinear el ultimo dato con la propia semana."""
    corrupto = feats.copy()
    corrupto["fecha_dato_ultimo"] = corrupto["fecha_origen"]
    with pytest.raises(backtest.FugaTemporalError):
        backtest.auditar_pit(corrupto)


def test_el_motor_aborta_si_una_etiqueta_se_publica_antes_de_su_semana(feats):
    """Adelantar la fecha de publicacion de la etiqueta al propio origen es lo
    que haria alguien buscando "mas ejemplos de entrenamiento".

    El filtro de entrenamiento por si solo NO lo detecta (las fechas siguen
    siendo <= origen); lo detecta la auditoria de coherencia: un precio no se
    puede publicar antes de la semana a la que se refiere.
    """
    from patata.predictores import NaiveUltimoPrecio

    corrupto = feats.copy()
    corrupto["fecha_pub_objetivo"] = corrupto["fecha_origen"]
    with pytest.raises(backtest.FugaTemporalError, match="antes de la semana"):
        backtest.walk_forward(
            corrupto,
            [NaiveUltimoPrecio()],
            inicio=str(corrupto["fecha_origen"].iloc[150].date()),
            min_entreno=10,
        )


def test_el_motor_aborta_si_el_horizonte_no_cuadra(feats):
    """Si el objetivo esta a 1 semana pero decimos que evaluamos a 4, las
    metricas serian irreales de forma silenciosa."""
    corrupto = feats.copy()
    corrupto["fecha_objetivo"] = corrupto["fecha_origen"] + pd.Timedelta(weeks=1)
    with pytest.raises(backtest.FugaTemporalError, match="semanas del origen"):
        backtest.auditar_pit(corrupto, horizonte_semanas=4)
