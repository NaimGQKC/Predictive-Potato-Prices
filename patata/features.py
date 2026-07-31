"""Construccion de la tabla de features con la regla de doble fecha.

El unico punto del proyecto donde se puede introducir fuga temporal es aqui.
Por eso la construccion es explicita y aburrida: para CADA lunes de origen se
saca un snapshot independiente de lo que era publicamente conocido ese lunes,
y la fila de features se calcula solo a partir de ese snapshot.

Es mas lento que calcular lags con un shift() sobre la serie entera. Es a
proposito: el shift() es exactamente el atajo que mete el dato de la semana T
en la prediccion de la semana T.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from patata import config, db

# --- columnas nucleo, declaradas en schema.sql -----------------------------
COLS_NUCLEO = (
    "fecha_origen",
    "fecha_asof",
    "max_fecha_pub_usada",
    "fecha_dato_ultimo",
    "precio_ref",
    "antiguedad_semanas",
    "n_obs_historicas",
    "semana_anyo",
    "mes",
    "sin_anual",
    "cos_anual",
    "fecha_objetivo",
    "precio_objetivo",
    "fecha_pub_objetivo",
    "etiqueta_comprar_ahora",
)

# columnas que NUNCA pueden entrar en un modelo: son objetivo, metadatos o
# identificadores. El motor de backtest las excluye de X.
COLS_NO_PREDICTORAS = (
    "fecha_origen",
    "fecha_asof",
    "max_fecha_pub_usada",
    "fecha_dato_ultimo",
    "fecha_objetivo",
    "precio_objetivo",
    "fecha_pub_objetivo",
    "etiqueta_comprar_ahora",
)


def snapshot(con, fecha_asof, fuente: str | None = None) -> pd.DataFrame:
    """Lo que era publicamente conocido el dia `fecha_asof`, ni un dato mas.

    Si una misma semana se ha publicado varias veces (revisiones), se queda la
    revision mas reciente ENTRE LAS VISIBLES a `fecha_asof`.
    """
    fecha_asof = pd.Timestamp(fecha_asof).date()
    filtro_fuente = "AND fuente = ?" if fuente else ""
    params: list = [fecha_asof] + ([fuente] if fuente else [])
    sql = f"""
        SELECT fecha_dato,
               arg_max(precio_eur_kg, fecha_publicacion) AS precio_eur_kg,
               max(fecha_publicacion)                    AS fecha_publicacion
        FROM raw_lonja
        WHERE fecha_publicacion <= ? {filtro_fuente}
        GROUP BY fecha_dato
        ORDER BY fecha_dato
    """
    df = con.execute(sql, params).df()
    if not df.empty:
        df["fecha_dato"] = pd.to_datetime(df["fecha_dato"])
        df["fecha_publicacion"] = pd.to_datetime(df["fecha_publicacion"])
    return df


def _fila_features(snap: pd.DataFrame, origen: pd.Timestamp) -> dict:
    """Features de un origen a partir de SU snapshot. Nada mas entra aqui."""
    fila: dict = {
        "fecha_origen": origen.date(),
        "fecha_asof": origen.date(),
        "semana_anyo": int(origen.isocalendar().week),
        "mes": int(origen.month),
        "sin_anual": float(np.sin(2 * np.pi * origen.dayofyear / 365.25)),
        "cos_anual": float(np.cos(2 * np.pi * origen.dayofyear / 365.25)),
        "n_obs_historicas": int(len(snap)),
    }

    if snap.empty:
        fila["max_fecha_pub_usada"] = None
        fila["fecha_dato_ultimo"] = None
        fila["precio_ref"] = np.nan
        fila["antiguedad_semanas"] = None
        for k in config.LAGS:
            fila[f"lag_{k}"] = np.nan
        for w in config.VENTANAS_MEDIA:
            fila[f"media_{w}"] = np.nan
            fila[f"desv_{w}"] = np.nan
        for k in (1, 4, 13, 52):
            fila[f"retorno_{k}"] = np.nan
        fila["pos_rango_52"] = np.nan
        fila["media_estacional"] = np.nan
        fila["desvio_estacional"] = np.nan
        return fila

    fila["max_fecha_pub_usada"] = snap["fecha_publicacion"].max().date()
    ultimo = snap.iloc[-1]
    f_ultimo = pd.Timestamp(ultimo["fecha_dato"])
    fila["fecha_dato_ultimo"] = f_ultimo.date()
    fila["precio_ref"] = float(ultimo["precio_eur_kg"])
    fila["antiguedad_semanas"] = int((origen - f_ultimo).days // 7)

    # lags alineados por CALENDARIO respecto al ultimo dato conocido: si falta
    # una semana el lag es NaN, no se cuela el vecino.
    por_fecha = dict(zip(snap["fecha_dato"], snap["precio_eur_kg"]))
    lags: dict[int, float] = {}
    for k in config.LAGS:
        lags[k] = float(por_fecha.get(f_ultimo - pd.Timedelta(weeks=k), np.nan))
        fila[f"lag_{k}"] = lags[k]

    serie = snap["precio_eur_kg"].to_numpy(dtype=float)
    for w in config.VENTANAS_MEDIA:
        corte = f_ultimo - pd.Timedelta(weeks=w - 1)
        ventana = snap.loc[snap["fecha_dato"] >= corte, "precio_eur_kg"].to_numpy(dtype=float)
        fila[f"media_{w}"] = float(ventana.mean()) if len(ventana) else np.nan
        fila[f"desv_{w}"] = float(ventana.std(ddof=1)) if len(ventana) > 1 else np.nan

    ref = fila["precio_ref"]
    for k in (1, 4, 13, 52):
        anterior = float(por_fecha.get(f_ultimo - pd.Timedelta(weeks=k), np.nan))
        fila[f"retorno_{k}"] = (ref / anterior - 1.0) if anterior and anterior > 0 else np.nan

    ult52 = serie[-52:]
    rango = ult52.max() - ult52.min()
    fila["pos_rango_52"] = float((ref - ult52.min()) / rango) if rango > 0 else np.nan

    # media historica de esta semana del anyo, calculada SOLO con el snapshot
    misma_semana = snap.loc[
        snap["fecha_dato"].dt.isocalendar().week == fila["semana_anyo"], "precio_eur_kg"
    ].to_numpy(dtype=float)
    if len(misma_semana):
        fila["media_estacional"] = float(misma_semana.mean())
        media_global = float(serie.mean())
        fila["desvio_estacional"] = fila["media_estacional"] - media_global
    else:
        fila["media_estacional"] = np.nan
        fila["desvio_estacional"] = np.nan

    return fila


def _tabla_objetivos(con, fuente: str | None = None) -> pd.DataFrame:
    """Valor y fecha de publicacion de cada semana objetivo.

    Se usa la PRIMERA publicacion de cada semana: es el valor que habrias
    tenido al entrenar en tiempo real, y `fecha_pub_objetivo` es el instante
    exacto en que la etiqueta paso a ser utilizable.
    """
    filtro = "WHERE fuente = ?" if fuente else ""
    params = [fuente] if fuente else []
    df = con.execute(
        f"""
        SELECT fecha_dato                                   AS fecha_objetivo,
               arg_min(precio_eur_kg, fecha_publicacion)    AS precio_objetivo,
               min(fecha_publicacion)                       AS fecha_pub_objetivo
        FROM raw_lonja {filtro}
        GROUP BY fecha_dato
        """,
        params,
    ).df()
    if not df.empty:
        df["fecha_objetivo"] = pd.to_datetime(df["fecha_objetivo"])
        df["fecha_pub_objetivo"] = pd.to_datetime(df["fecha_pub_objetivo"])
    return df


def construir_features(
    con,
    fuente: str | None = None,
    horizonte_semanas: int = config.HORIZONTE_SEMANAS,
    coste_almacenaje: float = config.COSTE_ALMACENAJE_EUR_KG,
    origenes: pd.DatetimeIndex | None = None,
    persistir: bool = True,
) -> pd.DataFrame:
    """Tabla de features semanal, una fila por lunes de origen."""
    rango = con.execute(
        "SELECT min(fecha_dato), max(fecha_dato) FROM raw_lonja"
        + (" WHERE fuente = ?" if fuente else ""),
        [fuente] if fuente else [],
    ).fetchone()
    if rango is None or rango[0] is None:
        raise ValueError("raw_lonja esta vacia: carga alguna fuente antes")

    if origenes is None:
        origenes = pd.date_range(pd.Timestamp(rango[0]), pd.Timestamp(rango[1]), freq="W-MON")

    filas = [_fila_features(snapshot(con, o, fuente=fuente), o) for o in origenes]
    feats = pd.DataFrame(filas)

    # --- objetivos ---------------------------------------------------------
    feats["fecha_origen"] = pd.to_datetime(feats["fecha_origen"])
    feats["fecha_objetivo"] = feats["fecha_origen"] + pd.Timedelta(weeks=horizonte_semanas)
    objetivos = _tabla_objetivos(con, fuente=fuente)
    feats = feats.merge(objetivos, on="fecha_objetivo", how="left")

    # Etiqueta de la decision real: 1 = "compra ahora", porque dentro de 4
    # semanas el precio superara el de hoy mas el coste de almacenaje.
    # La referencia es precio_ref (ultimo precio PUBLICADO en el origen), no el
    # precio verdadero de la semana del origen: ese todavia no lo conoces.
    umbral = feats["precio_ref"] + coste_almacenaje
    etiqueta = feats["precio_objetivo"] > umbral
    feats["etiqueta_comprar_ahora"] = etiqueta.where(
        feats["precio_objetivo"].notna() & feats["precio_ref"].notna()
    )

    for col in ("fecha_origen", "fecha_asof", "fecha_objetivo", "fecha_dato_ultimo",
                "max_fecha_pub_usada", "fecha_pub_objetivo"):
        feats[col] = pd.to_datetime(feats[col])

    feats = feats.sort_values("fecha_origen").reset_index(drop=True)

    if persistir:
        _asegurar_columnas(con, feats)
        db.guardar_df(con, feats, "features_semanal", reemplazar=True)
    return feats


def _asegurar_columnas(con, feats: pd.DataFrame) -> None:
    """Anade a features_semanal las columnas derivadas que no estan en el DDL.

    El DDL declara el nucleo auditable; los lags y ventanas son un parametro de
    config, asi que se materializan aqui en vez de duplicarlos en SQL.
    """
    existentes = {r[0] for r in con.execute("DESCRIBE features_semanal").fetchall()}
    for col in feats.columns:
        if col in existentes:
            continue
        tipo = "BOOLEAN" if feats[col].dtype == bool else "DOUBLE"
        con.execute(f'ALTER TABLE features_semanal ADD COLUMN "{col}" {tipo}')


def columnas_predictoras(feats: pd.DataFrame) -> list[str]:
    """Columnas que un modelo puede usar como X."""
    return [
        c
        for c in feats.columns
        if c not in COLS_NO_PREDICTORAS and pd.api.types.is_numeric_dtype(feats[c])
    ]
