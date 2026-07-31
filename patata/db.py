"""Acceso a DuckDB. Un solo fichero .db para todo el proyecto."""

from __future__ import annotations

from pathlib import Path

import duckdb

from patata import config

RUTA_SCHEMA = Path(__file__).resolve().parent / "schema.sql"


def conectar(ruta: str | Path | None = None, solo_lectura: bool = False):
    """Abre la base y garantiza que el esquema existe.

    `ruta=':memory:'` es valido y es lo que usan los tests.
    """
    if ruta is None:
        ruta = config.RUTA_DB
    if str(ruta) != ":memory:":
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(ruta), read_only=solo_lectura)
    if not solo_lectura:
        crear_esquema(con)
    return con


def crear_esquema(con) -> None:
    con.execute(RUTA_SCHEMA.read_text(encoding="utf-8"))


def vaciar_tablas(con, tablas: tuple[str, ...]) -> None:
    for t in tablas:
        con.execute(f"DELETE FROM {t}")


def guardar_df(con, df, tabla: str, reemplazar: bool = True) -> None:
    """Inserta un DataFrame en una tabla existente, casando por nombre de columna.

    Las columnas de la tabla que no esten en el DataFrame se dejan a su valor
    por defecto. Las columnas del DataFrame que no esten en la tabla son un
    error: casi siempre significan una falta de ortografia.
    """
    cols_tabla = [r[0] for r in con.execute(f"DESCRIBE {tabla}").fetchall()]
    sobran = [c for c in df.columns if c not in cols_tabla]
    if sobran:
        raise ValueError(f"columnas desconocidas para {tabla}: {sobran}")
    if reemplazar:
        con.execute(f"DELETE FROM {tabla}")
    cols = [c for c in cols_tabla if c in df.columns]
    lista = ", ".join(f'"{c}"' for c in cols)
    con.register("_tmp_guardar_df", df[cols])
    con.execute(f"INSERT INTO {tabla} ({lista}) SELECT {lista} FROM _tmp_guardar_df")
    con.unregister("_tmp_guardar_df")
