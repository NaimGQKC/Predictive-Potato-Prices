#!/usr/bin/env python3
"""Ejecuta la fase entera y saca la tabla de resultados.

    python scripts/run_eval.py

Responde a una sola pregunta: cuanto hay que batir para que un modelo valga la
pena. Todo lo que imprime es sobre datos SINTETICOS: sirve para validar el
esqueleto, no para decidir compras.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from patata import config, pipeline, predictores  # noqa: E402


def _fmt(tabla: pd.DataFrame) -> str:
    return tabla.to_string(index=False, float_format=lambda v: f"{v:,.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(config.RUTA_DB), help="ruta del fichero DuckDB")
    ap.add_argument("--inicio", default=config.INICIO_BACKTEST, help="primer lunes evaluado")
    ap.add_argument("--min-entreno", type=int, default=config.MIN_EJEMPLOS_ENTRENO)
    ap.add_argument("--reentrenar-cada", type=int, default=1)
    ap.add_argument("--csv", default=None, help="directorio donde volcar los CSV")
    args = ap.parse_args()

    salida = pipeline.ejecutar(
        ruta_db=args.db,
        inicio=args.inicio,
        min_entreno=args.min_entreno,
        reentrenar_cada=args.reentrenar_cada,
    )
    res = salida["resultado"]
    feats = salida["features"]

    print("=" * 78)
    print("FASE 0 - ESQUELETO DE EVALUACION (datos sinteticos)")
    print("=" * 78)
    print(f"DB                     : {args.db}")
    print(f"Filas raw_lonja        : {salida['n_filas_raw']:,}")
    print(f"Filas features         : {len(feats):,}")
    print(
        f"Rango de origenes      : {feats['fecha_origen'].min().date()} "
        f"-> {feats['fecha_origen'].max().date()}"
    )
    print(f"Horizonte              : {res.horizonte_semanas} semanas")
    print(f"Coste almacenaje       : {res.coste_almacenaje:.3f} EUR/kg")
    print(f"Origenes evaluados     : {res.origenes_evaluados:,}")
    print(f"Origenes saltados      : {res.origenes_saltados:,}  {res.motivos_salto}")
    vivas = int(res.predicciones["precio_real"].isna().sum())
    print(f"Predicciones vivas     : {vivas}  (objetivo aun no publicado, no puntuan)")

    ejemplo = res.predicciones.iloc[len(res.predicciones) // 2]
    print("\n-- Auditoria point-in-time (un origen cualquiera) --------------------")
    fila = feats.loc[feats["fecha_origen"] == ejemplo["fecha_origen"]].iloc[0]
    print(f"  origen                 : {fila['fecha_origen'].date()}")
    print(f"  ultimo dato conocido   : {fila['fecha_dato_ultimo'].date()}"
          f"  ({fila['antiguedad_semanas']} semanas de antiguedad)")
    print(f"  max fecha_publicacion  : {fila['max_fecha_pub_usada'].date()}  (<= origen)")
    print(f"  semana objetivo        : {fila['fecha_objetivo'].date()}")
    print(f"  etiqueta publicada el  : {fila['fecha_pub_objetivo'].date()}")
    print(f"  ejemplos de entreno    : {ejemplo['n_entreno']:,}")

    print("\n-- Tabla de resultados -----------------------------------------------")
    print(_fmt(salida["resultados"]))

    print("\n-- Estabilidad por anyo ----------------------------------------------")
    cols = ["predictor", "anyo", "n", "mape_pct", "rmse_eur_kg",
            "direccion_mayoritaria", "tasa_base_comprar"]
    print(_fmt(salida["por_anyo"][cols]))

    print("\n-- Comparacion contra el baseline ------------------------------------")
    print(_fmt(salida["comparacion"]))

    base = salida["resultados"].set_index("predictor").loc[
        predictores.NaiveUltimoPrecio.nombre
    ]
    print("\n" + "=" * 78)
    print("EL LISTON A BATIR")
    print("=" * 78)
    print(f"  (a) regresion  : MAPE {base['mape_pct']:.2f}% | "
          f"RMSE {base['rmse_eur_kg']:.4f} EUR/kg | "
          f"MAE {base['mae_eur_kg']:.4f} EUR/kg")
    print(f"  direccion      : la naive no opina nunca (cobertura "
          f"{base['cobertura_direccion']:.0%}), asi que el\n"
          f"                   liston es apostar siempre a la direccion mas "
          f"frecuente: {base['direccion_mayoritaria']:.1%}.")
    print(f"  (b) decision   : la naive nunca dice 'compra ahora', asi que su "
          f"accuracy es\n"
          f"                   {base['clf_accuracy']:.1%} y su recall {base['clf_recall']:.1%}. "
          f"El liston real es la\n"
          f"                   clase mayoritaria: {base['acierto_clase_mayoritaria']:.1%} "
          f"(tasa base {base['tasa_base_comprar']:.1%}).")
    print("\n  Un modelo solo cuenta como mejora si baja MAPE y RMSE, sube la")
    print("  accuracy balanceada por encima de 0.5, y aguanta anyo a anyo.")
    print("  Recordatorio: esto son datos sinteticos. El numero real cambiara.")

    if args.csv:
        dest = Path(args.csv)
        dest.mkdir(parents=True, exist_ok=True)
        salida["resultados"].to_csv(dest / "resultados.csv", index=False)
        salida["por_anyo"].to_csv(dest / "resultados_por_anyo.csv", index=False)
        res.predicciones.to_csv(dest / "predicciones.csv", index=False)
        print(f"\nCSV escritos en {dest}")

    salida["con"].close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
