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


pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)


def _fmt(tabla: pd.DataFrame) -> str:
    return tabla.to_string(index=False, float_format=lambda v: f"{v:,.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(config.RUTA_DB), help="ruta del fichero DuckDB")
    ap.add_argument("--inicio", default=config.INICIO_BACKTEST, help="primer lunes evaluado")
    ap.add_argument("--min-entreno", type=int, default=config.MIN_EJEMPLOS_ENTRENO)
    ap.add_argument("--reentrenar-cada", type=int, default=1)
    ap.add_argument("--csv", default=None, help="directorio donde volcar los CSV")
    ap.add_argument("--placebo", action="store_true",
                    help="baraja la variacion objetivo: todo debe caer a skill ~0")
    args = ap.parse_args()

    salida = pipeline.ejecutar(
        ruta_db=args.db,
        inicio=args.inicio,
        min_entreno=args.min_entreno,
        reentrenar_cada=args.reentrenar_cada,
        placebo=args.placebo,
    )
    res = salida["resultado"]
    feats = salida["features"]

    print("=" * 78)
    print("FASE 1 - BASELINE vs RETADORES (datos sinteticos)"
          + ("   [MODO PLACEBO]" if args.placebo else ""))
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

    print("\n-- (a) REGRESION: precio a 4 semanas ----------------------------------")
    cols_reg = ["predictor", "n", "mape_pct", "rmse_eur_kg", "mae_eur_kg",
                "acierto_direccion", "cobertura_direccion", "direccion_mayoritaria"]
    print(_fmt(salida["resultados"][cols_reg]))

    print("\n-- (b) DECISION: comprar ahora vs esperar -----------------------------")
    cols_clf = ["predictor", "n", "clf_accuracy", "clf_precision", "clf_recall",
                "clf_f1", "clf_acc_balanceada", "tasa_base_comprar",
                "acierto_clase_mayoritaria"]
    print(_fmt(salida["resultados"][cols_clf]))

    print("\n-- Comparacion contra el baseline naive -------------------------------")
    print(_fmt(salida["comparacion"]))
    print("  dm_t: test pareado sobre el error cuadratico (Newey-West).")
    print("  |dm_t| > 2 => la diferencia no cabe en el ruido. dm_t < 0 => mejor que la naive.")

    print("\n-- Estabilidad por anyo (MAPE) ----------------------------------------")
    pivote = salida["por_anyo"].pivot(index="anyo", columns="predictor", values="mape_pct")
    print(pivote.to_string(float_format=lambda v: f"{v:,.2f}"))

    print("\n-- Estabilidad por anyo (accuracy balanceada de la decision) ----------")
    pivote_b = salida["por_anyo"].pivot(index="anyo", columns="predictor",
                                        values="clf_acc_balanceada")
    print(pivote_b.to_string(float_format=lambda v: f"{v:,.3f}"))

    print("\n-- EN EUROS: coste medio de compra ------------------------------------")
    eco = salida["economia"]
    print(_fmt(eco[["estrategia", "n", "coste_medio_eur_kg", "pct_semanas_compra_ya",
                    "ahorro_eur_kg", "ahorro_anual_eur", "pct_del_ahorro_maximo"]]))
    print("  Supone 40.000 t/anyo, decisiones semanales independientes, sin")
    print("  restricciones de almacen ni contratos. Es un orden de magnitud.")

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
    if args.placebo:
        print("\n  MODO PLACEBO: la variacion objetivo esta barajada, no queda nada")
        print("  que predecir. Todos los skill_* tienen que salir <= 0 y las")
        print("  accuracy balanceadas ~0.5. Si algun modelo gana aqui, hay fuga.")
    print("\n  Un modelo solo cuenta como mejora si baja MAPE y RMSE, sube la")
    print("  accuracy balanceada por encima de 0.5, y aguanta anyo a anyo.")
    print("\n  AVISO: esto son datos SINTETICOS, generados con estacionalidad y")
    print("  reversion a la media conocidas. Que un modelo las encuentre prueba")
    print("  que el aparato detecta senyal cuando la hay; NO prueba que exista")
    print("  senyal en el mercado real. Ese numero solo sale con datos de lonja.")

    if args.csv:
        dest = Path(args.csv)
        dest.mkdir(parents=True, exist_ok=True)
        salida["resultados"].to_csv(dest / "resultados.csv", index=False)
        salida["por_anyo"].to_csv(dest / "resultados_por_anyo.csv", index=False)
        salida["comparacion"].to_csv(dest / "comparacion_baseline.csv", index=False)
        salida["economia"].to_csv(dest / "economia.csv", index=False)
        res.predicciones.to_csv(dest / "predicciones.csv", index=False)
        print(f"\nCSV escritos en {dest}")

    salida["con"].close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
