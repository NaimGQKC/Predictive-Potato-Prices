"""Metricas de evaluacion. Regresion, direccion y clasificacion.

Nada aqui sabe de modelos: entra numpy, sale un diccionario.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _limpiar(*arrays):
    arrs = [np.asarray(a, dtype=float) for a in arrays]
    valido = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        valido &= np.isfinite(a)
    return [a[valido] for a in arrs], int(valido.sum())


# --- regresion -------------------------------------------------------------

def mape(y_real, y_pred) -> float:
    """Error porcentual absoluto medio, en %. Ignora reales <= 0."""
    (yr, yp), _ = _limpiar(y_real, y_pred)
    ok = yr > 0
    if ok.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((yr[ok] - yp[ok]) / yr[ok])) * 100.0)


def rmse(y_real, y_pred) -> float:
    (yr, yp), n = _limpiar(y_real, y_pred)
    if n == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yr - yp) ** 2)))


def mae(y_real, y_pred) -> float:
    (yr, yp), n = _limpiar(y_real, y_pred)
    if n == 0:
        return float("nan")
    return float(np.mean(np.abs(yr - yp)))


def direccion(y_real, y_pred, referencia, tolerancia: float = 0.0) -> dict[str, float]:
    """Acierto de direccion (subio / bajo) respecto al precio de referencia.

    Matiz que importa: la naive predice exactamente el precio de referencia, es
    decir NO OPINA sobre la direccion. Contarle esas semanas como fallo daria
    un 0% enganyoso, y contarlas como acierto cuando el precio no se mueve daria
    un numero igual de inutil. Asi que el acierto se calcula solo sobre las
    semanas en las que el predictor SI se moja, y se reporta aparte la
    `cobertura` (que fraccion de semanas se moja) y `mayoritaria` (que sacaria
    quien apostase siempre a la direccion mas frecuente).

    Para la naive: cobertura 0 y acierto NaN. El liston en direccion es, por
    tanto, `mayoritaria`.
    """
    (yr, yp, ref), n = _limpiar(y_real, y_pred, referencia)
    if n == 0:
        return {"acierto_direccion": float("nan"), "cobertura_direccion": float("nan"),
                "direccion_mayoritaria": float("nan")}

    mov_real = yr - ref
    mov_pred = yp - ref
    signo_real = np.sign(np.where(np.abs(mov_real) <= tolerancia, 0.0, mov_real))
    signo_pred = np.sign(np.where(np.abs(mov_pred) <= tolerancia, 0.0, mov_pred))

    opina = signo_pred != 0
    acierto = float(np.mean(signo_real[opina] == signo_pred[opina])) if opina.any() else float("nan")
    subidas = float(np.mean(signo_real > 0))
    return {
        "acierto_direccion": acierto,
        "cobertura_direccion": float(np.mean(opina)),
        "direccion_mayoritaria": max(subidas, 1.0 - subidas),
    }


def acierto_direccion(y_real, y_pred, referencia, tolerancia: float = 0.0) -> float:
    """Atajo: solo el acierto de direccion. Ver `direccion` para el matiz."""
    return direccion(y_real, y_pred, referencia, tolerancia)["acierto_direccion"]


def metricas_regresion(y_real, y_pred, referencia) -> dict[str, float]:
    (_yr, _yp, _r), n = _limpiar(y_real, y_pred, referencia)
    return {
        "n": float(n),
        "mape_pct": mape(y_real, y_pred),
        "rmse_eur_kg": rmse(y_real, y_pred),
        "mae_eur_kg": mae(y_real, y_pred),
        **direccion(y_real, y_pred, referencia),
    }


# --- clasificacion ---------------------------------------------------------

def metricas_clasificacion(y_real, y_pred) -> dict[str, float]:
    """Metricas de la decision binaria "comprar ahora".

    `tasa_base` y `acierto_clase_mayoritaria` estan aqui a proposito: son la
    referencia honesta contra la que comparar la accuracy.
    """
    yr = pd.Series(y_real).astype("boolean")
    yp = pd.Series(y_pred).astype("boolean")
    valido = yr.notna() & yp.notna()
    yr = yr[valido].to_numpy(dtype=bool)
    yp = yp[valido].to_numpy(dtype=bool)
    n = len(yr)
    if n == 0:
        return {k: float("nan") for k in
                ("n", "accuracy", "precision", "recall", "f1", "accuracy_balanceada",
                 "tasa_base", "acierto_clase_mayoritaria", "vp", "fp", "vn", "fn")}

    vp = int(np.sum(yr & yp))
    fp = int(np.sum(~yr & yp))
    vn = int(np.sum(~yr & ~yp))
    fn = int(np.sum(yr & ~yp))

    precision = vp / (vp + fp) if (vp + fp) else float("nan")
    recall = vp / (vp + fn) if (vp + fn) else float("nan")
    especificidad = vn / (vn + fp) if (vn + fp) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and np.isfinite(precision) and np.isfinite(recall)
        else 0.0 if (vp + fp + fn) else float("nan")
    )
    tasa_base = float(np.mean(yr))
    return {
        "n": float(n),
        "accuracy": (vp + vn) / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy_balanceada": float(np.nanmean([recall, especificidad])),
        "tasa_base": tasa_base,
        "acierto_clase_mayoritaria": max(tasa_base, 1 - tasa_base),
        "vp": float(vp),
        "fp": float(fp),
        "vn": float(vn),
        "fn": float(fn),
    }


# --- comparacion contra el baseline ---------------------------------------

def skill_score(valor_modelo: float, valor_baseline: float, menor_es_mejor: bool = True) -> float:
    """Mejora relativa sobre el baseline. >0 significa batirlo.

    Para errores (MAPE, RMSE): 1 - modelo/baseline.
    Para aciertos: modelo/baseline - 1.
    """
    if not np.isfinite(valor_modelo) or not np.isfinite(valor_baseline) or valor_baseline == 0:
        return float("nan")
    if menor_es_mejor:
        return float(1.0 - valor_modelo / valor_baseline)
    return float(valor_modelo / valor_baseline - 1.0)


def diebold_mariano_simple(errores_a, errores_b) -> dict[str, float]:
    """Test pareado sobre las diferencias de error cuadratico.

    Version deliberadamente simple (t sobre la media de la diferencia, con
    correccion HAC de Newey-West para la autocorrelacion que introduce el
    solape de horizontes). No es el DM completo, pero evita cantar victoria
    con una diferencia que cabe dentro del ruido.
    """
    (ea, eb), n = _limpiar(errores_a, errores_b)
    if n < 30:
        return {"n": float(n), "dif_media": float("nan"), "t": float("nan")}
    d = ea**2 - eb**2
    dbar = float(np.mean(d))
    # Newey-West con retardo = horizonte - 1 (solape de 3 semanas)
    dc = d - dbar
    gamma0 = float(np.mean(dc**2))
    var = gamma0
    for lag in range(1, 4):
        g = float(np.mean(dc[lag:] * dc[:-lag]))
        var += 2 * (1 - lag / 4) * g
    var = max(var, 1e-18)
    t = dbar / np.sqrt(var / n)
    return {"n": float(n), "dif_media": dbar, "t": float(t)}
