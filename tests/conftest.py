from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from patata import db, features  # noqa: E402
from patata.loaders import falso  # noqa: E402

N_SEMANAS_TEST = 300


@pytest.fixture()
def con():
    """Base en memoria, recien cargada con la serie sintetica corta."""
    conexion = db.conectar(":memory:")
    falso.cargar(conexion, n_semanas=N_SEMANAS_TEST)
    yield conexion
    conexion.close()


@pytest.fixture()
def feats(con):
    return features.construir_features(con, fuente=falso.FUENTE)
