-- Esquema del almacen. Un solo fichero DuckDB.
--
-- REGLA INNEGOCIABLE: toda tabla `raw_*` lleva DOS fechas.
--   fecha_dato        -> a que semana se refiere el valor (lunes de esa semana)
--   fecha_publicacion -> cuando ese valor estuvo disponible publicamente
-- Ninguna consulta de entrenamiento puede mirar filas con
-- fecha_publicacion > lunes de la semana que se esta prediciendo.

-- Catalogo de fuentes. No es una fuente de datos, es el registro de las que
-- habra. Sirve para que el retardo de publicacion tipico este documentado.
CREATE TABLE IF NOT EXISTS fuentes (
    fuente              VARCHAR PRIMARY KEY,
    descripcion         VARCHAR NOT NULL,
    -- retardo tipico entre fecha_dato y fecha_publicacion, en dias
    retardo_dias_tipico INTEGER NOT NULL,
    es_sintetica        BOOLEAN NOT NULL DEFAULT FALSE
);

-- Fuente 1: precio semanal de lonja.
CREATE TABLE IF NOT EXISTS raw_lonja (
    fuente            VARCHAR  NOT NULL,
    fecha_dato        DATE     NOT NULL,   -- lunes de la semana de referencia
    fecha_publicacion DATE     NOT NULL,   -- cuando se pudo leer
    variedad          VARCHAR  NOT NULL,
    precio_eur_kg     DOUBLE   NOT NULL,
    unidad            VARCHAR  NOT NULL DEFAULT 'EUR/kg',
    ingesta_ts        TIMESTAMP NOT NULL DEFAULT now(),
    PRIMARY KEY (fuente, fecha_dato, variedad, fecha_publicacion)
);

-- Tabla de features semanal, ya materializada point-in-time.
-- Una fila por ORIGEN de prediccion (el lunes en el que decidimos).
-- `fecha_asof` es el corte de publicacion usado para construir la fila: por
-- construccion fecha_asof = fecha_origen, y ninguna columna de esta fila depende
-- de informacion publicada despues de fecha_asof.
CREATE TABLE IF NOT EXISTS features_semanal (
    fecha_origen            DATE   NOT NULL PRIMARY KEY,
    fecha_asof              DATE   NOT NULL,
    -- maxima fecha_publicacion realmente consumida al construir la fila
    max_fecha_pub_usada     DATE,
    -- fecha_dato del ultimo precio conocido en el momento del origen
    fecha_dato_ultimo       DATE,
    -- precio de referencia de la decision: el ultimo precio publicado
    precio_ref              DOUBLE,
    -- semanas transcurridas entre ese ultimo dato y el origen
    antiguedad_semanas      INTEGER,
    n_obs_historicas        INTEGER,
    semana_anyo             INTEGER,
    mes                     INTEGER,
    sin_anual               DOUBLE,
    cos_anual               DOUBLE,
    -- objetivos (se rellenan solo cuando ya son observables; ver
    -- fecha_pub_objetivo para saber CUANDO pasaron a serlo)
    fecha_objetivo          DATE,
    precio_objetivo         DOUBLE,
    fecha_pub_objetivo      DATE,
    etiqueta_comprar_ahora  BOOLEAN
);

-- Predicciones crudas del walk-forward, una fila por (predictor, origen).
CREATE TABLE IF NOT EXISTS backtest_predicciones (
    predictor            VARCHAR NOT NULL,
    fecha_origen         DATE    NOT NULL,
    fecha_objetivo       DATE    NOT NULL,
    n_entreno            INTEGER NOT NULL,
    precio_ref           DOUBLE,
    precio_pred          DOUBLE,
    precio_real          DOUBLE,
    prob_comprar_ahora   DOUBLE,
    pred_comprar_ahora   BOOLEAN,
    real_comprar_ahora   BOOLEAN,
    PRIMARY KEY (predictor, fecha_origen)
);

-- Metricas agregadas por predictor.
CREATE TABLE IF NOT EXISTS backtest_metricas (
    predictor VARCHAR NOT NULL,
    metrica   VARCHAR NOT NULL,
    valor     DOUBLE,
    PRIMARY KEY (predictor, metrica)
);
