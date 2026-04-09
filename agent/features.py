"""

1. satisfaccion_index
   Combina rating_actual (0.6) y nps_score (0.4).
   Justificación: correlación 0.887 — miden lo mismo.
   Rating tiene mayor peso porque es lo que ve el usuario en la app.

2. delivery_vs_vertical / ticket_vs_vertical / cancel_vs_vertical
   Z-scores por vertical.
   Justificación: Bebidas tiene 29% críticos vs 9% en Mercado.
   Los umbrales absolutos sesgan hacia Comida (127/200 registros).

3. critico_rate_vertical
   Tasa histórica de críticos por vertical.
   Justificación: vertical es predictiva, no solo contexto.

4. var_ordenes_pct como tendencia
   Se usa la variación semana vs semana, no el absoluto.
   Justificación: el cambio importa más que el volumen base.
5. Remover emojis de semaforo_riesgo
    Se usa regex para la eliminacion de emojis, debido a que el emoji no es un caracter valido en el dataset.

Variables eliminadas por multicolinealidad (heatmap > 0.80):
   - delta_rating       (0.90 con rating_actual)
   - nps_score          (0.89 con rating_actual → va en satisfaccion_index)
   - quejas_7d          (0.87 con tasa_cancelacion_pct)
   - tiempo_entrega     (0.82 con tasa_cancelacion_pct)
   - ordenes_7d         (0.70 con var_ordenes_pct)

Feature set final:
   satisfaccion_index, tasa_cancelacion_pct, var_ordenes_pct,
   valor_ticket_prom_mxn, vertical_encoded, critico_rate_vertical
─────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import re

# ─── Helpers ──────────────────────────────────────────────────────────────────

def normalize_minmax(series: pd.Series) -> pd.Series:
    """Min-max normalization a [0, 1]. Mayor valor = mayor score."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.0, index=series.index)
    return (series - min_val) / (max_val - min_val)


def zscore_by_group(df: pd.DataFrame, col: str, group: str) -> pd.Series:
    """
    Z-score de una variable dentro de cada grupo.
    Convierte valores absolutos en desviaciones relativas al grupo.
    Útil para comparar verticales con comportamientos base distintos.
    """
    group_mean = df.groupby(group)[col].transform('mean')
    group_std  = df.groupby(group)[col].transform('std').replace(0, 1)
    return (df[col] - group_mean) / group_std


# ─── Features de satisfacción ─────────────────────────────────────────────────

def add_satisfaccion_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Índice combinado de satisfacción del usuario.


    NPS se normaliza de [-100, 100] a [0, 1] antes de combinar.
    """
    rating_norm = normalize_minmax(df['rating_actual'])
    nps_norm    = normalize_minmax(df['nps_score'] + 100)  # shift a [0, 200] → norm a [0,1]

    df['satisfaccion_index'] = 0.6 * rating_norm + 0.4 * nps_norm
    return df


# ─── Features relativizadas por vertical ──────────────────────────────────────

def add_vertical_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-scores por vertical para las variables con mayor varianza entre verticales.



    Variables relativizadas:
    - tasa_cancelacion_pct  → cancel_vs_vertical
    - valor_ticket_prom_mxn → ticket_vs_vertical
    - var_ordenes_pct       → ordenes_vs_vertical
    """
    df['cancel_vs_vertical'] = zscore_by_group(df, 'tasa_cancelacion_pct', 'vertical')
    df['ticket_vs_vertical']  = zscore_by_group(df, 'valor_ticket_prom_mxn', 'vertical')
    df['ordenes_vs_vertical'] = zscore_by_group(df, 'var_ordenes_pct', 'vertical')
    return df


def add_critico_rate_vertical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tasa histórica de restaurantes críticos por vertical.
    Le da al modelo el contexto de qué tan riesgosa es cada vertical.


    """
    if 'semaforo_riesgo' not in df.columns:
        df['critico_rate_vertical'] = 0.0
        return df

    tasa = df.groupby('vertical').apply(
        lambda x: (x['semaforo_riesgo'].str.upper() == 'CRÍTICO').mean()
    ).rename('critico_rate_vertical')

    df = df.merge(tasa, on='vertical', how='left')
    return df


# ─── Encoding de vertical ─────────────────────────────────────────────────────

def add_vertical_encoded(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encoding ordinal de vertical para el árbol de decisión.

    """
    enc = OrdinalEncoder(
        categories=[['Comida', 'Bebidas', 'Farmacia', 'Mercado']],
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )
    df['vertical_encoded'] = enc.fit_transform(df[['vertical']])
    return df


# ─── Antigüedad ───────────────────────────────────────────────────────────────

def add_antiguedad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Días en la plataforma desde activo_desde.

    """
    if 'activo_desde' not in df.columns:
        df['antiguedad_dias'] = np.nan
        return df

    df['activo_desde'] = pd.to_datetime(df['activo_desde'], errors='coerce')
    df['antiguedad_dias'] = (
        pd.Timestamp.today() - df['activo_desde']
    ).dt.days.abs()

    return df



# ─── Pipeline completo ────────────────────────────────────────────────────────

FEATURE_SET = [
    'satisfaccion_index',       # rating + NPS combinados
    'tasa_cancelacion_pct',     # fricción operativa (proxy de quejas y tiempo entrega)
    'var_ordenes_pct',          # tendencia de demanda
    'valor_ticket_prom_mxn',    # ticket promedio (correlación moderada, aporta señal única)
    'cancel_vs_vertical',       # cancelación relativizada por vertical
    'ticket_vs_vertical',       # ticket relativizado por vertical
    'ordenes_vs_vertical',      # volumen relativizado por vertical
    'critico_rate_vertical',    # contexto histórico de la vertical
    'vertical_encoded',         # identidad de la vertical
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completo de feature engineering.
    Aplica todas las transformaciones en orden y retorna el dataframe enriquecido.

    Uso:
        from agent.features import build_features, FEATURE_SET
        df = build_features(df)
        X = df[FEATURE_SET]
    """
    df = df.copy()

    df = add_satisfaccion_index(df)
    df = add_vertical_zscores(df)
    df = add_critico_rate_vertical(df)
    df = add_vertical_encoded(df)
    df = add_antiguedad(df)

    # Verificar que todas las features del set estén presentes
    missing = [f for f in FEATURE_SET if f not in df.columns]
    if missing:
        raise ValueError(f"Features faltantes después del pipeline: {missing}")

    return df


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna solo la matriz de features lista para el modelo.
    Sin NaNs — rellena con mediana por columna.
    """
    df = build_features(df)
    X = df[FEATURE_SET].copy()
    X = X.fillna(X.median())
    return X