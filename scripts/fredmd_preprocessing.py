"""Reproduce the bundled McScan FRED-MD preprocessing in Python.

The implementation follows ``fbi::fredmd`` for transformation codes and
``FRED-MD data/fredmd_clean.R`` for balancing, standardisation, truncation,
and the six-factor augmented regression design.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _transform_series(values: np.ndarray, code: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = values.size
    result = np.full(n, np.nan)
    if code == 1:
        return values.copy()
    if code == 2:
        result[1:] = values[1:] - values[:-1]
    elif code == 3:
        result[2:] = values[2:] - 2.0 * values[1:-1] + values[:-2]
    elif code in {4, 5, 6}:
        if np.nanmin(values) > 1e-6:
            logged = np.log(values)
            if code == 4:
                result = logged
            elif code == 5:
                result[1:] = logged[1:] - logged[:-1]
            else:
                result[2:] = logged[2:] - 2.0 * logged[1:-1] + logged[:-2]
    elif code == 7:
        first = np.full(n, np.nan)
        first[1:] = (values[1:] - values[:-1]) / values[:-1]
        result[2:] = first[2:] - first[1:-1]
    else:
        raise ValueError(f"Unknown FRED-MD transformation code: {code}")
    return result


def load_transformed_fredmd(csv_path: str | Path) -> pd.DataFrame:
    """Load an official FRED-MD CSV and apply its transformation-code row."""
    csv_path = Path(csv_path)
    attributes = pd.read_csv(csv_path, header=None, nrows=2)
    names = ["date", *attributes.iloc[0, 1:].astype(str).tolist()]
    codes = attributes.iloc[1, 1:].astype(int).to_numpy()
    raw = pd.read_csv(csv_path, skiprows=2, header=None, names=names)
    raw["date"] = pd.to_datetime(raw["date"])
    transformed_columns = {"date": raw["date"]}
    for name, code in zip(names[1:], codes):
        transformed_columns[name] = _transform_series(raw[name].to_numpy(), int(code))
    return pd.DataFrame(transformed_columns)


def mcscan_design_from_csv(
    csv_path: str | Path,
    *,
    first_balanced_date: str = "1960-01",
    factors: int = 6,
    truncation_quantile: float = 0.999,
) -> dict:
    """Construct McScan's factor-augmented one-step prediction data."""
    transformed = load_transformed_fredmd(csv_path)
    data = transformed.loc[
        transformed["date"] >= pd.Timestamp(first_balanced_date)
    ].reset_index(drop=True)

    # Match fredmd_clean.R: retain columns observed in the first selected row,
    # then retain only columns with no missing observations over the panel.
    value_columns = [name for name in data.columns[1:] if pd.notna(data.loc[0, name])]
    value_columns = [name for name in value_columns if data[name].notna().all()]
    value_columns = [name for name in value_columns if name not in {"NONBORRES", "M1SL"}]
    if "INDPRO" not in value_columns:
        raise ValueError("The balanced FRED-MD panel does not contain INDPRO")

    x0 = data[value_columns].to_numpy(dtype=float).T
    centered = x0 - x0.mean(axis=1, keepdims=True)
    scaled = centered / centered.std(axis=1, ddof=1, keepdims=True)
    tau = float(np.quantile(np.abs(scaled), truncation_quantile))
    truncated = np.clip(scaled, -tau, tau)

    covariance = truncated @ truncated.T / truncated.shape[1]
    u, _, _ = np.linalg.svd(covariance, full_matrices=False)
    loadings = u[:, :factors]
    factor_scores = loadings.T @ truncated / np.sqrt(truncated.shape[0])
    idiosyncratic = truncated - loadings @ loadings.T @ truncated

    indpro_index = value_columns.index("INDPRO")
    response = truncated[indpro_index, 1:]
    idio_without_response = np.delete(idiosyncratic, indpro_index, axis=0)
    design = np.column_stack((factor_scores.T, idio_without_response.T))[:-1]
    dates = data["date"].dt.strftime("%Y-%m").to_numpy()[:-1]

    return {
        "X": design,
        "y": response,
        "date": dates,
        "series": value_columns,
        "tau": tau,
        "raw_balanced_end": data["date"].dt.strftime("%Y-%m").iloc[-1],
    }
