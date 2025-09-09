from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

try:
    from pandas_datareader import data as pdr
except Exception:  # pragma: no cover
    pdr = None  # fallback handled below


def _synthetic_macro(start: str, end: str, freq: str = "MS") -> pd.DataFrame:
    rng = pd.date_range(start=start, end=end, freq=freq)
    n = len(rng)
    rng_seed = abs(hash((start, end, freq))) % (2**32 - 1)
    rs = np.random.default_rng(rng_seed)

    def ar1(mu: float, phi: float, sigma: float, init: float) -> np.ndarray:
        x = np.empty(n, dtype=float)
        x[0] = init
        for t in range(1, n):
            x[t] = mu + phi * (x[t - 1] - mu) + rs.normal(0.0, sigma)
        return x

    unemployment = np.clip(ar1(5.5, 0.9, 0.15, 5.5), 2.5, 15.0)
    cpi_yoy = ar1(2.5, 0.7, 0.2, 2.5)
    gdp_growth_qoq_ann = ar1(2.0, 0.6, 0.8, 2.0)
    fed_funds = np.clip(ar1(2.0, 0.8, 0.25, 2.0), 0.0, 8.0)
    treasury_10y = np.clip(fed_funds + rs.normal(1.2, 0.2, n), 0.5, 10.0)
    credit_spread_bbb = np.clip(ar1(2.0, 0.85, 0.25, 2.0), 0.5, 8.0)
    hpi_yoy = ar1(3.0, 0.7, 0.6, 3.0)

    df = pd.DataFrame(
        {
            "asof_month": rng,
            "unemployment": unemployment,
            "cpi_yoy": cpi_yoy,
            "gdp_growth_qoq_ann": gdp_growth_qoq_ann,
            "fed_funds": fed_funds,
            "treasury_10y": treasury_10y,
            "credit_spread_bbb": credit_spread_bbb,
            "hpi_yoy": hpi_yoy,
        }
    )
    return df


def get_macro_data(
    start: str,
    end: str,
    freq: str = "MS",
    fred_api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Return macro DataFrame between start and end inclusive.

    Tries FRED first (if available), otherwise generates a synthetic but realistic series.
    """
    if pdr is None:
        return _synthetic_macro(start, end, freq)

    fred_api_key = fred_api_key or os.environ.get("FRED_API_KEY")
    symbols = {
        "unemployment": "UNRATE",
        "cpi": "CPIAUCSL",
        "fed_funds": "FEDFUNDS",
        "treasury_10y": "DGS10",
        "credit_spread_bbb": "BAMLC0A4CBBB",
        "hpi": "USSTHPI",
    }

    try:
        params = {}
        if fred_api_key:
            params["api_key"] = fred_api_key

        def fred(sym: str) -> pd.Series:
            s = pdr.DataReader(sym, "fred", start=start, end=end, api_key=fred_api_key)
            s = s.resample(freq).last()
            s.name = sym
            return s

        unrate = fred(symbols["unemployment"]).rename("unemployment")
        cpi = fred(symbols["cpi"])  # level
        cpi_yoy = (cpi.pct_change(12) * 100.0).rename("cpi_yoy")
        fed = fred(symbols["fed_funds"]).rename("fed_funds")
        t10 = fred(symbols["treasury_10y"]).rename("treasury_10y")
        bbb = fred(symbols["credit_spread_bbb"]).rename("credit_spread_bbb")
        hpi = fred(symbols["hpi"])  # level index
        hpi_yoy = (hpi.pct_change(12) * 100.0).rename("hpi_yoy")

        gdp_growth_qoq_ann = (
            3.0
            - 0.3 * (unrate - unrate.rolling(12).mean().fillna(unrate))
            - 0.1 * (fed - fed.rolling(6).mean().fillna(fed))
        ).rename("gdp_growth_qoq_ann")

        df = (
            pd.concat(
                [unrate, cpi_yoy, gdp_growth_qoq_ann, fed, t10, bbb, hpi_yoy], axis=1
            )
            .reset_index()
            .rename(columns={"index": "asof_month"})
        )
        df["asof_month"] = (
            pd.to_datetime(df["asof_month"]).dt.to_period("M").dt.to_timestamp()
        )
        return df.dropna().reset_index(drop=True)
    except Exception:
        return _synthetic_macro(start, end, freq)
