import os
import pandas as pd
import numpy as np
import yfinance as yf

WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def normalize_yahoo(t: str) -> str:
    # Yahoo uses BRK-B not BRK.B
    return t.replace(".", "-").strip()


def load_sp500_tickers() -> list[str]:
    # Add UA to avoid 403 from Wikipedia
    tables = pd.read_html(WIKI_SP500, storage_options={"User-Agent": "Mozilla/5.0"})
    # first table is constituents
    df = tables[0]
    tickers = [normalize_yahoo(x) for x in df["Symbol"].tolist()]
    return sorted(set(tickers))


def median_dollar_volume(ticker: str, lookback_days: int = 90) -> float:
    # Use last ~90 calendar days of daily bars
    try:
        df = yf.download(
            ticker,
            period=f"{lookback_days}d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if df is None or df.empty:
            return np.nan
        # require Close and Volume
        if "Close" not in df.columns or "Volume" not in df.columns:
            return np.nan
        dv = (df["Close"] * df["Volume"]).dropna()
        if len(dv) < 20:
            return np.nan
        return float(np.median(dv.values))
    except Exception:
        return np.nan


def main(out_path="data/meta/equities_top200.txt", n: int = 200):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tickers = load_sp500_tickers()
    rows = []
    for t in tickers:
        dv = median_dollar_volume(t, lookback_days=120)
        rows.append((t, dv))
    df = pd.DataFrame(rows, columns=["ticker", "median_dollar_vol"]).dropna()
    df = df.sort_values("median_dollar_vol", ascending=False).head(n)

    df["ticker"].to_csv(out_path, index=False, header=False)
    print(f"Wrote {len(df)} tickers -> {out_path}")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
