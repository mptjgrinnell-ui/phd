import os
import pandas as pd
import yfinance as yf
import yaml


def read_universe(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    fixed = cfg["fixed"]
    eq_file = cfg["equities_file"]

    equities = []
    if eq_file and os.path.exists(eq_file):
        with open(eq_file, "r") as f:
            equities = [line.strip() for line in f if line.strip()]

    tickers = sorted(set(fixed + equities))
    return cfg, tickers


def download_one(ticker: str, start: str, end: str | None):
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,  # adjusted OHLC
        progress=False,
        threads=True,
    )
    if df is None or df.empty:
        return None
    # yfinance can return MultiIndex columns even for a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    # Standardize columns
    keep = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep if c in df.columns]]
    df["Ticker"] = ticker
    return df


def main(cfg_path="configs/universe.yaml", out_dir="data/raw/yahoo_parquet"):
    os.makedirs(out_dir, exist_ok=True)
    cfg, tickers = read_universe(cfg_path)

    start = cfg["start"]
    end = cfg.get("end", None)

    ok, bad = 0, 0
    for t in tickers:
        fp = os.path.join(out_dir, f"{t}.parquet")
        if os.path.exists(fp):
            continue
        df = download_one(t, start=start, end=end)
        if df is None:
            bad += 1
            continue
        df.to_parquet(fp, index=False)
        ok += 1

    print(f"Downloaded: {ok} | Failed: {bad} | Total: {len(tickers)}")


if __name__ == "__main__":
    main()
