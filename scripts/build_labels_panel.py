import os
import glob
import numpy as np
import pandas as pd

RAW_DIR = "data/raw/yahoo_parquet"
OUT = "data/processed/panel_minimal.parquet"


def main():
    os.makedirs("data/processed", exist_ok=True)

    files = glob.glob(os.path.join(RAW_DIR, "*.parquet"))
    panels = []

    for fp in files:
        df = pd.read_parquet(fp)
        if df.empty:
            continue
        df = df.sort_values("Date")
        c = df["Close"].astype(float).values
        r1 = np.log(c[1:] / c[:-1])
        out = df.iloc[:-1].copy()
        out["r_t1"] = r1
        panels.append(out[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume", "r_t1"]])

    panel = pd.concat(panels, ignore_index=True)
    panel = panel.dropna().sort_values(["Date", "Ticker"])
    panel.to_parquet(OUT, index=False)
    print(f"Wrote {len(panel):,} rows -> {OUT}")


if __name__ == "__main__":
    main()
