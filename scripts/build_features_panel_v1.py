import os
from functools import reduce
import numpy as np
import pandas as pd
import yaml


def safe_div(a, b):
    return np.where(b == 0, np.nan, a / b)


def garman_klass_var(o, h, l, c):
    # Classic GK variance estimator (daily)
    lo = np.log(o)
    lh = np.log(h)
    ll = np.log(l)
    lc = np.log(c)
    term1 = 0.5 * (lh - ll) ** 2
    term2 = (2 * np.log(2) - 1) * (lc - lo) ** 2
    return term1 - term2


def rogers_satchell_var(o, h, l, c):
    lo = np.log(o)
    lh = np.log(h)
    ll = np.log(l)
    lc = np.log(c)
    return (lh - lc) * (lh - lo) + (ll - lc) * (ll - lo)


def build_features_for_ticker(df: pd.DataFrame, lookbacks: dict) -> pd.DataFrame:
    df = df.sort_values("Date").copy()
    o = df["Open"].astype(float).values
    h = df["High"].astype(float).values
    l = df["Low"].astype(float).values
    c = df["Close"].astype(float).values
    v = df["Volume"].astype(float).values

    # Basic logs
    logc = np.log(c)
    logv = np.log(np.maximum(v, 1.0))
    rng = np.log(safe_div(h, l))  # log(H/L)

    # Candle geometry
    hl = (h - l)
    clv = safe_div((c - l), hl)  # close location in range [0,1]-ish
    body = (c - o)
    body_pct = safe_div(body, c)
    gap = np.log(safe_div(o, np.r_[np.nan, c[:-1]]))  # log(O_t / C_{t-1})

    out = df[["Date", "Ticker", "r_t1"]].copy()

    # Returns over multiple horizons (close-to-close, log)
    for k in lookbacks["returns"]:
        ret_k = np.full_like(logc, np.nan, dtype=float)
        if k < len(logc):
            ret_k[k:] = logc[k:] - logc[:-k]
        out[f"ret_{k}"] = ret_k

    # Range/vol estimators
    gk = garman_klass_var(o, h, l, c)
    rs = rogers_satchell_var(o, h, l, c)
    out["gk_var_1"] = gk
    out["rs_var_1"] = rs
    out["range_log_hl"] = rng
    out["gap_log_oc1"] = gap
    out["clv"] = clv
    out["body_pct"] = body_pct
    out["vol_log"] = logv

    # Rolling vol-of-vol / range moments
    for w in lookbacks["vol_windows"]:
        out[f"gk_vol_{w}"] = pd.Series(gk).rolling(w).mean().to_numpy()
        out[f"rs_vol_{w}"] = pd.Series(rs).rolling(w).mean().to_numpy()
        out[f"range_mean_{w}"] = pd.Series(rng).rolling(w).mean().to_numpy()
        out[f"range_std_{w}"] = pd.Series(rng).rolling(w).std().to_numpy()
        out[f"absret_mean_{w}"] = pd.Series(np.abs(out["ret_1"])).rolling(w).mean().to_numpy()

    # Volume shocks (z-score like)
    for w in lookbacks["z_windows"]:
        m = pd.Series(logv).rolling(w).mean()
        s = pd.Series(logv).rolling(w).std()
        out[f"vol_z_{w}"] = ((pd.Series(logv) - m) / s).to_numpy()

    # Liquidity proxy (Amihud-ish): |ret1| / dollar volume
    dollar_vol = c * v
    out["illq_amihud"] = np.abs(out["ret_1"].to_numpy()) / np.maximum(dollar_vol, 1.0)

    return out


def main(cfg_path="configs/features.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    panel_in = cfg["panel_in"]
    panel_out = cfg["panel_out"]
    lookbacks = cfg["lookbacks"]
    os.makedirs(os.path.dirname(panel_out), exist_ok=True)

    panel = pd.read_parquet(panel_in)
    panel = panel.sort_values(["Ticker", "Date"]).drop_duplicates(["Ticker", "Date"])

    feats = []
    for t, g in panel.groupby("Ticker", sort=False):
        feats.append(build_features_for_ticker(g, lookbacks))
    feat_panel = pd.concat(feats, ignore_index=True).sort_values(["Date", "Ticker"])

    # Broadcast context: join selected tickers' features (excluding their targets) to all tickers by Date.
    bctx = cfg.get("broadcast_context", [])
    if bctx:
        ctx_frames = []
        for t in bctx:
            sub = feat_panel[feat_panel["Ticker"] == t].drop(columns=["Ticker", "r_t1"])
            sub = sub.add_prefix(f"CTX_{t}_")
            sub = sub.rename(columns={f"CTX_{t}_Date": "Date"})
            ctx_frames.append(sub)
        if ctx_frames:
            ctx_wide = reduce(lambda left, right: pd.merge(left, right, on="Date", how="outer"), ctx_frames)
            feat_panel = feat_panel.merge(ctx_wide, on="Date", how="left")

    feat_panel.to_parquet(panel_out, index=False)
    print(f"Wrote features: {len(feat_panel):,} rows -> {panel_out}")
    print("Columns:", len(feat_panel.columns))


if __name__ == "__main__":
    main()
