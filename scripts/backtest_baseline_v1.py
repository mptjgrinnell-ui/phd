import os
import argparse
import time
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed

Q_LIST = [0.05, 0.10, 0.50, 0.90, 0.95]
ALPHAS = [0.10, 0.05]  # 90%, 95%


def make_year_slices(df, first_test_year, last_test_year):
    years = sorted(df["Date"].dt.year.unique())
    if not years:
        return []
    max_year = max(years)
    if last_test_year is None:
        last_test_year = max_year - 1  # last full year typically
    for Y in range(first_test_year, last_test_year + 1):
        yield Y


def make_scaler():
    return RobustScaler(with_centering=True, with_scaling=True, quantile_range=(10, 90))


def make_regressor(q):
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=q,
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
    )


def fit_one_quantile(q, Xtr, ytr):
    m = make_regressor(q)
    m.fit(Xtr, ytr)
    return q, m


def parse_args():
    p = argparse.ArgumentParser(description="Baseline walk-forward backtest.")
    p.add_argument("--backtest-cfg", default="configs/backtest.yaml")
    p.add_argument("--features-cfg", default="configs/features.yaml")
    p.add_argument("--max-years", type=int, default=None, help="Limit number of test years processed.")
    p.add_argument(
        "--sample-tickers",
        type=int,
        default=None,
        help="If set, keep only the first N tickers (alphabetical) to speed smoke-tests.",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=5,
        help="Parallel workers for training the 5 quantile models (threading backend).",
    )
    p.add_argument(
        "--parallel-backend",
        default="threading",
        choices=["threading", "loky"],
        help="Joblib backend; use threading to avoid large data copies on Windows.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    backtest_cfg = args.backtest_cfg
    features_cfg = args.features_cfg
    max_years = args.max_years

    with open(backtest_cfg, "r") as f:
        bcfg = yaml.safe_load(f)
    with open(features_cfg, "r") as f:
        fcfg = yaml.safe_load(f)

    panel = pd.read_parquet(fcfg["panel_out"])
    panel["Date"] = pd.to_datetime(panel["Date"])
    panel = panel.sort_values(["Date", "Ticker"])
    panel = panel.dropna(axis=1, how="all")

    if args.sample_tickers:
        keep = sorted(panel["Ticker"].unique())[: args.sample_tickers]
        panel = panel[panel["Ticker"].isin(keep)]
        print(f"Sampling {len(keep)} tickers for fast run: {keep}")

    drop = {"Date", "Ticker", "r_t1"}
    feat_cols = [c for c in panel.columns if c not in drop]

    # Require target present; features are imputed downstream.
    panel = panel.dropna(subset=["r_t1"])
    years_arr = panel["Date"].dt.year.to_numpy()

    first_test_year = bcfg["walk_forward"]["first_test_year"]
    last_test_year = bcfg["walk_forward"]["last_test_year"]

    tape_rows = []

    processed = 0
    for Y in make_year_slices(panel, first_test_year, last_test_year):
        train_end = Y - 2
        cal_year = Y - 1
        test_year = Y

        train_mask = years_arr <= train_end
        cal_mask = years_arr == cal_year
        test_mask = years_arr == test_year

        train = panel[train_mask]
        cal = panel[cal_mask]
        test = panel[test_mask]

        if len(train) < 10000 or len(cal) < 1000 or len(test) < 1000:
            continue

        t0 = time.perf_counter()
        Xtr_df, ytr = train[feat_cols], train["r_t1"].to_numpy()
        Xcal_df, ycal = cal[feat_cols], cal["r_t1"].to_numpy()
        Xte_df, yte = test[feat_cols], test["r_t1"].to_numpy()

        # Fit preprocessing once per split (prevents ad hoc row dropping and avoids duplicated work per model).
        imputer = SimpleImputer(strategy="median")
        Xtr_imp = imputer.fit_transform(Xtr_df)
        Xcal_imp = imputer.transform(Xcal_df)
        Xte_imp = imputer.transform(Xte_df)

        scaler = make_scaler()
        Xtr = scaler.fit_transform(Xtr_imp)
        Xcal = scaler.transform(Xcal_imp)
        Xte = scaler.transform(Xte_imp)

        # Quantile models
        q_jobs = min(max(1, int(args.n_jobs)), len(Q_LIST))
        q_pairs = Parallel(n_jobs=q_jobs, backend=args.parallel_backend)(
            delayed(fit_one_quantile)(q, Xtr, ytr) for q in Q_LIST
        )
        q_models = dict(q_pairs)

        # Direction model (up/down) + calibration on the calibration year
        ytr_up = (ytr > 0).astype(int)
        ycal_up = (ycal > 0).astype(int)

        clf = LogisticRegression(
            solver="saga",
            penalty="l2",
            C=1.0,
            max_iter=2000,
            tol=1e-4,
            n_jobs=-1,
            random_state=42,
        )
        clf.fit(Xtr, ytr_up)

        # Predict on calibration for normalized conformal
        cal_q = {q: q_models[q].predict(Xcal) for q in Q_LIST}
        cal_sigma = np.maximum(1e-6, 0.5 * (cal_q[0.90] - cal_q[0.10]))
        cal_s = np.maximum(cal_q[0.05] - ycal, ycal - cal_q[0.95]) / cal_sigma

        # Mondrian buckets via volatility proxy
        def pick_proxy(df_slice):
            if "CTX_^VIX_ret_1" in df_slice:
                return np.abs(df_slice["CTX_^VIX_ret_1"].to_numpy())
            if "CTX_SPY_range_std_21" in df_slice:
                return df_slice["CTX_SPY_range_std_21"].to_numpy()
            if "range_std_21" in df_slice:
                return df_slice["range_std_21"].to_numpy()
            return None

        cal_proxy = pick_proxy(cal)
        te_proxy = pick_proxy(test)

        def make_edges(proxy_arr, n_bins=4):
            if proxy_arr is None:
                return None
            vals = proxy_arr[~np.isnan(proxy_arr)]
            if len(vals) < 100:
                return None
            qs = np.quantile(vals, [0, 0.25, 0.5, 0.75, 1.0])
            edges = np.unique(qs)
            if len(edges) < 3:
                return None
            return edges

        edges = make_edges(cal_proxy)

        def bucketize(arr, edges_in, n):
            if arr is None or edges_in is None:
                return np.zeros(n, dtype=int)
            breaks = edges_in[1:-1]
            b = np.digitize(arr, breaks, right=True)
            return np.where(np.isnan(arr), 0, b)

        cal_bucket = bucketize(cal_proxy, edges, len(cal_s))

        # Compute qhat per bucket with fallback to global
        qhat_global = {alpha: float(np.quantile(cal_s[~np.isnan(cal_s)], 1 - alpha)) for alpha in ALPHAS}
        qhat_bucket = {}
        for b in np.unique(cal_bucket):
            mask = cal_bucket == b
            scores_b = cal_s[mask]
            if scores_b.size == 0 or np.all(np.isnan(scores_b)):
                continue
            for alpha in ALPHAS:
                qhat_bucket.setdefault(alpha, {})
                qhat_bucket[alpha][b] = float(np.quantile(scores_b[~np.isnan(scores_b)], 1 - alpha))

        def get_qhat(alpha, b):
            if alpha in qhat_bucket and b in qhat_bucket[alpha]:
                return qhat_bucket[alpha][b]
            return qhat_global[alpha]

        # Predict on test
        preds = {q: q_models[q].predict(Xte) for q in Q_LIST}

        # Calibrate probabilities using the calibration year (Platt/sigmoid).
        # Guard against tiny samples where calibration labels collapse to 1 class.
        if np.unique(ycal_up).size >= 2:
            cal_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
            cal_clf.fit(Xcal, ycal_up)
            pup = cal_clf.predict_proba(Xte)[:, 1]
        else:
            pup = clf.predict_proba(Xte)[:, 1]

        te_sigma = np.maximum(1e-6, 0.5 * (preds[0.90] - preds[0.10]))
        te_bucket = bucketize(te_proxy, edges, len(yte))

        conf_bands = {}
        for alpha in ALPHAS:
            pct = int((1 - alpha) * 100)
            adj = np.array([get_qhat(alpha, b) for b in te_bucket]) * te_sigma
            conf_bands[f"conf{pct}_lo"] = preds[0.05] - adj
            conf_bands[f"conf{pct}_hi"] = preds[0.95] + adj

        out = test[["Date", "Ticker"]].copy()
        for q in Q_LIST:
            out[f"q{int(q * 100):02d}"] = preds[q]
        out["p_up"] = pup
        for k, v in conf_bands.items():
            out[k] = v
        out["realized_r1"] = yte
        out["test_year"] = Y
        tape_rows.append(out)

        # Quick console metrics (per year)
        mae = mean_absolute_error(yte, preds[0.50])
        cov90 = np.mean((yte >= conf_bands["conf90_lo"]) & (yte <= conf_bands["conf90_hi"]))
        cov95 = np.mean((yte >= conf_bands["conf95_lo"]) & (yte <= conf_bands["conf95_hi"]))
        exc05 = np.mean(yte < preds[0.05])
        exc95 = np.mean(yte > preds[0.95])
        dt = time.perf_counter() - t0
        print(
            f"Y={Y}  MAE(med)={mae:.6f}  cov90={cov90:.3f}  cov95={cov95:.3f}  "
            f"exc05={exc05:.3f}  exc95={exc95:.3f}  rows(tr/cal/te)={len(train)}/{len(cal)}/{len(test)}  "
            f"secs={dt:.1f}  jobs={q_jobs}/{args.parallel_backend}"
        )
        processed += 1
        if max_years is not None and processed >= max_years:
            print(f"Reached max_years={max_years}, stopping early.")
            break

    if not tape_rows:
        print("No valid walk-forward slices produced; check data coverage.")
        return

    tape = pd.concat(tape_rows, ignore_index=True).sort_values(["Date", "Ticker"])
    out_path = "data/reports/baseline_v1_tape.parquet"
    os.makedirs("data/reports", exist_ok=True)
    tape.to_parquet(out_path, index=False)
    print(f"\nWrote forecast tape -> {out_path}")
    print(tape.head())


if __name__ == "__main__":
    main()
