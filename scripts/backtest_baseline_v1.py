import os
import sys
import argparse
import atexit
import time
import warnings
import hashlib
import threading
import concurrent.futures
import re
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
Q_LIST = [0.05, 0.10, 0.50, 0.90, 0.95]
ALPHAS = [0.10, 0.05]  # 90%, 95%
SPLIT_CACHE_DIR = "data/meta/split_cache_v1"
PREP_CACHE_DIR = "data/meta/prep_cache_v1"
MIN_TRAIN_ROWS = 10000
MIN_CAL_ROWS = 1000
MIN_TEST_ROWS = 1000


class ETATracker:
    """
    ETA based on completed stage-units, with exponential moving average (EMA)
    of seconds per unit per stage.
    """

    def __init__(self, alpha=0.25, defaults=None):
        self.alpha = float(alpha)
        self.ema = {}  # stage -> seconds per unit
        self._t0 = None
        self._units = 1
        self.defaults = dict(
            {
                "fit_quantiles": 120.0,  # per quantile model until learned
                "fit_direction": 10.0,
                "calibrate_conformal": 10.0,
                "predict": 0.5,  # per predict substep
            },
            **(defaults or {}),
        )

    def start(self, units: int = 1):
        self._t0 = time.time()
        self._units = max(1, int(units))

    def end(self, stage: str):
        if self._t0 is None:
            return
        dt = max(1e-9, time.time() - self._t0)
        per_unit = dt / max(1, int(self._units))
        if stage not in self.ema:
            self.ema[stage] = per_unit
        else:
            self.ema[stage] = (1 - self.alpha) * float(self.ema[stage]) + self.alpha * per_unit
        self._t0 = None
        self._units = 1

    def eta_seconds(self, remaining_units_by_stage: dict) -> float:
        eta = 0.0
        for stage, rem_units in remaining_units_by_stage.items():
            per = self.ema.get(stage, self.defaults.get(stage, 1.0))
            eta += float(per) * float(rem_units)
        return float(max(0.0, eta))


def conformal_qhat(scores: np.ndarray, alpha: float, default: float = 0.0) -> float:
    """
    Finite-sample conformal quantile:
      q = k-th order statistic where k = ceil((n+1)*(1-alpha))
    Uses partition (O(n)) and clips at 0 so we never shrink bands.
    """
    v = scores[np.isfinite(scores)]
    if v.size == 0:
        return float(default)
    v = np.maximum(0.0, v)
    n = int(v.size)
    k = int(np.ceil((n + 1) * (1.0 - float(alpha))))  # 1..n+1
    k = min(max(k, 1), n)
    return float(np.partition(v, k - 1)[k - 1])


def sigma_from_proxy(proxy_arr, fallback_sigma, eps=1e-6):
    if proxy_arr is None:
        return fallback_sigma
    s = np.asarray(proxy_arr, dtype=float)
    s = np.where(np.isnan(s), np.nan, s)
    s = np.maximum(eps, s)
    return np.where(np.isnan(s), fallback_sigma, s)


def _setup_run_logging(log_dir="log", base_name="log"):
    os.makedirs(log_dir, exist_ok=True)

    existing = []
    pat = re.compile(rf"^{re.escape(base_name)}(\d+)\.txt$")
    for name in os.listdir(log_dir):
        m = pat.match(name)
        if m:
            existing.append(int(m.group(1)))
    n = (max(existing) + 1) if existing else 1

    run_path = os.path.join(log_dir, f"{base_name}{n}.txt")
    latest_path = os.path.join(log_dir, "latest.txt")

    run_f = open(run_path, "w", encoding="utf-8", buffering=1)
    latest_f = open(latest_path, "w", encoding="utf-8", buffering=1)

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams

        def write(self, s):
            for st in self._streams:
                st.write(s)
            return len(s)

        def flush(self):
            for st in self._streams:
                st.flush()

        def isatty(self):
            return False

    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_out, run_f, latest_f)
    sys.stderr = _Tee(orig_err, run_f, latest_f)

    def _cleanup():
        try:
            sys.stdout = orig_out
            sys.stderr = orig_err
        finally:
            try:
                run_f.close()
            finally:
                latest_f.close()

    atexit.register(_cleanup)

    print(f"[log] writing console output to {run_path} (also {latest_path})", flush=True)
    print(f"[log] cmd: {' '.join(sys.argv)}", flush=True)
    return run_path


class ProgressReporter:
    PHASE_STEPS = 6
    PHASE_TO_STEP = {
        "init": 0,
        "split": 0,
        "prep": 1,
        "fit_quantiles": 2,
        "fit_direction": 3,
        "calibrate_conformal": 4,
        "predict": 5,
        "skip_small": PHASE_STEPS,
        "done_year": PHASE_STEPS,
        "done_all": PHASE_STEPS,
        "no_valid_years": 0,
    }
    _SPIN = "|/-\\"

    def __init__(self, total, every_sec: float = 5.0):
        self.total = int(total)
        self.every_sec = float(every_sec)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._last_print_key = None
        self._last_print_ts = 0.0
        self._state = {
            "phase": "init",
            "year": None,
            "i": 0,
            "phase_step": 0,
            "sub_done": 0,
            "sub_total": 0,
            "sub_name": "",
            "rows_tr": 0,
            "rows_cal": 0,
            "rows_te": 0,
            "cum_test_rows": 0,
            "start_ts": time.time(),
            "year_start_ts": time.time(),
            "last_event_ts": time.time(),
            "eta_sec": None,
            "spin_i": 0,
        }

    def start(self):
        if self.every_sec > 0:
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def update(self, **kwargs):
        force_print = bool(kwargs.pop("force_print", False))
        event = bool(kwargs.pop("event", False))

        with self._lock:
            if "phase" in kwargs and "phase_step" not in kwargs:
                phase = kwargs["phase"]
                kwargs["phase_step"] = self.PHASE_TO_STEP.get(phase, 0)

                # Default sub-progress per phase (override by passing sub_* explicitly).
                if phase == "fit_quantiles":
                    kwargs.setdefault("sub_total", len(Q_LIST))
                    kwargs.setdefault("sub_done", 0)
                    kwargs.setdefault("sub_name", "q")
                elif phase == "predict":
                    kwargs.setdefault("sub_total", len(Q_LIST) + 2)  # quantiles + p_up + bands
                    kwargs.setdefault("sub_done", 0)
                    kwargs.setdefault("sub_name", "pred")
                else:
                    kwargs.setdefault("sub_total", 0)
                    kwargs.setdefault("sub_done", 0)
                    kwargs.setdefault("sub_name", "")

            if force_print or event:
                kwargs["last_event_ts"] = time.time()

            self._state.update(kwargs)

        if force_print:
            self.print_now()

    def _snapshot(self):
        with self._lock:
            return dict(self._state)

    def _next_spin(self, s: dict):
        with self._lock:
            self._state["spin_i"] = int(self._state.get("spin_i", 0)) + 1
            return self._SPIN[self._state["spin_i"] % len(self._SPIN)]

    def _format_secs(self, sec: float):
        sec = max(0.0, float(sec))
        if sec < 90:
            return f"{sec:,.0f}s"
        m, s = divmod(int(sec + 0.5), 60)
        if m < 90:
            return f"{m}m{s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m"

    def _format_line(self, s: dict, spin: str):
        total_years = max(self.total, 1)
        year_i = int(s.get("i", 0))
        years_done = min(max(year_i - 1, 0), total_years)

        phase_step = int(s.get("phase_step", 0))
        phase_step = min(max(phase_step, 0), self.PHASE_STEPS)

        sub_total = int(s.get("sub_total", 0))
        sub_done = int(s.get("sub_done", 0))
        sub_name = str(s.get("sub_name", ""))
        if sub_total > 0:
            sub_frac = min(max(sub_done / max(sub_total, 1), 0.0), 1.0)
        else:
            sub_frac = 0.0

        year_frac = min(max((phase_step + sub_frac) / max(self.PHASE_STEPS, 1), 0.0), 1.0)
        pct = 100.0 * ((years_done + year_frac) / total_years)

        now = time.time()
        elapsed = max(1e-6, now - s["start_ts"])
        since_evt = max(0.0, now - float(s.get("last_event_ts", s["start_ts"])))

        # "cum_test_rows" reflects completed years only. Provide an estimate as the
        # year advances based on phase/subprogress.
        year_te_rows = int(s.get("rows_te", 0))
        year_te_done_est = int(round(year_te_rows * year_frac))
        cum_done = int(s.get("cum_test_rows", 0))
        cum_te_est = cum_done + year_te_done_est

        eta_sec = s.get("eta_sec")
        eta_txt = "eta=?"
        if eta_sec is not None:
            eta_txt = f"eta={self._format_secs(float(eta_sec))}"

        y = s["year"]
        ytxt = "?" if y is None else str(y)
        subtxt = ""
        if sub_total > 0 and sub_name:
            subtxt = f"  {sub_name}={sub_done}/{sub_total}"
        return (
            f"{spin} [progress] {pct:5.1f}% (year {min(max(year_i, 0), total_years)}/{total_years})  "
            f"Y={ytxt}  phase={s['phase']}{subtxt}  "
            f"rows(tr/cal/te)={s['rows_tr']}/{s['rows_cal']}/{s['rows_te']}  "
            f"cum_te_est={cum_te_est} (done={cum_done})  yr_te_est={year_te_done_est}/{year_te_rows}  "
            f"{eta_txt}  t={self._format_secs(elapsed)}  idle={self._format_secs(since_evt)}"
        )

    def print_now(self):
        s = self._snapshot()
        spin = self._next_spin(s)
        line = self._format_line(s, spin)
        with self._lock:
            self._last_print_key = (
                s.get("i"),
                s.get("year"),
                s.get("phase"),
                s.get("sub_done"),
                s.get("sub_total"),
                s.get("rows_tr"),
                s.get("rows_cal"),
                s.get("rows_te"),
                s.get("cum_test_rows"),
            )
            self._last_print_ts = time.time()
        print(line, flush=True)

    def _run(self):
        while not self._stop.wait(self.every_sec):
            s = self._snapshot()
            key = (
                s.get("i"),
                s.get("year"),
                s.get("phase"),
                s.get("sub_done"),
                s.get("sub_total"),
                s.get("rows_tr"),
                s.get("rows_cal"),
                s.get("rows_te"),
                s.get("cum_test_rows"),
            )
            now = time.time()
            with self._lock:
                last_key = self._last_print_key
                last_ts = self._last_print_ts

            # Reduce spam: if nothing changed, print only every ~30s.
            if key == last_key and (now - last_ts) < 30.0:
                continue

            self.print_now()


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


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_or_make_split_indices(years_arr, test_year, cal_years: int = 1, cache_dir=SPLIT_CACHE_DIR):
    """
    For test year Y:
      train: years <= Y - cal_years - 1
      cal:   years in [Y-cal_years .. Y-1]
      test:  years == Y
    """
    ensure_dir(cache_dir)
    fp = os.path.join(cache_dir, f"splits_Y{test_year}_K{int(cal_years)}.npz")

    if os.path.exists(fp):
        z = np.load(fp)
        return z["tr"], z["cal"], z["te"]

    Y = int(test_year)
    K = int(cal_years)
    cal_start = Y - K

    tr = np.flatnonzero(years_arr < cal_start)
    cal = np.flatnonzero((years_arr >= cal_start) & (years_arr <= (Y - 1)))
    te = np.flatnonzero(years_arr == test_year)

    np.savez_compressed(fp, tr=tr.astype(np.int32), cal=cal.astype(np.int32), te=te.astype(np.int32))
    return tr, cal, te


def split_indices_K(years_arr: np.ndarray, Y: int, cal_years: int):
    cal_start = Y - int(cal_years)
    tr = np.flatnonzero(years_arr < cal_start)
    cal = np.flatnonzero((years_arr >= cal_start) & (years_arr <= (Y - 1)))
    te = np.flatnonzero(years_arr == Y)
    return tr, cal, te


def safe_split_indices_K(years_arr: np.ndarray, Y: int, cal_years: int):
    min_year = int(years_arr.min())
    max_possible_K = max(1, (Y - 1) - min_year)
    K_req = max(1, int(cal_years))
    K_use = min(K_req, max_possible_K)
    if K_use != K_req:
        print(
            f"[warn] Y={Y} requested cal_years={K_req} but only K={K_use} available from data; using K={K_use}.",
            flush=True,
        )
    return split_indices_K(years_arr, Y, K_use), K_use


def load_or_make_preprocessed(Y, Xtr, Xcal, Xte, imputer, scaler, feat_sig, split_sig, cache_dir=PREP_CACHE_DIR):
    ensure_dir(cache_dir)
    fp = os.path.join(cache_dir, f"prep_{feat_sig}_{split_sig}_Y{Y}.npz")
    if os.path.exists(fp):
        z = np.load(fp)
        return z["Xtr"], z["Xcal"], z["Xte"]

    Xtr_i = imputer.fit_transform(Xtr)
    Xcal_i = imputer.transform(Xcal)
    Xte_i = imputer.transform(Xte)

    Xtr_s = scaler.fit_transform(Xtr_i)
    Xcal_s = scaler.transform(Xcal_i)
    Xte_s = scaler.transform(Xte_i)

    np.savez_compressed(fp, Xtr=Xtr_s, Xcal=Xcal_s, Xte=Xte_s)
    return Xtr_s, Xcal_s, Xte_s


def parse_args():
    p = argparse.ArgumentParser(
        description="Baseline walk-forward backtest.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Run C (shortest valid run):\n"
            "  python scripts/backtest_baseline_v1.py --sample-tickers 10 --sample-seed 42 "
            "--max-years 1 --n-jobs 5 --parallel-backend threading\n"
        ),
    )
    p.add_argument("--backtest-cfg", default="configs/backtest.yaml")
    p.add_argument("--features-cfg", default="configs/features.yaml")
    p.add_argument("--max-years", type=int, default=None, help="Limit number of test years processed.")
    p.add_argument(
        "--sample-tickers",
        type=int,
        default=None,
        help="If set, keep only N randomly sampled tickers to speed smoke-tests.",
    )
    p.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="RNG seed for --sample-tickers random sampling.",
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
    p.add_argument("--cache-splits", action="store_true", help="Cache yearly split indices to disk.")
    p.add_argument("--cache-prep", action="store_true", help="Cache imputed+scaled arrays per year (iteration speed).")
    p.add_argument("--drop-constant-cols", action="store_true", help="Drop constant columns once (no signal loss).")
    p.add_argument(
        "--progress-every-sec",
        type=float,
        default=5.0,
        help="Print a progress heartbeat every N seconds (0 disables).",
    )
    p.add_argument(
        "--cal-years",
        type=int,
        default=None,
        help="Calibration window length in years. For test year Y, calibrate on [Y-cal_years .. Y-1].",
    )
    p.add_argument(
        "--n-buckets",
        type=int,
        default=3,
        help="Mondrian bucket count for conformal conditioning (default 3; try 5).",
    )
    p.add_argument(
        "--regime-buckets",
        type=int,
        default=None,
        help="Number of Mondrian regime buckets for conformal (e.g., 5).",
    )
    p.add_argument(
        "--sigma-proxy",
        choices=["pred", "rv", "hybrid"],
        default="pred",
        help="Sigma for conformal normalization: pred=(q90-q10)/2, rv=EWMA realized vol, hybrid=max(pred,rv).",
    )
    p.add_argument(
        "--first-test-year",
        type=int,
        default=None,
        help="Override configs/backtest.yaml walk_forward.first_test_year",
    )
    p.add_argument(
        "--last-test-year",
        type=int,
        default=None,
        help="Override configs/backtest.yaml walk_forward.last_test_year",
    )
    p.add_argument(
        "--min-panel-year",
        type=int,
        default=None,
        help="If set, drop panel rows with Date.year < this to reduce IO/RAM (debug speedup).",
    )
    p.add_argument("--regime-bins", type=int, default=4, help="Bins for regime/Mondrian proxy (default 4).")
    return p.parse_args()


def main():
    _setup_run_logging()
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
    panel = panel.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    panel = panel.dropna(axis=1, how="all")

    # Leak-free realized vol proxy (EWMA of squared daily return), shift(1) to avoid look-ahead.
    if "ret_1" in panel.columns:
        rv20 = (panel["ret_1"] ** 2).groupby(panel["Ticker"]).transform(lambda s: s.ewm(span=20, adjust=False).mean())
        panel["sigma_rv20"] = np.sqrt(rv20).shift(1)
    else:
        panel["sigma_rv20"] = np.nan

    if args.min_panel_year is not None:
        min_y = int(args.min_panel_year)
        panel = panel.loc[panel["Date"].dt.year >= min_y].copy()
        print(f"[data-cut] kept rows with Date.year >= {min_y}. rows={len(panel):,}")

    drop = {"Date", "Ticker", "r_t1"}
    feat_cols = [c for c in panel.columns if c not in drop]

    # Require target present; features are imputed downstream.
    panel = panel.dropna(subset=["r_t1"])
    panel = panel.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # --- FAST ARRAYS (convert once) ---
    years_arr = panel["Date"].dt.year.to_numpy(np.int16)
    dates_arr = panel["Date"].to_numpy()
    tickers_arr = panel["Ticker"].to_numpy()
    sigma_rv_arr = panel["sigma_rv20"].to_numpy(dtype=np.float32, copy=False)

    # Feature matrix + target as float32 (faster + less RAM).
    X_all = panel[feat_cols].to_numpy(dtype=np.float32, copy=False)
    y_all = panel["r_t1"].to_numpy(dtype=np.float32, copy=False)

    # --- optional ticker subsample (random, seeded) ---
    if args.sample_tickers is not None:
        rng = np.random.default_rng(args.sample_seed)

        uniq = np.unique(tickers_arr)
        if args.sample_tickers >= len(uniq):
            chosen = set(uniq.tolist())
        else:
            chosen = set(rng.choice(uniq, size=args.sample_tickers, replace=False).tolist())

        keep_mask = np.isin(tickers_arr, list(chosen))

        # IMPORTANT: keep arrays aligned
        years_arr = years_arr[keep_mask]
        dates_arr = dates_arr[keep_mask]
        tickers_arr = tickers_arr[keep_mask]
        X_all = X_all[keep_mask]
        y_all = y_all[keep_mask]
        sigma_rv_arr = sigma_rv_arr[keep_mask]

        print(f"Sampled {len(chosen)} tickers (seed={args.sample_seed}).")

    # Optional: drop constant columns once (no signal loss).
    if args.drop_constant_cols:
        stds = np.nanstd(X_all, axis=0)
        keep_mask = stds > 0
        if not np.all(keep_mask):
            dropped = int(np.sum(~keep_mask))
            feat_cols = [c for c, k in zip(feat_cols, keep_mask) if k]
            X_all = X_all[:, keep_mask]
            print(f"Dropped {dropped} constant cols; remaining={len(feat_cols)}")

    feat_sig = hashlib.sha1(("|".join(feat_cols)).encode("utf-8")).hexdigest()[:10]
    uniq_tickers = np.unique(tickers_arr)
    split_sig = hashlib.sha1(
        (
            "|".join(uniq_tickers.tolist())
            + f"|{years_arr.min()}|{years_arr.max()}|{len(years_arr)}|cal{int(cal_years)}"
        ).encode("utf-8")
    ).hexdigest()[:10]

    # Proxies for Mondrian bucketing (precompute once, if available).
    proxy_vix_absret1 = None
    proxy_vix_level = None
    proxy_spy_range_std21 = None
    proxy_spy_rvol21 = None
    proxy_own_range_std21 = None
    if "CTX_^VIX_ret_1" in panel.columns:
        proxy_vix_absret1 = np.abs(panel["CTX_^VIX_ret_1"].to_numpy(dtype=np.float32, copy=False))
    if "CTX_^VIX_close" in panel.columns:
        proxy_vix_level = panel["CTX_^VIX_close"].to_numpy(dtype=np.float32, copy=False)
    if "CTX_SPY_range_std_21" in panel.columns:
        proxy_spy_range_std21 = panel["CTX_SPY_range_std_21"].to_numpy(dtype=np.float32, copy=False)
    if "CTX_SPY_rvol_21" in panel.columns:
        proxy_spy_rvol21 = panel["CTX_SPY_rvol_21"].to_numpy(dtype=np.float32, copy=False)
    if "range_std_21" in panel.columns:
        proxy_own_range_std21 = panel["range_std_21"].to_numpy(dtype=np.float32, copy=False)

    first_test_year = (
        int(args.first_test_year)
        if args.first_test_year is not None
        else int(bcfg["walk_forward"]["first_test_year"])
    )
    last_test_year = (
        int(args.last_test_year)
        if args.last_test_year is not None
        else int(bcfg["walk_forward"]["last_test_year"])
    )
    default_cal_years = int(bcfg["walk_forward"].get("cal_years", 1))
    cal_years = int(args.cal_years) if args.cal_years is not None else default_cal_years
    if last_test_year is not None and first_test_year > last_test_year:
        raise ValueError(f"first_test_year ({first_test_year}) > last_test_year ({last_test_year})")
    cal_years = int(args.cal_years) if args.cal_years is not None else int(bcfg["walk_forward"].get("cal_years", 1))
    n_buckets = (
        int(args.regime_buckets)
        if args.regime_buckets is not None
        else int(bcfg.get("conformal", {}).get("n_buckets", 3))
    )
    if cal_years < 1:
        raise ValueError("--cal-years must be >= 1")
    if n_buckets < 2:
        raise ValueError("--regime-buckets must be >= 2")
    cal_years = int(args.cal_years) if args.cal_years is not None else int(bcfg["walk_forward"].get("cal_years", 1))
    n_buckets = (
        int(args.regime_buckets)
        if args.regime_buckets is not None
        else int(bcfg.get("conformal", {}).get("n_buckets", 3))
    )
    if cal_years < 1:
        raise ValueError("--cal-years must be >= 1")
    if n_buckets < 2:
        raise ValueError("--regime-buckets must be >= 2")

    tape_rows = []

    candidate_years = list(make_year_slices(panel, first_test_year, last_test_year))

    # Pre-filter to years that have enough rows for tr/cal/te. This avoids wasting
    # early years when sampling tickers with limited history.
    years_to_run = []
    for Y in candidate_years:
        tr_idx = np.flatnonzero(years_arr < (Y - cal_years))
        cal_idx = np.flatnonzero((years_arr >= (Y - cal_years)) & (years_arr <= (Y - 1)))
        te_idx = np.flatnonzero(years_arr == Y)
        if len(tr_idx) >= MIN_TRAIN_ROWS and len(cal_idx) >= MIN_CAL_ROWS and len(te_idx) >= MIN_TEST_ROWS:
            years_to_run.append(Y)

    if max_years is not None:
        years_to_run = years_to_run[: int(max_years)]

    progress = ProgressReporter(total=len(years_to_run), every_sec=args.progress_every_sec)
    progress.start()

    eta = ETATracker(alpha=0.25)

    processed = 0
    cum_test_rows = 0
    try:
        if not years_to_run:
            progress.update(phase="no_valid_years")
            print("No valid walk-forward slices produced; check data coverage.", flush=True)
            return
        for i, Y in enumerate(years_to_run, start=1):
            cal_year_start = Y - cal_years
            cal_year_end = Y - 1
            test_year = Y
            progress.update(i=i, year=Y, phase="split")

            if args.cache_splits:
                split_cache_dir = os.path.join(SPLIT_CACHE_DIR, split_sig)
                tr_idx, cal_idx, te_idx = load_or_make_split_indices(
                    years_arr, Y, cal_years=cal_years, cache_dir=split_cache_dir
                )
            else:
                tr_idx = np.flatnonzero(years_arr < cal_year_start)
                cal_idx = np.flatnonzero((years_arr >= cal_year_start) & (years_arr <= cal_year_end))
                te_idx = np.flatnonzero(years_arr == test_year)

            progress.update(rows_tr=len(tr_idx), rows_cal=len(cal_idx), rows_te=len(te_idx), cum_test_rows=cum_test_rows)
            if len(tr_idx) < 10000 or len(cal_idx) < 1000 or len(te_idx) < 1000:
                progress.update(phase="skip_small")
                ntr_t = int(np.unique(tickers_arr[tr_idx]).size) if len(tr_idx) else 0
                ncal_t = int(np.unique(tickers_arr[cal_idx]).size) if len(cal_idx) else 0
                nte_t = int(np.unique(tickers_arr[te_idx]).size) if len(te_idx) else 0
                print(
                    f"SKIP Y={Y} rows tr/cal/te={len(tr_idx)}/{len(cal_idx)}/{len(te_idx)} "
                    f"tickers tr/cal/te={ntr_t}/{ncal_t}/{nte_t}",
                    flush=True,
                )
                continue

            t_year_start = time.time()
            progress.update(year_start_ts=t_year_start, event=True)

            t0 = time.perf_counter()
            Xtr, ytr = X_all[tr_idx], y_all[tr_idx]
            Xcal, ycal = X_all[cal_idx], y_all[cal_idx]
            Xte, yte = X_all[te_idx], y_all[te_idx]
            Xtr = np.ascontiguousarray(Xtr, dtype=np.float32)
            Xcal = np.ascontiguousarray(Xcal, dtype=np.float32)
            Xte = np.ascontiguousarray(Xte, dtype=np.float32)
            dates_te = dates_arr[te_idx]
            tickers_te = tickers_arr[te_idx]

            total_years = len(years_to_run)
            remaining_years_after = max(0, total_years - i)

            def eta_remaining(stage: str, q_done: int = 0, pred_done: int = 0):
                rem = {}
                if stage == "fit_quantiles":
                    rem["fit_quantiles"] = max(0, len(Q_LIST) - int(q_done))
                    rem["fit_direction"] = 1
                    rem["calibrate_conformal"] = 1
                    rem["predict"] = 7
                elif stage == "fit_direction":
                    rem["fit_direction"] = 0
                    rem["calibrate_conformal"] = 1
                    rem["predict"] = 7
                elif stage == "calibrate_conformal":
                    rem["calibrate_conformal"] = 0
                    rem["predict"] = 7
                elif stage == "predict":
                    rem["predict"] = max(0, 7 - int(pred_done))

                if remaining_years_after:
                    rem["fit_quantiles"] = rem.get("fit_quantiles", 0) + remaining_years_after * len(Q_LIST)
                    rem["fit_direction"] = rem.get("fit_direction", 0) + remaining_years_after * 1
                    rem["calibrate_conformal"] = rem.get("calibrate_conformal", 0) + remaining_years_after * 1
                    rem["predict"] = rem.get("predict", 0) + remaining_years_after * 7
                return rem

            progress.update(phase="prep")
            # Fit preprocessing once per split (prevents ad hoc row dropping and avoids duplicated work per model).
            imputer = SimpleImputer(strategy="median")
            scaler = make_scaler()
            if args.cache_prep:
                Xtr, Xcal, Xte = load_or_make_preprocessed(Y, Xtr, Xcal, Xte, imputer, scaler, feat_sig, split_sig)
            else:
                Xtr_imp = imputer.fit_transform(Xtr)
                Xcal_imp = imputer.transform(Xcal)
                Xte_imp = imputer.transform(Xte)

                Xtr = scaler.fit_transform(Xtr_imp)
                Xcal = scaler.transform(Xcal_imp)
                Xte = scaler.transform(Xte_imp)

            progress.update(phase="fit_quantiles")
            # Quantile models
            q_jobs = min(max(1, int(args.n_jobs)), len(Q_LIST))
            if args.parallel_backend == "threading":
                progress.update(
                    sub_done=0,
                    sub_total=len(Q_LIST),
                    sub_name="q",
                    eta_sec=eta.eta_seconds(eta_remaining("fit_quantiles", q_done=0)),
                    force_print=True,
                )
                q_pairs = []
                eta.start(units=len(Q_LIST))
                with concurrent.futures.ThreadPoolExecutor(max_workers=q_jobs) as ex:
                    futures = [ex.submit(fit_one_quantile, q, Xtr, ytr) for q in Q_LIST]
                    for k, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
                        q_pairs.append(fut.result())
                        progress.update(
                            sub_done=k,
                            sub_total=len(Q_LIST),
                            sub_name="q",
                            eta_sec=eta.eta_seconds(eta_remaining("fit_quantiles", q_done=k)),
                            force_print=True,
                        )
                eta.end("fit_quantiles")
            else:
                eta.start(units=len(Q_LIST))
                q_pairs = Parallel(n_jobs=q_jobs, backend=args.parallel_backend)(
                    delayed(fit_one_quantile)(q, Xtr, ytr) for q in Q_LIST
                )
                eta.end("fit_quantiles")
                progress.update(
                    sub_done=len(Q_LIST),
                    sub_total=len(Q_LIST),
                    sub_name="q",
                    eta_sec=eta.eta_seconds(eta_remaining("fit_quantiles", q_done=len(Q_LIST))),
                    force_print=True,
                )
            q_models = dict(q_pairs)

            progress.update(phase="fit_direction", eta_sec=eta.eta_seconds(eta_remaining("fit_quantiles", q_done=len(Q_LIST))))
            # ---------------------------
            # Direction model (up/down) + calibration on calibration year
            # Fixes:
            #  - ConvergenceWarning: raise max_iter + stronger regularization
            #  - prefit deprecation: use FrozenEstimator with CalibratedClassifierCV
            # ---------------------------

            ytr_up = (ytr > 0).astype(int)
            ycal_up = (ycal > 0).astype(int)

            USE_CLASS_BALANCE = True

            # Fit LR on TRAIN only, retry once if it doesn't converge.
            eta.start()
            for max_iter, tol, C in [(5000, 1e-4, 0.2), (20000, 1e-3, 0.1)]:
                clf = LogisticRegression(
                    solver="saga",
                    penalty="l2",
                    C=C,
                    max_iter=max_iter,
                    tol=tol,
                    n_jobs=-1,
                    random_state=42,
                    class_weight=("balanced" if USE_CLASS_BALANCE else None),
                )
                with warnings.catch_warnings(record=True) as w:
                    warnings.filterwarnings("always", category=ConvergenceWarning)
                    clf.fit(Xtr, ytr_up)
                if not any(isinstance(wi.message, ConvergenceWarning) for wi in w):
                    break
            eta.end("fit_direction")

            progress.update(phase="calibrate_conformal", eta_sec=eta.eta_seconds(eta_remaining("fit_direction")))
            # Predict on calibration for normalized conformal
            eta.start()
            cal_q = {q: q_models[q].predict(Xcal) for q in Q_LIST}

            # Mondrian buckets via volatility proxy (precomputed once, then indexed).
            # Prefer a regime proxy that represents "state" (level/realized vol), not just a noisy 1-day move.
            cal_proxy = None
            te_proxy = None
            if proxy_vix_level is not None:
                cal_proxy = proxy_vix_level[cal_idx]
                te_proxy = proxy_vix_level[te_idx]
            elif proxy_spy_rvol21 is not None:
                cal_proxy = proxy_spy_rvol21[cal_idx]
                te_proxy = proxy_spy_rvol21[te_idx]
            elif proxy_spy_range_std21 is not None:
                cal_proxy = proxy_spy_range_std21[cal_idx]
                te_proxy = proxy_spy_range_std21[te_idx]
            elif proxy_vix_absret1 is not None:
                cal_proxy = proxy_vix_absret1[cal_idx]
                te_proxy = proxy_vix_absret1[te_idx]
            elif proxy_own_range_std21 is not None:
                cal_proxy = proxy_own_range_std21[cal_idx]
                te_proxy = proxy_own_range_std21[te_idx]

            # Heteroskedastic scale proxy (fallback to IQR, prefer regime proxy sigma).
            EPS = 1e-6
            cal_sigma_fallback = np.maximum(EPS, 0.5 * (cal_q[0.90] - cal_q[0.10]))
            cal_sigma_rv = sigma_rv_arr[cal_idx]

            def pick_sigma(pred_sigma, rv_sigma, mode: str, eps: float = 1e-6):
                pred_sigma = np.asarray(pred_sigma, dtype=np.float32)
                rv_sigma = np.asarray(rv_sigma, dtype=np.float32)
                if mode == "pred":
                    s = pred_sigma
                elif mode == "rv":
                    s = rv_sigma
                else:  # hybrid
                    s = np.maximum(pred_sigma, rv_sigma)
                s = np.where(np.isnan(s), pred_sigma, s)
                return np.maximum(eps, s)

            cal_sigma = pick_sigma(cal_sigma_fallback, cal_sigma_rv, mode=args.sigma_proxy, eps=EPS)

            # Two-sided CQR normalized nonconformity score (single score).
            # This avoids the "mass at 0 => qhat=0" failure mode from split tails.
            cal_s = np.maximum(0.0, np.maximum(cal_q[0.05] - ycal, ycal - cal_q[0.95]) / cal_sigma)

            def make_bucket_edges(cal_proxy_arr: np.ndarray, n_bins: int):
                if cal_proxy_arr is None:
                    return np.array([-np.inf, np.inf], dtype=np.float32)
                m = np.isfinite(cal_proxy_arr)
                x = cal_proxy_arr[m]
                if x.size < 50:
                    return np.array([-np.inf, np.inf], dtype=np.float32)
                qs = np.linspace(0.0, 1.0, int(n_bins) + 1)
                edges = np.quantile(x, qs).astype(np.float32)
                edges = np.unique(edges)
                if edges.size < 3:
                    return np.array([-np.inf, np.inf], dtype=np.float32)
                edges[0] = -np.inf
                edges[-1] = np.inf
                return edges

            def bucketize(arr, edges_arr, n):
                if arr is None or edges_arr is None:
                    return np.zeros(n, dtype=int)
                b = np.digitize(arr, edges_arr[1:-1], right=True)
                b = np.where(np.isfinite(arr), b, 0)
                return b.astype(int)

            edges = make_bucket_edges(cal_proxy, n_buckets)
            cal_bucket = bucketize(cal_proxy, edges, len(cal_s))
            te_bucket = bucketize(te_proxy, edges, len(yte))

            # Global fallback quantiles (CQR score).
            qhat_global = {alpha: conformal_qhat(cal_s, alpha) for alpha in ALPHAS}

            # Bucketed quantiles (Mondrian by regime).
            qhat_bucket = {alpha: {} for alpha in ALPHAS}
            for b in np.unique(cal_bucket):
                mask = cal_bucket == b
                s_b = cal_s[mask]
                for alpha in ALPHAS:
                    qhat_bucket[alpha][int(b)] = conformal_qhat(s_b, alpha, default=qhat_global[alpha])

            def get_qhat(alpha, b):
                b = int(b)
                return qhat_bucket.get(alpha, {}).get(b, qhat_global[alpha])
            eta.end("calibrate_conformal")

            progress.update(phase="predict", eta_sec=eta.eta_seconds(eta_remaining("calibrate_conformal")))
            progress.update(
                sub_done=0,
                sub_total=len(Q_LIST) + 2,
                sub_name="pred",
                eta_sec=eta.eta_seconds(eta_remaining("predict", pred_done=0)),
                force_print=True,
            )
            preds = {}
            for k, q in enumerate(Q_LIST, start=1):
                eta.start()
                preds[q] = q_models[q].predict(Xte)
                eta.end("predict")
                progress.update(
                    sub_done=k,
                    sub_total=len(Q_LIST) + 2,
                    sub_name="pred",
                    eta_sec=eta.eta_seconds(eta_remaining("predict", pred_done=k)),
                    force_print=True,
                )

            # Calibrate probabilities using the calibration year (Platt/sigmoid).
            # Guard against tiny samples where calibration labels collapse to 1 class.
            eta.start()
            if np.unique(ycal_up).size >= 2:
                min_class = int(np.min(np.bincount(ycal_up)))
                cv_splits = min(5, min_class)
                if cv_splits >= 2:
                    frozen = FrozenEstimator(clf)
                    cal_clf = CalibratedClassifierCV(
                        estimator=frozen,
                        method="sigmoid",
                        cv=cv_splits,
                    )
                    cal_clf.fit(Xcal, ycal_up)
                    pup = cal_clf.predict_proba(Xte)[:, 1]
                else:
                    pup = clf.predict_proba(Xte)[:, 1]
            else:
                pup = clf.predict_proba(Xte)[:, 1]
            eta.end("predict")
            progress.update(
                sub_done=len(Q_LIST) + 1,
                sub_total=len(Q_LIST) + 2,
                sub_name="pred",
                eta_sec=eta.eta_seconds(eta_remaining("predict", pred_done=len(Q_LIST) + 1)),
                force_print=True,
            )

            te_sigma_fallback = np.maximum(EPS, 0.5 * (preds[0.90] - preds[0.10]))
            te_sigma_rv = sigma_rv_arr[te_idx]
            te_sigma = pick_sigma(te_sigma_fallback, te_sigma_rv, mode=args.sigma_proxy, eps=EPS)
            te_bucket = bucketize(te_proxy, breaks, len(yte))

            conf_bands = {}
            eta.start()
            for alpha in ALPHAS:
                pct = int((1 - alpha) * 100)
                adj = np.array([get_qhat(alpha, b) for b in te_bucket], dtype=np.float32) * te_sigma

                conf_bands[f"conf{pct}_lo"] = preds[0.05] - adj
                conf_bands[f"conf{pct}_hi"] = preds[0.95] + adj
            eta.end("predict")
            progress.update(
                sub_done=len(Q_LIST) + 2,
                sub_total=len(Q_LIST) + 2,
                sub_name="pred",
                eta_sec=eta.eta_seconds(eta_remaining("predict", pred_done=7)),
                force_print=True,
            )

            out = pd.DataFrame({"Date": dates_arr[te_idx], "Ticker": tickers_arr[te_idx]})
            for q in Q_LIST:
                out[f"q{int(q * 100):02d}"] = preds[q]
            out["p_up"] = pup
            for k, v in conf_bands.items():
                out[k] = v
            out["realized_r1"] = yte
            out["test_year"] = Y
            tape_rows.append(out)

            mae = mean_absolute_error(yte, preds[0.50])
            cov90 = np.mean((yte >= conf_bands["conf90_lo"]) & (yte <= conf_bands["conf90_hi"]))
            cov95 = np.mean((yte >= conf_bands["conf95_lo"]) & (yte <= conf_bands["conf95_hi"]))
            exc05 = np.mean(yte < preds[0.05])
            exc95 = np.mean(yte > preds[0.95])
            cum_test_rows += len(te_idx)
            secs = time.time() - t_year_start
            te_rows_s = len(te_idx) / max(secs, 1e-9)
            dt = time.perf_counter() - t0
            print(
                f"Y={Y}  MAE(med)={mae:.6f}  cov90={cov90:.3f}  cov95={cov95:.3f}  "
                f"exc05={exc05:.3f}  exc95={exc95:.3f}  rows(tr/cal/te)={len(tr_idx)}/{len(cal_idx)}/{len(te_idx)}  "
                f"secs={secs:.1f}  te_rows/s={te_rows_s:.1f}  jobs={q_jobs}/{args.parallel_backend}"
                ,
                flush=True,
            )

            processed += 1
            progress.update(phase="done_year", cum_test_rows=cum_test_rows, event=True)

        progress.update(phase="done_all", cum_test_rows=cum_test_rows)
    finally:
        progress.stop()

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
