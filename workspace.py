#!/usr/bin/env python3
"""50 x 50 fixed-i_w scan; same dynamics and ICG score as supplied.
Run beside icg_functions.py: python grid_50ew_50phi_llr_tau.py
In a Linux notebook: %run grid_50ew_50phi_llr_tau.py
No simulations execute on import. Completed network jobs are resumable.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import json
import inspect
import warnings
import contextlib
import io
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from tqdm.auto import tqdm
from threadpoolctl import threadpool_limits
import powerlaw
import icg_functions as fn

start_dic = {
    "n_neurons": 2000, "T": 10, "dt": 0.01,
    "refractory_steps": 2, "ei_ratio": 0.2,
    "e_w": 11.7, "i_w": 22.4, "theta": 8.44,
    "p_ext": 0.00, "phi": 4.2, "smoothe": 0.05,
}
OUTDIR = Path(
    "/home/dburrows/DATA/BLNDEV-WILDTYPE/"
    "grid_ew_phi_iw22p4_sparse_avalanche_icg_50ew_50phis_20seeds_10000avals"
)
TRIALDIR = OUTDIR / "avalanche_trials_by_network"
ICGDIR = OUTDIR / "icg_rows_by_network"
JOBDIR = OUTDIR / "completed_network_jobs"
LLRDIR = OUTDIR / "truncated_powerlaw_vs_lognormal_fixed_iw_exact_llr_safe"

N_WORKERS = 20
N_SEEDS = 20
N_AVALANCHES_PER_NETWORK = 10000
MAX_STEPS = 1000
BASE_SEED = 91919

E_W_RANGE = (0.5, 18.0)
I_W_FIXED = float(start_dic["i_w"])
N_EW, N_IW, N_PHI = 50, 1, 50
EW_VALUES = np.linspace(*E_W_RANGE, N_EW)
IW_VALUES = np.array([I_W_FIXED])
PHI_VALUES = np.linspace(1.5, 6.0, N_PHI)

P_EXT_RUN = 0.0
SEED_ONLY_EXCITATORY = True
CLAMP_ZERO_PRESYNAPTIC_INPUT = True
DT = float(start_dic["dt"])
T_RUN = float(start_dic["T"])
SMOOTHE = float(start_dic["smoothe"])
BURN_IN_S = 2.0
TARGET_MV, TARGET_TAU = 1.50, 0.20
SAVE_AVALANCHE_TRIALS = True
SAVE_ICG_ROWS = True

SIZE_XMIN, DURATION_XMIN = 10, 0.1
EXTINCT_ONLY = True
MIN_N, MIN_TAIL_N, MIN_UNIQUE_TAIL = 100, 50, 3
RESUME = True
# Both estimands are retained. Pooled fits reproduce the previous
# analysis; mean per-network fits supply across-network SEMs.
SAVE_SEED_LLR = True
RUN_POOLED_LLR = True

KEYS = ["combo_i", "ew_idx", "iw_idx", "phi_idx", "phi", "e_w", "i_w"]
FIT_NAME = "truncated_powerlaw_vs_lognormal_by_ew_phi_fixed_iw_safe_extinct_only.csv"


def atomic_csv(df, path, compression=None):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, compression=compression)
    os.replace(tmp, path)


def atomic_json(value, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    # NaN follows Python JSON convention and is read by this script.
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)


def safe_values(x):
    return pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)


def sem(x):
    x = safe_values(x)
    return float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else np.nan


def safe_mean(x):
    x = safe_values(x)
    return float(np.mean(x)) if x.size else np.nan


def safe_percentile(x, q):
    x = safe_values(x)
    return float(np.percentile(x, q)) if x.size else np.nan


def safe_cv(x):
    x = safe_values(x)
    if x.size < 2 or np.mean(x) <= 0:
        return np.nan
    return float(np.std(x, ddof=1) / np.mean(x))


def response_decay_timescale(active_counts, dt):
    x = np.asarray(active_counts, dtype=float)
    if not x.size:
        return np.nan
    peak = float(x.max())
    if peak <= 0:
        return 0.0
    post = x[int(np.argmax(x)):]
    below = np.flatnonzero(post <= peak / np.e)
    return float((below[0] if below.size else post.size - 1) * dt)


def branching_metrics(active_counts, extinct):
    counts = np.asarray(active_counts, dtype=float)
    out = dict(sigma_origin=np.nan, br_reg_slope=np.nan,
               br_reg_intercept=np.nan, br_reg_r2=np.nan, br_n_pairs=0)
    if not counts.size:
        return out
    if extinct:
        x, y = counts, np.r_[counts[1:], 0.0]
    else:
        x, y = counts[:-1], counts[1:]
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[keep], y[keep]
    out["br_n_pairs"] = int(x.size)
    if x.size < 2:
        return out
    denom = float(np.sum(x * x))
    out["sigma_origin"] = float(np.sum(x * y) / denom) if denom > 0 else np.nan
    try:
        X = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out["br_reg_intercept"], out["br_reg_slope"] = map(float, beta)
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        out["br_reg_r2"] = (
            float(1 - np.sum((y - X @ beta) ** 2) / ss_tot)
            if ss_tot > 0 else np.nan
        )
    except Exception:
        pass
    return out


def local_connectivity_metrics(A, n_e):
    A = np.asarray(A, dtype=np.uint8)
    return {
        "edge_count": int(A.sum()),
        "mean_out_degree": float(A.sum(axis=1).mean()),
        "std_out_degree": float(A.sum(axis=1).std()),
        "mean_e_out_degree_all": float(A[:n_e, :].sum(axis=1).mean()),
        "mean_e_out_degree_e": float(A[:n_e, :n_e].sum(axis=1).mean()),
        "edge_density": float(A.sum() / max(A.shape[0] * (A.shape[0]-1), 1)),
    }


def pop_autocorr_tau(x, dt, max_lag_s=3.0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return np.nan
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom <= 0:
        return np.nan
    max_lag = min(int(max_lag_s / dt), x.size // 2)
    if max_lag < 2:
        return np.nan
    ac = np.empty(max_lag + 1)
    ac[0] = 1.0
    for lag in range(1, max_lag + 1):
        ac[lag] = np.sum(x[:-lag] * x[lag:]) / denom
    ac[~np.isfinite(ac)] = 0
    crossing = np.flatnonzero(ac[1:] <= 0)
    ac_use = ac[:crossing[0]+2] if crossing.size else ac
    return float(np.trapezoid(ac_use, dx=dt))


def fit_power_slope(g, y_col, x_col="mean_cluster_size",
                    exclude_first=True, exclude_last=False, min_points=3):
    d = g[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    d = d[(d[x_col] > 0) & (d[y_col] > 0)].sort_values(x_col)
    if exclude_first and len(d):
        d = d.iloc[1:]
    if exclude_last and len(d):
        d = d.iloc[:-1]
    if len(d) < min_points:
        return np.nan, np.nan, int(len(d))
    x, y = np.log10(d[x_col].to_numpy(float)), np.log10(d[y_col].to_numpy(float))
    slope, intercept = np.polyfit(x, y, 1)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - np.sum((y - intercept - slope*x)**2)/ss_tot if ss_tot > 0 else np.nan
    return float(slope), float(r2), int(len(d))


def standardise_gen_df(gen_df):
    g = gen_df.copy()
    for source, target in [
        ("corr_kurtosis", "kurtosis_corr"), ("kurtosis_corr", "corr_kurtosis"),
        ("MV_norm", "mean_variance_norm"), ("TAU_norm", "timescale_norm"),
        ("MV", "mean_variance"), ("TAU", "timescale"),
    ]:
        if target not in g and source in g:
            g[target] = g[source]
    return g


def run_seeded_avalanche_sparse(A_e_T, A_i_T, n, n_e, e_w, i_w,
                                theta, refractory_steps, dt, seed_node,
                                rng, max_steps):
    state = np.zeros(n, dtype=np.int16)
    state[int(seed_node)] = 1
    active_counts = np.empty(int(max_steps), dtype=np.int32)
    for step in range(int(max_steps)):
        active = state == 1
        n_active = int(active.sum())
        if n_active == 0:
            duration_steps, censored = int(step), False
            break
        active_counts[step] = n_active
        inp_e = A_e_T @ active[:n_e].astype(np.float32)
        inp_i = A_i_T @ active[n_e:].astype(np.float32)
        p_net = expit(e_w * inp_e - i_w * inp_i - theta)
        if CLAMP_ZERO_PRESYNAPTIC_INPUT:
            p_net[~((inp_e > 0) | (inp_i > 0))] = 0
        new_active = (state == 0) & (rng.random(n) < p_net)
        new_state = np.zeros_like(state)
        new_state[active] = 2
        refractory = state >= 2
        new_state[refractory] = state[refractory] + 1
        new_state[new_state > refractory_steps + 1] = 0
        new_state[new_active] = 1
        state = new_state
        if not np.any(state == 1):
            duration_steps, censored = int(step + 1), False
            break
    else:
        duration_steps, censored = int(max_steps), True

    active_counts = active_counts[:duration_steps].copy()
    extinct = bool(not censored)
    duration_s = float(duration_steps * dt)
    if active_counts.size >= 5 and np.var(active_counts) > 0:
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            warnings.filterwarnings("ignore", message="invalid value encountered in divide",
                                    category=RuntimeWarning)
            avalanche_tau_s = fn.timescale(
                active_counts[np.newaxis, :].astype(float),
                max_lag=min(3.0, max(2 * dt, duration_s / 2)), dt=float(dt))
    else:
        avalanche_tau_s = np.nan
    return {
        "size": int(active_counts.sum()),
        "duration_steps": int(duration_steps), "duration_s": duration_s,
        "lifetime_steps": int(duration_steps), "lifetime_s": duration_s,
        "peak_active": int(active_counts.max()) if active_counts.size else 0,
        "extinct": extinct, "censored": bool(censored),
        "persistent_at_cutoff": bool(censored),
        "mean_active_during_avalanche": safe_mean(active_counts),
        "avalanche_tau_s": float(avalanche_tau_s),
        "response_decay_tau_s": response_decay_timescale(active_counts, dt),
        **branching_metrics(active_counts, extinct),
    }


def empty_fit_result(n_all=0, n_tail=0, fail_reason="failed"):
    return {
        "R_truncated_vs_lognormal": np.nan, "p_truncated_vs_lognormal": np.nan,
        "xmin": np.nan, "alpha_truncated": np.nan, "lambda_truncated": np.nan,
        "mu_lognormal": np.nan, "sigma_lognormal": np.nan,
        "n_all": int(n_all), "n_tail": int(n_tail),
        "tail_fraction": float(n_tail/n_all) if n_all else np.nan,
        "fit_ok": False, "fail_reason": fail_reason,
    }


def fit_truncated_vs_lognormal(values, discrete=False, xmin=None):
    values = np.asarray(values, float)
    values = values[np.isfinite(values) & (values > 0)]
    n_all = int(values.size)
    tail = values[values >= xmin] if xmin is not None else values
    n_tail = int(tail.size)
    if n_all < MIN_N:
        return empty_fit_result(n_all, n_tail, "too_few_total")
    if n_tail < MIN_TAIL_N:
        return empty_fit_result(n_all, n_tail, "too_few_tail_pre")
    if np.unique(tail).size < MIN_UNIQUE_TAIL:
        return empty_fit_result(n_all, n_tail, "too_few_unique_tail_pre")
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            fit = powerlaw.Fit(values, discrete=discrete, xmin=xmin,
                               estimate_discrete=discrete, verbose=False)
            tail = values[values >= fit.xmin]
            n_tail = int(tail.size)
            if n_tail < MIN_TAIL_N:
                return empty_fit_result(n_all, n_tail, "too_few_tail_post")
            if np.unique(tail).size < MIN_UNIQUE_TAIL:
                return empty_fit_result(n_all, n_tail, "too_few_unique_tail_post")
            R, p = fit.distribution_compare(
                "truncated_power_law", "lognormal", normalized_ratio=True)
            out = {
                "R_truncated_vs_lognormal": float(R),
                "p_truncated_vs_lognormal": float(p), "xmin": float(fit.xmin),
                "alpha_truncated": float(fit.truncated_power_law.alpha),
                "lambda_truncated": float(fit.truncated_power_law.Lambda),
                "mu_lognormal": float(fit.lognormal.mu),
                "sigma_lognormal": float(fit.lognormal.sigma),
            }
        for key, value in out.items():
            if not np.isfinite(value):
                return empty_fit_result(n_all, n_tail, f"nonfinite_{key}")
        return {**out, "n_all": n_all, "n_tail": n_tail,
                "tail_fraction": float(n_tail/n_all), "fit_ok": True, "fail_reason": "ok"}
    except Exception as err:
        return empty_fit_result(n_all, n_tail, type(err).__name__)


METRICS = [("size", "size", True, SIZE_XMIN),
           ("duration", "lifetime_s", False, DURATION_XMIN)]


def job_stem(job):
    return (f"combo_{int(job['combo_i']):04d}_ewidx_{int(job['ew_idx']):02d}_"
            f"iwidx_{int(job['iw_idx']):02d}_phiidx_{int(job['phi_idx']):02d}_"
            f"seed_{int(job['seed_idx']):02d}")


def make_model_kwargs(phi, e_w, i_w, seed):
    pars = dict(start_dic)
    pars.update(phi=float(phi), e_w=float(e_w), i_w=float(i_w), p_ext=float(P_EXT_RUN))
    pars.pop("T", None)
    pars.pop("smoothe", None)
    valid_args = set(inspect.signature(fn.automata_EI_hiermod.__init__).parameters)
    if "phi" not in valid_args and "slope" in valid_args:
        pars["slope"] = pars.pop("phi")
    pars["seed"] = int(seed)
    return {k: v for k, v in pars.items() if k in valid_args}


def run_one_network(job):
    stem = job_stem(job)
    checkpoint = JOBDIR / f"{stem}.json"
    aval_path = TRIALDIR / f"{stem}_avalanches.csv.gz"
    icg_path = ICGDIR / f"{stem}_icg.csv.gz"
    if RESUME and checkpoint.exists():
        if (not SAVE_AVALANCHE_TRIALS or aval_path.exists()) and \
                (not SAVE_ICG_ROWS or icg_path.exists()):
            return json.loads(checkpoint.read_text())

    raw_seed = int(BASE_SEED + job["combo_i"]*100_000_000
                   + job["phi_idx"]*1_000_000 + job["seed_idx"]*10_000)
    # Legacy np.random.seed requires a uint32 seed. The 50-row
    # scan exceeds its limit; preserve original formula then wrap.
    network_seed = raw_seed % (2**32)
    model_kwargs = make_model_kwargs(job["phi"], job["e_w"], job["i_w"], network_seed)
    with threadpool_limits(limits=1):
        model = fn.automata_EI_hiermod(**model_kwargs)
        A = np.asarray(model.A, dtype=np.uint8)
        np.fill_diagonal(A, 0)
        n, n_e = int(model.n), int(model.e)
        conn = local_connectivity_metrics(A, n_e)
        A_e_T = sparse.csr_matrix(A[:n_e, :].T.astype(np.float32))
        A_i_T = sparse.csr_matrix(A[n_e:, :].T.astype(np.float32))
        rng = np.random.default_rng(network_seed + 123)
        identity = {**job, "network_seed": network_seed,
                    "e_w": float(model.e_w), "i_w": float(model.i_w),
                    "theta": float(model.theta), "p_ext": float(model.p_ext)}
        avalanche_rows = []
        for aval_i in range(N_AVALANCHES_PER_NETWORK):
            seed_node = int(rng.integers(0, n_e if SEED_ONLY_EXCITATORY else n))
            out = run_seeded_avalanche_sparse(
                A_e_T, A_i_T, n, n_e, float(model.e_w), float(model.i_w),
                float(model.theta), int(model.refractory_steps), float(model.dt),
                seed_node, rng, MAX_STEPS)
            row = {
                **identity, **conn, **out, "aval_i": aval_i, "seed_node": seed_node,
                "seed_cell_type": "E" if seed_node < n_e else "I",
                "n_neurons": n, "n_e": n_e, "dt": float(model.dt),
                "max_steps": MAX_STEPS, "cutoff_s": float(MAX_STEPS*model.dt),
                "refractory_steps": int(model.refractory_steps),
                "clamp_zero_presynaptic_input": CLAMP_ZERO_PRESYNAPTIC_INPUT,
            }
            row["lifetime_mean_active_density"] = row["size"] / (n*max(row["lifetime_steps"], 1))
            avalanche_rows.append(row)
        adf = pd.DataFrame(avalanche_rows)
        if SAVE_AVALANCHE_TRIALS:
            atomic_csv(adf, aval_path, compression="gzip")
        extinct_g = adf.loc[adf["extinct"]]
        av_summary = {
            **identity, **conn, "n_avalanches": len(adf), "avalanche_file": str(aval_path),
            "frac_extinct": float(adf["extinct"].mean()),
            "frac_persistent_at_cutoff": float(adf["persistent_at_cutoff"].mean()),
        }
        # Same original summaries, plus finite tau counts and
        # extinct-only tau, allowing transparent NaN accounting.
        for prefix, column, qs, include_cv in [
            ("size", "size", [50, 90, 95, 99], True),
            ("lifetime_s", "lifetime_s", [95, 99], True),
            ("peak_active", "peak_active", [95, 99], False),
            ("avalanche_tau_s", "avalanche_tau_s", [50, 90, 95, 99], False),
            ("response_decay_tau_s", "response_decay_tau_s", [95], False),
        ]:
            av_summary[f"mean_{prefix}"] = safe_mean(adf[column])
            for q in qs:
                key = f"median_{prefix}" if q == 50 else f"p{q}_{prefix}"
                av_summary[key] = safe_percentile(adf[column], q)
            if include_cv:
                av_summary[f"cv_{prefix}"] = safe_cv(adf[column])
        for col in ["size", "lifetime_s"]:
            for q in [95, 99]:
                av_summary[f"p{q}_{col}_extinct_only"] = safe_percentile(extinct_g[col], q)
        av_summary["mean_avalanche_tau_s_extinct_only"] = safe_mean(extinct_g["avalanche_tau_s"])
        av_summary["n_valid_avalanche_tau"] = int(safe_values(adf["avalanche_tau_s"]).size)
        av_summary["n_valid_avalanche_tau_extinct_only"] = int(safe_values(extinct_g["avalanche_tau_s"]).size)

        # Preserve original run_model, burn-in and smoothing behaviour.
        spikes, pop_rate = fn.run_model(model, T=T_RUN)
        dt = float(model.dt)
        burn = int(round(BURN_IN_S / dt))
        spikes_use = spikes[:, burn:]
        pop_rate_use = np.asarray(pop_rate[burn:], float)
        active_counts = spikes_use.sum(axis=0).astype(float)
        rho = active_counts / n
        mean_rate = float(spikes_use.mean()/dt)
        spikes_smooth = fn.exp_smooth_spikes(spikes, dt=dt, tau=SMOOTHE)
        # ICG uses full smoothed recording, as in supplied run.
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            warnings.filterwarnings("ignore", message="invalid value encountered in divide",
                                    category=RuntimeWarning)
            metric_row, gen_df = fn.compute_icg_metrics(spikes=spikes_smooth, dt=dt)
        gen_df = standardise_gen_df(gen_df)
        alpha, ar2, an = fit_power_slope(gen_df, "mean_variance_norm")
        beta, br2, bn = fit_power_slope(gen_df, "timescale_norm")
        kurt, kr2, kn = fit_power_slope(gen_df, "kurtosis_corr")
        score = (abs(alpha-TARGET_MV) + 2*abs(beta-TARGET_TAU)
                 if np.isfinite(alpha) and np.isfinite(beta) else np.nan)
        for key, value in {**identity, "smoothe": SMOOTHE, "mean_rate_hz": mean_rate}.items():
            gen_df[key] = value
        if SAVE_ICG_ROWS:
            atomic_csv(gen_df, icg_path, compression="gzip")
        icg_summary = {
            **identity, **conn, "icg_file": str(icg_path),
            "T": T_RUN, "burn_in_s": BURN_IN_S, "smoothe": SMOOTHE,
            "mean_rate_hz": mean_rate,
            "mean_rate_hz_pop_rate": float(pop_rate_use.mean()),
            "median_pop_rate_hz": float(np.median(pop_rate_use)),
            "max_pop_rate_hz": float(pop_rate_use.max()),
            "mean_rho": float(rho.mean()), "std_rho": float(rho.std()),
            "var_rho": float(rho.var()), "susceptibility_N_var_rho": float(n*rho.var()),
            "pop_rate_var": float(pop_rate_use.var()), "pop_rate_std": float(pop_rate_use.std()),
            "pop_rate_cv": safe_cv(pop_rate_use), "active_count_cv": safe_cv(active_counts),
            "autocorr_tau_rho_s": pop_autocorr_tau(rho, dt),
            "autocorr_tau_pop_rate_s": pop_autocorr_tau(pop_rate_use, dt),
            "silent_frac": float(np.mean(active_counts == 0)),
            "active_frac": float(np.mean(active_counts > 0)),
            "mean_active_count": float(active_counts.mean()),
            "p95_active_count": float(np.percentile(active_counts, 95)),
            "p99_active_count": float(np.percentile(active_counts, 99)),
            "max_active_count": float(active_counts.max()),
            "MV_alpha": alpha, "MV_r2": ar2, "MV_n_points": an,
            "TAU_beta": beta, "TAU_r2": br2, "TAU_n_points": bn,
            "KURT_slope": kurt, "KURT_r2": kr2, "KURT_n_points": kn, "score": score,
        }
        # Fit after dynamics, so fitting cannot affect simulation RNG.
        llr_rows = []
        if SAVE_SEED_LLR:
            selected = extinct_g if EXTINCT_ONLY else adf
            for metric, col, discrete, xmin in METRICS:
                res = fit_truncated_vs_lognormal(selected[col].to_numpy(float), discrete, xmin)
                llr_rows.append({**identity, "metric": metric, **res})
        result = {"avalanche": av_summary, "icg": icg_summary, "llr": llr_rows}
        atomic_json(result, checkpoint)  # written only after all stages finish
        return result


def run_pooled_cell(cell):
    rows = []
    metadata = {k: cell[0][k] for k in KEYS}
    values = {metric: [] for metric, *_ in METRICS}
    n_files = 0
    for job in cell:
        path = TRIALDIR / f"{job_stem(job)}_avalanches.csv.gz"
        if not path.exists():
            raise FileNotFoundError(f"Required pooled-fit input missing: {path}")
        d = pd.read_csv(path, usecols=["size", "lifetime_s", "extinct"])
        if EXTINCT_ONLY:
            d = d.loc[d["extinct"]]
        n_files += 1
        for metric, col, *_ in METRICS:
            x = d[col].to_numpy(float)
            values[metric].append(x[np.isfinite(x) & (x > 0)])
    with threadpool_limits(limits=1):
        for metric, _, discrete, xmin in METRICS:
            x = np.concatenate(values[metric])
            fit = fit_truncated_vs_lognormal(x, discrete, xmin)
            rows.append({**metadata, "metric": metric, "n_files": n_files, **fit})
    return rows


def aggregate(df, cols):
    agg = {"n": ("seed_idx", "count")}
    for col in cols:
        agg[f"{col}_mean"] = (col, safe_mean)
        agg[f"{col}_sem"] = (col, sem)
        agg[f"{col}_n_valid"] = (col, lambda x: len(safe_values(x)))
    return df.groupby(KEYS, as_index=False).agg(**agg).sort_values(["combo_i", "phi"])


def peak_width(phi, y, frac=0.5):
    phi, y = np.asarray(phi, float), np.asarray(y, float)
    keep = np.isfinite(phi) & np.isfinite(y)
    phi, y = phi[keep], y[keep]
    if y.size < 3:
        return np.nan, np.nan, np.nan, 0
    j = int(np.argmax(y))
    if y[j] <= 0:
        return float(phi[j]), float(y[j]), np.nan, 0
    above = y >= frac*y[j]
    return float(phi[j]), float(y[j]), float(np.ptp(phi[above])), int(above.sum())


def save_network_tables(results, partial=False):
    av = pd.DataFrame([r["avalanche"] for r in results])
    icg = pd.DataFrame([r["icg"] for r in results])
    llr = pd.DataFrame([row for r in results for row in r["llr"]])
    suffix = "_partial" if partial else ""
    for df, name in [(av, "avalanche_network_summary"), (icg, "icg_seed_summary")]:
        df = df.sort_values(["combo_i", "phi", "seed_idx"])
        atomic_csv(df, OUTDIR / f"{name}{suffix}.csv")
    if len(llr):
        atomic_csv(llr.sort_values(["combo_i", "phi", "seed_idx", "metric"]),
                   LLRDIR / f"normalized_llr_by_network{suffix}.csv")
    return av, icg, llr


def main():
    for directory in [OUTDIR, TRIALDIR, ICGDIR, JOBDIR, LLRDIR]:
        directory.mkdir(parents=True, exist_ok=True)
    config = {
        "scan_type": "e_w_phi_grid_fixed_i_w", "start_dic": start_dic,
        "N_WORKERS": N_WORKERS, "N_SEEDS": N_SEEDS,
        "N_AVALANCHES_PER_NETWORK": N_AVALANCHES_PER_NETWORK,
        "MAX_STEPS": MAX_STEPS, "BASE_SEED": BASE_SEED,
        "EW_VALUES": EW_VALUES.tolist(), "IW_VALUES": IW_VALUES.tolist(),
        "I_W_FIXED": I_W_FIXED, "PHI_VALUES": PHI_VALUES.tolist(),
        "P_EXT_RUN": P_EXT_RUN, "SEED_ONLY_EXCITATORY": SEED_ONLY_EXCITATORY,
        "CLAMP_ZERO_PRESYNAPTIC_INPUT": CLAMP_ZERO_PRESYNAPTIC_INPUT,
        "T_RUN": T_RUN, "SMOOTHE": SMOOTHE, "BURN_IN_S": BURN_IN_S,
        "TARGET_MV": TARGET_MV, "TARGET_TAU": TARGET_TAU,
        "SIZE_XMIN": SIZE_XMIN, "DURATION_XMIN": DURATION_XMIN,
        "EXTINCT_ONLY": EXTINCT_ONLY, "MIN_N": MIN_N,
        "MIN_TAIL_N": MIN_TAIL_N, "MIN_UNIQUE_TAIL": MIN_UNIQUE_TAIL,
        "SAVE_SEED_LLR": SAVE_SEED_LLR, "RUN_POOLED_LLR": RUN_POOLED_LLR,
        "SAVE_AVALANCHE_TRIALS": SAVE_AVALANCHE_TRIALS, "SAVE_ICG_ROWS": SAVE_ICG_ROWS,
        "network_seed_rule": "original_formula_modulo_2**32",
        "score_definition": "abs(alpha-1.5)+2*abs(beta-0.2)",
        "icg_includes_burn_in": True,
    }
    cfg_path = OUTDIR / "config.json"
    if cfg_path.exists():
        prior = json.loads(cfg_path.read_text())
        if prior != config:
            raise ValueError("Existing output config differs. Choose a new OUTDIR.")
    else:
        atomic_json(config, cfg_path)
    jobs = []
    combo_i = 0
    for ew_idx, e_w in enumerate(EW_VALUES):
        for iw_idx, i_w in enumerate(IW_VALUES):
            for phi_idx, phi in enumerate(PHI_VALUES):
                for seed_idx in range(N_SEEDS):
                    jobs.append(dict(combo_i=combo_i, ew_idx=ew_idx, iw_idx=iw_idx,
                                     phi_idx=phi_idx, seed_idx=seed_idx,
                                     e_w=float(e_w), i_w=float(i_w), phi=float(phi)))
            combo_i += 1
    design = pd.DataFrame(jobs)
    atomic_csv(design, OUTDIR / "grid_design.csv")
    # Validate wrapped seeds do not collide within this design.
    seeds = [(BASE_SEED+j["combo_i"]*100_000_000+j["phi_idx"]*1_000_000+
              j["seed_idx"]*10_000) % (2**32) for j in jobs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Network seed collision.")
    print("OUTDIR:", OUTDIR, flush=True)
    print(f"{N_EW} e_w x {N_PHI} phi x {N_SEEDS} seeds = {len(jobs):,} jobs", flush=True)
    print(f"Avalanches: {len(jobs)*N_AVALANCHES_PER_NETWORK:,}; workers: {N_WORKERS}", flush=True)
    print("Grid spacings:", EW_VALUES[1]-EW_VALUES[0], PHI_VALUES[1]-PHI_VALUES[0], flush=True)
    print("Score uses original weighted definition.", flush=True)
    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError("This script preserves the supplied Linux/fork execution mode.")
    ctx = mp.get_context("fork")
    results = []
    with ctx.Pool(processes=N_WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(run_one_network, jobs, chunksize=1),
                           total=len(jobs), desc="Networks + tau + ICG + seed LLR"):
            results.append(result)
            if len(results) % 100 == 0:
                save_network_tables(results, partial=True)
    av, icg, llr = save_network_tables(results)

    av_cols = [
        "frac_extinct", "frac_persistent_at_cutoff", "mean_size", "p95_size", "p99_size",
        "mean_lifetime_s", "p95_lifetime_s", "p99_lifetime_s", "mean_peak_active",
        "p95_peak_active", "mean_avalanche_tau_s", "median_avalanche_tau_s",
        "p95_avalanche_tau_s", "p99_avalanche_tau_s",
        "mean_response_decay_tau_s", "p95_response_decay_tau_s",
        "mean_avalanche_tau_s_extinct_only",
    ]
    icg_cols = [
        "mean_rate_hz", "mean_rho", "var_rho", "susceptibility_N_var_rho",
        "pop_rate_cv", "autocorr_tau_rho_s", "autocorr_tau_pop_rate_s",
        "MV_alpha", "MV_r2", "TAU_beta", "TAU_r2", "KURT_slope", "KURT_r2", "score",
    ]
    av_agg, icg_agg = aggregate(av, av_cols), aggregate(icg, icg_cols)
    atomic_csv(av_agg, OUTDIR / "avalanche_combo_phi_summary.csv")
    atomic_csv(icg_agg, OUTDIR / "icg_combo_phi_summary.csv")

    if len(llr):
        successful = llr.loc[llr["fit_ok"]].copy()
        grouped = successful.groupby(KEYS+["metric"], as_index=False).agg(
            R_normalized_mean=("R_truncated_vs_lognormal", safe_mean),
            R_normalized_sem=("R_truncated_vs_lognormal", sem),
            n_successful_seed_fits=("R_truncated_vs_lognormal", "count"),
        )
        total = llr.groupby(KEYS+["metric"], as_index=False).agg(
            n_seed_fits=("fit_ok", "size"), n_failed_seed_fits=("fit_ok", lambda x: int((~x).sum())))
        summary = total.merge(grouped, on=KEYS+["metric"], how="left")
        summary["n_successful_seed_fits"] = summary["n_successful_seed_fits"].fillna(0).astype(int)
        atomic_csv(summary, LLRDIR / "normalized_llr_summary_by_phi.csv")
        for metric, *_ in METRICS:
            d = summary.loc[summary["metric"] == metric].rename(
                columns={"R_normalized_mean": "R_mean", "R_normalized_sem": "R_sem"})
            atomic_csv(d.sort_values(["e_w", "phi"]),
                       LLRDIR / f"{metric}_mean_llr_phi_by_ew_fixed_iw_safe.csv")
            atomic_csv(d.sort_values(["phi", "e_w"]),
                       LLRDIR / f"{metric}_mean_llr_ew_by_phi_fixed_iw_safe.csv")

    combo_rows = []
    for combo, g in av_agg.groupby("combo_i", sort=True):
        g = g.sort_values("phi")
        ig = icg_agg.loc[icg_agg["combo_i"] == combo]
        pp, pt, width, nabove = peak_width(g["phi"], g["mean_avalanche_tau_s_mean"])
        finite = ig["score_mean"].dropna()
        combo_rows.append({
            **{k: g[k].iloc[0].item() if hasattr(g[k].iloc[0], "item") else g[k].iloc[0]
               for k in ["combo_i", "ew_idx", "iw_idx", "e_w", "i_w"]},
            "autocorr_peak_phi": pp, "autocorr_peak_tau_s": pt,
            "autocorr_peak_width_phi": width, "autocorr_n_phi_above_halfmax": nabove,
            "mean_persistent_fraction": safe_mean(g["frac_persistent_at_cutoff_mean"]),
            "max_persistent_fraction": float(g["frac_persistent_at_cutoff_mean"].max()),
            "mean_avalanche_tau_s_across_phi": safe_mean(g["mean_avalanche_tau_s_mean"]),
            "max_avalanche_tau_s_across_phi": float(g["mean_avalanche_tau_s_mean"].max()),
            "best_scaling_score": float(finite.min()) if len(finite) else np.nan,
            "best_scaling_phi": float(ig.loc[finite.idxmin(), "phi"]) if len(finite) else np.nan,
            "max_scaling_score": float(finite.max()) if len(finite) else np.nan,
            "max_scaling_phi": float(ig.loc[finite.idxmax(), "phi"]) if len(finite) else np.nan,
            "max_MV_alpha": float(ig["MV_alpha_mean"].max()),
            "max_TAU_beta": float(ig["TAU_beta_mean"].max()),
            "max_KURT_slope": float(ig["KURT_slope_mean"].max()),
            "max_autocorr_tau_rho_s": float(ig["autocorr_tau_rho_s_mean"].max()),
        })
    ranked = pd.DataFrame(combo_rows).sort_values(
        ["autocorr_peak_width_phi", "autocorr_peak_tau_s"], ascending=[True, False])
    atomic_csv(ranked, OUTDIR / "combo_summary_ranked.csv")

    finite_grid = icg_agg.loc[np.isfinite(icg_agg["score_mean"])].copy()
    extrema = []
    if len(finite_grid):
        for label, index in [("minimum", finite_grid["score_mean"].idxmin()),
                             ("maximum", finite_grid["score_mean"].idxmax())]:
            row = finite_grid.loc[index]
            extrema.append({"extremum": label,
                            **{key: row[key] for key in KEYS},
                            "score_mean": row["score_mean"],
                            "score_sem": row["score_sem"]})
    atomic_csv(pd.DataFrame(extrema, columns=["extremum"]+KEYS+["score_mean", "score_sem"]),
               OUTDIR / "score_grid_extrema.csv")

    if RUN_POOLED_LLR:
        cells = [g.to_dict("records") for _, g in design.groupby(["e_w", "i_w", "phi"], sort=True)]
        pooled_rows = []
        with ctx.Pool(processes=N_WORKERS) as pool:
            for rows in tqdm(pool.imap_unordered(run_pooled_cell, cells, chunksize=1),
                             total=len(cells), desc="Original pooled-seed LLR fits"):
                pooled_rows.extend(rows)
                if len(pooled_rows) % 100 == 0:
                    atomic_csv(pd.DataFrame(pooled_rows), LLRDIR / "pooled_llr_partial.csv")
        pooled = pd.DataFrame(pooled_rows).sort_values(["e_w", "phi", "metric"])
        atomic_csv(pooled, LLRDIR / FIT_NAME)
        # Pooled fits have one result per cell and NO seed-level SEM.

    print(ranked.head(20).to_string(index=False))
    print("Done. Saved simulations, tau, ICG, seed LLRs and pooled LLRs to:", OUTDIR)


if __name__ == "__main__":
    main()

