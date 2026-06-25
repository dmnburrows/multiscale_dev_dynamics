# ============================================================
# GRID e_w / i_w SEARCH
# Sparse seeded avalanches + per-avalanche autocorr
# + spontaneous ICG/static-dynamic scaling on same network
# Warning-safe ICG block
# ============================================================

import os
import json
import inspect
import warnings
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from tqdm.auto import tqdm
from threadpoolctl import threadpool_limits

import icg_functions as fn


# ============================================================
# Thread limits
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# ============================================================
# Settings
# ============================================================

OUTDIR = Path(
    "/home/dburrows/DATA/BLNDEV-WILDTYPE/"
    "grid_ew_iw_sparse_avalanche_icg_20x20_20phis_5seeds_10000avals"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

TRIALDIR = OUTDIR / "avalanche_trials_by_network"
TRIALDIR.mkdir(exist_ok=True)

ICGDIR = OUTDIR / "icg_rows_by_network"
ICGDIR.mkdir(exist_ok=True)

N_WORKERS = 30
N_SEEDS = 5
N_AVALANCHES_PER_NETWORK = 10000
MAX_STEPS = 1000
BASE_SEED = 91919

E_W_RANGE = (0.5, 20.0)
I_W_RANGE = (0.0, 150.0)

N_EW = 20
N_IW = 20

EW_VALUES = np.linspace(E_W_RANGE[0], E_W_RANGE[1], N_EW)
IW_VALUES = np.linspace(I_W_RANGE[0], I_W_RANGE[1], N_IW)

PHI_VALUES = np.linspace(1.5, 6.0, 20)

start_dic = {
    "n_neurons": 2000,
    "T": 10,
    "dt": 0.01,
    "refractory_steps": 2,
    "ei_ratio": 0.2,
    "e_w": 11.7,
    "i_w": 22.4,
    "theta": 8.44,
    "p_ext": 0.00,
    "phi": 4.2,
    "smoothe": 0.05,
}

# Force avalanche/spontaneous p_ext to zero
P_EXT_RUN = 0.0

SEED_ONLY_EXCITATORY = True
CLAMP_ZERO_PRESYNAPTIC_INPUT = True

DT = float(start_dic["dt"])
T_RUN = float(start_dic["T"])
SMOOTHE = float(start_dic["smoothe"])
BURN_IN_S = 2.0

TARGET_MV = 1.50
TARGET_TAU = 0.20

SAVE_AVALANCHE_TRIALS = True
SAVE_ICG_ROWS = True


# ============================================================
# Config
# ============================================================

config = {
    "N_WORKERS": N_WORKERS,
    "N_SEEDS": N_SEEDS,
    "N_AVALANCHES_PER_NETWORK": N_AVALANCHES_PER_NETWORK,
    "MAX_STEPS": MAX_STEPS,
    "BASE_SEED": BASE_SEED,
    "EW_VALUES": EW_VALUES.tolist(),
    "IW_VALUES": IW_VALUES.tolist(),
    "PHI_VALUES": PHI_VALUES.tolist(),
    "P_EXT_RUN": P_EXT_RUN,
    "SEED_ONLY_EXCITATORY": SEED_ONLY_EXCITATORY,
    "CLAMP_ZERO_PRESYNAPTIC_INPUT": CLAMP_ZERO_PRESYNAPTIC_INPUT,
    "start_dic": start_dic,
}

with open(OUTDIR / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print("OUTDIR:", OUTDIR)
print("Total ew/iw combos:", len(EW_VALUES) * len(IW_VALUES))
print("Total networks:", len(EW_VALUES) * len(IW_VALUES) * len(PHI_VALUES) * N_SEEDS)
print("Total avalanches:", len(EW_VALUES) * len(IW_VALUES) * len(PHI_VALUES) * N_SEEDS * N_AVALANCHES_PER_NETWORK)


# ============================================================
# Helpers
# ============================================================

def safe_values(x):
    return (
        pd.Series(x)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(float)
    )


def sem(x):
    x = safe_values(x)
    if x.size <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(x.size))


def safe_mean(x):
    x = safe_values(x)
    return float(np.mean(x)) if x.size else np.nan


def safe_percentile(x, q):
    x = safe_values(x)
    return float(np.percentile(x, q)) if x.size else np.nan


def safe_cv(x):
    x = safe_values(x)
    if x.size < 2:
        return np.nan
    mu = float(np.mean(x))
    if mu <= 0 or not np.isfinite(mu):
        return np.nan
    return float(np.std(x, ddof=1) / mu)


def response_decay_timescale(active_counts, dt):
    x = np.asarray(active_counts, dtype=float)

    if x.size == 0:
        return np.nan

    peak = float(np.max(x))
    if peak <= 0:
        return 0.0

    peak_idx = int(np.argmax(x))
    threshold = peak / np.e

    post_peak = x[peak_idx:]
    below = np.where(post_peak <= threshold)[0]

    if below.size:
        return float(below[0] * dt)

    return float((post_peak.size - 1) * dt)


def branching_metrics(active_counts, extinct):
    counts = np.asarray(active_counts, dtype=float)

    if counts.size == 0:
        return {
            "sigma_origin": np.nan,
            "br_reg_slope": np.nan,
            "br_reg_intercept": np.nan,
            "br_reg_r2": np.nan,
            "br_n_pairs": 0,
        }

    if extinct:
        x = counts
        y = np.concatenate([counts[1:], np.array([0.0])])
    else:
        if counts.size < 2:
            return {
                "sigma_origin": np.nan,
                "br_reg_slope": np.nan,
                "br_reg_intercept": np.nan,
                "br_reg_r2": np.nan,
                "br_n_pairs": 0,
            }

        x = counts[:-1]
        y = counts[1:]

    keep = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = x[keep]
    y = y[keep]

    if x.size < 2:
        return {
            "sigma_origin": np.nan,
            "br_reg_slope": np.nan,
            "br_reg_intercept": np.nan,
            "br_reg_r2": np.nan,
            "br_n_pairs": int(x.size),
        }

    denom = float(np.sum(x * x))
    sigma_origin = float(np.sum(x * y) / denom) if denom > 0 else np.nan

    X = np.column_stack([np.ones_like(x), x])

    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        intercept = float(beta[0])
        slope = float(beta[1])

        y_hat = X @ beta
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))

        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    except Exception:
        intercept = np.nan
        slope = np.nan
        r2 = np.nan

    return {
        "sigma_origin": sigma_origin,
        "br_reg_slope": slope,
        "br_reg_intercept": intercept,
        "br_reg_r2": r2,
        "br_n_pairs": int(x.size),
    }


def local_connectivity_metrics(A, n_e):
    A = np.asarray(A, dtype=np.uint8)

    A_e_all = A[:n_e, :]
    A_ee = A[:n_e, :n_e]

    return {
        "edge_count": int(A.sum()),
        "mean_out_degree": float(A.sum(axis=1).mean()),
        "std_out_degree": float(A.sum(axis=1).std()),
        "mean_e_out_degree_all": float(A_e_all.sum(axis=1).mean()),
        "mean_e_out_degree_e": float(A_ee.sum(axis=1).mean()),
        "edge_density": float(
            A.sum() / max(A.shape[0] * (A.shape[0] - 1), 1)
        ),
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

    ac = np.empty(max_lag + 1, dtype=float)
    ac[0] = 1.0

    for lag in range(1, max_lag + 1):
        ac[lag] = np.sum(x[:-lag] * x[lag:]) / denom

    ac[~np.isfinite(ac)] = 0.0

    crossing = np.where(ac[1:] <= 0)[0]

    if crossing.size > 0:
        ac_use = ac[:crossing[0] + 2]
    else:
        ac_use = ac

    if len(ac_use) < 2:
        return 0.0

    return float(np.trapezoid(ac_use, dx=dt))


def fit_power_slope(
    g,
    y_col,
    x_col="mean_cluster_size",
    exclude_first=True,
    exclude_last=False,
    min_points=3,
):
    dfit = g[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    dfit = dfit[(dfit[x_col] > 0) & (dfit[y_col] > 0)].sort_values(x_col)

    if exclude_first and len(dfit) > 0:
        dfit = dfit.iloc[1:]

    if exclude_last and len(dfit) > 0:
        dfit = dfit.iloc[:-1]

    if len(dfit) < min_points:
        return np.nan, np.nan, int(len(dfit))

    logx = np.log10(dfit[x_col].to_numpy(float))
    logy = np.log10(dfit[y_col].to_numpy(float))

    slope, intercept = np.polyfit(logx, logy, 1)
    pred = intercept + slope * logx

    ss_res = np.sum((logy - pred) ** 2)
    ss_tot = np.sum((logy - np.mean(logy)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return float(slope), float(r2), int(len(dfit))


def standardise_gen_df(gen_df):
    gen_df = gen_df.copy()

    if "corr_kurtosis" in gen_df.columns and "kurtosis_corr" not in gen_df.columns:
        gen_df["kurtosis_corr"] = gen_df["corr_kurtosis"]

    if "kurtosis_corr" in gen_df.columns and "corr_kurtosis" not in gen_df.columns:
        gen_df["corr_kurtosis"] = gen_df["kurtosis_corr"]

    if "mean_variance_norm" not in gen_df.columns and "MV_norm" in gen_df.columns:
        gen_df["mean_variance_norm"] = gen_df["MV_norm"]

    if "timescale_norm" not in gen_df.columns and "TAU_norm" in gen_df.columns:
        gen_df["timescale_norm"] = gen_df["TAU_norm"]

    if "mean_variance" not in gen_df.columns and "MV" in gen_df.columns:
        gen_df["mean_variance"] = gen_df["MV"]

    if "timescale" not in gen_df.columns and "TAU" in gen_df.columns:
        gen_df["timescale"] = gen_df["TAU"]

    return gen_df


# ============================================================
# Sparse avalanche runner
# ============================================================

def run_seeded_avalanche_sparse(
    A_e_T,
    A_i_T,
    n,
    n_e,
    e_w,
    i_w,
    theta,
    refractory_steps,
    dt,
    seed_node,
    rng,
    max_steps,
):
    state = np.zeros(n, dtype=np.int16)
    state[int(seed_node)] = 1

    active_counts = np.empty(int(max_steps), dtype=np.int32)

    for step in range(int(max_steps)):
        active = state == 1
        n_active = int(active.sum())

        if n_active == 0:
            duration_steps = int(step)
            censored = False
            break

        active_counts[step] = n_active

        active_e = active[:n_e].astype(np.float32)
        active_i = active[n_e:].astype(np.float32)

        inp_e = A_e_T @ active_e
        inp_i = A_i_T @ active_i

        net = (e_w * inp_e) - (i_w * inp_i)
        p_net = expit(net - theta)

        if CLAMP_ZERO_PRESYNAPTIC_INPUT:
            has_active_presynaptic_input = (inp_e > 0) | (inp_i > 0)
            p_net[~has_active_presynaptic_input] = 0.0

        quiescent = state == 0
        new_active = quiescent & (rng.random(n) < p_net)

        new_state = np.zeros_like(state)

        new_state[active] = 2

        refractory = state >= 2
        new_state[refractory] = state[refractory] + 1

        done_refrac = new_state > (refractory_steps + 1)
        new_state[done_refrac] = 0

        new_state[new_active] = 1
        state = new_state

        if not np.any(state == 1):
            duration_steps = int(step + 1)
            censored = False
            break

    else:
        duration_steps = int(max_steps)
        censored = True

    active_counts = active_counts[:duration_steps].copy()
    extinct = bool(not censored)

    size = int(np.sum(active_counts))
    duration_s = float(duration_steps * dt)
    peak_active = int(np.max(active_counts)) if active_counts.size else 0

    br = branching_metrics(active_counts, extinct)

    if active_counts.size >= 5 and np.var(active_counts) > 0:
        avalanche_tau_s = fn.timescale(
            active_counts[np.newaxis, :].astype(float),
            max_lag=min(3.0, max(2 * dt, duration_s / 2)),
            dt=float(dt),
        )
    else:
        avalanche_tau_s = np.nan

    response_decay_tau_s = response_decay_timescale(
        active_counts=active_counts,
        dt=float(dt),
    )

    return {
        "size": size,
        "duration_steps": int(duration_steps),
        "duration_s": duration_s,
        "lifetime_steps": int(duration_steps),
        "lifetime_s": duration_s,
        "peak_active": peak_active,

        "extinct": extinct,
        "censored": bool(censored),
        "persistent_at_cutoff": bool(censored),

        "mean_active_during_avalanche": (
            float(np.mean(active_counts))
            if active_counts.size
            else np.nan
        ),

        "avalanche_tau_s": float(avalanche_tau_s),
        "response_decay_tau_s": float(response_decay_tau_s),

        "sigma_origin": float(br["sigma_origin"]),
        "br_reg_slope": float(br["br_reg_slope"]),
        "br_reg_intercept": float(br["br_reg_intercept"]),
        "br_reg_r2": float(br["br_reg_r2"]),
        "br_n_pairs": int(br["br_n_pairs"]),
    }


# ============================================================
# Build model kwargs
# ============================================================

def make_model_kwargs(phi, e_w, i_w, seed):
    pars = dict(start_dic)

    pars["phi"] = float(phi)
    pars["e_w"] = float(e_w)
    pars["i_w"] = float(i_w)
    pars["p_ext"] = float(P_EXT_RUN)

    pars.pop("T", None)
    pars.pop("smoothe", None)

    sig = inspect.signature(fn.automata_EI_hiermod.__init__)
    valid_args = set(sig.parameters.keys())

    if "phi" not in valid_args and "slope" in valid_args:
        pars["slope"] = pars.pop("phi")

    pars["seed"] = int(seed)

    return {k: v for k, v in pars.items() if k in valid_args}


# ============================================================
# One network job: generate once, do avalanches + spontaneous ICG
# ============================================================

def run_one_network(job):
    combo_i = int(job["combo_i"])
    ew_idx = int(job["ew_idx"])
    iw_idx = int(job["iw_idx"])
    phi_idx = int(job["phi_idx"])
    seed_idx = int(job["seed_idx"])

    e_w = float(job["e_w"])
    i_w = float(job["i_w"])
    phi = float(job["phi"])

    network_seed = int(
        BASE_SEED
        + combo_i * 100_000_000
        + phi_idx * 1_000_000
        + seed_idx * 10_000
    )

    model_kwargs = make_model_kwargs(
        phi=phi,
        e_w=e_w,
        i_w=i_w,
        seed=network_seed,
    )

    # --------------------------------------------------------
    # Generate network once
    # --------------------------------------------------------
    with threadpool_limits(limits=1):
        model = fn.automata_EI_hiermod(**model_kwargs)

    A = np.asarray(model.A, dtype=np.uint8)
    np.fill_diagonal(A, 0)

    n = int(model.n)
    n_e = int(model.e)

    conn = local_connectivity_metrics(A, n_e=n_e)

    A_e_T = sparse.csr_matrix(A[:n_e, :].T.astype(np.float32))
    A_i_T = sparse.csr_matrix(A[n_e:, :].T.astype(np.float32))

    rng = np.random.default_rng(network_seed + 123)

    # --------------------------------------------------------
    # Sparse seeded avalanches
    # --------------------------------------------------------
    avalanche_rows = []

    for aval_i in range(N_AVALANCHES_PER_NETWORK):
        if SEED_ONLY_EXCITATORY:
            seed_node = int(rng.integers(0, n_e))
            seed_cell_type = "E"
        else:
            seed_node = int(rng.integers(0, n))
            seed_cell_type = "E" if seed_node < n_e else "I"

        out = run_seeded_avalanche_sparse(
            A_e_T=A_e_T,
            A_i_T=A_i_T,
            n=n,
            n_e=n_e,
            e_w=float(model.e_w),
            i_w=float(model.i_w),
            theta=float(model.theta),
            refractory_steps=int(model.refractory_steps),
            dt=float(model.dt),
            seed_node=seed_node,
            rng=rng,
            max_steps=MAX_STEPS,
        )

        row = {
            "combo_i": combo_i,
            "ew_idx": ew_idx,
            "iw_idx": iw_idx,
            "phi_idx": phi_idx,
            "seed_idx": seed_idx,
            "network_seed": network_seed,

            "aval_i": int(aval_i),
            "seed_node": seed_node,
            "seed_cell_type": seed_cell_type,

            "n_neurons": n,
            "n_e": n_e,
            "dt": float(model.dt),
            "max_steps": int(MAX_STEPS),
            "cutoff_s": float(MAX_STEPS * model.dt),

            "p_ext": float(model.p_ext),
            "phi": phi,
            "theta": float(model.theta),
            "e_w": float(model.e_w),
            "i_w": float(model.i_w),
            "refractory_steps": int(model.refractory_steps),
            "clamp_zero_presynaptic_input": bool(CLAMP_ZERO_PRESYNAPTIC_INPUT),
        }

        row.update(conn)
        row.update(out)

        row["lifetime_mean_active_density"] = (
            row["size"] / (n * max(row["lifetime_steps"], 1))
        )

        avalanche_rows.append(row)

    avalanche_df = pd.DataFrame(avalanche_rows)

    aval_path = TRIALDIR / (
        f"combo_{combo_i:04d}_ewidx_{ew_idx:02d}_iwidx_{iw_idx:02d}_"
        f"phiidx_{phi_idx:02d}_seed_{seed_idx:02d}_avalanches.csv.gz"
    )

    if SAVE_AVALANCHE_TRIALS:
        avalanche_df.to_csv(aval_path, index=False, compression="gzip")

    # --------------------------------------------------------
    # Avalanche network summary
    # --------------------------------------------------------
    extinct_g = avalanche_df[avalanche_df["extinct"]]

    avalanche_summary_row = {
        "combo_i": combo_i,
        "ew_idx": ew_idx,
        "iw_idx": iw_idx,
        "phi_idx": phi_idx,
        "seed_idx": seed_idx,
        "network_seed": network_seed,

        "e_w": float(model.e_w),
        "i_w": float(model.i_w),
        "phi": phi,
        "theta": float(model.theta),
        "p_ext": float(model.p_ext),

        "n_avalanches": int(len(avalanche_df)),
        "avalanche_file": str(aval_path),

        "frac_extinct": float(avalanche_df["extinct"].mean()),
        "frac_persistent_at_cutoff": float(avalanche_df["persistent_at_cutoff"].mean()),

        "mean_size": safe_mean(avalanche_df["size"]),
        "median_size": safe_percentile(avalanche_df["size"], 50),
        "p90_size": safe_percentile(avalanche_df["size"], 90),
        "p95_size": safe_percentile(avalanche_df["size"], 95),
        "p99_size": safe_percentile(avalanche_df["size"], 99),
        "cv_size": safe_cv(avalanche_df["size"]),

        "mean_lifetime_s": safe_mean(avalanche_df["lifetime_s"]),
        "p95_lifetime_s": safe_percentile(avalanche_df["lifetime_s"], 95),
        "p99_lifetime_s": safe_percentile(avalanche_df["lifetime_s"], 99),
        "cv_lifetime_s": safe_cv(avalanche_df["lifetime_s"]),

        "mean_peak_active": safe_mean(avalanche_df["peak_active"]),
        "p95_peak_active": safe_percentile(avalanche_df["peak_active"], 95),
        "p99_peak_active": safe_percentile(avalanche_df["peak_active"], 99),

        "mean_avalanche_tau_s": safe_mean(avalanche_df["avalanche_tau_s"]),
        "median_avalanche_tau_s": safe_percentile(avalanche_df["avalanche_tau_s"], 50),
        "p90_avalanche_tau_s": safe_percentile(avalanche_df["avalanche_tau_s"], 90),
        "p95_avalanche_tau_s": safe_percentile(avalanche_df["avalanche_tau_s"], 95),
        "p99_avalanche_tau_s": safe_percentile(avalanche_df["avalanche_tau_s"], 99),

        "mean_response_decay_tau_s": safe_mean(avalanche_df["response_decay_tau_s"]),
        "p95_response_decay_tau_s": safe_percentile(avalanche_df["response_decay_tau_s"], 95),

        "p95_size_extinct_only": (
            safe_percentile(extinct_g["size"], 95) if len(extinct_g) else np.nan
        ),
        "p99_size_extinct_only": (
            safe_percentile(extinct_g["size"], 99) if len(extinct_g) else np.nan
        ),
        "p95_lifetime_s_extinct_only": (
            safe_percentile(extinct_g["lifetime_s"], 95) if len(extinct_g) else np.nan
        ),
        "p99_lifetime_s_extinct_only": (
            safe_percentile(extinct_g["lifetime_s"], 99) if len(extinct_g) else np.nan
        ),
    }

    avalanche_summary_row.update(conn)

    # --------------------------------------------------------
    # Spontaneous run + ICG/static-dynamic scaling
    # --------------------------------------------------------
    with threadpool_limits(limits=1):
        spikes, pop_rate = fn.run_model(model, T=T_RUN)

    dt = float(model.dt)
    burn = int(round(BURN_IN_S / dt))

    spikes_use = spikes[:, burn:]
    pop_rate_use = np.asarray(pop_rate[burn:], dtype=float)

    active_counts = spikes_use.sum(axis=0).astype(float)
    rho = active_counts / float(n)

    mean_rate_hz_from_spikes = float(spikes_use.mean() / dt)
    mean_rate_hz_from_pop_rate = float(np.mean(pop_rate_use))

    mean_rho = float(np.mean(rho))
    var_rho = float(np.var(rho))
    std_rho = float(np.std(rho))
    susceptibility = float(n * var_rho)

    pop_rate_var = float(np.var(pop_rate_use))
    pop_rate_std = float(np.std(pop_rate_use))
    pop_rate_cv = safe_cv(pop_rate_use)

    active_count_cv = safe_cv(active_counts)

    autocorr_tau_rho_s = pop_autocorr_tau(rho, dt=dt, max_lag_s=3.0)
    autocorr_tau_pop_rate_s = pop_autocorr_tau(pop_rate_use, dt=dt, max_lag_s=3.0)

    spikes_smooth = fn.exp_smooth_spikes(
        spikes,
        dt=dt,
        tau=SMOOTHE,
    )

    # --------------------------------------------------------
    # Warning-safe ICG metrics
    # suppresses np.corrcoef zero-variance divide warnings
    # --------------------------------------------------------
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
        )

        with np.errstate(invalid="ignore", divide="ignore"):
            metric_row, gen_df = fn.compute_icg_metrics(
                spikes=spikes_smooth,
                dt=dt,
            )

    gen_df = standardise_gen_df(gen_df)

    MV_alpha, MV_r2, MV_n = fit_power_slope(
        gen_df,
        y_col="mean_variance_norm",
        x_col="mean_cluster_size",
        exclude_first=True,
        exclude_last=False,
    )

    TAU_beta, TAU_r2, TAU_n = fit_power_slope(
        gen_df,
        y_col="timescale_norm",
        x_col="mean_cluster_size",
        exclude_first=True,
        exclude_last=False,
    )

    KURT_slope, KURT_r2, KURT_n = fit_power_slope(
        gen_df,
        y_col="kurtosis_corr",
        x_col="mean_cluster_size",
        exclude_first=True,
        exclude_last=False,
    )

    score = (
        abs(MV_alpha - TARGET_MV) + 2.0 * abs(TAU_beta - TARGET_TAU)
        if np.isfinite(MV_alpha) and np.isfinite(TAU_beta)
        else np.nan
    )

    gen_df["combo_i"] = combo_i
    gen_df["ew_idx"] = ew_idx
    gen_df["iw_idx"] = iw_idx
    gen_df["phi_idx"] = phi_idx
    gen_df["seed_idx"] = seed_idx
    gen_df["network_seed"] = network_seed
    gen_df["e_w"] = float(model.e_w)
    gen_df["i_w"] = float(model.i_w)
    gen_df["phi"] = phi
    gen_df["p_ext"] = float(model.p_ext)
    gen_df["smoothe"] = float(SMOOTHE)
    gen_df["mean_rate_hz"] = mean_rate_hz_from_spikes

    icg_path = ICGDIR / (
        f"combo_{combo_i:04d}_ewidx_{ew_idx:02d}_iwidx_{iw_idx:02d}_"
        f"phiidx_{phi_idx:02d}_seed_{seed_idx:02d}_icg.csv.gz"
    )

    if SAVE_ICG_ROWS:
        gen_df.to_csv(icg_path, index=False, compression="gzip")

    icg_summary_row = {
        "combo_i": combo_i,
        "ew_idx": ew_idx,
        "iw_idx": iw_idx,
        "phi_idx": phi_idx,
        "seed_idx": seed_idx,
        "network_seed": network_seed,

        "e_w": float(model.e_w),
        "i_w": float(model.i_w),
        "phi": phi,
        "theta": float(model.theta),
        "p_ext": float(model.p_ext),

        "icg_file": str(icg_path),

        "T": float(T_RUN),
        "burn_in_s": float(BURN_IN_S),
        "smoothe": float(SMOOTHE),

        "mean_rate_hz": mean_rate_hz_from_spikes,
        "mean_rate_hz_pop_rate": mean_rate_hz_from_pop_rate,
        "median_pop_rate_hz": float(np.median(pop_rate_use)),
        "max_pop_rate_hz": float(np.max(pop_rate_use)),

        "mean_rho": mean_rho,
        "std_rho": std_rho,
        "var_rho": var_rho,
        "susceptibility_N_var_rho": susceptibility,

        "pop_rate_var": pop_rate_var,
        "pop_rate_std": pop_rate_std,
        "pop_rate_cv": pop_rate_cv,
        "active_count_cv": active_count_cv,

        "autocorr_tau_rho_s": autocorr_tau_rho_s,
        "autocorr_tau_pop_rate_s": autocorr_tau_pop_rate_s,

        "silent_frac": float(np.mean(active_counts == 0)),
        "active_frac": float(np.mean(active_counts > 0)),
        "mean_active_count": float(np.mean(active_counts)),
        "p95_active_count": float(np.percentile(active_counts, 95)),
        "p99_active_count": float(np.percentile(active_counts, 99)),
        "max_active_count": float(np.max(active_counts)),

        "MV_alpha": MV_alpha,
        "MV_r2": MV_r2,
        "MV_n_points": MV_n,

        "TAU_beta": TAU_beta,
        "TAU_r2": TAU_r2,
        "TAU_n_points": TAU_n,

        "KURT_slope": KURT_slope,
        "KURT_r2": KURT_r2,
        "KURT_n_points": KURT_n,

        "score": score,
    }

    icg_summary_row.update(conn)

    return avalanche_summary_row, icg_summary_row


# ============================================================
# Build design
# ============================================================

jobs = []
combo_i = 0

for ew_idx, e_w in enumerate(EW_VALUES):
    for iw_idx, i_w in enumerate(IW_VALUES):
        for phi_idx, phi in enumerate(PHI_VALUES):
            for seed_idx in range(N_SEEDS):
                jobs.append({
                    "combo_i": int(combo_i),
                    "ew_idx": int(ew_idx),
                    "iw_idx": int(iw_idx),
                    "phi_idx": int(phi_idx),
                    "seed_idx": int(seed_idx),
                    "e_w": float(e_w),
                    "i_w": float(i_w),
                    "phi": float(phi),
                })

        combo_i += 1

design = pd.DataFrame(jobs)
design.to_csv(OUTDIR / "grid_design.csv", index=False)

print("Saved design:", OUTDIR / "grid_design.csv")
print("N jobs:", len(jobs))


# ============================================================
# Run grid
# ============================================================

ctx = mp.get_context("fork")

avalanche_summary_rows = []
icg_summary_rows = []

with ctx.Pool(processes=N_WORKERS) as pool:
    for avalanche_row, icg_row in tqdm(
        pool.imap_unordered(run_one_network, jobs, chunksize=1),
        total=len(jobs),
        desc="e_w x i_w x phi x seed grid",
    ):
        avalanche_summary_rows.append(avalanche_row)
        icg_summary_rows.append(icg_row)

        n_done = len(avalanche_summary_rows)

        if n_done % 100 == 0:
            pd.DataFrame(avalanche_summary_rows).to_csv(
                OUTDIR / "avalanche_network_summary_partial.csv",
                index=False,
            )

            pd.DataFrame(icg_summary_rows).to_csv(
                OUTDIR / "icg_seed_summary_partial.csv",
                index=False,
            )


avalanche_network_summary = (
    pd.DataFrame(avalanche_summary_rows)
    .sort_values(["combo_i", "phi", "seed_idx"])
    .reset_index(drop=True)
)

icg_seed_summary = (
    pd.DataFrame(icg_summary_rows)
    .sort_values(["combo_i", "phi", "seed_idx"])
    .reset_index(drop=True)
)

avalanche_network_summary.to_csv(
    OUTDIR / "avalanche_network_summary.csv",
    index=False,
)

icg_seed_summary.to_csv(
    OUTDIR / "icg_seed_summary.csv",
    index=False,
)

print("Saved:", OUTDIR / "avalanche_network_summary.csv")
print("Saved:", OUTDIR / "icg_seed_summary.csv")


# ============================================================
# Aggregate by combo x phi
# ============================================================

av_cols = [
    "frac_extinct",
    "frac_persistent_at_cutoff",
    "mean_size",
    "p95_size",
    "p99_size",
    "mean_lifetime_s",
    "p95_lifetime_s",
    "p99_lifetime_s",
    "mean_peak_active",
    "p95_peak_active",
    "mean_avalanche_tau_s",
    "median_avalanche_tau_s",
    "p95_avalanche_tau_s",
    "p99_avalanche_tau_s",
    "mean_response_decay_tau_s",
    "p95_response_decay_tau_s",
]

agg = {
    "n": ("seed_idx", "count"),
    "e_w": ("e_w", "first"),
    "i_w": ("i_w", "first"),
}

for col in av_cols:
    agg[f"{col}_mean"] = (col, "mean")
    agg[f"{col}_sem"] = (col, sem)

avalanche_combo_phi_summary = (
    avalanche_network_summary
    .groupby(["combo_i", "ew_idx", "iw_idx", "phi_idx", "phi"], as_index=False)
    .agg(**agg)
    .sort_values(["combo_i", "phi"])
    .reset_index(drop=True)
)

avalanche_combo_phi_summary.to_csv(
    OUTDIR / "avalanche_combo_phi_summary.csv",
    index=False,
)

print("Saved:", OUTDIR / "avalanche_combo_phi_summary.csv")


icg_cols = [
    "mean_rate_hz",
    "mean_rho",
    "var_rho",
    "susceptibility_N_var_rho",
    "pop_rate_cv",
    "autocorr_tau_rho_s",
    "autocorr_tau_pop_rate_s",
    "MV_alpha",
    "MV_r2",
    "TAU_beta",
    "TAU_r2",
    "KURT_slope",
    "KURT_r2",
    "score",
]

agg = {
    "n": ("seed_idx", "count"),
    "e_w": ("e_w", "first"),
    "i_w": ("i_w", "first"),
}

for col in icg_cols:
    agg[f"{col}_mean"] = (col, "mean")
    agg[f"{col}_sem"] = (col, sem)

icg_combo_phi_summary = (
    icg_seed_summary
    .groupby(["combo_i", "ew_idx", "iw_idx", "phi_idx", "phi"], as_index=False)
    .agg(**agg)
    .sort_values(["combo_i", "phi"])
    .reset_index(drop=True)
)

icg_combo_phi_summary.to_csv(
    OUTDIR / "icg_combo_phi_summary.csv",
    index=False,
)

print("Saved:", OUTDIR / "icg_combo_phi_summary.csv")


# ============================================================
# Aggregate by combo across phi
# ============================================================

def peak_width(phi, y, frac=0.5):
    phi = np.asarray(phi, dtype=float)
    y = np.asarray(y, dtype=float)

    keep = np.isfinite(phi) & np.isfinite(y)
    phi = phi[keep]
    y = y[keep]

    if y.size < 3:
        return np.nan, np.nan, np.nan, 0

    peak_idx = int(np.nanargmax(y))
    peak_phi = float(phi[peak_idx])
    peak_y = float(y[peak_idx])

    if not np.isfinite(peak_y) or peak_y <= 0:
        return peak_phi, peak_y, np.nan, 0

    threshold = frac * peak_y
    above = y >= threshold

    width = float(np.max(phi[above]) - np.min(phi[above]))
    n_above = int(np.sum(above))

    return peak_phi, peak_y, width, n_above


combo_rows = []

for combo_i, g in avalanche_combo_phi_summary.groupby("combo_i", sort=True):
    g = g.sort_values("phi")

    peak_phi, peak_tau, width, n_above = peak_width(
        g["phi"],
        g["mean_avalanche_tau_s_mean"],
        frac=0.5,
    )

    icg_g = icg_combo_phi_summary[icg_combo_phi_summary["combo_i"] == combo_i]

    combo_rows.append({
        "combo_i": int(combo_i),
        "ew_idx": int(g["ew_idx"].iloc[0]),
        "iw_idx": int(g["iw_idx"].iloc[0]),
        "e_w": float(g["e_w"].iloc[0]),
        "i_w": float(g["i_w"].iloc[0]),

        "autocorr_peak_phi": peak_phi,
        "autocorr_peak_tau_s": peak_tau,
        "autocorr_peak_width_phi": width,
        "autocorr_n_phi_above_halfmax": n_above,

        "mean_persistent_fraction": float(g["frac_persistent_at_cutoff_mean"].mean()),
        "max_persistent_fraction": float(g["frac_persistent_at_cutoff_mean"].max()),

        "mean_avalanche_tau_s_across_phi": float(g["mean_avalanche_tau_s_mean"].mean()),
        "max_avalanche_tau_s_across_phi": float(g["mean_avalanche_tau_s_mean"].max()),

        "best_scaling_score": float(icg_g["score_mean"].min()),
        "best_scaling_phi": float(
            icg_g.loc[icg_g["score_mean"].idxmin(), "phi"]
            if len(icg_g) and icg_g["score_mean"].notna().any()
            else np.nan
        ),
        "max_MV_alpha": float(icg_g["MV_alpha_mean"].max()),
        "max_TAU_beta": float(icg_g["TAU_beta_mean"].max()),
        "max_KURT_slope": float(icg_g["KURT_slope_mean"].max()),
        "max_autocorr_tau_rho_s": float(icg_g["autocorr_tau_rho_s_mean"].max()),
    })

combo_summary = (
    pd.DataFrame(combo_rows)
    .sort_values(
        ["autocorr_peak_width_phi", "autocorr_peak_tau_s"],
        ascending=[True, False],
    )
    .reset_index(drop=True)
)

combo_summary.to_csv(
    OUTDIR / "combo_summary_ranked.csv",
    index=False,
)

print("Saved:", OUTDIR / "combo_summary_ranked.csv")
display(combo_summary.head(20))

print("\nDone.")
print("OUTDIR:", OUTDIR)