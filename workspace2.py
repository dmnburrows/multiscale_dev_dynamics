# ============================================================
# Exact LLR phi grid
# Seeded avalanches + per-avalanche autocorrelation timescale
# + spontaneous ICG on the same network
# ============================================================

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
# Settings
# ============================================================

start_dic = {
    "n_neurons": 2000,
    "T": 10,
    "dt": 0.01,
    "refractory_steps": 2,
    "ei_ratio": 0.2,
    "e_w": 11.7,
    "i_w": 22.4,
    "theta": 8.44,
    "p_ext": 0.0,
    "phi": 4.2,
    "smoothe": 0.05,
}



PHI_VALUES = np.array([
    2.605263,
    2.789474,
    2.973684,
    3.157895,
    3.342105,
    3.526316,
    3.710526,
    3.894737,
    4.078947,
    4.263158,
    4.447368,
])


N_WORKERS = 20
N_SEEDS = 50
N_AVALANCHES_PER_NETWORK = 10000
MAX_STEPS = 1000
BASE_SEED = 93939

DT = float(start_dic["dt"])
T_RUN = float(start_dic["T"])
SMOOTHE = float(start_dic["smoothe"])
BURN_IN_S = 2.0

TARGET_MV = 1.50
TARGET_TAU = 0.20

OUTDIR = Path(
    "/home/dburrows/DATA/BLNDEV-WILDTYPE/"
    "phi_ew11p7_iw22p4_N2000_"
    "llrgrid_11phis_50seeds_10000avals_"
    "max1000_avalanche_tau_icg"
)

TRIALDIR = OUTDIR / "avalanche_trials_by_network"
ICGDIR = OUTDIR / "icg_rows_by_network"

TRIALDIR.mkdir(parents=True, exist_ok=True)
ICGDIR.mkdir(parents=True, exist_ok=True)

config = {
    "N_WORKERS": N_WORKERS,
    "N_SEEDS": N_SEEDS,
    "N_AVALANCHES_PER_NETWORK": N_AVALANCHES_PER_NETWORK,
    "MAX_STEPS": MAX_STEPS,
    "BASE_SEED": BASE_SEED,
    "PHI_VALUES": PHI_VALUES.tolist(),
    "phi_grid_source": "existing llr DataFrame, unrounded",
    "start_dic": start_dic,
    "seed_only_excitatory": True,
    "clamp_zero_presynaptic_input": True,
    "avalanche_timescale_function": "icg_functions.timescale",
    "avalanche_max_lag": "min(3.0, max(2*dt, duration_s/2))",
    "avalanche_min_frames": 5,
    "avalanche_summary_includes": "extinct and censored trials",
    "burn_in_s": BURN_IN_S,
    "icg_uses_full_recording": True,
    "TARGET_MV": TARGET_MV,
    "TARGET_TAU": TARGET_TAU,
}

with open(OUTDIR / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print("OUTDIR:", OUTDIR)
print("Exact phi values:")
for phi in PHI_VALUES:
    print(f"  {phi:.16g}")

print("Workers:", N_WORKERS)
print("Total networks:", len(PHI_VALUES) * N_SEEDS)
print(
    "Total avalanches:",
    len(PHI_VALUES) * N_SEEDS * N_AVALANCHES_PER_NETWORK,
)


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


def safe_mean(x):
    x = safe_values(x)
    return float(np.mean(x)) if x.size else np.nan


def safe_percentile(x, q):
    x = safe_values(x)
    return float(np.percentile(x, q)) if x.size else np.nan


def sem(x):
    x = safe_values(x)
    return (
        float(np.std(x, ddof=1) / np.sqrt(x.size))
        if x.size > 1 else np.nan
    )


def response_decay_timescale(active_counts, dt):
    x = np.asarray(active_counts, dtype=float)

    if x.size == 0:
        return np.nan

    peak = float(np.max(x))
    if peak <= 0:
        return 0.0

    peak_idx = int(np.argmax(x))
    post_peak = x[peak_idx:]
    below = np.where(post_peak <= peak / np.e)[0]

    if below.size:
        return float(below[0] * dt)

    # Same fallback as the original example.
    return float((post_peak.size - 1) * dt)


def branching_metrics(active_counts, extinct):
    counts = np.asarray(active_counts, dtype=float)

    if extinct:
        x = counts
        y = np.concatenate([counts[1:], [0.0]])
    else:
        x = counts[:-1]
        y = counts[1:]

    keep = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[keep], y[keep]

    result = {
        "sigma_origin": np.nan,
        "br_reg_slope": np.nan,
        "br_reg_intercept": np.nan,
        "br_reg_r2": np.nan,
        "br_n_pairs": int(x.size),
    }

    if x.size < 2:
        return result

    denom = float(np.sum(x * x))
    result["sigma_origin"] = (
        float(np.sum(x * y) / denom)
        if denom > 0 else np.nan
    )

    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    result.update({
        "br_reg_intercept": float(beta[0]),
        "br_reg_slope": float(beta[1]),
        "br_reg_r2": (
            float(1.0 - ss_res / ss_tot)
            if ss_tot > 0 else np.nan
        ),
    })

    return result


def local_connectivity_metrics(A, n_e):
    return {
        "edge_count": int(A.sum()),
        "mean_out_degree": float(A.sum(axis=1).mean()),
        "std_out_degree": float(A.sum(axis=1).std()),
        "mean_e_out_degree_all": float(
            A[:n_e, :].sum(axis=1).mean()
        ),
        "mean_e_out_degree_e": float(
            A[:n_e, :n_e].sum(axis=1).mean()
        ),
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

    ac = np.empty(max_lag + 1)
    ac[0] = 1.0

    for lag in range(1, max_lag + 1):
        ac[lag] = np.sum(x[:-lag] * x[lag:]) / denom

    ac[~np.isfinite(ac)] = 0.0
    crossing = np.where(ac[1:] <= 0)[0]

    if crossing.size:
        ac = ac[:crossing[0] + 2]

    return float(np.trapezoid(ac, dx=dt))


def fit_power_slope(g, y_col):
    d = (
        g[["mean_cluster_size", y_col]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    d = d[
        (d["mean_cluster_size"] > 0) & (d[y_col] > 0)
    ].sort_values("mean_cluster_size")

    # Exclude first point; retain last.
    d = d.iloc[1:]

    if len(d) < 3:
        return np.nan, np.nan, int(len(d))

    logx = np.log10(d["mean_cluster_size"].to_numpy(float))
    logy = np.log10(d[y_col].to_numpy(float))

    slope, intercept = np.polyfit(logx, logy, 1)
    pred = intercept + slope * logx

    ss_res = np.sum((logy - pred) ** 2)
    ss_tot = np.sum((logy - logy.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return float(slope), float(r2), int(len(d))


def standardise_gen_df(gen_df):
    gen_df = gen_df.copy()

    aliases = {
        "MV": "mean_variance",
        "MV_norm": "mean_variance_norm",
        "TAU": "timescale",
        "TAU_norm": "timescale_norm",
        "corr_kurtosis": "kurtosis_corr",
    }

    for old, new in aliases.items():
        if new not in gen_df.columns and old in gen_df.columns:
            gen_df[new] = gen_df[old]

    return gen_df


def make_model_kwargs(phi, seed):
    pars = dict(start_dic)
    pars.pop("T", None)
    pars.pop("smoothe", None)

    pars["phi"] = float(phi)
    pars["seed"] = int(seed)
    pars["p_ext"] = 0.0

    valid_args = set(
        inspect.signature(
            fn.automata_EI_hiermod.__init__
        ).parameters
    )

    if "phi" not in valid_args:
        if "slope" in valid_args:
            pars["slope"] = pars.pop("phi")
        else:
            raise ValueError("Model exposes neither phi nor slope.")

    return {k: v for k, v in pars.items() if k in valid_args}


# ============================================================
# Seeded avalanche runner
# ============================================================

def run_seeded_avalanche_sparse(
    A_e_T, A_i_T, n, n_e,
    e_w, i_w, theta, refractory_steps,
    dt, seed_node, rng, max_steps,
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

        inp_e = A_e_T @ active[:n_e].astype(np.float32)
        inp_i = A_i_T @ active[n_e:].astype(np.float32)

        net = e_w * inp_e - i_w * inp_i
        p_net = expit(net - theta)

        # Remove baseline firing without active presynaptic input.
        has_input = (inp_e > 0) | (inp_i > 0)
        p_net[~has_input] = 0.0

        new_active = (
            (state == 0)
            & (rng.random(n) < p_net)
        )

        new_state = np.zeros_like(state)
        new_state[active] = 2

        refractory = state >= 2
        new_state[refractory] = state[refractory] + 1
        new_state[new_state > refractory_steps + 1] = 0
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
    extinct = not censored
    duration_s = float(duration_steps * dt)

    # Same per-avalanche fn.timescale() calculation as before.
    if active_counts.size >= 5 and np.var(active_counts) > 0:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in divide",
                category=RuntimeWarning,
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                avalanche_tau_s = fn.timescale(
                    active_counts[np.newaxis, :].astype(float),
                    max_lag=min(
                        3.0,
                        max(2 * dt, duration_s / 2),
                    ),
                    dt=float(dt),
                )
    else:
        avalanche_tau_s = np.nan

    return {
        "size": int(active_counts.sum()),
        "duration_steps": int(duration_steps),
        "duration_s": duration_s,
        "lifetime_steps": int(duration_steps),
        "lifetime_s": duration_s,
        "peak_active": (
            int(active_counts.max()) if active_counts.size else 0
        ),
        "extinct": bool(extinct),
        "censored": bool(censored),
        "persistent_at_cutoff": bool(censored),
        "mean_active_during_avalanche": (
            float(active_counts.mean())
            if active_counts.size else np.nan
        ),
        "avalanche_tau_s": float(avalanche_tau_s),
        "response_decay_tau_s": response_decay_timescale(
            active_counts, dt
        ),
        **branching_metrics(active_counts, extinct),
    }


# ============================================================
# One network: avalanche trials + spontaneous ICG
# ============================================================

def run_one_network(job):
    network_id = int(job["network_id"])
    phi_idx = int(job["phi_idx"])
    seed_idx = int(job["seed_idx"])
    phi = float(job["phi"])

    # Original phi-only avalanche seed construction.
    network_seed = int(
        BASE_SEED
        + int(round(phi * 1000)) * 100000
        + seed_idx * 1000
    )

    model_kwargs = make_model_kwargs(phi, network_seed)

    with threadpool_limits(limits=1):
        model = fn.automata_EI_hiermod(**model_kwargs)

    A = np.array(model.A, dtype=np.uint8, copy=True)
    np.fill_diagonal(A, 0)

    n = int(model.n)
    n_e = int(model.e)
    dt = float(model.dt)

    conn = local_connectivity_metrics(A, n_e)

    A_e_T = sparse.csr_matrix(
        A[:n_e, :].T.astype(np.float32)
    )
    A_i_T = sparse.csr_matrix(
        A[n_e:, :].T.astype(np.float32)
    )

    rng = np.random.default_rng(network_seed + 123)

    metadata = {
        "network_id": network_id,
        "phi_idx": phi_idx,
        "seed": seed_idx,
        "seed_idx": seed_idx,
        "network_seed": network_seed,
        "phi": phi,
        "e_w": float(model.e_w),
        "i_w": float(model.i_w),
        "theta": float(model.theta),
        "p_ext": 0.0,
        "n_neurons": n,
        "n_e": n_e,
        "dt": dt,
    }

    # --------------------------------------------------------
    # Avalanche trials
    # --------------------------------------------------------

    rows = []

    for aval_i in range(N_AVALANCHES_PER_NETWORK):
        seed_node = int(rng.integers(0, n_e))

        result = run_seeded_avalanche_sparse(
            A_e_T=A_e_T,
            A_i_T=A_i_T,
            n=n,
            n_e=n_e,
            e_w=float(model.e_w),
            i_w=float(model.i_w),
            theta=float(model.theta),
            refractory_steps=int(model.refractory_steps),
            dt=dt,
            seed_node=seed_node,
            rng=rng,
            max_steps=MAX_STEPS,
        )

        rows.append({
            **metadata,
            "aval_i": aval_i,
            "seed_node": seed_node,
            "seed_cell_type": "E",
            "max_steps": MAX_STEPS,
            "cutoff_s": MAX_STEPS * dt,
            "clamp_zero_presynaptic_input": True,
            **result,
        })

    avalanche_df = pd.DataFrame(rows)

    aval_path = TRIALDIR / (
        f"phiidx_{phi_idx:02d}_seed_{seed_idx:02d}_avalanches.csv.gz"
    )
    avalanche_df.to_csv(
        aval_path, index=False, compression="gzip"
    )

    avalanche_summary = {
        **metadata,
        **conn,
        "n_avalanches": len(avalanche_df),
        "avalanche_file": str(aval_path),
        "frac_extinct": float(avalanche_df["extinct"].mean()),
        "frac_persistent_at_cutoff": float(
            avalanche_df["persistent_at_cutoff"].mean()
        ),
        "n_valid_avalanche_tau": int(
            np.isfinite(avalanche_df["avalanche_tau_s"]).sum()
        ),
    }

    for column, name in [
        ("size", "size"),
        ("lifetime_s", "lifetime_s"),
        ("peak_active", "peak_active"),
        ("avalanche_tau_s", "avalanche_tau_s"),
        ("response_decay_tau_s", "response_decay_tau_s"),
    ]:
        avalanche_summary[f"mean_{name}"] = safe_mean(
            avalanche_df[column]
        )
        avalanche_summary[f"median_{name}"] = safe_percentile(
            avalanche_df[column], 50
        )

        for q in [90, 95, 99]:
            avalanche_summary[f"p{q}_{name}"] = safe_percentile(
                avalanche_df[column], q
            )

    extinct_df = avalanche_df.loc[avalanche_df["extinct"]]

    avalanche_summary["mean_avalanche_tau_s_extinct_only"] = (
        safe_mean(extinct_df["avalanche_tau_s"])
    )
    avalanche_summary["n_valid_avalanche_tau_extinct_only"] = int(
        np.isfinite(extinct_df["avalanche_tau_s"]).sum()
    )

    # --------------------------------------------------------
    # Spontaneous activity using the module's own step()
    # --------------------------------------------------------

    with threadpool_limits(limits=1):
        spikes, pop_rate = fn.run_model(model, T=T_RUN)

    burn = int(round(BURN_IN_S / dt))
    spikes_use = spikes[:, burn:]
    pop_rate_use = np.asarray(pop_rate[burn:], dtype=float)

    active_counts = spikes_use.sum(axis=0).astype(float)
    rho = active_counts / n

    # ICG uses the full recording, as in the previous block.
    spikes_smooth = fn.exp_smooth_spikes(
        spikes, dt=dt, tau=SMOOTHE
    )

    with threadpool_limits(limits=1):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in divide",
                category=RuntimeWarning,
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                _, gen_df = fn.compute_icg_metrics(
                    spikes=spikes_smooth,
                    dt=dt,
                )

    gen_df = standardise_gen_df(gen_df)

    MV_alpha, MV_r2, MV_n = fit_power_slope(
        gen_df, "mean_variance_norm"
    )
    TAU_beta, TAU_r2, TAU_n = fit_power_slope(
        gen_df, "timescale_norm"
    )
    KURT_slope, KURT_r2, KURT_n = fit_power_slope(
        gen_df, "kurtosis_corr"
    )

    score = (
        abs(MV_alpha - TARGET_MV)
        + 2.0 * abs(TAU_beta - TARGET_TAU)
        if np.isfinite(MV_alpha) and np.isfinite(TAU_beta)
        else np.nan
    )

    for key, value in metadata.items():
        gen_df[key] = value
    gen_df["smoothe"] = SMOOTHE

    icg_path = ICGDIR / (
        f"phiidx_{phi_idx:02d}_seed_{seed_idx:02d}_icg.csv.gz"
    )
    gen_df.to_csv(icg_path, index=False, compression="gzip")

    icg_summary = {
        **metadata,
        **conn,
        "icg_file": str(icg_path),
        "T": T_RUN,
        "burn_in_s": BURN_IN_S,
        "smoothe": SMOOTHE,
        "mean_rate_hz": float(spikes_use.mean() / dt),
        "mean_rate_hz_pop_rate": float(pop_rate_use.mean()),
        "mean_rho": float(rho.mean()),
        "var_rho": float(rho.var()),
        "susceptibility_N_var_rho": float(n * rho.var()),
        "silent_frac": float(np.mean(active_counts == 0)),
        "autocorr_tau_rho_s": pop_autocorr_tau(rho, dt),
        "autocorr_tau_pop_rate_s": pop_autocorr_tau(
            pop_rate_use, dt
        ),
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

    # Per-network checkpoints.
    pd.DataFrame([avalanche_summary]).to_csv(
        TRIALDIR / (
            f"phiidx_{phi_idx:02d}_seed_{seed_idx:02d}_summary.csv"
        ),
        index=False,
    )
    pd.DataFrame([icg_summary]).to_csv(
        ICGDIR / (
            f"phiidx_{phi_idx:02d}_seed_{seed_idx:02d}_summary.csv"
        ),
        index=False,
    )

    return avalanche_summary, icg_summary


# ============================================================
# Build design
# ============================================================

jobs = [
    {
        "network_id": phi_idx * N_SEEDS + seed_idx,
        "phi_idx": phi_idx,
        "seed_idx": seed_idx,
        "phi": float(phi),
    }
    for phi_idx, phi in enumerate(PHI_VALUES)
    for seed_idx in range(N_SEEDS)
]

pd.DataFrame(jobs).to_csv(
    OUTDIR / "design.csv", index=False
)


# ============================================================
# Run with 20 workers — Linux/fork
# ============================================================

ctx = mp.get_context("fork")

avalanche_rows = []
icg_rows = []

with ctx.Pool(processes=N_WORKERS) as pool:
    for av_row, icg_row in tqdm(
        pool.imap_unordered(run_one_network, jobs, chunksize=1),
        total=len(jobs),
        desc="Exact LLR phi grid: avalanche tau + ICG",
        unit="network",
    ):
        avalanche_rows.append(av_row)
        icg_rows.append(icg_row)

        if len(avalanche_rows) % 20 == 0:
            pd.DataFrame(avalanche_rows).to_csv(
                OUTDIR / "avalanche_network_summary_partial.csv",
                index=False,
            )
            pd.DataFrame(icg_rows).to_csv(
                OUTDIR / "icg_seed_summary_partial.csv",
                index=False,
            )

avalanche_network_summary = (
    pd.DataFrame(avalanche_rows)
    .sort_values(["phi", "seed_idx"])
    .reset_index(drop=True)
)

icg_seed_summary = (
    pd.DataFrame(icg_rows)
    .sort_values(["phi", "seed_idx"])
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


# ============================================================
# Aggregate by phi
# SEM is across network seeds, not individual trials.
# ============================================================

def count_valid(x):
    return int(len(safe_values(x)))


def summarise_by_phi(frame, columns):
    agg = {
        "n": ("seed_idx", "count"),
        "e_w": ("e_w", "first"),
        "i_w": ("i_w", "first"),
    }

    for column in columns:
        agg[f"{column}_mean"] = (column, safe_mean)
        agg[f"{column}_sem"] = (column, sem)
        agg[f"{column}_n_valid"] = (column, count_valid)

    return (
        frame.groupby(["phi_idx", "phi"], as_index=False)
        .agg(**agg)
        .sort_values("phi")
        .reset_index(drop=True)
    )


av_columns = [
    "frac_extinct",
    "frac_persistent_at_cutoff",
    "n_valid_avalanche_tau",
    "mean_size",
    "p95_size",
    "p99_size",
    "mean_lifetime_s",
    "p95_lifetime_s",
    "p99_lifetime_s",
    "mean_peak_active",
    "mean_avalanche_tau_s",
    "median_avalanche_tau_s",
    "p95_avalanche_tau_s",
    "p99_avalanche_tau_s",
    "mean_avalanche_tau_s_extinct_only",
    "mean_response_decay_tau_s",
    "p95_response_decay_tau_s",
]

avalanche_phi_summary = summarise_by_phi(
    avalanche_network_summary, av_columns
)

avalanche_phi_summary.to_csv(
    OUTDIR / "avalanche_combo_phi_summary.csv",
    index=False,
)

icg_columns = [
    "mean_rate_hz",
    "mean_rho",
    "var_rho",
    "susceptibility_N_var_rho",
    "autocorr_tau_rho_s",
    "autocorr_tau_pop_rate_s",
    "MV_alpha",
    "MV_r2",
    "TAU_beta",
    "TAU_r2",
    "KURT_slope",
    "score",
]

icg_phi_summary = summarise_by_phi(
    icg_seed_summary, icg_columns
)

icg_phi_summary.to_csv(
    OUTDIR / "icg_combo_phi_summary.csv",
    index=False,
)

print("\nAvalanche autocorrelation summary:")
print(
    avalanche_phi_summary[[
        "phi",
        "mean_avalanche_tau_s_mean",
        "mean_avalanche_tau_s_sem",
        "mean_avalanche_tau_s_n_valid",
        "frac_persistent_at_cutoff_mean",
    ]]
    .round(6)
    .to_string(index=False)
)

print("\nDone.")
print("Outputs:", OUTDIR)