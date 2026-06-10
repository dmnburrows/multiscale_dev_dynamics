import json, multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

import icg_functions as fn


# ============================================================
# Settings: single phi, multiprocessing over network seeds
# ============================================================

N_WORKERS = 30
N_SEEDS_PER_PHI = 50
N_AVALANCHES_PER_NETWORK = 10000
MAX_STEPS = 1000
BASE_SEED = 93939

PHI_VALUES = np.linspace(2.5, 3.5, 5)


OUTDIR = Path(
    "/home/dburrows/DATA/BLNDEV-WILDTYPE/"
    "new_ew11p7_iw22p4_phi2p5to3p5_N2000_5phis_50seeds_10000avals_max1000"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


start_dic = {
    "n_neurons": 2000,
    "T": 10,
    "dt": 0.01,
    "refractory_steps": 2,
    "ei_ratio": 0.2,
    "e_w": 11.7,
    "i_w": 22.4,
    "theta": 8.44,
    "p_ext": 0.02,
    "phi": 4.2,
    "smoothe": 0.05,

}

AVALANCHE_P_EXT = 0.0
DT = float(start_dic["dt"])
MAX_TIME_S = MAX_STEPS * DT

design_df = pd.DataFrame([
    {
        "network_id": int(i),
        "phi": float(phi),
        "seed": int(seed),
    }
    for i, (phi, seed) in enumerate(
        (p, s) for p in PHI_VALUES for s in range(N_SEEDS_PER_PHI)
    )
])

design_df.to_csv(OUTDIR / "design.csv", index=False)

with open(OUTDIR / "config.json", "w") as f:
    json.dump(
        {
            "N_WORKERS": N_WORKERS,
            "N_SEEDS_PER_PHI": N_SEEDS_PER_PHI,
            "N_AVALANCHES_PER_NETWORK": N_AVALANCHES_PER_NETWORK,
            "MAX_STEPS": MAX_STEPS,
            "MAX_TIME_S": MAX_TIME_S,
            "PHI_VALUES": list(PHI_VALUES),
            "AVALANCHE_P_EXT": AVALANCHE_P_EXT,
            "start_dic": start_dic,
        },
        f,
        indent=2,
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


def safe_cv(x):
    x = safe_values(x)
    if x.size < 2:
        return np.nan
    mu = float(np.mean(x))
    return float(np.std(x, ddof=1) / mu) if mu > 0 and np.isfinite(mu) else np.nan


def local_connectivity_metrics(A, n_e):
    A_e_all = A[:n_e, :]
    A_ee = A[:n_e, :n_e]
    return {
        "edge_count": int(A.sum()),
        "mean_out_degree": float(A.sum(axis=1).mean()),
        "mean_e_out_degree_all": float(A_e_all.sum(axis=1).mean()),
        "mean_e_out_degree_e": float(A_ee.sum(axis=1).mean()),
        "edge_density": float(A.sum() / max(A.shape[0] * (A.shape[0] - 1), 1)),
    }


# ============================================================
# Seeded avalanche runner
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

    size = 0
    peak_active = 1

    for step in range(int(max_steps)):
        active = state == 1
        n_active = int(active.sum())

        if n_active == 0:
            return {
                "size": int(size),
                "lifetime_steps": int(step),
                "lifetime_s": float(step * dt),
                "peak_active": int(peak_active),
                "extinct": True,
                "persistent_at_cutoff": False,
            }

        size += n_active
        peak_active = max(peak_active, n_active)

        active_e = active[:n_e].astype(np.float32)
        active_i = active[n_e:].astype(np.float32)

        inp_e = A_e_T @ active_e
        inp_i = A_i_T @ active_i

        net = (e_w * inp_e) - (i_w * inp_i)
        p_net = expit(net - theta)

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
            return {
                "size": int(size),
                "lifetime_steps": int(step + 1),
                "lifetime_s": float((step + 1) * dt),
                "peak_active": int(peak_active),
                "extinct": True,
                "persistent_at_cutoff": False,
            }

    return {
        "size": int(size),
        "lifetime_steps": int(max_steps),
        "lifetime_s": float(max_steps * dt),
        "peak_active": int(peak_active),
        "extinct": False,
        "persistent_at_cutoff": True,
    }


# ============================================================
# One network seed job
# ============================================================

def run_one_network_job(job):
    network_id = int(job["network_id"])
    phi = float(job["phi"])
    seed = int(job["seed"])

    pars = dict(start_dic)
    pars["phi"] = phi
    pars["p_ext"] = AVALANCHE_P_EXT

    model_kwargs = {
        k: v for k, v in pars.items()
        if k not in {"T", "smoothe"}
    }

    network_seed = int(BASE_SEED + int(round(phi * 1000)) * 100000 + seed * 1000)
    model_kwargs["seed"] = network_seed

    with threadpool_limits(limits=1):
        model = fn.automata_EI_hiermod(**model_kwargs)

    A = np.asarray(model.A, dtype=np.uint8)
    np.fill_diagonal(A, 0)

    n = int(model.n)
    n_e = int(model.e)

    A_e_T = sparse.csr_matrix(A[:n_e, :].T.astype(np.float32))
    A_i_T = sparse.csr_matrix(A[n_e:, :].T.astype(np.float32))

    conn = local_connectivity_metrics(A, n_e)
    rng = np.random.default_rng(network_seed + 123)

    rows = []

    for aval_i in range(N_AVALANCHES_PER_NETWORK):
        seed_node = int(rng.integers(0, n_e))

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
            "network_id": network_id,
            "aval_i": int(aval_i),
            "network_seed": network_seed,
            "seed": seed,
            "seed_node": seed_node,

            "n_neurons": n,
            "n_e": n_e,
            "dt": float(model.dt),
            "max_steps": int(MAX_STEPS),
            "cutoff_s": float(MAX_TIME_S),

            "p_ext": float(AVALANCHE_P_EXT),
            "phi": phi,
            "theta": float(model.theta),
            "ei_ratio": float(start_dic["ei_ratio"]),
            "e_w": float(model.e_w),
            "i_w": float(model.i_w),
            "refractory_steps": int(model.refractory_steps),
        }

        row.update(conn)
        row.update(out)

        row["lifetime_mean_active_density"] = (
            row["size"] / (n * max(row["lifetime_steps"], 1))
        )

        rows.append(row)

    return rows


# ============================================================
# Run: multiprocessing over network seeds
# ============================================================

jobs = design_df.to_dict("records")
trial_rows = []

ctx = mp.get_context("fork")

with ctx.Pool(processes=N_WORKERS) as pool:
    for rows_local in tqdm(
        pool.imap_unordered(run_one_network_job, jobs),
        total=len(jobs),
        desc="Single-phi avalanche sweep over network seeds",
    ):
        trial_rows.extend(rows_local)

avalanche_trials = (
    pd.DataFrame(trial_rows)
    .sort_values(["phi", "seed", "aval_i"])
    .reset_index(drop=True)
)

TRIALS_PATH = OUTDIR / "avalanche_trials_phi_only.csv"
SUMMARY_PATH = OUTDIR / "network_summary_phi_only.csv"
PHI_SUMMARY_PATH = OUTDIR / "phi_summary_phi_only.csv"

avalanche_trials.to_csv(TRIALS_PATH, index=False)

print("Saved trials:", TRIALS_PATH)
display(avalanche_trials.head())


# ============================================================
# Network-level summary
# ============================================================

summary_rows = []

for network_id, g in avalanche_trials.groupby("network_id", sort=True):
    extinct_g = g[g["extinct"]]

    row = {
        "network_id": int(network_id),
        "phi": float(g["phi"].iloc[0]),
        "seed": int(g["seed"].iloc[0]),
        "n_avalanches": int(len(g)),

        "e_w": float(g["e_w"].iloc[0]),
        "i_w": float(g["i_w"].iloc[0]),
        "theta": float(g["theta"].iloc[0]),
        "ei_ratio": float(g["ei_ratio"].iloc[0]),
        "p_ext": float(g["p_ext"].iloc[0]),

        "mean_out_degree": float(g["mean_out_degree"].iloc[0]),
        "mean_e_out_degree_all": float(g["mean_e_out_degree_all"].iloc[0]),
        "mean_e_out_degree_e": float(g["mean_e_out_degree_e"].iloc[0]),

        "frac_extinct": float(g["extinct"].mean()),
        "frac_persistent_at_cutoff": float(g["persistent_at_cutoff"].mean()),

        "mean_size": safe_mean(g["size"]),
        "median_size": safe_percentile(g["size"], 50),
        "p90_size": safe_percentile(g["size"], 90),
        "p95_size": safe_percentile(g["size"], 95),
        "p99_size": safe_percentile(g["size"], 99),
        "cv_size": safe_cv(g["size"]),

        "mean_lifetime_s": safe_mean(g["lifetime_s"]),
        "median_lifetime_s": safe_percentile(g["lifetime_s"], 50),
        "p90_lifetime_s": safe_percentile(g["lifetime_s"], 90),
        "p95_lifetime_s": safe_percentile(g["lifetime_s"], 95),
        "p99_lifetime_s": safe_percentile(g["lifetime_s"], 99),
        "cv_lifetime": safe_cv(g["lifetime_s"]),

        "mean_peak_active": safe_mean(g["peak_active"]),
        "p95_peak_active": safe_percentile(g["peak_active"], 95),
        "p99_peak_active": safe_percentile(g["peak_active"], 99),

        "mean_lifetime_active_density": safe_mean(g["lifetime_mean_active_density"]),

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

    summary_rows.append(row)

network_summary = (
    pd.DataFrame(summary_rows)
    .sort_values(["phi", "seed"])
    .reset_index(drop=True)
)

network_summary.to_csv(SUMMARY_PATH, index=False)

print("Saved network summary:", SUMMARY_PATH)
display(network_summary.head())


# ============================================================
# Phi-level summary
# ============================================================

phi_rows = []

for phi, g in network_summary.groupby("phi", sort=True):
    row = {
        "phi": float(phi),
        "n_networks": int(len(g)),
        "n_avalanches": int(g["n_avalanches"].sum()),
    }

    for col in [
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
        "mean_out_degree",
        "p95_size_extinct_only",
        "p99_size_extinct_only",
        "p95_lifetime_s_extinct_only",
        "p99_lifetime_s_extinct_only",
    ]:
        vals = g[col].replace([np.inf, -np.inf], np.nan)
        n_valid = vals.notna().sum()
        row[f"{col}_mean"] = float(vals.mean())
        row[f"{col}_sem"] = (
            float(vals.std(ddof=1) / np.sqrt(n_valid)) if n_valid > 1 else np.nan
        )

    phi_rows.append(row)

phi_summary = (
    pd.DataFrame(phi_rows)
    .sort_values("phi")
    .reset_index(drop=True)
)

phi_summary.to_csv(PHI_SUMMARY_PATH, index=False)

print("Saved phi summary:", PHI_SUMMARY_PATH)
display(phi_summary)