# ============================================================
# FINITE-SIZE SURVIVAL SCREEN — GEOMETRIC N, MORE SEEDS/AVALS
# Saves after each N; resumable
# ============================================================

import sys, os, time, json, warnings
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

import admin_functions as adfn
import trace_analyse as tfn
import icg_functions as fn

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# 1. Settings
# ============================================================

N_VALUES = np.unique(np.round(np.geomspace(100, 5000, 50)).astype(int))

# keep this broad enough for 0.95 -> 0.05 survival crossings
PHI_VALUES = np.linspace(2, 6.0, 20)

N_NETWORK_SEEDS = 10
N_AVALANCHES_PER_MODEL = 500

MAX_STEPS = 10000
DT = 0.01
MAX_TIME_S = MAX_STEPS * DT

EVAL_STEPS = sorted(set([
    int(round(10.0 / DT)),
    int(round(20.0 / DT)),
    int(round(30.0 / DT)),
    int(round(50.0 / DT)),
    int(MAX_STEPS),
]))
EVAL_STEPS = [s for s in EVAL_STEPS if s <= MAX_STEPS]
EVAL_TIMES_S = [float(s * DT) for s in EVAL_STEPS]

N_WORKERS = 30

BASE_NETWORK_SEED = 31000
BASE_TRIAL_SEED = 51000

start_dic = {
    "n_neurons": 2000,
    "T": 10,
    "dt": 0.01,
    "refractory_steps": 2,
    "ei_ratio": 0.2,
    "e_w": 21.4115,
    "i_w": 21.675,
    "theta": 8.44,
    "p_ext": 0.02,
    "phi": 4.2,
    "smoothe": 0.05,
}

AVALANCHE_P_EXT = 0.0
AVALANCHE_THETA = float(start_dic["theta"])

RUN_TAG = (
    f"GEOM_N{int(N_VALUES.min())}to{int(N_VALUES.max())}_"
    f"{len(N_VALUES)}N_"
    f"phi{PHI_VALUES.min():.2f}to{PHI_VALUES.max():.2f}_"
    f"{len(PHI_VALUES)}phi_"
    f"{N_NETWORK_SEEDS}netseeds_"
    f"{N_AVALANCHES_PER_MODEL}avals_"
    f"{MAX_TIME_S:.0f}s"
).replace(".", "p")

OUTDIR = Path("/home/dburrows/DATA/BLNDEV-WILDTYPE") / f"finite_size_survival_{RUN_TAG}"
OUTDIR.mkdir(parents=True, exist_ok=True)

TRIALS_PATH = OUTDIR / "survival_trials.csv"
SUMMARY_PATH = OUTDIR / "survival_summary_by_N_phi.csv"
MODEL_SUMMARY_PATH = OUTDIR / "model_level_summary.csv"
CONFIG_PATH = OUTDIR / "run_config.json"
DESIGN_PATH = OUTDIR / "design.csv"


# ============================================================
# 2. Save design/config
# ============================================================

design_df = pd.DataFrame(
    [
        {
            "n_neurons": int(n),
            "phi": float(phi),
            "network_seed_index": int(seed_idx),
        }
        for n in N_VALUES
        for phi in PHI_VALUES
        for seed_idx in range(N_NETWORK_SEEDS)
    ]
)
design_df.to_csv(DESIGN_PATH, index=False)

config = {
    "N_VALUES": list(map(int, N_VALUES)),
    "PHI_VALUES": list(map(float, PHI_VALUES)),
    "N_NETWORK_SEEDS": int(N_NETWORK_SEEDS),
    "N_AVALANCHES_PER_MODEL": int(N_AVALANCHES_PER_MODEL),
    "MAX_STEPS": int(MAX_STEPS),
    "DT": float(DT),
    "MAX_TIME_S": float(MAX_TIME_S),
    "EVAL_STEPS": list(map(int, EVAL_STEPS)),
    "EVAL_TIMES_S": list(map(float, EVAL_TIMES_S)),
    "N_WORKERS": int(N_WORKERS),
    "BASE_NETWORK_SEED": int(BASE_NETWORK_SEED),
    "BASE_TRIAL_SEED": int(BASE_TRIAL_SEED),
    "AVALANCHE_P_EXT": float(AVALANCHE_P_EXT),
    "AVALANCHE_THETA": float(AVALANCHE_THETA),
    "start_dic": start_dic,
    "RUN_TAG": RUN_TAG,
    "OUTDIR": str(OUTDIR),
}
with open(CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=2)


# ============================================================
# 3. Helpers
# ============================================================

def sem(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if x.size < 2:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(x.size))


def run_seeded_avalanche(
    A_e_T,
    A_i_T,
    n,
    n_e,
    e_w,
    i_w,
    theta,
    refractory_steps,
    seed_node,
    rng,
    max_steps,
    eval_steps,
):
    state = np.zeros(n, dtype=np.int16)
    state[int(seed_node)] = 1

    total_size = 0
    peak_active = 1

    eval_steps = sorted(int(s) for s in eval_steps)
    eval_active_counts = {int(s): 0 for s in eval_steps}

    for step in range(int(max_steps)):
        active = state == 1
        n_active = int(active.sum())

        if n_active == 0:
            return {
                "lifetime_steps": int(step),
                "lifetime_s": float(step * DT),
                "size": int(total_size),
                "peak_active": int(peak_active),
                "persistent_at_cutoff": False,
                "extinct": True,
                "eval_active_counts": eval_active_counts,
            }

        total_size += n_active
        peak_active = max(peak_active, n_active)

        active_e = active[:n_e].astype(np.float32)
        active_i = active[n_e:].astype(np.float32)

        inp_e = A_e_T @ active_e
        inp_i = A_i_T @ active_i

        net = (e_w * inp_e) - (i_w * inp_i)
        p_net = expit(net - theta)

        has_input = (inp_e > 0) | (inp_i > 0)
        p_net[~has_input] = 0.0

        quiescent = state == 0
        new_active = quiescent & (rng.random(n) < p_net)

        new_state = np.zeros_like(state)
        new_state[active] = 2

        refractory = state >= 2
        new_state[refractory] = state[refractory] + 1

        done_refractory = new_state > (refractory_steps + 1)
        new_state[done_refractory] = 0

        new_state[new_active] = 1
        state = new_state

        t_after_update = step + 1

        if t_after_update in eval_active_counts:
            eval_active_counts[t_after_update] = int(np.sum(state == 1))

        if not np.any(state == 1):
            return {
                "lifetime_steps": int(t_after_update),
                "lifetime_s": float(t_after_update * DT),
                "size": int(total_size),
                "peak_active": int(peak_active),
                "persistent_at_cutoff": False,
                "extinct": True,
                "eval_active_counts": eval_active_counts,
            }

    return {
        "lifetime_steps": int(max_steps),
        "lifetime_s": float(max_steps * DT),
        "size": int(total_size),
        "peak_active": int(peak_active),
        "persistent_at_cutoff": True,
        "extinct": False,
        "eval_active_counts": eval_active_counts,
    }


def run_one_network_job(job):
    n_neurons = int(job["n_neurons"])
    phi = float(job["phi"])
    seed_index = int(job["network_seed_index"])

    pars = dict(start_dic)
    pars["n_neurons"] = n_neurons
    pars["phi"] = phi
    pars["p_ext"] = AVALANCHE_P_EXT
    pars["theta"] = AVALANCHE_THETA

    model_kwargs = {
        key: value
        for key, value in pars.items()
        if key not in {"T", "smoothe"}
    }

    network_seed = int(BASE_NETWORK_SEED + n_neurons * 100 + seed_index)
    model_kwargs["seed"] = network_seed

    with threadpool_limits(limits=1):
        model = fn.automata_EI_hiermod(**model_kwargs)

    A = np.asarray(model.A, dtype=np.uint8)
    np.fill_diagonal(A, 0)

    n = int(model.n)
    n_e = int(model.e)

    A_e_T = sparse.csr_matrix(A[:n_e, :].T.astype(np.float32))
    A_i_T = sparse.csr_matrix(A[n_e:, :].T.astype(np.float32))

    mean_out_degree = float(A.sum(axis=1).mean())

    e_w = float(model.e_w)
    i_w = float(model.i_w)
    theta = float(model.theta)
    refractory_steps = int(model.refractory_steps)

    del A
    del model

    trial_rng = np.random.default_rng(
        BASE_TRIAL_SEED
        + n_neurons * 1000
        + int(round(phi * 1000)) * 10
        + seed_index
    )

    rows = []

    for aval_i in range(N_AVALANCHES_PER_MODEL):
        seed_node = int(trial_rng.integers(0, n_e))

        out = run_seeded_avalanche(
            A_e_T=A_e_T,
            A_i_T=A_i_T,
            n=n,
            n_e=n_e,
            e_w=e_w,
            i_w=i_w,
            theta=theta,
            refractory_steps=refractory_steps,
            seed_node=seed_node,
            rng=trial_rng,
            max_steps=MAX_STEPS,
            eval_steps=EVAL_STEPS,
        )

        row = {
            "n_neurons": n_neurons,
            "phi": phi,
            "network_seed_index": seed_index,
            "network_seed": network_seed,
            "aval_i": int(aval_i),
            "seed_node": seed_node,
            "mean_out_degree": mean_out_degree,
            "p_ext": AVALANCHE_P_EXT,
            "theta": AVALANCHE_THETA,
            "max_steps": MAX_STEPS,
            "cutoff_s": MAX_TIME_S,
            "lifetime_steps": out["lifetime_steps"],
            "lifetime_s": out["lifetime_s"],
            "size": out["size"],
            "peak_active": out["peak_active"],
            "persistent_at_cutoff": out["persistent_at_cutoff"],
            "extinct": out["extinct"],
            "lifetime_mean_active_density": (
                out["size"] / (n_neurons * max(out["lifetime_steps"], 1))
            ),
        }

        for eval_step, eval_time_s in zip(EVAL_STEPS, EVAL_TIMES_S):
            active_count = int(out["eval_active_counts"][eval_step])

            if eval_step >= MAX_STEPS:
                survived = bool(out["persistent_at_cutoff"])
            else:
                survived = bool(out["lifetime_steps"] >= eval_step)

            row[f"survived_{eval_time_s:g}s"] = survived
            row[f"active_count_{eval_time_s:g}s"] = active_count
            row[f"rho_uncond_{eval_time_s:g}s"] = active_count / n_neurons
            row[f"rho_cond_{eval_time_s:g}s"] = (
                active_count / n_neurons if survived else np.nan
            )

        rows.append(row)

    return rows


def append_rows_to_csv(rows, path):
    df = pd.DataFrame(rows)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)


def expected_rows_for_N():
    return int(len(PHI_VALUES) * N_NETWORK_SEEDS * N_AVALANCHES_PER_MODEL)


def already_completed_N():
    if not TRIALS_PATH.exists():
        return set()

    try:
        trials = pd.read_csv(TRIALS_PATH, usecols=["n_neurons"])
        counts = trials["n_neurons"].dropna().astype(int).value_counts()
        expected = expected_rows_for_N()
        return set(counts[counts >= expected].index.astype(int))
    except Exception:
        return set()


def rebuild_summary():
    if not TRIALS_PATH.exists():
        return None

    trials = pd.read_csv(TRIALS_PATH)

    model_rows = []

    for (n, phi, seed_idx), g in trials.groupby(
        ["n_neurons", "phi", "network_seed_index"],
        sort=True,
    ):
        row = {
            "n_neurons": int(n),
            "phi": float(phi),
            "network_seed_index": int(seed_idx),
            "n_avalanches": int(len(g)),
            "mean_out_degree": float(g["mean_out_degree"].iloc[0]),
            "frac_persistent_at_cutoff": float(g["persistent_at_cutoff"].mean()),
            "mean_lifetime_s": float(g["lifetime_s"].mean()),
            "mean_size": float(g["size"].mean()),
            "mean_peak_active": float(g["peak_active"].mean()),
            "lifetime_mean_active_density": float(
                g["lifetime_mean_active_density"].mean()
            ),
        }

        for eval_time_s in EVAL_TIMES_S:
            eval_step = int(round(eval_time_s / DT))

            surv_col = f"survived_{eval_time_s:g}s"
            uncond_col = f"rho_uncond_{eval_time_s:g}s"
            cond_col = f"rho_cond_{eval_time_s:g}s"
            active_col = f"active_count_{eval_time_s:g}s"

            if surv_col in g.columns:
                survived = g[surv_col].astype(bool)
            else:
                survived = (
                    g["persistent_at_cutoff"].astype(bool)
                    if eval_step >= MAX_STEPS
                    else g["lifetime_steps"] >= eval_step
                )

            row[f"P_surv_{eval_time_s:g}s"] = float(survived.mean())

            row[f"active_count_{eval_time_s:g}s"] = (
                float(g[active_col].mean()) if active_col in g.columns else np.nan
            )

            row[f"rho_uncond_{eval_time_s:g}s"] = (
                float(g[uncond_col].mean()) if uncond_col in g.columns else np.nan
            )

            row[f"rho_cond_{eval_time_s:g}s"] = (
                float(g[cond_col].mean(skipna=True)) if cond_col in g.columns else np.nan
            )

        model_rows.append(row)

    model_df = (
        pd.DataFrame(model_rows)
        .sort_values(["n_neurons", "phi", "network_seed_index"])
        .reset_index(drop=True)
    )
    model_df.to_csv(MODEL_SUMMARY_PATH, index=False)

    base_metrics = [
        "mean_out_degree",
        "frac_persistent_at_cutoff",
        "mean_lifetime_s",
        "mean_size",
        "mean_peak_active",
        "lifetime_mean_active_density",
    ]

    eval_metrics = []
    for eval_time_s in EVAL_TIMES_S:
        eval_metrics.extend([
            f"P_surv_{eval_time_s:g}s",
            f"active_count_{eval_time_s:g}s",
            f"rho_uncond_{eval_time_s:g}s",
            f"rho_cond_{eval_time_s:g}s",
        ])

    metrics = base_metrics + eval_metrics
    summary_rows = []

    for (n, phi), g in model_df.groupby(["n_neurons", "phi"], sort=True):
        row = {
            "n_neurons": int(n),
            "phi": float(phi),
            "n_networks": int(len(g)),
        }

        for metric in metrics:
            row[f"{metric}_mean"] = float(g[metric].mean())
            row[f"{metric}_sem"] = sem(g[metric])

        summary_rows.append(row)

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["n_neurons", "phi"])
        .reset_index(drop=True)
    )
    summary.to_csv(SUMMARY_PATH, index=False)

    print("Saved model summary:", MODEL_SUMMARY_PATH)
    print("Saved pooled summary:", SUMMARY_PATH)
    print("Available P_surv columns:")
    print([c for c in summary.columns if c.startswith("P_surv")])

    return summary


# ============================================================
# 4. Main
# ============================================================

def main():
    print("FINITE-SIZE SURVIVAL SCREEN — GEOMETRIC N, MORE SEEDS/AVALS")
    print(f"OUTDIR: {OUTDIR}")
    print(f"N_VALUES: {N_VALUES}")
    print(f"PHI_VALUES: {np.round(PHI_VALUES, 4)}")
    print(f"N_NETWORK_SEEDS: {N_NETWORK_SEEDS}")
    print(f"N_AVALANCHES_PER_MODEL: {N_AVALANCHES_PER_MODEL}")
    print(f"MAX_STEPS: {MAX_STEPS} = {MAX_TIME_S:.1f} s")
    print(f"EVAL_TIMES_S: {EVAL_TIMES_S}")
    print(f"N_WORKERS: {N_WORKERS}")
    print(f"Expected rows per N: {expected_rows_for_N():,}")
    print(f"Total networks: {len(N_VALUES) * len(PHI_VALUES) * N_NETWORK_SEEDS:,}")
    print(f"Total avalanches: {len(N_VALUES) * len(PHI_VALUES) * N_NETWORK_SEEDS * N_AVALANCHES_PER_MODEL:,}")

    completed = already_completed_N()
    print(f"Already completed N: {sorted(completed)}")

    ctx = mp.get_context("fork")

    for n_neurons in N_VALUES:
        n_neurons = int(n_neurons)

        if n_neurons in completed:
            print(f"\nSkipping N={n_neurons}; already complete in {TRIALS_PATH}")
            continue

        print("\n================================================")
        print(f"Starting N={n_neurons}")
        print("================================================")

        jobs = [
            {
                "n_neurons": int(n_neurons),
                "phi": float(phi),
                "network_seed_index": int(seed_index),
            }
            for phi in PHI_VALUES
            for seed_index in range(N_NETWORK_SEEDS)
        ]

        rng_jobs = np.random.default_rng(123 + n_neurons)
        rng_jobs.shuffle(jobs)

        t0 = time.perf_counter()
        rows_for_N = []

        with ctx.Pool(processes=N_WORKERS) as pool:
            for rows_local in tqdm(
                pool.imap_unordered(run_one_network_job, jobs),
                total=len(jobs),
                desc=f"N={n_neurons}",
            ):
                rows_for_N.extend(rows_local)

        append_rows_to_csv(rows_for_N, TRIALS_PATH)
        summary = rebuild_summary()

        elapsed = time.perf_counter() - t0

        print(f"Finished N={n_neurons}")
        print(f"Rows added: {len(rows_for_N):,}")
        print(f"Elapsed: {elapsed / 60:.2f} min")
        print(f"Saved trials: {TRIALS_PATH}")
        print(f"Saved summary: {SUMMARY_PATH}")

        if summary is not None:
            completed_now = sorted(summary["n_neurons"].unique().astype(int))
            print("Completed N in summary:", completed_now)

    print("\nAll requested N values complete or already present.")
    print(f"Final trials CSV: {TRIALS_PATH}")
    print(f"Final model summary CSV: {MODEL_SUMMARY_PATH}")
    print(f"Final pooled summary CSV: {SUMMARY_PATH}")
    print(f"Run config JSON: {CONFIG_PATH}")


if __name__ == "__main__":
    main()