# ============================================================
# FULL PHI x THETA x P_EXT SWEEP
# RUN + SAVE ONLY — NO PLOTTING
#
# Uses new icg_functions.py:
#   fn.model_outs(pars, seed)
#
# Assumes fn.model_outs now:
#   - pops "smoothe" from pars
#   - applies exp_smooth_spikes before compute_icg_metrics
#
# Sweep:
#   phi:   1.5 -> 10.0, 10 values
#   theta: 4.0 -> 12.0, 10 values
#   p_ext: 0.01 -> 0.10, 10 values
#   smoothe fixed at 0.05
#
# Total:
#   10 x 10 x 10 = 1000 parameter combos
#   with N_SEEDS=5 -> 5000 simulations
#
# Saves:
#   phi_theta_pext_smooth005_design.csv
#   phi_theta_pext_smooth005_seed_metrics.csv
#   phi_theta_pext_smooth005_generation_metrics.csv
#   phi_theta_pext_smooth005_summary_metrics.csv
#   phi_theta_pext_smooth005_generation_summary.csv
#   best50_phi_theta_pext_smooth005_norm_target.csv
# ============================================================

import os
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import icg_functions as fn


# ============================================================
# Settings
# ============================================================

start_dic = {'n_neurons': 2000,
 'T': 10,
 'dt': 0.01,
 'refractory_steps': 2,
 'ei_ratio': 0.2,
 'e_w': 21.4115,
 'i_w': 21.675,
 'theta': 7.5,
 'p_ext': 0.015,
 'slope': 2.25,
  'smoothe': 0.05}
start_dic

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

OUTDIR = Path("/home/dburrows/DATA/BLNDEV-WILDTYPE/phi_theta_pext_smooth005_10x10x10")
OUTDIR.mkdir(parents=True, exist_ok=True)

PHI_VALUES = np.linspace(1.5, 10.0, 10)
THETA_VALUES = np.linspace(4.0, 12.0, 10)
P_EXT_VALUES = np.linspace(0.01, 0.10, 10)

SMOOTHE_FIXED = 0.05

N_SEEDS = 1
N_WORKERS = 30
BASE_SEED = 891000

T_RUN = float(start_dic.get("T", 10.0))

TARGET_ALPHA = 1.5
TARGET_BETA = 0.2

print("phi:", PHI_VALUES)
print("theta:", THETA_VALUES)
print("p_ext:", P_EXT_VALUES)
print("smoothe fixed:", SMOOTHE_FIXED)
print("N seeds:", N_SEEDS)
print("workers:", N_WORKERS)
print("parameter combos:", len(PHI_VALUES) * len(THETA_VALUES) * len(P_EXT_VALUES))
print("total sims:", len(PHI_VALUES) * len(THETA_VALUES) * len(P_EXT_VALUES) * N_SEEDS)
print("OUTDIR:", OUTDIR)


# ============================================================
# Minimal wrapper
# ============================================================

def make_pars_from_start(updates):
    """
    Copy start_dic and apply updates.

    Converts old naming:
      slope -> phi

    Keeps smoothe in pars so fn.model_outs can pop/use it.
    """
    pars = dict(start_dic)

    if "slope" in pars and "phi" not in pars:
        pars["phi"] = float(pars.pop("slope"))
    else:
        pars.pop("slope", None)

    pars["T"] = float(pars.get("T", T_RUN))

    for k, v in updates.items():
        pars[k] = v

    return pars


def run_model_outs_with_updates(job):
    phi, theta, p_ext, seed = job

    pars = make_pars_from_start({
        "phi": float(phi),
        "theta": float(theta),
        "p_ext": float(p_ext),
        "smoothe": float(SMOOTHE_FIXED),
        "seed": int(seed),
        "n_neurons": int(start_dic["n_neurons"]),
        "T": float(T_RUN),
    })

    out, gen_df = fn.model_outs(pars, seed=int(seed))

    p_self_zero_input = float(1 / (1 + np.exp(float(theta))))

    out["phi"] = float(phi)
    out["theta"] = float(theta)
    out["p_ext"] = float(p_ext)
    out["smoothe"] = float(SMOOTHE_FIXED)
    out["p_self_zero_input"] = p_self_zero_input

    gen_df["phi"] = float(phi)
    gen_df["theta"] = float(theta)
    gen_df["p_ext"] = float(p_ext)
    gen_df["smoothe"] = float(SMOOTHE_FIXED)
    gen_df["p_self_zero_input"] = p_self_zero_input

    return out, gen_df


# ============================================================
# Build jobs/design
# ============================================================

jobs = []
job_i = 0

for phi in PHI_VALUES:
    for theta in THETA_VALUES:
        for p_ext in P_EXT_VALUES:
            for s in range(N_SEEDS):
                job_i += 1
                jobs.append((
                    float(phi),
                    float(theta),
                    float(p_ext),
                    int(BASE_SEED + job_i),
                ))

design = pd.DataFrame(jobs, columns=["phi", "theta", "p_ext", "seed"])
design["smoothe"] = SMOOTHE_FIXED
design["p_self_zero_input"] = 1 / (1 + np.exp(design["theta"]))
design["T"] = T_RUN

design_path = OUTDIR / "phi_theta_pext_smooth005_design.csv"
design.to_csv(design_path, index=False)

print("Saved design:", design_path)


# ============================================================
# Run jobs
# ============================================================

ctx = mp.get_context("fork")

rows = []
gen_rows = []

N_WORKERS_ACTUAL = min(N_WORKERS, len(jobs))

with ctx.Pool(processes=N_WORKERS_ACTUAL) as pool:
    for out, gen_df in tqdm(
        pool.imap_unordered(run_model_outs_with_updates, jobs, chunksize=1),
        total=len(jobs),
        desc="Running phi/theta/p_ext sweep with smoothe=0.05",
    ):
        rows.append(out)
        gen_rows.append(gen_df)


# ============================================================
# Save raw outputs
# ============================================================

df = (
    pd.DataFrame(rows)
    .sort_values(["phi", "theta", "p_ext", "seed"])
    .reset_index(drop=True)
)

gen_all = (
    pd.concat(gen_rows, ignore_index=True)
    .sort_values(["phi", "theta", "p_ext", "seed", "gen"])
    .reset_index(drop=True)
)

seed_metrics_path = OUTDIR / "phi_theta_pext_smooth005_seed_metrics.csv"
gen_metrics_path = OUTDIR / "phi_theta_pext_smooth005_generation_metrics.csv"

df.to_csv(seed_metrics_path, index=False)
gen_all.to_csv(gen_metrics_path, index=False)

print("Saved:", seed_metrics_path)
print("Saved:", gen_metrics_path)


# ============================================================
# Summaries using fn.sem
# ============================================================

summary = (
    df
    .groupby(["phi", "theta", "p_ext", "smoothe"], as_index=False)
    .agg(
        n=("seed", "count"),

        p_self_zero_input=("p_self_zero_input", "mean"),

        mv_alpha_mean=("mv_alpha", "mean"),
        mv_alpha_sem=("mv_alpha", fn.sem),
        mv_r2_mean=("mv_r2", "mean"),
        mv_r2_sem=("mv_r2", fn.sem),

        ts_beta_mean=("ts_beta", "mean"),
        ts_beta_sem=("ts_beta", fn.sem),
        ts_r2_mean=("ts_r2", "mean"),
        ts_r2_sem=("ts_r2", fn.sem),

        mv_alpha_norm_mean=("mv_alpha_norm", "mean"),
        mv_alpha_norm_sem=("mv_alpha_norm", fn.sem),
        mv_r2_norm_mean=("mv_r2_norm", "mean"),
        mv_r2_norm_sem=("mv_r2_norm", fn.sem),

        ts_beta_norm_mean=("ts_beta_norm", "mean"),
        ts_beta_norm_sem=("ts_beta_norm", fn.sem),
        ts_r2_norm_mean=("ts_r2_norm", "mean"),
        ts_r2_norm_sem=("ts_r2_norm", fn.sem),

        mean_variance_l0_mean=("mean_variance_l0", "mean"),
        timescale_l0_mean=("timescale_l0", "mean"),

        mean_rate_hz_mean=("mean_rate_hz", "mean"),
        mean_rate_hz_sem=("mean_rate_hz", fn.sem),

        pop_rate_mean_hz_mean=("pop_rate_mean_hz", "mean"),
        pop_rate_std_hz_mean=("pop_rate_std_hz", "mean"),

        frac_silent_frames_mean=("frac_silent_frames", "mean"),
        frac_silent_frames_sem=("frac_silent_frames", fn.sem),

        frac_active_neurons_mean=("frac_active_neurons", "mean"),
        frac_active_neurons_sem=("frac_active_neurons", fn.sem),

        n_icg_gens_mean=("n_icg_gens", "mean"),
    )
    .sort_values(["phi", "theta", "p_ext"])
    .reset_index(drop=True)
)

summary["target_score_raw"] = (
    np.abs(summary["mv_alpha_mean"] - TARGET_ALPHA)
    + 2.0 * np.abs(summary["ts_beta_mean"] - TARGET_BETA)
)

summary["target_score_norm"] = (
    np.abs(summary["mv_alpha_norm_mean"] - TARGET_ALPHA)
    + 2.0 * np.abs(summary["ts_beta_norm_mean"] - TARGET_BETA)
)

summary["target_score_norm_r2pen"] = (
    summary["target_score_norm"]
    + 0.5 * np.maximum(0, 0.8 - summary["mv_r2_norm_mean"])
    + 0.5 * np.maximum(0, 0.8 - summary["ts_r2_norm_mean"])
)

summary_path = OUTDIR / "phi_theta_pext_smooth005_summary_metrics.csv"
summary.to_csv(summary_path, index=False)

print("Saved:", summary_path)


gen_summary = (
    gen_all
    .groupby(["phi", "theta", "p_ext", "smoothe", "gen"], as_index=False)
    .agg(
        n=("seed", "count"),

        p_self_zero_input=("p_self_zero_input", "mean"),

        n_clusters_mean=("n_clusters", "mean"),

        mean_cluster_size_mean=("mean_cluster_size", "mean"),
        mean_cluster_size_sem=("mean_cluster_size", fn.sem),

        mean_activity_mean=("mean_activity", "mean"),
        mean_activity_sem=("mean_activity", fn.sem),

        mean_variance_mean=("mean_variance", "mean"),
        mean_variance_sem=("mean_variance", fn.sem),

        mean_variance_norm_mean=("mean_variance_norm", "mean"),
        mean_variance_norm_sem=("mean_variance_norm", fn.sem),

        timescale_mean=("timescale", "mean"),
        timescale_sem=("timescale", fn.sem),

        timescale_norm_mean=("timescale_norm", "mean"),
        timescale_norm_sem=("timescale_norm", fn.sem),

        corr_kurtosis_mean=("corr_kurtosis", "mean"),
        corr_kurtosis_sem=("corr_kurtosis", fn.sem),
    )
    .sort_values(["phi", "theta", "p_ext", "gen"])
    .reset_index(drop=True)
)

gen_summary_path = OUTDIR / "phi_theta_pext_smooth005_generation_summary.csv"
gen_summary.to_csv(gen_summary_path, index=False)

print("Saved:", gen_summary_path)


# ============================================================
# Best rows
# ============================================================

best50 = (
    summary
    .replace([np.inf, -np.inf], np.nan)
    .dropna(subset=["mv_alpha_norm_mean", "ts_beta_norm_mean", "target_score_norm"])
    .sort_values("target_score_norm")
    .head(50)
    .reset_index(drop=True)
)

best50["rank"] = np.arange(1, len(best50) + 1)

best_path = OUTDIR / "best50_phi_theta_pext_smooth005_norm_target.csv"
best50.to_csv(best_path, index=False)

print("\nBest 50:")
print(
    best50[
        [
            "rank",
            "phi", "theta", "p_ext", "smoothe", "p_self_zero_input",
            "target_score_norm",
            "mv_alpha_norm_mean", "mv_r2_norm_mean",
            "ts_beta_norm_mean", "ts_r2_norm_mean",
            "mean_rate_hz_mean",
            "frac_silent_frames_mean",
            "frac_active_neurons_mean",
            "n",
        ]
    ].round(6).to_string(index=False)
)

print("Saved:", best_path)


# ============================================================
# Done
# ============================================================

print("\nDone.")
print("smoothe used:", SMOOTHE_FIXED)
print("Total parameter combos:", len(PHI_VALUES) * len(THETA_VALUES) * len(P_EXT_VALUES))
print("Total simulations:", len(jobs))
print("Outputs saved to:", OUTDIR)

print("\nFiles:")
print("  design:             ", design_path)
print("  seed metrics:       ", seed_metrics_path)
print("  generation metrics: ", gen_metrics_path)
print("  summary:            ", summary_path)
print("  generation summary: ", gen_summary_path)
print("  best 50:            ", best_path)