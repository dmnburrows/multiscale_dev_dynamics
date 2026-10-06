import os

# Set before importing NumPy/Pandas
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import inspect
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import icg_functions as fn


# ============================================================
# Main parameters — same as previous run
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
    "p_ext": 0.00,
    "phi": 4.2,
    "smoothe": 0.05,
}


# ============================================================
# Sweep settings
# ============================================================

OUTDIR = Path(
    "/home/dburrows/DATA/BLNDEV-WILDTYPE/"
    "phi_scaling_newpars_ew11p7_iw22p4_pext0_"
    "phi2p5to4p5_30steps_40seeds_smooth005"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

PHI_VALUES = np.linspace(2.5, 4.5, 30)

N_SEEDS = 40
N_WORKERS = 30
BASE_SEED = 950000

SMOOTHE = float(start_dic["smoothe"])
T_RUN = float(start_dic["T"])

TARGET_MV = 1.50
TARGET_TAU = 0.20

with open(OUTDIR / "config.json", "w") as f:
    json.dump(
        {
            "PHI_VALUES": PHI_VALUES.tolist(),
            "N_SEEDS": N_SEEDS,
            "N_WORKERS": N_WORKERS,
            "BASE_SEED": BASE_SEED,
            "SMOOTHE": SMOOTHE,
            "T_RUN": T_RUN,
            "TARGET_MV": TARGET_MV,
            "TARGET_TAU": TARGET_TAU,
            "start_dic": start_dic,
        },
        f,
        indent=2,
    )

print("OUTDIR:", OUTDIR)
print("TOTAL SIMS:", len(PHI_VALUES) * N_SEEDS)
print("WORKERS:", N_WORKERS)
print("PHI SPACING:", PHI_VALUES[1] - PHI_VALUES[0])


# ============================================================
# Helpers
# ============================================================

def sem(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    return (
        float(x.std(ddof=1) / np.sqrt(len(x)))
        if len(x) > 1
        else np.nan
    )


def fit_power_slope(
    g,
    y_col,
    x_col="mean_cluster_size",
    exclude_first=True,
    exclude_last=False,
    min_points=3,
):
    dfit = (
        g[[x_col, y_col]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    dfit = dfit[
        (dfit[x_col] > 0) & (dfit[y_col] > 0)
    ].sort_values(x_col)

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


def make_pars(phi, seed):
    pars = dict(start_dic)
    pars.pop("T", None)
    pars.pop("smoothe", None)

    sig = inspect.signature(fn.automata_EI_hiermod.__init__)
    valid_args = set(sig.parameters.keys())

    if "slope" in valid_args:
        pars.pop("phi", None)
        pars["slope"] = float(phi)
    elif "phi" in valid_args:
        pars.pop("slope", None)
        pars["phi"] = float(phi)
    else:
        raise ValueError("Model expects neither 'slope' nor 'phi'.")

    pars["seed"] = int(seed)
    return {k: v for k, v in pars.items() if k in valid_args}


def run_one(job):
    sim_id, phi, seed, seed_idx = job

    pars = make_pars(phi=phi, seed=seed)

    model = fn.automata_EI_hiermod(**pars)
    spikes, pop_rate = fn.run_model(model, T=T_RUN)

    dt = float(pars["dt"])

    mean_rate_hz = float(spikes.mean() / dt)
    pop_rate_mean_hz = float(np.mean(pop_rate))
    pop_rate_std_hz = float(np.std(pop_rate))
    frac_silent_frames = float(np.mean(spikes.sum(axis=0) == 0))
    frac_active_neurons = float(np.mean(spikes.sum(axis=1) > 0))

    spikes_smooth = fn.exp_smooth_spikes(
        spikes,
        dt=dt,
        tau=SMOOTHE,
    )

    metric_row, gen_df = fn.compute_icg_metrics(
        spikes=spikes_smooth,
        dt=dt,
    )

    gen_df = gen_df.copy()

    rename_map = {
        "mean_variance": "MV",
        "mean_variance_norm": "MV_norm",
        "timescale": "TAU",
        "timescale_norm": "TAU_norm",
        "corr_kurtosis": "kurtosis_corr",
    }

    for old, new in rename_map.items():
        if old in gen_df.columns and new not in gen_df.columns:
            gen_df[new] = gen_df[old]

    if "MV" in gen_df.columns and "mean_variance" not in gen_df.columns:
        gen_df["mean_variance"] = gen_df["MV"]

    if "MV_norm" in gen_df.columns and "mean_variance_norm" not in gen_df.columns:
        gen_df["mean_variance_norm"] = gen_df["MV_norm"]

    if "TAU" in gen_df.columns and "timescale" not in gen_df.columns:
        gen_df["timescale"] = gen_df["TAU"]

    if "TAU_norm" in gen_df.columns and "timescale_norm" not in gen_df.columns:
        gen_df["timescale_norm"] = gen_df["TAU_norm"]

    if "kurtosis_corr" not in gen_df.columns and "corr_kurtosis" in gen_df.columns:
        gen_df["kurtosis_corr"] = gen_df["corr_kurtosis"]

    mv_y = (
        "mean_variance_norm"
        if "mean_variance_norm" in gen_df.columns
        else "MV_norm"
    )
    tau_y = (
        "timescale_norm"
        if "timescale_norm" in gen_df.columns
        else "TAU_norm"
    )

    MV_alpha, MV_r2, MV_n_points = fit_power_slope(
        gen_df,
        y_col=mv_y,
        x_col="mean_cluster_size",
        exclude_first=True,
        exclude_last=False,
    )

    TAU_beta, TAU_r2, TAU_n_points = fit_power_slope(
        gen_df,
        y_col=tau_y,
        x_col="mean_cluster_size",
        exclude_first=True,
        exclude_last=False,
    )

    score = (
        abs(MV_alpha - TARGET_MV)
        + 2.0 * abs(TAU_beta - TARGET_TAU)
        if np.isfinite(MV_alpha) and np.isfinite(TAU_beta)
        else np.nan
    )

    exp_row = {
        "sim_id": int(sim_id),
        "phi": float(phi),
        "seed": int(seed),
        "seed_idx": int(seed_idx),

        "MV_alpha": MV_alpha,
        "MV_r2": MV_r2,
        "MV_n_points": MV_n_points,

        "TAU_beta": TAU_beta,
        "TAU_r2": TAU_r2,
        "TAU_n_points": TAU_n_points,

        "score": score,

        "mean_rate_hz": mean_rate_hz,
        "pop_rate_mean_hz": pop_rate_mean_hz,
        "pop_rate_std_hz": pop_rate_std_hz,
        "frac_silent_frames": frac_silent_frames,
        "frac_active_neurons": frac_active_neurons,

        "dt": dt,
        "T": float(T_RUN),
        "smoothe": float(SMOOTHE),

        "theta": float(pars.get("theta", np.nan)),
        "p_ext": float(pars.get("p_ext", np.nan)),
        "e_w": float(pars.get("e_w", np.nan)),
        "i_w": float(pars.get("i_w", np.nan)),
        "ei_ratio": float(pars.get("ei_ratio", np.nan)),
        "refractory_steps": int(pars.get("refractory_steps", -1)),
    }

    gen_df["sim_id"] = int(sim_id)
    gen_df["phi"] = float(phi)
    gen_df["seed"] = int(seed)
    gen_df["seed_idx"] = int(seed_idx)
    gen_df["mean_rate_hz"] = mean_rate_hz
    gen_df["smoothe"] = float(SMOOTHE)

    return exp_row, gen_df


# ============================================================
# Build design
# ============================================================

jobs = []
sim_id = 0

for phi in PHI_VALUES:
    for seed_idx in range(N_SEEDS):
        seed = BASE_SEED + sim_id
        jobs.append(
            (int(sim_id), float(phi), int(seed), int(seed_idx))
        )
        sim_id += 1

design = pd.DataFrame(
    jobs,
    columns=["sim_id", "phi", "seed", "seed_idx"],
)

design.to_csv(OUTDIR / "phi_scaling_design.csv", index=False)


# ============================================================
# Run sweep — same Linux/fork workflow as before
# ============================================================

rows = []
gen_rows = []

ctx = mp.get_context("fork")

with ctx.Pool(processes=min(N_WORKERS, len(jobs))) as pool:
    for exp_row, gen_df in tqdm(
        pool.imap_unordered(run_one, jobs, chunksize=1),
        total=len(jobs),
        desc="Phi scaling sweep: 2.5–4.5",
    ):
        rows.append(exp_row)
        gen_rows.append(gen_df)

df_exp = (
    pd.DataFrame(rows)
    .sort_values(["phi", "seed_idx"])
    .reset_index(drop=True)
)

df_icg = (
    pd.concat(gen_rows, ignore_index=True)
    .sort_values(["phi", "seed_idx", "gen"])
    .reset_index(drop=True)
)

df_exp.to_csv(
    OUTDIR / "phi_scaling_seed_exponents.csv",
    index=False,
)
df_icg.to_csv(
    OUTDIR / "phi_scaling_icg_rows.csv",
    index=False,
)

print("Saved:", OUTDIR / "phi_scaling_seed_exponents.csv")
print("Saved:", OUTDIR / "phi_scaling_icg_rows.csv")


# ============================================================
# Summary by phi
# ============================================================

phi_summary = (
    df_exp
    .groupby("phi", as_index=False)
    .agg(
        n=("seed", "count"),

        MV_alpha_mean=("MV_alpha", "mean"),
        MV_alpha_sem=("MV_alpha", sem),
        MV_r2_mean=("MV_r2", "mean"),
        MV_r2_sem=("MV_r2", sem),

        TAU_beta_mean=("TAU_beta", "mean"),
        TAU_beta_sem=("TAU_beta", sem),
        TAU_r2_mean=("TAU_r2", "mean"),
        TAU_r2_sem=("TAU_r2", sem),

        score_mean=("score", "mean"),
        score_sem=("score", sem),

        mean_rate_hz=("mean_rate_hz", "mean"),
        mean_rate_hz_sem=("mean_rate_hz", sem),

        pop_rate_mean_hz=("pop_rate_mean_hz", "mean"),
        pop_rate_std_hz=("pop_rate_std_hz", "mean"),

        frac_silent_frames=("frac_silent_frames", "mean"),
        frac_active_neurons=("frac_active_neurons", "mean"),
    )
    .sort_values("phi")
    .reset_index(drop=True)
)

phi_summary.to_csv(
    OUTDIR / "phi_scaling_summary.csv",
    index=False,
)

print("\nBest phi values by score:")
print(
    phi_summary
    .sort_values("score_mean")
    [
        [
            "phi",
            "MV_alpha_mean",
            "MV_alpha_sem",
            "TAU_beta_mean",
            "TAU_beta_sem",
            "score_mean",
            "MV_r2_mean",
            "TAU_r2_mean",
            "mean_rate_hz",
            "n",
        ]
    ]
    .head(15)
    .round(4)
    .to_string(index=False)
)

print("\nDone.")
print("Outputs saved to:", OUTDIR)