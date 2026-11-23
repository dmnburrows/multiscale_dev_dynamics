import os
for key in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[key] = "1"

import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.special import expit
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits

import icg_functions as fn


# ================= SETTINGS =================
RUN_PATH = "/home/dburrows/DATA/BLNDEV-WILDTYPE/sequential_perturbations_phi4_sd005_run6"

PHI = 2.8
N_WORKERS = 30
N_SEEDS = 30
N_PERTURBATIONS = 5
EI_STEP_SD = 0.1
N_AVALANCHES = 10000
RANDOM_SEED = 42

GRID_DIR = Path(
    "/home/dburrows/DATA/BLNDEV-WILDTYPE/"
    "grid_ew_phi_iw22p4_sparse_avalanche_icg_"
    "50ew_50phis_20seeds_10000avals"
)

SAVEDIR = Path(RUN_PATH)

# ================= HELPERS =================
def atomic_csv(df, path, compression=None):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, compression=compression)
    os.replace(tmp, path)


def safe_mean(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else np.nan


def response_decay_timescale(counts, dt):
    x = np.asarray(counts, dtype=float)
    if not x.size:
        return np.nan
    peak = x.max()
    if peak <= 0:
        return 0.0
    post = x[np.argmax(x):]
    below = np.flatnonzero(post <= peak / np.e)
    return float((below[0] if below.size else post.size - 1) * dt)


def branching_metrics(counts, extinct):
    counts = np.asarray(counts, dtype=float)
    out = dict(
        sigma_origin=np.nan, br_reg_slope=np.nan,
        br_reg_intercept=np.nan, br_reg_r2=np.nan, br_n_pairs=0,
    )
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

    denom = np.sum(x * x)
    out["sigma_origin"] = float(np.sum(x * y) / denom)

    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    out["br_reg_intercept"], out["br_reg_slope"] = map(float, beta)
    ss = np.sum((y - y.mean()) ** 2)
    out["br_reg_r2"] = (
        float(1 - np.sum((y - X @ beta) ** 2) / ss)
        if ss > 0 else np.nan
    )
    return out


def run_avalanche(
    A_e_T, A_i_T, model, seed_node, rng, max_steps, clamp,
):
    # Same seeded dynamics and tau calculation as the original scan.
    state = np.zeros(model.n, dtype=np.int16)
    state[seed_node] = 1
    counts = np.empty(max_steps, dtype=np.int32)

    for step in range(max_steps):
        active = state == 1
        n_active = int(active.sum())
        if n_active == 0:
            duration_steps, censored = step, False
            break

        counts[step] = n_active
        inp_e = A_e_T @ active[:model.e].astype(np.float32)
        inp_i = A_i_T @ active[model.e:].astype(np.float32)
        p_net = expit(
            model.e_w * inp_e - model.i_w * inp_i - model.theta
        )
        if clamp:
            p_net[~((inp_e > 0) | (inp_i > 0))] = 0

        new_active = (state == 0) & (rng.random(model.n) < p_net)
        new_state = np.zeros_like(state)
        new_state[active] = 2
        refractory = state >= 2
        new_state[refractory] = state[refractory] + 1
        new_state[new_state > model.refractory_steps + 1] = 0
        new_state[new_active] = 1
        state = new_state

        if not np.any(state == 1):
            duration_steps, censored = step + 1, False
            break
    else:
        duration_steps, censored = max_steps, True

    counts = counts[:duration_steps].copy()
    duration_s = float(duration_steps * model.dt)
    extinct = not censored

    if counts.size >= 5 and np.var(counts) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(invalid="ignore", divide="ignore"):
                tau = fn.timescale(
                    counts[np.newaxis, :].astype(float),
                    max_lag=min(
                        3.0, max(2 * model.dt, duration_s / 2)
                    ),
                    dt=float(model.dt),
                )
    else:
        tau = np.nan

    return dict(
        size=int(counts.sum()),
        duration_steps=int(duration_steps),
        duration_s=duration_s,
        lifetime_steps=int(duration_steps),
        lifetime_s=duration_s,
        peak_active=int(counts.max()) if counts.size else 0,
        extinct=extinct,
        censored=censored,
        persistent_at_cutoff=censored,
        mean_active_during_avalanche=safe_mean(counts),
        avalanche_tau_s=float(tau),
        response_decay_tau_s=response_decay_timescale(counts, model.dt),
        **branching_metrics(counts, extinct),
    )


# ================= ONE NETWORK =================
def run_seed(seed_idx, settings):
    cfg = settings["source_config"]
    savedir = Path(settings["savedir"])
    network_seed = int(
        (
            cfg["BASE_SEED"]
            + settings["initial_ew_idx"] * 100_000_000
            + settings["phi_idx"] * 1_000_000
            + seed_idx * 10_000
        ) % (2**32)
    )

    pars = dict(cfg["start_dic"])
    pars.pop("T", None)
    pars.pop("smoothe", None)
    pars.update(
        phi=settings["phi_used"],
        e_w=settings["initial_e_w"],
        i_w=settings["i_w"],
        p_ext=float(cfg["P_EXT_RUN"]),
        seed=network_seed,
    )

    with threadpool_limits(limits=1):
        model = fn.automata_EI_hiermod(**pars)
        np.fill_diagonal(model.A, 0)

        A_e_T = sparse.csr_matrix(
            model.A[:model.e].T.astype(np.float32)
        )
        A_i_T = sparse.csr_matrix(
            model.A[model.e:].T.astype(np.float32)
        )

        walk_rng = np.random.default_rng(
            np.random.SeedSequence([
                settings["RANDOM_SEED"], seed_idx, 0
            ])
        )
        avalanche_rng = np.random.default_rng(network_seed + 123)
        ei = settings["initial_EI"]
        summaries = []
        max_steps = int(round(10.0 / model.dt))

        for step in range(settings["N_PERTURBATIONS"] + 1):
            previous_ei = ei
            proposed_delta = (
                float(walk_rng.normal(0, settings["EI_STEP_SD"]))
                if step else 0.0
            )
            proposed_ei = previous_ei + proposed_delta
            ei = float(np.clip(
                proposed_ei, settings["EI_min"], settings["EI_max"]
            ))
            model.e_w = ei * model.i * model.i_w / model.e

            identity = dict(
                seed_idx=seed_idx,
                network_seed=network_seed,
                step=step,
                phi=settings["phi_used"],
                EI=ei,
                e_w=float(model.e_w),
                i_w=float(model.i_w),
                proposed_delta_EI=proposed_delta,
                actual_delta_EI=ei - previous_ei,
                clipped=bool(ei != proposed_ei),
            )

            trials = []
            for aval_i in range(settings["N_AVALANCHES"]):
                node = int(avalanche_rng.integers(
                    0,
                    model.e if cfg["SEED_ONLY_EXCITATORY"]
                    else model.n,
                ))
                result = run_avalanche(
                    A_e_T, A_i_T, model, node,
                    avalanche_rng, max_steps,
                    cfg["CLAMP_ZERO_PRESYNAPTIC_INPUT"],
                )
                trials.append({
                    **identity,
                    "aval_i": aval_i,
                    "seed_node": node,
                    **result,
                })

            adf = pd.DataFrame(trials)
            atomic_csv(
                adf,
                savedir / f"seed_{seed_idx:03d}_step_{step:03d}.csv.gz",
                compression="gzip",
            )

            # Continuous-run FR: 10 s minus 2 s burn-in.
            # Continuous model state carries across steps.
            spikes, _ = fn.run_model(model, T=10.0)
            burn = int(round(2.0 / model.dt))
            fr = float(spikes[:, burn:].mean() / model.dt)

            extinct = adf["extinct"]
            tau = adf["avalanche_tau_s"]
            summaries.append({
                **identity,
                "mean_rate_hz": fr,
                "mean_avalanche_tau_s": safe_mean(tau),
                "mean_avalanche_tau_s_extinct_only":
                    safe_mean(tau[extinct]),
                "mean_avalanche_tau_s_censored_only":
                    safe_mean(tau[~extinct]),
                "n_valid_avalanche_tau": int(np.isfinite(tau).sum()),
                "n_extinct": int(extinct.sum()),
                "n_censored": int(adf["censored"].sum()),
                "frac_extinct": float(extinct.mean()),
                "frac_persistent_at_cutoff":
                    float(adf["censored"].mean()),
                "mean_size": float(adf["size"].mean()),
                "mean_lifetime_s": float(adf["lifetime_s"].mean()),
            })

            atomic_csv(
                pd.DataFrame(summaries),
                savedir / f"summary_seed_{seed_idx:03d}.csv",
            )

    return summaries


# ================= LOAD ORIGINAL RESULTS =================
# Starting-point criterion: same extinct-only tau used in your latest plots.
TAU_COLUMN = "mean_avalanche_tau_s_extinct_only"
# To select using extinct + censored trials instead:
# TAU_COLUMN = "mean_avalanche_tau_s"

if (
    N_SEEDS < 1 or N_WORKERS < 1 or N_PERTURBATIONS < 0
    or N_AVALANCHES < 1 or not np.isfinite(EI_STEP_SD)
    or EI_STEP_SD < 0 or not np.isfinite(PHI)
):
    raise ValueError("Invalid settings.")

cfg = json.loads((GRID_DIR / "config.json").read_text())
phis = np.asarray(cfg["PHI_VALUES"], float)
ews = np.asarray(cfg["EW_VALUES"], float)
iw = float(cfg["I_W_FIXED"])

if len(cfg["IW_VALUES"]) != 1:
    raise ValueError("Expected the original fixed-i_w grid.")

base = cfg["start_dic"]
n = int(base["n_neurons"])
n_e = int(n - n * base["ei_ratio"])
n_i = int(n * base["ei_ratio"])
ei_grid = n_e * ews / (n_i * iw)

phi_idx = int(np.argmin(abs(phis - PHI)))
phi_used = float(phis[phi_idx])

llr_rows, av_rows = [], []

for path in sorted((GRID_DIR / "completed_network_jobs").glob("*.json")):
    result = json.loads(path.read_text())

    llr_rows.extend(
        row for row in result.get("llr", [])
        if int(row["phi_idx"]) == phi_idx
    )

    av_row = result.get("avalanche", {})
    if av_row and int(av_row["phi_idx"]) == phi_idx:
        av_rows.append(av_row)

if not av_rows:
    raise ValueError("No saved avalanche results for this phi.")

av = pd.DataFrame(av_rows).replace([np.inf, -np.inf], np.nan)

if TAU_COLUMN not in av.columns:
    raise ValueError(f"Missing saved metric: {TAU_COLUMN}")

plot_df = pd.DataFrame({
    "EI": ei_grid,
    "e_w": ews,
})

# LLRs retained for display only; they do not determine selection.
plot_df["size"] = np.nan
plot_df["duration"] = np.nan

if llr_rows:
    llr = pd.DataFrame(llr_rows).replace([np.inf, -np.inf], np.nan)
    ok = (
        llr["fit_ok"].astype(str).str.lower()
        .isin(["true", "1", "1.0"])
    )

    for metric in ["size", "duration"]:
        means = (
            llr.loc[ok & (llr["metric"] == metric)]
            .groupby("ew_idx")["R_truncated_vs_lognormal"]
            .mean()
        )
        plot_df[metric] = means.reindex(plot_df.index)

plot_df["joint_min"] = np.minimum(
    plot_df["size"], plot_df["duration"]
)
plot_df["sum"] = plot_df["size"] + plot_df["duration"]

# Average the saved network-level tau estimates at each E/I.
tau_summary = av.groupby("ew_idx")[TAU_COLUMN].agg(
    mean="mean", sem="sem", n="count"
)
plot_df["tau_s"] = tau_summary["mean"].reindex(plot_df.index)
plot_df["tau_sem"] = tau_summary["sem"].reindex(plot_df.index)
plot_df["n_tau_seeds"] = tau_summary["n"].reindex(plot_df.index)

# Select the maximum mean TAU, independently of LLR fit success.
valid_tau = plot_df["tau_s"].dropna()
if valid_tau.empty:
    raise ValueError("No finite prior tau estimates for this phi.")

ew_idx = int(valid_tau.idxmax())
initial_ei = float(ei_grid[ew_idx])
initial_tau = float(valid_tau.loc[ew_idx])

print(
    f"Requested φ={PHI:g}; sampled φ={phi_used:.6g}\n"
    f"Starting criterion: maximum {TAU_COLUMN}\n"
    f"Selected E/I={initial_ei:.6g}; e_w={ews[ew_idx]:.6g}\n"
    f"Prior mean τ={initial_tau:.6g} s "
    f"from {int(plot_df.loc[ew_idx, 'n_tau_seeds'])} networks",
    flush=True,
)

# RUN_PATH must name a new folder.
SAVEDIR.mkdir(parents=True, exist_ok=False)

settings = dict(
    savedir=str(SAVEDIR),
    source_grid_dir=str(GRID_DIR),
    phi_requested=float(PHI),
    phi_used=phi_used,
    phi_idx=phi_idx,
    initial_ew_idx=ew_idx,
    initial_e_w=float(ews[ew_idx]),
    initial_EI=initial_ei,
    initial_prior_tau_s=initial_tau,
    initial_prior_tau_n_seeds=int(plot_df.loc[ew_idx, "n_tau_seeds"]),
    EI_min=float(ei_grid.min()),
    EI_max=float(ei_grid.max()),
    i_w=iw,
    N_WORKERS=N_WORKERS,
    N_SEEDS=N_SEEDS,
    N_PERTURBATIONS=N_PERTURBATIONS,
    EI_STEP_SD=EI_STEP_SD,
    N_AVALANCHES=N_AVALANCHES,
    RANDOM_SEED=RANDOM_SEED,
    objective="maximum_mean_tau_across_seeds",
    selection_tau_column=TAU_COLUMN,
    boundary_rule="clip",
    avalanche_cutoff_s=10.0,
    FR_recording_s=10.0,
    FR_burn_in_s=2.0,
    source_config=cfg,
)
(SAVEDIR / "config.json").write_text(json.dumps(settings, indent=2))
plot_df.to_csv(SAVEDIR / "starting_network_LLR_tau_slice.csv")


# ================= PLOT BEFORE RUNNING =================
fig, (ax_llr, ax_tau) = plt.subplots(
    1, 2, figsize=(11, 4), sharex=True
)
peaks = []

for metric, colour, style, ax, unit in [
    ("size", "#0072B2", "-", ax_llr, "LLR"),
    ("duration", "#D55E00", "-", ax_llr, "LLR"),
    ("joint_min", "0.4", "--", ax_llr, "LLR"),
    ("sum", "#8E44AD", "-.", ax_llr, "LLR"),
    ("tau_s", "#009E73", "-", ax_tau, "s"),
]:
    ax.plot(
        plot_df["EI"], plot_df[metric],
        marker="o", ms=3, lw=1.2,
        color=colour, ls=style, label=metric,
    )

    valid = plot_df[metric].dropna()
    if valid.empty:
        continue

    j = valid.idxmax()
    ax.scatter(
        plot_df.loc[j, "EI"], valid.loc[j],
        marker="*", s=150, color=colour,
        edgecolors="white", linewidths=0.6, zorder=5,
    )
    peaks.append({
        "metric": metric,
        "phi": phi_used,
        "EI_at_max": plot_df.loc[j, "EI"],
        "e_w_at_max": ews[int(j)],
        "maximum": valid.loc[j],
        "unit": unit,
    })

ax_tau.fill_between(
    plot_df["EI"],
    plot_df["tau_s"] - plot_df["tau_sem"],
    plot_df["tau_s"] + plot_df["tau_sem"],
    color="#009E73", alpha=0.18, linewidth=0,
    label="±1 SEM across seeds",
)

for ax in (ax_llr, ax_tau):
    ax.axvline(
        initial_ei, color="#009E73", ls=":", lw=1.5,
        label=f"Selected start E/I={initial_ei:.3f}",
    )
    ax.set_xlabel("Expected E/I input ratio")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)

ax_llr.axhline(0, color="0.7", lw=0.8)
ax_llr.set(
    title="Prior LLRs",
    ylabel="Mean seed normalised LLR / sum",
)
ax_tau.set(
    title=(
        "Prior extinct-only avalanche τ"
        if TAU_COLUMN.endswith("_extinct_only")
        else "Prior all-trial avalanche τ"
    ),
    ylabel="Mean avalanche τ (s)",
)

fig.suptitle(
    f"Requested φ={PHI:g}; sampled φ={phi_used:.3f}\n"
    f"Start at maximum τ: e_w={ews[ew_idx]:.3f}; fixed i_w={iw:g}"
)
fig.tight_layout()
fig.savefig(
    SAVEDIR / "starting_network_LLR_tau.png",
    dpi=300, bbox_inches="tight",
)
plt.show()
plt.close(fig)

peaks_df = pd.DataFrame(peaks)
peaks_df.to_csv(SAVEDIR / "starting_metric_peaks.csv", index=False)
print(peaks_df.to_string(index=False))


# ================= RUN IN PARALLEL =================
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import time

workers = min(N_WORKERS, N_SEEDS)
total_stages = N_SEEDS * (N_PERTURBATIONS + 1)

print(
    f"\nReading original results from:\n{GRID_DIR}\n\n"
    f"Saving NEW results to:\n{SAVEDIR}\n\n"
    f"Running {N_SEEDS} networks on {workers} processes\n"
    f"{total_stages} stages × {N_AVALANCHES:,} avalanches\n"
    f"Progress updates after each stage's avalanches, FR and saving finish.",
    flush=True,
)


def execute_parallel():
    # Configure joblib inside the thread that launches the workers.
    with parallel_config(backend="loky", inner_max_num_threads=1):
        return Parallel(n_jobs=workers, verbose=0)(
            delayed(run_seed)(seed_idx, settings)
            for seed_idx in range(N_SEEDS)
        )


def count_completed_stages():
    completed = 0
    finished_seeds = 0

    for seed_idx in range(N_SEEDS):
        path = SAVEDIR / f"summary_seed_{seed_idx:03d}.csv"
        if not path.exists():
            continue

        # Workers publish these files using atomic replacement.
        stages = pd.read_csv(path, usecols=["step"])["step"]
        n_done = stages[
            stages.between(0, N_PERTURBATIONS)
        ].nunique()

        completed += n_done
        finished_seeds += int(n_done == N_PERTURBATIONS + 1)

    return completed, finished_seeds


# Run simulations in the background while the notebook updates the bar.
with ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(execute_parallel)

    with tqdm(
        total=total_stages,
        desc="Completed network stages",
        unit="stage",
        dynamic_ncols=True,
    ) as progress:
        while not future.done():
            completed, finished_seeds = count_completed_stages()
            progress.update(max(0, completed - progress.n))
            progress.set_postfix(
                finished_networks=f"{finished_seeds}/{N_SEEDS}"
            )
            time.sleep(2)

        # Propagate any worker error instead of reporting false success.
        results = future.result()

        completed, finished_seeds = count_completed_stages()
        progress.update(max(0, completed - progress.n))
        progress.set_postfix(
            finished_networks=f"{finished_seeds}/{N_SEEDS}"
        )

summary = pd.DataFrame(
    [row for seed_rows in results for row in seed_rows]
).sort_values(["seed_idx", "step"])

atomic_csv(summary, SAVEDIR / "summary.csv")
print("\nDone. Saved to:", SAVEDIR)