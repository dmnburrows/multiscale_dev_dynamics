# ============================================================
# STANDALONE RANDOM e_w / i_w SEARCH FOR NARROWEST AUTOCORR PEAK
# ============================================================
#
# Runs indefinitely until manually cancelled.
#
# For each random e_w, i_w:
#   - runs seeded avalanche autocorr over 40 phi values
#   - uses 30 cores over phi x seed jobs
#   - saves candidate summary, model metrics, optional trials
#   - saves one small autocorr curve per candidate
#   - appends candidate-level result to CSV
#   - saves in batches
#
# Uses module only through:
#   fn.automata_EI_hiermod
#   fn.timescale
#   fn.sem
#
# Cancel with Ctrl+C.
# ============================================================

import os
import time
import signal
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.special import expit
from tqdm.auto import tqdm
from threadpoolctl import threadpool_limits

import icg_functions as fn


# ============================================================
# Settings
# ============================================================

SEARCH_OUTDIR = Path(
    "/home/dburrows/DATA/BLNDEV-WILDTYPE/"
    "random_search_ew_iw_narrow_autocorr_peak_40phis_30cores"
)
SEARCH_OUTDIR.mkdir(parents=True, exist_ok=True)

FIGDIR = SEARCH_OUTDIR / "autocorr_curves"
FIGDIR.mkdir(exist_ok=True)

BATCHDIR = SEARCH_OUTDIR / "batches"
BATCHDIR.mkdir(exist_ok=True)

RESULTS_PATH = SEARCH_OUTDIR / "search_results.csv"
BEST_PATH = SEARCH_OUTDIR / "search_results_sorted_best.csv"

N_WORKERS = 30
BATCH_SIZE = 10

N_SEEDS = 10
N_AVALANCHES_PER_MODEL = 100
MAX_STEPS = 2500
BASE_SEED = 91919

PHI_VALUES = np.linspace(1.5, 5.0, 40)

E_W_RANGE = (0.5, 100.0)
I_W_RANGE = (0.0, 60.0)
LOG_SAMPLE_E_W = True

WIDTH_FRAC = 0.5

SAVE_TRIALS = False
SAVE_MODEL_METRICS = True
SAVE_SUMMARY = True
SAVE_PLOTS = True

start_dic_base = {
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

AVALANCHE_P_EXT = 0.0
AVALANCHE_THETA = float(start_dic_base["theta"])
SEED_ONLY_EXCITATORY = True
CLAMP_ZERO_PRESYNAPTIC_INPUT = True


# ============================================================
# Generic helpers
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


def safe_sem(x):
    x = safe_values(x)
    return float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else np.nan


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


def clean_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)


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

    sigma_origin = (
        float(np.sum(x * y) / denom)
        if denom > 0
        else np.nan
    )

    X = np.column_stack([np.ones_like(x), x])

    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)

        intercept = float(beta[0])
        slope = float(beta[1])

        y_hat = X @ beta
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))

        r2 = (
            float(1.0 - ss_res / ss_tot)
            if ss_tot > 0
            else np.nan
        )

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
        "mean_e_out_degree_all": float(A_e_all.sum(axis=1).mean()),
        "mean_e_out_degree_e": float(A_ee.sum(axis=1).mean()),
        "edge_density": float(
            A.sum() / max(A.shape[0] * (A.shape[0] - 1), 1)
        ),
    }


# ============================================================
# Search helpers
# ============================================================

def sample_ew_iw(rng):
    if LOG_SAMPLE_E_W:
        lo, hi = np.log10(E_W_RANGE[0]), np.log10(E_W_RANGE[1])
        e_w = 10 ** rng.uniform(lo, hi)
    else:
        e_w = rng.uniform(*E_W_RANGE)

    i_w = rng.uniform(*I_W_RANGE)

    return float(e_w), float(i_w)


def peak_width(phi, y, frac=0.5):
    phi = np.asarray(phi, dtype=float)
    y = np.asarray(y, dtype=float)

    keep = np.isfinite(phi) & np.isfinite(y)
    phi = phi[keep]
    y = y[keep]

    if y.size < 3:
        return {
            "peak_phi": np.nan,
            "peak_tau": np.nan,
            "width": np.nan,
            "n_above": 0,
            "threshold": np.nan,
        }

    peak_idx = int(np.nanargmax(y))
    peak_tau = float(y[peak_idx])
    peak_phi = float(phi[peak_idx])

    if not np.isfinite(peak_tau) or peak_tau <= 0:
        return {
            "peak_phi": peak_phi,
            "peak_tau": peak_tau,
            "width": np.nan,
            "n_above": 0,
            "threshold": np.nan,
        }

    threshold = frac * peak_tau
    above = y >= threshold

    if not np.any(above):
        width = np.nan
        n_above = 0
    else:
        width = float(np.max(phi[above]) - np.min(phi[above]))
        n_above = int(np.sum(above))

    return {
        "peak_phi": peak_phi,
        "peak_tau": peak_tau,
        "width": width,
        "n_above": n_above,
        "threshold": threshold,
    }


def summarize_candidate(avalanche_trials):
    model_rows = []

    for (phi, seed), g in avalanche_trials.groupby(["phi", "seed"], sort=True):
        row = {
            "phi": float(phi),
            "seed": int(seed),
            "n_trials": int(len(g)),
            "frac_extinct": float(g["extinct"].mean()),
            "frac_persistent_at_cutoff": float(g["persistent_at_cutoff"].mean()),

            "mean_avalanche_tau_s": safe_mean(g["avalanche_tau_s"]),
            "median_avalanche_tau_s": safe_percentile(g["avalanche_tau_s"], 50),
            "p90_avalanche_tau_s": safe_percentile(g["avalanche_tau_s"], 90),
            "p95_avalanche_tau_s": safe_percentile(g["avalanche_tau_s"], 95),
            "p99_avalanche_tau_s": safe_percentile(g["avalanche_tau_s"], 99),

            "mean_size": safe_mean(g["size"]),
            "p95_size": safe_percentile(g["size"], 95),
            "mean_lifetime_s": safe_mean(g["lifetime_s"]),
            "p95_lifetime_s": safe_percentile(g["lifetime_s"], 95),
        }

        model_rows.append(row)

    model_metrics = (
        pd.DataFrame(model_rows)
        .sort_values(["phi", "seed"])
        .reset_index(drop=True)
    )

    summary_rows = []

    for phi, g in model_metrics.groupby("phi", sort=True):
        row = {
            "phi": float(phi),
            "n_models": int(len(g)),
            "n_trials": int(g["n_trials"].sum()),
        }

        for metric in [
            "frac_extinct",
            "frac_persistent_at_cutoff",
            "mean_avalanche_tau_s",
            "median_avalanche_tau_s",
            "p90_avalanche_tau_s",
            "p95_avalanche_tau_s",
            "p99_avalanche_tau_s",
            "mean_size",
            "p95_size",
            "mean_lifetime_s",
            "p95_lifetime_s",
        ]:
            vals = g[metric].replace([np.inf, -np.inf], np.nan)
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_sem"] = fn.sem(vals)

        summary_rows.append(row)

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values("phi")
        .reset_index(drop=True)
    )

    return model_metrics, summary


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
# One phi x seed network job
# ============================================================

def run_one_candidate_network_job(job):
    phi = float(job["phi"])
    seed = int(job["seed"])
    e_w = float(job["e_w"])
    i_w = float(job["i_w"])
    candidate_i = int(job["candidate_i"])

    pars = dict(start_dic_base)
    pars["phi"] = phi
    pars["e_w"] = e_w
    pars["i_w"] = i_w
    pars["p_ext"] = AVALANCHE_P_EXT
    pars["theta"] = AVALANCHE_THETA

    model_kwargs = {
        key: value
        for key, value in pars.items()
        if key not in {"T", "smoothe"}
    }

    network_seed = int(
        BASE_SEED
        + candidate_i * 10_000_000
        + int(round(phi * 1000)) * 10_000
        + seed * 1000
    )

    model_kwargs["seed"] = network_seed

    with threadpool_limits(limits=1):
        model = fn.automata_EI_hiermod(**model_kwargs)

    A = np.asarray(model.A, dtype=np.uint8)
    np.fill_diagonal(A, 0)

    n = int(model.n)
    n_e = int(model.e)

    A_e_T = sparse.csr_matrix(A[:n_e, :].T.astype(np.float32))
    A_i_T = sparse.csr_matrix(A[n_e:, :].T.astype(np.float32))

    connectivity = local_connectivity_metrics(A=A, n_e=n_e)

    rng = np.random.default_rng(network_seed + 123)

    rows = []

    for aval_i in range(N_AVALANCHES_PER_MODEL):
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
            "candidate_i": candidate_i,
            "phi": phi,
            "seed": seed,
            "network_seed": network_seed,
            "aval_i": int(aval_i),
            "seed_node": int(seed_node),
            "seed_cell_type": seed_cell_type,

            "n_neurons": n,
            "n_e": n_e,
            "dt": float(model.dt),
            "max_steps": int(MAX_STEPS),
            "cutoff_s": float(MAX_STEPS * model.dt),

            "p_ext": float(model.p_ext),
            "theta": float(model.theta),
            "e_w": float(model.e_w),
            "i_w": float(model.i_w),
            "refractory_steps": int(model.refractory_steps),

            "clamp_zero_presynaptic_input": bool(
                CLAMP_ZERO_PRESYNAPTIC_INPUT
            ),
        }

        row.update(connectivity)
        row.update(out)
        rows.append(row)

    return rows


# ============================================================
# Plot one candidate
# ============================================================

def plot_candidate(summary, candidate_i, e_w, i_w, width_info):
    x = summary["phi"].to_numpy(float)
    y = summary["mean_avalanche_tau_s_mean"].to_numpy(float)
    e = summary["mean_avalanche_tau_s_sem"].fillna(0).to_numpy(float)

    fig, ax = plt.subplots(figsize=(4.2, 3.0))

    ax.plot(
        x,
        y,
        marker="o",
        linewidth=1.6,
        markersize=3,
    )

    ax.fill_between(
        x,
        y - e,
        y + e,
        alpha=0.18,
        linewidth=0,
    )

    if np.isfinite(width_info["peak_phi"]):
        ax.axvline(
            width_info["peak_phi"],
            linestyle="--",
            linewidth=1.0,
            color="0.3",
        )

    if np.isfinite(width_info["threshold"]):
        ax.axhline(
            width_info["threshold"],
            linestyle=":",
            linewidth=1.0,
            color="0.3",
        )

    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel("Mean avalanche autocorr tau (s)")

    ax.set_title(
        rf"iter {candidate_i}: $e_w={e_w:.3g}$, $i_w={i_w:.3g}$"
        "\n"
        rf"width={width_info['width']:.3g}, peak $\phi$={width_info['peak_phi']:.3g}",
        fontsize=9,
    )

    clean_ax(ax)
    fig.tight_layout()

    path = FIGDIR / (
        f"iter_{candidate_i:06d}_"
        f"ew_{e_w:.4g}_iw_{i_w:.4g}_autocorr.png"
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return path


# ============================================================
# I/O helpers
# ============================================================

def get_start_iter():
    if not RESULTS_PATH.exists():
        return 0

    previous = pd.read_csv(RESULTS_PATH)

    if len(previous) == 0:
        return 0

    return int(previous["candidate_i"].max()) + 1


def append_results(rows):
    rows_df = pd.DataFrame(rows)

    if RESULTS_PATH.exists():
        old = pd.read_csv(RESULTS_PATH)
        out = pd.concat([old, rows_df], ignore_index=True)
        out = out.drop_duplicates(subset=["candidate_i"], keep="last")
    else:
        out = rows_df

    out = out.sort_values("candidate_i").reset_index(drop=True)
    out.to_csv(RESULTS_PATH, index=False)

    best = (
        out.sort_values(
            ["peak_width_phi", "peak_tau_s"],
            ascending=[True, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    best.to_csv(BEST_PATH, index=False)

    return out, best


def save_candidate_outputs(
    candidate_i,
    avalanche_trials,
    model_metrics,
    summary,
):
    candidate_dir = SEARCH_OUTDIR / f"candidate_{candidate_i:06d}"
    candidate_dir.mkdir(exist_ok=True)

    if SAVE_TRIALS:
        avalanche_trials.to_csv(
            candidate_dir / "avalanche_trials_fast_sparse.csv",
            index=False,
        )

    if SAVE_MODEL_METRICS:
        model_metrics.to_csv(
            candidate_dir / "avalanche_model_metrics_fast_sparse.csv",
            index=False,
        )

    if SAVE_SUMMARY:
        summary.to_csv(
            candidate_dir / "avalanche_summary_by_phi_fast_sparse.csv",
            index=False,
        )

    return candidate_dir


# ============================================================
# Run one candidate
# ============================================================

def run_candidate(candidate_i, e_w, i_w, ctx):
    jobs = [
        {
            "candidate_i": candidate_i,
            "e_w": e_w,
            "i_w": i_w,
            "phi": float(phi),
            "seed": int(seed),
        }
        for phi in PHI_VALUES
        for seed in range(N_SEEDS)
    ]

    trial_rows = []

    with ctx.Pool(processes=N_WORKERS) as pool:
        for rows_local in tqdm(
            pool.imap_unordered(run_one_candidate_network_job, jobs),
            total=len(jobs),
            desc=f"candidate {candidate_i}",
        ):
            trial_rows.extend(rows_local)

    avalanche_trials = (
        pd.DataFrame(trial_rows)
        .sort_values(["phi", "seed", "aval_i"])
        .reset_index(drop=True)
    )

    model_metrics, summary = summarize_candidate(avalanche_trials)

    width_info = peak_width(
        summary["phi"],
        summary["mean_avalanche_tau_s_mean"],
        frac=WIDTH_FRAC,
    )

    plot_path = None

    if SAVE_PLOTS:
        plot_path = plot_candidate(
            summary=summary,
            candidate_i=candidate_i,
            e_w=e_w,
            i_w=i_w,
            width_info=width_info,
        )

    candidate_dir = save_candidate_outputs(
        candidate_i=candidate_i,
        avalanche_trials=avalanche_trials,
        model_metrics=model_metrics,
        summary=summary,
    )

    result = {
        "candidate_i": int(candidate_i),
        "e_w": float(e_w),
        "i_w": float(i_w),
        "width_frac": float(WIDTH_FRAC),
        "peak_width_phi": float(width_info["width"]),
        "peak_phi": float(width_info["peak_phi"]),
        "peak_tau_s": float(width_info["peak_tau"]),
        "n_phi_above_threshold": int(width_info["n_above"]),
        "mean_persistent_fraction": float(
            summary["frac_persistent_at_cutoff_mean"].mean()
        ),
        "max_persistent_fraction": float(
            summary["frac_persistent_at_cutoff_mean"].max()
        ),
        "mean_tau_across_phi": float(
            summary["mean_avalanche_tau_s_mean"].mean()
        ),
        "max_tau_across_phi": float(
            summary["mean_avalanche_tau_s_mean"].max()
        ),
        "plot_path": str(plot_path) if plot_path is not None else "",
        "candidate_dir": str(candidate_dir),
    }

    return result


# ============================================================
# Main loop
# ============================================================

def main():
    print("=" * 80)
    print("RANDOM e_w / i_w SEARCH FOR NARROWEST AUTOCORR PEAK")
    print("=" * 80)
    print("SEARCH_OUTDIR:", SEARCH_OUTDIR)
    print("N_WORKERS:", N_WORKERS)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("N_SEEDS:", N_SEEDS)
    print("N_AVALANCHES_PER_MODEL:", N_AVALANCHES_PER_MODEL)
    print("MAX_STEPS:", MAX_STEPS)
    print("PHI_VALUES:", PHI_VALUES)
    print("E_W_RANGE:", E_W_RANGE)
    print("I_W_RANGE:", I_W_RANGE)
    print("WIDTH_FRAC:", WIDTH_FRAC)
    print("SAVE_TRIALS:", SAVE_TRIALS)

    rng_search = np.random.default_rng(12345)

    candidate_i = get_start_iter()

    # Advance RNG so resumed searches do not repeat previous parameter draws.
    for _ in range(candidate_i):
        sample_ew_iw(rng_search)

    ctx = mp.get_context("fork")

    batch_i = candidate_i // BATCH_SIZE
    batch_rows = []

    print("\nStarting at candidate:", candidate_i)

    try:
        while True:
            e_w, i_w = sample_ew_iw(rng_search)

            print("\n" + "=" * 80)
            print(f"Candidate {candidate_i}")
            print(f"e_w = {e_w:.8g}")
            print(f"i_w = {i_w:.8g}")

            t0 = time.time()

            result = run_candidate(
                candidate_i=candidate_i,
                e_w=e_w,
                i_w=i_w,
                ctx=ctx,
            )

            elapsed_min = (time.time() - t0) / 60.0
            result["elapsed_min"] = float(elapsed_min)

            batch_rows.append(result)

            print(
                f"Done candidate {candidate_i} "
                f"in {elapsed_min:.2f} min | "
                f"width={result['peak_width_phi']:.4g}, "
                f"peak_phi={result['peak_phi']:.4g}, "
                f"peak_tau={result['peak_tau_s']:.4g}"
            )

            candidate_i += 1

            if len(batch_rows) >= BATCH_SIZE:
                batch_path = BATCHDIR / f"batch_{batch_i:06d}.csv"
                pd.DataFrame(batch_rows).to_csv(batch_path, index=False)

                _, best = append_results(batch_rows)

                print("\nSaved batch:", batch_path)
                print("Saved aggregate:", RESULTS_PATH)
                print("Saved sorted best:", BEST_PATH)
                print("\nCurrent best:")
                print(best.head(10).to_string(index=False))

                batch_rows = []
                batch_i += 1

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Saving partial batch...")

        if batch_rows:
            batch_path = BATCHDIR / f"batch_{batch_i:06d}_partial.csv"
            pd.DataFrame(batch_rows).to_csv(batch_path, index=False)

            _, best = append_results(batch_rows)

            print("Saved partial batch:", batch_path)
            print("Saved aggregate:", RESULTS_PATH)
            print("Saved sorted best:", BEST_PATH)
            print("\nCurrent best:")
            print(best.head(10).to_string(index=False))
        else:
            print("No unsaved batch rows.")

        print("Stopped cleanly.")


if __name__ == "__main__":
    main()