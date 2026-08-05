# ============================================================
# MODULAR DEGREE-MATCHED CONTROL PHI SWEEP
#
# Purpose:
#   For each phi:
#     1. Build a hiermod reference network using phi
#     2. Count its directed edges
#     3. Build a FLAT MODULAR directed network with exactly the same
#        total number of directed edges, no self-loops
#     4. Run the same CA dynamics
#     5. Compute ICG static-dynamic scaling
#
# Control:
#   - Same n, E/I split, e_w, i_w, theta, p_ext, refractory, dt
#   - Same T, smoothing, ICG pipeline
#   - Same edge count as hiermod network at that phi/seed
#   - Has flat modules / communities
#   - But has NO nested hierarchical modular organisation
#
# Saves:
#   modular_degree_matched_phi_design.csv
#   modular_degree_matched_phi_seed_exponents.csv
#   modular_degree_matched_phi_icg_rows.csv
#   modular_degree_matched_phi_summary.csv
#   plots/
# ============================================================

import os
import inspect
import warnings
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

OUTDIR = Path(
    "/home/dburrows/DATA/BLNDEV-WILDTYPE/"
    "modular_degree_matched_newpars_ew11p7_iw22p4_pext0_phi1p5to10_15steps_20seeds_smooth005"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

FIGDIR = OUTDIR / "phi_plots"
FIGDIR.mkdir(parents=True, exist_ok=True)

N_WORKERS = 20
N_SEEDS = 20
N_PHI = 15
BASE_SEED = 888

PHI_MIN = 1.5
PHI_MAX = 10.0
PHI_VALUES = np.linspace(PHI_MIN, PHI_MAX, N_PHI)

SMOOTHE = float(start_dic["smoothe"])
T_RUN = float(start_dic["T"])

TARGET_MV = 1.50
TARGET_TAU = 0.20

# ------------------------------------------------------------
# Flat modular control parameters
# ------------------------------------------------------------
# N_MODULES controls the number of flat modules.
# WITHIN_EDGE_FRACTION controls how many edges are placed inside modules.
#
# Example:
#   0.70 means 70% of all matched edges are within-module edges,
#   30% are between-module edges.
#
# This gives modularity without hierarchy.
# ------------------------------------------------------------

N_MODULES = 32
WITHIN_EDGE_FRACTION = 0.70

print("OUTDIR:", OUTDIR)
print("N_PHI:", N_PHI)
print("PHI_VALUES:", PHI_VALUES)
print("N_SEEDS:", N_SEEDS)
print("N jobs:", N_PHI * N_SEEDS)
print("N_MODULES:", N_MODULES)
print("WITHIN_EDGE_FRACTION:", WITHIN_EDGE_FRACTION)
print("start_dic:", start_dic)


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


def safe_loglog_slope(
    x,
    y,
    exclude_first=True,
    exclude_last=False,
    min_points=3,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[ok]
    y = y[ok]

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if exclude_first and len(x) > 0:
        x = x[1:]
        y = y[1:]

    if exclude_last and len(x) > 0:
        x = x[:-1]
        y = y[:-1]

    if len(x) < min_points:
        return np.nan, np.nan, np.nan

    logx = np.log10(x)
    logy = np.log10(y)

    slope, intercept = np.polyfit(logx, logy, 1)
    pred = intercept + slope * logx

    ss_res = float(np.sum((logy - pred) ** 2))
    ss_tot = float(np.sum((logy - np.mean(logy)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return float(slope), float(r2), np.nan


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


def make_hiermod_model_kwargs(phi, seed):
    pars = dict(start_dic)

    pars["phi"] = float(phi)

    pars.pop("T", None)
    pars.pop("smoothe", None)

    pars["seed"] = int(seed)

    sig = inspect.signature(fn.automata_EI_hiermod.__init__)
    valid_args = set(sig.parameters.keys())

    if "phi" not in valid_args and "slope" in valid_args:
        pars["slope"] = pars.pop("phi")

    return {k: v for k, v in pars.items() if k in valid_args}


# ============================================================
# Flat modular degree-matched graph generator
# ============================================================

def make_flat_module_labels(n, n_modules, rng):
    """
    Randomly assigns neurons to approximately equal-sized flat modules.

    This is not hierarchical. It is a one-level partition:
        network -> modules
    """
    n = int(n)
    n_modules = int(n_modules)

    if n_modules < 2:
        raise ValueError("n_modules must be >= 2")

    if n_modules > n:
        raise ValueError("n_modules cannot exceed n")

    perm = rng.permutation(n)

    module_sizes = np.full(n_modules, n // n_modules, dtype=int)
    module_sizes[: n % n_modules] += 1

    module_id = np.empty(n, dtype=np.int32)

    start = 0
    for m, size in enumerate(module_sizes):
        nodes = perm[start:start + size]
        module_id[nodes] = m
        start += size

    return module_id, module_sizes


def edge_code_from_src_tgt(src, tgt, n):
    """
    Encode directed no-self-loop edge (src, tgt) as integer in [0, n*(n-1)).
    """
    src = np.asarray(src, dtype=np.int64)
    tgt = np.asarray(tgt, dtype=np.int64)

    rem = tgt - (tgt > src)
    return src * (n - 1) + rem


def src_tgt_from_edge_code(code, n):
    """
    Decode integer edge code into directed no-self-loop edge (src, tgt).
    """
    code = np.asarray(code, dtype=np.int64)

    src = code // (n - 1)
    rem = code % (n - 1)

    tgt = rem + (rem >= src)

    return src.astype(np.int64), tgt.astype(np.int64)


def sample_unique_edges_by_module_relation(
    n,
    module_id,
    n_edges,
    same_module,
    rng,
    batch_factor=4,
):
    """
    Samples unique directed edges either within modules or between modules.

    Parameters
    ----------
    same_module:
        True  -> sample src/tgt pairs with module_id[src] == module_id[tgt]
        False -> sample src/tgt pairs with module_id[src] != module_id[tgt]

    Notes
    -----
    Uses rejection sampling and a Python set of encoded edges.
    This avoids building the full n x n candidate list.
    """
    n = int(n)
    n_edges = int(n_edges)

    if n_edges <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    chosen = set()

    # Conservative batch size; grows with remaining need
    while len(chosen) < n_edges:
        remaining = n_edges - len(chosen)
        batch = max(10_000, int(batch_factor * remaining))

        src = rng.integers(0, n, size=batch, dtype=np.int64)
        tgt = rng.integers(0, n, size=batch, dtype=np.int64)

        ok = src != tgt

        if same_module:
            ok &= module_id[src] == module_id[tgt]
        else:
            ok &= module_id[src] != module_id[tgt]

        src = src[ok]
        tgt = tgt[ok]

        if src.size == 0:
            continue

        codes = edge_code_from_src_tgt(src, tgt, n)

        for c in codes:
            chosen.add(int(c))
            if len(chosen) >= n_edges:
                break

    codes = np.fromiter(chosen, dtype=np.int64, count=n_edges)
    src, tgt = src_tgt_from_edge_code(codes, n)

    return src, tgt


def make_modular_directed_exact_edges(
    n,
    n_edges,
    n_modules,
    within_edge_fraction,
    rng,
):
    """
    Directed flat modular random graph with exactly n_edges and no self-loops.

    Matches:
      - total directed edge count
      - mean degree / edge density

    Adds:
      - flat modular community structure

    Does NOT preserve:
      - hierarchical organisation
      - exact in/out degree sequence
      - Munn-style nested module wiring
    """
    n = int(n)
    n_edges = int(n_edges)

    max_edges = n * (n - 1)

    if n_edges > max_edges:
        raise ValueError(f"Requested {n_edges} edges, but max is {max_edges}")

    module_id, module_sizes = make_flat_module_labels(
        n=n,
        n_modules=n_modules,
        rng=rng,
    )

    max_within_edges = int(np.sum(module_sizes * (module_sizes - 1)))
    max_between_edges = int(max_edges - max_within_edges)

    n_within_target = int(round(n_edges * float(within_edge_fraction)))
    n_within = min(n_within_target, max_within_edges)

    n_between = n_edges - n_within

    if n_between > max_between_edges:
        n_between = max_between_edges
        n_within = n_edges - n_between

    if n_within < 0 or n_between < 0:
        raise ValueError("Invalid within/between edge split.")

    src_within, tgt_within = sample_unique_edges_by_module_relation(
        n=n,
        module_id=module_id,
        n_edges=n_within,
        same_module=True,
        rng=rng,
    )

    src_between, tgt_between = sample_unique_edges_by_module_relation(
        n=n,
        module_id=module_id,
        n_edges=n_between,
        same_module=False,
        rng=rng,
    )

    src = np.concatenate([src_within, src_between])
    tgt = np.concatenate([tgt_within, tgt_between])

    A = np.zeros((n, n), dtype=np.uint8)
    A[src, tgt] = 1
    np.fill_diagonal(A, 0)

    # Guard against accidental duplicate loss
    actual_edges = int(A.sum())

    if actual_edges != n_edges:
        raise RuntimeError(
            f"Expected {n_edges} edges, got {actual_edges}. "
            "Duplicate edge handling failed."
        )

    return A, module_id, {
        "n_modules": int(n_modules),
        "within_edge_fraction_target": float(within_edge_fraction),
        "n_within_edges": int(n_within),
        "n_between_edges": int(n_between),
        "within_edge_fraction_actual": float(n_within / max(n_edges, 1)),
        "max_within_edges": int(max_within_edges),
        "max_between_edges": int(max_between_edges),
        "module_size_min": int(module_sizes.min()),
        "module_size_max": int(module_sizes.max()),
        "module_size_mean": float(module_sizes.mean()),
    }


# ============================================================
# Modular CA model
# ============================================================

class automata_EI_modular_degree_matched:
    """
    Same CA dynamics as automata_EI_hiermod, but with a supplied flat modular A.
    """

    def __init__(
        self,
        A,
        n_neurons=2000,
        ei_ratio=0.2,
        e_w=11.7,
        i_w=22.4,
        theta=8.44,
        phi=np.nan,
        p_ext=0.0,
        refractory_steps=2,
        dt=0.01,
        seed=0,
    ):
        self.rng = np.random.default_rng(seed)

        self.A = np.asarray(A, dtype=np.uint8)
        np.fill_diagonal(self.A, 0)

        self.n = int(n_neurons)
        self.e = int(n_neurons - (n_neurons * ei_ratio))
        self.i = int(n_neurons * ei_ratio)

        if self.A.shape != (self.n, self.n):
            raise ValueError(f"A has shape {self.A.shape}, expected {(self.n, self.n)}")

        self.e_w = float(e_w)
        self.i_w = float(i_w)
        self.theta = float(theta)
        self.phi = float(phi)
        self.p_ext = float(p_ext)
        self.refractory_steps = int(refractory_steps)
        self.dt = float(dt)

        self.A_e = self.A[:self.e, :]
        self.A_i = self.A[self.e:, :]

        self.state = np.zeros(self.n, dtype=np.int16)

    def step(self):
        active = self.state == 1

        inp_e = self.A_e.T @ active[:self.e].astype(np.float32)
        inp_i = self.A_i.T @ active[self.e:].astype(np.float32)

        net = (self.e_w * inp_e) - (self.i_w * inp_i)
        p_net = 1.0 / (1.0 + np.exp(-(net - self.theta)))

        quiescent = self.state == 0

        ext_events = self.rng.random(self.n) < self.p_ext
        net_events = self.rng.random(self.n) < p_net

        new_active = quiescent & (ext_events | net_events)

        new_state = np.zeros_like(self.state)

        new_state[active] = 2

        refractory = self.state >= 2
        new_state[refractory] = self.state[refractory] + 1

        done_refrac = new_state > (self.refractory_steps + 1)
        new_state[done_refrac] = 0

        new_state[new_active] = 1

        self.state = new_state

        return (new_state == 1).astype(np.uint8)


def run_model_local(model, T=10.0):
    n_steps = int(T / model.dt)

    spikes = np.zeros((model.n, n_steps), dtype=np.uint8)
    pop_rate = np.zeros(n_steps, dtype=float)

    for t in range(n_steps):
        active = model.step()
        spikes[:, t] = active
        pop_rate[t] = active.mean() / model.dt

    return spikes, pop_rate


def compute_icg_and_exponents(spikes, dt, smoothe):
    spikes_icg = fn.exp_smooth_spikes(
        spikes,
        dt=float(dt),
        tau=float(smoothe),
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
        )

        with np.errstate(invalid="ignore", divide="ignore"):
            metric_row, gen_df = fn.compute_icg_metrics(
                spikes=spikes_icg,
                dt=float(dt),
            )

    metric_row = dict(metric_row)
    gen_df = standardise_gen_df(gen_df)

    if "mean_cluster_size" not in gen_df.columns:
        raise ValueError(f"No mean_cluster_size column. Columns: {gen_df.columns.tolist()}")

    if "mean_variance_norm" not in gen_df.columns:
        raise ValueError(f"No mean_variance_norm column. Columns: {gen_df.columns.tolist()}")

    if "timescale_norm" not in gen_df.columns:
        raise ValueError(f"No timescale_norm column. Columns: {gen_df.columns.tolist()}")

    MV_alpha, MV_r2, MV_p = safe_loglog_slope(
        gen_df["mean_cluster_size"],
        gen_df["mean_variance_norm"],
        exclude_first=True,
        exclude_last=False,
    )

    TAU_beta, TAU_r2, TAU_p = safe_loglog_slope(
        gen_df["mean_cluster_size"],
        gen_df["timescale_norm"],
        exclude_first=True,
        exclude_last=False,
    )

    if "corr_kurtosis" in gen_df.columns:
        KURT_slope, KURT_r2, KURT_p = safe_loglog_slope(
            gen_df["mean_cluster_size"],
            gen_df["corr_kurtosis"],
            exclude_first=True,
            exclude_last=False,
        )
    elif "kurtosis_corr" in gen_df.columns:
        KURT_slope, KURT_r2, KURT_p = safe_loglog_slope(
            gen_df["mean_cluster_size"],
            gen_df["kurtosis_corr"],
            exclude_first=True,
            exclude_last=False,
        )
    else:
        KURT_slope, KURT_r2, KURT_p = np.nan, np.nan, np.nan

    score = (
        abs(MV_alpha - TARGET_MV) + 2.0 * abs(TAU_beta - TARGET_TAU)
        if np.isfinite(MV_alpha) and np.isfinite(TAU_beta)
        else np.nan
    )

    metric_row["MV_alpha"] = MV_alpha
    metric_row["MV_r2"] = MV_r2
    metric_row["MV_p"] = MV_p

    metric_row["TAU_beta"] = TAU_beta
    metric_row["TAU_r2"] = TAU_r2
    metric_row["TAU_p"] = TAU_p

    metric_row["KURT_slope"] = KURT_slope
    metric_row["KURT_r2"] = KURT_r2
    metric_row["KURT_p"] = KURT_p

    metric_row["score"] = score

    # old aliases
    metric_row["mv_alpha_norm"] = MV_alpha
    metric_row["mv_r2_norm"] = MV_r2
    metric_row["ts_beta_norm"] = TAU_beta
    metric_row["ts_r2_norm"] = TAU_r2

    return metric_row, gen_df


# ============================================================
# One job
# ============================================================

def run_one_job(job):
    phi_idx = int(job["phi_idx"])
    seed_idx = int(job["seed_idx"])
    phi = float(job["phi"])
    seed = int(job["seed"])

    # separate seeds for reference topology and modular topology/dynamics
    hier_seed = int(seed)
    modular_seed = int(seed + 20_000_000)

    # --------------------------------------------------------
    # 1. Build hiermod reference network for this phi/seed
    # --------------------------------------------------------
    hier_kwargs = make_hiermod_model_kwargs(
        phi=phi,
        seed=hier_seed,
    )

    with threadpool_limits(limits=1):
        hier_model = fn.automata_EI_hiermod(**hier_kwargs)

    A_ref = np.asarray(hier_model.A, dtype=np.uint8)
    np.fill_diagonal(A_ref, 0)

    n = int(hier_model.n)
    n_e = int(hier_model.e)
    n_i = int(n - n_e)

    n_edges_ref = int(A_ref.sum())
    edge_density_ref = float(n_edges_ref / (n * (n - 1)))
    mean_out_degree_ref = float(A_ref.sum(axis=1).mean())
    mean_in_degree_ref = float(A_ref.sum(axis=0).mean())

    # --------------------------------------------------------
    # 2. Build flat modular network with exactly same edge count
    # --------------------------------------------------------
    rng_modular = np.random.default_rng(modular_seed)

    A_modular, module_id, modular_info = make_modular_directed_exact_edges(
        n=n,
        n_edges=n_edges_ref,
        n_modules=N_MODULES,
        within_edge_fraction=WITHIN_EDGE_FRACTION,
        rng=rng_modular,
    )

    n_edges_modular = int(A_modular.sum())
    edge_density_modular = float(n_edges_modular / (n * (n - 1)))
    mean_out_degree_modular = float(A_modular.sum(axis=1).mean())
    mean_in_degree_modular = float(A_modular.sum(axis=0).mean())

    # --------------------------------------------------------
    # 3. Run modular dynamics
    # --------------------------------------------------------
    modular_model = automata_EI_modular_degree_matched(
        A=A_modular,
        n_neurons=int(start_dic["n_neurons"]),
        ei_ratio=float(start_dic["ei_ratio"]),
        e_w=float(start_dic["e_w"]),
        i_w=float(start_dic["i_w"]),
        theta=float(start_dic["theta"]),
        phi=phi,
        p_ext=float(start_dic["p_ext"]),
        refractory_steps=int(start_dic["refractory_steps"]),
        dt=float(start_dic["dt"]),
        seed=modular_seed,
    )

    spikes, pop_rate = run_model_local(
        modular_model,
        T=T_RUN,
    )

    dt = float(start_dic["dt"])

    mean_rate_hz = float(spikes.mean() / dt)
    pop_rate_mean_hz = float(np.mean(pop_rate))
    pop_rate_std_hz = float(np.std(pop_rate))

    active_counts = spikes.sum(axis=0)
    frac_silent_frames = float(np.mean(active_counts == 0))
    frac_active_neurons = float(np.mean(spikes.sum(axis=1) > 0))

    # --------------------------------------------------------
    # 4. ICG metrics
    # --------------------------------------------------------
    metric_row, gen_df = compute_icg_and_exponents(
        spikes=spikes,
        dt=dt,
        smoothe=SMOOTHE,
    )

    out = {
        "condition": "modular_degree_matched",
        "phi_idx": phi_idx,
        "seed_idx": seed_idx,
        "phi": phi,
        "seed": seed,
        "hier_seed": hier_seed,
        "modular_seed": modular_seed,

        "n_neurons": n,
        "n_e": n_e,
        "n_i": n_i,
        "dt": dt,
        "T": T_RUN,

        "theta": float(start_dic["theta"]),
        "ei_ratio": float(start_dic["ei_ratio"]),
        "e_w": float(start_dic["e_w"]),
        "i_w": float(start_dic["i_w"]),
        "p_ext": float(start_dic["p_ext"]),
        "refractory_steps": int(start_dic["refractory_steps"]),
        "smoothe": SMOOTHE,

        "n_edges_ref_hiermod": n_edges_ref,
        "edge_density_ref_hiermod": edge_density_ref,
        "mean_out_degree_ref_hiermod": mean_out_degree_ref,
        "mean_in_degree_ref_hiermod": mean_in_degree_ref,

        "n_edges_modular": n_edges_modular,
        "edge_density_modular": edge_density_modular,
        "mean_out_degree_modular": mean_out_degree_modular,
        "mean_in_degree_modular": mean_in_degree_modular,

        "mean_rate_hz": mean_rate_hz,
        "pop_rate_mean_hz": pop_rate_mean_hz,
        "pop_rate_std_hz": pop_rate_std_hz,
        "frac_silent_frames": frac_silent_frames,
        "frac_active_neurons": frac_active_neurons,
    }

    out.update(modular_info)
    out.update(metric_row)

    gen_df = gen_df.copy()
    gen_df["condition"] = "modular_degree_matched"
    gen_df["phi_idx"] = phi_idx
    gen_df["seed_idx"] = seed_idx
    gen_df["phi"] = phi
    gen_df["seed"] = seed
    gen_df["hier_seed"] = hier_seed
    gen_df["modular_seed"] = modular_seed

    gen_df["n_neurons"] = n
    gen_df["p_ext"] = float(start_dic["p_ext"])
    gen_df["e_w"] = float(start_dic["e_w"])
    gen_df["i_w"] = float(start_dic["i_w"])
    gen_df["theta"] = float(start_dic["theta"])
    gen_df["smoothe"] = SMOOTHE

    gen_df["n_edges_ref_hiermod"] = n_edges_ref
    gen_df["edge_density_ref_hiermod"] = edge_density_ref
    gen_df["n_edges_modular"] = n_edges_modular
    gen_df["edge_density_modular"] = edge_density_modular

    gen_df["n_modules"] = int(N_MODULES)
    gen_df["within_edge_fraction_target"] = float(WITHIN_EDGE_FRACTION)
    gen_df["within_edge_fraction_actual"] = float(modular_info["within_edge_fraction_actual"])

    return out, gen_df


# ============================================================
# Build design
# ============================================================

jobs = []

for phi_idx, phi in enumerate(PHI_VALUES):
    for seed_idx in range(N_SEEDS):
        seed = int(BASE_SEED + phi_idx * 100_000 + seed_idx)

        jobs.append({
            "condition": "modular_degree_matched",
            "phi_idx": int(phi_idx),
            "seed_idx": int(seed_idx),
            "phi": float(phi),
            "seed": int(seed),
            "n_modules": int(N_MODULES),
            "within_edge_fraction": float(WITHIN_EDGE_FRACTION),
        })

design = pd.DataFrame(jobs)

for k, v in start_dic.items():
    design[k] = v

design_path = OUTDIR / "modular_degree_matched_phi_design.csv"
design.to_csv(design_path, index=False)

print("Saved design:", design_path)
print("N jobs:", len(jobs))


# ============================================================
# Run sweep
# ============================================================

rows = []
gen_rows = []

ctx = mp.get_context("fork")

with ctx.Pool(processes=min(N_WORKERS, len(jobs))) as pool:
    for out, gen_df in tqdm(
        pool.imap_unordered(run_one_job, jobs, chunksize=1),
        total=len(jobs),
        desc="Modular degree-matched phi sweep",
    ):
        rows.append(out)
        gen_rows.append(gen_df)

        if len(rows) % 50 == 0:
            pd.DataFrame(rows).to_csv(
                OUTDIR / "modular_degree_matched_phi_seed_exponents_partial.csv",
                index=False,
            )

            pd.concat(gen_rows, ignore_index=True).to_csv(
                OUTDIR / "modular_degree_matched_phi_icg_rows_partial.csv",
                index=False,
            )


# ============================================================
# Save raw outputs
# ============================================================

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

seed_path = OUTDIR / "modular_degree_matched_phi_seed_exponents.csv"
icg_path = OUTDIR / "modular_degree_matched_phi_icg_rows.csv"

df_exp.to_csv(seed_path, index=False)
df_icg.to_csv(icg_path, index=False)

print("Saved:", seed_path)
print("Saved:", icg_path)


# ============================================================
# Summary by phi
# ============================================================

phi_summary = (
    df_exp
    .groupby("phi", as_index=False)
    .agg(
        n=("seed_idx", "count"),

        MV_alpha_mean=("MV_alpha", "mean"),
        MV_alpha_sem=("MV_alpha", sem),
        MV_r2_mean=("MV_r2", "mean"),
        MV_r2_sem=("MV_r2", sem),

        TAU_beta_mean=("TAU_beta", "mean"),
        TAU_beta_sem=("TAU_beta", sem),
        TAU_r2_mean=("TAU_r2", "mean"),
        TAU_r2_sem=("TAU_r2", sem),

        KURT_slope_mean=("KURT_slope", "mean"),
        KURT_slope_sem=("KURT_slope", sem),

        score_mean=("score", "mean"),
        score_sem=("score", sem),

        mean_rate_hz=("mean_rate_hz", "mean"),
        mean_rate_hz_sem=("mean_rate_hz", sem),

        frac_silent_frames=("frac_silent_frames", "mean"),
        frac_silent_frames_sem=("frac_silent_frames", sem),

        frac_active_neurons=("frac_active_neurons", "mean"),
        frac_active_neurons_sem=("frac_active_neurons", sem),

        n_edges_ref_hiermod_mean=("n_edges_ref_hiermod", "mean"),
        n_edges_modular_mean=("n_edges_modular", "mean"),

        edge_density_ref_hiermod_mean=("edge_density_ref_hiermod", "mean"),
        edge_density_modular_mean=("edge_density_modular", "mean"),

        mean_out_degree_ref_hiermod_mean=("mean_out_degree_ref_hiermod", "mean"),
        mean_out_degree_modular_mean=("mean_out_degree_modular", "mean"),

        n_modules_mean=("n_modules", "mean"),
        within_edge_fraction_target_mean=("within_edge_fraction_target", "mean"),
        within_edge_fraction_actual_mean=("within_edge_fraction_actual", "mean"),
        n_within_edges_mean=("n_within_edges", "mean"),
        n_between_edges_mean=("n_between_edges", "mean"),
        module_size_min_mean=("module_size_min", "mean"),
        module_size_max_mean=("module_size_max", "mean"),
    )
    .sort_values("phi")
    .reset_index(drop=True)
)

summary_path = OUTDIR / "modular_degree_matched_phi_summary.csv"
phi_summary.to_csv(summary_path, index=False)

print("Saved:", summary_path)

best_idx = phi_summary["score_mean"].idxmin()
best_phi = float(phi_summary.loc[best_idx, "phi"])

print("\nBest modular degree-matched phi by score:")
print(
    phi_summary.loc[[best_idx], [
        "phi",
        "MV_alpha_mean",
        "MV_alpha_sem",
        "TAU_beta_mean",
        "TAU_beta_sem",
        "score_mean",
        "score_sem",
        "mean_rate_hz",
        "mean_rate_hz_sem",
        "frac_silent_frames",
        "frac_active_neurons",
        "MV_r2_mean",
        "TAU_r2_mean",
        "edge_density_ref_hiermod_mean",
        "edge_density_modular_mean",
        "within_edge_fraction_actual_mean",
        "n",
    ]]
    .round(4)
    .to_string(index=False)
)


# ============================================================
# Plotting helpers
# ============================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})


def finish(name):
    plt.tight_layout()
    path = FIGDIR / name
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.show()
    print("Saved:", path)


# ============================================================
# Main 2x2 phi summary plot
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.ravel()

# ----------------------------
# Score
# ----------------------------
ax = axes[0]
ax.errorbar(
    phi_summary["phi"],
    phi_summary["score_mean"],
    yerr=phi_summary["score_sem"],
    marker="o",
    linewidth=1.5,
    capsize=2,
)
ax.axvline(best_phi, linestyle="--", linewidth=1)
ax.set_xlabel(r"$\phi$ used to set matched degree")
ax.set_ylabel("Score")
ax.set_title("Scaling score")
ax.grid(alpha=0.3)

# ----------------------------
# MV alpha
# ----------------------------
ax = axes[1]
ax.errorbar(
    phi_summary["phi"],
    phi_summary["MV_alpha_mean"],
    yerr=phi_summary["MV_alpha_sem"],
    marker="o",
    linewidth=1.5,
    capsize=2,
)
ax.axhline(TARGET_MV, linestyle="--", linewidth=1, label="target")
ax.axvline(best_phi, linestyle="--", linewidth=1)
ax.set_xlabel(r"$\phi$ used to set matched degree")
ax.set_ylabel(r"MV exponent $\alpha$")
ax.set_title("Mean-variance scaling")
ax.legend(frameon=False)
ax.grid(alpha=0.3)

# ----------------------------
# TAU beta
# ----------------------------
ax = axes[2]
ax.errorbar(
    phi_summary["phi"],
    phi_summary["TAU_beta_mean"],
    yerr=phi_summary["TAU_beta_sem"],
    marker="o",
    linewidth=1.5,
    capsize=2,
)
ax.axhline(TARGET_TAU, linestyle="--", linewidth=1, label="target")
ax.axvline(best_phi, linestyle="--", linewidth=1)
ax.set_xlabel(r"$\phi$ used to set matched degree")
ax.set_ylabel(r"Timescale exponent $\beta$")
ax.set_title("Timescale scaling")
ax.legend(frameon=False)
ax.grid(alpha=0.3)

# ----------------------------
# Mean rate
# ----------------------------
ax = axes[3]
ax.errorbar(
    phi_summary["phi"],
    phi_summary["mean_rate_hz"],
    yerr=phi_summary["mean_rate_hz_sem"],
    marker="o",
    linewidth=1.5,
    capsize=2,
)
ax.axvline(best_phi, linestyle="--", linewidth=1)
ax.set_xlabel(r"$\phi$ used to set matched degree")
ax.set_ylabel("Mean rate Hz")
ax.set_title("Mean firing rate")
ax.grid(alpha=0.3)

fig.suptitle(
    f"Flat modular degree-matched control; p_ext = {start_dic['p_ext']}; "
    f"best matched phi = {best_phi:.3f}",
    y=1.02,
    fontsize=14,
)
finish("modular_degree_matched_phi_summary_2x2.png")


# ============================================================
# Separate clean plots
# ============================================================

fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(
    phi_summary["phi"],
    phi_summary["score_mean"],
    yerr=phi_summary["score_sem"],
    marker="o",
    capsize=2,
)
ax.axvline(best_phi, linestyle="--", linewidth=1)
ax.set_xlabel(r"$\phi$ used to set matched degree")
ax.set_ylabel("Score")
ax.set_title("Flat modular degree-matched scaling score")
ax.grid(alpha=0.3)
finish("modular_degree_matched_score_vs_phi.png")


fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(
    phi_summary["phi"],
    phi_summary["MV_alpha_mean"],
    yerr=phi_summary["MV_alpha_sem"],
    marker="o",
    capsize=2,
    label=r"$\alpha$",
)
ax.axhline(TARGET_MV, linestyle="--", linewidth=1, label="target")
ax.axvline(best_phi, linestyle="--", linewidth=1)
ax.set_xlabel(r"$\phi$ used to set matched degree")
ax.set_ylabel(r"MV exponent $\alpha$")
ax.set_title("Flat modular degree-matched MV scaling")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
finish("modular_degree_matched_MV_alpha_vs_phi.png")


fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(
    phi_summary["phi"],
    phi_summary["TAU_beta_mean"],
    yerr=phi_summary["TAU_beta_sem"],
    marker="o",
    capsize=2,
    label=r"$\beta$",
)
ax.axhline(TARGET_TAU, linestyle="--", linewidth=1, label="target")
ax.axvline(best_phi, linestyle="--", linewidth=1)
ax.set_xlabel(r"$\phi$ used to set matched degree")
ax.set_ylabel(r"Timescale exponent $\beta$")
ax.set_title("Flat modular degree-matched timescale scaling")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
finish("modular_degree_matched_TAU_beta_vs_phi.png")


fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(
    phi_summary["phi"],
    phi_summary["mean_rate_hz"],
    yerr=phi_summary["mean_rate_hz_sem"],
    marker="o",
    capsize=2,
)
ax.axvline(best_phi, linestyle="--", linewidth=1)
ax.set_xlabel(r"$\phi$ used to set matched degree")
ax.set_ylabel("Mean rate Hz")
ax.set_title("Flat modular degree-matched mean firing rate")
ax.grid(alpha=0.3)
finish("modular_degree_matched_mean_rate_vs_phi.png")


fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(
    phi_summary["phi"],
    phi_summary["frac_silent_frames"],
    marker="o",
    label="silent frames",
)
ax.plot(
    phi_summary["phi"],
    phi_summary["frac_active_neurons"],
    marker="o",
    label="active neurons",
)
ax.axvline(best_phi, linestyle="--", linewidth=1)
ax.set_xlabel(r"$\phi$ used to set matched degree")
ax.set_ylabel("Fraction")
ax.set_title("Flat modular degree-matched activity diagnostics")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
finish("modular_degree_matched_activity_diagnostics_vs_phi.png")


fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(
    phi_summary["phi"],
    phi_summary["edge_density_ref_hiermod_mean"],
    marker="o",
    label="reference hiermod density",
)
ax.plot(
    phi_summary["phi"],
    phi_summary["edge_density_modular_mean"],
    marker="o",
    linestyle="--",
    label="modular density",
)
ax.set_xlabel(r"$\phi$")
ax.set_ylabel("Directed edge density")
ax.set_title("Edge-density matching check")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
finish("modular_degree_matched_edge_density_check.png")


fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(
    phi_summary["phi"],
    phi_summary["within_edge_fraction_actual_mean"],
    marker="o",
)
ax.axhline(WITHIN_EDGE_FRACTION, linestyle="--", linewidth=1, label="target")
ax.set_xlabel(r"$\phi$")
ax.set_ylabel("Within-module edge fraction")
ax.set_title("Flat modularity check")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
finish("modular_degree_matched_within_edge_fraction_check.png")


# ============================================================
# Seed scatter overlays
# ============================================================

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df_exp["phi"], df_exp["score"], alpha=0.25, s=16)
ax.plot(phi_summary["phi"], phi_summary["score_mean"], marker="o", linewidth=2)
ax.axvline(best_phi, linestyle="--", linewidth=1)
ax.set_xlabel(r"$\phi$ used to set matched degree")
ax.set_ylabel("Score")
ax.set_title("Flat modular seed-level score")
ax.grid(alpha=0.3)
finish("modular_degree_matched_seed_score_scatter_vs_phi.png")


fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df_exp["phi"], df_exp["MV_alpha"], alpha=0.25, s=16)
ax.plot(phi_summary["phi"], phi_summary["MV_alpha_mean"], marker="o", linewidth=2)
ax.axhline(TARGET_MV, linestyle="--", linewidth=1)
ax.axvline(best_phi, linestyle="--", linewidth=1)
ax.set_xlabel(r"$\phi$ used to set matched degree")
ax.set_ylabel(r"MV exponent $\alpha$")
ax.set_title("Flat modular seed-level MV alpha")
ax.grid(alpha=0.3)
finish("modular_degree_matched_seed_MV_alpha_scatter_vs_phi.png")


fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df_exp["phi"], df_exp["TAU_beta"], alpha=0.25, s=16)
ax.plot(phi_summary["phi"], phi_summary["TAU_beta_mean"], marker="o", linewidth=2)
ax.axhline(TARGET_TAU, linestyle="--", linewidth=1)
ax.axvline(best_phi, linestyle="--", linewidth=1)
ax.set_xlabel(r"$\phi$ used to set matched degree")
ax.set_ylabel(r"Timescale exponent $\beta$")
ax.set_title("Flat modular seed-level TAU beta")
ax.grid(alpha=0.3)
finish("modular_degree_matched_seed_TAU_beta_scatter_vs_phi.png")


print("\nDone.")
print("OUTDIR:", OUTDIR)
print("Saved:")
print("  design:", design_path)
print("  seed exponents:", seed_path)
print("  icg rows:", icg_path)
print("  summary:", summary_path)
print("  plots:", FIGDIR)