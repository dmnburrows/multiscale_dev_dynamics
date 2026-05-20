#Import packages
#---------------------------------------
import sys
import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import glob
import admin_functions as adfn
import trace_analyse as tfn
from tqdm import tqdm
import re


#ICG
import numpy as np

# ============================================================
# Fast greedy ICG (Iterative Coarse-Graining) for X: (N, T)
# - Greedy pairing by Pearson correlation at each generation
# - Returns:
#   generations: list[int]
#   labels_by_gen: list[np.ndarray]  (each: (N_original,) cluster id at that generation)
#   mean_traces_by_gen: list[np.ndarray] (each: (n_clusters_gen, T) mean trace per cluster)
#   sizes_by_gen: list[np.ndarray]  (each: (n_clusters_gen,) #cells per cluster)
#
# Notes:
# - This is "fast" in the sense that each generation uses a single BLAS matmul to
#   form the correlation matrix, and greedy pairing runs without sorting edges.
# - Memory: correlation matrix at gen0 is ~N^2 float32 (N=10k -> ~400MB).
# ============================================================

def _zscore_rows(X, eps=1e-6):
    """Row-wise z-score in float32."""
    X = np.asarray(X, dtype=np.float32, order="C")
    mu = X.mean(axis=1, keepdims=True)
    Xc = X - mu
    sd = Xc.std(axis=1, keepdims=True)
    return Xc / (sd + eps)

def _greedy_pairs_from_corr_inplace(C):
    """
    Greedy pairing from correlation matrix C (modified in-place):
      - pick i in order; pair with argmax(C[i]) among remaining.
      - when i,j paired: set rows/cols i,j to -inf so they cannot be reused.
    Returns:
      pairs: (P, 2) int32 with i<j
      singles: (S,) int32 remaining (if odd n)
    """
    n = C.shape[0]
    pairs = []
    alive = np.ones(n, dtype=bool)

    # diagonal cannot pair with itself
    np.fill_diagonal(C, -np.inf)

    for i in range(n):
        if not alive[i]:
            continue

        j = int(np.argmax(C[i]))  # columns of dead nodes have been set to -inf
        if (j == i) or (not alive[j]) or (not np.isfinite(C[i, j])):
            # couldn't find a partner (should only happen if n==1)
            continue

        a, b = (i, j) if i < j else (j, i)
        pairs.append((a, b))

        # kill both nodes
        alive[i] = False
        alive[j] = False

        # remove them from further consideration: set rows+cols to -inf
        C[i, :] = -np.inf
        C[:, i] = -np.inf
        C[j, :] = -np.inf
        C[:, j] = -np.inf

    pairs = np.asarray(pairs, dtype=np.int32)
    singles = np.where(alive)[0].astype(np.int32)
    return pairs, singles

def icg_greedy_fast(
    X,
    n_gen=None,
    eps=1e-6,
    keep_singletons=True,
    return_gen0=True,
    verbose=False,
):
    """
    Parameters
    ----------
    X : array (N, T)
        Per-cell traces (can be dff, deconvolved, etc).
    n_gen : int or None
        Number of coarse-graining generations to compute (None -> until 1 cluster).
    eps : float
        Stabilizer for z-scoring.
    keep_singletons : bool
        If odd number of units at some generation, keep last as a singleton.
    return_gen0 : bool
        Include generation 0 outputs (labels=0..N-1, mean_traces=X, sizes=1).
    verbose : bool
        Print progress.

    Returns
    -------
    generations : list[int]
    labels_by_gen : list[np.ndarray]      # each (N_original,)
    mean_traces_by_gen : list[np.ndarray] # each (n_clusters_gen, T)
    sizes_by_gen : list[np.ndarray]       # each (n_clusters_gen,)
    """
    X0 = np.asarray(X, dtype=np.float32, order="C")
    N0, T = X0.shape

    # current level representation: units x T are SUMD traces of underlying cells
    X_curr = X0
    sizes_curr = np.ones(X_curr.shape[0], dtype=np.int32)

    # map original cells -> current unit id (starts as identity)
    labels_curr = np.arange(N0, dtype=np.int32)

    generations = []
    labels_by_gen = []
    sum_traces_by_gen = []
    sizes_by_gen = []

    def _append(gen, labels, traces, sizes):
        generations.append(gen)
        labels_by_gen.append(labels.copy())
        sum_traces_by_gen.append(np.asarray(traces, dtype=np.float32))
        sizes_by_gen.append(sizes.copy())

    gen = 0
    if return_gen0:
        _append(gen, labels_curr, X_curr, sizes_curr)

    # how many gens?
    if n_gen is None:
        # until we reach 1 cluster
        max_gen = 10_000  # hard cap safety
    else:
        max_gen = int(n_gen)

    while gen < max_gen:
        n_units = X_curr.shape[0]
        if n_units <= 1:
            break

        gen += 1
        if verbose:
            print(f"[ICG] gen={gen}  n_units={n_units}")

        # 1) correlation matrix for current units (float32)
        Z = _zscore_rows(X_curr, eps=eps)  # (n_units, T)
        C = (Z @ Z.T) / float(T)          # (n_units, n_units) float32

        # 2) greedy pairing on C in-place
        pairs, singles = _greedy_pairs_from_corr_inplace(C)

        # 3) build children lists for new clusters
        if pairs.size == 0 and singles.size == n_units:
            # nothing could be paired (shouldn't happen, but guard)
            if verbose:
                print("[ICG] no pairs formed; stopping.")
            break

        # Decide how to handle singles
        if (not keep_singletons) and (singles.size > 0):
            # drop the singles
            singles = np.empty((0,), dtype=np.int32)

        P = pairs.shape[0]
        S = singles.shape[0]
        n_new = P + S

        # children indices
        a = np.empty(n_new, dtype=np.int32)
        b = np.empty(n_new, dtype=np.int32)

        # pairs first
        if P > 0:
            a[:P] = pairs[:, 0]
            b[:P] = pairs[:, 1]

        # singletons: represent as (s, s) but with weight_b=0
        if S > 0:
            a[P:] = singles
            b[P:] = singles

        # 4) update cluster sizes
        sa = sizes_curr[a].astype(np.float32)
        sb = sizes_curr[b].astype(np.float32)
        if S > 0:
            # zero out sb for singleton rows (so mean stays unchanged)
            sb[P:] = 0.0

        sizes_new = (sa + sb).astype(np.int32)  # (#cells in each new cluster)
        denom = (sa + sb).reshape(-1, 1)        # float32

        # 5) update mean traces (weighted mean of children clusters)
        Xa = X_curr[a]  # (n_new, T)
        Xb = X_curr[b]  # (n_new, T)
        # X_new = (sa.reshape(-1, 1) * Xa + sb.reshape(-1, 1) * Xb) / denom

        X_new = Xa + Xb
        # keep singleton rows unchanged
        if S > 0:
            X_new[P:] = Xa[P:]


        # 6) mapping old units -> new units (for propagating original-cell labels)
        unit_to_new = np.full(n_units, -1, dtype=np.int32)
        unit_to_new[a] = np.arange(n_new, dtype=np.int32)
        if P > 0:
            unit_to_new[b[:P]] = np.arange(P, dtype=np.int32)  # second child of pairs

        # propagate labels for original cells
        labels_curr = unit_to_new[labels_curr]
        if np.any(labels_curr < 0):
            raise RuntimeError("Label propagation failed (found -1). Check singleton handling.")

        # save outputs at this generation
        _append(gen, labels_curr, X_new, sizes_new)

        # advance
        X_curr = X_new
        sizes_curr = sizes_new

    return generations, labels_by_gen, sum_traces_by_gen, sizes_by_gen

# ============================
# Example usage
# ============================
# X: (N,T) e.g. dff or deconvolved events
# gen, label_l, icgtr_l, size_l = icg_greedy_fast(
#     dff, n_gen=14, eps=1e-6, keep_singletons=True, return_gen0=True, verbose=True
# )
# #
# labels_by_gen[g] gives you an (N,) vector mapping each original cell -> cluster id at gen g
# mean_traces_by_gen[g] is (n_clusters_at_g, T) mean trace for each cluster at gen g
# sizes_by_gen[g] is (#cells in each cluster at gen g)


#======================
def kurtosis_corr(data):
#======================
    corr = np.corrcoef(data)
    x = corr[np.triu_indices(corr.shape[0], k=1)]
    
    #kurtosis 
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return np.nan
    m = x.mean()
    v = x.var()
    if v <= 0:
        return np.nan
    return np.mean((x - m) ** 4) / (v ** 2)


def mean_variance_sum(data):
    if data.size < 10:
        return np.nan
    return float(np.mean(np.var(data, axis=1)))

import numpy as np

def mean_autocorr_curve(X, max_lag_frames=10, eps=1e-8):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        return None

    N, T = X.shape
    L = min(max_lag_frames, T - 2)
    if N < 1 or L < 1:
        return None

    ac_sum = np.zeros(L + 1, dtype=np.float64)
    n_used = 0

    for i in range(N):
        x = X[i]

        # safer than removing bad points and compressing time
        if not np.all(np.isfinite(x)):
            continue
        if x.size < L + 2:
            continue

        x = x - x.mean()
        v = x.var(ddof=0)
        if v < eps:
            continue

        x = x / np.sqrt(v)

        ac = np.empty(L + 1, dtype=np.float64)
        ac[0] = 1.0
        for lag in range(1, L + 1):
            ac[lag] = np.dot(x[:-lag], x[lag:]) / (x.size - lag)

        ac_sum += ac
        n_used += 1

    if n_used == 0:
        return None

    return ac_sum / n_used


def timescale(X, max_lag=3.0, dt=1.0):
    max_lag_frames = int(np.floor(max_lag / dt))
    ac_mean = mean_autocorr_curve(X, max_lag_frames=max_lag_frames)
    if ac_mean is None:
        return np.nan

    t = np.arange(ac_mean.size) * dt
    return float(np.trapz(ac_mean, t))


def mapp(curr, n_neurons):
    lab = np.full(n_neurons, -1, dtype=int)
    lab[curr.ravel()] = np.repeat(np.arange(len(curr)), curr.shape[1])
    return lab

def label_neuron_bylevel(n_neurons, rng):
    # first level: neuron pairs
    perm0 = rng.permutation(n_neurons)
    # if odd, leave one neuron unlabeled at level 0
    n_used = n_neurons - (n_neurons % 2)
    curr_vec = perm0[:n_used].reshape(-1, 2)
    out_l = [curr_vec]
    
    
    # higher levels: pair previous rows, drop one if odd
    while len(curr_vec) > 1:
        perm = rng.permutation(len(curr_vec))
        if len(perm) % 2:
            perm = perm[:-1]
        curr_vec = perm.reshape(-1, 2)
        out_l.append(curr_vec)
    
    # map each level back to original neurons
    groupings = [out_l[0]]
    vals = out_l[0]
    
    for out in out_l[1:]:
        vals = vals[out].reshape(len(out), -1)
        groupings.append(vals)
    
    # labels for each neuron at each level
    labs = [mapp(g, n_neurons) for g in groupings]
    return(labs)

# draw A from defined P at each level
#first connect all pairs at l = 0
def wire_A(n_neurons, labs, t, rng):
    A = np.zeros((n_neurons, n_neurons), dtype=int)
    idx0 = np.where(labs[0] >= 0)[0]
    pair_idx = idx0[np.argsort(labs[0][idx0])].reshape(-1, 2)
    
    A[pair_idx[:, 0], pair_idx[:, 1]] = 1
    A[pair_idx[:, 1], pair_idx[:, 0]] = 1
    
    #now draw from P = t**l
    for x,l in enumerate(labs[1:]):
        P =  1 / (t**(x+1))
        n = 2 ** (x + 2)   # 4, 8, 16, ...
        idx = np.where(l >= 0)[0]
        pair_idx = idx[np.argsort(l[idx])].reshape(-1, n)
        
        for row in pair_idx:
            left = row[:len(row)//2]
            right = row[len(row)//2:]
            
            ii, jj = np.meshgrid(left, right, indexing="ij")
            comb_v = np.column_stack([ii.ravel(), jj.ravel()])
            A[comb_v[:,0] , comb_v[:,1]] = (P > rng.random(size = comb_v.shape[0])).astype(int)
            A[comb_v[:,1] , comb_v[:,0]] = (P > rng.random(size = comb_v.shape[0])).astype(int)
    return(A)

def exp_smooth_spikes(spk, dt=0.01, tau=0.2):
    """
    Causal exponential smoothing of binary spikes.
    
    spk: neurons x time, binary or counts
    dt: timestep in seconds
    tau: smoothing time constant in seconds
    """
    spk = np.asarray(spk, dtype=np.float32)
    alpha = np.exp(-dt / tau)

    X = np.zeros_like(spk, dtype=np.float32)
    X[:, 0] = spk[:, 0]

    for t in range(1, spk.shape[1]):
        X[:, t] = alpha * X[:, t - 1] + spk[:, t]

    return X

#====================
class automata_EI_hiermod:
#====================

    """
    3-state EI network:
      0 = quiescent
      1 = active
      2..(refractory_steps+1) = refractory countdown

    A quiescent neuron can become active from:
      - external Poisson-like drive p_ext
      - network input from active presynaptic neurons

    Network coupling is controlled by g. Higher g -> more synchronous / bursty activity.
    """

    def __init__(
        self,
        n_neurons=1000,
        ei_ratio = 0.2,
        e_w=0.8,
        i_w = 0.8,
        theta = 1.5,
        slope = 2,
        p_ext=0.001,
        
        refractory_steps=2,
        dt=0.01,
        seed=0
    ):
        self.rng = np.random.default_rng(seed)
        self.n = n_neurons
        self.e = int(n_neurons - (n_neurons*ei_ratio))
        self.i = int(n_neurons*ei_ratio)
        self.e_w=e_w
        self.i_w = i_w
        self.theta = theta
        self.slope = slope
        self.p_ext = p_ext
        self.refractory_steps = refractory_steps
        self.dt = dt

        #Hierarchical-modular connections
        self.labs = label_neuron_bylevel(self.n, rng=self.rng)
        self.A = wire_A(n_neurons=self.n, labs=self.labs, t=self.slope, rng=self.rng)
        self.A_e = self.A[:self.e, :]
        self.A_i = self.A[self.e:, :]
        # State vector: 0=Q, 1=A, >=2 refractory countdown states
        self.state = np.zeros(self.n, dtype=np.int16)


    def step(self):
        active = (self.state == 1)
        
        # Network input: number of active presynaptic neurons
        inp_e = self.A_e.T @ active[:self.e].astype(np.float32)#input from E
        inp_i = self.A_i.T @ active[self.e:].astype(np.float32) #input from I

        # Convert summed E & I into activation probability via sigmoid
        net = (self.e_w * inp_e) - (self.i_w * inp_i)
        p_net = 1 / (1 + np.exp(-(net - self.theta))) #sigmoid
            
        quiescent = (self.state == 0)

        # External drive
        ext_events = self.rng.random(self.n) < self.p_ext

        # Network-driven events
        net_events = self.rng.random(self.n) < p_net

        #make it combinatorial???
        new_active = quiescent & (ext_events | net_events) 

        # Update refractory dynamics
        new_state = np.zeros_like(self.state)

        # Active neurons enter refractory
        new_state[active] = 2

        # Refractory neurons advance countdown
        refractory = self.state >= 2
        new_state[refractory] = self.state[refractory] + 1

        # End refractory after refractory_steps
        done_refrac = new_state > (self.refractory_steps + 1)
        new_state[done_refrac] = 0

        # Newly activated neurons become active
        new_state[new_active] = 1

        self.state = new_state
        return (new_state == 1).astype(np.uint8)

#Sanity check for CA
#====================
def run_model(model, T=10.0):
    n_steps = int(T / model.dt)

    spikes = np.zeros((model.n, n_steps), dtype=np.uint8)
    pop_rate = np.zeros(n_steps, dtype=float)

    for t in range(n_steps):
        active = model.step()
        spikes[:, t] = active
        pop_rate[t] = active.mean() / model.dt

    return spikes, pop_rate


#Starting pars
top = pd.read_csv(
    "/home/dburrows/DATA/BLNDEV-WILDTYPE/automataEIhiermod_topsims.csv",
    index_col=0,
)

curr = (
    top.loc[top["original_sim_id"] == 83, ["p_ext", "slope", "e_w", "i_w", "theta"]]
    .iloc[0]
    .to_dict()
)

# OLD / base
start_dic = {
    "n_neurons": 2000,
    "T": 10,
    "dt": 0.01,
    "refractory_steps": 2,
    "ei_ratio": 0.2,
    "e_w": 25.19,
    "i_w": 25.5,
    "theta": 8.1,
    "p_ext": 0.015,
    "slope": 2.13,
}

# NEW: overwrite only p_ext, slope, e_w, i_w, theta
start_dic.update({k: float(v) for k, v in curr.items()})
start_dic.update({'slope': 2.35})
start_dic


# ============================================================
# FAST COMBINATORIAL BOUNDARY SWEEP
#
# Sweep:
#   phi/slope: 1 -> 5, 5 values
#   theta
#   ei_ratio
#   n_neurons
#
# Fixed from start_dic:
#   p_ext, e_w, i_w, dt, refractory_steps
#
# Saves:
#   boundary_fast_icg_rows.csv
#   boundary_fast_seed_exponents.csv
#   boundary_fast_summary.csv
#   boundary_fast_boundary_estimates.csv
#   plots/*.png
# ============================================================

import os
import json
import itertools
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


# ============================================================
# 0. Settings
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

OUTDIR = Path("/home/dburrows/DATA/BLNDEV-WILDTYPE/boundary_fast_phi_theta_eiratio_N")
OUTDIR.mkdir(parents=True, exist_ok=True)

PLOTDIR = OUTDIR / "plots"
PLOTDIR.mkdir(parents=True, exist_ok=True)

# Main requested sweep
PHI_VALUES = np.round(np.linspace(1.0, 5.0, 5), 3)

THETA_VALUES = np.array([6.5, 7.5, 8.5])
EI_RATIO_VALUES = np.array([0.10, 0.20, 0.30])
N_NEURONS_VALUES = np.array([500, 1000, 2000])

# Fast settings
N_SEEDS_PER_COND = 2
N_WORKERS = 20
BASE_SEED = 77000

# Use shorter T for fast boundary scan.
# Change to float(start_dic["T"]) for full-length runs.
T_RUN = 5.0

MAX_ICG_LEVELS = 8
MIN_CLUSTERS = 4

SMOOTH_TAU = 0.20
AC_MAX_LAG = 3.0

TARGET_MV = 1.50
TARGET_TAU = 0.20

EXCLUDE_FIRST_FIT_POINT = True
EXCLUDE_LAST_FIT_POINT = False

print("FAST combinatorial boundary sweep")
print("phis:", PHI_VALUES)
print("thetas:", THETA_VALUES)
print("ei_ratios:", EI_RATIO_VALUES)
print("n_neurons:", N_NEURONS_VALUES)
print("seeds per condition:", N_SEEDS_PER_COND)
print("workers:", N_WORKERS)
print("T_RUN:", T_RUN)
print("OUTDIR:", OUTDIR)

total_sims = (
    len(PHI_VALUES)
    * len(THETA_VALUES)
    * len(EI_RATIO_VALUES)
    * len(N_NEURONS_VALUES)
    * N_SEEDS_PER_COND
)
print("Total simulations:", total_sims)


# ============================================================
# 1. Helpers
# ============================================================

def sem(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) <= 1:
        return np.nan
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def fit_powerlaw_one_seed(
    g,
    y_col,
    x_col="mean_cluster_size",
    exclude_first=True,
    exclude_last=False,
    min_points=3,
):
    d = g[[x_col, y_col]].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    d = d[(d[x_col] > 0) & (d[y_col] > 0)]
    d = d.sort_values(x_col)

    if exclude_first and len(d) > 0:
        d = d.iloc[1:]

    if exclude_last and len(d) > 0:
        d = d.iloc[:-1]

    if len(d) < min_points:
        return {"alpha": np.nan, "r2": np.nan, "n_points": int(len(d))}

    logx = np.log10(d[x_col].to_numpy(float))
    logy = np.log10(d[y_col].to_numpy(float))

    alpha, intercept = np.polyfit(logx, logy, 1)
    pred = intercept + alpha * logx

    ss_res = np.sum((logy - pred) ** 2)
    ss_tot = np.sum((logy - np.mean(logy)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "alpha": float(alpha),
        "r2": float(r2),
        "n_points": int(len(d)),
    }


def estimate_saturation_phi(x, y, frac=0.10):
    """
    First phi where y is within frac of final plateau.
    Plateau = mean of final 2 phi points.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    if len(x) < 4:
        return np.nan

    order = np.argsort(x)
    x, y = x[order], y[order]

    plateau = np.nanmean(y[-2:])
    start = y[0]
    span = abs(plateau - start)

    if span <= 1e-12:
        return float(x[0])

    dist = np.abs(y - plateau)
    idx = np.where(dist <= frac * span)[0]

    if len(idx) == 0:
        return np.nan

    return float(x[idx[0]])


def add_normed_seed_metrics(df, group_col="sim_id"):
    d = df.copy()
    d = d.sort_values([group_col, "mean_cluster_size"])

    d["scaled_variance_norm"] = (
        d["scaled_variance"]
        / d.groupby(group_col)["scaled_variance"].transform("first")
    )

    d["timescale_norm"] = (
        d["timescale"]
        / d.groupby(group_col)["timescale"].transform("first")
    )

    return d


# ============================================================
# 2. Worker
# ============================================================

def run_one_boundary_job(job):
    """
    One job = one phi × theta × ei_ratio × n_neurons × seed.
    """
    phi, theta, ei_ratio, n_neurons, seed = job

    pars = dict(start_dic)
    pars.pop("T", None)

    pars["slope"] = float(phi)
    pars["theta"] = float(theta)
    pars["ei_ratio"] = float(ei_ratio)
    pars["n_neurons"] = int(n_neurons)
    pars["seed"] = int(seed)

    model = automata_EI_hiermod(**pars)
    spikes, pop_rate = run_model(model, T=float(T_RUN))

    X = exp_smooth_spikes(
        spikes,
        dt=float(pars["dt"]),
        tau=SMOOTH_TAU,
    )

    gen_l, label_l, icgtr_l, size_l = icg_greedy_fast(
        X,
        n_gen=MAX_ICG_LEVELS,
        eps=1e-6,
        keep_singletons=True,
        return_gen0=True,
        verbose=False,
    )

    rows = []

    sim_id = (
        f"phi{phi:.3f}_theta{theta:.3f}_eir{ei_ratio:.3f}_"
        f"N{int(n_neurons)}_seed{int(seed)}"
    )

    for gen, Xg, sg in zip(gen_l, icgtr_l, size_l):
        if Xg.shape[0] < MIN_CLUSTERS:
            break

        rows.append({
            "phi": float(phi),
            "slope": float(phi),
            "theta": float(theta),
            "ei_ratio": float(ei_ratio),
            "n_neurons": int(n_neurons),
            "seed": int(seed),
            "sim_id": sim_id,

            "icg_level": int(gen),
            "mean_cluster_size": float(np.mean(sg)),
            "n_clusters": int(Xg.shape[0]),

            "scaled_variance": float(mean_variance_sum(Xg)),
            "timescale": float(timescale(
                Xg,
                max_lag=AC_MAX_LAG,
                dt=float(pars["dt"]),
            )),
            "kurtosis_corr": float(kurtosis_corr(Xg)),

            "mean_rate_hz": float(spikes.mean() / float(pars["dt"])),
            "pop_rate_mean_hz": float(np.mean(pop_rate)),
            "pop_rate_std_hz": float(np.std(pop_rate)),

            "n_edges": int(model.A.sum()),
            "density": float(model.A.mean()),
            "mean_out_degree": float(model.A.sum(axis=1).mean()),
            "mean_in_degree": float(model.A.sum(axis=0).mean()),

            "p_ext": float(pars["p_ext"]),
            "e_w": float(pars["e_w"]),
            "i_w": float(pars["i_w"]),
            "dt": float(pars["dt"]),
            "T": float(T_RUN),
            "refractory_steps": int(pars["refractory_steps"]),
        })

    return pd.DataFrame(rows)


# ============================================================
# 3. Build jobs
# ============================================================

jobs = []

cond_i = 0

for phi, theta, ei_ratio, n_neurons in itertools.product(
    PHI_VALUES,
    THETA_VALUES,
    EI_RATIO_VALUES,
    N_NEURONS_VALUES,
):
    cond_i += 1
    seeds = BASE_SEED + cond_i * 1000 + np.arange(N_SEEDS_PER_COND)

    for seed in seeds:
        jobs.append((
            float(phi),
            float(theta),
            float(ei_ratio),
            int(n_neurons),
            int(seed),
        ))

print("Total jobs:", len(jobs))


# ============================================================
# 4. Run multiprocessing
# ============================================================

all_rows = []

ctx = mp.get_context("fork")

with ctx.Pool(processes=N_WORKERS) as pool:
    for out_df in tqdm(
        pool.imap_unordered(run_one_boundary_job, jobs),
        total=len(jobs),
        desc="Running boundary jobs",
    ):
        all_rows.append(out_df)

df = pd.concat(all_rows, ignore_index=True)

df = (
    df
    .sort_values(["n_neurons", "theta", "ei_ratio", "phi", "seed", "icg_level"])
    .reset_index(drop=True)
)

df = add_normed_seed_metrics(df, group_col="sim_id")

df.to_csv(OUTDIR / "boundary_fast_icg_rows.csv", index=False)

print("Saved:", OUTDIR / "boundary_fast_icg_rows.csv")
print("ICG rows:", len(df))
print("Unique sims:", df["sim_id"].nunique())


# ============================================================
# 5. Fit seed-level exponents
# ============================================================

exp_rows = []

for sim_id, g in df.groupby("sim_id"):
    mv_fit = fit_powerlaw_one_seed(
        g,
        y_col="scaled_variance_norm",
        exclude_first=EXCLUDE_FIRST_FIT_POINT,
        exclude_last=EXCLUDE_LAST_FIT_POINT,
    )

    tau_fit = fit_powerlaw_one_seed(
        g,
        y_col="timescale_norm",
        exclude_first=EXCLUDE_FIRST_FIT_POINT,
        exclude_last=EXCLUDE_LAST_FIT_POINT,
    )

    kurt_fit = fit_powerlaw_one_seed(
        g,
        y_col="kurtosis_corr",
        exclude_first=EXCLUDE_FIRST_FIT_POINT,
        exclude_last=EXCLUDE_LAST_FIT_POINT,
    )

    first = g.iloc[0]

    exp_rows.append({
        "sim_id": sim_id,
        "phi": float(first["phi"]),
        "theta": float(first["theta"]),
        "ei_ratio": float(first["ei_ratio"]),
        "n_neurons": int(first["n_neurons"]),
        "seed": int(first["seed"]),

        "MV_alpha": mv_fit["alpha"],
        "MV_r2": mv_fit["r2"],
        "MV_n_points": mv_fit["n_points"],

        "TAU_beta": tau_fit["alpha"],
        "TAU_r2": tau_fit["r2"],
        "TAU_n_points": tau_fit["n_points"],

        "KURT_gamma": kurt_fit["alpha"],
        "KURT_r2": kurt_fit["r2"],
        "KURT_n_points": kurt_fit["n_points"],

        "score": (
            abs(mv_fit["alpha"] - TARGET_MV)
            + 2.0 * abs(tau_fit["alpha"] - TARGET_TAU)
        ),

        "mean_rate_hz": float(first["mean_rate_hz"]),
        "pop_rate_mean_hz": float(first["pop_rate_mean_hz"]),
        "pop_rate_std_hz": float(first["pop_rate_std_hz"]),

        "n_edges": int(first["n_edges"]),
        "density": float(first["density"]),
        "mean_out_degree": float(first["mean_out_degree"]),

        "p_ext": float(first["p_ext"]),
        "e_w": float(first["e_w"]),
        "i_w": float(first["i_w"]),
        "dt": float(first["dt"]),
        "T": float(first["T"]),
        "refractory_steps": int(first["refractory_steps"]),
    })

exp_df = pd.DataFrame(exp_rows)

exp_df.to_csv(OUTDIR / "boundary_fast_seed_exponents.csv", index=False)

print("Saved:", OUTDIR / "boundary_fast_seed_exponents.csv")


# ============================================================
# 6. Summary by condition
# ============================================================

summary = (
    exp_df
    .groupby(["n_neurons", "theta", "ei_ratio", "phi"], as_index=False)
    .agg(
        MV_alpha_mean=("MV_alpha", "mean"),
        MV_alpha_sem=("MV_alpha", sem),
        MV_r2_mean=("MV_r2", "mean"),

        TAU_beta_mean=("TAU_beta", "mean"),
        TAU_beta_sem=("TAU_beta", sem),
        TAU_r2_mean=("TAU_r2", "mean"),

        KURT_gamma_mean=("KURT_gamma", "mean"),
        KURT_gamma_sem=("KURT_gamma", sem),
        KURT_r2_mean=("KURT_r2", "mean"),

        score_mean=("score", "mean"),
        score_sem=("score", sem),

        mean_rate_hz=("mean_rate_hz", "mean"),
        pop_rate_mean_hz=("pop_rate_mean_hz", "mean"),
        pop_rate_std_hz=("pop_rate_std_hz", "mean"),

        n_edges=("n_edges", "mean"),
        density=("density", "mean"),
        mean_out_degree=("mean_out_degree", "mean"),

        n=("seed", "count"),
    )
    .sort_values(["n_neurons", "theta", "ei_ratio", "phi"])
    .reset_index(drop=True)
)

summary.to_csv(OUTDIR / "boundary_fast_summary.csv", index=False)

print("Saved:", OUTDIR / "boundary_fast_summary.csv")

print("\nTop conditions by score:")
print(
    summary
    .sort_values("score_mean")
    .head(25)
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 7. Estimate saturation/boundary phi
# ============================================================

boundary_rows = []

for keys, g in summary.groupby(["n_neurons", "theta", "ei_ratio"]):
    n_neurons, theta, ei_ratio = keys
    g = g.sort_values("phi")

    boundary_rows.append({
        "n_neurons": int(n_neurons),
        "theta": float(theta),
        "ei_ratio": float(ei_ratio),

        "phi_sat_edges": estimate_saturation_phi(g["phi"], g["mean_out_degree"]),
        "phi_sat_rate": estimate_saturation_phi(g["phi"], g["mean_rate_hz"]),
        "phi_sat_MV": estimate_saturation_phi(g["phi"], g["MV_alpha_mean"]),
        "phi_sat_TAU": estimate_saturation_phi(g["phi"], g["TAU_beta_mean"]),
        "phi_sat_score": estimate_saturation_phi(g["phi"], g["score_mean"]),

        "final_mean_degree": float(np.nanmean(g["mean_out_degree"].tail(2))),
        "final_rate": float(np.nanmean(g["mean_rate_hz"].tail(2))),
        "final_MV_alpha": float(np.nanmean(g["MV_alpha_mean"].tail(2))),
        "final_TAU_beta": float(np.nanmean(g["TAU_beta_mean"].tail(2))),
        "final_score": float(np.nanmean(g["score_mean"].tail(2))),
    })

boundary_df = pd.DataFrame(boundary_rows)

boundary_df.to_csv(OUTDIR / "boundary_fast_boundary_estimates.csv", index=False)

print("\nBoundary estimates:")
print(boundary_df.round(4).to_string(index=False))

print("Saved:", OUTDIR / "boundary_fast_boundary_estimates.csv")


# ============================================================
# 8. Save run config
# ============================================================

run_config = {
    "mode": "fast_combinatorial_boundary_sweep",
    "PHI_VALUES": [float(x) for x in PHI_VALUES],
    "THETA_VALUES": [float(x) for x in THETA_VALUES],
    "EI_RATIO_VALUES": [float(x) for x in EI_RATIO_VALUES],
    "N_NEURONS_VALUES": [int(x) for x in N_NEURONS_VALUES],
    "N_SEEDS_PER_COND": int(N_SEEDS_PER_COND),
    "N_WORKERS": int(N_WORKERS),
    "BASE_SEED": int(BASE_SEED),
    "T_RUN": float(T_RUN),
    "MAX_ICG_LEVELS": int(MAX_ICG_LEVELS),
    "MIN_CLUSTERS": int(MIN_CLUSTERS),
    "SMOOTH_TAU": float(SMOOTH_TAU),
    "AC_MAX_LAG": float(AC_MAX_LAG),
    "TARGET_MV": float(TARGET_MV),
    "TARGET_TAU": float(TARGET_TAU),
    "OUTDIR": str(OUTDIR),
    "start_dic": {
        k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
        for k, v in start_dic.items()
    },
}

with open(OUTDIR / "run_config.json", "w") as f:
    json.dump(run_config, f, indent=2)

print("Saved:", OUTDIR / "run_config.json")


# ============================================================
# 9. Plotting
# ============================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})


def savefig(fig, name):
    path = PLOTDIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print("Saved:", path)


# ------------------------------------------------------------
# 9.1 Curves: one figure per N and theta, lines are ei_ratio
# ------------------------------------------------------------

metrics = [
    ("mean_out_degree", None, "Mean out-degree", None),
    ("mean_rate_hz", None, "Mean firing rate [Hz/neuron]", None),
    ("MV_alpha_mean", "MV_alpha_sem", "MV exponent α", TARGET_MV),
    ("TAU_beta_mean", "TAU_beta_sem", "Timescale exponent β", TARGET_TAU),
    ("KURT_gamma_mean", "KURT_gamma_sem", "Kurtosis slope γ", None),
    ("score_mean", "score_sem", "Target score", None),
]

for n_neurons in sorted(summary["n_neurons"].unique()):
    for theta in sorted(summary["theta"].unique()):

        d = summary[
            (summary["n_neurons"] == n_neurons)
            & (summary["theta"] == theta)
        ].copy()

        fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)

        for ax, (metric, sem_col, ylabel, target) in zip(axes.ravel(), metrics):
            for ei_ratio, g in d.groupby("ei_ratio"):
                g = g.sort_values("phi")

                yerr = None
                if sem_col is not None and sem_col in g.columns:
                    yerr = g[sem_col]

                ax.errorbar(
                    g["phi"],
                    g[metric],
                    yerr=yerr,
                    marker="o",
                    linewidth=2,
                    capsize=3,
                    label=f"EI ratio={ei_ratio:g}",
                )

            if target is not None:
                ax.axhline(target, linestyle="--", color="black", linewidth=1.2)

            ax.set_xlabel("Φ / slope")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.grid(alpha=0.25)

        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"N={n_neurons}, theta={theta:g}", fontsize=14)

        savefig(fig, f"curves_N{n_neurons}_theta{theta:g}.png")
        plt.show()


# ------------------------------------------------------------
# 9.2 Boundary scatter summaries
# ------------------------------------------------------------

for metric in ["phi_sat_edges", "phi_sat_rate", "phi_sat_MV", "phi_sat_TAU", "phi_sat_score"]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    sc = axes[0].scatter(
        boundary_df["n_neurons"],
        boundary_df[metric],
        c=boundary_df["theta"],
        s=120,
    )
    axes[0].set_xlabel("n_neurons")
    axes[0].set_ylabel(metric)
    axes[0].set_title(f"{metric} vs N")
    plt.colorbar(sc, ax=axes[0], label="theta")

    sc = axes[1].scatter(
        boundary_df["theta"],
        boundary_df[metric],
        c=boundary_df["ei_ratio"],
        s=120,
    )
    axes[1].set_xlabel("theta")
    axes[1].set_ylabel(metric)
    axes[1].set_title(f"{metric} vs theta")
    plt.colorbar(sc, ax=axes[1], label="ei_ratio")

    sc = axes[2].scatter(
        boundary_df["ei_ratio"],
        boundary_df[metric],
        c=boundary_df["n_neurons"],
        s=120,
    )
    axes[2].set_xlabel("ei_ratio")
    axes[2].set_ylabel(metric)
    axes[2].set_title(f"{metric} vs EI ratio")
    plt.colorbar(sc, ax=axes[2], label="N")

    for ax in axes:
        ax.grid(alpha=0.25)

    savefig(fig, f"boundary_scatter_{metric}.png")
    plt.show()


# ------------------------------------------------------------
# 9.3 Heatmaps per theta/ei_ratio: phi × N
# ------------------------------------------------------------

def plot_heatmap_phi_N(df, metric, theta, ei_ratio):
    d = df[
        (df["theta"] == theta)
        & (df["ei_ratio"] == ei_ratio)
    ].copy()

    if len(d) == 0:
        return

    mat = (
        d.pivot(index="n_neurons", columns="phi", values=metric)
        .sort_index()
        .sort_index(axis=1)
    )

    vals = mat.to_numpy(float)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    im = ax.imshow(
        vals,
        origin="lower",
        aspect="auto",
    )

    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels([f"{x:g}" for x in mat.columns])

    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels([str(int(y)) for y in mat.index])

    ax.set_xlabel("Φ / slope")
    ax.set_ylabel("n_neurons")
    ax.set_title(f"{metric} | theta={theta:g}, ei_ratio={ei_ratio:g}")

    for yi in range(mat.shape[0]):
        for xi in range(mat.shape[1]):
            val = mat.iloc[yi, xi]
            if np.isfinite(val):
                ax.text(
                    xi,
                    yi,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)

    savefig(fig, f"heatmap_{metric}_theta{theta:g}_eir{ei_ratio:g}.png")
    plt.show()


for theta in sorted(summary["theta"].unique()):
    for ei_ratio in sorted(summary["ei_ratio"].unique()):
        for metric in ["MV_alpha_mean", "TAU_beta_mean", "score_mean", "mean_rate_hz", "mean_out_degree"]:
            plot_heatmap_phi_N(summary, metric, theta, ei_ratio)


# ------------------------------------------------------------
# 9.4 Best conditions
# ------------------------------------------------------------

best = summary.sort_values("score_mean").head(50)

best_cols = [
    "phi",
    "theta",
    "ei_ratio",
    "n_neurons",
    "MV_alpha_mean",
    "TAU_beta_mean",
    "KURT_gamma_mean",
    "score_mean",
    "mean_rate_hz",
    "pop_rate_std_hz",
    "mean_out_degree",
    "n",
]

print("\nTop 50 conditions:")
print(best[best_cols].round(4).to_string(index=False))

best.to_csv(PLOTDIR / "top_50_conditions.csv", index=False)

print("\nSaved:", PLOTDIR / "top_50_conditions.csv")
print("\nDone. Outputs saved to:")
print(OUTDIR)