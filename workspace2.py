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
start_dic




# ============================================================
# FULL TOPOLOGY-MATCHED PAIRED TEST:
# HM vs random / small-world / scale-free
# Random neuron addition
# DR + response span + communicability
# 20 matched baseline network instantiations
# ============================================================
#
# Assumes ONLY:
#   - automata_EI_hiermod
#   - start_dic
#
# Main design:
#   For each seed_i:
#       1. Build ONE canonical HM baseline network.
#       2. Build matched topology baselines with same N and same edge count:
#           - hm
#           - random
#           - small_world
#           - scale_free
#       3. For each topology baseline:
#           - add random neurons at each add_neuron_frac
#           - new-involving edges sampled at that topology baseline density
#           - preserve old-old edges exactly
#
# DR protocol:
#   Same as previous HM-only random-neuron run:
#       - 20 seeds
#       - same stimulus grid
#       - same burn-in / response / stim duration
#       - explicit no-stim condition gives F0
#       - S10/S90 interpolated in log10 stimulus space
#
# Main comparisons:
#   1. Absolute DR by topology and growth level
#   2. DR change relative to each topology's own baseline
#   3. DR ratio relative to each topology's own baseline
#   4. Baseline shift relative to same-seed HM baseline
#   5. Max DR achieved by topology
#   6. Response span / F0 / Fmax / AUC / gain
#   7. Communicability proxy
#
# Outputs:
#   - topo_dr_trials
#   - topo_network_metrics
#   - topo_by_network
#   - topo_paired_by_network
#   - topo_summary
#   - topo_response_curves_by_network
#   - topo_response_curves
#   - topo_max_summary
#
# ============================================================

import os
import json
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from scipy import sparse


# ============================================================
# 0. Settings
# ============================================================

TOPOLOGY_TYPES = [
    "hm",
    "random",
    "small_world",
    "scale_free",
]

ADD_NEURON_FRACS_TOPO = np.array([
    0.00,
    0.05,
    0.10,
    0.20,
    0.40,
], dtype=float)

# Same explicit F0 + positive stimulus grid as previous HM-only run.
STIM_RATES_HZ_TOPO = np.array([
    0.0,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    3e-1,
    1.0,
    10 ** 0.5,
], dtype=float)

N_SEEDS_TOPO = 20
N_TRIALS_TOPO = 1

# Match previous HM-only run.
BASE_SEED_TOPO = 91000

# Same protocol as previous HM-only paired test.
BURN_IN_TOPO = 0.50
RESPONSE_DURATION_TOPO = 1.00
STIM_DURATION_TOPO = 0.50

# DR thresholds.
TARGET_X_LOW = 0.10
TARGET_X_HIGH = 0.90

# Reliability flags. Raw DR is still stored regardless.
MIN_ABS_SPAN_HZ = 0.25
MIN_FRAC_SPAN = 0.05

# Communicability proxy.
DO_COMMUNICABILITY = True
COMM_USE_DEGREE_NORMALIZATION = True
COMM_SUBTRACT_DIAGONAL = True
COMM_K_MAX = 3

# Topology parameters.
SMALL_WORLD_REWIRE_P = 0.10
SCALE_FREE_ALPHA_OUT = 1.8
SCALE_FREE_ALPHA_IN = 1.8

SAVE_TOPO = True
OUTDIR_TOPO = "/home/dburrows/DATA/BLNDEV-WILDTYPE/proper_topology_dr_comm_n20"

print("FULL topology-matched paired DR + response-span + communicability test")
print("Topologies:", TOPOLOGY_TYPES)
print("Fractions:", ADD_NEURON_FRACS_TOPO)
print("Stim rates:", STIM_RATES_HZ_TOPO)
print("Seeds:", N_SEEDS_TOPO)
print("Trials per stim:", N_TRIALS_TOPO)
print("Burn-in:", BURN_IN_TOPO)
print("Response duration:", RESPONSE_DURATION_TOPO)
print("Stim duration:", STIM_DURATION_TOPO)
print("Communicability:", DO_COMMUNICABILITY)
print("Communicability k_max:", COMM_K_MAX)
print("Output dir:", OUTDIR_TOPO)

expected_condition_networks = (
    N_SEEDS_TOPO
    * len(TOPOLOGY_TYPES)
    * len(ADD_NEURON_FRACS_TOPO)
)

expected_evoked = (
    expected_condition_networks
    * len(STIM_RATES_HZ_TOPO)
    * N_TRIALS_TOPO
)

print("")
print("Expected condition networks:", expected_condition_networks)
print("Expected evoked simulations:", expected_evoked)


# ============================================================
# 1. Required object check
# ============================================================

_REQUIRED = [
    "automata_EI_hiermod",
    "start_dic",
]

_missing = [x for x in _REQUIRED if x not in globals()]
if len(_missing) > 0:
    raise NameError(
        "Missing required existing functions/objects:\n"
        + "\n".join([f"  - {x}" for x in _missing])
    )


# ============================================================
# 2. Generic helpers
# ============================================================

def sem(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) <= 1:
        return np.nan
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def remove_self_edges_local(A):
    A = np.asarray(A, dtype=bool).copy()
    np.fill_diagonal(A, False)
    return A


def edge_count_density(A):
    A = remove_self_edges_local(A)
    N = int(A.shape[0])
    E = int(A.sum())
    density = float(E / max(N * (N - 1), 1))
    mean_out = float(A.sum(axis=1).mean())
    mean_in = float(A.sum(axis=0).mean())
    return E, density, mean_out, mean_in


def make_start_model_local(seed, p_ext=None):
    """
    Build your exact canonical automata_EI_hiermod baseline.
    Used only to generate the canonical HM baseline adjacency.
    """
    pars = dict(start_dic)
    pars.pop("T", None)

    if p_ext is not None:
        pars["p_ext"] = float(p_ext)

    pars["seed"] = int(seed)

    return automata_EI_hiermod(**pars)


def make_model_from_A_fast(A_fixed, e_fixed, seed, p_ext=None):
    """
    Fast model construction without rebuilding hierarchical A.

    Manually creates an automata_EI_hiermod object with fields needed by .step().

    E rows use start_dic["e_w"].
    I rows use start_dic["i_w"].
    """
    if p_ext is None:
        p_ext = float(start_dic["p_ext"])

    A_fixed = remove_self_edges_local(A_fixed).astype(np.uint8)

    model = automata_EI_hiermod.__new__(automata_EI_hiermod)

    model.rng = np.random.default_rng(int(seed))

    model.n = int(A_fixed.shape[0])
    model.e = int(e_fixed)
    model.i = int(model.n - model.e)

    model.e_w = float(start_dic["e_w"])
    model.i_w = float(start_dic["i_w"])
    model.theta = float(start_dic["theta"])
    model.slope = float(start_dic["slope"])
    model.p_ext = float(p_ext)

    model.refractory_steps = int(start_dic["refractory_steps"])
    model.dt = float(start_dic["dt"])

    model.A = A_fixed
    model.A_e = A_fixed[:model.e, :]
    model.A_i = A_fixed[model.e:, :]

    model.state = np.zeros(model.n, dtype=np.int16)

    return model


def directed_edges_from_offdiag_indices(idx, N):
    """
    Map indices in [0, N*(N-1)) to directed off-diagonal pairs.
    """
    idx = np.asarray(idx, dtype=np.int64)
    rows = idx // (N - 1)
    cols = idx % (N - 1)
    cols = cols + (cols >= rows)
    return rows.astype(np.int64), cols.astype(np.int64)


def force_exact_edge_count(A, target_edges, rng):
    """
    Add/drop random directed off-diagonal edges until exact target edge count.
    """
    A = remove_self_edges_local(A).astype(bool)
    N = int(A.shape[0])
    target_edges = int(target_edges)

    max_edges = N * (N - 1)
    target_edges = int(min(max(target_edges, 0), max_edges))

    current_edges = int(A.sum())

    if current_edges == target_edges:
        return A

    if current_edges > target_edges:
        edge_idx = np.flatnonzero(A.ravel())
        drop_n = current_edges - target_edges
        drop = rng.choice(edge_idx, size=drop_n, replace=False)
        A.ravel()[drop] = False
        np.fill_diagonal(A, False)
        return A

    need = target_edges - current_edges
    possible = N * (N - 1)

    # Add until exact. This is robust for sparse/moderate graphs.
    while need > 0:
        chunk = int(max(need * 5, 10_000))
        chunk = min(chunk, possible)

        idx = rng.choice(possible, size=chunk, replace=False)
        ii, jj = directed_edges_from_offdiag_indices(idx, N)

        missing = ~A[ii, jj]
        ii = ii[missing]
        jj = jj[missing]

        if len(ii) == 0:
            continue

        take = min(need, len(ii))
        A[ii[:take], jj[:take]] = True
        np.fill_diagonal(A, False)

        need = target_edges - int(A.sum())

    np.fill_diagonal(A, False)
    return A


# ============================================================
# 3. Topology builders
# ============================================================

def build_random_directed_A(N, target_edges, seed):
    """
    Directed ER-like graph with exact target edge count.
    """
    rng = np.random.default_rng(int(seed))

    N = int(N)
    target_edges = int(target_edges)

    max_edges = N * (N - 1)
    target_edges = min(target_edges, max_edges)

    idx = rng.choice(max_edges, size=target_edges, replace=False)
    ii, jj = directed_edges_from_offdiag_indices(idx, N)

    A = np.zeros((N, N), dtype=bool)
    A[ii, jj] = True
    A = remove_self_edges_local(A)

    return A


def build_small_world_directed_A(
    N,
    target_edges,
    seed,
    rewire_p=SMALL_WORLD_REWIRE_P,
):
    """
    Directed small-world-like graph:
      - start from directed ring lattice with approximately target_edges
      - rewire edges with probability rewire_p
      - force exact edge count at end
    """
    rng = np.random.default_rng(int(seed))

    N = int(N)
    target_edges = int(target_edges)

    A = np.zeros((N, N), dtype=bool)

    k_out = max(1, int(round(target_edges / max(N, 1))))
    k_out = min(k_out, N - 1)

    src = np.arange(N)

    for shift in range(1, k_out + 1):
        dst = (src + shift) % N
        A[src, dst] = True

    A = remove_self_edges_local(A)

    ii, jj = np.where(A)
    n_edges = len(ii)

    rewire_mask = rng.random(n_edges) < float(rewire_p)
    rewire_indices = np.where(rewire_mask)[0]

    for idx in rewire_indices:
        src_i = int(ii[idx])
        old_dst = int(jj[idx])

        A[src_i, old_dst] = False

        # Try to replace with another destination.
        replaced = False
        for _ in range(100):
            new_dst = int(rng.integers(0, N))
            if new_dst != src_i and not A[src_i, new_dst]:
                A[src_i, new_dst] = True
                replaced = True
                break

        # If no replacement found, leave it to force_exact_edge_count().
        if not replaced:
            pass

    A = force_exact_edge_count(A, target_edges=target_edges, rng=rng)
    A = remove_self_edges_local(A)

    return A


def build_scale_free_static_directed_A(
    N,
    target_edges,
    seed,
    alpha_out=SCALE_FREE_ALPHA_OUT,
    alpha_in=SCALE_FREE_ALPHA_IN,
):
    """
    Static directed scale-free-like graph with exact target edge count.

    Heavy-tailed out-weights and in-weights are sampled from Pareto distributions.
    Edges are then sampled according to source/destination weights.
    """
    rng = np.random.default_rng(int(seed))

    N = int(N)
    target_edges = int(target_edges)

    max_edges = N * (N - 1)
    target_edges = min(target_edges, max_edges)

    out_w = rng.pareto(float(alpha_out), size=N) + 1.0
    in_w = rng.pareto(float(alpha_in), size=N) + 1.0

    out_w = out_w / out_w.sum()
    in_w = in_w / in_w.sum()

    A = np.zeros((N, N), dtype=bool)

    while int(A.sum()) < target_edges:
        need = target_edges - int(A.sum())
        chunk = int(max(need * 5, 10_000))

        src = rng.choice(N, size=chunk, replace=True, p=out_w)
        dst = rng.choice(N, size=chunk, replace=True, p=in_w)

        keep = src != dst
        src = src[keep]
        dst = dst[keep]

        A[src, dst] = True
        np.fill_diagonal(A, False)

    A = force_exact_edge_count(A, target_edges=target_edges, rng=rng)
    A = remove_self_edges_local(A)

    return A


def build_topology_A_from_hm_baseline(A_hm, topology, seed):
    """
    Build topology-matched A with:
      - same N as HM baseline
      - same directed edge count as HM baseline
    """
    A_hm = remove_self_edges_local(A_hm)
    N = int(A_hm.shape[0])
    target_edges = int(A_hm.sum())

    topology = str(topology)

    if topology == "hm":
        A = A_hm.copy()

    elif topology == "random":
        A = build_random_directed_A(
            N=N,
            target_edges=target_edges,
            seed=int(seed),
        )

    elif topology == "small_world":
        A = build_small_world_directed_A(
            N=N,
            target_edges=target_edges,
            seed=int(seed),
            rewire_p=SMALL_WORLD_REWIRE_P,
        )

    elif topology == "scale_free":
        A = build_scale_free_static_directed_A(
            N=N,
            target_edges=target_edges,
            seed=int(seed),
            alpha_out=SCALE_FREE_ALPHA_OUT,
            alpha_in=SCALE_FREE_ALPHA_IN,
        )

    else:
        raise ValueError(f"Unknown topology: {topology}")

    A = remove_self_edges_local(A)

    actual_edges = int(A.sum())
    if actual_edges != target_edges:
        raise RuntimeError(
            f"Topology {topology} has wrong edge count: "
            f"{actual_edges} != {target_edges}"
        )

    return A


# ============================================================
# 4. Communicability proxy
# ============================================================

def compute_communicability_metrics(A, k_max=COMM_K_MAX):
    """
    Fast truncated communicability proxy.

    Approximates:
        exp(B) @ 1

    using:
        1 + B1 + B^2 1 / 2! + ... + B^k 1 / k!

    where:
        B = A / mean_out_degree

    Main paired metric:
        communicability_mean_offdiag
    """
    A = remove_self_edges_local(A).astype(np.float32)
    N = int(A.shape[0])

    edge_count = int(A.sum())
    mean_out_degree = float(A.sum(axis=1).mean())
    mean_in_degree = float(A.sum(axis=0).mean())
    density = float(edge_count / max(N * (N - 1), 1))

    out_degrees = A.sum(axis=1)
    in_degrees = A.sum(axis=0)

    base = {
        "comm_N": N,
        "comm_edge_count": edge_count,
        "comm_density": density,
        "comm_mean_out_degree": mean_out_degree,
        "comm_mean_in_degree": mean_in_degree,
        "comm_out_degree_std": float(np.std(out_degrees)),
        "comm_in_degree_std": float(np.std(in_degrees)),
        "comm_out_degree_cv": float(np.std(out_degrees) / mean_out_degree) if mean_out_degree > 0 else np.nan,
        "comm_in_degree_cv": float(np.std(in_degrees) / mean_in_degree) if mean_in_degree > 0 else np.nan,
        "communicability_k_max": int(k_max),
    }

    if (
        not DO_COMMUNICABILITY
        or N <= 1
        or edge_count == 0
        or mean_out_degree <= 0
    ):
        base.update({
            "communicability_total": np.nan,
            "communicability_mean_all": np.nan,
            "communicability_mean_offdiag": np.nan,
            "communicability_input_scale": np.nan,
        })
        return base

    scale = mean_out_degree if COMM_USE_DEGREE_NORMALIZATION else 1.0
    scale = float(max(scale, 1e-12))

    B = sparse.csr_matrix(A / scale)
    ones = np.ones(N, dtype=np.float64)

    v_total = ones.copy()
    v_power = ones.copy()
    factorial = 1.0

    for k in range(1, int(k_max) + 1):
        v_power = B @ v_power
        factorial *= k
        v_total += v_power / factorial

    total = float(np.sum(v_total))
    mean_all = float(total / (N * N))

    if COMM_SUBTRACT_DIAGONAL:
        mean_offdiag = float((total - N) / max(N * (N - 1), 1))
    else:
        mean_offdiag = float(total / max(N * (N - 1), 1))

    base.update({
        "communicability_total": total,
        "communicability_mean_all": mean_all,
        "communicability_mean_offdiag": mean_offdiag,
        "communicability_input_scale": scale,
    })

    return base


# ============================================================
# 5. Random neuron addition from fixed topology A
# ============================================================

def add_random_neurons_same_density_from_A(
    A_old,
    e0,
    add_neuron_frac,
    seed,
):
    """
    Random addition from a fixed baseline A.

    New ordering:
        old E
        new E
        old I
        new I

    Dynamics/weights:
        all E rows, old and new, use global e_w
        all I rows, old and new, use global i_w

    New-involving edges:
        sampled at original baseline directed density.
    """
    rng = np.random.default_rng(int(seed))

    A_old = remove_self_edges_local(A_old)

    N0 = int(A_old.shape[0])
    e0 = int(e0)
    i0 = N0 - e0

    n_add = int(round(float(add_neuron_frac) * N0))
    n_add = max(n_add, 0)

    baseline_edge_density = float(A_old.sum() / max(N0 * (N0 - 1), 1))

    if n_add == 0:
        info = {
            "N0": N0,
            "N1": N0,
            "n_add": 0,
            "add_neuron_frac": 0.0,
            "e0": e0,
            "i0": i0,
            "e1": e0,
            "i1": i0,
            "n_add_e": 0,
            "n_add_i": 0,
            "baseline_edge_density": baseline_edge_density,
            "n_edges_old": int(A_old.sum()),
            "n_edges_new_total": int(A_old.sum()),
            "n_edges_added": 0,
            "mean_out_degree_old": float(A_old.sum(axis=1).mean()),
            "mean_out_degree_new": float(A_old.sum(axis=1).mean()),
            "mean_in_degree_new": float(A_old.sum(axis=0).mean()),
            "mean_added_out_degree": np.nan,
            "mean_added_in_degree": np.nan,
        }

        return A_old.copy(), e0, info

    N1 = N0 + n_add

    # In your class, ei_ratio is inhibitory fraction.
    e1 = int(N1 - (N1 * float(start_dic["ei_ratio"])))
    i1 = N1 - e1

    n_add_e = e1 - e0
    n_add_i = i1 - i0

    if n_add_e < 0 or n_add_i < 0:
        raise ValueError(
            f"E/I split went negative: n_add_e={n_add_e}, n_add_i={n_add_i}"
        )

    # Preserve old E/I identity while using the class convention:
    # E rows first, then I rows.
    old_to_new = np.empty(N0, dtype=int)

    old_E_old = np.arange(0, e0)
    old_I_old = np.arange(e0, N0)

    old_E_new = np.arange(0, e0)
    old_I_new = np.arange(e1, e1 + i0)

    old_to_new[old_E_old] = old_E_new
    old_to_new[old_I_old] = old_I_new

    A_new = np.zeros((N1, N1), dtype=bool)

    # Preserve old-old edges exactly under new ordering.
    ii, jj = np.where(A_old)
    A_new[old_to_new[ii], old_to_new[jj]] = True

    # New neuron indices.
    new_E_idx = np.arange(e0, e1)
    new_I_idx = np.arange(e1 + i0, N1)
    added_idx = np.concatenate([new_E_idx, new_I_idx]).astype(int)

    # Candidate directed edges where at least one endpoint is new.
    candidate = np.zeros((N1, N1), dtype=bool)
    candidate[added_idx, :] = True
    candidate[:, added_idx] = True
    np.fill_diagonal(candidate, False)
    candidate[A_new] = False

    random_edges = candidate & (rng.random((N1, N1)) < baseline_edge_density)

    A_new |= random_edges
    A_new = remove_self_edges_local(A_new)

    info = {
        "N0": N0,
        "N1": N1,
        "n_add": int(n_add),
        "add_neuron_frac": float(add_neuron_frac),
        "e0": e0,
        "i0": i0,
        "e1": int(e1),
        "i1": int(i1),
        "n_add_e": int(n_add_e),
        "n_add_i": int(n_add_i),
        "baseline_edge_density": baseline_edge_density,
        "n_edges_old": int(A_old.sum()),
        "n_edges_new_total": int(A_new.sum()),
        "n_edges_added": int(A_new.sum() - A_old.sum()),
        "mean_out_degree_old": float(A_old.sum(axis=1).mean()),
        "mean_out_degree_new": float(A_new.sum(axis=1).mean()),
        "mean_in_degree_new": float(A_new.sum(axis=0).mean()),
        "mean_added_out_degree": float(A_new[added_idx, :].sum(axis=1).mean()),
        "mean_added_in_degree": float(A_new[:, added_idx].sum(axis=0).mean()),
    }

    return A_new, e1, info


# ============================================================
# 6. Evoked simulation
# ============================================================

def run_evoked_response_from_A(
    A_fixed,
    e_fixed,
    stim_rate_hz,
    seed,
    p_ext=None,
):
    if p_ext is None:
        p_ext = float(start_dic["p_ext"])

    model = make_model_from_A_fast(
        A_fixed=A_fixed,
        e_fixed=e_fixed,
        seed=int(seed),
        p_ext=float(p_ext),
    )

    dt = float(model.dt)

    n_burn = int(round(float(BURN_IN_TOPO) / dt))
    n_resp = int(round(float(RESPONSE_DURATION_TOPO) / dt))
    n_stim = int(round(float(STIM_DURATION_TOPO) / dt))

    for _ in range(n_burn):
        model.step()

    p_ext_base = float(model.p_ext)
    p_stim_per_step = float(1.0 - np.exp(-float(stim_rate_hz) * dt))
    p_ext_stim = float(np.clip(p_ext_base + p_stim_per_step, 0.0, 1.0))

    active_count = 0
    pop_rates = np.zeros(n_resp, dtype=float)

    for t in range(n_resp):
        if t < n_stim:
            model.p_ext = p_ext_stim
        else:
            model.p_ext = p_ext_base

        active = model.step()
        n_active = int(active.sum())

        active_count += n_active
        pop_rates[t] = active.mean() / dt

    response_hz = active_count / float(model.n * n_resp * dt)

    return {
        "response_hz": float(response_hz),
        "response_pop_rate_mean_hz": float(pop_rates.mean()),
        "response_pop_rate_std_hz": float(pop_rates.std()),
        "p_stim_per_step": float(p_stim_per_step),
        "p_ext_used": float(p_ext_base),
        "p_ext_plus_stim": float(p_ext_stim),
    }


# ============================================================
# 7. DR / response-curve metrics
# ============================================================

def compute_dr_metrics_from_curve(curve_df):
    """
    Quick DR estimate.

    Uses:
        F0 from stim_rate_hz == 0.
        Fmax from positive stimuli.
        S10/S90 interpolated in log10 stimulus space.

    Also stores:
        response span
        fractional response span
        AUC over log stimulus
        high-low gain
    """
    d = curve_df.copy()
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=["stim_rate_hz", "response_hz_mean"])
    d = d.sort_values("stim_rate_hz")

    zero = d[np.isclose(d["stim_rate_hz"], 0.0)]
    pos = d[d["stim_rate_hz"] > 0].copy()

    if len(zero) > 0:
        F0 = float(zero["response_hz_mean"].mean())
    else:
        F0 = float(d["response_hz_mean"].min()) if len(d) else np.nan

    if len(pos) < 3:
        return {
            "F0": F0,
            "Fmax": np.nan,
            "response_span": np.nan,
            "frac_response_span": np.nan,
            "S_10": np.nan,
            "S_90": np.nan,
            "dynamic_range_db_raw": np.nan,
            "dynamic_range_db": np.nan,
            "reliable_dr": False,
            "auc_logstim": np.nan,
            "gain_low_high": np.nan,
        }

    stim = pos["stim_rate_hz"].to_numpy(float)
    resp = pos["response_hz_mean"].to_numpy(float)

    Fmax = float(np.nanmax(resp))
    response_span = float(Fmax - F0)
    frac_response_span = float(response_span / F0) if F0 > 0 else np.nan

    auc_logstim = float(np.trapz(resp, np.log10(stim)))
    gain_low_high = float(resp[-1] - resp[0])

    if (not np.isfinite(response_span)) or response_span <= 0:
        return {
            "F0": F0,
            "Fmax": Fmax,
            "response_span": response_span,
            "frac_response_span": frac_response_span,
            "S_10": np.nan,
            "S_90": np.nan,
            "dynamic_range_db_raw": np.nan,
            "dynamic_range_db": np.nan,
            "reliable_dr": False,
            "auc_logstim": auc_logstim,
            "gain_low_high": gain_low_high,
        }

    F10 = F0 + TARGET_X_LOW * response_span
    F90 = F0 + TARGET_X_HIGH * response_span

    # Enforce monotonic response for S10/S90 interpolation.
    resp_mono = np.maximum.accumulate(resp)

    def interp_stim_at_response(target):
        if target < resp_mono.min() or target > resp_mono.max():
            return np.nan

        idx = np.where(resp_mono >= target)[0]
        if len(idx) == 0:
            return np.nan

        hi = int(idx[0])

        if hi == 0:
            return float(stim[0])

        lo = hi - 1

        r0 = resp_mono[lo]
        r1 = resp_mono[hi]
        s0 = stim[lo]
        s1 = stim[hi]

        if r1 == r0:
            return float(s1)

        x0 = np.log10(s0)
        x1 = np.log10(s1)

        frac = (target - r0) / (r1 - r0)
        xs = x0 + frac * (x1 - x0)

        return float(10 ** xs)

    S_10 = interp_stim_at_response(F10)
    S_90 = interp_stim_at_response(F90)

    if (
        np.isfinite(S_10)
        and np.isfinite(S_90)
        and S_10 > 0
        and S_90 > 0
        and S_90 > S_10
    ):
        dynamic_range_db_raw = float(10.0 * np.log10(S_90 / S_10))
    else:
        dynamic_range_db_raw = np.nan

    reliable_dr = bool(
        np.isfinite(dynamic_range_db_raw)
        and response_span >= MIN_ABS_SPAN_HZ
        and np.isfinite(frac_response_span)
        and frac_response_span >= MIN_FRAC_SPAN
    )

    dynamic_range_db = dynamic_range_db_raw if reliable_dr else np.nan

    return {
        "F0": F0,
        "Fmax": Fmax,
        "response_span": response_span,
        "frac_response_span": frac_response_span,
        "S_10": S_10,
        "S_90": S_90,
        "dynamic_range_db_raw": dynamic_range_db_raw,
        "dynamic_range_db": dynamic_range_db,
        "reliable_dr": reliable_dr,
        "auc_logstim": auc_logstim,
        "gain_low_high": gain_low_high,
    }


# ============================================================
# 8. Build condition tasks
# ============================================================

condition_tasks = []

for seed_i in range(N_SEEDS_TOPO):
    for topology in TOPOLOGY_TYPES:
        for frac_i, add_frac in enumerate(ADD_NEURON_FRACS_TOPO):
            condition_tasks.append({
                "seed_i": int(seed_i),
                "topology": str(topology),
                "frac_i": int(frac_i),
                "add_neuron_frac": float(add_frac),
            })

total_evoked = len(condition_tasks) * len(STIM_RATES_HZ_TOPO) * N_TRIALS_TOPO

print("")
print("Condition tasks:", len(condition_tasks))
print("Total evoked runs:", total_evoked)


# ============================================================
# 9. Run serial with visible progress
# ============================================================

network_rows = []
trial_rows = []
curve_rows = []
by_network_rows = []

pbar = tqdm(
    total=len(condition_tasks) + total_evoked,
    desc="Topology networks + evoked runs",
)

for seed_i in range(N_SEEDS_TOPO):

    # --------------------------------------------------------
    # Build one canonical HM baseline for this seed.
    # All topology baselines are matched to this HM network's N/E.
    # --------------------------------------------------------
    base_seed = int(BASE_SEED_TOPO + seed_i * 100_000)

    base_model = make_start_model_local(
        seed=base_seed,
        p_ext=float(start_dic["p_ext"]),
    )

    A_hm = remove_self_edges_local(base_model.A)
    e_base = int(base_model.e)

    hm_edges, hm_density, hm_mean_out, hm_mean_in = edge_count_density(A_hm)

    # --------------------------------------------------------
    # Build topology baselines for this seed.
    # --------------------------------------------------------
    base_topology_As = {}

    for topology in TOPOLOGY_TYPES:
        topology_seed = int(
            base_seed
            + 5_000_000
            + 100_000 * TOPOLOGY_TYPES.index(topology)
        )

        A_topo = build_topology_A_from_hm_baseline(
            A_hm=A_hm,
            topology=topology,
            seed=topology_seed,
        )

        base_topology_As[topology] = A_topo

    seed_tasks = [t for t in condition_tasks if int(t["seed_i"]) == seed_i]

    for task in seed_tasks:

        topology = str(task["topology"])
        add_frac = float(task["add_neuron_frac"])
        frac_i = int(task["frac_i"])

        A_base_topo = base_topology_As[topology]
        topo_edges, topo_density, topo_mean_out, topo_mean_in = edge_count_density(A_base_topo)

        perturb_seed = int(
            base_seed
            + 1_000_000 * TOPOLOGY_TYPES.index(topology)
            + frac_i * 10_000
            + 999
        )

        A_fixed, e_fixed, info = add_random_neurons_same_density_from_A(
            A_old=A_base_topo,
            e0=e_base,
            add_neuron_frac=add_frac,
            seed=perturb_seed,
        )

        comm = compute_communicability_metrics(A_fixed)

        network_row = {
            "seed_i": seed_i,
            "base_seed": base_seed,
            "perturb_seed": perturb_seed,
            "topology": topology,
            "add_neuron_frac": add_frac,
            "frac_i": frac_i,

            "e_fixed": int(e_fixed),
            "e_w_used": float(start_dic["e_w"]),
            "i_w_used": float(start_dic["i_w"]),
            "theta_used": float(start_dic["theta"]),
            "p_ext_used_for_evoked": float(start_dic["p_ext"]),

            "hm_baseline_edges": int(hm_edges),
            "hm_baseline_density": float(hm_density),
            "hm_baseline_mean_out_degree": float(hm_mean_out),
            "hm_baseline_mean_in_degree": float(hm_mean_in),

            "topology_baseline_edges": int(topo_edges),
            "topology_baseline_density": float(topo_density),
            "topology_baseline_mean_out_degree": float(topo_mean_out),
            "topology_baseline_mean_in_degree": float(topo_mean_in),

            **info,
            **comm,
        }

        network_rows.append(network_row)
        pbar.update(1)

        # ----------------------------------------------------
        # Evoked response curve
        # ----------------------------------------------------
        trial_rows_this = []

        for stim_i, stim_rate_hz in enumerate(STIM_RATES_HZ_TOPO):
            for trial in range(N_TRIALS_TOPO):

                # Same evoked seed across topologies/fractions/stimuli for matched stochastic drive.
                trial_seed = int(
                    base_seed
                    + stim_i * 1_000
                    + trial * 10
                    + 12345
                )

                out = run_evoked_response_from_A(
                    A_fixed=A_fixed,
                    e_fixed=e_fixed,
                    stim_rate_hz=float(stim_rate_hz),
                    seed=trial_seed,
                    p_ext=float(start_dic["p_ext"]),
                )

                row = {
                    "seed_i": seed_i,
                    "base_seed": base_seed,
                    "perturb_seed": perturb_seed,
                    "trial_seed": trial_seed,

                    "topology": topology,
                    "add_neuron_frac": add_frac,
                    "frac_i": frac_i,
                    "stim_i": int(stim_i),
                    "stim_rate_hz": float(stim_rate_hz),
                    "trial": int(trial),

                    "e_w_used": float(start_dic["e_w"]),
                    "i_w_used": float(start_dic["i_w"]),
                    "theta_used": float(start_dic["theta"]),

                    "hm_baseline_edges": int(hm_edges),
                    "hm_baseline_density": float(hm_density),
                    "topology_baseline_edges": int(topo_edges),
                    "topology_baseline_density": float(topo_density),

                    **out,
                    **info,
                    **comm,
                }

                trial_rows.append(row)
                trial_rows_this.append(row)

                pbar.update(1)

        trials_this = pd.DataFrame(trial_rows_this)

        curve_this = (
            trials_this
            .groupby(
                [
                    "seed_i",
                    "topology",
                    "add_neuron_frac",
                    "stim_rate_hz",
                    "p_stim_per_step",
                ],
                as_index=False,
            )
            .agg(
                response_hz_mean=("response_hz", "mean"),
                response_hz_sem=("response_hz", sem),
                response_hz_std=("response_hz", "std"),
                response_pop_rate_mean_hz=("response_pop_rate_mean_hz", "mean"),
                response_pop_rate_std_hz=("response_pop_rate_std_hz", "mean"),
                n_trials=("response_hz", "count"),

                base_seed=("base_seed", "first"),
                perturb_seed=("perturb_seed", "first"),
                p_ext_used=("p_ext_used", "first"),
                e_w_used=("e_w_used", "first"),
                i_w_used=("i_w_used", "first"),
                theta_used=("theta_used", "first"),

                N0=("N0", "first"),
                N1=("N1", "first"),
                n_add=("n_add", "first"),
                n_edges_added=("n_edges_added", "first"),
                mean_out_degree_old=("mean_out_degree_old", "first"),
                mean_out_degree_new=("mean_out_degree_new", "first"),
                mean_in_degree_new=("mean_in_degree_new", "first"),
                mean_added_out_degree=("mean_added_out_degree", "first"),
                mean_added_in_degree=("mean_added_in_degree", "first"),
                baseline_edge_density=("baseline_edge_density", "first"),

                communicability_total=("communicability_total", "first"),
                communicability_mean_all=("communicability_mean_all", "first"),
                communicability_mean_offdiag=("communicability_mean_offdiag", "first"),
                comm_density=("comm_density", "first"),
                comm_mean_out_degree=("comm_mean_out_degree", "first"),
                comm_mean_in_degree=("comm_mean_in_degree", "first"),
                comm_out_degree_cv=("comm_out_degree_cv", "first"),
                comm_in_degree_cv=("comm_in_degree_cv", "first"),

                hm_baseline_edges=("hm_baseline_edges", "first"),
                hm_baseline_density=("hm_baseline_density", "first"),
                topology_baseline_edges=("topology_baseline_edges", "first"),
                topology_baseline_density=("topology_baseline_density", "first"),
            )
        )

        curve_rows.append(curve_this)

        dr = compute_dr_metrics_from_curve(curve_this)

        by_network_row = {
            "seed_i": seed_i,
            "base_seed": base_seed,
            "perturb_seed": perturb_seed,
            "topology": topology,
            "add_neuron_frac": add_frac,
            "frac_i": frac_i,

            "e_w_used": float(start_dic["e_w"]),
            "i_w_used": float(start_dic["i_w"]),
            "theta_used": float(start_dic["theta"]),
            "p_ext_used_for_evoked": float(start_dic["p_ext"]),

            "hm_baseline_edges": int(hm_edges),
            "hm_baseline_density": float(hm_density),
            "hm_baseline_mean_out_degree": float(hm_mean_out),
            "hm_baseline_mean_in_degree": float(hm_mean_in),

            "topology_baseline_edges": int(topo_edges),
            "topology_baseline_density": float(topo_density),
            "topology_baseline_mean_out_degree": float(topo_mean_out),
            "topology_baseline_mean_in_degree": float(topo_mean_in),

            **dr,
            **info,
            **comm,
        }

        by_network_rows.append(by_network_row)

pbar.close()


# ============================================================
# 10. Collect outputs
# ============================================================

topo_dr_trials = (
    pd.DataFrame(trial_rows)
    .sort_values(["seed_i", "topology", "add_neuron_frac", "stim_rate_hz", "trial"])
    .reset_index(drop=True)
)

topo_network_metrics = (
    pd.DataFrame(network_rows)
    .sort_values(["seed_i", "topology", "add_neuron_frac"])
    .reset_index(drop=True)
)

topo_by_network = (
    pd.DataFrame(by_network_rows)
    .sort_values(["seed_i", "topology", "add_neuron_frac"])
    .reset_index(drop=True)
)

topo_response_curves_by_network = (
    pd.concat(curve_rows, ignore_index=True)
    .sort_values(["seed_i", "topology", "add_neuron_frac", "stim_rate_hz"])
    .reset_index(drop=True)
)

print("")
print("Output shapes:")
print("topo_dr_trials:", topo_dr_trials.shape)
print("topo_network_metrics:", topo_network_metrics.shape)
print("topo_by_network:", topo_by_network.shape)
print("topo_response_curves_by_network:", topo_response_curves_by_network.shape)

display(topo_dr_trials.head())
display(topo_network_metrics.head())
display(topo_by_network.head())
display(topo_response_curves_by_network.head())


# ============================================================
# 11. Paired baseline deltas and ratios
# ============================================================
#
# For each metric:
#   A. topology baseline comparison:
#       metric - same-seed/same-topology/add_frac=0
#
#   B. HM baseline comparison:
#       metric - same-seed/HM/add_frac=0
#
# This gives:
#   - change relative to each topology's own baseline
#   - shift relative to HM baseline
# ============================================================

metric_cols = [
    "dynamic_range_db_raw",
    "dynamic_range_db",
    "response_span",
    "frac_response_span",
    "F0",
    "Fmax",
    "S_10",
    "S_90",
    "auc_logstim",
    "gain_low_high",

    "communicability_mean_offdiag",
    "communicability_mean_all",
    "communicability_total",

    "comm_density",
    "comm_mean_out_degree",
    "comm_mean_in_degree",
    "comm_out_degree_cv",
    "comm_in_degree_cv",

    "mean_out_degree_new",
    "mean_in_degree_new",
    "n_edges_new_total",
]

topo_paired_by_network = topo_by_network.copy()

hm_baseline = (
    topo_by_network
    .loc[
        (topo_by_network["topology"] == "hm")
        & np.isclose(topo_by_network["add_neuron_frac"], 0.0)
    ]
    .set_index("seed_i")
)

topology_baseline = (
    topo_by_network
    .loc[np.isclose(topo_by_network["add_neuron_frac"], 0.0)]
    .set_index(["seed_i", "topology"])
)

for col in metric_cols:
    if col not in topo_paired_by_network.columns:
        continue

    hm_map = hm_baseline[col].to_dict()
    topo_map = topology_baseline[col].to_dict()

    # Same-seed HM baseline.
    topo_paired_by_network[f"{col}_hm_baseline"] = (
        topo_paired_by_network["seed_i"].map(hm_map)
    )

    topo_paired_by_network[f"{col}_shift_vs_hm_baseline"] = (
        topo_paired_by_network[col]
        - topo_paired_by_network[f"{col}_hm_baseline"]
    )

    topo_paired_by_network[f"{col}_ratio_vs_hm_baseline"] = (
        topo_paired_by_network[col]
        / topo_paired_by_network[f"{col}_hm_baseline"].replace(0, np.nan)
    )

    # Same-seed, same-topology baseline.
    topo_paired_by_network[f"{col}_topology_baseline"] = [
        topo_map.get((int(seed_i), str(topology)), np.nan)
        for seed_i, topology in zip(
            topo_paired_by_network["seed_i"],
            topo_paired_by_network["topology"],
        )
    ]

    topo_paired_by_network[f"{col}_increase_vs_topology_baseline"] = (
        topo_paired_by_network[col]
        - topo_paired_by_network[f"{col}_topology_baseline"]
    )

    topo_paired_by_network[f"{col}_ratio_vs_topology_baseline"] = (
        topo_paired_by_network[col]
        / topo_paired_by_network[f"{col}_topology_baseline"].replace(0, np.nan)
    )


# ============================================================
# 12. Summary by topology x neuron-addition fraction
# ============================================================

summary_metric_cols = [
    "dynamic_range_db_raw",
    "dynamic_range_db",
    "response_span",
    "frac_response_span",
    "F0",
    "Fmax",
    "S_10",
    "S_90",
    "auc_logstim",
    "gain_low_high",

    "communicability_mean_offdiag",
    "communicability_mean_all",
    "communicability_total",

    "comm_density",
    "comm_mean_out_degree",
    "comm_mean_in_degree",
    "comm_out_degree_cv",
    "comm_in_degree_cv",

    "mean_out_degree_new",
    "mean_in_degree_new",
    "n_edges_new_total",
]

agg = {}

for col in summary_metric_cols:
    if col not in topo_paired_by_network.columns:
        continue

    agg[col] = (col, "mean")
    agg[f"{col}_sem"] = (col, sem)
    agg[f"{col}_std"] = (col, "std")

    for suffix in [
        "shift_vs_hm_baseline",
        "ratio_vs_hm_baseline",
        "increase_vs_topology_baseline",
        "ratio_vs_topology_baseline",
    ]:
        full_col = f"{col}_{suffix}"
        if full_col in topo_paired_by_network.columns:
            agg[full_col] = (full_col, "mean")
            agg[f"{full_col}_sem"] = (full_col, sem)
            agg[f"{full_col}_std"] = (full_col, "std")

agg.update({
    "N0": ("N0", "mean"),
    "N1": ("N1", "mean"),
    "n_add": ("n_add", "mean"),
    "n_edges_added": ("n_edges_added", "mean"),
    "mean_added_out_degree": ("mean_added_out_degree", "mean"),
    "mean_added_in_degree": ("mean_added_in_degree", "mean"),
    "baseline_edge_density": ("baseline_edge_density", "mean"),
    "hm_baseline_density": ("hm_baseline_density", "mean"),
    "topology_baseline_density": ("topology_baseline_density", "mean"),
    "topology_baseline_mean_out_degree": ("topology_baseline_mean_out_degree", "mean"),
    "reliable_fraction": ("reliable_dr", "mean"),
    "n_seeds": ("seed_i", "nunique"),
})

topo_summary = (
    topo_paired_by_network
    .groupby(["topology", "add_neuron_frac"], as_index=False)
    .agg(**agg)
    .sort_values(["topology", "add_neuron_frac"])
    .reset_index(drop=True)
)

topo_summary["usable_dynamic_range_db"] = (
    topo_summary["dynamic_range_db_raw"]
    * topo_summary["frac_response_span"]
    .replace([np.inf, -np.inf], np.nan)
    .clip(lower=0, upper=1)
)

display(topo_summary)


# ============================================================
# 13. Average response curves
# ============================================================

topo_response_curves = (
    topo_response_curves_by_network
    .groupby(["topology", "add_neuron_frac", "stim_rate_hz", "p_stim_per_step"], as_index=False)
    .agg(
        response_hz_mean=("response_hz_mean", "mean"),
        response_hz_sem=("response_hz_mean", sem),
        response_hz_std_across_seeds=("response_hz_mean", "std"),
        n_seeds=("seed_i", "nunique"),

        N1=("N1", "mean"),
        n_add=("n_add", "mean"),

        communicability_mean_offdiag=("communicability_mean_offdiag", "mean"),
        comm_density=("comm_density", "mean"),
        comm_mean_out_degree=("comm_mean_out_degree", "mean"),
    )
    .sort_values(["topology", "add_neuron_frac", "stim_rate_hz"])
    .reset_index(drop=True)
)

display(topo_response_curves.head())


# ============================================================
# 14. Max-achieved summary by topology
# ============================================================

topo_max_rows = []

for topology, g in topo_summary.groupby("topology"):
    g = g.sort_values("add_neuron_frac").copy()

    baseline = g[np.isclose(g["add_neuron_frac"], 0.0)]
    if len(baseline) == 0:
        continue

    baseline = baseline.iloc[0]

    metric = "dynamic_range_db_raw"

    valid = g.dropna(subset=[metric]).copy()
    if len(valid) == 0:
        continue

    idxmax = valid[metric].idxmax()
    max_row = valid.loc[idxmax]

    topo_max_rows.append({
        "topology": topology,
        "n_seeds": baseline.get("n_seeds", np.nan),

        "baseline_dr": baseline.get(metric, np.nan),
        "baseline_dr_sem": baseline.get(f"{metric}_sem", np.nan),

        "baseline_shift_vs_hm_dr": baseline.get(f"{metric}_shift_vs_hm_baseline", np.nan),
        "baseline_shift_vs_hm_dr_sem": baseline.get(f"{metric}_shift_vs_hm_baseline_sem", np.nan),
        "baseline_ratio_vs_hm_dr": baseline.get(f"{metric}_ratio_vs_hm_baseline", np.nan),
        "baseline_ratio_vs_hm_dr_sem": baseline.get(f"{metric}_ratio_vs_hm_baseline_sem", np.nan),

        "max_dr_achieved": max_row.get(metric, np.nan),
        "max_dr_achieved_sem": max_row.get(f"{metric}_sem", np.nan),
        "add_neuron_frac_at_max_dr": max_row.get("add_neuron_frac", np.nan),

        "absolute_dr_increase_from_topology_baseline": (
            max_row.get(metric, np.nan) - baseline.get(metric, np.nan)
        ),

        "dr_ratio_max_vs_topology_baseline": (
            max_row.get(metric, np.nan) / baseline.get(metric, np.nan)
            if np.isfinite(baseline.get(metric, np.nan)) and baseline.get(metric, np.nan) != 0
            else np.nan
        ),

        "response_span_baseline": baseline.get("response_span", np.nan),
        "response_span_baseline_sem": baseline.get("response_span_sem", np.nan),
        "response_span_max_dr_point": max_row.get("response_span", np.nan),
        "response_span_max_dr_point_sem": max_row.get("response_span_sem", np.nan),

        "response_span_increase_from_topology_baseline": (
            max_row.get("response_span", np.nan) - baseline.get("response_span", np.nan)
        ),

        "F0_baseline": baseline.get("F0", np.nan),
        "F0_max_dr_point": max_row.get("F0", np.nan),
        "Fmax_baseline": baseline.get("Fmax", np.nan),
        "Fmax_max_dr_point": max_row.get("Fmax", np.nan),

        "communicability_baseline": baseline.get("communicability_mean_offdiag", np.nan),
        "communicability_baseline_sem": baseline.get("communicability_mean_offdiag_sem", np.nan),
        "communicability_max_dr_point": max_row.get("communicability_mean_offdiag", np.nan),
        "communicability_max_dr_point_sem": max_row.get("communicability_mean_offdiag_sem", np.nan),

        "comm_ratio_max_vs_topology_baseline": (
            max_row.get("communicability_mean_offdiag", np.nan)
            / baseline.get("communicability_mean_offdiag", np.nan)
            if np.isfinite(baseline.get("communicability_mean_offdiag", np.nan))
            and baseline.get("communicability_mean_offdiag", np.nan) != 0
            else np.nan
        ),
    })

topo_max_summary = (
    pd.DataFrame(topo_max_rows)
    .sort_values("topology")
    .reset_index(drop=True)
)

print("")
print("Topology max-achieved summary:")
display(topo_max_summary)


# ============================================================
# 15. Compact comparison tables
# ============================================================

compact_topo_cols = [
    "topology",
    "add_neuron_frac",
    "n_seeds",
    "N1",
    "n_add",
    "n_edges_added",
    "mean_out_degree_new",
    "mean_added_out_degree",
    "mean_added_in_degree",

    "dynamic_range_db_raw",
    "dynamic_range_db_raw_sem",
    "dynamic_range_db_raw_std",
    "dynamic_range_db_raw_increase_vs_topology_baseline",
    "dynamic_range_db_raw_increase_vs_topology_baseline_sem",
    "dynamic_range_db_raw_ratio_vs_topology_baseline",
    "dynamic_range_db_raw_ratio_vs_topology_baseline_sem",
    "dynamic_range_db_raw_shift_vs_hm_baseline",
    "dynamic_range_db_raw_shift_vs_hm_baseline_sem",
    "dynamic_range_db_raw_ratio_vs_hm_baseline",
    "dynamic_range_db_raw_ratio_vs_hm_baseline_sem",

    "response_span",
    "response_span_sem",
    "response_span_increase_vs_topology_baseline",
    "response_span_increase_vs_topology_baseline_sem",
    "response_span_ratio_vs_topology_baseline",
    "response_span_ratio_vs_topology_baseline_sem",

    "F0",
    "F0_sem",
    "F0_ratio_vs_topology_baseline",
    "F0_ratio_vs_topology_baseline_sem",

    "Fmax",
    "Fmax_sem",
    "Fmax_ratio_vs_topology_baseline",
    "Fmax_ratio_vs_topology_baseline_sem",

    "auc_logstim",
    "auc_logstim_sem",
    "auc_logstim_ratio_vs_topology_baseline",
    "auc_logstim_ratio_vs_topology_baseline_sem",

    "gain_low_high",
    "gain_low_high_sem",
    "gain_low_high_ratio_vs_topology_baseline",
    "gain_low_high_ratio_vs_topology_baseline_sem",

    "communicability_mean_offdiag",
    "communicability_mean_offdiag_sem",
    "communicability_mean_offdiag_ratio_vs_topology_baseline",
    "communicability_mean_offdiag_ratio_vs_topology_baseline_sem",

    "reliable_fraction",
]

compact_topo_cols = [c for c in compact_topo_cols if c in topo_summary.columns]

print("")
print("Compact topology x growth table:")
display(topo_summary[compact_topo_cols])

compact_max_cols = [
    "topology",
    "n_seeds",
    "baseline_dr",
    "baseline_dr_sem",
    "baseline_shift_vs_hm_dr",
    "baseline_shift_vs_hm_dr_sem",
    "baseline_ratio_vs_hm_dr",
    "baseline_ratio_vs_hm_dr_sem",
    "max_dr_achieved",
    "max_dr_achieved_sem",
    "add_neuron_frac_at_max_dr",
    "absolute_dr_increase_from_topology_baseline",
    "dr_ratio_max_vs_topology_baseline",
    "response_span_baseline",
    "response_span_max_dr_point",
    "response_span_increase_from_topology_baseline",
    "communicability_baseline",
    "communicability_max_dr_point",
    "comm_ratio_max_vs_topology_baseline",
]

compact_max_cols = [c for c in compact_max_cols if c in topo_max_summary.columns]

print("")
print("Compact max-achieved topology comparison:")
display(topo_max_summary[compact_max_cols])


# ============================================================
# 16. Diagnostic plots
# ============================================================

plot_df = topo_summary.copy()


# ------------------------------------------------------------
# A. Baseline DR shift relative to same-seed HM baseline.
# ------------------------------------------------------------
baseline_plot = plot_df[np.isclose(plot_df["add_neuron_frac"], 0.0)].copy()

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    baseline_plot["topology"],
    baseline_plot["dynamic_range_db_raw_shift_vs_hm_baseline"],
    yerr=baseline_plot.get("dynamic_range_db_raw_shift_vs_hm_baseline_sem", None),
    capsize=3,
)

ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
ax.set_ylabel("Baseline DR shift vs HM baseline, dB")
ax.set_xlabel("Topology")
ax.set_title("Baseline topology effect on dynamic range")
ax.grid(axis="y", alpha=0.25)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# B. DR increase relative to each topology baseline.
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))

for topology, g in plot_df.groupby("topology"):
    g = g.sort_values("add_neuron_frac")

    ax.errorbar(
        g["add_neuron_frac"],
        g["dynamic_range_db_raw_increase_vs_topology_baseline"],
        yerr=g.get("dynamic_range_db_raw_increase_vs_topology_baseline_sem", None),
        marker="o",
        linewidth=2.2,
        capsize=3,
        label=topology,
    )

ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
ax.set_xlabel("Added neurons / original total neurons")
ax.set_ylabel("DR increase vs topology baseline, dB")
ax.set_title("Dynamic-range increase under random neuron addition")
ax.grid(alpha=0.25)
ax.legend(frameon=False)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# C. DR ratio relative to each topology baseline.
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))

for topology, g in plot_df.groupby("topology"):
    g = g.sort_values("add_neuron_frac")

    ax.errorbar(
        g["add_neuron_frac"],
        g["dynamic_range_db_raw_ratio_vs_topology_baseline"],
        yerr=g.get("dynamic_range_db_raw_ratio_vs_topology_baseline_sem", None),
        marker="o",
        linewidth=2.2,
        capsize=3,
        label=topology,
    )

ax.axhline(1, color="black", linestyle="--", linewidth=1.5)
ax.set_xlabel("Added neurons / original total neurons")
ax.set_ylabel("DR / topology baseline")
ax.set_title("Relative dynamic-range change under random neuron addition")
ax.grid(alpha=0.25)
ax.legend(frameon=False)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# D. Absolute DR achieved.
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))

for topology, g in plot_df.groupby("topology"):
    g = g.sort_values("add_neuron_frac")

    ax.errorbar(
        g["add_neuron_frac"],
        g["dynamic_range_db_raw"],
        yerr=g.get("dynamic_range_db_raw_sem", None),
        marker="o",
        linewidth=2.2,
        capsize=3,
        label=topology,
    )

ax.set_xlabel("Added neurons / original total neurons")
ax.set_ylabel("Absolute raw DR, dB")
ax.set_title("Absolute dynamic range achieved by topology")
ax.grid(alpha=0.25)
ax.legend(frameon=False)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# E. Response span increase relative to topology baseline.
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))

for topology, g in plot_df.groupby("topology"):
    g = g.sort_values("add_neuron_frac")

    ax.errorbar(
        g["add_neuron_frac"],
        g["response_span_increase_vs_topology_baseline"],
        yerr=g.get("response_span_increase_vs_topology_baseline_sem", None),
        marker="o",
        linewidth=2.2,
        capsize=3,
        label=topology,
    )

ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
ax.set_xlabel("Added neurons / original total neurons")
ax.set_ylabel("Response span increase vs topology baseline, Hz")
ax.set_title("Response-span increase under random neuron addition")
ax.grid(alpha=0.25)
ax.legend(frameon=False)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# F. Communicability ratio relative to topology baseline.
# ------------------------------------------------------------
if "communicability_mean_offdiag_ratio_vs_topology_baseline" in plot_df.columns:
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for topology, g in plot_df.groupby("topology"):
        g = g.sort_values("add_neuron_frac")

        ax.errorbar(
            g["add_neuron_frac"],
            g["communicability_mean_offdiag_ratio_vs_topology_baseline"],
            yerr=g.get("communicability_mean_offdiag_ratio_vs_topology_baseline_sem", None),
            marker="o",
            linewidth=2.2,
            capsize=3,
            label=topology,
        )

    ax.axhline(1, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Added neurons / original total neurons")
    ax.set_ylabel("Communicability / topology baseline")
    ax.set_title("Communicability change under random neuron addition")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# G. Absolute max DR achieved by topology.
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    topo_max_summary["topology"],
    topo_max_summary["max_dr_achieved"],
    yerr=topo_max_summary.get("max_dr_achieved_sem", None),
    capsize=3,
)

ax.set_ylabel("Max raw DR achieved, dB")
ax.set_xlabel("Topology")
ax.set_title("Maximum dynamic range achieved")
ax.grid(axis="y", alpha=0.25)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# H. Absolute increase from topology baseline to max.
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    topo_max_summary["topology"],
    topo_max_summary["absolute_dr_increase_from_topology_baseline"],
)

ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
ax.set_ylabel("Max DR - topology baseline DR, dB")
ax.set_xlabel("Topology")
ax.set_title("Absolute DR gain from random neuron addition")
ax.grid(axis="y", alpha=0.25)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()


# ============================================================
# 17. Response curves
# ============================================================

for topology in TOPOLOGY_TYPES:
    curve_df = topo_response_curves[topo_response_curves["topology"] == topology].copy()

    if len(curve_df) == 0:
        continue

    fig, ax = plt.subplots(figsize=(8, 5.5))

    fracs = np.sort(curve_df["add_neuron_frac"].unique())

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=float(fracs.min()), vmax=float(fracs.max()))

    for add_frac, g in curve_df.groupby("add_neuron_frac"):
        g = g.sort_values("stim_rate_hz")
        plot_g = g[g["stim_rate_hz"] > 0].copy()

        if len(plot_g) == 0:
            continue

        if np.isclose(add_frac, 0.0):
            color = "black"
            linewidth = 3.0
            label = "baseline"
            zorder = 10
        else:
            color = cmap(norm(add_frac))
            linewidth = 2.0
            label = f"{add_frac:g}"
            zorder = 3

        ax.plot(
            plot_g["stim_rate_hz"],
            plot_g["response_hz_mean"],
            marker="o",
            linewidth=linewidth,
            color=color,
            label=label,
            zorder=zorder,
        )

        if plot_g["response_hz_sem"].notna().any():
            ax.fill_between(
                plot_g["stim_rate_hz"],
                plot_g["response_hz_mean"] - plot_g["response_hz_sem"],
                plot_g["response_hz_mean"] + plot_g["response_hz_sem"],
                color=color,
                alpha=0.12,
                linewidth=0,
                zorder=zorder - 1,
            )

    ax.set_xscale("log")

    vals = curve_df["response_hz_mean"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) > 0 and np.all(vals > 0):
        ax.set_yscale("log")

    ax.set_xlabel("Stimulus rate, Hz")
    ax.set_ylabel("Response rate, Hz")
    ax.set_title(f"Stimulus-response curves: {topology}")
    ax.grid(alpha=0.25)

    smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    smap.set_array([])
    cbar = plt.colorbar(smap, ax=ax)
    cbar.set_label("Added neurons / original total neurons")

    ax.legend(frameon=False, fontsize=8, ncol=2)

    plt.tight_layout()
    plt.show()


# ============================================================
# 18. Sanity check: HM branch table
# ============================================================

hm_check = (
    topo_summary
    .loc[topo_summary["topology"] == "hm"]
    .sort_values("add_neuron_frac")
    .copy()
)

hm_check_cols = [
    "topology",
    "add_neuron_frac",
    "n_seeds",
    "dynamic_range_db_raw",
    "dynamic_range_db_raw_sem",
    "dynamic_range_db_raw_increase_vs_topology_baseline",
    "dynamic_range_db_raw_increase_vs_topology_baseline_sem",
    "dynamic_range_db_raw_ratio_vs_topology_baseline",
    "dynamic_range_db_raw_ratio_vs_topology_baseline_sem",
    "response_span",
    "response_span_sem",
    "response_span_ratio_vs_topology_baseline",
    "response_span_ratio_vs_topology_baseline_sem",
    "communicability_mean_offdiag",
    "communicability_mean_offdiag_ratio_vs_topology_baseline",
    "reliable_fraction",
]

hm_check_cols = [c for c in hm_check_cols if c in hm_check.columns]

print("")
print("HM branch sanity check:")
display(hm_check[hm_check_cols])


# ============================================================
# 19. Quick interpretation helper
# ============================================================

comparison_cols = [
    "topology",
    "add_neuron_frac",
    "n_seeds",

    "dynamic_range_db_raw",
    "dynamic_range_db_raw_sem",
    "dynamic_range_db_raw_increase_vs_topology_baseline",
    "dynamic_range_db_raw_increase_vs_topology_baseline_sem",
    "dynamic_range_db_raw_ratio_vs_topology_baseline",
    "dynamic_range_db_raw_ratio_vs_topology_baseline_sem",

    "response_span",
    "response_span_sem",
    "response_span_increase_vs_topology_baseline",
    "response_span_increase_vs_topology_baseline_sem",
    "response_span_ratio_vs_topology_baseline",
    "response_span_ratio_vs_topology_baseline_sem",

    "communicability_mean_offdiag",
    "communicability_mean_offdiag_sem",
    "communicability_mean_offdiag_ratio_vs_topology_baseline",
    "communicability_mean_offdiag_ratio_vs_topology_baseline_sem",

    "reliable_fraction",
]

comparison_cols = [c for c in comparison_cols if c in topo_summary.columns]

topo_comparison = (
    topo_summary
    .sort_values(["topology", "add_neuron_frac"])
    [comparison_cols]
    .reset_index(drop=True)
)

print("")
print("Topology paired summary:")
display(topo_comparison)


# ============================================================
# 20. Save outputs
# ============================================================

if SAVE_TOPO:
    OUTDIR_TOPO = Path(OUTDIR_TOPO)
    OUTDIR_TOPO.mkdir(parents=True, exist_ok=True)

    outputs_to_save = {
        "topo_dr_trials": topo_dr_trials,
        "topo_network_metrics": topo_network_metrics,
        "topo_by_network": topo_by_network,
        "topo_paired_by_network": topo_paired_by_network,
        "topo_summary": topo_summary,
        "topo_response_curves_by_network": topo_response_curves_by_network,
        "topo_response_curves": topo_response_curves,
        "topo_max_summary": topo_max_summary,
        "topo_comparison": topo_comparison,
    }

    for name, df in outputs_to_save.items():
        csv_path = OUTDIR_TOPO / f"{name}.csv"
        pkl_path = OUTDIR_TOPO / f"{name}.pkl"

        df.to_csv(csv_path, index=False)
        df.to_pickle(pkl_path)

        print(f"Saved {name}:")
        print(f"  CSV: {csv_path}")
        print(f"  PKL: {pkl_path}")

    all_outputs_path = OUTDIR_TOPO / "all_topology_outputs.pkl"

    with open(all_outputs_path, "wb") as f:
        pickle.dump(outputs_to_save, f)

    metadata = {
        "saved_at": datetime.now().isoformat(),
        "outdir": str(OUTDIR_TOPO),

        "TOPOLOGY_TYPES": list(TOPOLOGY_TYPES),
        "ADD_NEURON_FRACS_TOPO": ADD_NEURON_FRACS_TOPO.tolist(),
        "STIM_RATES_HZ_TOPO": STIM_RATES_HZ_TOPO.tolist(),

        "N_SEEDS_TOPO": int(N_SEEDS_TOPO),
        "N_TRIALS_TOPO": int(N_TRIALS_TOPO),
        "BASE_SEED_TOPO": int(BASE_SEED_TOPO),

        "BURN_IN_TOPO": float(BURN_IN_TOPO),
        "RESPONSE_DURATION_TOPO": float(RESPONSE_DURATION_TOPO),
        "STIM_DURATION_TOPO": float(STIM_DURATION_TOPO),

        "TARGET_X_LOW": float(TARGET_X_LOW),
        "TARGET_X_HIGH": float(TARGET_X_HIGH),
        "MIN_ABS_SPAN_HZ": float(MIN_ABS_SPAN_HZ),
        "MIN_FRAC_SPAN": float(MIN_FRAC_SPAN),

        "DO_COMMUNICABILITY": bool(DO_COMMUNICABILITY),
        "COMM_USE_DEGREE_NORMALIZATION": bool(COMM_USE_DEGREE_NORMALIZATION),
        "COMM_SUBTRACT_DIAGONAL": bool(COMM_SUBTRACT_DIAGONAL),
        "COMM_K_MAX": int(COMM_K_MAX),

        "SMALL_WORLD_REWIRE_P": float(SMALL_WORLD_REWIRE_P),
        "SCALE_FREE_ALPHA_OUT": float(SCALE_FREE_ALPHA_OUT),
        "SCALE_FREE_ALPHA_IN": float(SCALE_FREE_ALPHA_IN),

        "n_condition_tasks": int(len(condition_tasks)),
        "n_evoked_runs": int(total_evoked),

        "expected_condition_networks": int(expected_condition_networks),
        "expected_evoked": int(expected_evoked),

        "start_dic": {
            str(k): (
                float(v)
                if isinstance(v, (int, float, np.integer, np.floating))
                else str(v)
            )
            for k, v in start_dic.items()
        },

        "output_shapes": {
            name: {
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
            }
            for name, df in outputs_to_save.items()
        },
    }

    metadata_path = OUTDIR_TOPO / "run_metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("")
    print(f"Saved combined pickle: {all_outputs_path}")
    print(f"Saved metadata: {metadata_path}")

    reload_test = pd.read_csv(OUTDIR_TOPO / "topo_summary.csv")

    print("")
    print("Reload test:")
    print("topo_summary shape:", reload_test.shape)
    display(reload_test.head())

    print("")
    print("DONE. Saved FULL topology comparison outputs to:")
    print(OUTDIR_TOPO)