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
# FULL PAIRED TEST:
# Random neuron addition only
# DR + response span + communicability
#
# Design:
#   - 20 matched baseline network instantiations
#   - 20 random-addition values from 0.05 to 0.40
#   - For each baseline seed:
#       1. baseline network
#       2. same baseline A with random neurons added at each fraction
#
# Counts:
#   - Baseline networks: 20
#   - Random-addition fractions: 20
#   - Random condition networks: 20 seeds x 20 fractions = 400
#   - Total condition networks: 420
#   - Stimulus values: 13
#   - Evoked simulations: 420 x 13 x 1 = 5460
#
# Assumes ONLY:
#   - automata_EI_hiermod
#   - start_dic
#
# Main outputs:
#   - random_dr_trials
#   - random_network_metrics
#   - random_by_network
#   - random_paired_by_network
#   - random_summary
#   - random_paired_summary
#   - random_response_curves_by_network
#   - random_response_curves
#   - random_compact_summary
#   - comparison
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

# One explicit baseline at 0.00.
# Twenty random-addition values from 0.05 to 0.40.
ADD_NEURON_FRACS_RANDOM = np.concatenate([
    np.array([0.00]),
    np.linspace(0.05, 0.40, 20),
]).astype(float)

GROWTH_MODES_RANDOM = [
    "baseline",
    "random",
]

STIM_RATES_HZ_RANDOM = np.array([
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

N_SEEDS_RANDOM = 20
N_TRIALS_RANDOM = 1

BASE_SEED_RANDOM = 91000

BURN_IN_RANDOM = 0.50
RESPONSE_DURATION_RANDOM = 1.00
STIM_DURATION_RANDOM = 0.50

TARGET_X_LOW = 0.10
TARGET_X_HIGH = 0.90

MIN_ABS_SPAN_HZ = 0.25
MIN_FRAC_SPAN = 0.05

COMM_USE_DEGREE_NORMALIZATION = True
COMM_SUBTRACT_DIAGONAL = True
COMM_K_MAX = 3

SAVE_RANDOM = True
OUTDIR_RANDOM = "/home/dburrows/DATA/BLNDEV-WILDTYPE/proper_dr_comm_random_only_n20_20frac_005_040"

print("FULL paired DR + response-span + communicability test: RANDOM ONLY")
print("Fractions including baseline:", ADD_NEURON_FRACS_RANDOM)
print("Random-addition fractions only:", ADD_NEURON_FRACS_RANDOM[ADD_NEURON_FRACS_RANDOM > 0])
print("Number of random-addition fractions:", int(np.sum(ADD_NEURON_FRACS_RANDOM > 0)))
print("Seeds per fraction:", N_SEEDS_RANDOM)
print("Growth modes:", GROWTH_MODES_RANDOM)
print("Stim rates:", STIM_RATES_HZ_RANDOM)
print("Trials per stim:", N_TRIALS_RANDOM)
print("Response duration:", RESPONSE_DURATION_RANDOM)
print("Communicability k_max:", COMM_K_MAX)
print("Output dir:", OUTDIR_RANDOM)

expected_baseline_networks = N_SEEDS_RANDOM
expected_random_condition_networks = N_SEEDS_RANDOM * int(np.sum(ADD_NEURON_FRACS_RANDOM > 0))
expected_total_condition_networks = expected_baseline_networks + expected_random_condition_networks
expected_total_evoked = expected_total_condition_networks * len(STIM_RATES_HZ_RANDOM) * N_TRIALS_RANDOM

print("")
print("Expected baseline networks:", expected_baseline_networks)
print("Expected random condition networks:", expected_random_condition_networks)
print("Expected total condition networks:", expected_total_condition_networks)
print("Expected total evoked simulations:", expected_total_evoked)


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


def make_start_model_local(seed, p_ext=None):
    """
    Build exact canonical automata_EI_hiermod baseline.
    Used only to generate baseline hierarchical A.
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

    New and old neurons use same global scalar weights:
        E rows -> e_w
        I rows -> i_w
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


# ============================================================
# 3. Communicability proxy
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
    """
    A = remove_self_edges_local(A).astype(np.float32)
    N = int(A.shape[0])

    edge_count = int(A.sum())
    mean_out_degree = float(A.sum(axis=1).mean())
    density = float(edge_count / max(N * (N - 1), 1))

    if N <= 1 or edge_count == 0 or mean_out_degree <= 0:
        return {
            "comm_N": N,
            "comm_edge_count": edge_count,
            "comm_density": density,
            "comm_mean_out_degree": mean_out_degree,
            "communicability_total": np.nan,
            "communicability_mean_all": np.nan,
            "communicability_mean_offdiag": np.nan,
            "communicability_input_scale": np.nan,
            "communicability_k_max": int(k_max),
        }

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

    return {
        "comm_N": N,
        "comm_edge_count": edge_count,
        "comm_density": density,
        "comm_mean_out_degree": mean_out_degree,
        "communicability_total": total,
        "communicability_mean_all": mean_all,
        "communicability_mean_offdiag": mean_offdiag,
        "communicability_input_scale": scale,
        "communicability_k_max": int(k_max),
    }


# ============================================================
# 4. Random neuron addition from fixed baseline A
# ============================================================

def add_random_neurons_same_density_from_A(
    A_old,
    e0,
    add_neuron_frac,
    seed,
):
    """
    Random addition from fixed baseline A.

    New ordering:
        old E
        new E
        old I
        new I

    Dynamics/weights:
        all E rows use global e_w
        all I rows use global i_w
    """
    rng = np.random.default_rng(int(seed))

    A_old = np.asarray(A_old, dtype=bool)

    N0 = int(A_old.shape[0])
    e0 = int(e0)
    i0 = N0 - e0

    n_add = int(round(float(add_neuron_frac) * N0))
    n_add = max(n_add, 0)

    baseline_edge_density = float(A_old.sum() / max(N0 * (N0 - 1), 1))

    if n_add == 0:
        info = {
            "growth_mode": "baseline",
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
        "growth_mode": "random",
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


def build_A_for_condition_from_baseline(
    A_base,
    e_base,
    add_neuron_frac,
    growth_mode,
    perturb_seed,
):
    growth_mode = str(growth_mode)
    add_neuron_frac = float(add_neuron_frac)

    if growth_mode == "baseline":
        A_new, e_new, info = add_random_neurons_same_density_from_A(
            A_old=A_base,
            e0=e_base,
            add_neuron_frac=0.0,
            seed=int(perturb_seed),
        )
        info["growth_mode"] = "baseline"

    elif growth_mode == "random":
        A_new, e_new, info = add_random_neurons_same_density_from_A(
            A_old=A_base,
            e0=e_base,
            add_neuron_frac=add_neuron_frac,
            seed=int(perturb_seed),
        )
        info["growth_mode"] = "random"

    else:
        raise ValueError(f"Unknown growth_mode: {growth_mode}")

    return A_new, e_new, info


# ============================================================
# 5. Evoked simulation
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

    n_burn = int(round(float(BURN_IN_RANDOM) / dt))
    for _ in range(n_burn):
        model.step()

    p_ext_base = float(model.p_ext)
    p_stim_per_step = float(1.0 - np.exp(-float(stim_rate_hz) * dt))

    n_resp = int(round(float(RESPONSE_DURATION_RANDOM) / dt))
    n_stim = int(round(float(STIM_DURATION_RANDOM) / dt))

    active_count = 0
    pop_rates = np.zeros(n_resp, dtype=float)

    for t in range(n_resp):
        if t < n_stim:
            model.p_ext = float(np.clip(p_ext_base + p_stim_per_step, 0.0, 1.0))
        else:
            model.p_ext = p_ext_base

        active = model.step()
        active_count += int(active.sum())
        pop_rates[t] = active.mean() / dt

    response_hz = active_count / float(model.n * n_resp * dt)

    return {
        "response_hz": float(response_hz),
        "response_pop_rate_mean_hz": float(pop_rates.mean()),
        "response_pop_rate_std_hz": float(pop_rates.std()),
        "p_stim_per_step": float(p_stim_per_step),
        "p_ext_used": float(p_ext_base),
        "p_ext_plus_stim": float(np.clip(p_ext_base + p_stim_per_step, 0.0, 1.0)),
    }


# ============================================================
# 6. DR / response-curve metrics
# ============================================================

def compute_dr_metrics_from_curve(curve_df):
    """
    Quick DR estimate.

    Uses:
        F0 from stim_rate_hz == 0
        Fmax from positive stimuli
        S10/S90 interpolated in log10 stimulus space
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
# 7. Build condition tasks
# ============================================================

condition_tasks = []

for seed_i in range(N_SEEDS_RANDOM):
    for growth_mode in GROWTH_MODES_RANDOM:
        for frac_i, add_frac in enumerate(ADD_NEURON_FRACS_RANDOM):

            # Only one baseline condition.
            if growth_mode == "baseline" and not np.isclose(add_frac, 0.0):
                continue

            # Avoid duplicate zero-fraction random rows.
            if growth_mode != "baseline" and np.isclose(add_frac, 0.0):
                continue

            condition_tasks.append({
                "seed_i": int(seed_i),
                "frac_i": int(frac_i),
                "growth_mode": str(growth_mode),
                "add_neuron_frac": float(add_frac),
            })

total_evoked = len(condition_tasks) * len(STIM_RATES_HZ_RANDOM) * N_TRIALS_RANDOM

print("")
print("Condition tasks:", len(condition_tasks))
print("Total evoked runs:", total_evoked)


# ============================================================
# 8. Run serial with visible progress
# ============================================================

network_rows = []
trial_rows = []
curve_rows = []
by_network_rows = []

pbar = tqdm(
    total=len(condition_tasks) + total_evoked,
    desc="Random-only networks + evoked runs",
)

for seed_i in range(N_SEEDS_RANDOM):

    # Build one matched baseline network for this seed.
    base_seed = int(BASE_SEED_RANDOM + seed_i * 100_000)

    base_model = make_start_model_local(
        seed=base_seed,
        p_ext=float(start_dic["p_ext"]),
    )

    A_base = remove_self_edges_local(base_model.A)
    e_base = int(base_model.e)

    seed_tasks = [t for t in condition_tasks if int(t["seed_i"]) == seed_i]

    for task in seed_tasks:

        growth_mode = str(task["growth_mode"])
        add_frac = float(task["add_neuron_frac"])
        frac_i = int(task["frac_i"])

        # Same baseline seed.
        # Perturb seed only controls random added-neuron edges.
        perturb_seed = int(
            base_seed
            + frac_i * 10_000
            + {"baseline": 0, "random": 1_000_000}[growth_mode]
            + 999
        )

        A_fixed, e_fixed, info = build_A_for_condition_from_baseline(
            A_base=A_base,
            e_base=e_base,
            add_neuron_frac=add_frac,
            growth_mode=growth_mode,
            perturb_seed=perturb_seed,
        )

        comm = compute_communicability_metrics(A_fixed)

        network_row = {
            "seed_i": seed_i,
            "base_seed": base_seed,
            "perturb_seed": perturb_seed,
            "growth_mode": growth_mode,
            "add_neuron_frac": add_frac,
            "frac_i": frac_i,
            "e_fixed": int(e_fixed),
            "e_w_used": float(start_dic["e_w"]),
            "i_w_used": float(start_dic["i_w"]),
            "theta_used": float(start_dic["theta"]),
            "p_ext_used_for_evoked": float(start_dic["p_ext"]),
            **info,
            **comm,
        }

        network_rows.append(network_row)
        pbar.update(1)

        trial_rows_this = []

        for stim_i, stim_rate_hz in enumerate(STIM_RATES_HZ_RANDOM):
            for trial in range(N_TRIALS_RANDOM):

                # Same evoked seed across baseline/random/fractions for matched noise.
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
                    "growth_mode": growth_mode,
                    "add_neuron_frac": add_frac,
                    "frac_i": frac_i,
                    "stim_i": int(stim_i),
                    "stim_rate_hz": float(stim_rate_hz),
                    "trial": int(trial),
                    "e_w_used": float(start_dic["e_w"]),
                    "i_w_used": float(start_dic["i_w"]),
                    "theta_used": float(start_dic["theta"]),
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
                    "growth_mode",
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
            )
        )

        curve_rows.append(curve_this)

        dr = compute_dr_metrics_from_curve(curve_this)

        by_network_row = {
            "seed_i": seed_i,
            "base_seed": base_seed,
            "perturb_seed": perturb_seed,
            "growth_mode": growth_mode,
            "add_neuron_frac": add_frac,
            "frac_i": frac_i,
            "e_w_used": float(start_dic["e_w"]),
            "i_w_used": float(start_dic["i_w"]),
            "theta_used": float(start_dic["theta"]),
            "p_ext_used_for_evoked": float(start_dic["p_ext"]),
            **dr,
            **info,
            **comm,
        }

        by_network_rows.append(by_network_row)

pbar.close()


# ============================================================
# 9. Collect outputs
# ============================================================

random_dr_trials = (
    pd.DataFrame(trial_rows)
    .sort_values(["seed_i", "growth_mode", "add_neuron_frac", "stim_rate_hz", "trial"])
    .reset_index(drop=True)
)

random_network_metrics = (
    pd.DataFrame(network_rows)
    .sort_values(["seed_i", "growth_mode", "add_neuron_frac"])
    .reset_index(drop=True)
)

random_by_network = (
    pd.DataFrame(by_network_rows)
    .sort_values(["seed_i", "growth_mode", "add_neuron_frac"])
    .reset_index(drop=True)
)

random_response_curves_by_network = (
    pd.concat(curve_rows, ignore_index=True)
    .sort_values(["seed_i", "growth_mode", "add_neuron_frac", "stim_rate_hz"])
    .reset_index(drop=True)
)

print("")
print("Output shapes:")
print("random_dr_trials:", random_dr_trials.shape)
print("random_network_metrics:", random_network_metrics.shape)
print("random_by_network:", random_by_network.shape)
print("random_response_curves_by_network:", random_response_curves_by_network.shape)

display(random_dr_trials.head())
display(random_network_metrics.head())
display(random_by_network.head())
display(random_response_curves_by_network.head())


# ============================================================
# 10. Paired baseline deltas and ratios
# ============================================================

baseline_metrics = (
    random_by_network
    .loc[random_by_network["growth_mode"] == "baseline"]
    .set_index("seed_i")
)

paired_metric_cols = [
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
    "mean_out_degree_new",
    "mean_in_degree_new",
]

random_paired_by_network = random_by_network.copy()

for col in paired_metric_cols:
    if col not in random_paired_by_network.columns:
        continue

    base_map = baseline_metrics[col].to_dict()

    random_paired_by_network[f"{col}_baseline"] = (
        random_paired_by_network["seed_i"].map(base_map)
    )

    random_paired_by_network[f"{col}_delta"] = (
        random_paired_by_network[col]
        - random_paired_by_network[f"{col}_baseline"]
    )

    random_paired_by_network[f"{col}_ratio"] = (
        random_paired_by_network[col]
        / random_paired_by_network[f"{col}_baseline"].replace(0, np.nan)
    )


# ============================================================
# 11. Summary across seeds
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
    "mean_out_degree_new",
    "mean_in_degree_new",
]

agg = {}

for col in summary_metric_cols:
    if col not in random_paired_by_network.columns:
        continue

    agg[col] = (col, "mean")
    agg[f"{col}_sem"] = (col, sem)
    agg[f"{col}_std"] = (col, "std")

    if f"{col}_delta" in random_paired_by_network.columns:
        agg[f"{col}_delta"] = (f"{col}_delta", "mean")
        agg[f"{col}_delta_sem"] = (f"{col}_delta", sem)
        agg[f"{col}_delta_std"] = (f"{col}_delta", "std")

    if f"{col}_ratio" in random_paired_by_network.columns:
        agg[f"{col}_ratio"] = (f"{col}_ratio", "mean")
        agg[f"{col}_ratio_sem"] = (f"{col}_ratio", sem)
        agg[f"{col}_ratio_std"] = (f"{col}_ratio", "std")

agg.update({
    "N0": ("N0", "mean"),
    "N1": ("N1", "mean"),
    "n_add": ("n_add", "mean"),
    "n_edges_added": ("n_edges_added", "mean"),
    "mean_added_out_degree": ("mean_added_out_degree", "mean"),
    "mean_added_in_degree": ("mean_added_in_degree", "mean"),
    "baseline_edge_density": ("baseline_edge_density", "mean"),
    "reliable_fraction": ("reliable_dr", "mean"),
    "n_seeds": ("seed_i", "nunique"),
})

random_summary = (
    random_paired_by_network
    .groupby(["growth_mode", "add_neuron_frac"], as_index=False)
    .agg(**agg)
    .sort_values(["growth_mode", "add_neuron_frac"])
    .reset_index(drop=True)
)

random_paired_summary = random_summary.copy()

random_summary["usable_dynamic_range_db"] = (
    random_summary["dynamic_range_db_raw"]
    * random_summary["frac_response_span"]
    .replace([np.inf, -np.inf], np.nan)
    .clip(lower=0, upper=1)
)

display(random_summary)


# ============================================================
# 12. Average response curves
# ============================================================

random_response_curves = (
    random_response_curves_by_network
    .groupby(["growth_mode", "add_neuron_frac", "stim_rate_hz", "p_stim_per_step"], as_index=False)
    .agg(
        response_hz_mean=("response_hz_mean", "mean"),
        response_hz_sem=("response_hz_mean", sem),
        response_hz_std_across_seeds=("response_hz_mean", "std"),
        n_seeds=("seed_i", "nunique"),
        N1=("N1", "mean"),
        n_add=("n_add", "mean"),
        communicability_mean_offdiag=("communicability_mean_offdiag", "mean"),
    )
    .sort_values(["growth_mode", "add_neuron_frac", "stim_rate_hz"])
    .reset_index(drop=True)
)

display(random_response_curves.head())


# ============================================================
# 13. Compact comparison table
# ============================================================

compact_cols = [
    "growth_mode",
    "add_neuron_frac",
    "n_seeds",
    "n_add",
    "N1",
    "n_edges_added",
    "mean_out_degree_new",
    "mean_added_out_degree",
    "mean_added_in_degree",

    "dynamic_range_db_raw",
    "dynamic_range_db_raw_sem",
    "dynamic_range_db_raw_std",
    "dynamic_range_db_raw_delta",
    "dynamic_range_db_raw_delta_sem",
    "dynamic_range_db_raw_delta_std",
    "dynamic_range_db_raw_ratio",
    "dynamic_range_db_raw_ratio_sem",

    "response_span",
    "response_span_sem",
    "response_span_std",
    "response_span_delta",
    "response_span_delta_sem",
    "response_span_delta_std",
    "response_span_ratio",
    "response_span_ratio_sem",

    "F0_ratio",
    "F0_ratio_sem",
    "Fmax_ratio",
    "Fmax_ratio_sem",
    "auc_logstim_ratio",
    "auc_logstim_ratio_sem",
    "gain_low_high_ratio",
    "gain_low_high_ratio_sem",

    "communicability_mean_offdiag",
    "communicability_mean_offdiag_sem",
    "communicability_mean_offdiag_std",
    "communicability_mean_offdiag_delta",
    "communicability_mean_offdiag_delta_sem",
    "communicability_mean_offdiag_ratio",
    "communicability_mean_offdiag_ratio_sem",

    "reliable_fraction",
]

compact_cols = [c for c in compact_cols if c in random_summary.columns]

random_compact_summary = random_summary[compact_cols].copy()

print("")
print("Compact paired table:")
display(random_compact_summary)


# ============================================================
# 14. Diagnostic plots
# ============================================================

plot_summary = random_summary.copy()
plot_summary = plot_summary[plot_summary["growth_mode"] == "random"].copy()

diagnostic_specs = [
    ("dynamic_range_db_raw_delta", "dynamic_range_db_raw_delta_sem", "Δ raw DR, dB", 0.0),
    ("dynamic_range_db_raw_ratio", "dynamic_range_db_raw_ratio_sem", "Raw DR / baseline", 1.0),
    ("response_span_delta", "response_span_delta_sem", "Δ response span, Hz", 0.0),
    ("response_span_ratio", "response_span_ratio_sem", "Response span / baseline", 1.0),
    ("F0_ratio", "F0_ratio_sem", "F0 / baseline", 1.0),
    ("Fmax_ratio", "Fmax_ratio_sem", "Fmax / baseline", 1.0),
    ("auc_logstim_ratio", "auc_logstim_ratio_sem", "AUC / baseline", 1.0),
    ("gain_low_high_ratio", "gain_low_high_ratio_sem", "High-low gain / baseline", 1.0),
    ("communicability_mean_offdiag_ratio", "communicability_mean_offdiag_ratio_sem", "Communicability / baseline", 1.0),
    ("communicability_mean_offdiag_delta", "communicability_mean_offdiag_delta_sem", "Δ communicability", 0.0),
]

for y_col, sem_col, ylabel, ref in diagnostic_specs:
    if y_col not in plot_summary.columns:
        continue

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    g = plot_summary.sort_values("add_neuron_frac")
    yerr = g[sem_col] if sem_col in g.columns else None

    ax.errorbar(
        g["add_neuron_frac"],
        g[y_col],
        yerr=yerr,
        marker="o",
        linewidth=2.5,
        capsize=3,
        label="random",
    )

    ax.axhline(ref, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Added neurons / original total neurons")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()


# ============================================================
# 15. Response curves
# ============================================================

for mode in ["baseline", "random"]:
    curve_df = random_response_curves[random_response_curves["growth_mode"] == mode].copy()

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

        if mode == "baseline" or np.isclose(add_frac, 0):
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
    ax.set_title(f"Stimulus-response curves: {mode}")
    ax.grid(alpha=0.25)

    if mode != "baseline":
        smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        smap.set_array([])
        cbar = plt.colorbar(smap, ax=ax)
        cbar.set_label("Added neurons / original total neurons")

    ax.legend(frameon=False, fontsize=8, ncol=2)

    plt.tight_layout()
    plt.show()


# ============================================================
# 16. Quick interpretation helper
# ============================================================

comparison_cols = [
    "add_neuron_frac",
    "n_seeds",
    "dynamic_range_db_raw_delta",
    "dynamic_range_db_raw_delta_sem",
    "dynamic_range_db_raw_ratio",
    "response_span_delta",
    "response_span_delta_sem",
    "response_span_ratio",
    "communicability_mean_offdiag_delta",
    "communicability_mean_offdiag_delta_sem",
    "communicability_mean_offdiag_ratio",
]

comparison_cols = [c for c in comparison_cols if c in random_summary.columns]

comparison = (
    random_summary
    .loc[random_summary["growth_mode"] == "random"]
    .sort_values("add_neuron_frac")
    [comparison_cols]
)

print("")
print("Random addition paired summary:")
display(comparison)


# ============================================================
# 17. Save outputs
# ============================================================

if SAVE_RANDOM:
    OUTDIR_RANDOM = Path(OUTDIR_RANDOM)
    OUTDIR_RANDOM.mkdir(parents=True, exist_ok=True)

    outputs_to_save = {
        "random_dr_trials": random_dr_trials,
        "random_network_metrics": random_network_metrics,
        "random_by_network": random_by_network,
        "random_paired_by_network": random_paired_by_network,
        "random_summary": random_summary,
        "random_paired_summary": random_paired_summary,
        "random_response_curves_by_network": random_response_curves_by_network,
        "random_response_curves": random_response_curves,
        "random_compact_summary": random_compact_summary,
        "comparison": comparison,
    }

    for name, df in outputs_to_save.items():
        csv_path = OUTDIR_RANDOM / f"{name}.csv"
        pkl_path = OUTDIR_RANDOM / f"{name}.pkl"

        df.to_csv(csv_path, index=False)
        df.to_pickle(pkl_path)

        print(f"Saved {name}:")
        print(f"  CSV: {csv_path}")
        print(f"  PKL: {pkl_path}")

    all_outputs_path = OUTDIR_RANDOM / "all_random_outputs.pkl"

    with open(all_outputs_path, "wb") as f:
        pickle.dump(outputs_to_save, f)

    print("")
    print(f"Saved combined output pickle: {all_outputs_path}")

    metadata = {
        "saved_at": datetime.now().isoformat(),
        "outdir": str(OUTDIR_RANDOM),

        "ADD_NEURON_FRACS_RANDOM": ADD_NEURON_FRACS_RANDOM.tolist(),
        "random_addition_values_only": ADD_NEURON_FRACS_RANDOM[ADD_NEURON_FRACS_RANDOM > 0].tolist(),
        "n_random_addition_values": int(np.sum(ADD_NEURON_FRACS_RANDOM > 0)),

        "GROWTH_MODES_RANDOM": list(GROWTH_MODES_RANDOM),
        "STIM_RATES_HZ_RANDOM": STIM_RATES_HZ_RANDOM.tolist(),

        "N_SEEDS_RANDOM": int(N_SEEDS_RANDOM),
        "N_TRIALS_RANDOM": int(N_TRIALS_RANDOM),
        "BASE_SEED_RANDOM": int(BASE_SEED_RANDOM),

        "BURN_IN_RANDOM": float(BURN_IN_RANDOM),
        "RESPONSE_DURATION_RANDOM": float(RESPONSE_DURATION_RANDOM),
        "STIM_DURATION_RANDOM": float(STIM_DURATION_RANDOM),

        "TARGET_X_LOW": float(TARGET_X_LOW),
        "TARGET_X_HIGH": float(TARGET_X_HIGH),
        "MIN_ABS_SPAN_HZ": float(MIN_ABS_SPAN_HZ),
        "MIN_FRAC_SPAN": float(MIN_FRAC_SPAN),

        "COMM_USE_DEGREE_NORMALIZATION": bool(COMM_USE_DEGREE_NORMALIZATION),
        "COMM_SUBTRACT_DIAGONAL": bool(COMM_SUBTRACT_DIAGONAL),
        "COMM_K_MAX": int(COMM_K_MAX),

        "n_condition_tasks": int(len(condition_tasks)),
        "n_evoked_runs": int(total_evoked),

        "expected_baseline_networks": int(expected_baseline_networks),
        "expected_random_condition_networks": int(expected_random_condition_networks),
        "expected_total_condition_networks": int(expected_total_condition_networks),
        "expected_total_evoked_simulations": int(expected_total_evoked),

        "actual_random_dr_trials_rows": int(random_dr_trials.shape[0]),
        "actual_network_metric_rows": int(random_network_metrics.shape[0]),
        "actual_by_network_rows": int(random_by_network.shape[0]),

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

    metadata_path = OUTDIR_RANDOM / "run_metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata: {metadata_path}")

    reload_test = pd.read_csv(OUTDIR_RANDOM / "random_summary.csv")

    print("")
    print("Reload test:")
    print("random_summary shape:", reload_test.shape)
    display(reload_test.head())

    print("")
    print("DONE. Saved all outputs to:")
    print(OUTDIR_RANDOM)