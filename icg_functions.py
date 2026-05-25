import numpy as np
import pandas as pd
from scipy.stats import linregress

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
      pairs: (P, 2) int32 with i<j-
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
    """
    Returns:
      labs: labels per level, same as before
      groupings: actual nested neuron groupings per level
    """
    # level 0: pairs
    perm0 = rng.permutation(n_neurons)
    n_used = n_neurons - (n_neurons % 2)
    curr_groups = perm0[:n_used].reshape(-1, 2)

    groupings = [curr_groups.copy()]

    # higher levels: pair previous groups, preserving nested structure
    while len(curr_groups) > 1:
        perm = rng.permutation(len(curr_groups))
        if len(perm) % 2:
            perm = perm[:-1]

        paired = perm.reshape(-1, 2)
        curr_groups = curr_groups[paired].reshape(len(paired), -1)

        groupings.append(curr_groups.copy())

    # labels for compatibility
    labs = []
    for g in groupings:
        lab = np.full(n_neurons, -1, dtype=int)
        for k, row in enumerate(g):
            lab[row] = k
        labs.append(lab)

    return labs, groupings


def wire_A(n_neurons, groupings, t, rng):
    """
    Munn-style hierarchical-modular wiring:
      - level 0 pairs are always connected
      - higher-level edges are added between nested child modules
      - previous edges are never erased
    """
    A = np.zeros((n_neurons, n_neurons), dtype=np.uint8)

    # level 0: force pair edges
    pairs = groupings[0]

    A[pairs[:, 0], pairs[:, 1]] = 1
    A[pairs[:, 1], pairs[:, 0]] = 1

    # higher levels
    for level_idx, groups in enumerate(groupings[1:], start=1):
        P = 1.0 / (float(t) ** level_idx)

        for row in groups:
            half = len(row) // 2

            left = row[:half]
            right = row[half:]

            ii, jj = np.meshgrid(left, right, indexing="ij")
            comb_v = np.column_stack([ii.ravel(), jj.ravel()])

            fwd = (rng.random(comb_v.shape[0]) < P).astype(np.uint8)
            rev = (rng.random(comb_v.shape[0]) < P).astype(np.uint8)

            # ADD edges, never overwrite existing ones
            A[comb_v[:, 0], comb_v[:, 1]] = np.maximum(
                A[comb_v[:, 0], comb_v[:, 1]],
                fwd,
            )

            A[comb_v[:, 1], comb_v[:, 0]] = np.maximum(
                A[comb_v[:, 1], comb_v[:, 0]],
                rev,
            )

    return A

class automata_EI_hiermod:
    def __init__(
        self,
        n_neurons=1000,
        ei_ratio=0.2,
        e_w=0.8,
        i_w=0.8,
        theta=1.5,
        phi=2,
        p_ext=0.001,
        refractory_steps=2,
        dt=0.01,
        seed=0,
    ):
        self.rng = np.random.default_rng(seed)

        self.n = int(n_neurons)
        self.e = int(n_neurons - (n_neurons * ei_ratio))
        self.i = int(n_neurons * ei_ratio)

        self.e_w = float(e_w)
        self.i_w = float(i_w)
        self.theta = float(theta)
        self.phi = float(phi)
        self.p_ext = float(p_ext)
        self.refractory_steps = int(refractory_steps)
        self.dt = float(dt)

        self.labs, self.groupings = label_neuron_bylevel(self.n, rng=self.rng)

        self.A = wire_A(
            n_neurons=self.n,
            groupings=self.groupings,
            t=self.phi,
            rng=self.rng,
        )

        self.A_e = self.A[:self.e, :]
        self.A_i = self.A[self.e:, :]

        self.state = np.zeros(self.n, dtype=np.int16)

    def step(self):
        active = self.state == 1

        inp_e = self.A_e.T @ active[:self.e].astype(np.float32)
        inp_i = self.A_i.T @ active[self.e:].astype(np.float32)

        net = (self.e_w * inp_e) - (self.i_w * inp_i)
        p_net = 1 / (1 + np.exp(-(net - self.theta)))

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


def compute_icg_metrics(
    spikes,
    dt,
    max_lag_s=3.0,
    fit_exclude_first=True,
    fit_exclude_last=True,
):
    generations, labels_by_gen, sum_traces_by_gen, sizes_by_gen = icg_greedy_fast(spikes)
    
    rows_by_gen = []

    for gen, Xg, sizes in zip(generations[:-2], sum_traces_by_gen[:-2], sizes_by_gen[:-2]):
        Xg = np.asarray(Xg)
        sizes = np.asarray(sizes)

        mean_cluster_size = float(np.mean(sizes))
        mean_activity = float(np.mean(Xg))
        mean_variance = float(np.mean(np.var(Xg, axis=1)))

        tau = float(timescale(Xg, max_lag=max_lag_s, dt=dt))
        kcorr = float(kurtosis_corr(Xg))

        rows_by_gen.append({
            "gen": int(gen),
            "n_clusters": int(Xg.shape[0]),
            "mean_cluster_size": mean_cluster_size,
            "mean_activity": mean_activity,
            "mean_variance": mean_variance,
            "timescale": tau,
            "corr_kurtosis": kcorr,
        })

    gen_df = pd.DataFrame(rows_by_gen)

    mv0 = float(gen_df.loc[gen_df["gen"].idxmin(), "mean_variance"])
    tau0 = float(gen_df.loc[gen_df["gen"].idxmin(), "timescale"])

    gen_df["mean_variance_l0"] = mv0
    gen_df["timescale_l0"] = tau0

    gen_df["mean_variance_norm"] = gen_df["mean_variance"] / mv0
    gen_df["timescale_norm"] = gen_df["timescale"] / tau0

    gen_df.loc[~np.isfinite(gen_df["mean_variance_norm"]), "mean_variance_norm"] = np.nan
    gen_df.loc[~np.isfinite(gen_df["timescale_norm"]), "timescale_norm"] = np.nan

    mv_alpha, mv_r2, mv_p = safe_loglog_slope(
        gen_df["mean_cluster_size"],
        gen_df["mean_variance"],
        exclude_first=fit_exclude_first,
        exclude_last=fit_exclude_last,
    )
    
    ts_beta, ts_r2, ts_p = safe_loglog_slope(
        gen_df["mean_cluster_size"],
        gen_df["timescale"],
        exclude_first=fit_exclude_first,
        exclude_last=fit_exclude_last,
    )
    
    mv_alpha_norm, mv_r2_norm, mv_p_norm = safe_loglog_slope(
        gen_df["mean_cluster_size"],
        gen_df["mean_variance_norm"],
        exclude_first=fit_exclude_first,
        exclude_last=fit_exclude_last,
    )
    
    ts_beta_norm, ts_r2_norm, ts_p_norm = safe_loglog_slope(
        gen_df["mean_cluster_size"],
        gen_df["timescale_norm"],
        exclude_first=fit_exclude_first,
        exclude_last=fit_exclude_last,
    )

    metric_row = {
        "mv_alpha": mv_alpha,
        "mv_r2": mv_r2,
        "mv_p": mv_p,

        "ts_beta": ts_beta,
        "ts_r2": ts_r2,
        "ts_p": ts_p,

        "mv_alpha_norm": mv_alpha_norm,
        "mv_r2_norm": mv_r2_norm,
        "mv_p_norm": mv_p_norm,

        "ts_beta_norm": ts_beta_norm,
        "ts_r2_norm": ts_r2_norm,
        "ts_p_norm": ts_p_norm,

        "mean_variance_l0": mv0,
        "timescale_l0": tau0,
        "n_icg_gens": int(len(gen_df)),
    }

    return metric_row, gen_df

def model_outs(pars, seed):
    pars = dict(pars)
    T = float(pars.pop("T"))
    smoothe = float(pars.pop("smoothe", 0.05))
    pars["seed"] = int(seed)

    model = automata_EI_hiermod(**pars)
    spikes, pop_rate = run_model(model, T=T)
    
    mean_rate_hz = float(spikes.mean() / float(pars["dt"]))
    pop_rate_mean_hz = float(np.mean(pop_rate))
    pop_rate_std_hz = float(np.std(pop_rate))
    
    frac_silent_frames = float(np.mean(spikes.sum(axis=0) == 0))
    frac_active_neurons = float(np.mean(spikes.sum(axis=1) > 0))

    spikes_icg = exp_smooth_spikes(
        spikes,
        dt=float(pars["dt"]),
        tau=smoothe,
    )
    
    metric_row, gen_df = compute_icg_metrics(
        spikes=spikes_icg,
        dt=float(pars["dt"]),
    )
    
    out = {
        "phi": float(pars['phi']),
        "p_ext": float(pars['p_ext']),
        "seed": int(seed),
    
        "n_neurons": int(pars["n_neurons"]),
        "dt": float(pars["dt"]),
        "T": T,
    
        "theta": float(pars["theta"]),
        "ei_ratio": float(pars["ei_ratio"]),
        "e_w": float(pars["e_w"]),
        "i_w": float(pars["i_w"]),
        "refractory_steps": int(pars["refractory_steps"]),

        "smoothe": smoothe,
    
        "mean_rate_hz": mean_rate_hz,
        "pop_rate_mean_hz": pop_rate_mean_hz,
        "pop_rate_std_hz": pop_rate_std_hz,
        "frac_silent_frames": frac_silent_frames,
        "frac_active_neurons": frac_active_neurons,
    }
    
    out.update(metric_row)
    
    gen_df["phi"] = float(pars["phi"])
    gen_df["p_ext"] = float(pars["p_ext"])
    gen_df["seed"] = int(seed)
    gen_df["n_neurons"] = int(pars["n_neurons"])
    gen_df["smoothe"] = smoothe

    return out, gen_df
    

def safe_loglog_slope(x, y, exclude_first=True, exclude_last=True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[ok]
    y = y[ok]

    if exclude_first and len(x) > 3:
        x = x[1:]
        y = y[1:]

    if exclude_last and len(x) > 3:
        x = x[:-1]
        y = y[:-1]

    if len(x) < 3:
        return np.nan, np.nan, np.nan

    fit = linregress(np.log10(x), np.log10(y))
    return float(fit.slope), float(fit.rvalue ** 2), float(fit.pvalue)


def sem(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) <= 1:
        return np.nan
    return float(x.std(ddof=1) / np.sqrt(len(x)))



def exp_smooth_spikes(spikes, dt=0.01, tau=0.05):
    """
    Exponential causal smoothing of binary spike matrix.

    Parameters
    ----------
    spikes : array, shape (n_neurons, n_time)
        Binary or count spike raster.
    dt : float
        Simulation timestep in seconds.
    tau : float
        Exponential smoothing time constant in seconds.

    Returns
    -------
    X : array, shape (n_neurons, n_time)
        Smoothed activity traces.
    """
    import numpy as np

    spikes = np.asarray(spikes, dtype=np.float32)

    if tau <= 0:
        return spikes.copy()

    alpha = np.exp(-float(dt) / float(tau))

    X = np.zeros_like(spikes, dtype=np.float32)
    X[:, 0] = spikes[:, 0]

    for t in range(1, spikes.shape[1]):
        X[:, t] = alpha * X[:, t - 1] + spikes[:, t]

    return X