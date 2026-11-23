"""Check the modal identity behind the alpha/beta decomposition.

Uses linear_theory.py unmodified: imports its greedy_pairs / pool / coarse_grain /
fit_scaling / stationary_covariances / make_network and only ADDS membership
tracking alongside, so the cascade is bit-for-bit the one run_sweep uses.
"""
import sys
sys.path.insert(0, "/Users/k2585057/Library/CloudStorage/Dropbox/PhD/Analysis/"
                   "my_scripts/GitHub/multiscale_dev_dynamics")
import numpy as np
import pandas as pd
from scipy.linalg import eigh
import linear_theory as lt

np.seterr(all="ignore")

N = 512
MIN_CLUSTERS = 4


def cascade(S, L, min_clusters=MIN_CLUSTERS):
    """lt.coarse_grain, but also returning the unit->cluster membership per level."""
    memb = np.arange(len(S))
    levels, K = [], 1
    while len(S) >= min_clusters:
        levels.append(dict(K=K, clusters=len(S),
                           variance=np.trace(S) / len(S),
                           tau=np.trace(L) / np.trace(S),
                           memb=memb.copy()))
        if len(S) == 1:
            break
        pairs = lt.greedy_pairs(S)
        a, b = pairs.T
        relabel = np.empty(len(S), int)
        relabel[a] = np.arange(len(pairs))
        relabel[b] = np.arange(len(pairs))
        memb = relabel[memb]
        S, L = lt.pool(S, pairs), lt.pool(L, pairs)
        K *= 2
    return levels


def overlaps(memb, eigenvectors):
    """abar_k = < (v_k . 1_S)^2 > averaged over the clusters at this level."""
    n_clusters = memb.max() + 1
    indicator = np.zeros((len(memb), n_clusters))
    indicator[np.arange(len(memb)), memb] = 1.0
    projected = eigenvectors.T @ indicator          # (modes, clusters)
    return np.mean(projected ** 2, axis=1)


def modal_weights(eigenvalues, m):
    lam = m * eigenvalues / np.max(np.abs(eigenvalues))
    w = 1.0 / ((1.0 - lam) * (1.0 + lam))
    return w, 1.0 / (1.0 - lam)


def slopes(K, y, drop_first=1, drop_last=1):
    end = len(K) - drop_last if drop_last else len(K)
    x, yy = np.log(np.asarray(K[drop_first:end], float)), np.log(np.asarray(y[drop_first:end], float))
    return float(np.polyfit(x, yy, 1)[0])


NETWORKS = [("hm2.83", "hm", 2.83), ("er6.2", "er", 6.2), ("ring4", "ring", 4)]
M_VALUES = [0.5, 0.9, 0.99, 0.999]

print("=" * 92)
print("TEST 1  modal identity:  V(K) = sum_k w_k abar_k(K)   and   tau(K) = sum w_k tau_k abar_k / sum w_k abar_k")
print("=" * 92)
print(f"{'network':<9}{'m':<8}{'max rel err V(K)':<20}{'max rel err tau(K)':<22}"
      f"{'alpha identity':<18}{'beta identity'}")

store = {}
for name, fam, par in NETWORKS:
    A = lt.make_network(N, fam, par, 7)
    ev, V = eigh(A, check_finite=False)
    store[name] = (ev, V)
    for m in M_VALUES:
        S, L = lt.stationary_covariances(ev, V, m)
        levels = cascade(S, L)
        w, tk = modal_weights(ev, m)
        Ks = [d["K"] for d in levels]
        v_true = np.array([d["variance"] for d in levels])
        t_true = np.array([d["tau"] for d in levels])
        v_pred, t_pred = [], []
        for d in levels:
            ab = overlaps(d["memb"], V)
            num_v = float(np.sum(w * ab))
            v_pred.append(num_v)
            t_pred.append(float(np.sum(w * tk * ab)) / num_v)
        v_pred, t_pred = np.array(v_pred), np.array(t_pred)
        # alpha/beta from the module's own fitter vs from the modal reconstruction
        df = pd.DataFrame({"K": Ks, "variance": v_true, "tau": t_true})
        ref = lt.fit_scaling(df)
        a_mod, b_mod = slopes(Ks, v_pred), slopes(Ks, t_pred)
        print(f"{name:<9}{m:<8}{np.abs(v_pred/v_true-1).max():<20.3e}"
              f"{np.abs(t_pred/t_true-1).max():<22.3e}"
              f"{abs(a_mod-ref['alpha']):<18.3e}{abs(b_mod-ref['beta']):.3e}")

print()
print("=" * 92)
print("TEST 2  how much of abar_k(K) is itself m-dependent (the partition is chosen from S)")
print("        'frozen' = take the partitions found at m_ref, re-score them with the w,tau of m")
print("=" * 92)
print(f"{'network':<9}{'m_ref':<8}{'m':<8}{'alpha true':<12}{'alpha frozen':<14}"
      f"{'beta true':<11}{'beta frozen':<13}{'cluster agreement at K=64'}")

for name, fam, par in NETWORKS:
    ev, V = store[name]
    parts = {}
    for m in M_VALUES:
        S, L = lt.stationary_covariances(ev, V, m)
        parts[m] = cascade(S, L)
    for m_ref in (0.5, 0.999):
        for m in M_VALUES:
            w, tk = modal_weights(ev, m)
            Ks = [d["K"] for d in parts[m_ref]]
            v_f, t_f = [], []
            for d in parts[m_ref]:
                ab = overlaps(d["memb"], V)
                nv = float(np.sum(w * ab))
                v_f.append(nv)
                t_f.append(float(np.sum(w * tk * ab)) / nv)
            df_t = pd.DataFrame({"K": [d["K"] for d in parts[m]],
                                 "variance": [d["variance"] for d in parts[m]],
                                 "tau": [d["tau"] for d in parts[m]]})
            ref = lt.fit_scaling(df_t)
            # agreement of the K=64 partitions: fraction of unit pairs co-clustered in both
            lv = {d["K"]: d["memb"] for d in parts[m]}
            lr = {d["K"]: d["memb"] for d in parts[m_ref]}
            if 64 in lv and 64 in lr:
                same_v = lv[64][:, None] == lv[64][None, :]
                same_r = lr[64][:, None] == lr[64][None, :]
                agree = float((same_v == same_r).mean())
            else:
                agree = np.nan
            print(f"{name:<9}{m_ref:<8}{m:<8}{ref['alpha']:<12.4f}{slopes(Ks, v_f):<14.4f}"
                  f"{ref['beta']:<11.4f}{slopes(Ks, t_f):<13.4f}{agree:.4f}")

print()
print("=" * 92)
print("TEST 3  where the weight sits: Perron mode share of V(K), and abar_1(K) vs abar_bulk(K)")
print("=" * 92)
for name, fam, par in NETWORKS:
    ev, V = store[name]
    print(f"\n{name}   normalised spectrum s_k = eig/rho : "
          f"s_max=1.0000  s_2={ev[-2]/np.abs(ev).max():.4f}  s_min={ev[0]/np.abs(ev).max():.4f}")
    print(f"{'m':<8}{'K':<6}{'abar_1':<12}{'w_1 abar_1':<14}{'V(K)':<14}"
          f"{'p = Perron share':<18}{'tau_1':<10}{'tau(K)'}")
    for m in (0.9, 0.999):
        S, L = lt.stationary_covariances(ev, V, m)
        levels = cascade(S, L)
        w, tk = modal_weights(ev, m)
        for d in levels:
            if d["K"] not in (1, 8, 64):
                continue
            ab = overlaps(d["memb"], V)
            tot = float(np.sum(w * ab))
            print(f"{m:<8}{d['K']:<6}{ab[-1]:<12.5f}{w[-1]*ab[-1]:<14.4f}{tot:<14.4f}"
                  f"{w[-1]*ab[-1]/tot:<18.4f}{tk[-1]:<10.2f}{d['tau']:.3f}")
