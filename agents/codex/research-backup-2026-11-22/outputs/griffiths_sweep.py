"""Exact stationary sweep for symmetric VAR(1) networks with unit white noise."""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components


def graph(family, n, rng):
    nodes = np.arange(n)
    distance = np.abs(nodes[:, None] - nodes)
    if family == 'ring':
        return (np.minimum(distance, n - distance) <= 2) - np.eye(n)
    if family == 'grid':
        side = int(np.sqrt(n))
        row, col = nodes // side, nodes % side
        dr = np.abs(row[:, None] - row)
        dc = np.abs(col[:, None] - col)
        return (np.minimum(dr, side - dr) + np.minimum(dc, side - dc) == 1).astype(float)
    if family == 'hierarchy':
        xor = nodes[:, None] ^ nodes
        level = np.floor(np.log2(np.maximum(xor, 1))) + 1
        return np.where(xor > 0, 4.0 ** (1 - level), 0)
    if family in ('ER', 'HM'):
        probability = 8 / (n - 1)
        if family == 'HM':
            probability = np.where(nodes[:, None] // 16 == nodes // 16, .4,
                                   np.where(nodes[:, None] // 64 == nodes // 64, .025, .008))
        upper = np.triu(rng.random((n, n)) < probability, 1)
        adj = (upper | upper.T).astype(float)
        count, labels = connected_components(adj)
        representatives = [np.flatnonzero(labels == i)[0] for i in range(count)]
        for i, j in zip(representatives, representatives[1:]):
            adj[i, j] = adj[j, i] = 1
        return adj
    adj = np.zeros((n, n))
    adj[:4, :4] = 1 - np.eye(4)
    degrees = adj.sum(axis=1)
    for node in range(4, n):
        neighbors = rng.choice(node, size=3, replace=False, p=degrees[:node] / degrees[:node].sum())
        adj[node, neighbors] = adj[neighbors, node] = 1
        degrees[node] = 3
        degrees[neighbors] += 1
    return adj


def groups(covariance, policy, n, order):
    membership = np.eye(n)[order]
    yield membership
    while len(membership) > 2:
        size = len(membership)
        if policy == 'greedy':
            cov = membership @ covariance @ membership.T
            sd = np.sqrt(np.diag(cov))
            ix, jx = np.triu_indices(size, 1)
            # Stable endpoint ordering breaks exact ties reproducibly.
            ranked = np.argsort(-(cov[ix, jx] / (sd[ix] * sd[jx])), kind='stable')
            used = set()
            pairs = []
            for index in ranked:
                i, j = int(ix[index]), int(jx[index])
                if i not in used and j not in used:
                    used.update((i, j))
                    pairs.append((i, j))
                    if len(used) == size:
                        break
            membership = membership[np.array(pairs)].sum(axis=1)
        else:
            membership = membership.reshape(size // 2, 2, n).sum(axis=1)
        yield membership


def curves(weights, lam, m):
    a = m * lam
    gain = 1 / (1 - a * a)
    t = 1 / (1 - a)
    variance = np.array([q.mean(axis=0) @ gain for q in weights])
    weighted_time = np.array([q.mean(axis=0) @ (gain * t) for q in weights])
    mean_time = np.array([np.mean((q @ (gain * t)) / (q @ gain)) for q in weights])
    response = np.array([q.mean(axis=0) @ (gain * gain * (a * a + a))
                         for q in weights]) / variance
    np.testing.assert_allclose(1 + response, weighted_time / variance, rtol=1e-8)
    return variance, weighted_time, mean_time


def slope(values):
    x = np.arange(len(values), dtype=float)
    x -= x.mean()
    return x @ np.log2(values) / (x @ x)


def run(n, seeds, points, output):
    ms = np.unique(np.r_[0, np.linspace(.05, .95, points), 1 - np.geomspace(.05, 1e-7, points)])
    windows = [(0, 4), (1, 5), (2, 6), (3, 7), (4, 7)]
    fields = ('n', 'family', 'seed', 'policy', 'm', 'k_min', 'k_max', 'alpha',
              'beta', 'beta_mean_time', 'gamma', 'target_error')
    with output.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        for family in ('ring', 'grid', 'hierarchy', 'ER', 'HM', 'SF'):
            for seed in range(seeds if family in ('ER', 'HM', 'SF') else 1):
                rng = np.random.default_rng(seed)
                adj = graph(family, n, rng)
                lam, modes = eigh(adj)
                lam /= np.max(np.abs(lam))
                fixed = {}
                for policy, order in [('indexed', np.arange(n)), ('random', rng.permutation(n)),
                                      ('fixed_greedy', np.arange(n))]:
                    cov = (modes / (1 - (.95 * lam)**2)) @ modes.T if policy == 'fixed_greedy' else None
                    fixed[policy] = [(g @ modes)**2 for g in groups(
                        cov, 'greedy' if cov is not None else policy, n, order)]
                for m in ms:
                    gain = 1 / (1 - (m * lam)**2)
                    cov = (modes * gain) @ modes.T
                    greedy = [(g @ modes)**2 for g in groups(cov, 'greedy', n, np.arange(n))]
                    for policy, weights in [*fixed.items(), ('greedy', greedy)]:
                        v, w, t = curves(weights, lam, m)
                        for lo, hi in windows:
                            if hi >= len(v):
                                continue
                            win = slice(lo, hi + 1)
                            alpha, gamma = slope(v[win]), slope(w[win])
                            beta = gamma - alpha
                            writer.writerow((n, family, seed, policy, m, 2**lo, 2**hi,
                                             alpha, beta, slope(t[win]), gamma,
                                             np.hypot(alpha - 1.5, beta - .2)))
            file.flush()
            print(n, family, 'done', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, choices=[256, 1024], default=256)
    parser.add_argument('--seeds', type=int, default=12)
    parser.add_argument('--points', type=int, default=100)
    parser.add_argument('--output', type=Path, default=Path('sweep.csv'))
    args = parser.parse_args()
    run(args.n, args.seeds, args.points, args.output)
