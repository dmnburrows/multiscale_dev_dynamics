# How coupling links the variance and timescale exponents

In the symmetric linear model with white isotropic drive, \(m\) changes how strongly each network mode fluctuates and how long it persists. Cluster size \(K\) changes how much each cluster samples those modes. The variance exponent \(\alpha\) measures how quickly their contributions grow in aggregate. The timescale exponent \(\beta\) measures whether this growth shifts variance toward longer-lived modes.

## Exact relationship

Let \(x_{t+1}=mBx_t+\eta_t\), with \(B\) symmetric, spectral radius one, \(0\le m<1\), and \(\eta_t\) white with identity covariance. For each eigenmode \(u_i\) of eigenvalue \(\mu_i\), let \(q_i(K)=\langle(\mathbf 1_C^\top u_i)^2\rangle_{|C|=K}\) be its mean squared overlap with clusters of size \(K\). Its variance contribution is \(s_i=q_i/(1-m^2\mu_i^2)\). Its variance share is \(p_i=s_i/\sum_j s_j\), and its integrated timescale is \(\tau_i=(1-m\mu_i)^{-1}\).

Mean variance is \(V=\sum_i s_i\). The one-sided, lag-zero-inclusive integrated autocorrelation of the variance-weighted cluster average is \(T=\sum_i p_i\tau_i\). For smooth positive overlaps, define \(e_i=\partial_{\log K}\log q_i\). At fixed \(m\),

$$
\alpha_{\rm local}=\mathbb E_p[e_i],\qquad
\beta_{\rm local}=\frac{\operatorname{Cov}_p(e_i,\tau_i)}{\mathbb E_p[\tau_i]}.
$$

This follows from \(\partial_{\log K}p_i=p_i(e_i-\mathbb E_p[e_i])\). Large \(\alpha\) can coexist with small \(\beta\) if the same slow modes dominate every fitted size. \(\beta\) rises when the modes whose shares grow with \(K\) also have long \(\tau_i\).

For dyadic cluster sizes, write \(W=\sum_i s_i\tau_i\), the mean summed autocovariance. Since \(W=VT\), its log–log slope is exactly \(\alpha+\beta\) on the same fitting window. This identity works even when the local derivatives are undefined.

A second exact link expresses \(T\) through the response of variance to two changes in the dynamics. Let \(V(K;m,r)\) be the variance under update \(mB+rI\). Holding cluster membership fixed during differentiation gives

$$
T(K,m)=1+\frac12\left(m\partial_m+\partial_r\right)\log V(K;m,r)\bigg|_{r=0}.
$$

The common diagonal perturbation distinguishes modes of opposite sign, which the unperturbed covariance weights identically. The code checks this identity with finite differences and direct matrix inversion.

### A counterexample with nonnegative weights

Take two blocks of four nodes, put a perfect matching in each block, and connect every cross-block pair with weight \(1/4\). Divide the adjacency matrix by two to set the row sum to one. Choosing different matchings gives two connected, loop-free graphs with identical \(B^2\), hence identical equal-time covariance for every \(m\). Hold the coarse-graining pairs fixed. At \(m=0.9\) over \(K=1,2\), both give \(\alpha=1.34754\), while \(\beta\) is \(0.56916\) in one and \(0.49723\) in the other. A universal mapping \(\beta=F(\alpha,m)\) therefore fails even within connected nonnegative networks.

## Ring crossover

For a diffusive line with damping \(\delta=1-m\) and diffusion coefficient \(D\), the stationary covariance is proportional to \(\exp(-|r|/\xi)/\sqrt{\delta}\), with \(\xi=\sqrt{D/\delta}\). For an interval of length \(K\) and \(u=K/\xi\), double integration gives \(V\propto\delta^{-3/2}(u-1+e^{-u})\) at fixed \(D\). Each mode with decay rate \(\delta+Dk^2\) contributes an extra inverse power of its decay rate to \(W\), so \(W=-\partial_\delta V\). Thus

$$
\alpha_{\rm local}=\frac{u(1-e^{-u})}{u-1+e^{-u}},\qquad
\delta T=\frac{3-\alpha_{\rm local}}2,\qquad
\beta_{\rm local}=-\frac{\partial_{\log u}\alpha_{\rm local}}{3-\alpha_{\rm local}}.
$$

The local peak is at \(u=2.014\) and \((\alpha,\beta)=(1.521,0.217)\). Fitting five dyadic sizes spanning a factor of 16 moves the peak to \(K_{\rm mid}/\xi=1.743\) and \((1.547,0.179)\).

The discrete ring with first and second neighbours has \(\mu(k)=(\cos k+\cos 2k)/2\) and \(D=1.25m\) at low wavenumber. Exact Fourier sums on 32,768 nodes agree with the matched continuum fit to 0.00275 in \(\alpha\) and 0.00169 in \(\beta\) for \(K=16\)–256 and \(0.3\le K_{\rm mid}/\xi\le5\). At \(K=128\)–2048 the errors fall below 0.00004 and 0.00003. Along the crossover, \(m=[1+1.25(u/K)^2]^{-1}\). Near \(m=1\), doubling \(K\) at fixed \(u\) approximately quarters \(1-m\). [Ring collapse figure](ring_collapse.png)

## Cross-network sweep

The discrete sweep produced 164,640 fitted rows from 48 graph instances, six families, four clustering policies and five fit windows. I used 12 seeds per random family at \(N=256\), two at \(N=1024\), and one realization per deterministic family and size. Greedy clustering is recomputed for each \(m\), fixed greedy clustering uses pairs found at \(m=0.95\), and the other two policies pair by index order or a fixed random permutation.

These are the median nearest points at \(N=256\), greedy clustering and \(K=2\)–32. The last column is the median of each network's peak across \(m\).

| Family | Nearest \(\alpha\) | Nearest \(\beta\) | Peak \(\beta\) |
|---|---:|---:|---:|
| Ring | 1.495 | 0.231 | 0.257 |
| Grid | 1.468 | 0.318 | 0.330 |
| Weighted hierarchy | 1.524 | 0.344 | 0.348 |
| ER | 1.643 | 0.348 | 0.594 |
| Random HM | 1.639 | 0.333 | 0.513 |
| Scale-free | 1.604 | 0.325 | 0.562 |

I measured nearest points by Euclidean distance to \((1.5,0.2)\). Across the five greedy windows and both sizes, only the ring enters \(|\alpha-1.5|\le0.05,\ |\beta-0.2|\le0.05\) on the tested \(m\) grid. The random examples overshoot \(\beta=0.2\), and the reconstructed HM peaks differ from the best HM reported in the browser chat. In the \(N=256\) greedy rows, averaging individual integrated timescales instead of weighting by each cluster's variance changes fitted \(\beta\) by more than 0.04 in roughly 10% of cases. [Cross-network plots](exponent_sweep.png) · [Nearest fits and peaks](nearest_to_target.csv)

## Dyadic hierarchy

In this weighted hierarchy, nodes whose first shared block is level \(j\) have weight \(\Phi^{-(j-1)}\). Haar eigenmodes give exact overlaps for aligned dyadic clusters. At \(\Phi=4\), I checked variance and integrated autocovariance against dense eigendecomposition. I evaluated 25,110,040 combinations of \(m\), \(\Phi\), size and fit window, then refined the grid fits with a numerical root solver.

| Fitted \(K\), \(N=32{,}768\) | \(\Phi\) yielding \((1.5,0.2)\) | \(m\) yielding \((1.5,0.2)\) |
|---|---:|---:|
| 2–32 | 9.6421 | 0.984098 |
| 4–64 | 9.0044 | 0.995685 |
| 8–128 | 8.8630 | 0.998963 |
| 16–256 | 8.8304 | 0.999761 |
| 32–512 | 8.8230 | 0.999946 |
| 64–1024 | 8.8213 | 0.999988 |

These solutions vary both the hierarchy parameter and fitting window. The shift in \(m\) remains large while the fitted \(\Phi\) converges. With \(\Phi=4\) fixed, this implementation gives \(\beta\approx0.325\) near \(\alpha=1.5\) on \(K=2\)–32. That differs from the \(\Phi=4\) figure quoted in the browser chat. The original definition of \(\Phi\), cluster window and timescale calculation would need to match before interpreting the discrepancy. [Hierarchy scan](hierarchy_scan.png) · [Refined target solutions](hierarchy_target_solutions.csv)

## Scope and reproduction

I used reconstructed graphs because I could not locate the original model or Figure 3 files in the accessible workspace. These calculations use stationary linear dynamics, unit white noise and a one-sided integrated autocorrelation. The empirical benchmark is rounded. Meshulam et al. estimate the timescale by collapsing rescaled autocorrelation curves, which is a different estimator. The exact modal formulas yield the full autocorrelation at any lag, so the original estimator can be applied directly once its conventions are available. These runs address the exponent link in a stable model. Locating a Griffiths phase would require separate evidence. [Meshulam et al., equations 24–25, Figure 10 and Table I](https://arxiv.org/pdf/1812.11904)

From the deliverables directory, using Python with NumPy, SciPy, pandas and matplotlib:

    python griffiths_sweep.py --output sweep_n256.csv
    python griffiths_sweep.py --n 1024 --seeds 2 --points 24 --output sweep_n1024.csv
    python analyze_sweep.py
    python ring_collapse.py
    python hierarchy_scan.py
