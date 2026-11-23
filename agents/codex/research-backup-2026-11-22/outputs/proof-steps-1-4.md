# Proof and audit of steps 1–4

The finite symmetric network identities in steps 1–2 are exact under the assumptions below. Step 3 is an exact differential identity for a differentiable extension of the cluster overlaps. The fitted-slope identity is exact on discrete sizes. Step 4 is exact within a separately defined continuum diffusion model. Applying that model to the discrete ring requires a scaling approximation.

## 1. Stationary dynamics and modal covariance

Assume \(B\) is a real symmetric \(N\times N\) matrix with spectral radius one, \(0\le m<1\), and \(\eta_t\) has zero mean and covariance \(I\), with different times uncorrelated. For independence claims assume the noise vectors are independent across time; for independent modes also assume Gaussian noise. Identity covariance by itself guarantees uncorrelated modes.

Write \(B=U\Lambda U^\top\), \(U^\top U=I\), and let column \(v_i\) of \(U\) have eigenvalue \(\lambda_i\). Define \(z_t=U^\top x_t\), \(\epsilon_t=U^\top\eta_t\), and \(a_i=m\lambda_i\). Then

\[
z_{t+1}=m\Lambda z_t+\epsilon_t,\qquad
\mathbb E[\epsilon_t\epsilon_t^\top]=I.
\]

The stationary solution exists in mean square and is

\[
z_i(t)=\sum_{r=0}^{\infty}a_i^r\epsilon_i(t-1-r).
\]

Convergence follows from \(|a_i|\le m<1\) and \(\sum_r a_i^{2r}<\infty\). Uncorrelated noise contributions give

\[
\operatorname{Var}(z_i)=\sum_{r=0}^{\infty}a_i^{2r}=\frac1{1-a_i^2}.
\]

Distinct modes have zero cross-covariance at every lag. For \(\ell\ge0\), iteration gives

\[
z_i(t+\ell)=a_i^\ell z_i(t)
+\sum_{r=0}^{\ell-1}a_i^r\epsilon_i(t+\ell-1-r).
\]

Each noise term in the latter sum is uncorrelated with \(z_i(t)\). Thus

\[
\operatorname{Cov}(z_i(t+\ell),z_i(t))
=\frac{a_i^\ell}{1-a_i^2},
\qquad C_i(\ell)=a_i^\ell.
\]

The one-sided, lag-zero-inclusive integrated normalized autocorrelation is

\[
\tau_i=\sum_{\ell=0}^{\infty}C_i(\ell)=\frac1{1-a_i}.
\]

Absolute convergence follows from \(|a_i|<1\). Negative \(a_i\) produce alternating correlations. This integral differs from the envelope decay time \(-1/\log|a_i|\) and from the convention \(1+2\sum_{\ell\ge1}C_i(\ell)\). No fitted exponential decay assumption has been used.

In neuron coordinates,

\[
S=\operatorname{Cov}(x_t)=U\operatorname{diag}\left(\frac1{1-a_i^2}\right)U^\top
=(I-m^2B^2)^{-1}.
\]

It obeys the discrete Lyapunov equation \(S=m^2BSB^\top+I\).

## 2. Cluster overlap and the exact timescale being calculated

Let \(c_C\) be the zero–one membership vector of cluster \(C\). Its summed activity is

\[
X_C(t)=c_C^\top x_t=\sum_i b_{Ci}z_i(t),
\qquad b_{Ci}=c_C^\top v_i.
\]

Uncorrelated modes imply

\[
V_C=\sum_i\frac{b_{Ci}^2}{1-a_i^2},\qquad
G_C(\ell)=\operatorname{Cov}(X_C(t+\ell),X_C(t))
=\sum_i\frac{b_{Ci}^2a_i^\ell}{1-a_i^2}.
\]

Let angle brackets denote the same arithmetic average over clusters of size \(K\) in every equation, and set \(q_i(K)=\langle b_{Ci}^2\rangle\). Then

\[
V(K)=\langle V_C\rangle
=\sum_i\frac{q_i(K)}{1-a_i^2}
=\sum_i s_i(K).
\]

Define the normalized mean autocovariance by

\[
R_K(\ell)=\frac{\langle G_C(\ell)\rangle}{\langle V_C\rangle}.
\]

Its integral is

\[
T(K)=\sum_{\ell\ge0}R_K(\ell)
=\frac{\sum_iq_i(K)/[(1-a_i^2)(1-a_i)]}{\sum_iq_i(K)/(1-a_i^2)}
=\sum_ip_i(K)\tau_i,
\quad p_i=s_i/V.
\]

This follows by summing finitely many absolutely convergent modal series. More explicitly, if \(T_C=\sum_\ell G_C(\ell)/V_C\), then

\[
T(K)=\frac{\langle V_CT_C\rangle}{\langle V_C\rangle}.
\]

Thus the formula integrates a variance-weighted average of individual normalized cluster autocorrelations. It does not calculate the autocorrelation of the sum over all clusters, which would include cross-cluster covariance. It also differs in general from \(\langle T_C\rangle\).

Parseval's identity provides a useful check:

\[
\sum_i b_{Ci}^2=\|c_C\|^2=K,\qquad \sum_iq_i(K)=K.
\]

At \(m=0\), this gives \(V(K)=K\), \(T(K)=1\), and fitted \(\alpha=1,\beta=0\).

## 3. Exact local and fitted identities

Fix the network, \(m\), noise model, and an extension of the clustering overlaps to continuous \(x=\log K\). Suppose \(q_i(x)\) are differentiable and positive near the evaluation point. Set

\[
e_i=\frac{d\log q_i}{dx}.
\]

Because the mode gains and \(\tau_i\) are independent of \(K\),

\[
s_i'=s_ie_i,\quad V'=\sum_i s_ie_i,\quad
\alpha_{\rm local}=\frac{V'}V=\sum_ip_ie_i.
\]

Differentiating \(p_i=s_i/V\) yields \(p_i'=p_i(e_i-\alpha_{\rm local})\). Therefore

\[
T'=\sum_ip_i(e_i-\alpha_{\rm local})\tau_i,
\]

and division by \(T>0\) proves

\[
\boxed{\beta_{\rm local}
=\frac{\operatorname{Cov}_p(e_i,\tau_i)}{\mathbb E_p[\tau_i]}}.
\]

If an overlap vanishes, its logarithm is undefined. A safer equivalent expression uses \(s_i'\) directly:

\[
\alpha_{\rm local}=\frac{\sum_i s_i'}V,\qquad
\beta_{\rm local}=\frac{\sum_i s_i'\tau_i}{W}-\frac{\sum_i s_i'}V,
\quad W=\sum_i s_i\tau_i.
\]

Actual greedy clustering produces overlaps at discrete sizes. An interpolated derivative requires specifying an interpolation. This differential identity alone does not establish a unique local exponent for such data.

For discrete fitted sizes define \(x_j=\log K_j\), fixed positive regression weights \(w_j\), \(\bar x_w=\sum_jw_jx_j/\sum_jw_j\), and the slope operator

\[
\mathcal L[f]=
\frac{\sum_jw_j(x_j-\bar x_w)f_j}
{\sum_jw_j(x_j-\bar x_w)^2}.
\]

This is weighted least squares with a fitted intercept. Since \(\log T=\log W-\log V\), linearity proves

\[
\boxed{\beta_{\rm fit}=\mathcal L[\log W]-\alpha_{\rm fit}},
\qquad \alpha_{\rm fit}=\mathcal L[\log V].
\]

Use the same sizes and regression weights for both slopes. This identity requires neither a smooth interpolation nor an exact power law. It concerns the stated integrated-timescale estimator.

## 4. A precise continuum model and its proof

Define a stationary field on the infinite real line obeying

\[
\partial_t x(r,t)=-(\delta-D\partial_r^2)x(r,t)+\eta(r,t),
\quad \delta,D>0,
\]
\[
\mathbb E[\eta(r,t)\eta(r',t')]=2Q\,\delta_{\rm Dirac}(r-r')
\delta_{\rm Dirac}(t-t'),\quad Q>0.
\]

A Fourier mode of wavenumber \(k\) has relaxation rate \(h(k)=\delta+Dk^2\), stationary variance spectrum \(Q/h(k)\), and integrated time \(1/h(k)\). With inverse Fourier measure \(dk/(2\pi)\),

\[
C(r)=\int_{\mathbb R}\frac{Qe^{ikr}}{\delta+Dk^2}\frac{dk}{2\pi}
=\frac{Q}{2\sqrt{D\delta}}e^{-|r|/\xi},
\qquad \xi=\sqrt{D/\delta}.
\]

For \(X_K=\int_0^Kx(r)\,dr\),

\[
V(K)=2\int_0^K(K-r)C(r)\,dr.
\]

The elementary integral is

\[
\int_0^K(K-r)e^{-r/\xi}\,dr
=K\xi-\xi^2(1-e^{-K/\xi})
=\xi^2F(u),
\quad F(u)=u-1+e^{-u},\quad u=K/\xi.
\]

Hence the exact continuum variance is

\[
\boxed{V(K)=Q\sqrt D\,\delta^{-3/2}F(u)}.
\]

Let \(H_K(k)=\int_0^K e^{ikr}dr\). The variance and integrated autocovariance have Fourier representations

\[
V=\int_{\mathbb R}\frac{Q|H_K(k)|^2}{h(k)}\frac{dk}{2\pi},
\qquad
W=\int_{\mathbb R}\frac{Q|H_K(k)|^2}{h(k)^2}\frac{dk}{2\pi}.
\]

Differentiation under the integral at fixed \(D,K,Q\) is valid for \(\delta>0\), since the derivative is integrable and locally dominated. It proves \(W=-\partial_\delta V\), so \(T=-\partial_\delta\log V\).

At fixed \(\delta,D\), \(d\log u/d\log K=1\). Thus

\[
\boxed{\alpha_{\rm local}(u)=\frac{uF'(u)}{F(u)}
=\frac{u(1-e^{-u})}{u-1+e^{-u}}}.
\]

At fixed \(D,K\), \(\partial_\delta u=u/(2\delta)\). Consequently,

\[
\partial_\delta\log V
=-\frac3{2\delta}+\frac{F'(u)}{F(u)}\frac{u}{2\delta},
\]

which proves

\[
\boxed{\delta T=\frac{3-\alpha_{\rm local}(u)}2}.
\]

Taking a log derivative with respect to \(K\), still at fixed \(\delta,D\), proves

\[
\boxed{\beta_{\rm local}(u)
=-\frac{u\alpha_{\rm local}'(u)}{3-\alpha_{\rm local}(u)}}.
\]

The derivative can be evaluated analytically using

\[
u\alpha'(u)=
u\frac{[F'(u)+uF''(u)]F(u)-uF'(u)^2}{F(u)^2},
\quad F''(u)=e^{-u}.
\]

As \(u\to0\), \(F(u)=u^2/2-u^3/6+\cdots\), giving \(\alpha\to2\), \(T\to1/(2\delta)\), and \(\beta\to0\). As \(u\to\infty\), \(F(u)\sim u-1\), giving \(\alpha\to1\), \(T\to1/\delta\), and \(\beta\to0\). These are exact limits in this continuum model.

The reported peak locations are numerical optimizations of these formulas. A numerical optimizer and derivative checks corroborate their values without establishing a formal theorem that the peak is unique.

## 5. Relationship to the discrete ring and necessary corrections

The original simulation ring connects each node to its first and second neighbours with equal weight. Its normalized dispersion is

\[
\lambda(k)=\frac{\cos k+\cos 2k}{2}
=1-\frac54k^2+O(k^4).
\]

Thus \(1-m\lambda(k)=\delta+Dk^2+O(mk^4)\), with \(\delta=1-m\) and \(D=5m/4\).

However its exact stationary variance spectrum is

\[
\frac1{1-a(k)^2}
=\frac1{[1-a(k)][1+a(k)]}.
\]

The continuum approximation additionally uses \(1+a(k)\simeq2\) for the dominant modes near \(a=1\). It therefore has \(Q=1/2\) in the convention above. That factor cancels in \(\alpha,\beta,T\). The relevant regime has \(1\ll K,\xi\ll N\), with the ratio \(K/\xi\) controlled. Finite-size and lattice corrections remain outside this limit.

For the discrete ring, the exact interval expressions are

\[
V_K=\frac1N\sum_{r=0}^{N-1}\frac{|H_K(k_r)|^2}{1-m^2\lambda(k_r)^2},
\quad
W_K=\frac1N\sum_{r=0}^{N-1}
\frac{|H_K(k_r)|^2}{[1-m^2\lambda(k_r)^2][1-m\lambda(k_r)]},
\]
\[
k_r=2\pi r/N,\quad
H_K(k)=\sum_{j=0}^{K-1}e^{ikj},\quad T_K=W_K/V_K.
\]

In particular, \(W=-\partial_\delta V\) is an identity of the continuum model at fixed \(D\). It is not an exact derivative identity for the original discrete model as \(m\) varies. In that model, \(D\) changes with \(m\), and the variance has the additional \(1+a\) factor.

A nearest-neighbour-only ring is bipartite for even \(N\), has a second edge at eigenvalue \(-1\), and has exactly zero equal-time covariance between opposite partitions. For example adjacent nodes have zero covariance for every \(m<1\). The simple single-exponential covariance across all sites cannot be assumed for that ring. The first-and-second-neighbour ring avoids this second critical spectral edge.

At fixed \(u\), the diffusion approximation with \(D=5m/4\) gives

\[
m=\frac1{1+(5/4)(u/K)^2}.
\]

The assertion that doubling \(K\) quarters \(1-m\) is asymptotic near \(m=1\). Within this approximation, the exact relation after doubling is \(m_{\rm new}=4m/(1+3m)\).

## 6. Independent checks and verdict

The accompanying script checks the modal covariance against a discrete Lyapunov solver, cluster formulas against direct matrix quadratic forms, lag covariance against matrix powers, local identities against central finite differences, and the fitted identity with unequal regression weights and nonuniform sizes. These checks include negative eigenvalues and unequal cluster variances.

For the continuum model it checks interval variance by real-space quadrature and integrated autocovariance by an independent Fourier quadrature, then checks the local timescale slope by finite differences. It also evaluates the nearest-neighbour bipartite counterexample.

Run with NumPy and SciPy:

    python check_derivation.py

[Numerical check results](proof-check-results.json) · [Check script](check_derivation.py)

The corrected claim is that steps 1–2 are exact second-order identities, step 3 has the differentiability and estimator conditions specified above, and step 4 is exact for the stated infinite-line continuum model and asymptotic for the simulated discrete ring. Identifying these conditions corrects the earlier suggestion that the entire four-step derivation was exact for the discrete network.
