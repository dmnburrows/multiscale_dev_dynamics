# A relationship between fitted exponents, coupling, and the fitting window

## 1. Definition of the fitted exponents

Choose cluster sizes \(K_j\), positive regression weights \(w_j\), and set \(x_j=\log K_j\). Define the linear slope operator

\[
\mathcal L[f]=\frac{\sum_jw_j(x_j-\bar x_w)f_j}
{\sum_jw_j(x_j-\bar x_w)^2},
\qquad \bar x_w=\frac{\sum_jw_jx_j}{\sum_jw_j}.
\]

Then \(\mathcal L[1]=0\), so multiplying a positive curve by a constant leaves its log slope unchanged. Throughout, the regression includes an intercept and uses the same weights and sizes for variance and timescale.

Define \(W(K)\) to be the summed one-sided autocovariance and \(T=W/V\). The globally fitted exponents are exactly

\[
\alpha_{\rm fit}=\mathcal L[\log V],\qquad
\beta_{\rm fit}=\mathcal L[\log W]-\mathcal L[\log V].
\]

These definitions require neither a differentiable overlap curve nor exact power laws.

## 2. Direct ring formula

In the infinite-line continuum diffusion model, set \(\delta=1-m>0\), \(D>0\), \(\xi=\sqrt{D/\delta}\), and \(u=K/\xi\). The spatial covariance is \(C(r)=Qe^{-|r|/\xi}/(2\sqrt{D\delta})\).

For the interval sum \(X_K=\int_0^Kx(r)\,dr\),

\[
V(K)=2\int_0^K(K-r)C(r)\,dr
=Q\sqrt D\,\delta^{-3/2}F(u),
\qquad F(u)=u-1+e^{-u}.
\]

In this model the spectrum of summed autocovariance has one additional factor \((\delta+Dk^2)^{-1}\), so \(W=-\partial_\delta V\) at fixed \(D,Q,K\). Direct differentiation gives

\[
W(K)=\frac{Q\sqrt D}{2}\delta^{-5/2}H(u),
\qquad
H(u)=3F(u)-uF'(u)=2u-3+(u+3)e^{-u}.
\]

Consequently

\[
T(K)=\frac1{2\delta}\frac{H(u)}{F(u)}.
\]

This obtains both curves without defining any local exponent. The derivative with respect to the model parameter \(\delta\) is a spectrum identity, rather than a derivative with respect to cluster size.

At fixed \(m,D\), the multiplicative prefactors are independent of \(K\). Therefore

\[
\boxed{\alpha_{\rm fit}=\mathcal L[\log F(K/\xi)]},
\qquad
\boxed{\beta_{\rm fit}=\mathcal L[\log H(K/\xi)]
-\mathcal L[\log F(K/\xi)]}.
\]

For \(K_j=K_{\rm mid}r_j\), fixed ratios \(r_j\), and a fixed weighting rule, put \(u_{\rm mid}=K_{\rm mid}/\xi\). Both exponents become functions of that single variable:

\[
(\alpha_{\rm fit},\beta_{\rm fit})
=(A_{\{r,w\}}(u_{\rm mid}),B_{\{r,w\}}(u_{\rm mid})).
\]

This is a smooth parametric curve for each specified fitting window. It need not pass through the empirical target exactly, and different windows define different curves. A fitted exponent alone does not establish that the underlying log–log curve is straight over its window.

For the first-and-second-neighbour discrete ring, the diffusion approximation has \(D=5m/4\), giving

\[
u_{\rm mid}=K_{\rm mid}\sqrt{\frac{1-m}{1.25m}},
\qquad
m=\frac1{1+1.25(u_{\rm mid}/K_{\rm mid})^2}.
\]

Thus the fitted exponent pair constrains a combination of observation scale and coupling. For a known window, inference of \(m\) can be attempted by fitting both outputs, subject to uncertainty and model mismatch. A unique inverse must be checked in the parameter range of interest.

## 3. Finite-window limits

At small \(u\),

\[
F(u)=u^2/2+O(u^3),\qquad H(u)=u^2/2+O(u^4).
\]

For any fixed collection of ratios \(r_j\), these give \(\alpha_{\rm fit}\to2\), \(\beta_{\rm fit}\to0\) as \(u_{\rm mid}\to0\). At large \(u\), \(F(u)\sim u\) and \(H(u)\sim2u\), giving \(\alpha_{\rm fit}\to1\), \(\beta_{\rm fit}\to0\).

The variance-weighted integrated timescale grows across an intermediate range. The exact height and position of its fitted-slope maximum depend on the complete window and weights. A continuum curve cannot predict abrupt jumps caused by changes of greedy cluster membership.

## 4. General finite-network formula without local slopes

For a real symmetric network with spectral radius one, stable coupling \(0\le m<1\), unit white noise covariance, and orthonormal eigenvectors \(v_i\) of eigenvalues \(\lambda_i\), define

\[
q_i(K)=\left\langle(c_C^\top v_i)^2\right\rangle_{|C|=K},
\quad
s_i(K,m)=\frac{q_i(K)}{1-m^2\lambda_i^2},
\quad
\tau_i=\frac1{1-m\lambda_i}.
\]

Then

\[
V(K,m)=\sum_i s_i(K,m),\qquad
W(K,m)=\sum_i s_i(K,m)\tau_i.
\]

Substitute these sums into the regression definition:

\[
\boxed{\alpha_{\rm fit}(m)=
\mathcal L\!\left[\log\sum_i\frac{q_i(K,m)}{1-m^2\lambda_i^2}\right]},
\]
\[
\boxed{\beta_{\rm fit}(m)=
\mathcal L\!\left[\log\sum_i
\frac{q_i(K,m)}{(1-m^2\lambda_i^2)(1-m\lambda_i)}\right]
-\alpha_{\rm fit}(m)}.
\]

The explicit \(m\) argument on \(q_i\) allows greedy clusters to be recomputed at every coupling. These formulas are exact at the discrete cluster sizes for the specified integrated-timescale estimator. They describe the calculation for a known network; a predictive reduction across architectures still needs a theory for their eigenvalues and overlaps.

A useful equivalent interpretation follows by selecting a reference size \(K_0\) for which all relevant \(q_i(K_0)>0\). Define

\[
R_i(K)=\frac{q_i(K)}{q_i(K_0)},\quad
p_i^0=\frac{s_i(K_0)}{V(K_0)},\quad
\widetilde p_i^0=\frac{s_i(K_0)\tau_i}{W(K_0)}.
\]

Then

\[
\frac{V(K)}{V(K_0)}=\sum_i p_i^0R_i(K),\qquad
\frac{W(K)}{W(K_0)}=\sum_i\widetilde p_i^0R_i(K).
\]

Therefore

\[
\alpha_{\rm fit}=\mathcal L[\log\mathbb E_{p^0}R_i(K)],
\]
\[
\beta_{\rm fit}=
\mathcal L[\log\mathbb E_{\widetilde p^0}R_i(K)]
-\mathcal L[\log\mathbb E_{p^0}R_i(K)].
\]

Both exponents depend on the same finite-size overlap growth factors. The summed autocovariance gives additional weight to slow modes through \(\widetilde p_i^0\). This is a global version of the modal explanation, with no differentiation of the overlaps. When reference overlaps vanish, use the direct sums above or choose the singleton reference \(K_0=1\), for which \(q_i(1)=1/N\) for a partition of all nodes.

## 5. Checks

The script evaluates the ring relations directly and compares them with independent discrete Fourier sums. It also verifies the general overlap-ratio identity against directly regressed modal sums on 100 random tables. Fourier comparison measures the continuum approximation error; it is not an assertion of equality with the finite discrete ring.

[Check script](global_ring_relation.py) · [Numerical results](global-ring-checks.json)
