# Spatial-Field AIA (SF-AIA)

This document derives the bias that a spatially varying phase-step error introduces into the piston-model AIA solution of `aia.md` (Wang and Han, 2004), and the SF-AIA estimator, which recovers this error from the AIA residual and corrects the solution iteratively. The phase-step error field is part of the full model of `interference_model.md`, Eqs. (9a)–(9b); `aia.md` solves its piston limit $\Delta_n\equiv0$ (its Eq. 20).

## Setup

### Phase-step error field

The phase step of frame $n$ is $\delta_n+\Delta_n(x,y)$: a piston $\delta_n$ and a spatial remainder $\Delta_n$, expanded in a polynomial basis (`interference_model.md`, Eqs. 9a–9b),

$$\Delta_n(x,y)=\sum_{j=1}^Jc_{jn}\,p_j(x,y).\tag{T1}$$

Here $p_1,\dots,p_J$ are the monomials of total degree $1$ through $M$ ($x$, $y$, $x^2$, $xy$, $y^2$, …), so $J=(M+1)(M+2)/2-1$, and $c_{jn}$ is the coefficient of term $j$ in frame $n$. Degree $0$ is excluded, since it is the piston. $M=1$ describes a pure tilt; $M=0$ gives $\Delta_n\equiv0$, the piston model.

The measured intensities are (`interference_model.md`, Eq. 17, with $\alpha_n$ divided out as in `aia.md`)

$$I_n=a+g_nb\cos\big(\Phi+\delta_n+\Delta_n\big),\tag{T2}$$

with background $a$, fringe amplitude $b$, frame contrast $g_n$, and static phase $\Phi$. The dependence of $a,b,\Phi,\Delta_n$ on $(x,y)$ is suppressed; every equation holds pixel by pixel unless stated otherwise.

### Conventions

**Spatial mean** (`interference_model.md`, Eq. 9a). Place the origin of $(x,y)$ at the field centroid and subtract from each basis function its spatial mean:

$$\langle p_j\rangle_{x,y}=0,\qquad j=1,\dots,J.\tag{T3}$$

Thus $\langle\Delta_n\rangle_{x,y}=0$, and $\delta_n$ is the spatial mean of the phase step. For $M=1$, $p_1=x$ and $p_2=y$ already satisfy Eq. (T3); higher monomials are mean-subtracted, e.g. $x^2-\langle x^2\rangle_{x,y}$, which is generally nonzero at the centroid.

**Frame mean** (`interference_model.md`, Eq. 9b). Replacing $c_{jn}$ by $c_{jn}-\bar c_j$ and adding $\bar c_jp_j$ to $\Phi$ leaves every intensity unchanged. Remove this ambiguity by

$$\langle c_{jn}\rangle_n=0,\qquad j=1,\dots,J.\tag{T3b}$$

The per-frame fit of the estimator cannot impose this condition, so it is applied after the fit. The phase-origin, contrast-scale, and quadrature-frame conventions of the AIA solve in `aia.md` are unchanged.

**Scope.** The bias analysis propagates $\Delta_n$ through the pixel step of `aia.md` with $\delta_n$ and $g_n$ held at their true values. The effect of $\Delta_n$ on its frame step is treated at the end of the bias analysis.

### Linearization

With the quadrature fields and design-matrix rows of `aia.md` (Eqs. 2, 3, 6), write

$$P_n=g_n\cos\delta_n,\qquad Q_n=g_n\sin\delta_n,\qquad u=b\cos\Phi,\qquad v=-b\sin\Phi,\qquad I_n^{(0)}=a+P_nu+Q_nv,\tag{T4}$$

where $I_n^{(0)}$ is the piston prediction. Expanding the cosine of Eq. (T2) to first order in $\Delta_n$,

$$\cos(\Phi+\delta_n+\Delta_n)\approx\cos(\Phi+\delta_n)-\Delta_n\sin(\Phi+\delta_n),\tag{T5}$$

gives $I_n=I_n^{(0)}+\Delta I_n^{(1)}+O(\Delta_n^2)$ with

$$\Delta I_n^{(1)}=-g_nb\,\Delta_n\sin(\Phi+\delta_n).\tag{T6}$$

Since $g_nb\sin(\Phi+\delta_n)=uQ_n-vP_n$,

$$\Delta I_n^{(1)}=-w_n\Delta_n,\qquad w_n=uQ_n-vP_n.\tag{T7}$$

The coefficient $w_n=-\partial I_n^{(0)}/\partial\delta_n$ is the sensitivity of the piston prediction to the phase step.

## Bias of the piston AIA solve

### Frame moments

Define the $\Delta_n$-weighted frame moments

$$R_\Delta=\big\langle g_n\Delta_n\,e^{i\delta_n}\big\rangle_n,\qquad
\langle g^2\Delta\rangle=\big\langle g_n^2\Delta_n\big\rangle_n,\qquad
R_\Delta^{(2)}=\big\langle g_n^2\Delta_n\,e^{i2\delta_n}\big\rangle_n,\tag{T8}$$

and their unweighted counterparts, obtained with $\Delta_n\to1$,

$$R=\big\langle g_n\,e^{i\delta_n}\big\rangle_n,\qquad
\langle g^2\rangle=\big\langle g_n^2\big\rangle_n,\qquad
R^{(2)}=\big\langle g_n^2\,e^{i2\delta_n}\big\rangle_n.\tag{T8'}$$

Real and imaginary parts are marked by subscripts $c$ and $s$, e.g. $R_\Delta=R_{\Delta,c}+iR_{\Delta,s}$. The weighted moments are fields; by Eq. (T1), each is a combination of the same basis,

$$R_\Delta=\sum_j\rho_j\,p_j,\qquad
\langle g^2\Delta\rangle=\sum_j\mu_j\,p_j,\qquad
R_\Delta^{(2)}=\sum_j\rho_j^{(2)}\,p_j,\tag{T8a}$$

with $\rho_j=\langle g_nc_{jn}e^{i\delta_n}\rangle_n$, $\mu_j=\langle g_n^2c_{jn}\rangle_n$, and $\rho_j^{(2)}=\langle g_n^2c_{jn}e^{i2\delta_n}\rangle_n$.

### Pixel-step bias

The pixel step of `aia.md` solves $(a,u,v)^\top=A_p^{-1}A^\top(I_1,\dots,I_N)^\top$, where $A$ is the design matrix of `aia.md`, Eq. (6), with rows $(1,P_n,Q_n)$, and $A_p=A^\top A$. The piston prediction $I_n^{(0)}$ lies in the column space of $A$, so for $I_n=I_n^{(0)}+\Delta I_n^{(1)}$ the solve returns the true fields plus the bias

$$\begin{pmatrix}\Delta a\\\Delta u\\\Delta v\end{pmatrix}
=A_p^{-1}A^\top\begin{pmatrix}\Delta I_1^{(1)}\\\vdots\\\Delta I_N^{(1)}\end{pmatrix}.\tag{T10}$$

Measurement noise adds an independent term and is not carried here. With Eq. (T7) and the identities $P_n^2=\tfrac12g_n^2(1+\cos2\delta_n)$, $Q_n^2=\tfrac12g_n^2(1-\cos2\delta_n)$, $P_nQ_n=\tfrac12g_n^2\sin2\delta_n$, the entries of $A^\top(\Delta I_1^{(1)},\dots,\Delta I_N^{(1)})^\top/N$ are

$$\begin{aligned}
\big\langle\Delta I_n^{(1)}\big\rangle_n&=vR_{\Delta,c}-uR_{\Delta,s},\\
\big\langle P_n\Delta I_n^{(1)}\big\rangle_n&=\tfrac12v\langle g^2\Delta\rangle+\tfrac12\big(vR_{\Delta,c}^{(2)}-uR_{\Delta,s}^{(2)}\big),\\
\big\langle Q_n\Delta I_n^{(1)}\big\rangle_n&=-\tfrac12u\langle g^2\Delta\rangle+\tfrac12\big(vR_{\Delta,s}^{(2)}+uR_{\Delta,c}^{(2)}\big),
\end{aligned}\tag{T11}$$

and the same identities with $\Delta_n\to1$ give

$$\frac{A_p}{N}=\begin{pmatrix}
1&R_c&R_s\\
R_c&\tfrac12\big(\langle g^2\rangle+R_c^{(2)}\big)&\tfrac12R_s^{(2)}\\
R_s&\tfrac12R_s^{(2)}&\tfrac12\big(\langle g^2\rangle-R_c^{(2)}\big)
\end{pmatrix}.\tag{T12}$$

**Uniform steps.** Assume uniform steps $\delta_n=2\pi n/N$ and constant $g_n$, the good-coverage condition recommended in `aia.md`. For $N\ge3$, $R=R^{(2)}=0$, so

$$A_p=N\operatorname{diag}\big(1,\ \tfrac12\langle g^2\rangle,\ \tfrac12\langle g^2\rangle\big).\tag{T13}$$

For other step sets, the off-diagonal terms of Eq. (T12) add further bias. Combining Eqs. (T10)–(T13),

$$\Delta a=vR_{\Delta,c}-uR_{\Delta,s},\qquad
\Delta u=\frac{v\langle g^2\Delta\rangle+vR_{\Delta,c}^{(2)}-uR_{\Delta,s}^{(2)}}{\langle g^2\rangle},\qquad
\Delta v=\frac{-u\langle g^2\Delta\rangle+vR_{\Delta,s}^{(2)}+uR_{\Delta,c}^{(2)}}{\langle g^2\rangle}.\tag{T14}$$

### Bias in $a$, $b$, and $\Phi$

To first order in $(\Delta u,\Delta v)$, $b=\sqrt{u^2+v^2}$ and $\Phi=\operatorname{atan2}(-v,u)$ give

$$\Delta b=\cos\Phi\,\Delta u-\sin\Phi\,\Delta v,\qquad
\Delta\Phi=\frac{-\sin\Phi\,\Delta u-\cos\Phi\,\Delta v}{b}.\tag{T15}$$

Substituting Eq. (T14) and using

$$u\cos\Phi-v\sin\Phi=b,\qquad u\cos\Phi+v\sin\Phi=b\cos2\Phi,\qquad v\cos\Phi+u\sin\Phi=0,\qquad v\cos\Phi-u\sin\Phi=-b\sin2\Phi\tag{T16}$$

gives

$$\Delta a=-b\operatorname{Im}\big[R_\Delta e^{i\Phi}\big],\tag{T17}$$

$$\Delta b=-\frac{b}{\langle g^2\rangle}\operatorname{Im}\big[R_\Delta^{(2)}e^{i2\Phi}\big],\tag{T18}$$

$$\Delta\Phi=\frac{\langle g^2\Delta\rangle}{\langle g^2\rangle}-\frac{1}{\langle g^2\rangle}\operatorname{Re}\big[R_\Delta^{(2)}e^{i2\Phi}\big].\tag{T19}$$

The bias in $a$ oscillates at the first harmonic of $\Phi$, and the biases in $b$ and $\Phi$ at the second. The phase-independent term $\langle g^2\Delta\rangle/\langle g^2\rangle$ of $\Delta\Phi$ vanishes for constant $g_n$; when $g_n$ varies, it has the form of a smooth carrier phase and is removed with it (`interference_model.md`, Eq. 15). By Eq. (T8a), every envelope is a polynomial of degree at most $M$ with zero field mean. A residual ripple with uniform amplitude across the field therefore cannot come from $\Delta_n$; a ripple whose amplitude varies smoothly with position indicates an unmodeled phase-step error.

### Frame step

The frame step of `aia.md` regresses each frame on $(1,u,v)$ across pixels. The phase-step signal $-w_n\Delta_n=-Q_nu\Delta_n+P_nv\Delta_n$ adds $-Q_n\sum_{x,y}u^2\Delta_n/\sum_{x,y}u^2$ to the coefficient of $u$ and $P_n\sum_{x,y}v^2\Delta_n/\sum_{x,y}v^2$ to the coefficient of $v$, up to terms containing $uv$. With many fringes, $u^2\approx v^2\approx b^2/2$ and $uv$ averages out, so the fitted $(P_n,Q_n)$ are rotated by

$$\varepsilon_n=\frac{\langle b^2\Delta_n\rangle_{x,y}}{\langle b^2\rangle_{x,y}}=\sum_jc_{jn}\,\frac{\langle b^2p_j\rangle_{x,y}}{\langle b^2\rangle_{x,y}},\tag{T19a}$$

and the AIA piston is $\delta_n+\varepsilon_n$. The pixel step then sees the phase-step error $\Delta_n-\varepsilon_n$, and $\varepsilon_n$ has zero frame mean by Eq. (T3b). Eqs. (T10)–(T19) assume $\langle b^2p_j\rangle_{x,y}=0$ for every $j$, which holds, for example, for a tilt under illumination symmetric about the centroid. Otherwise, $\Delta_n$ is replaced by $\Delta_n-\varepsilon_n$ in Eq. (T8).

## SF-AIA estimator

### Per-frame normal equations

Run the AIA solve of `aia.md` to obtain $(a,u,v,\delta_n,g_n)$, and form the residual

$$r_n=I_n-\big(a+P_nu+Q_nv\big).$$

When the gain is jointly fitted, $I_n$ is taken net of the per-frame offset of `aia.md`, Eq. (7).

By Eq. (T7), $r_n\approx-w_n\Delta_n$ to first order. With $(a,u,v,\delta_n,g_n)$ held at the AIA estimates, $w_n=uQ_n-vP_n$ is known at every pixel, and substituting Eq. (T1) gives a residual model that is linear in the coefficients:

$$r_n\approx-\sum_{j=1}^Jc_{jn}\,w_np_j.$$

The coefficients $c_{1n},\dots,c_{Jn}$ of frame $n$ enter only the residual of frame $n$, so each frame is fitted separately by minimizing

$$\mathcal L_n(c_{1n},\dots,c_{Jn})=\sum_{x,y}\Big(r_n+w_n\sum_{j=1}^Jc_{jn}\,p_j\Big)^2.$$

Index the $N_p$ pixels by $k$, and write $w_{nk}$, $p_{jk}$, and $r_{nk}$ for the values at pixel $k$. The design matrix and its normal matrix are

$$D_n=\begin{pmatrix}
w_{n1}p_{11}&\cdots&w_{n1}p_{J1}\\
\vdots&&\vdots\\
w_{nN_p}p_{1N_p}&\cdots&w_{nN_p}p_{JN_p}
\end{pmatrix},\qquad
G^{(n)}=D_n^\top D_n=\begin{pmatrix}
\sum_{x,y}w_n^2p_1p_1&\cdots&\sum_{x,y}w_n^2p_1p_J\\
\vdots&&\vdots\\
\sum_{x,y}w_n^2p_Jp_1&\cdots&\sum_{x,y}w_n^2p_Jp_J
\end{pmatrix}.$$

Setting the $J$ derivatives of $\mathcal L_n$ to zero gives the normal equations $G^{(n)}(c_{1n},\dots,c_{Jn})^\top=-h^{(n)}$, with $h^{(n)}=D_n^\top(r_{n1},\dots,r_{nN_p})^\top$. If $G^{(n)}$ is invertible, their solution is

$$\begin{pmatrix}c_{1n}\\\vdots\\c_{Jn}\end{pmatrix}
=-\big(G^{(n)}\big)^{-1}h^{(n)}
=-\big(G^{(n)}\big)^{-1}\begin{pmatrix}\sum_{x,y}w_nr_np_1\\\vdots\\\sum_{x,y}w_nr_np_J\end{pmatrix}.\tag{E1}$$

Eq. (E1) is the pixel step of `aia.md` run in the other direction: a regression across pixels at a fixed frame, with regressors $w_np_j$, instead of a regression across frames at a fixed pixel, with regressors $(1,P_n,Q_n)$. For $M=1$ it is the $2\times2$ system for the tilt coefficients of frame $n$.

**Bias of a single pass.** AIA has already absorbed part of $-w_n\Delta_n$ into $(a,u,v)$ (Eq. T10), so the residual contains only the remainder, and one pass of Eq. (E1) underestimates $c_{jn}$. For uniform steps, the first and second temporal harmonics of $c_{jn}$ over $n$ are recovered at about half their size, and the other harmonics almost fully. The iteration below recovers the missing part; for uniform steps, the error in these harmonics halves in each round. The Variable-Projection AIA (`vp_aia.md`) removes this bias in a single pass by fitting the pixel corrections jointly.

### Gauge fixing

Eq. (E1) is solved independently per frame and cannot impose Eq. (T3b). Before the fitted field is used, remove the frame mean of each coefficient:

$$c_{jn}\leftarrow c_{jn}-\bar c_j,\qquad\bar c_j=\langle c_{jn}\rangle_n.\tag{E4}$$

The removed static part $\sum_j\bar c_jp_j$ belongs to $\Phi$ and is recovered by the next re-solve. Without this step the intensities are unchanged, but the split between $\Phi$ and $c_{jn}$ is arbitrary and can drift between rounds.

### Conditioning

$G^{(n)}$ is the Gram matrix of the basis weighted by $w_n^2$. Since $w_n$ vanishes on the fringe nulls of frame $n$, basis functions that are well separated under the plain inner product can become nearly dependent under this weight. The condition number

$$\kappa_\Delta^{(n)}=\operatorname{cond}\big(G^{(n)}\big)\tag{E3}$$

flags unreliable frames, and $\max_n\kappa_\Delta^{(n)}$ is a single summary. A sharp increase when $M$ is raised indicates that the added basis terms are not resolved by the fringe pattern.

### Algorithm

Eq. (E1) uses only the first-order model. Each round corrects the data and re-solves, so the next fit is linearized around the corrected estimates and also absorbs higher-order effects.

1. **Initial solve.** Run AIA to convergence.
2. **Basis.** Build an orthonormal basis $p_1,\dots,p_J$ of degree $1$ to $M$ on centered, unit-scaled coordinates, each with zero spatial mean (Eq. T3). $M=0$ returns the AIA result.
3. **Per-frame fit.** Compute $w_n$ and the residual $r_n$ of the original stack with the current fields, solve Eq. (E1) for every frame, and record $\kappa_\Delta^{(n)}$ (Eq. E3).
4. **Score.** Rebuild each frame with the step $\delta_n+\Delta_n$ and compute the residual RMS relative to the data RMS over a border-cropped region. Keep the fields of the best-scoring round.
5. **Gauge fixing.** Apply Eq. (E4).
6. **Correct and re-solve.** Form $I_n'=I_n+w_n\Delta_n$ from the original stack; each round estimates the total field, not an increment. Rerun the AIA pixel and frame steps on $I_n'$ with the phase-origin, whitening, and contrast-scale normalizations, re-centering the per-frame offsets when the gain is jointly fitted.
7. **Stop.** Repeat steps 3–6 until the score improvement falls below a tolerance or a round budget is spent, and report the best-scoring round.

## Noise of the corrected solve

Consider noise $\varepsilon_n(x,y)$ with variance $\sigma_0^2(x,y)$, independent across frames and pixels but not of equal size across the field (`aia_noise.md`, Eqs. 1 and 16), and exact $P_n,Q_n$. At one pixel, the pixel-step estimate of $\Phi$ is linear in the intensity column. By Eqs. (T10) and (T15), its error is $\sum_ns_n\varepsilon_n$ with

$$\begin{pmatrix}s_1\\\vdots\\s_N\end{pmatrix}=AA_p^{-1}\begin{pmatrix}0\\-\sin\Phi/b\\-\cos\Phi/b\end{pmatrix},$$

so plain AIA gives $\sigma_\Phi^2=\sigma_0^2\sum_ns_n^2$.

**Noise in the fitted field.** Let $\Pi=I-AA_p^{-1}A^\top$ be the $N\times N$ matrix that removes from a column over frames its pixel-step fit (`vp_aia.md`, Eq. 10). The residual noise at each pixel is $\Pi(\varepsilon_1,\dots,\varepsilon_N)^\top$, and Eq. (E1) maps the residual noise of frame $n$ to coefficient errors through $-\big(G^{(n)}\big)^{-1}D_n^\top$. The correction adds $w_np^\top(c_{1n},\dots,c_{Jn})^\top$ to frame $n$, where $p=(p_1,\dots,p_J)^\top$ at the pixel. Since $(s_1,\dots,s_N)^\top$ lies in the column space of $A$, $\Pi$ removes it, and the added term is uncorrelated with the phase error $\sum_ns_n\varepsilon_n$ of the same pixel. The correction therefore cannot reduce the noise; it adds a variance term:

$$\sigma_\Phi^2=\sigma_0^2\sum_ns_n^2+\sum_{n,m}t_nt_m\,\Pi_{nm}\,p^\top\big(G^{(n)}\big)^{-1}D_n^\top\Lambda D_m\big(G^{(m)}\big)^{-1}p,\qquad\Lambda=\operatorname{diag}\big(\sigma_0^2(x,y)\big),\tag{E7}$$

with $t_n=s_nw_n$ for the fit of Eq. (E1), and $t_n=s_nw_n-\tfrac1N\sum_{n'}s_{n'}w_{n'}$ after the gauge fixing of Eq. (E4). Here $D_n^\top\Lambda D_m$ is the $J\times J$ matrix with entries $\sum_{x,y}\sigma_0^2w_nw_mp_jp_{j'}$: the noise enters the coefficient covariance weighted by where in the field it sits, while $G^{(n)}=D_n^\top D_n$ stays unweighted, Eq. (E1) being an unweighted fit. The two coincide up to a factor when $\sigma_0$ is constant, $D_n^\top\Lambda D_m=\sigma_0^2D_n^\top D_m$, and the noise then factors out of the second term as it does out of the first. Where it is not constant the distinction matters: a field whose noise is concentrated where $w_n$ is large is fitted worse than its unweighted Gram suggests.

**Uniform steps.** Assume uniform steps with $N\ge5$, constant $\sigma_0$, $g_n$ and $b$, many fringes, and an orthonormal basis, $\langle p_jp_{j'}\rangle_{x,y}=\delta_{jj'}$. Then $G^{(n)}\approx\tfrac12g^2b^2N_pI_J$, $D_n^\top D_m\approx\tfrac12g^2b^2N_p\cos(\delta_n-\delta_m)\,I_J$, and $s_n=-2w_n/(Ng^2b^2)$, where $I_J$ is the $J\times J$ identity matrix. Averaging Eq. (E7) over the fringe phase gives

$$\frac{\sigma_\Phi^2}{\sigma_0^2\sum_ns_n^2}\approx1+\frac{|p(x,y)|^2}{4N_p},\qquad
\bigg\langle\frac{\sigma_\Phi^2}{\sigma_0^2\sum_ns_n^2}\bigg\rangle_{x,y}\approx1+\frac{J}{4N_p}.\tag{E8}$$

The correction increases the phase noise by a relative amount of order $J/N_p$, largest where $|p|$ is large and negligible for $N_p\gg J$. It can only increase it: the field is fitted to $\Pi\varepsilon$, the part of the noise the pixel step has already removed, so nothing it adds cancels the phase error of the same pixel.

Equation (E8) is a reading of Eq. (E7) under the four conditions above, not a substitute for it. Away from them — irregular steps, per-frame gains, a contrast or a noise level that varies across the field, or a basis the fringe pattern resolves unevenly — Eq. (E7) is the form to use. It costs one $J\times J$ solve per frame and one $NJ\times NJ$ quadratic form per pixel, with the Gram matrices $D_n^\top D_m$ a single reduction over the field.

### Composition with the fitted-step error

Equation (E7) holds $P_n$ and $Q_n$ at their true values. Their own noise is the subject of `aia_noise.md`, whose Eqs. (24) and (29) give the exact per-pixel phase variance when $\delta_n$, or $\delta_n$ and $g_n$, is fitted from the same frames. Both are first-order in the same noise and are built on the same $(u,v)$, and they describe displacements of $\Phi$ from different sources — a perturbed pixel-step design, and a fitted step field — so the SF-AIA phase variance is

$$\sigma_\Phi^2=\sigma_\Phi^2\Big|_{\texttt{aia\_noise.md}\ \text{Eq. (24) or (29)}}
+\sum_{n,m}t_nt_m\,\Pi_{nm}\,p^\top\big(G^{(n)}\big)^{-1}D_n^\top\Lambda D_m\big(G^{(m)}\big)^{-1}p.\tag{E9}$$

The first term reduces to `aia_noise.md` Eq. (10) when the steps are known, so Eq. (E9) contains the plain-AIA result as its own special case at $J=0$.

### Validity

Equation (E9) is a sum of two exact terms with a third omitted. Their cross-correlation is not derived here: the step-field fit is driven by $\Pi\varepsilon$, which the pixel step removes, while $e_{\delta_n}$ comes from the frame step's regression on $(1,u,v)$ across pixels, and those two projections are not orthogonal in general. The omitted term is $O(1/N_p)$, the same order as either of the two kept.

Both terms are first order in the noise and describe one corrected pass taken from the true fields, so they carry the conditions of `aia_noise.md` §"Validity" unchanged: they are variances about each estimator's own mean, in the gauge where $\Phi$ is free up to one additive constant.

Equation (E9) is also a single pass, and the refinement loop of §"Algorithm" does not leave it where it is. Each round removes part of the bias of §"Bias of a single pass", and with it part of the attenuation that kept the one-pass noise small, so the added variance grows round by round. It passes the $J/N_p$ of `vp_aia.md` Eq. (30) — the level of the unbiased one-pass estimator — after about three rounds and keeps rising, because the alternation's fixed point is not the joint least-squares solution that Eq. (30) describes. Equation (E9) is therefore a floor for a solve that has run several rounds, not an estimate of it, and the loop's own fixed point is not derived here. This is why the algorithm reports the best-scoring round rather than the last.

## References

- Z. Wang and B. Han, "Advanced iterative algorithm for phase extraction of randomly phase-shifted interferograms," *Optics Letters* **29**(14), 1671–1673 (2004).
