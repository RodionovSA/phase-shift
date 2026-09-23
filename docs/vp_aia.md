# Variable-Projection AIA (VP-AIA)

VP-AIA estimates a spatially varying phase-step error as a first-order correction to the AIA solution of the piston model (`aia.md`; Wang and Han, 2004). The pixel fields are eliminated by projection, as in the variable projection method for separable least-squares problems (Golub and Pereyra, 1973), which leaves one small linear system for the frame corrections and the step-error coefficients.

## Model and conventions

The measured intensities are modeled as

$$I_n(x, y) = a(x, y) + g_n\,b(x, y)\,\cos\big(\Phi(x, y) + \delta_n + \Delta_n(x, y)\big), \qquad n = 1 \dots N, \tag{1}$$

with background $a$, fringe amplitude $b$, static phase $\Phi$, per-frame contrast $g_n$, piston step $\delta_n$, and spatially varying phase-step error $\Delta_n$. These symbols denote the true parameters; AIA estimates carry a hat, as in $\hat\Phi$, and the corrected VP-AIA estimates a tilde, as in $\tilde\Phi$ or $\tilde\Delta_n$.

### Gauge conventions

The intensity depends on $\Phi$, $\delta_n$, and $\Delta_n$ only through their sum, and on $g_n$ and $b$ only through their product. Four conventions select one parameter set:

- **Spatial mean.** A uniform part of $\Delta_n$ is indistinguishable from $\delta_n$ and is assigned to it: $\langle\Delta_n\rangle_{x,y}=0$.
- **Frame mean.** A pattern shared by all frames is indistinguishable from $\Phi$ and is assigned to it: $\langle\Delta_n(x,y)\rangle_n=0$.
- **Phase origin.** A constant exchanged between $\Phi$ and all $\delta_n$ is fixed by $\delta_1=0$.
- **Contrast scale.** A factor exchanged between $g_n$ and $b$ is fixed by $\operatorname{median}_ng_n=1$. The median, as in `aia.md` Eq. (16), keeps a single collapsed frame from moving the scale of the others.

The AIA parametrization adds four freedoms of the quadrature frame, fixed by the conventions of Appendix D. An unrestricted $\Delta_n$ has more unknowns than the data provide equations (Appendix A), so its spatial form must be restricted.

### Cost of the full model

The piston limit $\Delta_n\equiv0$ is linear in the pixel fields for fixed frame parameters and in the frame parameters for fixed pixel fields, and AIA alternates these two fits, each with one small normal matrix. A spatially varying $\Delta_n$ breaks this split: it enters inside the cosine and depends on both frame and pixel. The full model can be solved as a general nonlinear least-squares problem, as the general iterative algorithm (GIA) of Chen and Kemao (2021) does, at about 72 times the computation time of AIA (558.8 s against 7.7 s at equal iteration counts). VP-AIA instead keeps the AIA solution as a baseline and recovers $\Delta_n$ together with the bias of that baseline as small corrections.

## AIA baseline

With $\Delta_n\equiv0$ and the quadrature variables

$$u=b\cos\Phi,\qquad v=-b\sin\Phi,\qquad P_n=g_n\cos\delta_n,\qquad Q_n=g_n\sin\delta_n,\tag{2}$$

Eq. (1) is the piston model

$$I_n(x,y)=a(x,y)+P_nu(x,y)+Q_nv(x,y).\tag{3}$$

AIA (`aia.md`) fits it and returns $\hat a,\hat u,\hat v,\hat P_n,\hat Q_n$, brought to the conventions above and to those of Appendix D. Its last pixel step solves, at each pixel,

$$\big(\hat a,\hat u,\hat v\big)^\top=A_p^{-1}A^\top\mathbf i,\qquad A_p=A^\top A,\tag{4}$$

where $\mathbf i=(I_1,\dots,I_N)^\top$ and $A$ has rows $(1,\hat P_n,\hat Q_n)$. Since the data contain $\Delta_n$, the AIA estimates absorb part of it and are biased at first order (Appendix B).

## First-order corrections

For small $|\Delta_n|$, each AIA estimate is the exact parameter plus a bias ordered by powers of $\Delta_n$ (Appendix B):

$$\hat f=f^{(0)}+f^{(1)}+O(\Delta_n^2),\qquad f^{(k)}=O(\Delta_n^k),\qquad f\in\{a,u,v,P_n,Q_n\}.\tag{5}$$

$f^{(0)}$ is the exact parameter the method must recover, and $f^{(1)}$ the bias the fit absorbed. The biases $a^{(1)},u^{(1)},v^{(1)}$ are fields shared by all frames; $P_n^{(1)},Q_n^{(1)}$ are scalars shared by all pixels. $\Delta_n$ itself is first order throughout.

### Spatial modes

To reduce the unknowns, $\Delta_n$ is expanded on $J$ fixed, linearly independent spatial functions with frame-dependent amplitudes,

$$\Delta_n(x,y)=\sum_{j=1}^J\alpha_{nj}H_j(x,y),\qquad
\langle H_j\rangle_{x,y}=0,\qquad
\langle\alpha_{nj}\rangle_n=0,\tag{6}$$

which enforce the spatial-mean and frame-mean conventions. A unique solution requires (Appendix A)

$$J\le J_{\max}\equiv\left\lfloor\frac{(N-3)(K-2)}{N-1}\right\rfloor,\tag{7}$$

and identifying any mode requires $N\ge5$. Since $K\gg N$ in practice, $J_{\max}\approx K(N-3)/(N-1)$, about $K/2$ at $N=5$.

### First-order model of the residual

With the baseline residual

$$r_n=I_n-\big(\hat a+\hat P_n\hat u+\hat Q_n\hat v\big),\tag{8}$$

substituting Eqs. (5) and (6) into Eq. (1) and keeping first-order terms gives, up to noise (Appendix B),

$$r_n=w_n\sum_{j=1}^J\alpha_{nj}H_j-\big(a^{(1)}+\hat P_nu^{(1)}+\hat Q_nv^{(1)}\big)-\hat uP_n^{(1)}-\hat vQ_n^{(1)},\qquad w_n=\hat P_n\hat v-\hat Q_n\hat u.\tag{9}$$

The step-error signal enters with a plus sign and every absorbed bias with a minus: what the fit took out of the data is missing from the residual. $w_n$ is the first-order sensitivity to $\Delta_n$ evaluated at the AIA estimates; using the true parameters instead changes Eq. (9) only at second order.

### Removing the pixel corrections

At one pixel, define the $N\times N$ residual-maker of the pixel step (Hoaglin and Welsch, 1978),

$$\Pi=I-AA_p^{-1}A^\top.\tag{10}$$

By Eq. (4), $\Pi\mathbf i=(r_1,\dots,r_N)^\top$. $\Pi$ depends only on $\hat P_n,\hat Q_n$, is the same at every pixel, and satisfies $\Pi A=0$, $\Pi^2=\Pi$, $\Pi^\top=\Pi$; it has rank $N-3$. The pixel-side bias of Eq. (9) is a column $A\big(a^{(1)},u^{(1)},v^{(1)}\big)^\top$, which $\Pi$ removes, while $\Pi$ leaves the residual unchanged. Applying $\Pi$ to Eq. (9) gives

$$\begin{pmatrix}r_1\\\vdots\\r_N\end{pmatrix}
=\Pi\begin{pmatrix}z_1\\\vdots\\z_N\end{pmatrix},\qquad
z_n=w_n\sum_j\alpha_{nj}H_j-\hat uP_n^{(1)}-\hat vQ_n^{(1)}.\tag{11}$$

Eq. (11) holds at every pixel and contains only the $2N+NJ$ unknowns $P_n^{(1)},Q_n^{(1)},\alpha_{nj}$, all shared by the pixels. Eliminating a group of regressors by applying the residual-maker to the data and the remaining regressors does not change the least-squares estimates of the rest (Frisch and Waugh, 1933; Lovell, 1963), so a fit of Eq. (11) gives the same unknowns as a joint fit of Eq. (9).

### Fit of the frame corrections and phase-step coefficients

Eq. (11) is fitted over all pixels by least squares, with the loss

$$\mathcal L_1=\sum_{x,y}\sum_{n=1}^N\Big(r_n-\sum_{m=1}^N\Pi_{nm}z_m\Big)^2
=\sum_{x,y}\Big(\sum_nr_n^2-2\sum_nr_nz_n+\sum_{n,m}z_n\Pi_{nm}z_m\Big).\tag{12}$$

Collect the unknowns in $\beta=\big(P_1^{(1)},\dots,P_N^{(1)},\,Q_1^{(1)},\dots,Q_N^{(1)},\,\alpha_{11},\dots,\alpha_{NJ}\big)^\top$ of length $2N+NJ$. Setting the gradient of Eq. (12) to zero gives the normal equations

$$M\beta=\rho,\tag{13}$$

whose matrix and right-hand side are pixel sums of the baseline fields, the modes, and the residual, all accumulated in one pass (Appendix C).

**Undetermined directions.** $\Pi$ removes the columns $(1,\dots,1)^\top$, $(\hat P_1,\dots,\hat P_N)^\top$, and $(\hat Q_1,\dots,\hat Q_N)^\top$, so adding any combination of them to $\big(P_n^{(1)}\big)$ or $\big(Q_n^{(1)}\big)$ leaves Eq. (11) unchanged: these six directions are the linearized freedoms of the quadrature frame. A frame-constant $\alpha_{nj}=c_j$ adds $\big(\sum_jc_jH_j\big)w_n$, which $\Pi$ also removes. Fix these $6+J$ directions by

$$A^\top\big(P_1^{(1)},\dots,P_N^{(1)}\big)^\top=A^\top\big(Q_1^{(1)},\dots,Q_N^{(1)}\big)^\top=0,\qquad
\sum_n\alpha_{nj}=0,\tag{14}$$

written $C\beta=0$. They only select among solutions of equal loss, so they are appended to Eq. (13), and

$$\beta=\big(M+C^\top C\big)^{-1}\rho,\tag{15}$$

when the matrix is invertible, which requires $N\ge5$ and $J\le J_{\max}$. The system has size $2N+NJ$, independent of $K$.

### Pixel corrections and normalization

With $\tilde\alpha_{nj}$ from Eq. (15) and $\tilde\Delta_n=\sum_j\tilde\alpha_{nj}H_j$, Eq. (9) at one pixel is a pixel step for $a^{(1)},u^{(1)},v^{(1)}$. The residual and, by Eq. (14), the frame corrections are removed by $A^\top$, so only the step-error term remains:

$$\begin{pmatrix}a^{(1)}\\u^{(1)}\\v^{(1)}\end{pmatrix}
=A_p^{-1}\begin{pmatrix}\sum_nw_n\tilde\Delta_n\\\sum_n\hat P_nw_n\tilde\Delta_n\\\sum_n\hat Q_nw_n\tilde\Delta_n\end{pmatrix}.\tag{16}$$

By the Frisch–Waugh–Lovell theorem, Eqs. (15) and (16) together are the joint least-squares solution of Eq. (9). Subtracting the biases gives the corrected estimates

$$\tilde a=\hat a-a^{(1)},\qquad \tilde u=\hat u-u^{(1)},\qquad \tilde v=\hat v-v^{(1)},\qquad \tilde P_n=\hat P_n-P_n^{(1)},\qquad \tilde Q_n=\hat Q_n-Q_n^{(1)},\tag{17}$$

each equal to $f^{(0)}$ up to $O(\Delta_n^2)$. They satisfy Eq. (14) rather than the conventions, from which they deviate only at first order; the normalization of Appendix D restores the conventions, leaves $\tilde\Delta_n$ unchanged, and changes the intensities only at second order. The final estimates are $\tilde\delta_n=\operatorname{atan2}(\tilde Q_n,\tilde P_n)$, $\tilde g_n=\sqrt{\tilde P_n^2+\tilde Q_n^2}$, $\tilde b=\sqrt{\tilde u^2+\tilde v^2}$, $\tilde\Phi=\operatorname{atan2}(-\tilde v,\tilde u)$, and $\tilde\Delta_n$, with errors of second order in $\Delta_n$.

### VP-AIA algorithm and cost

1. **AIA baseline.** Run AIA to convergence, finish with a pixel step, and normalize (Appendix D).
2. **Residual and projector.** Compute $r_n$ (Eq. 8), $w_n$ (Eq. 9), and $\Pi$ (Eq. 10).
3. **Pixel sums.** In one pass over the pixels, accumulate the sums of $M$ and $\rho$ (Appendix C).
4. **Frame corrections and coefficients.** Solve Eq. (15).
5. **Pixel corrections.** Evaluate $\tilde\Delta_n$ and Eq. (16) at every pixel, and form Eq. (17).
6. **Normalization.** Normalize the corrected estimates (Appendix D).

One AIA iteration costs $O(NK)$. Steps 2–6 add $O(NKJ)$ for the residual, the right-hand sides, and Eq. (16), $O(KJ^2)$ for the pixel sums, and $O\big((2N+NJ)^3\big)$ for Eq. (15): about $J+J^2/N$ AIA iterations. Besides the stack and the AIA fields, the pass stores the $J$ modes and matrices of size $2N+NJ$; $\tilde\Delta_n$ is evaluated from $\tilde\alpha_{nj}$ and $H_j$ when needed.

**Larger phase-step errors.** One pass leaves an error of second order. When that matters, remove the current step-error signal from the stack with the full model,

$$I_n'=I_n-\tilde g_n\tilde b\big[\cos(\tilde\Phi+\tilde\delta_n+\tilde\Delta_n)-\cos(\tilde\Phi+\tilde\delta_n)\big],\tag{18}$$

where $\tilde\Delta_n$ is the accumulated step error; $I'_n$ then contains only the remaining error $\Delta_n-\tilde\Delta_n$. One round runs AIA on $I'_n$ from the current $\tilde P_n,\tilde Q_n$, applies steps 2–6, adds the increment $\delta\tilde\Delta_n=\sum_j\delta\tilde\alpha_{nj}H_j$ to $\tilde\Delta_n$, and replaces the other estimates by the corrected, normalized ones, re-estimated from $I'_n$ rather than accumulated. Each increment has zero spatial and frame means, so $\tilde\Delta_n$ keeps the conventions of Eq. (6).

The remaining error enters $I'_n$ with the sensitivity at $\tilde\Delta_n$, while Eq. (9) uses the sensitivity at zero step error; the two differ at first order, so each round reduces the remaining error by a factor of order $\max_n|\Delta_n|$. The rounds are a fixed-point scheme, not a Newton step, and converge when that factor is well below one. Evaluating $w_n$ at $\tilde\Delta_n$ would make the convergence quadratic, but $w_n$ would no longer be a frame-weighted combination of $\hat u$ and $\hat v$, and the reductions of Appendix C would be lost. At convergence the solution is unbiased to all orders in $\Delta_n$.

## Noise of the corrected estimates

Consider noise $\varepsilon_n(x,y)$ of variance $\sigma_0^2$, independent across frames and pixels, and treat $P_n,Q_n$ as exact. At one pixel, the pixel-step estimate of $\Phi=\operatorname{atan2}(-v,u)$ is linear in the intensities, with error $\sum_ns_n\varepsilon_n$ by Eq. (4), where

$$\begin{pmatrix}s_1\\\vdots\\s_N\end{pmatrix}=AA_p^{-1}\begin{pmatrix}0\\-\sin\Phi/b\\-\cos\Phi/b\end{pmatrix},$$

so plain AIA gives $\sigma_\Phi^2=\sigma_0^2\sum_ns_n^2$.

**Noise of the fitted unknowns.** The residual noise at each pixel is $\Pi(\varepsilon_1,\dots,\varepsilon_N)^\top$, and each entry of $\rho$ is a pixel sum of the residual weighted by one column of Eq. (11). Since $\Pi^\top\Pi=\Pi$, the noise of $\rho$ has covariance $\sigma_0^2M$, and by Eq. (15) that of $\beta$ is $\sigma_0^2\big(M+C^\top C\big)^{-1}M\big(M+C^\top C\big)^{-1}$.

**Effect on the phase.** By Eq. (16), noise $\delta\tilde\Delta_n$ in the fitted step error changes $\Phi$ by $-\sum_ns_nw_n\,\delta\tilde\Delta_n$. Since $(s_n)$ lies in the column space of $A$, which $\Pi$ removes, this change is uncorrelated with $\sum_ns_n\varepsilon_n$ at the same pixel, and the correction adds a variance term:

$$\sigma_\Phi^2=\sigma_0^2\sum_ns_n^2+\sigma_0^2\,\gamma^\top\big(M+C^\top C\big)^{-1}M\big(M+C^\top C\big)^{-1}\gamma,\tag{19}$$

where $\gamma$, ordered as $\beta$, has zeros at the positions of $P_n^{(1)},Q_n^{(1)}$ and $s_nw_nH_j$ at the position of $\alpha_{nj}$, evaluated at the pixel.

**Uniform steps.** For uniform steps with $N\ge5$, constant $g_n$ and $b$, many fringes, and $\langle H_jH_{j'}\rangle_{x,y}=\delta_{jj'}$, the frame corrections decouple from the coefficients, the coefficient block of $M$ has entries $\tfrac12g^2b^2K\,\Pi_{nm}\cos(\delta_n-\delta_m)\,\delta_{jj'}$, and $s_nw_n=\big(1-\cos(2\Phi+2\delta_n)\big)/N$. The matrix with entries $\Pi_{nm}\cos(\delta_n-\delta_m)$ has eigenvalue $0$ on frame-constant columns, $\tfrac12$ on the first and second temporal harmonics, and $1$ on the others. The frame-varying part of $s_nw_n$ lies in the second harmonic, where the eigenvalue $\tfrac12$ doubles its contribution. Averaging Eq. (19) over the fringe phase gives

$$\frac{\sigma_\Phi^2}{\sigma_0^2\sum_ns_n^2}\approx1+\frac{\sum_jH_j(x,y)^2}{K},\qquad
\bigg\langle\frac{\sigma_\Phi^2}{\sigma_0^2\sum_ns_n^2}\bigg\rangle_{x,y}\approx1+\frac{J}{K}.\tag{20}$$

The correction raises the phase noise by a relative amount of order $J/K$, largest where $\sum_jH_j^2$ is large. Eq. (20) reads Eq. (19) under the four conditions above; for irregular steps, per-frame gains, varying contrast, or unevenly resolved modes, use Eq. (19).

**Noise of the baseline steps.** The first term of Eq. (19) is `aia_noise.md` Eq. (10), the phase variance at known $\delta_n$ and $g_n$. Estimating both from the same frames adds the $O(1/K)$ term of `aia_noise.md` Eqs. (24) and (29); the zeros of $\gamma$ say only that the first-order frame corrections leave $\Phi$ unchanged to first order. The two terms add; their cross-correlation is not derived here.

## Appendix A. Unknowns and identifiability

For $N$ frames and $K$ pixels, Eq. (1) has $3K+2N+NK$ unknowns: $a,b,\Phi$ at every pixel, $g_n,\delta_n$ for every frame, and $\Delta_n$ at every pixel of every frame. The $NK$ intensities give at most $NK$ equations. The spatial-mean and frame-mean conventions impose $N+K-1$ independent conditions, since their total-sum condition is shared; the phase origin and contrast scale add two. At least

$$(3K+2N+NK)-NK-(N+K-1)-2=2K+N-1\tag{A1}$$

degrees of freedom remain, so an unrestricted $\Delta_n$ cannot be recovered. The same count holds for the first-order unknowns, $3K$ pixel biases, $2N$ frame biases, and $NK$ values of $\Delta_n$; the four quadrature-frame conditions remove four more, leaving at least $2K+N-5$.

With Eq. (6), the unknowns are $3K+2N+NJ$. With the $NK$ equations, the $J$ coefficient-mean constraints, and the six conditions of phase origin, contrast scale, and quadrature frame,

$$\nu=3K+2N+NJ-NK-J-6\tag{A2}$$

degrees of freedom remain, and $\nu\le0$ gives Eq. (7), with $N\ge3$. At $N=3$, $J_{\max}=0$: AIA is determined, but no mode is. At $N=4$ the count permits modes but none is identifiable: $\operatorname{span}\{1,P_n,Q_n\}$ has codimension one in $\mathbb R^4$, so for each $H_j$ some nonzero $\alpha_{nj}$ with zero frame mean has $\alpha_{nj}P_n$ and $\alpha_{nj}Q_n$ in that span, and its intensity change is absorbed by $a^{(1)},u^{(1)},v^{(1)}$. Identifying modes therefore requires $N\ge5$. The count is necessary only; the fit must still determine all unknowns.

## Appendix B. Perturbation series and baseline bias

At each pixel, write $\theta_n=\Phi+\delta_n$. The Taylor expansion about $\Delta_n=0$,

$$\cos(\theta_n+\Delta_n)=\sum_{k=0}^{\infty}\frac{\Delta_n^k}{k!}\cos\left(\theta_n+\frac{k\pi}{2}\right),\tag{B1}$$

turns Eq. (1), in the variables of Eq. (2), into

$$I_n=a+P_nu+Q_nv+\sum_{k=1}^{\infty}\kappa_{n,k}\Delta_n^k,\qquad
\kappa_{n,k}=\begin{cases}
\dfrac{(-1)^{(k-1)/2}}{k!}\big(P_nv-Q_nu\big), & k\text{ odd},\\[2ex]
\dfrac{(-1)^{k/2}}{k!}\big(P_nu+Q_nv\big), & k\text{ even},
\end{cases}\tag{B2}$$

since $g_nb\cos\theta_n=P_nu+Q_nv$ and $-g_nb\sin\theta_n=P_nv-Q_nu$. The zeroth order is the piston model, Eq. (3); the rest, $e_n=\sum_{k\ge1}\kappa_{n,k}\Delta_n^k$, is what the piston model cannot represent.

**Bias of the pixel step.** With the frame-parameter errors $\varepsilon_{P_n}=\hat P_n-P_n$ and $\varepsilon_{Q_n}=\hat Q_n-Q_n$, the data are $I_n=a+\hat P_nu+\hat Q_nv+s_n$ with $s_n=e_n-\varepsilon_{P_n}u-\varepsilon_{Q_n}v$, and Eq. (4) gives

$$\begin{pmatrix}\hat a\\\hat u\\\hat v\end{pmatrix}=\begin{pmatrix}a\\u\\v\end{pmatrix}+A_p^{-1}A^\top\begin{pmatrix}s_1\\\vdots\\s_N\end{pmatrix}.\tag{B3}$$

The pixel fields are biased by the projection of $s_n$ onto the regressors: the leakage of the step-error intensity $e_n$, and the frame-parameter errors that the frame step produces. If the frame parameters carry no bias, the pixel step takes the part of $e$ in the column space of $A$, and the residual keeps only $\Pi e$; fitting the residual with $e_n$ itself would therefore underestimate $\Delta_n$, which is why Eq. (11) compares the residual with the leftover of the signal.

**Ordering.** Only the estimates depend on $\Delta_n$; the true parameters have no expansion. To give orders a meaning, scale a fixed shape, $\Delta_n=\epsilon D_n$, so that each estimate is a function of the scalar $\epsilon$. At $\epsilon=0$ the piston model describes the data exactly and the conventions leave one solution, so a noiseless fit returns the exact parameters. The estimates solve AIA's normal equations with the normalization conditions, which determine them uniquely at $\epsilon=0$, so they depend analytically on $\epsilon$ near it, which gives Eq. (5).

**First-order residual.** Substituting Eq. (5) into $r_n=I_n-\big(\hat a+\hat P_n\hat u+\hat Q_n\hat v\big)$ with Eq. (B2) and keeping first-order terms gives Eq. (9). The sensitivity $\kappa_{n,1}=P_nv-Q_nu$ multiplies the first-order $\Delta_n$, so evaluating it at the AIA estimates, as $w_n$, costs $O(\Delta_n^2)$.

## Appendix C. Normal equations

In this appendix $u,v,P_n,Q_n$ denote the baseline estimates, hats suppressed, and $z_n$ is the column of Eq. (11). Setting the derivatives of Eq. (12) with respect to $P_n^{(1)}$, $Q_n^{(1)}$, and $\alpha_{nj}$ to zero gives, for $n=1,\dots,N$ and $j=1,\dots,J$,

$$\begin{aligned}
\sum_m\Pi_{nm}\Big[P_m^{(1)}\sum_{x,y}u^2+Q_m^{(1)}\sum_{x,y}uv-\sum_{j'}\alpha_{mj'}\sum_{x,y}uw_mH_{j'}\Big]&=-\sum_{x,y}ur_n,\\
\sum_m\Pi_{nm}\Big[P_m^{(1)}\sum_{x,y}uv+Q_m^{(1)}\sum_{x,y}v^2-\sum_{j'}\alpha_{mj'}\sum_{x,y}vw_mH_{j'}\Big]&=-\sum_{x,y}vr_n,\\
\sum_m\Pi_{nm}\Big[-P_m^{(1)}\sum_{x,y}uw_nH_j-Q_m^{(1)}\sum_{x,y}vw_nH_j+\sum_{j'}\alpha_{mj'}\sum_{x,y}w_nw_mH_jH_{j'}\Big]&=\sum_{x,y}w_nH_jr_n.
\end{aligned}\tag{C1}$$

Since $w_n=P_nv-Q_nu$, every pixel sum containing $w_n$ reduces to sums without a frame index:

$$\begin{aligned}
\sum_{x,y}uw_mH_j&=P_m\sum_{x,y}uvH_j-Q_m\sum_{x,y}u^2H_j,\qquad
\sum_{x,y}vw_mH_j=P_m\sum_{x,y}v^2H_j-Q_m\sum_{x,y}uvH_j,\\
\sum_{x,y}w_nw_mH_jH_{j'}&=P_nP_m\sum_{x,y}v^2H_jH_{j'}-\big(P_nQ_m+Q_nP_m\big)\sum_{x,y}uvH_jH_{j'}+Q_nQ_m\sum_{x,y}u^2H_jH_{j'},\\
\sum_{x,y}w_nH_jr_n&=P_n\sum_{x,y}vH_jr_n-Q_n\sum_{x,y}uH_jr_n.
\end{aligned}\tag{C2}$$

The matrix therefore needs three scalar sums, three sums per mode, and three per pair of modes; the right-hand sides need four sums per frame and mode, all in one pass. Under whitening, $\sum_{x,y}uv=0$ and $\sum_{x,y}u^2=\sum_{x,y}v^2$. In block form, Eq. (13) has

$$M=\begin{pmatrix}
\Pi\sum_{x,y}u^2&\Pi\sum_{x,y}uv&-D_u\\
\Pi\sum_{x,y}uv&\Pi\sum_{x,y}v^2&-D_v\\
-D_u^\top&-D_v^\top&E
\end{pmatrix},$$

$$\rho=\Big(-\sum_{x,y}ur_1,\dots,-\sum_{x,y}ur_N,\ -\sum_{x,y}vr_1,\dots,-\sum_{x,y}vr_N,\ \sum_{x,y}w_1H_1r_1,\dots,\sum_{x,y}w_NH_Jr_N\Big)^\top,$$

where $\Pi\sum_{x,y}u^2$ is $\Pi$ times a scalar. The coupling and coefficient blocks are

$$D_u=\begin{pmatrix}
\Pi_{11}d^u_1&\cdots&\Pi_{1N}d^u_N\\
\vdots&&\vdots\\
\Pi_{N1}d^u_1&\cdots&\Pi_{NN}d^u_N
\end{pmatrix},\qquad
d^u_m=\Big(\sum_{x,y}uw_mH_1,\ \dots,\ \sum_{x,y}uw_mH_J\Big),$$

of size $N\times NJ$, with $D_v$ the same for $v$, and

$$E=\begin{pmatrix}
\Pi_{11}E_{11}&\cdots&\Pi_{1N}E_{1N}\\
\vdots&&\vdots\\
\Pi_{N1}E_{N1}&\cdots&\Pi_{NN}E_{NN}
\end{pmatrix},\qquad
E_{nm}=\begin{pmatrix}
\sum_{x,y}w_nw_mH_1H_1&\cdots&\sum_{x,y}w_nw_mH_1H_J\\
\vdots&&\vdots\\
\sum_{x,y}w_nw_mH_JH_1&\cdots&\sum_{x,y}w_nw_mH_JH_J
\end{pmatrix}.$$

The conditions of Eq. (14) are $C\beta=0$ with

$$C=\begin{pmatrix}
A^\top&0&0\\
0&A^\top&0\\
0&0&\big(I_J\ \cdots\ I_J\big)
\end{pmatrix},$$

where $I_J$, the $J\times J$ identity, is repeated $N$ times; $C$ has size $(6+J)\times(2N+NJ)$.

## Appendix D. Quadrature frame and normalization

### Quadrature frame

The piston model, Eq. (3), is unchanged by $(u,v)\to(u,v)T$, $(P_n,Q_n)\to(P_n,Q_n)T^{-\top}$ for any invertible $2\times2$ matrix $T$, and by the shifts $(P_n,Q_n)\to(P_n+p,Q_n+q)$, $a\to a-pu-qv$. The phase origin and contrast scale fix a common rotation and scale of $T$. The remaining four freedoms, shear, anisotropic scaling, and the two shifts, are fixed by the quadrature-frame conventions

$$\sum_{x,y}(a-\bar a)u=\sum_{x,y}(a-\bar a)v=0,\qquad\sum_{x,y}u^2=\sum_{x,y}v^2,\qquad\sum_{x,y}uv=0,$$

where bars denote pixel means. Both the baseline and the corrected estimates are brought to them.

### Normalization

Apply the following steps in order; each preserves the conditions of the previous ones and leaves $a+P_nu+Q_nv$ unchanged.

1. **Shift.** Solve
   $$\begin{pmatrix}
   \sum_{x,y}(u-\bar u)^2&\sum_{x,y}(u-\bar u)(v-\bar v)\\
   \sum_{x,y}(u-\bar u)(v-\bar v)&\sum_{x,y}(v-\bar v)^2
   \end{pmatrix}
   \begin{pmatrix}p\\q\end{pmatrix}
   =\begin{pmatrix}\sum_{x,y}(a-\bar a)u\\\sum_{x,y}(a-\bar a)v\end{pmatrix},$$
   and apply $a\leftarrow a-pu-qv$, $P_n\leftarrow P_n+p$, $Q_n\leftarrow Q_n+q$.
2. **Whitening.** With $G=\begin{pmatrix}\sum_{x,y}u^2&\sum_{x,y}uv\\\sum_{x,y}uv&\sum_{x,y}v^2\end{pmatrix}$, apply $(u,v)^\top\leftarrow G^{-1/2}(u,v)^\top$ and $(P_n,Q_n)^\top\leftarrow G^{1/2}(P_n,Q_n)^\top$. The shift condition is preserved, since $a$ is unchanged and the new $u,v$ are combinations of the old.
3. **Phase origin.** With $\delta_1=\operatorname{atan2}(Q_1,P_1)$ and $R=\begin{pmatrix}\cos\delta_1&\sin\delta_1\\-\sin\delta_1&\cos\delta_1\end{pmatrix}$, apply $(P_n,Q_n)^\top\leftarrow R(P_n,Q_n)^\top$ and $(u,v)^\top\leftarrow R(u,v)^\top$, so that $\delta_1=0$.
4. **Contrast scale.** With $s=\operatorname{median}_n\sqrt{P_n^2+Q_n^2}$, apply $P_n\leftarrow P_n/s$, $Q_n\leftarrow Q_n/s$, $u\leftarrow su$, $v\leftarrow sv$.

## References

- R. Frisch and F. V. Waugh, "Partial time regressions as compared with individual trends," *Econometrica* **1**(4), 387–401 (1933).
- M. C. Lovell, "Seasonal adjustment of economic time series and multiple regression analysis," *Journal of the American Statistical Association* **58**(304), 993–1010 (1963).
- G. H. Golub and V. Pereyra, "The differentiation of pseudo-inverses and nonlinear least squares problems whose variables separate," *SIAM Journal on Numerical Analysis* **10**(2), 413–432 (1973).
- D. C. Hoaglin and R. E. Welsch, "The hat matrix in regression and ANOVA," *The American Statistician* **32**(1), 17–22 (1978).
- Z. Wang and B. Han, "Advanced iterative algorithm for phase extraction of randomly phase-shifted interferograms," *Optics Letters* **29**(14), 1671–1673 (2004).
- Y. Chen and Q. Kemao, "General iterative algorithm for phase-extraction from fringe patterns with random phase-shifts, intensity harmonics and non-uniform phase-shift distribution," *Optics Express* **29**(19), 30905–30926 (2021).
