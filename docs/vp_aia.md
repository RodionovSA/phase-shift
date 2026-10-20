# Variable-Projection AIA (VP-AIA)

The advanced iterative algorithm (AIA; `aia.md`; Wang and Han, 2004) is a simple and very efficient method for phase recovery in phase-shifting interferometry. However, its model assumes spatially uniform phase steps, which limits the accuracy of the reconstructed phase. In many real scenarios, high stability of the interferometer is challenging, and this error becomes a bottleneck.

VP-AIA improves the standard AIA to overcome this problem while preserving its high efficiency. It expands the full-model solution around the AIA baseline and computes higher-order corrections in powers of the non-uniformity. These corrections are similar to Gauss–Newton steps from the AIA solution with a fixed Jacobian. However, VP-AIA splits the frame and pixel variables with the variable projection method for separable least-squares problems (Golub and Pereyra, 1973), which significantly reduces the cost and increases the computation speed.

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

### Spatial modes

An unrestricted $\Delta_n$ has more unknowns than the data provide equations (Appendix A). To reduce the unknowns, $\Delta_n$ is expanded on $J$ fixed, linearly independent spatial functions with frame-dependent amplitudes,

$$\Delta_n(x,y)=\sum_{j=1}^J\alpha_{nj}H_j(x,y),\qquad
\langle H_j\rangle_{x,y}=0,\qquad
\langle\alpha_{nj}\rangle_n=0,\tag{2}$$

which enforce the spatial-mean and frame-mean conventions. A unique solution requires (Appendix A)

$$J\le J_{\max}\equiv\left\lfloor\frac{(N-3)K}{N-1}\right\rfloor-2,\tag{3}$$

and identifying any mode requires $N\ge5$. Since $K\gg N$ in practice, $J_{\max}\approx K(N-3)/(N-1)$, about $K/2$ at $N=5$.

### Cost of the full model

The piston limit $\Delta_n\equiv0$ is linear in the pixel fields for fixed frame parameters and in the frame parameters for fixed pixel fields, and AIA alternates these two fits, each with one small normal matrix. A spatially varying $\Delta_n$ breaks this split: it enters inside the cosine and depends on both frame and pixel. The full model can be solved as a general nonlinear least-squares problem, as the general iterative algorithm (GIA) of Chen and Kemao (2021) does, at about 72 times the computation time of AIA (558.8 s against 7.7 s at equal iteration counts). VP-AIA instead keeps the AIA solution as a baseline, moves it to the quadrature frame of the true parameters, and recovers $\Delta_n$ together with the remaining bias as small corrections.

## AIA baseline

With $\Delta_n\equiv0$ and the quadrature variables

$$u=b\cos\Phi,\qquad v=-b\sin\Phi,\qquad P_n=g_n\cos\delta_n,\qquad Q_n=g_n\sin\delta_n,\tag{4}$$

Eq. (1) is the piston model

$$I_n(x,y)=a(x,y)+P_nu(x,y)+Q_nv(x,y).\tag{5}$$

AIA (`aia.md`) fits it and returns $\hat a,\hat u,\hat v,\hat P_n,\hat Q_n$, brought to the conventions above and to those of Appendix B. Its last pixel step solves, at each pixel,

$$\big(\hat a,\hat u,\hat v\big)^\top=A_p^{-1}A^\top\mathbf i,\qquad A_p=A^\top A,\tag{6}$$

where $\mathbf i=(I_1,\dots,I_N)^\top$ and $A$ has rows $(1,\hat P_n,\hat Q_n)$. Since the data contain $\Delta_n$, the AIA estimates absorb part of it and are biased at first order (Appendix C).

## Expansion around the AIA solution

The piston model, Eq. (5), fits the measured data, and its parameters absorb a bias from the model mismatch caused by the spatial non-uniformity of the phase steps. However, if the phase steps are spatially uniform, $\Delta_n(x,y)=0$, the true solution equals the AIA solution. Based on this, for small $\epsilon=\max_{n,x,y}|\Delta_n|$ we expand each true parameter as the AIA estimate plus corrections ordered by powers of $\epsilon$:

$$f=\hat f+\sum_{k\ge1}f^{(k)},\qquad \alpha_{nj}=\sum_{k\ge1}\alpha_{nj}^{(k)},\qquad f^{(k)},\,\alpha_{nj}^{(k)}=O(\epsilon^k),\qquad f\in\{a,u,v,P_n,Q_n\}.\tag{7}$$

The corrections are found one order at a time. They enter Eq. (1) nonlinearly and cannot be fitted directly. Instead, at order $k$ we subtract from the data the model evaluated at everything already known, and fit only what is left.

Let $f_{k-1}=\hat f+\sum_{m<k}f^{(m)}$ and $\Delta_{n,k-1}=\sum_{m<k}\Delta_n^{(m)}$, with $\Delta_n^{(m)}=\sum_j\alpha_{nj}^{(m)}H_j$, be the estimates accumulated up to order $k-1$, with components $a_{k-1},u_{k-1},v_{k-1},P_{n,k-1},Q_{n,k-1}$. The part of the data they leave unexplained is the residual of the full model, Eq. (1) in the variables of Eq. (4),

$$r_n^{(k)}=I_n-a_{k-1}-\big(P_{n,k-1}u_{k-1}+Q_{n,k-1}v_{k-1}\big)\cos\Delta_{n,k-1}-\big(P_{n,k-1}v_{k-1}-Q_{n,k-1}u_{k-1}\big)\sin\Delta_{n,k-1}.\tag{8}$$

At $k=1$ nothing is accumulated yet, $f_0=\hat f$ and $\Delta_{n,0}=0$, so $r_n^{(1)}$ is the residual of the AIA pixel step, Eq. (6). For $k\ge2$, $r_n^{(k)}$ is what remains after the first $k-1$ orders. Since the accumulated estimates match the true parameters up to order $k-1$, $r_n^{(k)}$ is of order $\epsilon^k$. Since Eq. (8) evaluates the model exactly, it contains all products of the lower-order corrections.

Substituting the true parameters, Eq. (7), into Eq. (1) and subtracting Eq. (8) leaves the order-$k$ corrections at leading order:

$$r_n^{(k)}=\big(a^{(k)}+\hat P_nu^{(k)}+\hat Q_nv^{(k)}\big)+\hat uP_n^{(k)}+\hat vQ_n^{(k)}+\hat w_n\Delta_n^{(k)}+O(\epsilon^{k+1}),\qquad\hat w_n=\hat P_n\hat v-\hat Q_n\hat u.\tag{9}$$

The coefficients in Eq. (9) are taken at the AIA solution rather than at $f_{k-1}$. The two differ by $O(\epsilon)$, which only adds to the $O(\epsilon^{k+1})$ term. The right-hand side is therefore the same linear operator at every order, and only the residual changes. Each order is one linear least-squares fit of $r_n^{(k)}$ for $f^{(k)}$ and $\alpha_{nj}^{(k)}$, always with the same matrix.

### Quadrature frame of the expansion point

Eq. (7) holds only up to a freedom of the piston model. Eq. (5) is unchanged by

$$(u,v)\to(u,v)T,\qquad(P_n,Q_n)\to(P_n,Q_n)T^{-\top},\qquad(P_n,Q_n)\to(P_n+p,Q_n+q),\qquad a\to a-pu-qv,\tag{10}$$

for any invertible $2\times2$ matrix $T$ and shifts $p,q$. The phase origin and contrast scale fix a common rotation and scale of $T$, and the conventions of Appendix B fix the remaining four parameters. Eq. (1) is not unchanged by Eq. (10) when $\Delta_n\ne0$, since the move changes $w_n=P_nv-Q_nu$, so the true parameters generally do not satisfy these conventions. The expansion is taken about the AIA solution in its conventional frame. Its difference from the true frame does not vanish with $\epsilon$ and sets a floor on the accuracy that higher orders do not remove (Appendix D). The size of this floor depends on how closely the true fields satisfy the conventions of Appendix B.

### Removing the pixel corrections

At each order, Eq. (9) contains two kinds of unknowns. The pixel corrections $a^{(k)},u^{(k)},v^{(k)}$ take a separate value at every pixel, while $P_n^{(k)},Q_n^{(k)}$ and $\alpha_{nj}^{(k)}$ are shared by all pixels. The pixel corrections are by far the most numerous, but at one pixel they enter only as $A\big(a^{(k)},u^{(k)},v^{(k)}\big)^\top$, the same combination that the pixel step of Eq. (6) fits. They can therefore be removed by a projection.

At one pixel, define the $N\times N$ residual-maker of the pixel step (Hoaglin and Welsch, 1978),

$$\Pi=I-AA_p^{-1}A^\top.\tag{11}$$

$\Pi$ depends only on $\hat P_n,\hat Q_n$, so it is the same at every pixel. It satisfies $\Pi A=0$, $\Pi^2=\Pi$, $\Pi^\top=\Pi$, and has rank $N-3$. Applying $\Pi$ to Eq. (9) at one pixel removes the pixel corrections:

$$\Pi\mathbf r^{(k)}=\Pi\mathbf z^{(k)},\qquad
z_n^{(k)}=\hat w_n\Delta_n^{(k)}+\hat uP_n^{(k)}+\hat vQ_n^{(k)},\tag{12}$$

where $\mathbf r^{(k)}=\big(r_1^{(k)},\dots,r_N^{(k)}\big)^\top$, and $\mathbf z^{(k)}$ is formed in the same way. At $k=1$ the residual is that of the pixel step, $\mathbf r^{(1)}=\Pi\mathbf i$, so $\Pi$ leaves it unchanged. For $k\ge2$ it does not, and $\Pi$ must be applied to the residual as well.

Eq. (12) holds at every pixel and contains only the unknowns shared by all pixels. Eliminating a group of regressors by applying the residual-maker to the data and to the remaining regressors does not change the least-squares estimates of the rest (Frisch and Waugh, 1933; Lovell, 1963). A fit of Eq. (12) therefore gives the same shared unknowns as a joint fit of Eq. (9). The pixel corrections then follow from Eq. (9), one pixel at a time.

### Fit of the shared unknowns

Eq. (12) holds at every pixel, and the shared unknowns are fitted over all pixels by least squares, with the loss

$$\mathcal L_k=\sum_{x,y}\big\|\Pi\big(\mathbf r^{(k)}-\mathbf z^{(k)}\big)\big\|^2.\tag{13}$$

Collect the unknowns of order $k$ in $\beta=\big(P_1^{(k)},\dots,P_N^{(k)},\,Q_1^{(k)},\dots,Q_N^{(k)},\,\alpha_{11}^{(k)},\dots,\alpha_{NJ}^{(k)}\big)^\top$. Since $\mathbf z^{(k)}$ is linear in $\beta$, setting the gradient of Eq. (13) to zero gives the normal equations

$$M\beta=\rho.\tag{14}$$

Their matrix and right-hand side are pixel sums of the expansion point, the modes, and the projected residual (Appendix E). The matrix depends only on the AIA solution and the modes, so it is the same at every order. Only $\rho$ is recomputed, in one pass over the pixels.

**Undetermined directions.** $\Pi$ removes the columns $(1,\dots,1)^\top$, $(\hat P_1,\dots,\hat P_N)^\top$, and $(\hat Q_1,\dots,\hat Q_N)^\top$, so adding any combination of them to $\big(P_n^{(k)}\big)$ or $\big(Q_n^{(k)}\big)$ leaves Eq. (12) unchanged. These six directions are infinitesimal changes of the quadrature frame, which leave the piston model unchanged. A frame-constant $\alpha_{nj}^{(k)}=c_j$ adds $\big(\sum_jc_jH_j\big)\hat w_n$, which $\Pi$ also removes. Fix these $6+J$ directions by

$$A^\top\big(P_1^{(k)},\dots,P_N^{(k)}\big)^\top=A^\top\big(Q_1^{(k)},\dots,Q_N^{(k)}\big)^\top=0,\qquad
\sum_n\alpha_{nj}^{(k)}=0,\tag{15}$$

written $C\beta=0$. The conditions only select among solutions of equal loss, so they are appended to Eq. (14), and

$$\beta=\big(M+C^\top C\big)^{-1}\rho,\tag{16}$$

when the matrix is invertible, which requires $N\ge5$ and $J\le J_{\max}$. The system has size $2N+NJ$, independent of $K$.

### Pixel corrections and accumulated estimates

With $\beta$ from Eq. (16), Eq. (9) at one pixel is a pixel step for $a^{(k)},u^{(k)},v^{(k)}$. By Eq. (15), $A^\top$ removes the frame corrections, and only the terms with the step error remain:

$$\begin{pmatrix}a^{(k)}\\u^{(k)}\\v^{(k)}\end{pmatrix}=A_p^{-1}A^\top\big(\mathbf r^{(k)}-\mathbf d^{(k)}\big),\qquad
d_n^{(k)}=\hat w_n\Delta_n^{(k)},\tag{17}$$

At $k=1$ the residual is that of the pixel step, so $A^\top\mathbf r^{(1)}=0$; for $k\ge2$ it is not. By the Frisch–Waugh–Lovell theorem, Eqs. (16) and (17) together are the joint least-squares solution of Eq. (9).

The corrections of order $k$ are then added to the estimates,

$$f_k=f_{k-1}+f^{(k)},\qquad \alpha_{nj}=\sum_{m\le k}\alpha_{nj}^{(m)},\qquad f\in\{a,u,v,P_n,Q_n\}.\tag{18}$$

Each $\alpha_{nj}^{(k)}$ has zero frame mean by Eq. (15), so the accumulated $\Delta_n$ keeps the conventions of Eq. (2).

After the last order $k_{\max}$, the estimates are normalized by Appendix B. The corrections move them off the conventions at first order, and the normalization returns them to the conventional frame that the expansion describes (Appendix D); it leaves $\tilde\Delta_n$ unchanged. The final estimates are $\tilde\delta_n=\operatorname{atan2}(\tilde Q_n,\tilde P_n)$, $\tilde g_n=\sqrt{\tilde P_n^2+\tilde Q_n^2}$, $\tilde b=\sqrt{\tilde u^2+\tilde v^2}$, $\tilde\Phi=\operatorname{atan2}(-\tilde v,\tilde u)$, and $\tilde\Delta_n$. The error of $\tilde\Delta_n$ is of order $\epsilon^{k_{\max}+1}$ above the floor of §"Quadrature frame of the expansion point"; the other estimates improve more slowly with $k_{\max}$.

## VP-AIA algorithm and cost

The number of orders $k_{\max}$ is chosen by the user, as the number of iterations would be for Gauss–Newton. Each order reduces the error by a factor of order $\epsilon$, so more orders are worth computing only while the remaining error $\epsilon^{k+1}$ is above the noise.

1. **AIA baseline.** Run AIA to convergence, finish with a pixel step, normalize (Appendix B), and form $\Pi$ (Eq. 11).
2. **Matrix.** In one pass over the pixels, accumulate the sums $F$ (Appendix E), form $M$ and $C$, and factor $M+C^\top C$ once.
3. **Orders $k=1,\dots,k_{\max}$.** In one pass over the pixels, compute $\mathbf r^{(k)}$ (Eq. 8) at the accumulated estimates and accumulate the sums $L$. Solve Eq. (16), evaluate Eq. (17) at every pixel, and accumulate by Eq. (18). The pass of $k=1$ can be merged with step 2.
4. **Normalization.** Normalize by Appendix B and form the final estimates.

One AIA iteration costs $O(NK)$. The sums $F$ cost $O(KJ^2)$ once, and the factorization $O\big((2N+NJ)^3\big)$ once. Each order costs one pass of $O(NKJ)$ for the residual, the sums $L$, and Eq. (17), and one solve with the factored matrix. In AIA iterations, the first order costs about $J+J^2/N$, and each further order about $J$. Besides the stack and the current estimates, the method stores the $J$ modes, the sums $F$, and matrices of size $2N+NJ$; $\Delta_n$ is evaluated from $\alpha_{nj}$ and $H_j$ when needed.

## Noise of the corrected estimates

Consider noise $\varepsilon_n(x,y)$ of variance $\sigma_0^2$, independent across frames and pixels, and treat $P_n,Q_n$ as exact. At one pixel, the pixel-step estimate of $\Phi=\operatorname{atan2}(-v,u)$ is linear in the intensities, with error $\sum_ns_n\varepsilon_n$ by Eq. (6), where

$$\begin{pmatrix}s_1\\\vdots\\s_N\end{pmatrix}=AA_p^{-1}\begin{pmatrix}0\\-\sin\Phi/b\\-\cos\Phi/b\end{pmatrix},$$

so plain AIA gives $\sigma_\Phi^2=\sigma_0^2\sum_ns_n^2$.

**Higher orders.** Noise enters the corrections through the first order. To first order in the noise, the noise in $\mathbf r^{(k)}$ for $k\ge2$ is the part of $\mathbf r^{(1)}$ that the first order did not fit, and the normal equations of the first order map it to zero. The orders $k\ge2$ use the same matrix, so they leave the noise unchanged up to a relative correction of order $\epsilon$. The noise below is therefore that of the first order and holds for any $k_{\max}$.

**Noise of the fitted unknowns.** The residual noise at each pixel is $\Pi(\varepsilon_1,\dots,\varepsilon_N)^\top$, and each entry of $\rho$ at $k=1$ is a pixel sum of the residual weighted by one column of Eq. (12). Since $\Pi^\top\Pi=\Pi$, the noise of $\rho$ has covariance $\sigma_0^2M$, and by Eq. (16) that of $\beta$ is $\sigma_0^2\big(M+C^\top C\big)^{-1}M\big(M+C^\top C\big)^{-1}$.

**Effect on the phase.** By Eq. (17), noise $\delta\tilde\Delta_n$ in the fitted step error changes $\Phi$ by $-\sum_ns_n\hat w_n\,\delta\tilde\Delta_n$. Since $(s_n)$ lies in the column space of $A$, which $\Pi$ removes, this change is uncorrelated with $\sum_ns_n\varepsilon_n$ at the same pixel, and the correction adds a variance term:

$$\sigma_\Phi^2=\sigma_0^2\sum_ns_n^2+\sigma_0^2\,\gamma^\top\big(M+C^\top C\big)^{-1}M\big(M+C^\top C\big)^{-1}\gamma,\tag{19}$$

where $\gamma$, ordered as $\beta$, has zeros at the positions of $P_n^{(1)},Q_n^{(1)}$ and $s_n\hat w_nH_j$ at the position of $\alpha_{nj}^{(1)}$, evaluated at the pixel.

**Uniform steps.** For uniform steps with $N\ge5$, constant $g_n$ and $b$, many fringes, and $\langle H_jH_{j'}\rangle_{x,y}=\delta_{jj'}$, the frame corrections decouple from the coefficients, the coefficient block of $M$ has entries $\tfrac12g^2b^2K\,\Pi_{nm}\cos(\delta_n-\delta_m)\,\delta_{jj'}$, and $s_n\hat w_n=\big(1-\cos(2\Phi+2\delta_n)\big)/N$. The matrix with entries $\Pi_{nm}\cos(\delta_n-\delta_m)$ has eigenvalue $0$ on frame-constant columns, $\tfrac12$ on the first and second temporal harmonics, and $1$ on the others. The frame-varying part of $s_n\hat w_n$ lies in the second harmonic, where the eigenvalue $\tfrac12$ doubles its contribution. Averaging Eq. (19) over the fringe phase gives

$$\frac{\sigma_\Phi^2}{\sigma_0^2\sum_ns_n^2}\approx1+\frac{\sum_jH_j(x,y)^2}{K},\qquad
\bigg\langle\frac{\sigma_\Phi^2}{\sigma_0^2\sum_ns_n^2}\bigg\rangle_{x,y}\approx1+\frac{J}{K}.\tag{20}$$

The correction raises the phase noise by a relative amount of order $J/K$, largest where $\sum_jH_j^2$ is large. Eq. (20) reads Eq. (19) under the four conditions above; for irregular steps, per-frame gains, varying contrast, or unevenly resolved modes, use Eq. (19).

**Noise of the baseline steps.** The first term of Eq. (19) is `aia_noise.md` Eq. (10), the phase variance at known $\delta_n$ and $g_n$. Estimating both from the same frames adds the $O(1/K)$ term of `aia_noise.md` Eqs. (24) and (29); the zeros of $\gamma$ say only that the frame corrections of the first order leave $\Phi$ unchanged to first order. The two terms add; their cross-correlation is not derived here.

## Appendix A. Unknowns and identifiability

For $N$ frames and $K$ pixels, Eq. (1) has $3K+2N+NK$ unknowns: $a,b,\Phi$ at every pixel, $g_n,\delta_n$ for every frame, and $\Delta_n$ at every pixel of every frame. The $NK$ intensities give at most $NK$ equations. The spatial-mean and frame-mean conventions impose $N+K-1$ independent conditions, since their total-sum condition is shared; the phase origin and contrast scale add two. At least

$$(3K+2N+NK)-NK-(N+K-1)-2=2K+N-1\tag{A1}$$

degrees of freedom remain, so an unrestricted $\Delta_n$ cannot be recovered. The quadrature-frame freedoms of Eq. (10) are not counted: they are not freedoms of Eq. (1).

With Eq. (2), the unknowns are $3K+2N+NJ$. The spatial mean holds by construction, the frame mean imposes $J$ constraints, and the phase origin and contrast scale two more. With the $NK$ equations,

$$\nu=3K+2N+NJ-NK-J-2\tag{A2}$$

degrees of freedom remain, and $\nu\le0$ gives Eq. (3). At $N=3$ the bound is negative, and no mode can be recovered. At $N=4$ the count permits modes, but none is identifiable near $\Delta_n=0$. There, a small change $c_n$ of $\alpha_{nj}$ changes $I_n$ by $c_nw_nH_j$, with $w_n=P_nv-Q_nu$. Since $\operatorname{span}\{1,P_n,Q_n\}$ has codimension one in $\mathbb R^4$, some nonzero $c_n$ with zero frame mean has $c_nP_n$ and $c_nQ_n$ in that span. The intensity change then has the form $\delta a+P_n\delta u+Q_n\delta v$ and is absorbed by the pixel fields. Identifying modes therefore requires $N\ge5$. The count is a necessary condition only.

## Appendix B. Quadrature frame and normalization

### Quadrature frame

The AIA baseline is brought to the quadrature-frame conventions

$$\sum_{x,y}(a-\bar a)u=\sum_{x,y}(a-\bar a)v=0,\qquad\sum_{x,y}u^2=\sum_{x,y}v^2,\qquad\sum_{x,y}uv=0,$$

where bars denote pixel means. They select one member of the family of Eq. (10).

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

## Appendix C. Perturbation series and baseline bias

Write $\theta_n=\Phi+\delta_n$ at each pixel. Expanding the cosine about $\Delta_n=0$,

$$\cos(\theta_n+\Delta_n)=\sum_{k=0}^{\infty}\frac{\Delta_n^k}{k!}\cos\left(\theta_n+\frac{k\pi}{2}\right),\tag{C1}$$

turns Eq. (1), in the variables of Eq. (4), into

$$I_n=a+P_nu+Q_nv+\sum_{k=1}^{\infty}\kappa_{n,k}\Delta_n^k,\qquad
\kappa_{n,k}=\begin{cases}
\dfrac{(-1)^{(k-1)/2}}{k!}\big(P_nv-Q_nu\big), & k\text{ odd},\\[2ex]
\dfrac{(-1)^{k/2}}{k!}\big(P_nu+Q_nv\big), & k\text{ even},
\end{cases}\tag{C2}$$

where $g_nb\cos\theta_n=P_nu+Q_nv$ and $-g_nb\sin\theta_n=P_nv-Q_nu$. The zeroth-order term is the piston model, Eq. (5). The rest, $e_n=\sum_{k\ge1}\kappa_{n,k}\Delta_n^k$, is the part of the intensity that the piston model cannot represent. Summing the series gives the exact form

$$I_n=a+\big(P_nu+Q_nv\big)\cos\Delta_n+\big(P_nv-Q_nu\big)\sin\Delta_n.\tag{C3}$$

**Bias of the pixel step.** AIA fits the piston model to data that also contain $e_n$, and its frame parameters carry the errors $\varepsilon_{P_n}=\hat P_n-P_n$ and $\varepsilon_{Q_n}=\hat Q_n-Q_n$. The data can then be written as $I_n=a+\hat P_nu+\hat Q_nv+s_n$ with $s_n=e_n-\varepsilon_{P_n}u-\varepsilon_{Q_n}v$, and the pixel step of Eq. (6) gives

$$\begin{pmatrix}\hat a\\\hat u\\\hat v\end{pmatrix}=\begin{pmatrix}a\\u\\v\end{pmatrix}+A_p^{-1}A^\top\begin{pmatrix}s_1\\\vdots\\s_N\end{pmatrix}.\tag{C4}$$

The pixel fields are therefore biased by the part of $s_n$ that the regressors can represent. It has two sources: the step-error intensity $e_n$, which leaks into the fit, and the errors of the frame parameters. Even with exact frame parameters, the pixel step absorbs the part of $\mathbf e=(e_1,\dots,e_N)^\top$ in the column space of $A$, and only $\Pi\mathbf e$ remains in the residual. Fitting the residual with $e_n$ itself would therefore underestimate $\Delta_n$. This is why Eq. (12) compares the projected residual with the projected model.

## Appendix D. Validity of the expansion

Only the estimates depend on $\Delta_n$; the true parameters are fixed. To give the orders of Eq. (7) a meaning, scale a fixed shape, $\Delta_n=\epsilon D_n$ with $\max_{n,x,y}|D_n|=1$, so that every estimate becomes a function of the single number $\epsilon$.

At $\epsilon=0$ the piston model describes the data exactly. A noiseless AIA fit then returns the true parameters moved by Eq. (10) to the conventions of Appendix B. Call these the conventional true parameters. They are fixed by the true parameters and do not depend on $\epsilon$.

The AIA estimates solve AIA's normal equations together with the normalization conditions. These determine them uniquely at $\epsilon=0$, so the estimates depend analytically on $\epsilon$ near it. The differences between the conventional true parameters and the AIA estimates therefore vanish at $\epsilon=0$ and expand in powers of $\epsilon$, which is Eq. (7) for the conventional true parameters. The move from them to the true parameters does not vanish with $\epsilon$. For $\Delta_n\ne0$ it changes $w_n$ and so the data, which is the floor of §"Quadrature frame of the expansion point".

## Appendix E. Normal equations

In this appendix $u,v,P_n,Q_n$ denote the AIA solution, with hats suppressed. At order $k$, Eq. (12) is fitted over all pixels by minimizing $\sum_{x,y}\big\|\Pi\big(\mathbf r^{(k)}-\mathbf z^{(k)}\big)\big\|^2$. Let $z_{c,n}$ be the coefficient of the unknown $c$ in $z_n^{(k)}$. Setting the gradient to zero gives $M\beta=\rho$ with

$$M_{cc'}=\sum_{x,y}\sum_{n,m}z_{c,n}\Pi_{nm}z_{c',m},\qquad\rho_c=\sum_{x,y}\sum_nz_{c,n}\big(\Pi\mathbf r^{(k)}\big)_n.\tag{E1}$$

**Reduction to frame-free sums.** Every column has the same form,

$$z_{c,n}=\big(\lambda_{c,n}u+\mu_{c,n}v\big)\sum_{j=0}^J\eta_{c,n,j}H_j,\qquad H_0=1,$$

with coefficients that depend on the frame but not on the pixel:

| unknown $c$ | $\lambda_{c,n}$ | $\mu_{c,n}$ | $\sum_j\eta_{c,n,j}H_j$ |
|---|---|---|---|
| $P_m^{(k)}$ | $\delta_{nm}$ | $0$ | $1$ |
| $Q_m^{(k)}$ | $0$ | $\delta_{nm}$ | $1$ |
| $\alpha_{mj}^{(k)}$ | $-Q_n\delta_{nm}$ | $P_n\delta_{nm}$ | $H_j$ |

The rows of $\alpha_{mj}^{(k)}$ come from $w_n=P_nv-Q_nu$. Substituting into Eq. (E1) gives

$$\begin{aligned}
M_{cc'}&=\sum_{n,m}\Pi_{nm}\sum_{j,j'}\eta_{c,n,j}\,\eta_{c',m,j'}\Big[\lambda_{c,n}\lambda_{c',m}F^{uu}_{jj'}+\big(\lambda_{c,n}\mu_{c',m}+\mu_{c,n}\lambda_{c',m}\big)F^{uv}_{jj'}+\mu_{c,n}\mu_{c',m}F^{vv}_{jj'}\Big],\\
\rho_c&=\sum_n\sum_j\eta_{c,n,j}\big(\lambda_{c,n}L^u_{nj}+\mu_{c,n}L^v_{nj}\big),
\end{aligned}\tag{E2}$$

with the pixel sums

$$F^{uu}_{jj'}=\sum_{x,y}u^2H_jH_{j'},\qquad F^{uv}_{jj'}=\sum_{x,y}uvH_jH_{j'},\qquad F^{vv}_{jj'}=\sum_{x,y}v^2H_jH_{j'},$$

$$L^u_{nj}=\sum_{x,y}uH_j\big(\Pi\mathbf r^{(k)}\big)_n,\qquad L^v_{nj}=\sum_{x,y}vH_j\big(\Pi\mathbf r^{(k)}\big)_n.$$

The matrix needs only the three symmetric $(J+1)\times(J+1)$ arrays $F$. They are accumulated once and do not change between orders, so $M$ is the same at every order. The right-hand side needs the $2N(J+1)$ sums $L$, recomputed at each order in one pass over the pixels. At $k=1$, $\Pi\mathbf r^{(1)}=\mathbf r^{(1)}$.

**Conditions.** The conditions of Eq. (15) are $C\beta=0$ with

$$C=\begin{pmatrix}
A^\top&0&0\\
0&A^\top&0\\
0&0&\big(I_J\ \cdots\ I_J\big)
\end{pmatrix},$$

where $I_J$, the $J\times J$ identity, is repeated $N$ times; $C$ has size $(6+J)\times(2N+NJ)$.

## References

- R. Frisch and F. V. Waugh, "Partial time regressions as compared with individual trends," *Econometrica* **1**(4), 387–401 (1933).
- M. C. Lovell, "Seasonal adjustment of economic time series and multiple regression analysis," *Journal of the American Statistical Association* **58**(304), 993–1010 (1963).
- G. H. Golub and V. Pereyra, "The differentiation of pseudo-inverses and nonlinear least squares problems whose variables separate," *SIAM Journal on Numerical Analysis* **10**(2), 413–432 (1973).
- D. C. Hoaglin and R. E. Welsch, "The hat matrix in regression and ANOVA," *The American Statistician* **32**(1), 17–22 (1978).
- Z. Wang and B. Han, "Advanced iterative algorithm for phase extraction of randomly phase-shifted interferograms," *Optics Letters* **29**(14), 1671–1673 (2004).
- Y. Chen and Q. Kemao, "General iterative algorithm for phase-extraction from fringe patterns with random phase-shifts, intensity harmonics and non-uniform phase-shift distribution," *Optics Express* **29**(19), 30905–30926 (2021).
