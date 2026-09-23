# Advanced Iterative Algorithm (AIA)

This document derives the advanced iterative algorithm (AIA) for phase-shifting interferometry (Wang and Han, 2004) from the per-frame model of `interference_model.md`, together with the accuracy diagnostics of Chen and Kemao (2019). The noise analysis of the solution is in `aia_noise.md`.

## Starting point

`interference_model.md`, Eq. (17), models each frame as

$$I_n(x,y)=\alpha_n\Big[a(x,y)+g_n\,b(x,y)\cos\big(\Phi(x,y)+\delta_n+\Delta_n(x,y)\big)\Big],\qquad n=1\dots N.$$

AIA solves the uniform-piston limit $\Delta_n\equiv0$ (`interference_model.md`, Eq. 20), in which each frame's phase step is a scalar $\delta_n$. When $\Delta_n\not\equiv0$, the recovered $\Phi$ and $b$ carry a low-order bias that does not shrink with more frames; `sf_aia.md` and `vp_aia.md` correct it.

The source-power factor $\alpha_n$ is not fitted. With many fringes, $\langle a\rangle\gg\langle b\rangle$, so $\langle I_n\rangle\approx\alpha_n\langle a\rangle$; $\alpha_n$ is estimated from the frame mean and divided out, with $\operatorname{median}(\alpha_n)=1$. The per-frame offset $c_n$ below absorbs what remains. AIA then solves

$$I_n(x,y)=a(x,y)+g_n\,b(x,y)\cos\big(\Phi(x,y)+\delta_n\big)\tag{1}$$

for the fields $a,b,\Phi$ and the per-frame steps $\delta_n$ and gains $g_n$.

## Quadrature form

With the quadrature fields

$$u=b\cos\Phi,\qquad v=-b\sin\Phi,\tag{2}$$

Eq. (1) becomes

$$I_n=a+g_n\big(u\cos\delta_n+v\sin\delta_n\big)=a+P_nu+Q_nv,\qquad
P_n=g_n\cos\delta_n,\qquad Q_n=g_n\sin\delta_n.\tag{3}$$

Eq. (3) is linear in $(a,u,v)$ for fixed $(P_n,Q_n)$ and linear in $(P_n,Q_n)$ for fixed $(a,u,v)$, but not in both at once. This bilinearity motivates alternating least squares.

## Objective

AIA minimizes the squared residual of Eq. (3), extended by a free per-frame offset $c_n$ that absorbs the background level and the drift left by the $\alpha_n$ normalization:

$$\mathcal L\big(a,u,v,\{c_n,\delta_n,g_n\}\big)
=\sum_{n=1}^N\sum_{x,y}\Big[I_n(x,y)-a(x,y)-c_n-g_n\big(u\cos\delta_n+v\sin\delta_n\big)\Big]^2.\tag{4}$$

The fit solves for $(a,u,v)$; $b$ and $\Phi$ follow at the end.

## Alternating least squares

AIA alternates two linear solves until $\delta_n$, and $g_n$ when fitted, stop changing, or an iteration limit is reached.

### Pixel step

Eq. (4) has no cross terms between pixels, so for fixed $\{c_n,\delta_n,g_n\}$ each pixel minimizes

$$\mathcal L_{x,y}(a,u,v)=\sum_{n=1}^N\big[I_n(x,y)-c_n-a-g_n\big(u\cos\delta_n+v\sin\delta_n\big)\big]^2,\tag{5}$$

a linear regression of its $N$ offset-corrected frame values on the design matrix

$$A=\begin{pmatrix}
1&g_1\cos\delta_1&g_1\sin\delta_1\\
\vdots&\vdots&\vdots\\
1&g_N\cos\delta_N&g_N\sin\delta_N
\end{pmatrix},\tag{6}$$

with columns $(1,P_n,Q_n)$ and normal matrix $A_p=A^\top A$, the same at every pixel. The recovered $(u,v)$ are fixed only up to the freedoms of "Identifiability and gauge" and are whitened before the frame step.

### Frame step

Eq. (4) has no cross terms between frames, so for fixed $(a,u,v)$ each frame is fitted separately. AIA regresses the frame with the scalar offset $c_n$ in place of the background field:

$$\mathcal L_n(c_n,P_n,Q_n)=\sum_{x,y}\big[I_n(x,y)-c_n-P_nu(x,y)-Q_nv(x,y)\big]^2,\tag{7}$$

a linear regression of the $N_p$ pixel values on the design matrix

$$B=\begin{pmatrix}
1&u(x_1,y_1)&v(x_1,y_1)\\
\vdots&\vdots&\vdots\\
1&u(x_{N_p},y_{N_p})&v(x_{N_p},y_{N_p})
\end{pmatrix},\tag{8}$$

the same for every frame, with normal matrix $A_{ps}=B^\top B$. Its solution gives $(c_n,P_n,Q_n)$, and

$$\delta_n=\operatorname{atan2}(Q_n,P_n),\qquad g_n=\sqrt{P_n^2+Q_n^2}.\tag{9}$$

**Dropping the background field.** Decompose the background against the columns of $B$,

$$a=a_0+a_u\,u+a_v\,v+a_\perp,\qquad \textstyle\sum_{x,y}a_\perp=\sum_{x,y}a_\perp u=\sum_{x,y}a_\perp v=0,$$

with $(a_0,a_u,a_v)$ the least-squares coefficients of $a$ on $\{1,u,v\}$, the same for every frame. Since $a_\perp$ is orthogonal to the columns of $B$, the exact minimizer of Eq. (7) is

$$\hat c_n=c_n+a_0,\qquad \hat P_n=P_n+a_u,\qquad \hat Q_n=Q_n+a_v,$$

whatever $a_\perp$ is. The structure $a_\perp$ is invisible to the frame step. The constant $a_0$ is removed by the convention $\overline{c_n}=0$ below. The pair $(a_u,a_v)$ displaces every frame's $(\hat P_n,\hat Q_n)$ by the same vector, a step along the background freedom of "Identifiability and gauge"; it is small when $a$ varies slowly compared with a fringe period.

Up to that common displacement, both steps minimize Eq. (4), so its residual decreases monotonically. Whitening between them changes neither objective, since it re-bases the same subspace $\{1,u,v\}$.

### Phase-origin convention

Adding a constant to every $\delta_n$ and subtracting it from $\Phi$ leaves Eq. (3) unchanged. Each iteration sets $\delta_n\leftarrow\delta_n-\delta_1$, so that $\delta_1=0$.

### Convergence

With $\delta_n^{(k)}$ the steps after iteration $k$, re-referenced to $\delta_1=0$, the iteration stops when

$$\max_n\Big|\arg e^{i\left(\delta_n^{(k)}-\delta_n^{(k-1)}\right)}\Big|<\varepsilon_\delta,$$

where the argument keeps the difference in $(-\pi,\pi]$, so a wrap of $\delta_n$ is not read as a large change. When the gain is fitted, $\max_n\big|g_n^{(k)}-g_n^{(k-1)}\big|<\varepsilon_g$ is required as well. Stopping at the iteration limit means a stalled fit. The phase and amplitude follow from the quadrature fields,

$$\Phi(x,y)=\operatorname{atan2}\big(-v(x,y),u(x,y)\big),\qquad b(x,y)=\sqrt{u(x,y)^2+v(x,y)^2}.\tag{10}$$

## Accuracy diagnostics

The accuracy of the two solves is governed by the conditioning of their normal matrices (Chen and Kemao, 2019):

- $\kappa_p=\operatorname{cond}(A_p)$ measures how well the phase steps $\{\delta_n,g_n\}$ condition the pixel step.
- $\kappa_{ps}=\operatorname{cond}(A_{ps})$, evaluated on the normalized design with columns $\cos\Phi$ and $\sin\Phi$, measures how well the recovered phase covers the unit circle. It is at least $2$, with equality when $\Phi$ is spread evenly over $2\pi$. Large values mean the field spans less than about one fringe, so the frame step cannot separate $\delta_n$ from noise.

Chen and Kemao (2019) predict the RMS phase error as

$$\sigma_\Phi\approx0.42\big(\sqrt{\kappa_p}+2\big)\,\frac{\sigma}{b}\,\frac{1}{\sqrt N},\tag{11}$$

with $\sigma$ the RMS residual of the final pixel step and $b$ the median fringe amplitude. Eq. (11) is an empirical fit, and $\kappa_p$ alone cannot be an exact predictor: solving for $3a$ instead of $a$ leaves $\Phi$ unchanged but moves $\kappa_p$ from $2$ to $18$. In the ideal case it agrees with the exact result, `aia_noise.md` Eq. (14), to $1.4\%$, and its $\sqrt{\kappa_p}$ growth is the correct worst-case shape, `aia_noise.md` Eq. (15).

A third diagnostic, $\min_ng_n/\operatorname{median}(g_n)$, flags a frame nearly uncorrelated with the recovered fringe pattern: its $\delta_n$ is poorly determined even when the fit reports convergence.

## How many frames are needed

Stacking the model over all $N$ frames and $N_p$ pixels,

$$I=\mathbf 1_N\,a^\top+c\,\mathbf 1_{N_p}^\top+P\,u^\top+Q\,v^\top,\tag{12}$$

with $a,u,v\in\mathbb R^{N_p}$ and $c,P,Q\in\mathbb R^N$, a sum of four rank-1 terms. Since the fields are free, the frame-side vectors $\{\mathbf 1_N,P,Q\}$ decide what is reachable. For $N\le3$ they span $\mathbb R^N$ for generic steps, every stack is reproduced exactly with $c=0$, and the data constrain no per-frame parameter.

For $N\ge4$, some $m\ne0$ is orthogonal to $\mathbf 1_N$, $P$, and $Q$, and Eq. (12) gives

$$m^\top I=\big(m^\top c\big)\,\mathbf 1_{N_p}^\top,$$

so $m^\top I$ must be constant across pixels, $N_p-1$ conditions that a generic stack fails. The per-frame parameters are therefore determined from $N\ge4$ frames on. The offset $c_n$ costs no extra frame, since its pixel-side factor is fixed to $\mathbf 1_{N_p}$.

The condition is necessary, not sufficient: poorly distributed $\{\delta_n\}$ or a field spanning too little phase leave the per-frame parameters weakly determined, which $\kappa_p$ and $\kappa_{ps}$ measure.

## Identifiability and gauge

The minimizer of Eq. (4) is not unique: with $(a,u,v)$ and $(c_n,P_n,Q_n)$ free, nine independent directions leave every intensity unchanged, together with one discrete symmetry.

**Quadrature basis (four directions).** For any invertible $2\times2$ matrix $T$,

$$(u,v)\to(u,v)\,T,\qquad (P,Q)\to(P,Q)\,T^{-\top},\tag{13}$$

since $\{1,u,v\}$ and $\{1,uT,vT\}$ span the same subspace. Left alone, the alternation can converge to any basis of it, with $(P_n,Q_n)$ tracing no physical $\delta_n$.

**Background along the fringe pattern (two directions).** $a\to a+\lambda u$ with $P_n\to P_n-\lambda$, and $a\to a+\mu v$ with $Q_n\to Q_n-\mu$. These leave $(u,v)$ and $\Phi$ unchanged and displace every $(P_n,Q_n)$ by the same vector, as the frame step does each iteration.

**Shift of the quadratures (two directions).** For any scalar $s$,

$$u\to u+s,\qquad a\to a-s\,\overline{P},\qquad c_n\to c_n-s\big(P_n-\overline{P}\big),\tag{14}$$

and likewise for $v$ against $Q$. Unlike the previous two, this moves $\Phi=\operatorname{atan2}(-v,u)$.

**Constant between background and offset (one direction).** $a\to a+\kappa$ with $c_n\to c_n-\kappa$.

**Discrete.** $(\Phi,\delta_n)\to(-\Phi,-\delta_n)$, since the cosine is even. The alternation does not pin it, so two solves of the same data may land on opposite branches.

The algorithm fixes five of the nine. The phase origin $\delta_1=0$ fixes the rotation in Eq. (13). Whitening $(u,v)$ after every pixel step,

$$\sum_{x,y}u^2=\sum_{x,y}v^2,\qquad \sum_{x,y}uv=0,\tag{15}$$

fixes its shear and anisotropic scaling, leaving rotations and reflections; it preserves $\sum_{x,y}(u^2+v^2)$ and hence the overall scale. That scale and the background-offset constant are fixed by

$$g_n=\sqrt{P_n^2+Q_n^2},\qquad g_n\leftarrow g_n/\operatorname{median}(g_n),\qquad c_n\leftarrow c_n-\overline{c_n},\tag{16}$$

after which the pixel step is re-solved against $I-c$, so both steps minimize the same objective.

The remaining four directions, the two shifts of Eq. (14) and the two background directions, are pinned by no convention. Only the shifts move $\Phi$. The alternation starts from $c_n\equiv0$, and since these are flat directions of Eq. (4), not descent directions, it does not drift along them; but nothing resolves them explicitly either.

## Known gain as a special case

If $g_n$ is measured independently, it can be held fixed. The offset $c_n$ is then unnecessary, since $a(x,y)$ absorbs the background exactly; Eq. (6) is built from the known $g_n$; and whitening is unnecessary, because a fixed $g_n$ confines $(P_n,Q_n)$ to a circle of known radius and removes the freedom of Eq. (13). The frame count is unchanged: the data constrain the steps from $N\ge4$ frames on.

## References

- Z. Wang and B. Han, "Advanced iterative algorithm for phase extraction of randomly phase-shifted interferograms," *Optics and Lasers in Engineering* **42**(1), 87–97 (2004).
- Y. Chen and Q. Kemao, "Advanced iterative algorithm for phase extraction: performance evaluation and enhancement," *Optics Express* **27**(26), 37634–37651 (2019).
