# Advanced Iterative Algorithm (AIA)

This document derives the advanced iterative algorithm (AIA) for phase-shifting interferometry (Wang and Han, 2004), together with the accuracy analysis of Chen and Kemao (2019), from the per-frame model of `interference_model.md`.

## Starting point

`interference_model.md`, Eq. (17), models each frame as

$$I_n(x,y)=\alpha_n\Big[a(x,y)+g_n\,b(x,y)\cos\big(\Phi(x,y)+\delta_n+\Delta_n(x,y)\big)\Big],\qquad n=1\dots N.$$

AIA solves its uniform-piston limit $\Delta_n\equiv0$ (`interference_model.md`, Eq. 20), in which the phase step of a frame is a single scalar $\delta_n$ rather than a field. When $\Delta_n\not\equiv0$, the recovered $\Phi$ and $b$ carry a low-order bias that does not shrink with more frames; `sf_aia.md` and `vp_aia.md` quantify and correct it.

The per-frame source-power factor $\alpha_n$ is not fitted. With many fringes across the field, $\langle a\rangle\gg\langle b\rangle$, so the frame mean gives $\langle I_n\rangle\approx\alpha_n\langle a\rangle$; $\alpha_n$ is estimated from that mean and divided out before the fit, under the convention $\operatorname{median}(\alpha_n)=1$. Whatever frame-to-frame level this leaves is absorbed by the per-frame offset $c_n$ below.

With $\alpha_n$ divided out and $\Delta_n\equiv0$, AIA solves

$$I_n(x,y)=a(x,y)+g_n\,b(x,y)\cos\big(\Phi(x,y)+\delta_n\big)\tag{1}$$

for the fields $a,b,\Phi$ and the per-frame steps $\delta_n$ and gains $g_n$, none of which are known in advance.

## Quadrature form

Expanding $\cos(\Phi+\delta_n)=\cos\Phi\cos\delta_n-\sin\Phi\sin\delta_n$ and defining the quadrature fields

$$u=b\cos\Phi,\qquad v=-b\sin\Phi,\tag{2}$$

Eq. (1) becomes

$$I_n=a+g_n\big(u\cos\delta_n+v\sin\delta_n\big)=a+P_nu+Q_nv,\qquad
P_n=g_n\cos\delta_n,\qquad Q_n=g_n\sin\delta_n.\tag{3}$$

Equation (3) is linear in $(a,u,v)$ for fixed $(P_n,Q_n)$ and linear in $(P_n,Q_n)$ for fixed $(a,u,v)$, but not in both at once, since their product appears. This bilinearity rules out a closed-form solution and motivates alternating least squares.

## Objective

AIA minimizes the squared residual against Eq. (3), extended by a free per-frame offset $c_n$ that absorbs the background level and the frame-to-frame drift left by the $\alpha_n$ normalization:

$$\mathcal L\big(a,u,v,\{c_n,\delta_n,g_n\}\big)
=\sum_{n=1}^N\sum_{x,y}\Big[I_n(x,y)-a(x,y)-c_n-g_n\big(u\cos\delta_n+v\sin\delta_n\big)\Big]^2.\tag{4}$$

The fit is parametrized by $(a,u,v)$, the variables each linear sub-problem below solves for; $b$ and $\Phi$ follow from $(u,v)$ at the end.

## Alternating least squares

AIA alternates two linear solves until $\delta_n$, and $g_n$ when it is fitted, stop changing, or an iteration limit is reached.

### Pixel step

Equation (4) sums over pixels with no cross terms between them, so minimizing it over the fields $(a,u,v)$ splits into an independent problem at each pixel, a sum over frames alone. With $\{c_n,\delta_n,g_n\}$ fixed, each pixel minimizes

$$\mathcal L_{x,y}(a,u,v)=\sum_{n=1}^N\big[I_n(x,y)-c_n-a-g_n\big(u\cos\delta_n+v\sin\delta_n\big)\big]^2,\tag{5}$$

a linear regression of the pixel's $N$ frame values, each with $c_n$ subtracted, on the design matrix

$$A=\begin{pmatrix}
1&g_1\cos\delta_1&g_1\sin\delta_1\\
\vdots&\vdots&\vdots\\
1&g_N\cos\delta_N&g_N\sin\delta_N
\end{pmatrix},\tag{6}$$

whose columns are $(1,P_n,Q_n)$. Setting the three derivatives to zero gives the normal equations with the pixel-step normal matrix $A_p=A^\top A$; the same matrix serves every pixel. The recovered $(u,v)$ are determined only up to the reparametrization of "Identifiability and gauge" below, and are whitened before the frame step.

### Frame step

With $(a,u,v)$ fixed, Eq. (4) sums over frames with no cross terms, so minimizing it over $\{c_n,\delta_n,g_n\}$ splits into an independent problem at each frame, a sum over pixels alone. AIA regresses the measured frame directly, with the scalar offset $c_n$ in place of the background field:

$$\mathcal L_n(c_n,P_n,Q_n)=\sum_{x,y}\big[I_n(x,y)-c_n-P_nu(x,y)-Q_nv(x,y)\big]^2,\tag{7}$$

a linear regression of the $N_p$ pixel values on the design matrix

$$B=\begin{pmatrix}
1&u(x_1,y_1)&v(x_1,y_1)\\
\vdots&\vdots&\vdots\\
1&u(x_{N_p},y_{N_p})&v(x_{N_p},y_{N_p})
\end{pmatrix},\tag{8}$$

the same for every frame, with the frame-step normal matrix $A_{ps}=B^\top B$. Its solution gives $(c_n,P_n,Q_n)$, and

$$\delta_n=\operatorname{atan2}(Q_n,P_n),\qquad g_n=\sqrt{P_n^2+Q_n^2},\tag{9}$$

where $\delta_n$ is unchanged by a common positive scale on $(P_n,Q_n)$.

**Dropping the background field.** Write $\langle f,h\rangle=\sum_{x,y}f h$ and decompose the background against the columns of $B$,

$$a=a_0+a_u\,u+a_v\,v+a_\perp,\qquad \langle a_\perp,1\rangle=\langle a_\perp,u\rangle=\langle a_\perp,v\rangle=0,$$

with $(a_0,a_u,a_v)$ the least-squares coefficients of $a$ on $\{1,u,v\}$. The decomposition is the same for every frame, since $a$, $u$, and $v$ do not depend on $n$. Because $a_\perp$ is orthogonal to every column of $B$, it does not enter the normal equations of Eq. (7), whose exact minimizer is therefore

$$\hat c_n=c_n+a_0,\qquad \hat P_n=P_n+a_u,\qquad \hat Q_n=Q_n+a_v,$$

whatever the size or shape of $a_\perp$. Three consequences follow. The structure of the background, $a_\perp$, is invisible to the frame step and appears only in that step's own residual, which is discarded. The constant $a_0$ is the same for every frame and is removed by the convention $\overline{c_n}=0$ below. The pair $(a_u,a_v)$ displaces every frame's $(\hat P_n,\hat Q_n)$ by the same vector, a step along the shift freedom of "Identifiability and gauge"; it measures the overlap of the background with the fringe pattern and is small whenever $a$ varies slowly compared with a fringe period, the regime that a large $\kappa_{ps}$ flags.

Up to that frame-independent displacement, the two steps minimize the same objective, Eq. (4), so its residual decreases monotonically. Whitening between them changes neither objective, since it re-bases the same subspace $\{1,u,v\}$ against which both regress.

### Phase-origin convention

Adding a constant to every $\delta_n$ while subtracting it from $\Phi$ leaves Eq. (3) unchanged. Each iteration re-references $\delta_n\leftarrow\delta_n-\delta_1$, so that $\delta_1=0$.

### Convergence

Iterate the two steps until the phase steps stop changing. Let $\delta_n^{(k)}$ be the value after iteration $k$, re-referenced to $\delta_1=0$. The criterion is

$$\max_n\Big|\arg e^{i\left(\delta_n^{(k)}-\delta_n^{(k-1)}\right)}\Big|<\varepsilon_\delta,$$

where the argument keeps the comparison in $(-\pi,\pi]$, so that a wrap of $\delta_n$ is not read as a large change. When the gain is fitted, $\max_n\big|g_n^{(k)}-g_n^{(k-1)}\big|<\varepsilon_g$ is required as well. The iteration also stops at a fixed iteration limit, in which case the result is a stalled fit rather than a converged one. The phase map and fringe amplitude then follow from the quadrature fields,

$$\Phi(x,y)=\operatorname{atan2}\big(-v(x,y),u(x,y)\big),\qquad b(x,y)=\sqrt{u(x,y)^2+v(x,y)^2}.\tag{10}$$

## Accuracy diagnostics

The accuracy of the two solves is governed by the conditioning of their normal matrices (Chen and Kemao, 2019):

- $\kappa_p=\operatorname{cond}(A_p)$ measures how well the phase-step distribution $\{\delta_n,g_n\}$ conditions the pixel step.
- $\kappa_{ps}=\operatorname{cond}(A_{ps})$, evaluated on the normalized design with columns $\cos\Phi$ and $\sin\Phi$, measures how well the recovered phase covers the unit circle. It is bounded below by $2$, with equality when $\Phi$ is spread evenly over $2\pi$. Large values mean the field spans less than about one fringe, so the frame step cannot separate $\delta_n$ from noise.

Chen and Kemao (2019) predict the RMS phase error as

$$\sigma_\Phi\approx0.42\big(\sqrt{\kappa_p}+2\big)\,\frac{\sigma}{b}\,\frac{1}{\sqrt N},\tag{11}$$

with $\sigma$ the RMS residual of the final pixel step and $b$ the median fringe amplitude. Equation (11) is an empirical fit rather than a consequence of the normal equations, and $\kappa_p$ alone cannot be an exact predictor: scaling the constant column of $A$, which amounts to solving for $3a$ instead of $a$, leaves the recovered $\Phi$ unchanged while moving $\kappa_p$ from $2$ to $18$. In the ideal case it agrees with the exact result of Eq. (30) to $1.4\%$, and its $\sqrt{\kappa_p}$ growth is the correct worst-case shape (Eq. 31).

A third diagnostic, $\min_ng_n/\operatorname{median}(g_n)$, flags a frame nearly uncorrelated with the recovered fringe pattern: its $\delta_n$ is poorly determined even when the fit reports convergence.

## How many frames are needed

Stack the per-frame model over all $N$ frames and $N_p$ pixels,

$$I=\mathbf 1_N\,a^\top+c\,\mathbf 1_{N_p}^\top+P\,u^\top+Q\,v^\top,\tag{12}$$

with $a,u,v\in\mathbb R^{N_p}$ and $c,P,Q\in\mathbb R^N$. The right side is a sum of four rank-1 terms, so $I$ has rank at most 4. Two of the four carry a fixed vector: $\mathbf 1_N$ multiplies $a$, and $\mathbf 1_{N_p}$ multiplies $c$.

Whether the model constrains the data at all depends on the frame side. The fields $(a,u,v)$ are free, so the three frame-side vectors $\{\mathbf 1_N,P,Q\}$ decide what is reachable. For $N\le3$ they span $\mathbb R^N$ for generic steps, every stack is reproduced exactly with $c=0$, and no per-frame parameter is constrained by the data.

For $N\ge4$ there are vectors $m\ne0$ orthogonal to $\mathbf 1_N$, $P$, and $Q$. Multiplying Eq. (12) by such an $m$ leaves

$$m^\top I=\big(m^\top c\big)\,\mathbf 1_{N_p}^\top,$$

so the model requires $m^\top I$ to be constant across pixels, $N_p-1$ conditions that a generic stack fails. The model therefore constrains the data, and the per-frame parameters are determined, from $N\ge4$ frames on. The free offset $c_n$ does not cost an extra frame, since its pixel-side factor is fixed to $\mathbf 1_{N_p}$ rather than free.

This is a necessary condition, not a sufficient one: with $N\ge4$ a poorly distributed $\{\delta_n\}$ or a field spanning too little phase still leaves the per-frame parameters weakly determined, which is what $\kappa_p$ and $\kappa_{ps}$ measure.

## Identifiability and gauge

Even when the data constrain the model, the minimizer of Eq. (4) is not unique: with $(a,u,v)$ and $(c_n,P_n,Q_n)$ both free, nine independent directions leave every predicted intensity unchanged, together with one discrete symmetry.

**Quadrature basis (four directions).** For any invertible $2\times2$ matrix $T$,

$$(u,v)\to(u,v)\,T,\qquad (P,Q)\to(P,Q)\,T^{-\top},\tag{13}$$

since $\{1,u,v\}$ and $\{1,uT,vT\}$ span the same subspace. Left alone, the alternation can converge to any basis of that subspace, with $(P_n,Q_n)$ tracing no physically meaningful $\delta_n$.

**Background along the fringe pattern (two directions).** $a\to a+\lambda u$ with $P_n\to P_n-\lambda$, and $a\to a+\mu v$ with $Q_n\to Q_n-\mu$. These leave $(u,v)$, and therefore $\Phi$, untouched; they displace every frame's $(P_n,Q_n)$ by the same vector. The frame step takes exactly such a step each iteration, by the displacement of "Dropping the background field".

**Shift of the quadratures (two directions).** For any scalar $s$,

$$u\to u+s,\qquad a\to a-s\,\overline{P},\qquad c_n\to c_n-s\big(P_n-\overline{P}\big),\tag{14}$$

and symmetrically for $v$ against $Q$. Unlike the previous two, this one does move $\Phi=\operatorname{atan2}(-v,u)$.

**Constant between background and offset (one direction).** $a\to a+\kappa$ with $c_n\to c_n-\kappa$.

**Discrete.** $(\Phi,\delta_n)\to(-\Phi,-\delta_n)$, since the cosine is even. The alternation does not pin it, so two solves of the same data may land on opposite branches.

The algorithm fixes five of the nine. The phase origin $\delta_1=0$ fixes the rotation in Eq. (13). Whitening, applied to $(u,v)$ after every pixel step,

$$\sum_{x,y}u^2=\sum_{x,y}v^2,\qquad \sum_{x,y}uv=0,\tag{15}$$

fixes its shear and anisotropic scaling, collapsing Eq. (13) to rotations and reflections; it preserves $\sum_{x,y}(u^2+v^2)$, so it does not change the overall scale. The remaining scale of Eq. (13) and the constant of the last direction are fixed by

$$g_n=\sqrt{P_n^2+Q_n^2},\qquad g_n\leftarrow g_n/\operatorname{median}(g_n),\qquad c_n\leftarrow c_n-\overline{c_n}.\tag{16}$$

The pixel step is then re-solved against $I-c$, so both steps minimize the same objective.

The four remaining directions — the two shifts of Eq. (14) and the two background directions — are pinned by no convention above. Only the shifts move $\Phi$; the alternation starts from $c_n\equiv0$ and they are flat directions of Eq. (4) rather than descent directions, so the iteration does not drift along them, but nothing resolves them explicitly either.

## Known gain as a special case

If $g_n$ is measured independently, it can be held fixed instead of fitted. The offset $c_n$ is then unnecessary, since the pixel step's own $a(x,y)$ absorbs the background exactly; the design matrix of Eq. (6) is built from the known $g_n$; and whitening is unnecessary, because a fixed $g_n$ confines $(P_n,Q_n)$ to a circle of known radius and removes the freedom of Eq. (13). The frame count is unchanged: $\{\mathbf 1_N,P,Q\}$ still spans $\mathbb R^N$ for $N\le3$, so the data constrain the steps only from $N\ge4$ frames on.

## Phase-error covariance

This section derives the phase and amplitude error from the normal equations themselves, in three stages: $\delta_n$ and $g_n$ known; $\delta_n$ fitted with $g_n$ fixed; both fitted. Throughout, $\mathbf i(x,y)\in\mathbb R^N$ is the per-pixel vector of measured intensities.

### Noise model

The camera adds zero-mean noise, independent from pixel to pixel and frame to frame, but not of equal variance:

$$I_n^{\text{meas}}=I_n+\varepsilon_n,\qquad
\operatorname E\big[\varepsilon_n(x,y)\big]=0,\qquad
\operatorname{Cov}\big(\varepsilon_n(x,y),\varepsilon_m(x',y')\big)=\sigma_n^2(x,y)\,\delta_{nm}\delta_{xx'}\delta_{yy'}.\tag{17}$$

Only these two moments are used below. Independence rules out correlation within a frame or across frames: banded or common-mode readout noise, inter-pixel crosstalk, and speckle or source fluctuation correlated across the field.

### Stage 1: known steps and gains

With $\delta_n$ and $g_n$ at their true values, the pixel step at one pixel is the linear solve

$$(a,u,v)^\top=A_p^{-1}A^\top\mathbf i,\qquad A_p=A^\top A,\tag{18}$$

so its error $(e_a,e_u,e_v)^\top=A_p^{-1}A^\top\varepsilon$ has the sandwich covariance

$$\operatorname{Cov}\big((e_a,e_u,e_v)^\top\big)=A_p^{-1}A^\top D\,A\,A_p^{-1},\qquad D=\operatorname{diag}\big(\sigma_n^2(x,y)\big),\tag{19}$$

which collapses to

$$\operatorname{Cov}\big((e_a,e_u,e_v)^\top\big)=\sigma^2A_p^{-1}\tag{20}$$

exactly when $D=\sigma^2I_N$, that is, equal variance across the $N$ frames at that pixel — a weaker condition than equal variance across pixels. Write $S$ for the $\{u,v\}$ block of $A_p^{-1}$, so that $\operatorname{Cov}\big((e_u,e_v)^\top\big)=\sigma^2S$, with $\sigma$ read as a per-pixel quantity.

Differentiating Eq. (10) to first order in $(e_u,e_v)$,

$$e_\Phi=-\frac{\sin\Phi\,e_u+\cos\Phi\,e_v}{b},\qquad
e_b=\cos\Phi\,e_u-\sin\Phi\,e_v,\tag{21}$$

so with the orthonormal pair $w=(\sin\Phi,\cos\Phi)^\top$ and $\tilde w=(\cos\Phi,-\sin\Phi)^\top$,

$$\sigma_\Phi^2(x,y)=\frac{\sigma^2}{b^2}\,w^\top Sw,\qquad
\sigma_b^2(x,y)=\sigma^2\,\tilde w^\top S\tilde w.\tag{22}$$

### Closed form

Write the $g$-weighted circular moments of the phase-step distribution,

$$R_c=\langle g_n\cos\delta_n\rangle_n,\quad R_s=\langle g_n\sin\delta_n\rangle_n,\quad
R_c^{(2)}=\langle g_n^2\cos2\delta_n\rangle_n,\quad R_s^{(2)}=\langle g_n^2\sin2\delta_n\rangle_n.\tag{23}$$

The columns of Eq. (6) then give

$$A_p=N\begin{pmatrix}
1&R_c&R_s\\
R_c&\tfrac12\big(\langle g^2\rangle_n+R_c^{(2)}\big)&\tfrac12R_s^{(2)}\\
R_s&\tfrac12R_s^{(2)}&\tfrac12\big(\langle g^2\rangle_n-R_c^{(2)}\big)
\end{pmatrix},\tag{24}$$

and eliminating the $a$ row by Schur complement gives $S$ in closed form,

$$S=\frac1N\,C^{-1},\qquad
C=\big\langle g_n^2\,d_nd_n^\top\big\rangle_n-\big\langle g_nd_n\big\rangle_n\big\langle g_nd_n\big\rangle_n^\top,\qquad
d_n=\begin{pmatrix}\cos\delta_n\\\sin\delta_n\end{pmatrix}.\tag{25}$$

$C$ is the frame-side covariance of the $g$-weighted unit-circle directions: solving for the background $a$ costs exactly a mean-centering of those directions. Its trace, $\operatorname{tr}C=\langle g^2\rangle_n-|R|^2$ with $|R|=\sqrt{R_c^2+R_s^2}$, shrinks as $|R|$ approaches $\langle g\rangle_n$, the failure mode $\kappa_{ps}$ flags seen from the frame side. Combining Eqs. (22) and (25),

$$\sigma_\Phi^2(x,y)=\frac{\sigma^2}{N\,b(x,y)^2}\,w^\top C^{-1}w,\qquad
\sigma_b^2(x,y)=\frac{\sigma^2}{N}\,\tilde w^\top C^{-1}\tilde w,\tag{26}$$

and, since $w$ and $\tilde w$ are orthonormal, the $\Phi$ dependence cancels in

$$b(x,y)^2\sigma_\Phi^2(x,y)+\sigma_b^2(x,y)=\frac{\sigma^2}{N}\operatorname{tr}\big(C^{-1}\big),\tag{27}$$

the same total error budget at every pixel, split between phase and amplitude by how $\Phi$ aligns with the principal axes of $C$.

### Bounds and limiting cases

Since $w$ is a unit vector, Eq. (26) is bounded by the eigenvalues $\lambda_\pm$ of $C$,

$$\frac{\sigma}{b\sqrt{N\lambda_+}}\;\le\;\sigma_\Phi\;\le\;\frac{\sigma}{b\sqrt{N\lambda_-}},\tag{28}$$

with equality when $\Phi$ aligns with an eigenvector. Whenever $C$ is anisotropic, $\sigma_\Phi$ therefore varies between pixels of equal $b$ with period $\pi$ in $\Phi$, a modulation no single scalar can express. Averaging Eq. (26) over the field, with $\Phi$ decorrelated from the local illumination — many fringes spread across whatever gradient exists — replaces $w^\top C^{-1}w$ by $\tfrac12\operatorname{tr}(C^{-1})=\operatorname{tr}C/(2\det C)$ and keeps the per-pixel noise and contrast together:

$$\big\langle\sigma_\Phi^2\big\rangle_{x,y}=\frac{1}{2N}\Big\langle\frac{\sigma^2(x,y)}{b(x,y)^2}\Big\rangle_{x,y}\,\frac{\operatorname{tr}C}{\det C}.\tag{29}$$

The two factors must stay under one spatial average: replacing it by $\langle\sigma^2\rangle\langle1/b^2\rangle$ overstates the error whenever $\sigma$ and $b$ are positively correlated, which shot noise makes the rule rather than the exception.

**Ideal case.** For evenly spaced steps $\delta_n=2\pi n/N$ with $N\ge3$ and unit gain, $R=R^{(2)}=0$, so $C=\tfrac12I_2$ and

$$\sigma_\Phi=\sqrt{\frac2N}\,\frac{\sigma}{b},\tag{30}$$

independent of $\Phi$. This configuration has $\kappa_p=2$, and Eq. (11) evaluates to $1.014\,\sigma_\Phi$ for every $N$.

**Bound in terms of $\kappa_p$.** By Cauchy interlacing, $\lambda_{\max}(S)\le1/\lambda_{\min}(A_p)=\kappa_p/\lambda_{\max}(A_p)$, and $\lambda_{\max}(A_p)\ge\operatorname{tr}(A_p)/3=N\big(1+\langle g^2\rangle_n\big)/3$ by Eq. (24), so

$$\sigma_\Phi\;\le\;\frac{\sigma}{b}\sqrt{\frac{3\kappa_p}{N\big(1+\langle g^2\rangle_n\big)}},\tag{31}$$

which at $g_n\equiv1$ is $1.22\sqrt{\kappa_p}\,\sigma/(b\sqrt N)$.

### Shot noise

At a fixed pixel, $\sigma_n^2(x,y)$ genuinely differs between frames, since shot noise follows that frame's own fringe-modulated illumination. Equation (20) requires $D\propto I_N$, which such noise satisfies only approximately; the stand-in is the quadrature mean across frames,

$$\sigma_0(x,y)^2=\frac1N\sum_{n=1}^N\sigma_n^2(x,y),\tag{32}$$

used in place of $\sigma$ in Eq. (26), giving a per-pixel noise map paired pointwise with $b(x,y)$. How good the approximation is depends on the spread of $\sigma_n^2$ about $\sigma_0^2$, which calibrated per-frame noise maps give directly.

### Stage 2: fitted steps, known gains

Stage 1 held $\delta_n$ at its true value. In practice the pixel step uses the frame step's estimate $\hat\delta_n=\delta_n+e_{\delta_n}$, so its design matrix is already perturbed. Differentiating the normal equations at the true, noiseless data, where the model's own zero residual cancels the leading term, gives

$$\frac{\partial X}{\partial\delta_n}=-A_p^{-1}a_n\,\frac{\partial I_n}{\partial\delta_n},\qquad
\frac{\partial I_n}{\partial\delta_n}=-g_n\,b(x,y)\sin\big(\Phi(x,y)+\delta_n\big),\tag{33}$$

with $X=(a,u,v)^\top$ and $a_n$ the $n$-th row of $A$: an error in $\delta_n$ displaces the pixel-step solution exactly as extra noise on frame $n$ would, scaled by the model's sensitivity to $\delta_n$. The $(u,v)$ part of that displacement is the leverage vector

$$k_n=\big(A_p^{-1}a_n\big)_{\{u,v\}}=\frac1N\,C^{-1}\big(g_nd_n-R\big),\qquad R=\big\langle g_nd_n\big\rangle_n.\tag{34}$$

**Size of the frame-step error.** It comes from the frame step, which is the pixel step transposed: a regression over $N_p$ pixels with design $B$ in place of $A$. Holding $(u,v)$ at their noise-free values, the algebra of Eqs. (18)–(19) gives, per frame,

$$\operatorname{Cov}\big((e_{c_n},e_{P_n},e_{Q_n})^\top\big)=A_{ps}^{-1}\big(B^\top DB\big)A_{ps}^{-1},\qquad
A_{ps}=B^\top B,\qquad D=\operatorname{diag}\big(\sigma_0^2(x,y)\big).\tag{35}$$

Differentiating $\delta_n=\operatorname{atan2}(Q_n,P_n)$ as in Eq. (21), with $w_n=(-\sin\delta_n,\cos\delta_n)^\top$,

$$\sigma_{\delta_n}^2=\frac{1}{g_n^2}\,w_n^\top\Big[A_{ps}^{-1}\big(B^\top DB\big)A_{ps}^{-1}\Big]_{\{P,Q\}}w_n.\tag{36}$$

With many fringes and no strongly preferred orientation, $\sum_{x,y}u^2\approx\sum_{x,y}v^2\approx N_p\langle b^2\rangle/2$, $\sum_{x,y}uv\approx0$, and $\Phi$ decorrelated from the local illumination, Eq. (36) becomes isotropic:

$$\sigma_{\delta_n}^2\approx\frac{2\sigma_{\text{eff}}^2}{N_p\langle b^2\rangle\,g_n^2},\qquad
\sigma_{\text{eff}}^2=\frac{\big\langle\sigma_0^2b^2\big\rangle}{\big\langle b^2\big\rangle}.\tag{37}$$

The $b^2$ weighting says that bright, high-contrast pixels dominate the frame-step fit, so their noise level sets the uncertainty of $\delta_n$.

**Effect on the phase.** The estimate $\hat\delta_n$ is fitted from the same frames the pixel step then uses, so $e_{\delta_n}$ is correlated with the noise at every pixel. Treating the two as independent — adding $\sum_n(\partial I_n/\partial\delta_n)^2\operatorname{Var}(e_{\delta_n})\,k_nk_n^\top$ to $\sigma^2S$ — gives the right size but the wrong sign, because the correlation contributes twice that term with the opposite sign. For evenly spaced steps, unit gain, and $\Phi$ spread evenly over $2\pi$, the net effect is a reduction:

$$\sigma_\Phi^2\big|_{\text{ideal}}=\sigma_\Phi^2\big|_{\text{Eq. (30)}}\Big(1-\frac{3}{2N_p}\Big).\tag{38}$$

Away from that configuration the per-pixel covariance is not settled here: with irregular phase coverage the ratio to Stage 1 falls on either side of $1$, and neither the independent-error expression nor its sign-corrected counterpart reproduces it. What holds in every case is the order of the effect: fitting $\delta_n$ changes the per-pixel phase variance by $O(1/N_p)$, a few parts in $10^3$ for a megapixel field, and the correction vanishes as $N_p$ grows. Frame-step errors are nonetheless spatially coherent — one number per frame, not one per pixel — so they do not average away across the field the way per-pixel noise does, and they matter when a reconstruction is differenced against another acquisition with correlated $\delta_n$ error. The effective $N_p$ is also smaller than the raw pixel count whenever a small region is used or the noise is spatially correlated.

### Stage 3: fitted steps and gains

Perturbing the assumed $g_n$ instead of $\delta_n$ gives the same leverage vector, only the sensitivity differs:

$$\frac{\partial X}{\partial g_n}=-A_p^{-1}a_n\,\frac{\partial I_n}{\partial g_n},\qquad
\frac{\partial I_n}{\partial g_n}=b(x,y)\cos\big(\Phi(x,y)+\delta_n\big).\tag{39}$$

Under the same well-spread approximation, $e_{\delta_n}$ and $e_{g_n}$ are uncorrelated to leading order: $\delta_n$ is an angle and $g_n$ a radius, so their errors are orthogonal projections of one isotropic error in $(P_n,Q_n)$. The gain error follows from Eq. (35) in the same way as Eq. (37),

$$\sigma_{g_n}^2\approx\frac{2\sigma_{\text{eff}}^2}{N_p\langle b^2\rangle},\tag{40}$$

without the $1/g_n^2$, since a radius does not gain precision as it grows the way an angle does at fixed arc-length error. The correlation of Stage 2 applies here as well, and in the ideal configuration the two contributions combine into

$$\sigma_\Phi^2\big|_{\text{ideal}}=\sigma_\Phi^2\big|_{\text{Eq. (30)}}\Big(1-\frac{2}{N_p}\Big).\tag{41}$$

### Validity

Equation (21) linearizes $\operatorname{atan2}$ and the square root about the noise-free solution, which holds while $b/\sigma\gg1$, roughly $\sigma_\Phi\lesssim0.3$ rad. Below that the exact phasor distribution takes over, and Eq. (26) understates the probability of a $2\pi$ phase slip. The independence in Eq. (17) is not broken by shot noise, which is independent per pixel and frame, but by effects the model excludes: banded or common-mode readout, crosstalk, and speckle or source fluctuation correlated across the field. Dividing out $\alpha_n$ rescales frame $n$'s variance by $1/\alpha_n^2$, which is diagonal but not proportional to $I_N$, so Eq. (20) does not apply exactly; since $\alpha_n$ is estimated rather than assumed, this is corrected by replacing $A_p$ with $A^\top WA$, $W=\operatorname{diag}(\alpha_n^2)$, throughout.

### Interpreting $\sigma$ and $b$

Two gaps separate the quantities above from their usual estimates. The pixel-step residual has $N-3$ degrees of freedom per pixel, so dividing the summed squared residual by $N\,N_p$ instead of $(N-3)N_p$ biases $\sigma$ low by $\sqrt{(N-3)/N}$: $24\%$ at $N=7$ and $37\%$ at $N=5$. And scaling by $\operatorname{median}(b)$ is not the field average of Eq. (29), which keeps $\sigma^2(x,y)/b(x,y)^2$ together: low-contrast, low-illumination pixels dominate the true RMS more than two separately aggregated scalars suggest.

## References

- Z. Wang and B. Han, "Advanced iterative algorithm for phase extraction of randomly phase-shifted interferograms," *Optics and Lasers in Engineering* **42**(1), 87–97 (2004).
- Y. Chen and Q. Kemao, "Advanced iterative algorithm for phase extraction: performance evaluation and enhancement," *Optics Express* **27**(26), 37634–37651 (2019).
