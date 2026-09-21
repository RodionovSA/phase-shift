# Variable-Projection AIA (VP-AIA)

This document derives the Variable-Projection AIA (VP-AIA), which estimates a spatially varying phase-step error as a first-order correction to the advanced iterative algorithm (AIA) solution of the piston model (Wang and Han, 2004). The pixel fields are eliminated by projection, as in the variable projection method for separable least-squares problems (Golub and Pereyra, 1973).

## Interference model

The measured intensities are modeled as

$$I_n(x, y) = a(x, y) + g_n\,b(x, y)\,\cos\big(\Phi(x, y) + \delta_n + \Delta_n(x, y)\big), \qquad n = 1 \dots N, \tag{1}$$

where

- $a(x, y)$ is the static background (DC) intensity,
- $b(x, y)$ is the static fringe amplitude,
- $g_n$ is the spatially uniform per-frame contrast factor; it multiplies only the modulation term,
- $\Phi(x, y)$ is the total static phase,
- $\delta_n$ is the piston phase step,
- $\Delta_n(x, y)$ is the spatially varying phase-step error in frame $n$.

### Gauge conventions

The intensity depends on $\Phi$, $\delta_n$, and $\Delta_n$ only through their sum, and on $g_n$ and $b$ only through their product. Different parameter sets therefore predict the same intensities. The following conventions select one of them; each operation below leaves every intensity unchanged.

**Spatial mean.** A spatially uniform part of $\Delta_n$ is indistinguishable from the piston $\delta_n$. For each frame, let $m_n=\langle\Delta_n\rangle_{x,y}$ and assign that part to the piston:

$$\delta_n\leftarrow\delta_n+m_n,\qquad
\Delta_n(x,y)\leftarrow\Delta_n(x,y)-m_n.
\qquad\text{Thus }\langle\Delta_n\rangle_{x,y}=0.$$

**Frame mean.** A spatial pattern shared by all frames is indistinguishable from the static phase $\Phi$. After removing the spatial means, let $h(x,y)=\langle\Delta_n(x,y)\rangle_n$ and assign that pattern to the static phase:

$$\Phi(x,y)\leftarrow\Phi(x,y)+h(x,y),\qquad
\Delta_n(x,y)\leftarrow\Delta_n(x,y)-h(x,y).
\qquad\text{Thus }\langle\Delta_n(x,y)\rangle_n=0.$$

Since $h$ has zero spatial mean, this operation preserves the spatial-mean condition.

**Phase origin.** A common additive constant can still be exchanged between $\Phi$ and all piston steps. Set the first step to zero by taking $d=\delta_1$ and applying

$$\delta_n\leftarrow\delta_n-d,\qquad\Phi(x,y)\leftarrow\Phi(x,y)+d.
\qquad\text{Thus }\delta_1=0.$$

This preserves both zero-mean conditions on $\Delta_n$.

**Contrast scale.** A common positive factor can be exchanged between the frame contrasts and the fringe amplitude. Take $s=\operatorname{median}_ng_n$ and apply

$$g_n\leftarrow g_n/s,\qquad b(x,y)\leftarrow s\,b(x,y).
\qquad\text{Thus }\operatorname{median}_ng_n=1.$$

This leaves all phase conventions unchanged. The median is used rather than the mean, following `aia.md` Eq. (16), so that a single frame whose contrast collapses — the failure the diagnostic $\min_ng_n/\operatorname{median}_ng_n$ flags — does not move the scale of every other frame.

These conventions define which parts are called static phase, piston, phase-step error, contrast, and amplitude. The AIA parametrization introduces four further freedoms, fixed by the quadrature-frame conventions of the zeroth-order solution.

### Small phase-step errors

For $N$ frames and $K$ pixels, Eq. (1) contains $3K+2N+NK$ unknown scalar values: the fields $a,b,\Phi$ at every pixel, $g_n$ and $\delta_n$ for every frame, and $\Delta_n$ at every pixel of every frame. The $NK$ measured intensities provide at most $NK$ independent equations. The zero spatial mean in each frame and the zero frame mean at each pixel impose $N+K-1$ independent conditions, since their total-sum condition is shared; the phase origin and the contrast scale add two more. The number of undetermined degrees of freedom is therefore at least

$$(3K+2N+NK)-NK-(N+K-1)-2=2K+N-1.\tag{2}$$

An unrestricted $\Delta_n$ therefore cannot be recovered from the intensities alone, and its spatial form must be restricted. In addition, assume $|\Delta_n(x,y)|\ll1$ radian and organize its effect by perturbation order; this assumption alone does not resolve the ambiguity.

At each pixel, write $\theta_n=\Phi+\delta_n$ and suppress $(x,y)$ for brevity. Holding the other parameters fixed, expand directly about $\Delta_n=0$:

$$\cos(\theta_n+\Delta_n)=\sum_{k=0}^{\infty}\frac{\Delta_n^k}{k!}\cos\left(\theta_n+\frac{k\pi}{2}\right).\tag{3}$$

With these parameters fixed, the intensity expansion in the phase error alone is

$$I_n=a+g_n b\cos\theta_n+\sum_{k=1}^{\infty}w_n^{(k)}\Delta_n^k,\qquad w_n^{(k)}=\frac{g_n b}{k!}\cos\left(\theta_n+\frac{k\pi}{2}\right).\tag{4}$$

Here $w_n^{(k)}$ describes sensitivity to the phase error with the other parameters fixed.

## Zeroth-order solution

Set $\Delta_n=0$ and fit the measured stack with the AIA model. Superscripts $(0)$ on the fitted parameters are suppressed in this section:

$$I_n^{(0)}(x,y)=a(x,y)+P_nu(x,y)+Q_nv(x,y),\tag{5}$$

where $u=b\cos\Phi$, $v=-b\sin\Phi$, $P_n=g_n\cos\delta_n$, and $Q_n=g_n\sin\delta_n$. AIA alternates a fit across frames at each pixel with a fit across pixels in each frame.

### Pixel step

With $(P_n,Q_n)$ fixed, minimize independently at each pixel

$$\mathcal L_p(a,u,v)=\sum_{n=1}^N\big[I_n(x,y)-a-P_nu-Q_nv\big]^2.\tag{6}$$

The design matrix and its normal matrix are

$$A=\begin{pmatrix}
1&P_1&Q_1\\
\vdots&\vdots&\vdots\\
1&P_N&Q_N
\end{pmatrix},\qquad
A_p=A^\top A=\begin{pmatrix}
N&\sum_nP_n&\sum_nQ_n\\
\sum_nP_n&\sum_nP_n^2&\sum_nP_nQ_n\\
\sum_nQ_n&\sum_nP_nQ_n&\sum_nQ_n^2
\end{pmatrix}.\tag{7}$$

Setting the three derivatives of $\mathcal L_p$ to zero gives the normal equations. If $A_p$ is invertible, their solution is

$$\begin{pmatrix}a(x,y)\\u(x,y)\\v(x,y)\end{pmatrix}
=A_p^{-1}A^\top\begin{pmatrix}I_1(x,y)\\\vdots\\I_N(x,y)\end{pmatrix}
=A_p^{-1}\begin{pmatrix}\sum_nI_n(x,y)\\\sum_nP_nI_n(x,y)\\\sum_nQ_nI_n(x,y)\end{pmatrix}.\tag{8}$$

The sums run over frames at one fixed pixel; $A^\top$ denotes the transpose of $A$.

### Frame step

With $(u,v)$ fixed, fit each frame across its $K$ pixels, indexed by $j$, using a scalar intercept $c_n$:

$$\mathcal L_{ps}(c_n,P_n,Q_n)=\sum_{j=1}^K\big[I_{nj}-c_n-P_nu_j-Q_nv_j\big]^2.\tag{9}$$

The intercept accommodates the background in this regression. Replacing the spatial background $a_j$ by a scalar assigns background components correlated with the quadratures to $(P_n,Q_n)$.

The spatial design and normal matrices are

$$B=\begin{pmatrix}
1&u_1&v_1\\
\vdots&\vdots&\vdots\\
1&u_K&v_K
\end{pmatrix},\qquad
A_{ps}=B^\top B=\begin{pmatrix}
K&\sum_ju_j&\sum_jv_j\\
\sum_ju_j&\sum_ju_j^2&\sum_ju_jv_j\\
\sum_jv_j&\sum_ju_jv_j&\sum_jv_j^2
\end{pmatrix}.\tag{10}$$

Setting the derivatives of $\mathcal L_{ps}$ to zero gives the frame normal equations. If $A_{ps}$ is invertible, their solution is

$$\begin{pmatrix}c_n\\P_n\\Q_n\end{pmatrix}
=A_{ps}^{-1}B^\top\begin{pmatrix}I_{n1}\\\vdots\\I_{nK}\end{pmatrix}
=A_{ps}^{-1}\begin{pmatrix}\sum_jI_{nj}\\\sum_ju_jI_{nj}\\\sum_jv_jI_{nj}\end{pmatrix}.\tag{11}$$

Update $\delta_n=\operatorname{atan2}(Q_n,P_n)$ and $g_n=\sqrt{P_n^2+Q_n^2}$, then repeat the pixel step, the whitening of $(u,v)$ described below, and the frame step until the estimates stabilize. The recovered fields are $b=\sqrt{u^2+v^2}$ and $\Phi=\operatorname{atan2}(-v,u)$.

### Quadrature frame

With $g_n$ free, $(P_n,Q_n)$ is an unconstrained point in the plane rather than a point on a circle of known radius. Equation (5) is then unchanged under any invertible $2\times2$ matrix $T$,

$$\begin{pmatrix}u\\v\end{pmatrix}\leftarrow T\begin{pmatrix}u\\v\end{pmatrix},\qquad
\begin{pmatrix}P_n\\Q_n\end{pmatrix}\leftarrow T^{-\top}\begin{pmatrix}P_n\\Q_n\end{pmatrix},$$

and under the shifts $(P_n,Q_n)\leftarrow(P_n+p,\,Q_n+q)$, $a\leftarrow a-pu-qv$. Without further conditions, the recovered $\delta_n$ and $g_n$ need not be the physical ones. The scalar intercept in Eq. (9) fixes the shifts, and whitening $(u,v)$ after every pixel step fixes the shear and anisotropic scaling of $T$. At convergence,

$$\sum_j(a_j-\bar a)u_j=\sum_j(a_j-\bar a)v_j=0,\qquad\sum_ju_j^2=\sum_jv_j^2,\qquad\sum_ju_jv_j=0,$$

with $\bar a=\langle a\rangle_{x,y}$. The remaining rotations, reflections, and common scalings of $T$ are fixed by $\delta_1=0$, the step direction, and $\operatorname{median}_ng_n=1$. Unlike rotations and common scalings, shifts, shears, and anisotropic scalings change the first-order sensitivity $w_n^{(1)}=P_nv-Q_nu$ of Eq. (4). These four conditions are therefore assumptions on the physical fields, typically met by a pattern with many fringes and a background uncorrelated with the quadratures. Small transformations still leave the intensities unchanged to first order, so the conditions also fix the first-order problem.

## First-order corrections to the AIA solution

The known inputs are the measured stack and the zeroth-order AIA estimates, now marked by $(0)$. The estimates satisfy all conventions above, and $(a^{(0)},u^{(0)},v^{(0)})$ come from a final pixel step with $P_n^{(0)},Q_n^{(0)}$. A small $\Delta_n$ changes the intensities, and AIA has already absorbed part of this change into the fitted parameters. Expand all of them to first order, using $\epsilon$ only to label perturbation order:

$$\begin{aligned}
f&=f^{(0)}+\epsilon f^{(1)}+O(\epsilon^2),
&&f\in\{a,u,v,P_n,Q_n\},\\
\Delta_n&=\epsilon\Delta_n^{(1)}+O(\epsilon^2).
\end{aligned}\tag{12}$$

The corrections to $(a,u,v)$ are spatial fields shared across frames; corrections to $(P_n,Q_n)$ are scalars shared across pixels. Set $\epsilon=1$ after collecting the linear terms.

### Spatial modes

The count of Eq. (2) applies equally to the first-order unknowns: $3K$ pixel-field corrections $a^{(1)},u^{(1)},v^{(1)}$, $2N$ frame corrections $P_n^{(1)},Q_n^{(1)}$, and $NK$ phase-step errors $\Delta_n^{(1)}$. At first order, the four quadrature-frame conditions remove four more directions, so at least $2K+N-5$ degrees of freedom remain undetermined, and an unrestricted $\Delta_n^{(1)}$ cannot be recovered. To reduce the number of unknowns, choose $J$ fixed, linearly independent spatial functions $H_j(x,y)$ and let their amplitudes vary between frames:

$$\Delta_n^{(1)}(x,y)=\sum_{j=1}^J\alpha_{nj}H_j(x,y),\qquad
\langle H_j\rangle_{x,y}=0,\qquad
\langle\alpha_{nj}\rangle_n=0.\tag{13}$$

The functions are prescribed and shared by all frames; the $\alpha_{nj}$ are signed amplitudes. The zero spatial means of $H_j$ enforce the spatial convention, and $\langle\alpha_{nj}\rangle_n=0$ supplies $J$ constraints for the temporal convention.

The unknowns now contain $3K+2N+NJ$ scalars. With the $NK$ intensity equations, the $J$ coefficient-mean constraints, and the six conditions of the phase origin, the contrast scale, and the quadrature frame, the number of remaining degrees of freedom is

$$\nu=3K+2N+NJ-NK-J-6.\tag{14}$$

A unique solution is possible only if $\nu\le0$, that is,

$$J\le J_{\max}\equiv\left\lfloor\frac{(N-3)(K-2)}{N-1}\right\rfloor,\qquad N\ge3.\tag{15}$$

At $N=3$, $J_{\max}=0$: the AIA fit is determined, but no phase-error mode is. At $N=4$ the count permits modes, but none is identifiable: $\operatorname{span}\{1,P_n,Q_n\}$ has codimension one in $\mathbb R^4$, so for each $H_j$ some nonzero $\alpha_{nj}$ with zero frame mean has $\alpha_{nj}P_n$ and $\alpha_{nj}Q_n$ in that span, and its intensity change is absorbed by $a^{(1)},u^{(1)},v^{(1)}$. Identifying modes therefore requires $N\ge5$. Since $K\gg N$ in practice, $J_{\max}\approx K(N-3)/(N-1)$, for example about $K/2$ at $N=5$. The count is only necessary; the fit must still determine all remaining unknowns uniquely.

### First-order model of the residual

Define the baseline residual

$$r_n=I_n-\big(a^{(0)}+P_n^{(0)}u^{(0)}+Q_n^{(0)}v^{(0)}\big).\tag{16}$$

Substituting Eqs. (12) and (13) into the model and collecting the terms linear in $\epsilon$ gives, up to noise and second-order terms,

$$r_n=a^{(1)}+P_n^{(0)}u^{(1)}+Q_n^{(0)}v^{(1)}+u^{(0)}P_n^{(1)}+v^{(0)}Q_n^{(1)}+w_n\sum_{j=1}^J\alpha_{nj}H_j,\tag{17}$$

where $w_n=P_n^{(0)}v^{(0)}-Q_n^{(0)}u^{(0)}$ is the first-order sensitivity $w_n^{(1)}$ of Eq. (4) evaluated at the AIA estimates.

### Removing the pixel corrections

At one pixel, collect the intensities of all frames in the column of Eq. (8), and let $A$ and $A_p$ be the matrices of Eq. (7) built from $P_n^{(0)},Q_n^{(0)}$. Define the $N\times N$ matrix

$$\Pi=I-AA_p^{-1}A^\top,\tag{18}$$

where $I$ is the $N\times N$ identity matrix. In regression terms, $AA_p^{-1}A^\top$ is the hat matrix of the pixel-step fit and $\Pi$ is its residual-maker matrix (Hoaglin and Welsch, 1978). By Eq. (8), the pixel step fits the intensity column by $AA_p^{-1}A^\top$ applied to it, so the column splits into the fitted part and the leftover:

$$\begin{pmatrix}I_1\\\vdots\\I_N\end{pmatrix}
=AA_p^{-1}A^\top\begin{pmatrix}I_1\\\vdots\\I_N\end{pmatrix}
+\Pi\begin{pmatrix}I_1\\\vdots\\I_N\end{pmatrix},\qquad
\Pi\begin{pmatrix}I_1\\\vdots\\I_N\end{pmatrix}=\begin{pmatrix}r_1\\\vdots\\r_N\end{pmatrix}.$$

The leftover is the residual of Eq. (16). $\Pi$ depends only on $P_n^{(0)},Q_n^{(0)}$ and is the same at every pixel. Since $A_p=A^\top A$,

$$\Pi A=0,\qquad A^\top\Pi=0,\qquad\Pi^2=\Pi,\qquad\Pi^\top=\Pi.$$

The first relation states that $\Pi$ removes every column of the form $A(a,u,v)^\top$. The second states that a leftover satisfies the pixel-step normal equations and cannot be fitted further. The third states that $\Pi$ leaves a leftover unchanged; in particular, $\Pi$ leaves the residual column unchanged. $\Pi$ has rank $N-3$, so each pixel retains $N-3$ independent combinations of its $N$ frame values; for $N=3$, $\Pi=0$.

**Leftover of the AIA fit.** Consider one pixel with exact $P_n,Q_n$, true fields $a,u,v$, and intensities $I_n=a+P_nu+Q_nv+e_n$, where $e_n=w_n\Delta_n$ is the first-order phase-step signal. By Eq. (8) and $\Pi A=0$, the pixel step returns

$$\begin{pmatrix}a^{(0)}\\u^{(0)}\\v^{(0)}\end{pmatrix}=\begin{pmatrix}a\\u\\v\end{pmatrix}+A_p^{-1}A^\top\begin{pmatrix}e_1\\\vdots\\e_N\end{pmatrix},\qquad
\begin{pmatrix}r_1\\\vdots\\r_N\end{pmatrix}=\Pi\begin{pmatrix}e_1\\\vdots\\e_N\end{pmatrix}.\tag{19}$$

The fitted part of the phase-step signal is absorbed into the AIA estimates; only its leftover remains in the residual. Fitting the residual by $e_n$ with the pixel corrections set to zero therefore underestimates $\Delta_n$. The residual must instead be compared with the leftover of the signal.

**Elimination.** At one pixel, the first three terms of Eq. (17) form the column $A\big(a^{(1)},u^{(1)},v^{(1)}\big)^\top$, which $\Pi$ removes, while $\Pi$ leaves the residual column unchanged. Applying $\Pi$ to Eq. (17) gives

$$\begin{pmatrix}r_1\\\vdots\\r_N\end{pmatrix}
=\Pi\begin{pmatrix}
u^{(0)}P_1^{(1)}+v^{(0)}Q_1^{(1)}+w_1\sum_j\alpha_{1j}H_j\\
\vdots\\
u^{(0)}P_N^{(1)}+v^{(0)}Q_N^{(1)}+w_N\sum_j\alpha_{Nj}H_j
\end{pmatrix}.\tag{20}$$

Equation (20) holds at every pixel and contains only the $2N+NJ$ unknowns $P_n^{(1)},Q_n^{(1)},\alpha_{nj}$, all shared across pixels. A single pixel provides $N-3$ independent equations, fewer than the $N-1$ zero-mean values $\Delta_n^{(1)}$ at that pixel; the shared spatial functions of Eq. (13) combine the equations of all pixels.

Eliminating a group of regressors by applying the residual-maker matrix to both the data and the remaining regressors does not change the least-squares estimates of the remaining unknowns (Frisch and Waugh, 1933; Lovell, 1963). A least-squares fit of Eq. (20) therefore gives the same $P_n^{(1)},Q_n^{(1)},\alpha_{nj}$ as a joint fit of Eq. (17) that includes the pixel corrections. The same elimination underlies the variable projection method for separable least-squares problems (Golub and Pereyra, 1973).

### Fit of the frame corrections and phase-step coefficients

Eq. (20) is fitted over all pixels by least squares. In this subsection, $u,v,P_n,Q_n$ denote the zeroth-order estimates $u^{(0)},v^{(0)},P_n^{(0)},Q_n^{(0)}$. Write

$$z_n=uP_n^{(1)}+vQ_n^{(1)}+w_n\sum_{j=1}^J\alpha_{nj}H_j$$

for the entries of the column inside Eq. (20). The loss is

$$\mathcal L_1=\sum_{x,y}\sum_{n=1}^N\Big(r_n-\sum_{m=1}^N\Pi_{nm}z_m\Big)^2
=\sum_{x,y}\Big(\sum_nr_n^2-2\sum_nr_nz_n+\sum_{n,m}z_n\Pi_{nm}z_m\Big),\tag{21}$$

where the second form uses $\Pi^\top\Pi=\Pi$ and the fact that $\Pi$ leaves the residual column unchanged. Setting the derivatives with respect to $P_n^{(1)}$, $Q_n^{(1)}$, and $\alpha_{nj}$ to zero gives, for $n=1,\dots,N$ and $j=1,\dots,J$,

$$\begin{aligned}
\sum_m\Pi_{nm}\Big[P_m^{(1)}\sum_{x,y}u^2+Q_m^{(1)}\sum_{x,y}uv+\sum_{j'}\alpha_{mj'}\sum_{x,y}uw_mH_{j'}\Big]&=\sum_{x,y}ur_n,\\
\sum_m\Pi_{nm}\Big[P_m^{(1)}\sum_{x,y}uv+Q_m^{(1)}\sum_{x,y}v^2+\sum_{j'}\alpha_{mj'}\sum_{x,y}vw_mH_{j'}\Big]&=\sum_{x,y}vr_n,\\
\sum_m\Pi_{nm}\Big[P_m^{(1)}\sum_{x,y}uw_nH_j+Q_m^{(1)}\sum_{x,y}vw_nH_j+\sum_{j'}\alpha_{mj'}\sum_{x,y}w_nw_mH_jH_{j'}\Big]&=\sum_{x,y}w_nH_jr_n.
\end{aligned}\tag{22}$$

Since $w_n=P_nv-Q_nu$, every pixel sum in Eq. (22) that contains $w_n$ reduces to sums without a frame index:

$$\begin{aligned}
\sum_{x,y}uw_mH_j&=P_m\sum_{x,y}uvH_j-Q_m\sum_{x,y}u^2H_j,\qquad
\sum_{x,y}vw_mH_j=P_m\sum_{x,y}v^2H_j-Q_m\sum_{x,y}uvH_j,\\
\sum_{x,y}w_nw_mH_jH_{j'}&=P_nP_m\sum_{x,y}v^2H_jH_{j'}-\big(P_nQ_m+Q_nP_m\big)\sum_{x,y}uvH_jH_{j'}+Q_nQ_m\sum_{x,y}u^2H_jH_{j'},\\
\sum_{x,y}w_nH_jr_n&=P_n\sum_{x,y}vH_jr_n-Q_n\sum_{x,y}uH_jr_n.
\end{aligned}\tag{23}$$

The matrix of Eq. (22) therefore requires three scalar sums, three sums per spatial function, and three sums per pair of spatial functions, all obtained in one pass over the pixels. The right-hand sides require four sums per frame and spatial function. Under the whitening conditions, $\sum_{x,y}uv=0$ and $\sum_{x,y}u^2=\sum_{x,y}v^2$.

**Undetermined directions.** $\Pi$ removes the columns $(1,\dots,1)^\top$, $(P_1,\dots,P_N)^\top$, and $(Q_1,\dots,Q_N)^\top$. Adding any combination of them to $(P_1^{(1)},\dots,P_N^{(1)})^\top$ or to $(Q_1^{(1)},\dots,Q_N^{(1)})^\top$ leaves Eq. (20) unchanged; these six directions are the linearized shifts and $2\times2$ transformations of the quadrature frame. A frame-constant $\alpha_{nj}=c_j$ adds $\big(\sum_jc_jH_j\big)\big(P_nv-Q_nu\big)$, which $\Pi$ also removes. Fix these $6+J$ directions by

$$\sum_nP_n^{(1)}=\sum_nP_nP_n^{(1)}=\sum_nQ_nP_n^{(1)}=0,\qquad
\sum_nQ_n^{(1)}=\sum_nP_nQ_n^{(1)}=\sum_nQ_nQ_n^{(1)}=0,\qquad
\sum_n\alpha_{nj}=0.\tag{24}$$

The first six conditions state $A^\top(P_1^{(1)},\dots,P_N^{(1)})^\top=A^\top(Q_1^{(1)},\dots,Q_N^{(1)})^\top=0$. They select one representative; the gauge and quadrature-frame conventions are imposed when the pixel corrections are recovered.

**Solution.** Collect the unknowns in the column

$$\beta=\big(P_1^{(1)},\dots,P_N^{(1)},\,Q_1^{(1)},\dots,Q_N^{(1)},\,\alpha_{11},\dots,\alpha_{1J},\dots,\alpha_{N1},\dots,\alpha_{NJ}\big)^\top$$

of length $2N+NJ$. Eq. (22) is $M\beta=b$, with

$$M=\begin{pmatrix}
\Pi\sum_{x,y}u^2&\Pi\sum_{x,y}uv&D_u\\
\Pi\sum_{x,y}uv&\Pi\sum_{x,y}v^2&D_v\\
D_u^\top&D_v^\top&E
\end{pmatrix},$$

$$b=\Big(\sum_{x,y}ur_1,\dots,\sum_{x,y}ur_N,\ \sum_{x,y}vr_1,\dots,\sum_{x,y}vr_N,\ \sum_{x,y}w_1H_1r_1,\dots,\sum_{x,y}w_1H_Jr_1,\dots,\sum_{x,y}w_NH_1r_N,\dots,\sum_{x,y}w_NH_Jr_N\Big)^\top.$$

Here $\Pi\sum_{x,y}u^2$ is the $N\times N$ matrix $\Pi$ multiplied by the scalar $\sum_{x,y}u^2$. The blocks coupling the frame corrections to the coefficients are

$$D_u=\begin{pmatrix}
\Pi_{11}d^u_1&\cdots&\Pi_{1N}d^u_N\\
\vdots&&\vdots\\
\Pi_{N1}d^u_1&\cdots&\Pi_{NN}d^u_N
\end{pmatrix},\qquad
d^u_m=\Big(\sum_{x,y}uw_mH_1,\ \dots,\ \sum_{x,y}uw_mH_J\Big),$$

of size $N\times NJ$, and $D_v$ is the same with $v$ in place of $u$. The coefficient block is

$$E=\begin{pmatrix}
\Pi_{11}E_{11}&\cdots&\Pi_{1N}E_{1N}\\
\vdots&&\vdots\\
\Pi_{N1}E_{N1}&\cdots&\Pi_{NN}E_{NN}
\end{pmatrix},\qquad
E_{nm}=\begin{pmatrix}
\sum_{x,y}w_nw_mH_1H_1&\cdots&\sum_{x,y}w_nw_mH_1H_J\\
\vdots&&\vdots\\
\sum_{x,y}w_nw_mH_JH_1&\cdots&\sum_{x,y}w_nw_mH_JH_J
\end{pmatrix},$$

of size $NJ\times NJ$ with $J\times J$ blocks $E_{nm}$. All sums are evaluated with Eq. (23). Eq. (24) is $C\beta=0$, with

$$C=\begin{pmatrix}
A^\top&0&0\\
0&A^\top&0\\
0&0&\big(I_J\ \cdots\ I_J\big)
\end{pmatrix},\qquad
A^\top=\begin{pmatrix}
1&\cdots&1\\
P_1&\cdots&P_N\\
Q_1&\cdots&Q_N
\end{pmatrix},$$

where $I_J$ is the $J\times J$ identity matrix, repeated $N$ times; $C$ has size $(6+J)\times(2N+NJ)$. The conditions only select among solutions with the same loss, so they can be appended as extra equations with zero right-hand side. If $M+C^\top C$ is invertible, the solution is

$$\beta=\big(M+C^\top C\big)^{-1}b.\tag{25}$$

$M+C^\top C$ is invertible exactly when Eq. (20) has no undetermined directions besides those fixed by Eq. (24); this requires $N\ge5$ and $J\le J_{\max}$ (Eq. 15). The system has size $2N+NJ$ and does not grow with $K$.

### Pixel corrections and normalization

With $P_n^{(1)},Q_n^{(1)},\alpha_{nj}$ known, Eq. (17) at one pixel is a pixel step for $a^{(1)},u^{(1)},v^{(1)}$ with the column $r_n-uP_n^{(1)}-vQ_n^{(1)}-w_n\Delta_n^{(1)}$ as data. By Eq. (8), its solution is $A_p^{-1}A^\top$ applied to this column. The residual column satisfies $A^\top(r_1,\dots,r_N)^\top=0$, and Eq. (24) gives $A^\top(P_1^{(1)},\dots,P_N^{(1)})^\top=A^\top(Q_1^{(1)},\dots,Q_N^{(1)})^\top=0$, so only the phase-step term remains:

$$\begin{pmatrix}a^{(1)}\\u^{(1)}\\v^{(1)}\end{pmatrix}
=-A_p^{-1}\begin{pmatrix}\sum_nw_n\Delta_n^{(1)}\\\sum_nP_nw_n\Delta_n^{(1)}\\\sum_nQ_nw_n\Delta_n^{(1)}\end{pmatrix},\qquad
\Delta_n^{(1)}=\sum_{j=1}^J\alpha_{nj}H_j.\tag{26}$$

This is the negative of the shift of the AIA estimates in Eq. (19). By the Frisch–Waugh–Lovell theorem, Eqs. (25) and (26) together give the joint least-squares solution of Eq. (17). Setting $\epsilon=1$, the corrected fields are

$$a=a^{(0)}+a^{(1)},\qquad u=u^{(0)}+u^{(1)},\qquad v=v^{(0)}+v^{(1)},\qquad P_n=P_n^{(0)}+P_n^{(1)},\qquad Q_n=Q_n^{(0)}+Q_n^{(1)}.\tag{27}$$

**Normalization.** The corrected fields satisfy Eq. (24) rather than the gauge and quadrature-frame conventions. They deviate from those conventions only at first order, so the transformations that restore them leave $\Delta_n^{(1)}$ unchanged and change the intensities only at second order. The zero-mean conventions on $\Delta_n^{(1)}$ already hold by Eq. (13). Apply the following steps in order; each preserves the conditions established by the previous ones.

1. **Shift.** Solve

   $$\begin{pmatrix}
   \sum_{x,y}(u-\bar u)^2&\sum_{x,y}(u-\bar u)(v-\bar v)\\
   \sum_{x,y}(u-\bar u)(v-\bar v)&\sum_{x,y}(v-\bar v)^2
   \end{pmatrix}
   \begin{pmatrix}p\\q\end{pmatrix}
   =\begin{pmatrix}\sum_{x,y}(a-\bar a)u\\\sum_{x,y}(a-\bar a)v\end{pmatrix},$$

   where bars denote pixel means, and apply $a\leftarrow a-pu-qv$, $P_n\leftarrow P_n+p$, $Q_n\leftarrow Q_n+q$. Thus $\sum_{x,y}(a-\bar a)u=\sum_{x,y}(a-\bar a)v=0$.

2. **Whitening.** With $G=\begin{pmatrix}\sum_{x,y}u^2&\sum_{x,y}uv\\\sum_{x,y}uv&\sum_{x,y}v^2\end{pmatrix}$ and its symmetric square root $G^{1/2}$, apply

   $$\begin{pmatrix}u\\v\end{pmatrix}\leftarrow G^{-1/2}\begin{pmatrix}u\\v\end{pmatrix},\qquad
   \begin{pmatrix}P_n\\Q_n\end{pmatrix}\leftarrow G^{1/2}\begin{pmatrix}P_n\\Q_n\end{pmatrix}.$$

   Thus $\sum_{x,y}u^2=\sum_{x,y}v^2$ and $\sum_{x,y}uv=0$. The shift condition is preserved because $a$ is unchanged and the new $u,v$ are linear combinations of the old ones.

3. **Phase origin.** With $\delta_1=\operatorname{atan2}(Q_1,P_1)$, apply

   $$\begin{pmatrix}P_n\\Q_n\end{pmatrix}\leftarrow R\begin{pmatrix}P_n\\Q_n\end{pmatrix},\qquad
   \begin{pmatrix}u\\v\end{pmatrix}\leftarrow R\begin{pmatrix}u\\v\end{pmatrix},\qquad
   R=\begin{pmatrix}\cos\delta_1&\sin\delta_1\\-\sin\delta_1&\cos\delta_1\end{pmatrix}.$$

   This is the phase-origin convention, $\delta_n\leftarrow\delta_n-\delta_1$ and $\Phi\leftarrow\Phi+\delta_1$, written in quadrature variables. Thus $\delta_1=0$.

4. **Contrast scale.** With $s=\operatorname{median}_n\sqrt{P_n^2+Q_n^2}$, apply $P_n\leftarrow P_n/s$, $Q_n\leftarrow Q_n/s$, $u\leftarrow su$, $v\leftarrow sv$. Thus $\operatorname{median}_ng_n=1$.

The final estimates are $\delta_n=\operatorname{atan2}(Q_n,P_n)$, $g_n=\sqrt{P_n^2+Q_n^2}$, $b=\sqrt{u^2+v^2}$, $\Phi=\operatorname{atan2}(-v,u)$, and the phase-step error $\Delta_n^{(1)}$ of Eq. (26). Their remaining errors are of second order in $\Delta_n$.

### VP-AIA algorithm and cost

1. **AIA baseline.** Run the pixel step, whitening, and frame step to convergence, finish with a pixel step, and apply normalization steps 1–4.
2. **Residual and projector.** Compute $r_n$ (Eq. 16), $w_n$ (Eq. 17), and $\Pi$ (Eq. 18).
3. **Pixel sums.** In one pass over the pixels, compute the sums of Eq. (23) and the right-hand sides of Eq. (22).
4. **Frame corrections and coefficients.** Assemble $M$, $b$, and $C$, and solve Eq. (25).
5. **Pixel corrections.** Evaluate $\Delta_n^{(1)}$ and Eq. (26) at every pixel, and form the corrected fields of Eq. (27).
6. **Normalization.** Apply normalization steps 1–4 to the corrected fields.

One AIA iteration costs $O(NK)$ operations. Steps 2–6 add $O(NKJ)$ operations for the residual, the right-hand sides, and Eq. (26); $O(KJ^2)$ operations for the sums of Eq. (23); and $O\big((2N+NJ)^3\big)$ operations for the solve of Eq. (25). The first-order pass therefore costs about as much as $J+J^2/N$ AIA iterations. Besides the stack and the fields already used by AIA, it stores the $J$ spatial functions and matrices of size $2N+NJ$. $\Delta_n^{(1)}$ need not be stored, since it is evaluated from $\alpha_{nj}$ and $H_j$.

**Larger phase-step errors.** When $\Delta_n$ is not small enough for second-order errors to be negligible, re-linearize around the corrected estimates. Remove the current phase-step signal from the measured stack with the full model, Eq. (1):

$$I_n'=I_n-g_nb\big[\cos(\Phi+\delta_n+\Delta_n)-\cos(\Phi+\delta_n)\big],\tag{28}$$

where $a,b,\Phi,\delta_n,g_n$ are the current estimates and $\Delta_n$ is the accumulated phase-step error. Run AIA on $I_n'$ starting from the current $P_n,Q_n$, apply steps 2–6 to obtain an increment $\Delta_n^{(1)}$, and update $\Delta_n\leftarrow\Delta_n+\Delta_n^{(1)}$. The increments have zero spatial and frame means, so the accumulated $\Delta_n$ keeps both conventions. Each round reduces the remaining error by a factor of the order of $\max|\Delta_n|$; stop when the increment becomes negligible.

## Noise of the corrected estimates

Consider noise $\varepsilon_n(x,y)$ with variance $\sigma_0^2$, independent across frames and pixels, and treat $P_n,Q_n$ as exact; the noise of their estimates is not included here. At one pixel, the pixel-step estimate of $\Phi=\operatorname{atan2}(-v,u)$ is linear in the intensity column. By Eq. (8), its error is $\sum_ns_n\varepsilon_n$ with

$$\begin{pmatrix}s_1\\\vdots\\s_N\end{pmatrix}=AA_p^{-1}\begin{pmatrix}0\\-\sin\Phi/b\\-\cos\Phi/b\end{pmatrix},$$

so plain AIA gives $\sigma_\Phi^2=\sigma_0^2\sum_ns_n^2$.

**Noise of the fitted unknowns.** The residual noise at each pixel is $\Pi(\varepsilon_1,\dots,\varepsilon_N)^\top$. Each entry of $b$ is a pixel sum of the residual weighted by one column of Eq. (20) after applying $\Pi$. Since $\Pi^\top\Pi=\Pi$, the noise of $b$ has covariance $\sigma_0^2M$, and by Eq. (25) the noise of $\beta$ has covariance

$$\sigma_0^2\big(M+C^\top C\big)^{-1}M\big(M+C^\top C\big)^{-1}.$$

**Effect on the phase.** By Eq. (26), noise $\delta\Delta_n^{(1)}$ in the fitted phase-step error changes $(a,u,v)$ by $-A_p^{-1}A^\top$ applied to the column $\big(w_1\delta\Delta_1^{(1)},\dots,w_N\delta\Delta_N^{(1)}\big)^\top$, and therefore changes $\Phi$ by $-\sum_ns_nw_n\,\delta\Delta_n^{(1)}$. Since $(s_1,\dots,s_N)^\top$ lies in the column space of $A$, $\Pi$ removes it, and this change is uncorrelated with $\sum_ns_n\varepsilon_n$ at the same pixel. The correction therefore adds a variance term:

$$\sigma_\Phi^2=\sigma_0^2\sum_ns_n^2+\sigma_0^2\,\gamma^\top\big(M+C^\top C\big)^{-1}M\big(M+C^\top C\big)^{-1}\gamma,\tag{29}$$

where $\gamma$ is the column of length $2N+NJ$, ordered as $\beta$, with zeros at the positions of $P_n^{(1)},Q_n^{(1)}$ and the entry $s_nw_nH_j$ at the position of $\alpha_{nj}$, evaluated at the pixel.

**Uniform steps.** Assume uniform steps with $N\ge5$, constant $g_n$ and $b$, many fringes, and $\langle H_jH_{j'}\rangle_{x,y}=\delta_{jj'}$. Then the frame corrections decouple from the coefficients, the coefficient block of $M$ has entries $\tfrac12g^2b^2K\,\Pi_{nm}\cos(\delta_n-\delta_m)\,\delta_{jj'}$, and $s_nw_n=\big(1-\cos(2\Phi+2\delta_n)\big)/N$. The $N\times N$ matrix with entries $\Pi_{nm}\cos(\delta_n-\delta_m)$ has eigenvalue $0$ on frame-constant columns, $\tfrac12$ on the first and second temporal harmonics, and $1$ on the others. The frame-varying part of $s_nw_n$ lies in the second harmonic, so the eigenvalue $\tfrac12$ doubles its contribution. Averaging Eq. (29) over the fringe phase gives

$$\frac{\sigma_\Phi^2}{\sigma_0^2\sum_ns_n^2}\approx1+\frac{\sum_jH_j(x,y)^2}{K},\qquad
\bigg\langle\frac{\sigma_\Phi^2}{\sigma_0^2\sum_ns_n^2}\bigg\rangle_{x,y}\approx1+\frac{J}{K}.\tag{30}$$

The correction increases the phase noise by a relative amount of order $J/K$, largest where $\sum_jH_j^2$ is large and negligible for $K\gg J$. Equation (30) is a reading of Eq. (29) under the four conditions above, not a substitute for it: away from them — irregular steps, per-frame gains, a contrast that varies across the field, or modes the fringe pattern resolves unevenly — Eq. (29) is the form to use.

**Noise of the zeroth-order steps.** The first term of Eq. (29) is `aia.md` Eq. (26), the phase variance at known $\delta_n$ and $g_n$. The zeroth-order solve estimates both from the same frames, which adds the $O(1/K)$ term of `aia.md` Eqs. (40) and (45); the zeros of $\gamma$ at the positions of $P_n^{(1)},Q_n^{(1)}$ say only that the first-order frame corrections leave $\Phi$ unchanged to first order, not that the steps carry no noise. The two terms add, as in `sf_aia.md` Eq. (E9), and their cross-correlation is not derived here.

**Comparison with SF-AIA.** For the same $J$-mode field fitted from the same data, `sf_aia.md` Eq. (E8) gives $1+J/(4N_p)$ where Eq. (30) gives $1+J/K$, with $K=N_p$. The factor of four is the estimator, not a disagreement between the two derivations. Both reduce to a quadratic form in the $N\times N$ matrix with entries $\Pi_{nm}\cos(\delta_n-\delta_m)$, whose eigenvalue on the second temporal harmonic — where the frame-varying part of $s_nw_n$ lies — is $\tfrac12$. SF-AIA's per-frame fit multiplies that harmonic by the eigenvalue; VP-AIA's joint fit inverts it. The two therefore differ by $2/\tfrac12=4$ in variance, and it is the same factor as in `sf_aia.md` §"Bias of a single pass": one SF-AIA pass recovers the first and second harmonics of $c_{jn}$ at half their size, and attenuates their noise by that same half. Paying $J/K$ rather than $J/(4N_p)$ is therefore the price of the unbiased field, not an extra cost of the method: SF-AIA buys its smaller variance with the attenuation that is exactly its bias. Its refinement loop gives that attenuation back round by round, and its noise passes Eq. (30) rather than settling at it, since the alternation's fixed point is not the joint least-squares solution fitted here. VP-AIA reaches the unbiased field and Eq. (30) together, in one pass.

## References

- R. Frisch and F. V. Waugh, "Partial time regressions as compared with individual trends," *Econometrica* **1**(4), 387–401 (1933).
- M. C. Lovell, "Seasonal adjustment of economic time series and multiple regression analysis," *Journal of the American Statistical Association* **58**(304), 993–1010 (1963).
- G. H. Golub and V. Pereyra, "The differentiation of pseudo-inverses and nonlinear least squares problems whose variables separate," *SIAM Journal on Numerical Analysis* **10**(2), 413–432 (1973).
- D. C. Hoaglin and R. E. Welsch, "The hat matrix in regression and ANOVA," *The American Statistician* **32**(1), 17–22 (1978).
- Z. Wang and B. Han, "Advanced iterative algorithm for phase extraction of randomly phase-shifted interferograms," *Optics Letters* **29**(14), 1671–1673 (2004).
