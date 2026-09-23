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

Throughout, these symbols denote the true parameters; AIA estimates of them carry a hat, as in $\hat\Phi$, and the corrected estimates of VP-AIA a tilde, as in $\tilde\Phi$ or $\tilde\Delta_n$.

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

These conventions define which parts are called static phase, piston, phase-step error, contrast, and amplitude. The AIA parametrization introduces four further freedoms, fixed by the quadrature-frame conventions of the AIA baseline.

### Constraints for $\Delta_n$

For $N$ frames and $K$ pixels, Eq. (1) contains $3K+2N+NK$ unknown scalar values: the fields $a,b,\Phi$ at every pixel, $g_n$ and $\delta_n$ for every frame, and $\Delta_n$ at every pixel of every frame. The $NK$ measured intensities provide at most $NK$ independent equations. The zero spatial mean in each frame and the zero frame mean at each pixel impose $N+K-1$ independent conditions, since their total-sum condition is shared; the phase origin and the contrast scale add two more. The number of undetermined degrees of freedom is therefore at least

$$(3K+2N+NK)-NK-(N+K-1)-2=2K+N-1.\tag{2}$$

An unrestricted $\Delta_n$ therefore cannot be recovered from the intensities alone, and its spatial form must be restricted.

### Cost of the full model

Even with the gauge conventions and a restricted spatial form of $\Delta_n$, Eq. (1) is much harder to solve than its piston limit $\Delta_n\equiv0$. The piston limit is linear in the pixel fields when the frame parameters are fixed, and linear in the frame parameters when the pixel fields are fixed; AIA (Wang and Han, 2004) alternates these two linear least-squares fits, each with one small normal matrix shared by all pixels or all frames. A spatially varying $\Delta_n$ breaks this split. It depends on both frame and pixel and enters inside the cosine, so the fit across pixels in each frame becomes nonlinear in the step-error unknowns, and the fit across frames needs its own normal matrix at every pixel.

The full model can still be solved as a general nonlinear least-squares problem. The general iterative algorithm (GIA) of Chen and Kemao (2021) does so by alternating Levenberg–Marquardt fits over groups of unknowns; with equal iteration counts it reports about 72 times the computation time of AIA (558.8 s against 7.7 s). A gap of this size rules out such solvers for long measurement series and for real-time phase extraction.

VP-AIA takes a different route. The phase-step error is usually small, so the AIA solution is already close to the exact one; VP-AIA keeps it as a baseline and recovers $\Delta_n$ together with the bias of that baseline as small perturbative corrections.

## VP-AIA perturbation model

First, assume $|\Delta_n(x,y)|\ll1$ rad, so that the effect of $\Delta_n$ on the intensities can be ordered by its powers. At each pixel, write $\theta_n=\Phi+\delta_n$ and suppress $(x,y)$. With all other parameters held fixed, the Taylor expansion about $\Delta_n=0$ is

$$\cos(\theta_n+\Delta_n)=\sum_{k=0}^{\infty}\frac{\Delta_n^k}{k!}\cos\left(\theta_n+\frac{k\pi}{2}\right),\tag{3}$$

and Eq. (1) becomes

$$I_n=a+g_n b\cos\theta_n+\sum_{k=1}^{\infty}\kappa_{n,k}\Delta_n^k,\qquad \kappa_{n,k}=\frac{g_n b}{k!}\cos\left(\theta_n+\frac{k\pi}{2}\right).\tag{4}$$

The coefficient $\kappa_{n,k}$ is the sensitivity of the intensity to the $k$-th power of the phase-step error. 

**Quadrature form.** Define the pixel fields and frame parameters

$$u=b\cos\Phi,\qquad v=-b\sin\Phi,\qquad P_n=g_n\cos\delta_n,\qquad Q_n=g_n\sin\delta_n.\tag{4a}$$

Then $g_nb\cos\theta_n=P_nu+Q_nv$ and $-g_nb\sin\theta_n=P_nv-Q_nu$, and Eq. (4) becomes

$$I_n=a+P_nu+Q_nv+\sum_{k=1}^{\infty}\kappa_{n,k}\Delta_n^k,\tag{4b}$$

where, since $\cos(\theta_n+k\pi/2)$ alternates between $\pm\cos\theta_n$ and $\mp\sin\theta_n$,

$$\kappa_{n,k}=\begin{cases}
\dfrac{(-1)^{(k-1)/2}}{k!}\big(P_nv-Q_nu\big), & k\text{ odd},\\[2ex]
\dfrac{(-1)^{k/2}}{k!}\big(P_nu+Q_nv\big), & k\text{ even}.
\end{cases}\tag{4c}$$

The zeroth-order term is the piston model, linear in the pixel fields for fixed frame parameters and in the frame parameters for fixed pixel fields. 

### AIA baseline

With $\Delta_n\equiv0$, Eq. (4b) reduces to the piston model in quadrature form,

$$I_n(x,y)=a(x,y)+P_nu(x,y)+Q_nv(x,y).\tag{5}$$

This equation can be solved through AIA, however due to the fit of underparametrized equation to real intensities, AIA variables absorb error from $\Delta_n$. Let us name AIA variables in the following way: ($\hat a, \hat u, \hat v, \{ \hat P_n \}, \{ \hat Q_n \}$), so AIA intensity become:

$$\hat I_n(x,y)=\hat a(x,y)+ \hat P_n \hat u(x,y)+\hat Q_n \hat v(x,y).\tag{6}$$

Now, we can show how error affects AIA variables. 

Write the measured intensity as the piston model plus the part it cannot represent,

$$I_n=a+P_nu+Q_nv+e_n,\qquad e_n=\sum_{k=1}^{\infty}\kappa_{n,k}\Delta_n^k,\tag{7}$$

by Eq. (4b). AIA fits Eq. (6) to $I_n$, and the fit assigns part of $e_n$ to its variables.

AIA's last pixel step holds $\hat P_n,\hat Q_n$ fixed and solves, at each pixel, the linear regression of `aia.md` Eqs. (5)–(6), with design matrix $\hat A$ of rows $(1,\hat P_n,\hat Q_n)$. Write the frame-parameter errors as $\varepsilon_{P_n}=\hat P_n-P_n$ and $\varepsilon_{Q_n}=\hat Q_n-Q_n$. By Eq. (7),

$$I_n=a+\hat P_nu+\hat Q_nv+s_n,\qquad s_n=e_n-\varepsilon_{P_n}u-\varepsilon_{Q_n}v,$$

where the first three terms form the column $\hat A(a,u,v)^\top$ and $s_n$ is the bias source of the pixel step. Since $(\hat A^\top\hat A)^{-1}\hat A^\top\hat A$ is the identity, the pixel-step solution is

$$\begin{pmatrix}\hat a\\\hat u\\\hat v\end{pmatrix}=\big(\hat A^\top\hat A\big)^{-1}\hat A^\top\begin{pmatrix}I_1\\\vdots\\I_N\end{pmatrix}
=\begin{pmatrix}a\\u\\v\end{pmatrix}+\big(\hat A^\top\hat A\big)^{-1}\hat A^\top\begin{pmatrix}s_1\\\vdots\\s_N\end{pmatrix}.\tag{8}$$

The pixel fields are therefore biased by the projection of $s_n$ onto the regressors $(1,\hat P_n,\hat Q_n)$. The bias source has two parts: the leakage of the unmodeled intensity $e_n$ of the phase-step error, and the bias carried over from the frame-parameter errors $\varepsilon_{P_n},\varepsilon_{Q_n}$, which the frame step produces.

## First-order corrections to the AIA solution

The known inputs are the measured stack and the AIA baseline estimates, now marked by a hat. The estimates satisfy all conventions above, and $(\hat a,\hat u,\hat v)$ come from a final pixel step with $\hat P_n,\hat Q_n$. A small $\Delta_n$ changes the intensities, and AIA has already absorbed part of this change into the fitted parameters.

Only the estimates depend on $\Delta_n$; the true parameters are properties of the experiment and have no expansion of their own. To give "order in $\Delta_n$" a meaning for a field, fix its spatial shape and scale it, $\Delta_n(x,y)=\epsilon D_n(x,y)$ with $D_n$ fixed, so that the stack is analytic in the single scalar $\epsilon$ and each estimate is a function of it. At $\epsilon=0$ the piston model describes the data exactly, its residual vanishes at the exact parameters, and the conventions above leave a single solution, so a noiseless fit returns them: $\hat f(0)=f^{(0)}$. This is a property of the fit, not a choice of notation. The estimates also depend smoothly on $\epsilon$, since they solve AIA's own normal equations together with the normalization conditions, and those determine the solution uniquely at $\epsilon=0$; a single analytic branch therefore passes through that point, and its Taylor expansion is

$$\hat f=f^{(0)}+f^{(1)}+O(\Delta_n^2),\qquad f^{(k)}=O(\Delta_n^k),\qquad f\in\{a,u,v,P_n,Q_n\},\tag{12}$$

Here $f^{(0)}$ is the exact parameter, the quantity the method must recover, and $f^{(1)}=O(\Delta_n)$ is the bias the fit absorbed, so that $\hat f-f^{(0)}=O(\Delta_n)$ — the one fact every perturbative statement below rests on. The step-error field $\Delta_n$ is itself first order throughout and so carries no expansion of its own. The biases $a^{(1)},u^{(1)},v^{(1)}$ are spatial fields shared across frames; $P_n^{(1)},Q_n^{(1)}$ are scalars shared across pixels.

### Spatial modes

The count of Eq. (2) applies equally to the first-order unknowns: $3K$ pixel-field corrections $a^{(1)},u^{(1)},v^{(1)}$, $2N$ frame corrections $P_n^{(1)},Q_n^{(1)}$, and $NK$ phase-step errors $\Delta_n$. At first order, the four quadrature-frame conditions remove four more directions, so at least $2K+N-5$ degrees of freedom remain undetermined, and an unrestricted $\Delta_n$ cannot be recovered. To reduce the number of unknowns, choose $J$ fixed, linearly independent spatial functions $H_j(x,y)$ and let their amplitudes vary between frames:

$$\Delta_n(x,y)=\sum_{j=1}^J\alpha_{nj}H_j(x,y),\qquad
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

$$r_n=I_n-\big(\hat a+\hat P_n\hat u+\hat Q_n\hat v\big).\tag{16}$$

Substituting Eqs. (12) and (13) into the model and collecting the first-order terms gives, up to noise and second-order terms,

$$r_n=w_n\sum_{j=1}^J\alpha_{nj}H_j-\big(a^{(1)}+\hat P_nu^{(1)}+\hat Q_nv^{(1)}\big)-\hat uP_n^{(1)}-\hat vQ_n^{(1)},\tag{17}$$

in which the step-error signal enters with a plus sign and every absorbed bias with a minus: what the fit took out of the data is missing from the residual. Here $w_n=\hat P_n\hat v-\hat Q_n\hat u$ is the sensitivity $\kappa_{n,1}$ of Eq. (4) written in quadrature variables and evaluated at the AIA estimates. Built from the true parameters it would differ at first order, but $w_n$ multiplies only first-order quantities, so the substitution costs $O(\Delta_n^2)$.

### Removing the pixel corrections

At one pixel, collect the intensities of all frames in the column of Eq. (8), and let $A$ and $A_p$ be the matrices of Eq. (7) built from $\hat P_n,\hat Q_n$. Define the $N\times N$ matrix

$$\Pi=I-AA_p^{-1}A^\top,\tag{18}$$

where $I$ is the $N\times N$ identity matrix. In regression terms, $AA_p^{-1}A^\top$ is the hat matrix of the pixel-step fit and $\Pi$ is its residual-maker matrix (Hoaglin and Welsch, 1978). By Eq. (8), the pixel step fits the intensity column by $AA_p^{-1}A^\top$ applied to it, so the column splits into the fitted part and the leftover:

$$\begin{pmatrix}I_1\\\vdots\\I_N\end{pmatrix}
=AA_p^{-1}A^\top\begin{pmatrix}I_1\\\vdots\\I_N\end{pmatrix}
+\Pi\begin{pmatrix}I_1\\\vdots\\I_N\end{pmatrix},\qquad
\Pi\begin{pmatrix}I_1\\\vdots\\I_N\end{pmatrix}=\begin{pmatrix}r_1\\\vdots\\r_N\end{pmatrix}.$$

The leftover is the residual of Eq. (16). $\Pi$ depends only on $\hat P_n,\hat Q_n$ and is the same at every pixel. Since $A_p=A^\top A$,

$$\Pi A=0,\qquad A^\top\Pi=0,\qquad\Pi^2=\Pi,\qquad\Pi^\top=\Pi.$$

The first relation states that $\Pi$ removes every column of the form $A(a,u,v)^\top$. The second states that a leftover satisfies the pixel-step normal equations and cannot be fitted further. The third states that $\Pi$ leaves a leftover unchanged; in particular, $\Pi$ leaves the residual column unchanged. $\Pi$ has rank $N-3$, so each pixel retains $N-3$ independent combinations of its $N$ frame values; for $N=3$, $\Pi=0$.

**Leftover of the AIA fit.** Consider one pixel with exact $P_n,Q_n$, exact fields $a^{(0)},u^{(0)},v^{(0)}$, and intensities $I_n=a^{(0)}+P_nu^{(0)}+Q_nv^{(0)}+e_n$, where $e_n=w_n\Delta_n$ is the first-order phase-step signal, and suppose the frame coefficients carry no bias, $P_n^{(1)}=Q_n^{(1)}=0$, so that the pixel side carries all of it. By Eq. (8) and $\Pi A=0$, the pixel step returns

$$\begin{pmatrix}a^{(1)}\\u^{(1)}\\v^{(1)}\end{pmatrix}=A_p^{-1}A^\top\begin{pmatrix}e_1\\\vdots\\e_N\end{pmatrix},\qquad
\begin{pmatrix}r_1\\\vdots\\r_N\end{pmatrix}=\Pi\begin{pmatrix}e_1\\\vdots\\e_N\end{pmatrix}.\tag{19}$$

The part of $e$ lying in the column space of $A$ is what the fit can represent, and that part is the first-order bias; the leftover $\Pi e$ is what remains in the residual. Fitting the residual by $e_n$ with the pixel-side bias set to zero therefore underestimates $\Delta_n$. The residual must instead be compared with the leftover of the signal.

**Elimination.** At one pixel, the pixel-side bias in Eq. (17) forms the column $A\big(a^{(1)},u^{(1)},v^{(1)}\big)^\top$, which $\Pi$ removes, while $\Pi$ leaves the residual column unchanged. Applying $\Pi$ to Eq. (17) gives

$$\begin{pmatrix}r_1\\\vdots\\r_N\end{pmatrix}
=\Pi\begin{pmatrix}
w_1\sum_j\alpha_{1j}H_j-\hat uP_1^{(1)}-\hat vQ_1^{(1)}\\
\vdots\\
w_N\sum_j\alpha_{Nj}H_j-\hat uP_N^{(1)}-\hat vQ_N^{(1)}
\end{pmatrix}.\tag{20}$$

Equation (20) holds at every pixel and contains only the $2N+NJ$ unknowns $P_n^{(1)},Q_n^{(1)},\alpha_{nj}$, all shared across pixels. A single pixel provides $N-3$ independent equations, fewer than the $N-1$ zero-mean values $\Delta_n$ at that pixel; the shared spatial functions of Eq. (13) combine the equations of all pixels.

Eliminating a group of regressors by applying the residual-maker matrix to both the data and the remaining regressors does not change the least-squares estimates of the remaining unknowns (Frisch and Waugh, 1933; Lovell, 1963). A least-squares fit of Eq. (20) therefore gives the same $P_n^{(1)},Q_n^{(1)},\alpha_{nj}$ as a joint fit of Eq. (17) that carries the pixel-side bias along. The same elimination underlies the variable projection method for separable least-squares problems (Golub and Pereyra, 1973).

### Fit of the frame corrections and phase-step coefficients

Eq. (20) is fitted over all pixels by least squares. In this subsection, $u,v,P_n,Q_n$ denote the baseline estimates $\hat u,\hat v,\hat P_n,\hat Q_n$ with their hats suppressed. Write

$$z_n=w_n\sum_{j=1}^J\alpha_{nj}H_j-uP_n^{(1)}-vQ_n^{(1)}$$

for the entries of the column inside Eq. (20). The loss is

$$\mathcal L_1=\sum_{x,y}\sum_{n=1}^N\Big(r_n-\sum_{m=1}^N\Pi_{nm}z_m\Big)^2
=\sum_{x,y}\Big(\sum_nr_n^2-2\sum_nr_nz_n+\sum_{n,m}z_n\Pi_{nm}z_m\Big),\tag{21}$$

where the second form uses $\Pi^\top\Pi=\Pi$ and the fact that $\Pi$ leaves the residual column unchanged. Setting the derivatives with respect to $P_n^{(1)}$, $Q_n^{(1)}$, and $\alpha_{nj}$ to zero gives, for $n=1,\dots,N$ and $j=1,\dots,J$,

$$\begin{aligned}
\sum_m\Pi_{nm}\Big[P_m^{(1)}\sum_{x,y}u^2+Q_m^{(1)}\sum_{x,y}uv-\sum_{j'}\alpha_{mj'}\sum_{x,y}uw_mH_{j'}\Big]&=-\sum_{x,y}ur_n,\\
\sum_m\Pi_{nm}\Big[P_m^{(1)}\sum_{x,y}uv+Q_m^{(1)}\sum_{x,y}v^2-\sum_{j'}\alpha_{mj'}\sum_{x,y}vw_mH_{j'}\Big]&=-\sum_{x,y}vr_n,\\
\sum_m\Pi_{nm}\Big[-P_m^{(1)}\sum_{x,y}uw_nH_j-Q_m^{(1)}\sum_{x,y}vw_nH_j+\sum_{j'}\alpha_{mj'}\sum_{x,y}w_nw_mH_jH_{j'}\Big]&=\sum_{x,y}w_nH_jr_n.
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

The first six conditions state $A^\top(P_1^{(1)},\dots,P_N^{(1)})^\top=A^\top(Q_1^{(1)},\dots,Q_N^{(1)})^\top=0$. They select one representative; the gauge and quadrature-frame conventions are imposed when the pixel-side bias is recovered.

**Solution.** Collect the unknowns in the column

$$\beta=\big(P_1^{(1)},\dots,P_N^{(1)},\,Q_1^{(1)},\dots,Q_N^{(1)},\,\alpha_{11},\dots,\alpha_{1J},\dots,\alpha_{N1},\dots,\alpha_{NJ}\big)^\top$$

of length $2N+NJ$. Eq. (22) is $M\beta=\rho$, with

$$M=\begin{pmatrix}
\Pi\sum_{x,y}u^2&\Pi\sum_{x,y}uv&-D_u\\
\Pi\sum_{x,y}uv&\Pi\sum_{x,y}v^2&-D_v\\
-D_u^\top&-D_v^\top&E
\end{pmatrix},$$

$$\rho=\Big(-\sum_{x,y}ur_1,\dots,-\sum_{x,y}ur_N,\ -\sum_{x,y}vr_1,\dots,-\sum_{x,y}vr_N,\ \sum_{x,y}w_1H_1r_1,\dots,\sum_{x,y}w_1H_Jr_1,\dots,\sum_{x,y}w_NH_1r_N,\dots,\sum_{x,y}w_NH_Jr_N\Big)^\top.$$

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

$$\beta=\big(M+C^\top C\big)^{-1}\rho.\tag{25}$$

$M+C^\top C$ is invertible exactly when Eq. (20) has no undetermined directions besides those fixed by Eq. (24); this requires $N\ge5$ and $J\le J_{\max}$ (Eq. 15). The system has size $2N+NJ$ and does not grow with $K$.

### Pixel corrections and normalization

Write $\tilde\alpha_{nj}$ for the amplitudes returned by Eq. (25) and $\tilde\Delta_n$ for the fitted step field they define. With $P_n^{(1)},Q_n^{(1)},\tilde\alpha_{nj}$ known, Eq. (17) at one pixel is a pixel step for the pixel-side bias $a^{(1)},u^{(1)},v^{(1)}$ with the column $w_n\tilde\Delta_n-\hat uP_n^{(1)}-\hat vQ_n^{(1)}-r_n$ as data. By Eq. (8), its solution is $A_p^{-1}A^\top$ applied to this column. The residual column satisfies $A^\top(r_1,\dots,r_N)^\top=0$, and Eq. (24) gives $A^\top(P_1^{(1)},\dots,P_N^{(1)})^\top=A^\top(Q_1^{(1)},\dots,Q_N^{(1)})^\top=0$, so only the phase-step term remains:

$$\begin{pmatrix}a^{(1)}\\u^{(1)}\\v^{(1)}\end{pmatrix}
=A_p^{-1}\begin{pmatrix}\sum_nw_n\tilde\Delta_n\\\sum_n\hat P_nw_n\tilde\Delta_n\\\sum_n\hat Q_nw_n\tilde\Delta_n\end{pmatrix},\qquad
\tilde\Delta_n=\sum_{j=1}^J\tilde\alpha_{nj}H_j.\tag{26}$$

This reproduces the pixel-side bias of Eq. (19), now with the frame-side bias accounted for as well. By the Frisch–Waugh–Lovell theorem, Eqs. (25) and (26) together give the joint least-squares solution of Eq. (17). Subtracting the first-order bias gives the debiased fields

$$\tilde a=\hat a-a^{(1)},\qquad \tilde u=\hat u-u^{(1)},\qquad \tilde v=\hat v-v^{(1)},\qquad \tilde P_n=\hat P_n-P_n^{(1)},\qquad \tilde Q_n=\hat Q_n-Q_n^{(1)},\tag{27}$$

each equal to the corresponding $f^{(0)}$ up to $O(\Delta_n^2)$.

**Normalization.** The debiased fields of Eq. (27) satisfy Eq. (24) rather than the gauge and quadrature-frame conventions. They deviate from those conventions only at first order, so the transformations that restore them leave $\tilde\Delta_n$ unchanged and change the intensities only at second order. The zero-mean conventions on $\tilde\Delta_n$ already hold by Eqs. (13) and (24). Apply the following steps in order; each preserves the conditions established by the previous ones.

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

The final estimates are $\tilde\delta_n=\operatorname{atan2}\big(\tilde Q_n,\tilde P_n\big)$, $\tilde g_n=\sqrt{\tilde P_n^2+\tilde Q_n^2}$, $\tilde b=\sqrt{\tilde u^2+\tilde v^2}$, $\tilde\Phi=\operatorname{atan2}\big(-\tilde v,\tilde u\big)$, and the phase-step error $\tilde\Delta_n$ of Eq. (26). Their remaining errors are of second order in $\Delta_n$.

### VP-AIA algorithm and cost

1. **AIA baseline.** Run the pixel step, whitening, and frame step to convergence, finish with a pixel step, and apply normalization steps 1–4.
2. **Residual and projector.** Compute $r_n$ (Eq. 16), $w_n$ (Eq. 17), and $\Pi$ (Eq. 18).
3. **Pixel sums.** In one pass over the pixels, compute the sums of Eq. (23) and the right-hand sides of Eq. (22).
4. **Frame-side bias and coefficients.** Assemble $M$, $\rho$, and $C$, and solve Eq. (25).
5. **Pixel-side bias.** Evaluate $\tilde\Delta_n$ and Eq. (26) at every pixel, and form the debiased fields of Eq. (27).
6. **Normalization.** Apply normalization steps 1–4 to the debiased fields.

One AIA iteration costs $O(NK)$ operations. Steps 2–6 add $O(NKJ)$ operations for the residual, the right-hand sides, and Eq. (26); $O(KJ^2)$ operations for the sums of Eq. (23); and $O\big((2N+NJ)^3\big)$ operations for the solve of Eq. (25). The first-order pass therefore costs about as much as $J+J^2/N$ AIA iterations. Besides the stack and the fields already used by AIA, it stores the $J$ spatial functions and matrices of size $2N+NJ$. $\tilde\Delta_n$ need not be stored, since it is evaluated from $\tilde\alpha_{nj}$ and $H_j$.

**Larger phase-step errors.** One pass removes $f^{(1)}$ and leaves $\sum_{k\ge2}f^{(k)}$, so when $\Delta_n$ is large enough that the second-order term of the bias series is not negligible, that remainder must be attacked in turn. Iterate: remove the current phase-step signal from the measured stack with the full model, Eq. (1):

$$I_n'=I_n-\tilde g_n\tilde b\big[\cos(\tilde\Phi+\tilde\delta_n+\tilde\Delta_n)-\cos(\tilde\Phi+\tilde\delta_n)\big],\tag{28}$$

where $\tilde b,\tilde\Phi,\tilde\delta_n,\tilde g_n$ are the current estimates and $\tilde\Delta_n$ the accumulated step-error field. Up to the errors of the current estimates, $I_n'$ is a stack whose remaining step error is $\Delta_n-\tilde\Delta_n$. One round consists of three operations:

1. run AIA on $I_n'$, starting from the current $\tilde P_n,\tilde Q_n$;
2. apply steps 2–6 to its output; the amplitudes returned by Eq. (25), written $\delta\tilde\alpha_{nj}$, define the increment $\delta\tilde\Delta_n=\sum_j\delta\tilde\alpha_{nj}H_j$;
3. update $\tilde\Delta_n\leftarrow\tilde\Delta_n+\delta\tilde\Delta_n$, and replace $\tilde a,\tilde u,\tilde v,\tilde P_n,\tilde Q_n$ by the corrected, normalized parameters of step 6, which are re-estimated from $I_n'$ rather than accumulated.

Each increment has zero spatial and frame means, so the accumulated field keeps both conventions of Eq. (13).

The remaining step error enters $I_n'$ with the sensitivity $-g_nb\sin(\Phi+\delta_n+\tilde\Delta_n)$, whereas steps 2–6 use the $w_n$ of Eq. (17), which approximates the sensitivity at zero step error, $-g_nb\sin(\Phi+\delta_n)$. The two differ by $O(\Delta_n)$, so each increment carries a relative error of that order. The iteration is therefore a fixed-point scheme rather than a Newton step about the current solution: each round reduces the remaining error by a factor of order $\max_n|\Delta_n|$, and the scheme converges when that factor is well below one. Evaluating $w_n$ at $\tilde\Phi+\tilde\delta_n+\tilde\Delta_n$ instead would make the convergence quadratic, but $w_n$ would then no longer be a frame-weighted combination of $\hat u$ and $\hat v$, and the reductions of Eq. (23) would be lost. Stop when the increment becomes negligible. Only in this limit, not after one pass, is the solution unbiased in $\Delta_n$ to all orders.

## Noise of the corrected estimates

Consider noise $\varepsilon_n(x,y)$ with variance $\sigma_0^2$, independent across frames and pixels, and treat $P_n,Q_n$ as exact; the noise of their estimates is not included here. At one pixel, the pixel-step estimate of $\Phi=\operatorname{atan2}(-v,u)$ is linear in the intensity column. By Eq. (8), its error is $\sum_ns_n\varepsilon_n$ with

$$\begin{pmatrix}s_1\\\vdots\\s_N\end{pmatrix}=AA_p^{-1}\begin{pmatrix}0\\-\sin\Phi/b\\-\cos\Phi/b\end{pmatrix},$$

so plain AIA gives $\sigma_\Phi^2=\sigma_0^2\sum_ns_n^2$.

**Noise of the fitted unknowns.** The residual noise at each pixel is $\Pi(\varepsilon_1,\dots,\varepsilon_N)^\top$. Each entry of $\rho$ is a pixel sum of the residual weighted by one column of Eq. (20) after applying $\Pi$. Since $\Pi^\top\Pi=\Pi$, the noise of $\rho$ has covariance $\sigma_0^2M$, and by Eq. (25) the noise of $\beta$ has covariance

$$\sigma_0^2\big(M+C^\top C\big)^{-1}M\big(M+C^\top C\big)^{-1}.$$

**Effect on the phase.** By Eq. (26), noise $\delta\tilde\Delta_n$ in the fitted phase-step error changes $(\tilde a,\tilde u,\tilde v)$ by $-A_p^{-1}A^\top$ applied to the column $\big(w_1\delta\tilde\Delta_1,\dots,w_N\delta\tilde\Delta_N\big)^\top$, and therefore changes $\Phi$ by $-\sum_ns_nw_n\,\delta\tilde\Delta_n$. Since $(s_1,\dots,s_N)^\top$ lies in the column space of $A$, $\Pi$ removes it, and this change is uncorrelated with $\sum_ns_n\varepsilon_n$ at the same pixel. The correction therefore adds a variance term:

$$\sigma_\Phi^2=\sigma_0^2\sum_ns_n^2+\sigma_0^2\,\gamma^\top\big(M+C^\top C\big)^{-1}M\big(M+C^\top C\big)^{-1}\gamma,\tag{29}$$

where $\gamma$ is the column of length $2N+NJ$, ordered as $\beta$, with zeros at the positions of $P_n^{(1)},Q_n^{(1)}$ and the entry $s_nw_nH_j$ at the position of $\alpha_{nj}$, evaluated at the pixel.

**Uniform steps.** Assume uniform steps with $N\ge5$, constant $g_n$ and $b$, many fringes, and $\langle H_jH_{j'}\rangle_{x,y}=\delta_{jj'}$. Then the frame corrections decouple from the coefficients, the coefficient block of $M$ has entries $\tfrac12g^2b^2K\,\Pi_{nm}\cos(\delta_n-\delta_m)\,\delta_{jj'}$, and $s_nw_n=\big(1-\cos(2\Phi+2\delta_n)\big)/N$. The $N\times N$ matrix with entries $\Pi_{nm}\cos(\delta_n-\delta_m)$ has eigenvalue $0$ on frame-constant columns, $\tfrac12$ on the first and second temporal harmonics, and $1$ on the others. The frame-varying part of $s_nw_n$ lies in the second harmonic, so the eigenvalue $\tfrac12$ doubles its contribution. Averaging Eq. (29) over the fringe phase gives

$$\frac{\sigma_\Phi^2}{\sigma_0^2\sum_ns_n^2}\approx1+\frac{\sum_jH_j(x,y)^2}{K},\qquad
\bigg\langle\frac{\sigma_\Phi^2}{\sigma_0^2\sum_ns_n^2}\bigg\rangle_{x,y}\approx1+\frac{J}{K}.\tag{30}$$

The correction increases the phase noise by a relative amount of order $J/K$, largest where $\sum_jH_j^2$ is large and negligible for $K\gg J$. Equation (30) is a reading of Eq. (29) under the four conditions above, not a substitute for it: away from them — irregular steps, per-frame gains, a contrast that varies across the field, or modes the fringe pattern resolves unevenly — Eq. (29) is the form to use.

**Noise of the baseline steps.** The first term of Eq. (29) is `aia.md` Eq. (26), the phase variance at known $\delta_n$ and $g_n$. The baseline solve estimates both from the same frames, which adds the $O(1/K)$ term of `aia.md` Eqs. (40) and (45); the zeros of $\gamma$ at the positions of $P_n^{(1)},Q_n^{(1)}$ say only that the first-order frame corrections leave $\Phi$ unchanged to first order, not that the steps carry no noise. The two terms add, as in `sf_aia.md` Eq. (E9), and their cross-correlation is not derived here.

**Comparison with SF-AIA.** For the same $J$-mode field fitted from the same data, `sf_aia.md` Eq. (E8) gives $1+J/(4N_p)$ where Eq. (30) gives $1+J/K$, with $K=N_p$. The factor of four is the estimator, not a disagreement between the two derivations. Both reduce to a quadratic form in the $N\times N$ matrix with entries $\Pi_{nm}\cos(\delta_n-\delta_m)$, whose eigenvalue on the second temporal harmonic — where the frame-varying part of $s_nw_n$ lies — is $\tfrac12$. SF-AIA's per-frame fit multiplies that harmonic by the eigenvalue; VP-AIA's joint fit inverts it. The two therefore differ by $2/\tfrac12=4$ in variance, and it is the same factor as in `sf_aia.md` §"Bias of a single pass": one SF-AIA pass recovers the first and second harmonics of $c_{jn}$ at half their size, and attenuates their noise by that same half. Paying $J/K$ rather than $J/(4N_p)$ is therefore the price of the unbiased field, not an extra cost of the method: SF-AIA buys its smaller variance with the attenuation that is exactly its bias. Its refinement loop gives that attenuation back round by round, and its noise passes Eq. (30) rather than settling at it, since the alternation's fixed point is not the joint least-squares solution fitted here. VP-AIA reaches the unbiased field and Eq. (30) together, in one pass.

## References

- R. Frisch and F. V. Waugh, "Partial time regressions as compared with individual trends," *Econometrica* **1**(4), 387–401 (1933).
- M. C. Lovell, "Seasonal adjustment of economic time series and multiple regression analysis," *Journal of the American Statistical Association* **58**(304), 993–1010 (1963).
- G. H. Golub and V. Pereyra, "The differentiation of pseudo-inverses and nonlinear least squares problems whose variables separate," *SIAM Journal on Numerical Analysis* **10**(2), 413–432 (1973).
- D. C. Hoaglin and R. E. Welsch, "The hat matrix in regression and ANOVA," *The American Statistician* **32**(1), 17–22 (1978).
- Z. Wang and B. Han, "Advanced iterative algorithm for phase extraction of randomly phase-shifted interferograms," *Optics Letters* **29**(14), 1671–1673 (2004).
- Y. Chen and Q. Kemao, "General iterative algorithm for phase-extraction from fringe patterns with random phase-shifts, intensity harmonics and non-uniform phase-shift distribution," *Optics Express* **29**(19), 30905–30926 (2021).
