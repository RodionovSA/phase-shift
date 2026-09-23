# AIA phase-error covariance

This document derives the phase and amplitude errors of the AIA solution of `aia.md` from its normal equations, in three stages: $\delta_n$ and $g_n$ known; $\delta_n$ fitted with $g_n$ known; both fitted. Equations of `aia.md` are cited as `aia.md` Eq. (n). Throughout, $\mathbf i(x,y)\in\mathbb R^N$ is the per-pixel vector of measured intensities.

## Noise model

The camera adds zero-mean noise, independent between pixels and frames but not of equal variance:

$$I_n^{\text{meas}}=I_n+\varepsilon_n,\qquad
\operatorname E\big[\varepsilon_n(x,y)\big]=0,\qquad
\operatorname{Cov}\big(\varepsilon_n(x,y),\varepsilon_m(x',y')\big)=\sigma_n^2(x,y)\,\delta_{nm}\delta_{xx'}\delta_{yy'}.\tag{1}$$

Only these two moments are used. Independence excludes banded or common-mode readout noise, inter-pixel crosstalk, and speckle or source fluctuation correlated across the field.

## Stage 1: known steps and gains

With $\delta_n$ and $g_n$ at their true values, the pixel step at one pixel is

$$(a,u,v)^\top=A_p^{-1}A^\top\mathbf i,\qquad A_p=A^\top A,\tag{2}$$

with $A$ of `aia.md` Eq. (6). Its error $(e_a,e_u,e_v)^\top=A_p^{-1}A^\top\varepsilon$ has the covariance

$$\operatorname{Cov}\big((e_a,e_u,e_v)^\top\big)=A_p^{-1}A^\top D\,A\,A_p^{-1},\qquad D=\operatorname{diag}\big(\sigma_n^2(x,y)\big),\tag{3}$$

which reduces to

$$\operatorname{Cov}\big((e_a,e_u,e_v)^\top\big)=\sigma^2A_p^{-1}\tag{4}$$

when $D=\sigma^2I_N$, that is, when the variance is equal across the $N$ frames at that pixel; it may still vary between pixels. Write $S$ for the $\{u,v\}$ block of $A_p^{-1}$, so that $\operatorname{Cov}\big((e_u,e_v)^\top\big)=\sigma^2S$.

Linearizing `aia.md` Eq. (10) in $(e_u,e_v)$,

$$e_\Phi=-\frac{\sin\Phi\,e_u+\cos\Phi\,e_v}{b},\qquad
e_b=\cos\Phi\,e_u-\sin\Phi\,e_v,\tag{5}$$

so with the orthonormal pair $w=(\sin\Phi,\cos\Phi)^\top$ and $\tilde w=(\cos\Phi,-\sin\Phi)^\top$,

$$\sigma_\Phi^2(x,y)=\frac{\sigma^2}{b^2}\,w^\top Sw,\qquad
\sigma_b^2(x,y)=\sigma^2\,\tilde w^\top S\tilde w.\tag{6}$$

### Closed form

With the $g$-weighted circular moments of the phase steps,

$$R_c=\langle g_n\cos\delta_n\rangle_n,\quad R_s=\langle g_n\sin\delta_n\rangle_n,\quad
R_c^{(2)}=\langle g_n^2\cos2\delta_n\rangle_n,\quad R_s^{(2)}=\langle g_n^2\sin2\delta_n\rangle_n,\tag{7}$$

the columns of $A$ give

$$A_p=N\begin{pmatrix}
1&R_c&R_s\\
R_c&\tfrac12\big(\langle g^2\rangle_n+R_c^{(2)}\big)&\tfrac12R_s^{(2)}\\
R_s&\tfrac12R_s^{(2)}&\tfrac12\big(\langle g^2\rangle_n-R_c^{(2)}\big)
\end{pmatrix}.\tag{8}$$

Eliminating the $a$ row by Schur complement gives

$$S=\frac1N\,C^{-1},\qquad
C=\big\langle g_n^2\,d_nd_n^\top\big\rangle_n-\big\langle g_nd_n\big\rangle_n\big\langle g_nd_n\big\rangle_n^\top,\qquad
d_n=\begin{pmatrix}\cos\delta_n\\\sin\delta_n\end{pmatrix}.\tag{9}$$

$C$ is the covariance of the $g$-weighted step directions: fitting the background $a$ costs a mean-centering of those directions. Its trace, $\operatorname{tr}C=\langle g^2\rangle_n-|R|^2$ with $|R|=\sqrt{R_c^2+R_s^2}$, shrinks as the steps bunch together. Combining Eqs. (6) and (9),

$$\sigma_\Phi^2(x,y)=\frac{\sigma^2}{N\,b(x,y)^2}\,w^\top C^{-1}w,\qquad
\sigma_b^2(x,y)=\frac{\sigma^2}{N}\,\tilde w^\top C^{-1}\tilde w,\tag{10}$$

and, since $w$ and $\tilde w$ are orthonormal,

$$b(x,y)^2\sigma_\Phi^2(x,y)+\sigma_b^2(x,y)=\frac{\sigma^2}{N}\operatorname{tr}\big(C^{-1}\big):\tag{11}$$

the same error budget at every pixel, split between phase and amplitude by how $\Phi$ aligns with the principal axes of $C$.

### Bounds and limiting cases

With $\lambda_\pm$ the eigenvalues of $C$, Eq. (10) is bounded by

$$\frac{\sigma}{b\sqrt{N\lambda_+}}\;\le\;\sigma_\Phi\;\le\;\frac{\sigma}{b\sqrt{N\lambda_-}},\tag{12}$$

with equality when $\Phi$ aligns with an eigenvector. For anisotropic $C$, $\sigma_\Phi$ therefore varies with period $\pi$ in $\Phi$ between pixels of equal $b$. Averaged over the field, with $\Phi$ decorrelated from the local illumination, $w^\top C^{-1}w$ becomes $\tfrac12\operatorname{tr}(C^{-1})=\operatorname{tr}C/(2\det C)$, and

$$\big\langle\sigma_\Phi^2\big\rangle_{x,y}=\frac{1}{2N}\Big\langle\frac{\sigma^2(x,y)}{b(x,y)^2}\Big\rangle_{x,y}\,\frac{\operatorname{tr}C}{\det C}.\tag{13}$$

The ratio $\sigma^2/b^2$ must be averaged as one quantity: $\langle\sigma^2\rangle\langle1/b^2\rangle$ overstates the error when $\sigma$ and $b$ are positively correlated, as shot noise makes them.

Both conditions of Eq. (13) are sharper than "many fringes". The first holds once the second harmonic of $\Phi$ averages out, so for $\Phi$ spanning any interval of length $\pi$; it is $25\%$ off over $\pi/2$ and a factor $2$ off for a nearly flat field. The second fails when the contrast is locked to the phase, whatever the spread of $\Phi$: for $b=1+0.6\cos\Phi$, Eq. (13) reads $16\%$ low.

**Ideal case.** For evenly spaced steps $\delta_n=2\pi n/N$, $N\ge3$, and unit gain, $R=R^{(2)}=0$, so $C=\tfrac12I_2$ and

$$\sigma_\Phi=\sqrt{\frac2N}\,\frac{\sigma}{b},\tag{14}$$

independent of $\Phi$. This configuration has $\kappa_p=2$, and `aia.md` Eq. (11) evaluates to $1.014\,\sigma_\Phi$ for every $N$.

**Bound in terms of $\kappa_p$.** By Cauchy interlacing, $\lambda_{\max}(S)\le1/\lambda_{\min}(A_p)=\kappa_p/\lambda_{\max}(A_p)$, and $\lambda_{\max}(A_p)\ge\operatorname{tr}(A_p)/3=N\big(1+\langle g^2\rangle_n\big)/3$ by Eq. (8), so

$$\sigma_\Phi\;\le\;\frac{\sigma}{b}\sqrt{\frac{3\kappa_p}{N\big(1+\langle g^2\rangle_n\big)}},\tag{15}$$

which at $g_n\equiv1$ is $1.22\sqrt{\kappa_p}\,\sigma/(b\sqrt N)$.

### Shot noise

Shot noise follows each frame's fringe-modulated illumination, so $\sigma_n^2(x,y)$ differs between frames at a fixed pixel and Eq. (4) holds only approximately. The stand-in is the quadrature mean over frames,

$$\sigma_0(x,y)^2=\frac1N\sum_{n=1}^N\sigma_n^2(x,y),\tag{16}$$

used for $\sigma$ in Eq. (10), which gives a per-pixel noise map paired with $b(x,y)$. Its accuracy depends on the spread of $\sigma_n^2$ about $\sigma_0^2$, which calibrated per-frame noise maps give directly.

## Stage 2: fitted steps, known gains

In practice the pixel step uses the fitted $\hat\delta_n=\delta_n+e_{\delta_n}$, so its design matrix is perturbed. Differentiating the normal equations at the true, noiseless data, where the zero model residual cancels the leading term,

$$\frac{\partial X}{\partial\delta_n}=-A_p^{-1}a_n\,\frac{\partial I_n}{\partial\delta_n},\qquad
\frac{\partial I_n}{\partial\delta_n}=-g_n\,b(x,y)\sin\big(\Phi(x,y)+\delta_n\big),\tag{17}$$

with $X=(a,u,v)^\top$ and $a_n$ the $n$-th row of $A$: an error in $\delta_n$ displaces the solution as extra noise on frame $n$ would. The $(u,v)$ part of that displacement is the leverage vector

$$k_n=\big(A_p^{-1}a_n\big)_{\{u,v\}}=\frac1N\,C^{-1}\big(g_nd_n-R\big),\qquad R=\big\langle g_nd_n\big\rangle_n.\tag{18}$$

**Size of the frame-step error.** The frame step is the pixel step transposed: a regression over $N_p$ pixels with the design $B$ of `aia.md` Eq. (8). With $(u,v)$ at their noise-free values, the algebra of Eqs. (2)–(3) gives, per frame,

$$\operatorname{Cov}\big((e_{c_n},e_{P_n},e_{Q_n})^\top\big)=A_{ps}^{-1}\big(B^\top DB\big)A_{ps}^{-1},\qquad
A_{ps}=B^\top B,\qquad D=\operatorname{diag}\big(\sigma_0^2(x,y)\big).\tag{19}$$

Linearizing $\delta_n=\operatorname{atan2}(Q_n,P_n)$ as in Eq. (5), with $w_n=(-\sin\delta_n,\cos\delta_n)^\top$,

$$\sigma_{\delta_n}^2=\frac{1}{g_n^2}\,w_n^\top\Big[A_{ps}^{-1}\big(B^\top DB\big)A_{ps}^{-1}\Big]_{\{P,Q\}}w_n.\tag{20}$$

With many fringes in no preferred orientation, $\sum_{x,y}u^2\approx\sum_{x,y}v^2\approx N_p\langle b^2\rangle/2$, $\sum_{x,y}uv\approx0$, and $\Phi$ decorrelated from the illumination, Eq. (20) becomes isotropic:

$$\sigma_{\delta_n}^2\approx\frac{2\sigma_{\text{eff}}^2}{N_p\langle b^2\rangle\,g_n^2},\qquad
\sigma_{\text{eff}}^2=\frac{\big\langle\sigma_0^2b^2\big\rangle}{\big\langle b^2\big\rangle}.\tag{21}$$

Bright, high-contrast pixels dominate the frame step, so their noise sets the uncertainty of $\delta_n$.

Whitening, `aia.md` Eq. (15), imposes the first two conditions exactly. The condition that fails first is the unstated one, $\sum_{x,y}u\approx\sum_{x,y}v\approx0$, which makes $A_{ps}$ diagonal. It holds once the field spans a full fringe and fails fast below: Eq. (21) is exact at one fringe, $3.3$ times off at half a fringe, and $22$ times off at a quarter, where $\kappa_{ps}$ is already large. Eq. (20) is exact throughout and costs one $3\times3$ inverse.

**Effect on the phase.** $\hat\delta_n$ is fitted from the same frames the pixel step uses, so $e_{\delta_n}$ is correlated with the noise at every pixel. Both errors are linear in the same noise, and writing the frame-step solution out resolves the correlation. By Eqs. (2)–(3) applied to $B$, it is a pixel-weighted sum of the frame's own noise,

$$\binom{e_{P_n}}{e_{Q_n}}=\sum_{x,y}\ell(x,y)\,\varepsilon_n(x,y),\qquad
\ell(x,y)=\Big[A_{ps}^{-1}\big(1,u(x,y),v(x,y)\big)^{\!\top}\Big]_{\{P,Q\}},\tag{22}$$

with $M=\sum_{x,y}\ell\,\ell^\top\sigma^2(x,y)$ the $\{P,Q\}$ block of Eq. (19), its noise read as in Eq. (16). Eq. (22) is Eq. (18) transposed: $k_n$ weights the $N$ frames at one pixel, $\ell$ the $N_p$ pixels of one frame. It contains frame $n$'s noise alone, so $e_{\delta_n}$ and $e_{\delta_m}$ are uncorrelated for $n\ne m$.

At one pixel, the pixel step's own noise term is $\sum_nk_n\varepsilon_n(x,y)$, and the design error adds Eq. (17) once per frame. With $q=(u,v)^\top$, the sensitivity is $\partial I_n/\partial\delta_n=g_n\,w_n^\top q$ and $e_{\delta_n}=w_n^\top(e_{P_n},e_{Q_n})^\top/g_n$, so the factors of $g_n$ cancel and

$$\binom{e_u}{e_v}=\sum_{n=1}^Nk_n\,\eta_n,\qquad
\eta_n=\varepsilon_n(x,y)-q^\top\Pi_n\binom{e_{P_n}}{e_{Q_n}},\qquad
\Pi_n=w_nw_n^\top.\tag{23}$$

$\eta_n$ is the effective noise of frame $n$ at that pixel: its own, less the part the frame step spent on $\delta_n$. The projector $\Pi_n$ keeps the component of the frame-step error that moves $\delta_n$ and discards the one that moves the fixed $g_n$; Stage 3 replaces it by $I_2$. Since the $\eta_n$ are uncorrelated across frames, and $\varepsilon_n(x,y)$ enters $(e_{P_n},e_{Q_n})$ with the weight $\ell(x,y)$,

$$\sigma_\Phi^2(x,y)=\frac{1}{b^2}\sum_{n=1}^N\big(w^\top k_n\big)^2\,\Gamma_n,\qquad
\Gamma_n=\sigma^2\Big[1-2\,q^\top\Pi_n\ell\Big]+q^\top\Pi_nM\Pi_nq,\tag{24}$$

every quantity read at $(x,y)$. The bracket is the correlation between the pixel's noise and the step fitted from it; the last term is the variance the fitted step injects. $\Pi_n=0$ returns Stage 1, since $\sum_nk_nk_n^\top=S$. Beyond $\ell$ and $q$, nothing in Eq. (24) is pixel-sized: $A_{ps}$, $M$, and $C$ are computed once.

**Ideal case.** For evenly spaced steps, unit gain, and $\Phi$ spread evenly over $2\pi$, Eq. (22) gives $\ell=2q/N_p$ and $M=(2\sigma^2/N_p)I_2$, and Eq. (24) becomes, at every pixel,

$$\sigma_\Phi^2\big|_{\text{ideal}}=\sigma_\Phi^2\big|_{\text{Eq. (14)}}\Big(1-\frac{3}{2N_p}\Big),\tag{25}$$

for $N\ge3$; only at $N=4$ does the coefficient vary across the field, as $-\tfrac32-\tfrac12\cos4\Phi$, with field average $-\tfrac32$. The variance term contributes $+3/(2N_p)$ and the correlation $-3/N_p$; Stage 3 splits the same way, $+2/N_p$ against $-4/N_p$. The factor of two holds in this configuration, where $M\Pi_nq=\sigma^2\ell$; in general the two terms of $\Gamma_n$ are unrelated and their sum has either sign.

Away from it, Eq. (24) is not a fixed multiple of Stage 1, and the ratio varies from pixel to pixel on either side of $1$, but fitting $\delta_n$ always changes the per-pixel phase variance by $O(1/N_p)$, a few parts in $10^3$ for a megapixel field. Frame-step errors are, however, one number per frame, so they do not average away across the field, and they matter when a reconstruction is differenced against another acquisition with correlated $\delta_n$ error. The effective $N_p$ is smaller than the pixel count when a small region is used or the noise is spatially correlated.

## Stage 3: fitted steps and gains

Perturbing $g_n$ instead of $\delta_n$ gives the same leverage vector with a different sensitivity:

$$\frac{\partial X}{\partial g_n}=-A_p^{-1}a_n\,\frac{\partial I_n}{\partial g_n},\qquad
\frac{\partial I_n}{\partial g_n}=b(x,y)\cos\big(\Phi(x,y)+\delta_n\big).\tag{26}$$

**Size of the gain error.** The gain is the radius of $(P_n,Q_n)$ and $\delta_n$ its angle, so the gain error is the radial projection of the covariance that Eq. (20) projects tangentially. With $d_n$ of Eq. (9) orthogonal to $w_n$, and $M$ the $\{P,Q\}$ block of Eq. (19),

$$\sigma_{g_n}^2=d_n^\top Md_n,\qquad
\operatorname{Cov}\big(e_{\delta_n},e_{g_n}\big)=\frac1{g_n}w_n^\top Md_n,\qquad
g_n^2\sigma_{\delta_n}^2+\sigma_{g_n}^2=\operatorname{tr}M,\tag{27}$$

the last being the frame-side counterpart of Eq. (11): step and gain share one error budget, and are uncorrelated when $d_n$ is a principal axis of $M$. Under the approximation of Eq. (21), $M$ is isotropic, the covariance vanishes, and

$$\sigma_{g_n}^2\approx\frac{2\sigma_{\text{eff}}^2}{N_p\langle b^2\rangle},\tag{28}$$

without the $1/g_n^2$ of Eq. (21): a radius does not gain precision as it grows, as an angle does at fixed arc-length error.

**Effect on the phase.** Both errors now perturb the design, and together they are all the frame step solves for, so Eq. (23) holds with $\Pi_n=I_2$, the cross term of Eq. (27) included. $\Gamma_n$ in Eq. (24) then no longer depends on $n$, and $\sum_nk_nk_n^\top=S$ collapses the sum to one factor per pixel:

$$\sigma_\Phi^2(x,y)=\sigma_\Phi^2\big|_{\text{Eq. (10)}}\left[1-2\,q^\top\ell+\frac{q^\top Mq}{\sigma^2}\right].\tag{29}$$

**Ideal case.** With $\ell=2q/N_p$, $M=(2\sigma^2/N_p)I_2$, and $q^\top q=b^2$, the bracket is $1-4/N_p+2/N_p$, so at every pixel and for every $N\ge3$,

$$\sigma_\Phi^2\big|_{\text{ideal}}=\sigma_\Phi^2\big|_{\text{Eq. (14)}}\Big(1-\frac{2}{N_p}\Big).\tag{30}$$

## Validity

**Stages 2 and 3.** Eqs. (24) and (29) are first order in the noise and read the frame step with $(u,v)$ at their noise-free values, Eq. (19). They describe one pass of the alternation from the true fields, not its converged fixed point, and give variances about each estimator's own mean. Three effects lie outside them.

The first is a bias. The converged frame step regresses on quadratures estimated from the same frames, and a regression on an estimated regressor is biased. By Eq. (4) the regressor error is $\sigma^2S$ at every pixel, so the bias is $O(\sigma^2/b^2)$ and does not shrink as the field grows. Since $\sigma_{\delta_n}$ falls as $N_p^{-1/2}$ by Eq. (21), the bias dominates the step error beyond $N_p$ of order $b^2/\sigma^2$. This affects $\delta_n$ only: $\sigma_\Phi$ has no $N_p$ in it, so the bias stays a small fixed fraction of the phase noise, and Eq. (10) remains the predictor of $\sigma_\Phi$ at any field size. The bias vanishes for a noise-free solve and for a frame step given the exact $(u,v)$, so it belongs to the alternation, not the model. It is not derived here.

The second is the frame-independent displacement $(a_u,a_v)$ of `aia.md` §"Dropping the background field", which biases $\hat\delta_n$ whenever the background overlaps the fringe pattern. It is a gauge shift when $g_n$ is fitted, but not when $g_n$ is fixed, so in Stage 2 an RMS deviation from the true step exceeds Eq. (20). It is deterministic and survives at zero noise.

The third is the phase-origin convention $\delta_n\leftarrow\delta_n-\delta_1$, which adds $-e_{\delta_1}$ to every step and shifts $\Phi$ by one constant across the field; Eqs. (24) and (29) give the variance of $\Phi$ up to that constant, which is what a relative phase map uses.

**Linearization and noise model.** Eq. (5) linearizes $\operatorname{atan2}$ and the square root, which holds while $b/\sigma\gg1$, roughly $\sigma_\Phi\lesssim0.3$ rad; below that the exact phasor distribution takes over, and Eq. (10) understates the probability of a $2\pi$ phase slip. Shot noise does not break the independence of Eq. (1); correlated readout, crosstalk, and correlated speckle or source fluctuation do. Dividing out $\alpha_n$ rescales frame $n$'s variance by $1/\alpha_n^2$, so Eq. (4) does not hold exactly; since $\alpha_n$ is estimated, replace $A_p$ by $A^\top WA$, $W=\operatorname{diag}(\alpha_n^2)$, throughout.

## Interpreting $\sigma$ and $b$

The pixel-step residual has $N-3$ degrees of freedom per pixel, so dividing the summed squared residual by $N\,N_p$ instead of $(N-3)N_p$ biases $\sigma$ low by $\sqrt{(N-3)/N}$: $24\%$ at $N=7$, $37\%$ at $N=5$. And scaling by $\operatorname{median}(b)$ is not the field average of Eq. (13), which keeps $\sigma^2(x,y)/b(x,y)^2$ together: low-contrast, low-illumination pixels weigh more in the true RMS than two separate scalars suggest.

## References

- Y. Chen and Q. Kemao, "Advanced iterative algorithm for phase extraction: performance evaluation and enhancement," *Optics Express* **27**(26), 37634–37651 (2019).
