# Carrier removal

Carrier removal fits a smooth, low-order phase field to a recovered phase map and subtracts it. The fit is posed on $e^{i\Phi}$ rather than on $\Phi$, which makes it exact under $2\pi$ wraps and needs no unwrapping.

## Model

A solve of `interference_model.md` Eq. (17), by AIA (`aia.md`) or VP-AIA (`vp_aia.md`), returns the total static phase of `interference_model.md` Eq. (15),

$$\Phi(x,y)=\phi(x,y)+\phi_{\text{inst}}(x,y)+\phi_{\text{carrier}}(x,y).$$

The data cannot separate the three terms. $\phi_{\text{carrier}}$ is low-order by construction and $\phi_{\text{inst}}$ usually is too, so carrier removal fits a low-order model of $\Phi$ itself and treats everything it captures as instrumental.

The intensities depend on $\Phi$ only through a cosine, so at each pixel $\Phi$ is known only up to an integer multiple of $2\pi$: the data determine $e^{i\Phi}$, not $\Phi$. A carrier estimate must be invariant under $\Phi\to\Phi+2\pi m(x,y)$ for any integer field $m$, and a criterion built from $e^{i\Phi}$ is invariant without unwrapping.

## Basis

Let $p_1,\dots,p_L$ be the monomials in $x,y$ of total degree $1$ to $M$, on coordinates centred at the field centroid and scaled to about $[-1,1]$, each made zero-mean over the field and orthonormalized in ascending degree, so $L=\tfrac{(M+1)(M+2)}{2}-1$. Add the constant $p_0=1$, which is orthogonal to the others since they have zero mean. The carrier field is

$$\Psi(x,y)=\sum_{l=0}^{L}\eta_l\,p_l(x,y),\tag{1}$$

with piston $\eta_0$ and low-order shape $\eta_1,\dots,\eta_L$.

## Objective

With a reliability weight $\omega(x,y)\ge0$, define the weighted complex field and the residual phase

$$\zeta=\omega\,e^{i\Phi},\qquad\chi=\Phi-\Psi.\tag{2}$$

The natural weight is the fringe amplitude, $\omega=b$. With the quadrature fields of `aia.md` Eq. (2), $u-iv=b\,e^{i\Phi}$, so then $\zeta=u-iv$: no division by $b$ is needed, and pixels without fringe contrast drop out.

Carrier removal maximizes

$$F(\boldsymbol\eta)=\sum_{x,y}\omega\cos\chi=\operatorname{Re}\sum_{x,y}\zeta\,e^{-i\Psi}.\tag{3}$$

Since $\sum_{x,y}\omega(1-\cos\chi)=\tfrac12\sum_{x,y}\omega\,|e^{i\Phi}-e^{i\Psi}|^2$, this is weighted least squares between points on the unit circle; for small $\chi$ it reduces to weighted least squares on $\chi$ (Appendix A).

The piston can be profiled out. With $\Psi'=\sum_{l\ge1}\eta_lp_l$,

$$\max_{\eta_0}F=\Big|\sum_{x,y}\omega\,e^{i(\Phi-\Psi')}\Big|,\qquad\eta_0=\arg\sum_{x,y}\omega\,e^{i(\Phi-\Psi')}.\tag{4}$$

## Newton iteration

Setting the gradient of Eq. (3) to zero gives

$$h_l\equiv\sum_{x,y}\omega\,p_l\sin\chi=0,\qquad l=0,\dots,L.\tag{5}$$

The sine of the residual, not the residual, is orthogonal to every basis function. The $l=0$ row at a maximum gives $\arg\sum_{x,y}\omega\,e^{i\chi}=0$: the weighted circular mean of the corrected phase is zero, which fixes the piston.

With the matrix

$$S_{lm}=\sum_{x,y}\omega\cos\chi\,p_lp_m,\tag{6}$$

minus the Hessian of $F$, each Newton step updates

$$\boldsymbol\eta\leftarrow\boldsymbol\eta+S^{-1}\mathbf h.\tag{7}$$

$S$ is positive definite only where $\omega\cos\chi>0$ dominates, that is, near the maximum; there it tends to the weighted Gram matrix $\sum_{x,y}\omega\,p_lp_m$, and $\operatorname{cond}(S)$ is the diagnostic of the fit. The corrected phase is $\arg e^{i(\Phi-\Psi)}$.

## Initialization

$F$ has about one local maximum per fringe along each basis direction, and Eq. (7) converges to the nearest one, so every coefficient, curvature included, must start within about half a fringe of the truth. A linear fit of the phase differences between neighbouring pixels gives such a start.

Let $\mathcal W(x,y)$ be the $K\times K$ window of pixels centred on $(x,y)$, truncated to the field. Summing the products of horizontal and vertical neighbours over the window,

$$\begin{aligned}
c_x(x,y)&=\sum_{(x',y')\in\mathcal W(x,y)}\zeta(x'+1,y')\,\overline{\zeta(x',y')},\\
c_y(x,y)&=\sum_{(x',y')\in\mathcal W(x,y)}\zeta(x',y'+1)\,\overline{\zeta(x',y')},
\end{aligned}\tag{8}$$

gives the phase differences and their weights

$$d_x=\arg c_x,\quad\omega_x=|c_x|,\qquad d_y=\arg c_y,\quad\omega_y=|c_y|.\tag{9}$$

Each product in Eq. (8) is $\omega(x'+1,y')\,\omega(x',y')\,e^{i[\Phi(x'+1,y')-\Phi(x',y')]}$, so $d_x$ needs no unwrapping and approximates the weighted mean of $\Phi(x'+1,y')-\Phi(x',y')$ over the window wherever these differences are smaller than $\pi$ in magnitude and vary across the window by much less than one radian; likewise for $d_y$. For $K=1$ it is the single difference $\Phi(x+1,y)-\Phi(x,y)$. With the basis differences $p^x_l(x,y)=p_l(x+1,y)-p_l(x,y)$ and $p^y_l(x,y)=p_l(x,y+1)-p_l(x,y)$, the piston drops out, and the weighted least-squares fit of $d_x,d_y$ by $\sum_{l\ge1}\eta_lp^x_l$, $\sum_{l\ge1}\eta_lp^y_l$ has the normal equations

$$\sum_{m=1}^{L}\Big[\sum_{x,y}\big(\omega_xp^x_lp^x_m+\omega_yp^y_lp^y_m\big)\Big]\eta_m=\sum_{x,y}\big(\omega_xp^x_ld_x+\omega_yp^y_ld_y\big),\qquad l=1,\dots,L,\tag{10}$$

each sum taken over the pixels where the neighbour exists. Eq. (4) then gives $\eta_0$.

The window sum in Eq. (8) is what makes the start reliable under noise. The argument of a single noisy product wraps whenever the noise carries it past $\pm\pi$, and a wrapped value lands on the far side of zero. The differences of Eq. (9) with $K=1$ are therefore biased towards zero gradient, and so is the fit of Eq. (10). The bias is a small fraction of the gradient, but $\Psi$ accumulates the gradient across the whole field, so the error at the field edge grows with the field size and can exceed the half fringe the start allows. Summing the complex products first averages the noise on the circle, where nothing wraps, and the argument is taken once, of a sum whose relative noise is about $K$ times smaller. For the same reason a sample-phase edge inside a window lowers $\omega_x$ or $\omega_y$ instead of producing a wrapped difference. The price is resolution: $d_x,d_y$ describe the gradient averaged over the window, so $K$ must be small enough that the gradient of $\Phi$ varies little across one window.

Differencing amplifies noise and weights the pixels less well than Eq. (3), so Eq. (10) serves only as the start; a few steps of Eq. (7) reach the maximum of $F$.

**Subsampled start.** The start needs only half-fringe accuracy, so it can be computed on every $s$-th pixel in each direction, with $\zeta$ and $p_0,\dots,p_L$ sampled there, not rebuilt. Eqs. (8)–(10) and (4) then apply to that grid, whose neighbours are $s$ pixels apart, so the phase must change by less than $\pi$ over $s$ pixels. Steps of Eq. (7) on the same grid reach the maximum of $F$ restricted to it, which differs from the full-grid maximum only by noise and so lies well inside its half-fringe basin; one or two steps of Eq. (7) on the full grid finish the fit. The costly full-grid passes are thereby reduced to those final steps.

## Identifiability

- **Piston.** Fixed by Eq. (4), equivalently the $l=0$ row of Eq. (5).
- **Aliasing.** On the pixel grid, $\eta$ and $\eta'$ give the same data when $\Psi-\Psi'$ is a multiple of $2\pi$ at every pixel; a tilt, for example, is defined only modulo one cycle per pixel. Among carrier fields with
  $$\max_{x,y}|\nabla\Psi|<\pi\ \text{rad/pixel},\tag{11}$$
  the fit is unique up to the piston (Appendix A).
- **Sample phase.** Any part of $\phi$ in the span of $p_1,\dots,p_L$ is removed with the carrier. $M$ is a modelling choice, not a fit-quality setting.
- **Sign.** Under $(\Phi,\delta_n)\to(-\Phi,-\delta_n)$ (`aia.md`, "Identifiability and gauge"), every $\eta_l$ changes sign. Carrier removal does not resolve this branch; maps to be compared must be brought to the same branch first.

## Relation to AIA and VP-AIA

By the frame-mean convention of `vp_aia.md`, a phase-step pattern shared by all frames belongs to $\Phi$, so it is part of what carrier removal fits. When the step-error modes $H_j$ lie in the span of $p_1,\dots,p_L$, that shared pattern is removed with the carrier.

AIA applied to data with $\Delta_n\not\equiv0$ biases $\Phi$ (`aia.md`, "Starting point"). Carrier removal takes out only the part of that bias in the span of the basis, and then the fitted $\boldsymbol\eta$ no longer describes the instrument alone. VP-AIA removes the bias to first order before carrier removal.

## Appendix A. Derivations

**Chord distance.** From $|e^{i\theta}-1|^2=2-2\cos\theta$,

$$\sum_{x,y}\omega(1-\cos\chi)=\tfrac12\sum_{x,y}\omega\,|e^{i\Phi}-e^{i\Psi}|^2=\sum_{x,y}\omega-F(\boldsymbol\eta),\tag{A1}$$

and $\sum_{x,y}\omega$ does not depend on $\boldsymbol\eta$. For small $\chi$, $1-\cos\chi\approx\chi^2/2$.

**Piston profile.** With $p_0=1$, $F=\operatorname{Re}\big(e^{-i\eta_0}R\big)$ for $R=\sum_{x,y}\omega\,e^{i(\Phi-\Psi')}$, maximized at $\eta_0=\arg R$ with value $|R|$, which is Eq. (4).

**Gradient and Hessian.** Since $\partial\chi/\partial\eta_l=-p_l$,

$$\frac{\partial F}{\partial\eta_l}=\sum_{x,y}\omega\,p_l\sin\chi,\qquad
\frac{\partial^2F}{\partial\eta_l\,\partial\eta_m}=-\sum_{x,y}\omega\cos\chi\,p_lp_m,\tag{A2}$$

which give Eqs. (5)–(7). Replacing $\sin\chi\to\chi$ and $\cos\chi\to1$, with $\chi$ wrapped to $(-\pi,\pi]$, turns Eq. (7) into the weighted least-squares fit of the wrapped residual onto the basis: the familiar wrap-and-refit loop is the linearization of Eq. (7). Least squares on the wrapped residual itself is not a substitute for Eq. (3), because the wrapped residual jumps whenever a pixel crosses $\pm\pi$, whereas $F$ is smooth in $\boldsymbol\eta$.

**Uniqueness under Eq. (11).** Let $\Psi$ and $\Psi'$ both satisfy Eq. (11) and agree modulo $2\pi$ at every pixel, and let $D=\Psi-\Psi'$. If $D$ is not the same multiple of $2\pi$ at all pixels, it differs by at least $2\pi$ between two neighbouring pixels, so somewhere on the segment between them $|\nabla D|\ge2\pi$. Then $|\nabla\Psi|+|\nabla\Psi'|\ge2\pi$ there, contradicting Eq. (11). Hence $D$ is a constant multiple of $2\pi$, a piston.
