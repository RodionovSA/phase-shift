# SF-AIA: perturbative recovery without a polynomial basis

Discussion note; not approved package theory or an implementation specification.
The established theory documents remain unchanged.

This note derives a possible first-order estimator from the
[interference model](../docs/interference_model.md), the
[AIA pixel solve](../docs/aia.md), and the
[SF-AIA perturbation expansion](../docs/sf_aia.md), especially §§2, 4, and 8.
The projected residual estimator below is a proposed extension of those equations.

## 1. Two different approximations

Expanding in the magnitude of the phase error does not require expanding its
spatial shape in polynomials. SF-AIA §2 already gives a pointwise perturbation
expansion valid for an arbitrary spatial shape.

However, removing the polynomial basis removes a constraint used to estimate the
unknown field. Perturbation theory provides a forward sensitivity; it does not
by itself supply the missing information needed for inversion.

Write the unknown as $\Delta_n(x,y)$: each frame has its own field. A field that
is identical in every frame is indistinguishable from an addition to the static
phase $\Phi(x,y)$.

## 2. Pointwise first-order expansion

Let $I_n$ denote the intensity after source-power normalization and subtraction
of any fitted additive frame offset. Initially hold the frame gains $g_n$ and
piston steps $\delta_n$ fixed. The model is

$$
I_n=a+g_n b\cos(\Phi+\delta_n+\Delta_n).
$$

Define

$$
I_n^{(0)}=a+g_n b\cos(\Phi+\delta_n),
\qquad
w_n=g_n b\sin(\Phi+\delta_n).
$$

The expansion is

$$
I_n=I_n^{(0)}-w_n\Delta_n
-\frac12 g_n b\cos(\Phi+\delta_n)\Delta_n^2
+O(\Delta_n^3).
$$

If the baseline fields were independently known, the first-order estimate would be

$$
\widehat\Delta_n\simeq-\frac{I_n-I_n^{(0)}}{w_n}.
$$

This division is unstable near $w_n=0$. At fringe maxima and minima, first-order
phase sensitivity vanishes. The quadratic term there does not resolve the sign
of a small perturbation on its own.

More fundamentally, an AIA baseline fitted to the same stack has already absorbed
part of the perturbation into its recovered background, amplitude, and phase.
Its residual is not the difference from the independently known baseline above.

## 3. What remains in a fitted AIA residual

At one pixel, collect the $N$ frame intensities into a vector. Use the quadrature
convention from the main docs,

$$
\beta=\begin{pmatrix}a\\u\\v\end{pmatrix},
\qquad u=b\cos\Phi,\qquad v=-b\sin\Phi,
$$

and define

$$
A_{n,:}=[1,\ g_n\cos\delta_n,\ g_n\sin\delta_n],
\qquad D=\operatorname{diag}(w_1,\ldots,w_N).
$$

With $\Delta=(\Delta_1,\ldots,\Delta_N)^\top$, the first-order data model is

$$
I=A\beta-D\Delta+\varepsilon+O(\|\Delta\|^2).
$$

For the ordinary least-squares pixel solve, define

$$
\widehat\beta=A^+I,\qquad H=AA^+,\qquad R=\mathrm{Id}_N-H.
$$

Here $A^+$ is the pseudoinverse. The fitted residual satisfies

$$
\boxed{r=I-A\widehat\beta=RI
\simeq-RD\Delta+R\varepsilon.}
$$

The component $HD\Delta$ is absorbed into the fitted fields; only $RD\Delta$
remains in the residual. In particular,

$$
\widehat\beta-\beta\simeq-A^+D\Delta+A^+\varepsilon.
$$

These identities assume the pixel coefficients are the least-squares solution
for the fixed $A$. If subsequent normalization or gauge operations alter that
representation, the residual and design matrix must be made consistent first.

### Identifiability

If $A$ has rank three, then

$$
\operatorname{rank}(RD)\leq\operatorname{rank}(R)=N-3.
$$

There are $N$ unknown phase-error values at this pixel, but at most $N-3$
independent residual constraints. If every $w_n$ is nonzero, an explicit
unobservable family is

$$
\Delta_{\mathrm{null}}=D^{-1}Az,\qquad z\in\mathbb R^3,
\qquad RD\Delta_{\mathrm{null}}=0.
$$

A frame-mean-zero constraint removes at most one degree of freedom per pixel.
Spatial mean constraints couple pixels but do not generally identify arbitrary
fields across an entire image. More frames alone do not fix the problem: an
unrestricted model also introduces a new unknown field with every frame.

### Implication for the current SF-AIA derivation

The relation $r_n\simeq-w_n\Delta_n$ in SF-AIA §8.1 is conditional on treating
the other fields as the baseline. For a residual after a pixel fit to the same
perturbed data, the projection above must be accounted for in a complete
first-order inverse formulation. This distinction should be discussed before
revising the established document.

## 4. Proposed estimator on the pixel grid

One replacement for a polynomial basis is a spatial smoothness assumption.
Keep a separate unknown $\Delta_n(x,y)$ at every pixel and frame, and estimate
the fields jointly by

$$
\boxed{
\widehat\Delta=\arg\min_\Delta
\left\{
\sum_{x,y}\left\|r(x,y)+R D(x,y)\Delta(x,y)\right\|_2^2
+\lambda\sum_n\left\|\nabla_{x,y}\Delta_n\right\|_2^2
\right\}.
}
$$

Impose the spatial and temporal conventions corresponding to the main model:

$$
\langle\Delta_n\rangle_{x,y}=0\quad\text{for every }n,
\qquad
\langle\Delta_n(x,y)\rangle_n=0\quad\text{for every pixel}.
$$

The gradient is a discrete spatial derivative with an explicitly chosen pixel
spacing and boundary treatment. The parameter $\lambda>0$ controls how strongly
rapid spatial variation is penalized. This is a general-form Tikhonov
regularization construction; see
[Hansen's treatment of inverse problems](https://www.imm.dtu.dk/~pcha/Book/mm04.html).

This estimator does not assume a polynomial degree, but it does assume smoothness.
The regularizer selects among solutions that the measurements cannot distinguish;
it does not make their previously invisible components experimentally measured.
Low-contrast regions depend especially strongly on that assumption.

For clarity, vectorize all the fields as $d$, let $B$ apply $RD(x,y)$ at each
pixel, let $L$ apply the spatial gradients, and let $Cd=0$ collect independent
mean constraints. The normal equations can be written

$$
\begin{pmatrix}
B^\top B+\lambda L^\top L&C^\top\\
C&0
\end{pmatrix}
\begin{pmatrix}d\\\mu\end{pmatrix}
=
\begin{pmatrix}-B^\top r\\0\end{pmatrix}.
$$

One mean constraint is redundant when both families use consistent uniform
averages; remove dependent constraints or use a null-space parameterization.
Uniqueness requires

$$
\ker B\cap\ker L\cap\ker C=\{0\}.
$$

For a connected pixel grid, a gradient penalty has spatial constants as its
null space; the per-frame zero-spatial-mean constraints eliminate those constants.
Disconnected masks require corresponding care.

## 5. A single perturbative correction

With the starting AIA solution frozen, the proposed procedure is:

1. Form a consistent pixel-fit residual and evaluate $w_n$ from the AIA estimate.
2. Solve the constrained, regularized linear problem for $\widehat\Delta_n$.
3. Correct the original normalized, offset-subtracted stack to first order:

$$
I_n^{\mathrm{corr}}=I_n+w_n\widehat\Delta_n.
$$

4. Perform the pixel solve with the same fixed gains and piston steps:

$$
\beta_{\mathrm{corr}}=A^+I^{\mathrm{corr}}
=\widehat\beta+A^+D\widehat\Delta,
$$

then recover

$$
b_{\mathrm{corr}}=\sqrt{u_{\mathrm{corr}}^2+v_{\mathrm{corr}}^2},
\qquad
\Phi_{\mathrm{corr}}=\operatorname{atan2}(-v_{\mathrm{corr}},u_{\mathrm{corr}}).
$$

This replaces the nonlinear outer refinement loop with one linear inverse
problem. Solving that potentially large linear system may still use numerical
iterations; it is not necessarily cheaper than small polynomial fits.

Only if the first-order field is recovered sufficiently accurately does the
remaining model error become second order. Regularization bias, noise, and
unobservable components can leave first-order errors. Smallness means a small
phase perturbation in radians, not merely a small error relative to a large carrier.

## 6. If gains and piston steps also need correction

The fixed-$g_n$, fixed-$\delta_n$ analysis isolates the main issue. A more complete
first-order solve would include corrections to all fitted nuisance parameters.
At a baseline with $\Delta_n=0$, put $\theta_n=\Phi+\delta_n$. Then

$$
\begin{aligned}
r_n\simeq{}&\eta_a+\eta_{c_n}
+g_n\cos\theta_n\,\eta_b
+b\cos\theta_n\,\eta_{g_n}\\
&-w_n\left(\eta_\Phi+\eta_{\delta_n}+\Delta_n\right).
\end{aligned}
$$

Here $\eta$ denotes a first-order parameter correction, and $c_n$ is the additive
frame offset. The static-field corrections depend on position; the gain, piston,
and offset corrections are shared across each frame.

These terms form a joint Jacobian. One can either solve jointly with the spatial
penalty or eliminate the nuisance columns before estimating $\Delta$. The latter
generalizes the projector $R$ above. The gain, background, phase-origin, and other
gauge freedoms also need explicit treatment; this note does not prescribe a
complete joint gauge implementation.

## 7. Questions to settle before implementation

- What spatial variation is physically plausible, and on what length scale?
- Is there useful temporal structure, or must every frame's field be independent?
- Can calibration provide information that the acquisition cannot identify?
- Are the perturbations small enough for a single linearization?
- Should the initial gains and piston steps be held fixed or corrected jointly?

The essential distinction is between propagating a known perturbation and
estimating an unknown one. The former is already available pointwise in SF-AIA;
the latter needs a justified constraint or additional measurement.
