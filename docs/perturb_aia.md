## Introduction

The $N$ measured intensities are modeled with a static background $a$, fringe amplitude $b$, and phase $\Phi$, a spatially uniform per-frame contrast $g_n$, and a phase-step field $\Delta_n$ that includes the piston:

$$I_n(x, y) = a(x, y) + g_n\,b(x, y)\,\cos\big(\Phi(x, y) + \Delta_n(x, y)\big), \qquad n = 1 \dots N. \tag{1}$$

Expanding the cosine gives the quadrature form

$$I_n = a + P^{(\Delta)}_n u + Q^{(\Delta)}_n v, \qquad u = b\cos\Phi,\quad v = -b\sin\Phi,\quad P^{(\Delta)}_n = g_n\cos\Delta_n,\quad Q^{(\Delta)}_n = g_n\sin\Delta_n. \tag{2}$$

Index the pixels by $k = 1 \dots K$ and write $f_k = f(x_k, y_k)$. Collecting the $N$ frames of pixel $k$ gives one linear system per pixel,

$$\mathbf I_k = \mathbf A_k\,\mathbf x_k,\qquad
\mathbf I_k = \begin{pmatrix} I_{1k} \\ \vdots \\ I_{Nk} \end{pmatrix},\qquad
\mathbf A_k = \begin{pmatrix} 1 & P^{(\Delta)}_{1k} & Q^{(\Delta)}_{1k} \\ \vdots & \vdots & \vdots \\ 1 & P^{(\Delta)}_{Nk} & Q^{(\Delta)}_{Nk} \end{pmatrix},\qquad
\mathbf x_k = \begin{pmatrix} a_k \\ u_k \\ v_k \end{pmatrix}. \tag{3}$$

A pixel alone gives $N$ equations for its three pixel unknowns and $N$ phase steps, so the phase-step field must be shared between pixels. We expand it on $J+1$ linearly independent spatial modes,

$$\Delta_n(x, y) = \sum_{j=0}^{J} c_{jn}\,H_j(x, y), \qquad H_0 = 1, \qquad c_{j1} = 0, \tag{4}$$

where $H_0$ is the piston mode and $c_{j1} = 0$ fixes the gauge $\Phi \to \Phi + f$, $\Delta_n \to \Delta_n - f$ by $\Delta_1 = 0$. The contrast scale $g_n \to g_n/s$, $b \to s\,b$ is fixed by $\operatorname{median}_n g_n = 1$. Appendix A counts the unknowns, bounds $J$, and shows that the method needs $N \ge 5$ frames.

The frame parameters $\boldsymbol\theta = (g_n, c_{jn})$ are shared by all pixels and enter every $\mathbf A_k$. With noisy data we solve

$$\min_{\boldsymbol\theta,\ \mathbf x_1, \dots, \mathbf x_K}\ \sum_{k=1}^{K} \big\|\mathbf I_k - \mathbf A_k(\boldsymbol\theta)\,\mathbf x_k\big\|^2. \tag{5}$$

For fixed $\boldsymbol\theta$, Eq. (5) splits into $K$ independent linear fits of three unknowns; all coupling between pixels passes through $\boldsymbol\theta$, in which the problem is nonlinear.

## AIA baseline

With the piston $\delta_n = c_{0n}$ alone, $\Delta_n = \delta_n$, Eq. (3) becomes the piston model $I_{nk} = a_k + P_n u_k + Q_n v_k$ with $P_n = g_n\cos\delta_n$, $Q_n = g_n\sin\delta_n$. AIA (Wang and Han, 2004) fits it by minimizing

$$L_{\text{AIA}} = \sum_{k=1}^{K}\sum_{n=1}^{N}\big(I_{nk} - a_k - P_n u_k - Q_n v_k\big)^2. \tag{6}$$

The loss is quadratic in $\mathbf x_k$ for fixed $P_n, Q_n$, and quadratic in $P_n, Q_n$ for fixed $\mathbf x_k$, so AIA alternates two linear least-squares steps, each obtained by setting the gradient over its own unknowns to zero.

**Pixel step.** For fixed $P_n, Q_n$, each pixel solves

$$\mathbf M\,\mathbf x_k = \mathbf A^{(0)T}\mathbf I_k, \qquad \mathbf M = \mathbf A^{(0)T}\mathbf A^{(0)}, \qquad
\mathbf A^{(0)} = \begin{pmatrix} 1 & P_1 & Q_1 \\ \vdots & \vdots & \vdots \\ 1 & P_N & Q_N \end{pmatrix}, \tag{7}$$

with one $3\times3$ matrix $\mathbf M$ shared by all pixels.

**Frame step.** For fixed $\mathbf x_k$, each frame solves

$$\mathbf C\begin{pmatrix} P_n \\ Q_n \end{pmatrix} = \sum_{k=1}^{K}\big(I_{nk} - a_k\big)\begin{pmatrix} u_k \\ v_k \end{pmatrix}, \qquad
\mathbf C = \sum_{k=1}^{K}\begin{pmatrix} u_k^2 & u_kv_k \\ u_kv_k & v_k^2 \end{pmatrix}, \tag{8}$$

with one $2\times2$ matrix $\mathbf C$ shared by all frames.

Each step minimizes Eq. (6) exactly over its own unknowns, so the loss never increases. The fixed point gives the AIA estimates $\hat P_n, \hat Q_n, \hat{\mathbf x}_k$, which are brought to the frame conventions: shift, $\sum_k (a_k - \bar a)u_k = \sum_k (a_k - \bar a)v_k = 0$; whitening, $\sum_k u_k^2 = \sum_k v_k^2$ and $\sum_k u_kv_k = 0$; phase origin $\hat\delta_1 = 0$; and $\operatorname{median}_n\hat g_n = 1$. Bars denote pixel means.

## Idea: expansion around AIA

The piston $\delta_n$ is a full phase step and is not small, but the spatially varying remainder usually is:

$$\Delta_{nk} = \delta_n + \varepsilon_{nk}, \qquad \varepsilon_{nk} = \sum_{j=1}^{J} c_{jn}H_{jk}, \qquad |\varepsilon_{nk}| \ll 1. \tag{9}$$

At $\varepsilon = 0$ the model is the piston model of AIA. The AIA estimates $\hat P_n, \hat Q_n, \hat{\mathbf x}_k$ are known, but biased at first order in $\varepsilon$, since the data contain the step error. Adding the mode coefficients to the same alternation converges slowly: part of the step-error signal can be reproduced by the pixel unknowns, so each pixel step absorbs it and each frame step sees only the remainder.

Instead, we expand around the AIA solution. Its frame parameters are the known centre,

$$P_n = \hat P_n + p_n, \qquad Q_n = \hat Q_n + q_n, \tag{10}$$

and the corrections $p_n, q_n$ and the step error $\varepsilon_{nk}$ are the small quantities, all of first order. Expanding Eq. (2) in them (Appendix B),

$$\mathbf A_k = \mathbf A^{(0)} + \mathbf A^{(1)}_k + \mathbf A^{(2)}_k + O(\varepsilon^3), \qquad
\mathbf A^{(0)}:\ \big(1,\ \hat P_n,\ \hat Q_n\big), \qquad
\mathbf A^{(1)}_k:\ \big(0,\ p_n - \hat Q_n\varepsilon_{nk},\ q_n + \hat P_n\varepsilon_{nk}\big), \tag{11}$$

where the rows $n$ are shown. The zeroth order $\mathbf A^{(0)}$ is known and the same at every pixel. The unknowns, collected into a vector $\mathbf t = (p_n, q_n, c_{jn})$ of length $T$, enter the first order linearly.

## One pass: first-order corrections

**Pixel unknowns.** For fixed $\boldsymbol\theta$, $\mathbf x_k$ solves the normal equations $\mathbf A_k^T\mathbf A_k\mathbf x_k = \mathbf A_k^T\mathbf I_k$. Expanding $\mathbf x_k = \sum_m \mathbf x^{(m)}_k$ with Eq. (11) gives, order by order (Appendix C),

$$\mathbf M\,\mathbf x^{(0)}_k = \mathbf A^{(0)T}\mathbf I_k, \qquad
\mathbf M\,\mathbf x^{(m)}_k = \mathbf A^{(m)T}_k\mathbf I_k - \sum_{i=1}^{m}\mathbf M^{(i)}_k\,\mathbf x^{(m-i)}_k, \qquad
\mathbf M^{(m)}_k = \sum_{i=0}^{m}\mathbf A^{(i)T}_k\mathbf A^{(m-i)}_k, \tag{12}$$

with $\mathbf M = \mathbf M^{(0)} = \mathbf A^{(0)T}\mathbf A^{(0)}$. The zeroth order is the AIA pixel solution, $\mathbf x^{(0)}_k = \hat{\mathbf x}_k$, and every order is solved with the same $3\times3$ matrix $\mathbf M$. The pixel unknowns are therefore known functions of $\mathbf t$.

**Reduced loss.** Substituting them into Eq. (5) leaves a loss in $\mathbf t$ alone. It is built from the AIA residual and the first-order step-error signal,

$$\mathbf r_k = \mathbf I_k - \mathbf A^{(0)}\mathbf x^{(0)}_k = \boldsymbol\Pi\mathbf I_k, \qquad \boldsymbol\Pi = \mathbf 1 - \mathbf A^{(0)}\mathbf M^{-1}\mathbf A^{(0)T}, \tag{13}$$

$$\mathbf A^{(1)}_k\mathbf x^{(0)}_k = \mathbf G_k\,\mathbf t, \tag{14}$$

where the $N\times T$ matrix $\mathbf G_k$ is built from the AIA solution. For data that follow Eq. (1), up to small noise, $\mathbf r_k$ is itself of first order, and to second order (Appendix D)

$$L(\mathbf t) = \sum_{k=1}^{K}\big\|\boldsymbol\Pi\big(\mathbf r_k - \mathbf G_k\mathbf t\big)\big\|^2 + O(\varepsilon^3). \tag{15}$$

Eq. (15) fits the AIA residual with the step-error signal, after removing from both the part that the pixel unknowns absorb. This is the variable-projection form of the problem (Golub and Pereyra, 1973). Its minimum is one linear system,

$$\mathbf H\,\mathbf t = \mathbf h, \qquad
\mathbf H = \sum_{k=1}^{K}\mathbf G_k^T\boldsymbol\Pi\mathbf G_k, \qquad
\mathbf h = \sum_{k=1}^{K}\mathbf G_k^T\mathbf r_k. \tag{16}$$

$\mathbf H$ and $\mathbf h$ are accumulated in one pass over the pixels, and their size does not grow with $K$. The solution $\mathbf t$ has an error of second order in $\varepsilon$, and so do the corrected estimates $\hat P_n + p_n$, $\hat Q_n + q_n$, $\varepsilon_{nk}$, and the pixel unknowns $\mathbf x^{(0)}_k + \mathbf x^{(1)}_k$.

**Gauge.** Eq. (16) is singular. By Eq. (13), $\boldsymbol\Pi$ removes the columns $(1, \dots, 1)^T$, $(\hat P_1, \dots, \hat P_N)^T$, and $(\hat Q_1, \dots, \hat Q_N)^T$, so adding any combination of $1, \hat P_n, \hat Q_n$ to $p_n$, or to $q_n$, leaves Eq. (15) unchanged. These six directions are the linearized shifts and $2\times2$ transformations of the AIA quadrature frame. Two of them are fixed by holding frame 1 at the AIA values, $p_1 = q_1 = 0$, so that $T = (J+2)(N-1)$; the other four by requiring the corrected pixel fields to satisfy the same frame conventions as the AIA solution, four linear conditions on $\mathbf t$ (Appendix E). Finally, the corrected contrasts are rescaled to $\operatorname{median}_n g_n = 1$.

## Higher accuracy: repeated passes

One pass leaves an error of second order. To reduce it, remove the current step-error signal from the data with the full model and repeat. With the current estimates, marked by a tilde, and the accumulated step error $\tilde\varepsilon_{nk}$,

$$I'_{nk} = I_{nk} - \big(\tilde P^{(\Delta)}_{nk} - \tilde P_n\big)\tilde u_k - \big(\tilde Q^{(\Delta)}_{nk} - \tilde Q_n\big)\tilde v_k, \tag{17}$$

where $\tilde P^{(\Delta)}_{nk}, \tilde Q^{(\Delta)}_{nk}$ follow from Eq. (2) with $\Delta_{nk} = \tilde\delta_n + \tilde\varepsilon_{nk}$. The corrected stack $I'$ contains only the remaining step error $\varepsilon - \tilde\varepsilon$. One round runs AIA on $I'$, starting from the current $\tilde P_n, \tilde Q_n$, solves Eq. (16), adds the new step error to $\tilde\varepsilon$, and replaces the other estimates by the corrected ones.

The rounds converge linearly. Eq. (14) uses the step-error sensitivity at zero step error, while the remaining error enters $I'$ with the sensitivity at $\tilde\varepsilon$; the two differ at first order, so each round reduces the remaining error by a factor of order $\max|\varepsilon|$. At convergence the solution is unbiased to all orders in $\varepsilon$.

The method is therefore:

1. Run AIA and bring its solution to the frame conventions.
2. Accumulate $\mathbf H$ and $\mathbf h$, Eqs. (13)–(16), in one pass over the pixels.
3. Solve Eq. (16) under the gauge conditions, and form the corrected estimates.
4. If the step error is large, correct the stack, Eq. (17), and repeat from step 1.

## Appendix A. Unknowns and gauges of the full model

Pixel $k$ gives $N$ equations for its own $N+3$ unknowns $a_k, u_k, v_k, \Delta_{1k}, \dots, \Delta_{Nk}$; the $N$ contrasts $g_n$ add only $N/K$ unknowns per pixel. One of the three missing equations is the gauge $\Phi \to \Phi + f(x, y)$, $\Delta_n \to \Delta_n - f(x, y)$, fixed by $\Delta_1 = 0$. The other two must come from coupling between pixels, which Eq. (4) provides.

With Eq. (4) and the contrast scale fixed, the stack gives $NK$ equations for $3K + (J+2)(N-1)$ unknowns: the fields $a, b, \Phi$, the contrasts less the median condition, and $c_{jn}$ with $n \ge 2$. A unique solution requires $NK \ge 3K + (J+2)(N-1)$, which bounds the number of modes by

$$J + 1 \le \frac{N-3}{N-1}\,K - 1, \tag{A1}$$

and needs $N \ge 4$. The count is only necessary. The first-order fit of Eq. (15) needs $N \ge 5$: at $N = 4$, $\operatorname{span}\{1, \hat P_n, \hat Q_n\}$ has codimension one in $\mathbb R^4$, so for each mode some nonzero $c_{jn}$ with $c_{j1} = 0$ has both $c_{jn}\hat P_n$ and $c_{jn}\hat Q_n$ in that span, and its step-error signal is removed by $\boldsymbol\Pi$.

## Appendix B. Expansion of the design matrix

The angle-sum identities give $P^{(\Delta)}_{nk} = P_n\cos\varepsilon_{nk} - Q_n\sin\varepsilon_{nk}$ and $Q^{(\Delta)}_{nk} = Q_n\cos\varepsilon_{nk} + P_n\sin\varepsilon_{nk}$. Substituting Eq. (10) and expanding in $p_n, q_n, \varepsilon_{nk}$, row $n$ of $\mathbf A^{(m)}_k$ is $\big(0,\ P^{(m)}_{nk},\ Q^{(m)}_{nk}\big)$ for $m \ge 1$, with

$$\begin{pmatrix} P^{(m)}_{nk} \\ Q^{(m)}_{nk} \end{pmatrix}
= \frac{\varepsilon_{nk}^m}{m!}\,\mathbf R\Big(\frac{m\pi}{2}\Big)\begin{pmatrix} \hat P_n \\ \hat Q_n \end{pmatrix}
+ \frac{\varepsilon_{nk}^{m-1}}{(m-1)!}\,\mathbf R\Big(\frac{(m-1)\pi}{2}\Big)\begin{pmatrix} p_n \\ q_n \end{pmatrix}, \qquad
\mathbf R(\phi) = \begin{pmatrix} \cos\phi & -\sin\phi \\ \sin\phi & \cos\phi \end{pmatrix}. \tag{B1}$$

For $m = 1$ this gives the rows of Eq. (11), and for $m = 2$ the rows $-\big(0,\ \tfrac12\hat P_n\varepsilon_{nk}^2 + q_n\varepsilon_{nk},\ \tfrac12\hat Q_n\varepsilon_{nk}^2 - p_n\varepsilon_{nk}\big)$.

## Appendix C. Pixel unknowns order by order

For fixed $\boldsymbol\theta$, only $L_k = \|\mathbf I_k - \mathbf A_k\mathbf x_k\|^2$ depends on $\mathbf x_k$, and setting its gradient $-2\,\mathbf A_k^T(\mathbf I_k - \mathbf A_k\mathbf x_k)$ to zero gives $\mathbf A_k^T\mathbf A_k\,\mathbf x_k = \mathbf A_k^T\mathbf I_k$. The data $\mathbf I_k$ are not expanded. Substituting $\mathbf A_k = \sum_m \mathbf A^{(m)}_k$ and $\mathbf x_k = \sum_m \mathbf x^{(m)}_k$, the $m$-th order of $\mathbf A_k^T\mathbf A_k$ is $\mathbf M^{(m)}_k$ of Eq. (12), and collecting the terms of order $m$ on both sides gives

$$\sum_{i=0}^{m}\mathbf M^{(i)}_k\,\mathbf x^{(m-i)}_k = \mathbf A^{(m)T}_k\mathbf I_k,$$

which is Eq. (12) with the $i = 0$ term moved to the left.

## Appendix D. Reduced loss

At the solution of the normal equations the residual is orthogonal to the columns of $\mathbf A_k$, so $L_k = \mathbf I_k^T\mathbf I_k - \mathbf I_k^T\mathbf A_k\mathbf x_k$. The $m$-th order of $\mathbf A_k\mathbf x_k$ is $\sum_{i=0}^{m}\mathbf A^{(i)}_k\mathbf x^{(m-i)}_k$. In each order, $\mathbf I_k^T\mathbf A^{(0)}\mathbf x^{(m)}_k = \mathbf x^{(0)T}_k\mathbf M\,\mathbf x^{(m)}_k$ is replaced using Eq. (12), which gives

$$L^{(0)}_k = \|\mathbf r_k\|^2, \qquad
L^{(1)}_k = -2\,\mathbf r_k^T\mathbf A^{(1)}_k\mathbf x^{(0)}_k,$$

$$L^{(2)}_k = -2\,\mathbf r_k^T\mathbf A^{(2)}_k\mathbf x^{(0)}_k + \big\|\mathbf A^{(1)}_k\mathbf x^{(0)}_k\big\|^2 - \big\|\mathbf A^{(0)}\mathbf x^{(1)}_k\big\|^2,$$

so that, to second order,

$$L = \sum_{k=1}^{K}\Big(\big\|\mathbf r_k - \mathbf A^{(1)}_k\mathbf x^{(0)}_k\big\|^2 - \big\|\mathbf A^{(0)}\mathbf x^{(1)}_k\big\|^2 - 2\,\mathbf r_k^T\mathbf A^{(2)}_k\mathbf x^{(0)}_k\Big) + O(\varepsilon^3). \tag{D1}$$

For data that follow Eq. (1), $\mathbf I_k = \mathbf A_k\mathbf x_k$ at the true parameters, and the AIA pixel step removes the zeroth-order part exactly, so

$$\mathbf r_k = \boldsymbol\Pi\mathbf A^{(1)}_k\mathbf x_k + O(\varepsilon^2)$$

is of first order. The terms of Eq. (D1) that contain $\mathbf r_k$ together with a correction are then of third order. Dropping them, Eq. (12) with $m = 1$, written as $\mathbf M\,\mathbf x^{(1)}_k = \mathbf A^{(1)T}_k\mathbf r_k - \mathbf A^{(0)T}\mathbf A^{(1)}_k\mathbf x^{(0)}_k$, leaves $\mathbf A^{(0)}\mathbf x^{(1)}_k = -\big(\mathbf 1 - \boldsymbol\Pi\big)\mathbf G_k\mathbf t$, and Eq. (D1) becomes Eq. (15).

## Appendix E. Gauge conditions for the corrections

The piston model is unchanged by $(u, v) \to (u, v)\mathbf T$, $(P_n, Q_n) \to (P_n, Q_n)\mathbf T^{-T}$ for any invertible $2\times2$ matrix $\mathbf T$, and by the shifts $(P_n, Q_n) \to (P_n + s_1, Q_n + s_2)$, $a \to a - s_1 u - s_2 v$. Linearized, these six freedoms are the null directions of Eq. (16). In the full model, row $n$ of $\mathbf A_k$ rotates $(P_n, Q_n)$ by $\varepsilon_{nk}$, so only the transformations $\mathbf T$ that commute with rotations, a common rotation and scale, remain exact; the first-order system does not resolve the others, and the conventions select the solution.

The AIA solution satisfies the frame conventions of the AIA baseline section. The corrected fields $\mathbf x^{(0)}_k + \mathbf x^{(1)}_k$ must satisfy the shift and whitening conventions as well. Since the AIA fields satisfy them, to first order this requires

$$\begin{aligned}
&\sum_k\big(a^{(1)}_k - \bar a^{(1)}\big)u^{(0)}_k + \big(a^{(0)}_k - \bar a^{(0)}\big)u^{(1)}_k = 0, \qquad
\sum_k\big(a^{(1)}_k - \bar a^{(1)}\big)v^{(0)}_k + \big(a^{(0)}_k - \bar a^{(0)}\big)v^{(1)}_k = 0, \\
&\sum_k\big(u^{(0)}_ku^{(1)}_k - v^{(0)}_kv^{(1)}_k\big) = 0, \qquad
\sum_k\big(u^{(0)}_kv^{(1)}_k + u^{(1)}_kv^{(0)}_k\big) = 0,
\end{aligned} \tag{E1}$$

with $\mathbf x^{(1)}_k = -\mathbf M^{-1}\mathbf A^{(0)T}\mathbf G_k\mathbf t$ from Appendix D. These are four linear conditions on $\mathbf t$, and Eq. (16) is solved on the subspace where they hold.

## References

- Z. Wang and B. Han, "Advanced iterative algorithm for phase extraction of randomly phase-shifted interferograms," *Optics Letters* **29**(14), 1671–1673 (2004).
- G. H. Golub and V. Pereyra, "The differentiation of pseudo-inverses and nonlinear least squares problems whose variables separate," *SIAM Journal on Numerical Analysis* **10**(2), 413–432 (1973).
