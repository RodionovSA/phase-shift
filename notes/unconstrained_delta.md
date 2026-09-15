# Why unrestricted phase-error corrections are not unique

A linear system has a unique solution only if its columns are independent.
Here, different parameter corrections can cancel one another exactly.

## An explicit cancellation

Choose any small spatial function $h(x,y)$ with zero spatial mean. Add the
following changes to any proposed first-order solution:

$$\begin{aligned}
a^{(1)}&\longrightarrow a^{(1)},\\
u^{(1)}&\longrightarrow u^{(1)}+v^{(0)}h,\\
v^{(1)}&\longrightarrow v^{(1)}-u^{(0)}h,\\
\Delta_n^{(1)}&\longrightarrow\Delta_n^{(1)}-h,
\end{aligned}$$

leaving $P_n^{(1)}$ and $Q_n^{(1)}$ unchanged. The change in the predicted
first-order intensity is

$$\begin{aligned}
\text{intensity change}
&=P_n^{(0)}v^{(0)}h-Q_n^{(0)}u^{(0)}h
-\big(P_n^{(0)}v^{(0)}-Q_n^{(0)}u^{(0)}\big)h\\
&=0.
\end{aligned}$$

Thus every choice of $h$ gives another solution with exactly the same residual
and least-squares loss. The zero spatial mean of $\Delta_n$ is preserved.

Physically, this moves a static spatial phase pattern between the recovered phase
and the phase-error maps. In the full nonlinear model, the corresponding exact
ambiguity is

$$\Phi\longrightarrow\Phi+h,\qquad
\Delta_n\longrightarrow\Delta_n-h.$$

Their sum, and therefore every measured intensity, is unchanged. This ambiguity
is not caused by truncating the expansion.

## What this means for the matrix solve

The nonzero corrections above produce zero intensity change: they form a null
direction of the linear system. Consequently, the unconstrained joint normal
matrix is singular. A numerical method can select one solution, but the data
do not identify it as the physical one.

Requiring zero frame mean of $\Delta_n$ at every pixel would eliminate this
particular ambiguity. That restriction has not been selected here, and removing
this one ambiguity alone does not establish uniqueness of the entire fit.

For example, even with frame coefficients fixed, a pixel has $N$ intensity
equations and $N+3$ unknown corrections: $N$ phase errors plus corrections to
$a,u,v$. One temporal-mean constraint still leaves at least two local degrees
of freedom before cross-pixel restrictions are applied. The explicit cancellation
above is already sufficient to show why linearity alone does not guarantee a
unique solution.
