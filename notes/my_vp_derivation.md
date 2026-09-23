
True intensity, when Delta is small:
$$I_n = a + P_n u + Q_n v + w_n \Delta_n + O(\Delta^2_n)$$

AIA solution:
$$\hat I_n = \hat a + \hat P_n \hat u + \hat Q_n \hat v$$

Lets split true intensity in proper perturbation orders, because delta is at first order. 
$$I_n = I^{(0)}_n + I^{(1)}_n + O(\Delta^2_n)$$
where $I^{(0)}_n = a^{(0)} + P^{(0)}_n u^{(0)} + Q^{(0)}_n v^{(0)}$, and $I^{(1)}_n = a^{(1)}+P_n^{(0)}u^{(1)}+Q_n^{(0)}v^{(1)}+u^{(0)}P_n^{(1)}+v^{(0)}Q_n^{(1)}+w^{(0)}_n \Delta^{(1)}_n$.

## Expanding the AIA solution

The true parameters are properties of the experiment and do not depend on
$\Delta_n$, so they have no expansion. The AIA estimates do: they are computed
from a stack that contains $\Delta_n$, and they move as $\Delta_n$ moves. The
expansion is therefore of the estimates, about the true parameters.

To give "order in $\Delta_n$" a meaning for a field, fix its spatial shape and
scale it,

$$\Delta_n(x,y)=\epsilon\,D_n(x,y),$$

with $D_n$ fixed and $\epsilon$ a single scalar. Then $O(\Delta_n^2)$ means
$O(\epsilon^2)$, and the measured stack

$$I_n(\epsilon)=a+g_n b\cos\big(\Phi+\delta_n+\epsilon D_n\big)$$

is a smooth function of $\epsilon$ alone. AIA maps the stack to
$\hat a,\hat u,\hat v,\hat P_n,\hat Q_n$, so each estimate is likewise a
function of $\epsilon$, written $\hat f(\epsilon)$.

**The estimates are exact at $\epsilon=0$.** With $\Delta_n\equiv0$ the piston
model describes the data exactly, its residual vanishes at the true parameters,
and the conventions leave only one solution. A noiseless AIA fit therefore
returns the true parameters,

$$\hat f(0)=f.$$

This is a property of the fit, not a choice of notation.

**The estimates are smooth in $\epsilon$.** What AIA returns is a solution of
its own normal equations together with the normalization conditions. These
depend smoothly on the estimates and on $\epsilon$, and at $\epsilon=0$ they fix
the solution uniquely — the same identifiability the conventions were
introduced to supply. A single smooth branch therefore passes through
$\hat f(0)=f$, and

$$\hat f(\epsilon)=f+\epsilon\,\hat f'(0)+O(\epsilon^2),
\qquad\text{so}\qquad
\hat f-f=O(\Delta_n).$$

The gap between the estimates and the truth is first order in the phase-step
error. Everything perturbative below rests on this one statement.

**The same expansion read backwards.** Solving for the true parameter,

$$f=\hat f-\epsilon\,\hat f'(0)+O(\epsilon^2),$$

so the correction that carries the estimates back to the truth,

$$f^{(1)}:=f-\hat f,$$

is the negative of the first Taylor coefficient of $\hat f$. The two displays
are one expansion read in two directions; only the sign of $f^{(1)}$
distinguishes them.

**Anchoring the split.** Take $f^{(0)}:=\hat f$. Then the zeroth-order intensity
is the AIA prediction,

$$I^{(0)}_n=\hat a+\hat P_n\hat u+\hat Q_n\hat v=\hat I_n,$$

and the residual is the first-order intensity,

$$r_n=I_n-\hat I_n=I_n-I^{(0)}_n=I^{(1)}_n+O(\Delta_n^2).$$

Both hold by construction, not merely to some order, and the true parameters are
recovered by adding: $f=f^{(0)}+f^{(1)}+O(\Delta_n^2)$.

One assumption is worth naming. The frame step regresses on a scalar intercept
$c_n$ rather than on the background field $a(x,y)$, so what AIA converges to is a
stationary point of the joint least-squares problem only when the background is
uncorrelated with the quadratures — the same condition the quadrature frame
already requires.
