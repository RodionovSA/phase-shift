# The residual equation: correct perturbation structure

Working note. Supersedes the earlier version of this file, which was built on
the wrong convention. The LaTeX has not been changed.

## 0. What I had wrong

I had defined $f^{(0)}$ as the *exact* parameter and the AIA output as
$\hat f=f^{(0)}+f^{(1)}$. That inverts the perturbation problem. With that
convention the data expansion reads $I_n=a^{(0)}+P^{(0)}_nu^{(0)}+Q^{(0)}_nv^{(0)}+w_n\Delta^{(1)}_n$,
which silently asserts that the undecorated $a$ of Eq. (3) equals $a^{(0)}$ —
i.e. it drops $a^{(1)},u^{(1)},v^{(1)},P^{(1)}_n,Q^{(1)}_n$ from the model
entirely. That is the error you flagged.

The correct structure is the one already in `docs/vp_aia.md` and in the
implementation:

$$a=a^{(0)}+a^{(1)}+a^{(2)}+\cdots,\qquad\text{and likewise for }u,v,P_n,Q_n,$$

with $f^{(0)}$ the **zeroth-order solution, i.e. what AIA returns**, $f^{(1)}$
the first-order correction, and the exact parameter the sum of the series. The
step field has no zeroth-order term, since the piston model has none:

$$\Delta_n=\Delta_n^{(1)}+\Delta_n^{(2)}+\cdots$$

A consequence worth stating: because $f^{(0)}$ *is* the AIA output, it is known.
No hatted symbols are needed anywhere — $\hat a$ and $a^{(0)}$ are the same
object.

## 1. Representing the AIA solution through orders

Everything is written in orders of $\Delta_n$. No symbol appears that is not an
order of something.

**The exact parameters.** $a,u,v,P_n,Q_n$ in paper Eq. (5) are properties of
the experiment. They do not depend on $\Delta_n$, so their expansion terminates
at order zero:

$$a=a^{(0)},\quad u=u^{(0)},\quad v=v^{(0)},\quad P_n=P_n^{(0)},\quad Q_n=Q_n^{(0)}$$

The step field has no order-zero term, since the piston model has none:
$\Delta_n=\Delta_n^{(1)}+\Delta_n^{(2)}+\cdots$

**The AIA solution.** AIA is a least-squares fit to data that depends on
$\Delta_n$, so the AIA output depends on $\Delta_n$ and *does* have a
non-trivial expansion. Denote its terms by the same superscripts:

$$a_{\text{AIA}}=a^{(0)}_{\text{AIA}}+a^{(1)}_{\text{AIA}}+a^{(2)}_{\text{AIA}}+\cdots,
\qquad a^{(k)}_{\text{AIA}}=O(\Delta^k),$$

and likewise for $u,v,P_n,Q_n$. Its order-zero term is its value at
$\Delta_n\equiv0$; there the piston model is exact and a noiseless fit returns
the generating parameters, so

$$a^{(0)}_{\text{AIA}}=a^{(0)},\quad u^{(0)}_{\text{AIA}}=u^{(0)},\quad
v^{(0)}_{\text{AIA}}=v^{(0)},\quad P^{(0)}_{n,\text{AIA}}=P_n^{(0)},\quad
Q^{(0)}_{n,\text{AIA}}=Q_n^{(0)}\tag{A0}$$

**This is what "AIA is not a pure order" means.** AIA does not return
$a^{(0)}$; it returns the whole series $a^{(0)}+a^{(1)}_{\text{AIA}}+\cdots$.
Its order-zero term is the quantity we want, and everything from order one on
is contamination. §2 computes $a^{(1)}_{\text{AIA}}$ and shows what produces it.

**The data.** Paper Eq. (3) in the quadrature variables of Eq. (5), with the
exact parameters written in order notation by the first display above:

$$I_n=\underbrace{a^{(0)}+P_n^{(0)}u^{(0)}+Q_n^{(0)}v^{(0)}}_{\text{order }0}
+\underbrace{w_n^{(0)}\Delta_n^{(1)}}_{\text{order }1}+O(\Delta^2),
\qquad w_n^{(0)}=P_n^{(0)}v^{(0)}-Q_n^{(0)}u^{(0)}\tag{M}$$

The single first-order term is not a truncation: by the first display the exact
parameters have no order-one part, so there is nothing else at that order to
keep. The order-one terms $a^{(1)}_{\text{AIA}}$ etc. belong to the AIA
*solution*, not to the data, and they enter in §2 through the AIA prediction.

## 2. The order-one contamination of the AIA solution

At order zero, (M) says the data is exactly the piston model, and (A0) says AIA
reproduces it. At order one the data carries

$$e_n:=w_n^{(0)}\Delta_n^{(1)},$$

which the piston model cannot represent as a whole. With $A$ and $A_p$ of paper
Eq. (6) built from $P_n^{(0)},Q_n^{(0)}$, least squares splits it in two:

$$e=\underbrace{AA_p^{-1}A^\top e}_{\text{inside the model's span}}
+\underbrace{\Pi e}_{\text{outside it}}$$

The first part is indistinguishable from a change of $a,u,v$, so the fit
absorbs it. That absorbed part **is** the order-one term of the AIA solution:

$$\begin{pmatrix}a^{(1)}_{\text{AIA}}\\u^{(1)}_{\text{AIA}}\\v^{(1)}_{\text{AIA}}\end{pmatrix}
=A_p^{-1}A^\top e\tag{C}$$

which is paper Eq. (23). The second part cannot be absorbed and is left over as
the residual of paper Eq. (18),

$$r_n:=I_n-\big(a_{\text{AIA}}+P_{n,\text{AIA}}u_{\text{AIA}}+Q_{n,\text{AIA}}v_{\text{AIA}}\big),
\qquad r=\Pi e+O(\Delta^2)\tag{R0}$$

So $r_n$ is order one — it vanishes with the step error — and it is *smaller*
than the signal $e$ that produced it, by exactly the absorbed part. This is why
fitting $r_n$ as though it were $e_n$ underestimates $\Delta_n$, and why the
order-one terms of the AIA solution cannot be dropped: they are the absorbed
part, carried with the sign that puts it back.

## 3. Order one: the residual equation

The order-one terms of (M) must account for exactly that defect. Substituting
(R0) into (M):

$$r_n=a^{(1)}+P_n^{(0)}u^{(1)}+Q_n^{(0)}v^{(1)}+u^{(0)}P_n^{(1)}+v^{(0)}Q_n^{(1)}
+w_n\sum_{j=1}^J\alpha_{nj}H_j\tag{O1}$$

using paper Eq. (14) for $\Delta_n^{(1)}$. This is `docs/vp_aia.md` Eq. (17),
and it is the equation the fit should be built on. Every term carries a plus
sign: the corrections and the step-error signal together reproduce the defect
left by the order-zero solve.

Note what this says about the paper's Eq. (19), which currently reads

$$r_n=w_n\Delta_n^{(1)}-\big(a^{(1)}+\hat P_nu^{(1)}+\hat Q_nv^{(1)}\big)-\hat uP_n^{(1)}-\hat vQ_n^{(1)}$$

It is (O1) with every $f^{(1)}$ negated, which is consistent only under the
inverted convention of §0. Under the correct convention the paper's Eq. (19) has
the wrong signs.

## 4. Eliminating the pixel-side corrections

At one pixel, collect the $N$ frame values. With $A$ and $A_p$ of paper Eq. (6)
built from $P_n^{(0)},Q_n^{(0)}$, the first three terms of (O1) form the column
$A\big(a^{(1)},u^{(1)},v^{(1)}\big)^\top$. Two facts from paper Eq. (22):

$$\Pi A=0,\qquad A^\top r=0\ \Rightarrow\ \Pi r=r$$

the second holding exactly, at any size of $\Delta_n$, because the alternation
ends with a pixel step. Applying $\Pi$ to (O1) therefore removes the pixel-side
corrections and leaves the residual untouched:

$$r_n=\Pi\Big[u^{(0)}P_n^{(1)}+v^{(0)}Q_n^{(1)}+w_n\sum_j\alpha_{nj}H_j\Big]\tag{O1P}$$

which is `docs/vp_aia.md` Eq. (20) — and the paper's Eq. (24) with the signs of
$P_n^{(1)},Q_n^{(1)}$ restored.

## 5. What is determined and what is fitted

Applying $A^\top$ to (O1) instead of $\Pi$, and using $A^\top r=0$:

$$\begin{pmatrix}a^{(1)}\\u^{(1)}\\v^{(1)}\end{pmatrix}
=-A_p^{-1}A^\top\begin{pmatrix}u^{(0)}P_1^{(1)}+v^{(0)}Q_1^{(1)}+w_1\Delta_1^{(1)}\\ \vdots\\
u^{(0)}P_N^{(1)}+v^{(0)}Q_N^{(1)}+w_N\Delta_N^{(1)}\end{pmatrix}$$

and with the constraints of paper Eq. (29), $A^\top(P^{(1)})=A^\top(Q^{(1)})=0$,
the frame-side part drops and this is `docs/vp_aia.md` Eq. (26) — the minus sign
the implementation uses at `vp_aia.py:154`. So the pixel-side corrections are
**determined**, not free; the paper reaches the same formula but presents it as
an independent result rather than as a consequence of $A^\top r=0$.

The frame-side $P_n^{(1)},Q_n^{(1)}$ are **not** determined this way, because
the frame step, paper Eq. (8), regresses on $(1,u^{(0)},v^{(0)})$ with a scalar
intercept $c_n$ rather than the background field $a(x,y)$. There is no
frame-side analogue of $A^\top r=0$; in particular $\sum_{x,y}u^{(0)}r_n\neq0$,
and those quantities are the first $2N$ entries of the right-hand side of paper
Eq. (28). They must be fitted, which is what the reduced system does.

## 6. Consequences for the paper

1. **Revert the convention.** $f^{(0)}$ = AIA output, $f^{(1)}$ = correction,
   exact $=f^{(0)}+f^{(1)}+O(\Delta^2)$. This restores agreement with
   `docs/vp_aia.md` and with `methods/vp_aia.py`, and removes the hats
   entirely.
2. **Fix the signs** of Eqs. (19), (24), (27)–(28) and (32) accordingly: all
   $f^{(1)}$ terms positive, $D_u,D_v$ back to $+$, the first $2N$ entries of
   $\rho$ back to $+$, and Eq. (32) back to $-A_p^{-1}(\cdots)$.
3. **Replace the appendix derivation** with §§1–3 above: decompose every
   variable, collect order zero (what AIA solves) and order one (the residual
   equation). This is the derivation you asked for and the one the structure
   actually supports.
4. Optionally add §5's distinction — pixel-side determined, frame-side fitted —
   and the frame-step caveat.

Awaiting your go-ahead before touching `paper/vp_aia.tex`.
