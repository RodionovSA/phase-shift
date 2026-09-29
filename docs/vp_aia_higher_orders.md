

Equation (8) in `vp_aia.md` gives residual for all order in $\Delta_n$ and equation (9) show specific formula for the first-order residual that leaves second order error. Now, let us form a second order residual:

$$r^{(2)}_n=I_n-\big(\hat a+\hat P_n\hat u+\hat Q_n\hat v\big)-r^{(1)}_n,\tag{1}$$

where $r^{(1)}_n$ is the right-hand side of `vp_aia.md` Eq. (9), the first-order model of the residual. Expanding all the intensity terms up to the second order, we have:

$$\begin{aligned}
r^{(2)}_n={}&\big(a^{(2)}+\hat P_nu^{(2)}+\hat Q_nv^{(2)}\big)+\hat uP_n^{(2)}+\hat vQ_n^{(2)}+P_n^{(1)}u^{(1)}+Q_n^{(1)}v^{(1)}\\
&+\big(\hat P_nv^{(1)}-\hat Q_nu^{(1)}+\hat vP_n^{(1)}-\hat uQ_n^{(1)}\big)\Delta_n-\tfrac12\big(\hat P_n\hat u+\hat Q_n\hat v\big)\Delta_n^2+O(\Delta_n^3).
\end{aligned}\tag{2}$$

Split the step error by order, $\Delta_n=\Delta_n^{(1)}+\Delta_n^{(2)}+O(\Delta_n^3)$, with $\Delta_n^{(k)}=\sum_j\alpha_{nj}^{(k)}H_j$. In the last two terms of Eq. (2), the coefficients are first and zeroth order, so only $\Delta_n^{(1)}$ contributes at second order. $\Delta_n^{(2)}$ enters only through $r^{(1)}_n$, as $w_n\Delta_n^{(2)}$. Taking $r^{(1)}_n$ with $\Delta_n^{(1)}$ moves this term into Eq. (2). Collecting the terms known from the first order into

$$k_n=P_n^{(1)}u^{(1)}+Q_n^{(1)}v^{(1)}+\big(\hat P_nv^{(1)}-\hat Q_nu^{(1)}+\hat vP_n^{(1)}-\hat uQ_n^{(1)}\big)\Delta_n^{(1)}-\tfrac12\big(\hat P_n\hat u+\hat Q_n\hat v\big)\big(\Delta_n^{(1)}\big)^2,$$

Eq. (2) becomes

$$r^{(2)}_n-k_n=\underbrace{\big(a^{(2)}+\hat P_nu^{(2)}+\hat Q_nv^{(2)}\big)}_{\text{pixel}}
+\underbrace{\hat uP_n^{(2)}+\hat vQ_n^{(2)}+w_n\Delta_n^{(2)}}_{\text{frame and step error}}+O(\Delta_n^3).\tag{3}$$

The first-order quantities in $r^{(1)}_n$ and $k_n$ are those of the first-order fit, `vp_aia.md` Eqs. (15) and (16).

## Second-order fit

Eq. (3) is `vp_aia.md` Eq. (9) with $r_n\to r^{(2)}_n-k_n$ and second-order unknowns. Unlike $r_n$, the left-hand side is not a pixel-step residual, so $\Pi$ does not leave it unchanged.

1. **Frame corrections and step error.** `vp_aia.md` Eqs. (11)–(15) hold with $r_n\to\big(\Pi(\mathbf r^{(2)}-\mathbf k)\big)_n$. $M$ and $C$ are unchanged; only $\rho$ is recomputed. This gives $P_n^{(2)},Q_n^{(2)},\alpha_{nj}^{(2)}$.
2. **Pixel corrections.** With Eq. (14) of `vp_aia.md` for $P_n^{(2)},Q_n^{(2)}$, the pixel step of Eq. (3) gives
$$\begin{pmatrix}a^{(2)}\\u^{(2)}\\v^{(2)}\end{pmatrix}=A_p^{-1}A^\top\begin{pmatrix}r^{(2)}_1-k_1-w_1\Delta_1^{(2)}\\\vdots\\r^{(2)}_N-k_N-w_N\Delta_N^{(2)}\end{pmatrix}.\tag{4}$$
3. **Corrected estimates.** $\tilde f=\hat f+f^{(1)}+f^{(2)}$ and $\tilde\Delta_n=\Delta_n^{(1)}+\Delta_n^{(2)}$, normalized by `vp_aia.md` Appendix D. Their error is $O(\Delta_n^3)$.
