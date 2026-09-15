# Full-rank test for the first-order system

Equation (20) defines a linear map from the first-order parameter corrections to
the measured intensity corrections. Index pixels by $p=1,\ldots,K$ and spatial
modes by $\ell=1,\ldots,J$. Define

$$s_{np}=P_n^{(0)}v_p^{(0)}-Q_n^{(0)}u_p^{(0)}.\tag{1}$$

For observation $(n,p)$, the row of the design matrix $\mathcal M$ has these
entries; all unlisted entries are zero:

$$
\begin{array}{c|c}
\text{unknown correction} & \text{entry in row }(n,p)\\
\hline
a_p^{(1)} & 1\\
u_p^{(1)} & P_n^{(0)}\\
v_p^{(1)} & Q_n^{(0)}\\
P_n^{(1)} & u_p^{(0)}\\
Q_n^{(1)} & v_p^{(0)}\\
\alpha_{n\ell}^{(1)} & s_{np}H_\ell(p)
\end{array}
$$

If $z$ stacks the $3K+2N+NJ$ unknown corrections, Eq. (20) is

$$\Delta I=\mathcal Mz.\tag{2}$$

Build a constraint matrix $C$ for the coefficient means and phase origin,

$$\sum_{n=1}^N\alpha_{n\ell}^{(1)}=0\quad(\ell=1,\ldots,J),
\qquad Q_1^{(1)}=0.\tag{3}$$

The constrained first-order solution is unique exactly when

$$
\operatorname{rank}\begin{pmatrix}\mathcal M\\C\end{pmatrix}
=3K+2N+NJ.\tag{4}
$$

Equivalently, let $Z$ contain a basis for $\ker C$. Then check

$$
\operatorname{rank}(\mathcal MZ)=3K+2N+NJ-J-1.\tag{5}
$$

In numerical work, compute the rank from an SVD using a relative tolerance. The
degree-of-freedom bound in the main document is only necessary; this test uses
the actual zeroth-order fields, frame coefficients, and chosen spatial
functions.
