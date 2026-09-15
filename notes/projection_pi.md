# The projection $\Pi$, step by step

Companion note to `docs/vp_aia.md`, Eqs. (17)–(19). Everything below happens at **one pixel**, and all quantities are zeroth-order AIA estimates (superscripts $(0)$ dropped) unless marked with $(1)$.

Notation: at one pixel, a *column* is a list of $N$ values, one per frame:

$$\mathbf 1=\begin{pmatrix}1\\\vdots\\1\end{pmatrix},\quad
\mathbf P=\begin{pmatrix}P_1\\\vdots\\P_N\end{pmatrix},\quad
\mathbf Q=\begin{pmatrix}Q_1\\\vdots\\Q_N\end{pmatrix},\quad
\mathbf r=\begin{pmatrix}r_1\\\vdots\\r_N\end{pmatrix},\quad
A=\begin{pmatrix}\mathbf 1&\mathbf P&\mathbf Q\end{pmatrix}.$$

The three columns $\mathbf 1,\mathbf P,\mathbf Q$ depend only on the frames, so they are the same at every pixel.

---

## Step 1. The pixel step splits any column into two parts

Take any column $\mathbf y$. The pixel step (Eq. 8) finds the numbers $a,u,v$ that best fit it:

$$\begin{pmatrix}a\\u\\v\end{pmatrix}=A_p^{-1}A^\top\mathbf y,\qquad A_p=A^\top A.$$

The fitted column is $a\mathbf 1+u\mathbf P+v\mathbf Q=A(a,u,v)^\top$:

$$\text{fitted part}=AA_p^{-1}A^\top\mathbf y\equiv F\mathbf y.$$

What is left over:

$$\text{leftover}=\mathbf y-F\mathbf y=(I-F)\mathbf y\equiv\Pi\mathbf y.$$

So every column splits as

$$\boxed{\mathbf y=F\mathbf y+\Pi\mathbf y}\qquad\text{(what the pixel step can explain + what it cannot).}$$

$F$ and $\Pi$ are $N\times N$ matrices built only from $P_n,Q_n$, so they are the same at every pixel.

---

## Step 2. Four properties (each is one line of algebra)

**(a) Anything built from $\mathbf 1,\mathbf P,\mathbf Q$ is removed completely.**

$$\Pi A=A-AA_p^{-1}(A^\top A)=A-A=0.$$

So $\Pi\mathbf 1=\Pi\mathbf P=\Pi\mathbf Q=0$, and for **any** numbers $a,u,v$:

$$\Pi\big(a\mathbf 1+u\mathbf P+v\mathbf Q\big)=0.$$

**(b) The leftover has nothing along $\mathbf 1,\mathbf P,\mathbf Q$.**

$$A^\top\Pi=A^\top-(A^\top A)A_p^{-1}A^\top=0,$$

that is, $\sum_n(\Pi\mathbf y)_n=0$, $\sum_nP_n(\Pi\mathbf y)_n=0$, $\sum_nQ_n(\Pi\mathbf y)_n=0$. These are exactly the pixel-step normal equations: the leftover cannot be fitted any further.

**(c) A column that already has nothing along $\mathbf 1,\mathbf P,\mathbf Q$ is not changed.**

If $A^\top\mathbf y=0$, then $\Pi\mathbf y=\mathbf y-AA_p^{-1}(A^\top\mathbf y)=\mathbf y$.

**(d) Applying $\Pi$ twice changes nothing, and $\Pi$ is symmetric.**

$\Pi\Pi=\Pi-AA_p^{-1}(A^\top\Pi)=\Pi$ by (b), and $\Pi^\top=\Pi$ because $F^\top=F$.

---

## Step 3. A worked example with $N=4$

Take uniform steps $\delta_n=0,\ \pi/2,\ \pi,\ 3\pi/2$ and $g_n=1$:

$$\mathbf P=\begin{pmatrix}1\\0\\-1\\0\end{pmatrix},\qquad
\mathbf Q=\begin{pmatrix}0\\1\\0\\-1\end{pmatrix},\qquad
A_p=A^\top A=\begin{pmatrix}4&0&0\\0&2&0\\0&0&2\end{pmatrix}.$$

Because $A_p$ is diagonal, $F=\tfrac14\mathbf 1\mathbf 1^\top+\tfrac12\mathbf P\mathbf P^\top+\tfrac12\mathbf Q\mathbf Q^\top$:

$$F=\frac14\begin{pmatrix}3&1&-1&1\\1&3&1&-1\\-1&1&3&1\\1&-1&1&3\end{pmatrix},\qquad
\Pi=I-F=\frac14\begin{pmatrix}1&-1&1&-1\\-1&1&-1&1\\1&-1&1&-1\\-1&1&-1&1\end{pmatrix}=\frac14\,\mathbf c\,\mathbf c^\top,\quad
\mathbf c=\begin{pmatrix}1\\-1\\1\\-1\end{pmatrix}.$$

So for $N=4$:

$$\Pi\mathbf y=\frac{y_1-y_2+y_3-y_4}{4}\,\mathbf c.$$

Only **one number per pixel** survives, the alternating combination. Four values minus three fitted numbers leaves one. In general, $N$ values minus 3 leaves $N-3$.

Check of property (a): $\mathbf c^\top\mathbf 1=0$, $\mathbf c^\top\mathbf P=1-1=0$, $\mathbf c^\top\mathbf Q=-1+1=0$.

**Numbers.** Take $\mathbf y=(5,2,1,4)^\top$. Then $y_1-y_2+y_3-y_4=0$, so $\Pi\mathbf y=0$: the pixel step explains it exactly with $a=3,\ u=2,\ v=-1$ (frame 1: $3+2=5$; frame 2: $3-1=2$; frame 3: $3-2=1$; frame 4: $3+1=4$).

Change one value, $\mathbf y=(6,2,1,4)^\top$. Now $y_1-y_2+y_3-y_4=1$:

$$\Pi\mathbf y=\begin{pmatrix}\tfrac14\\-\tfrac14\\\tfrac14\\-\tfrac14\end{pmatrix},\qquad
F\mathbf y=\mathbf y-\Pi\mathbf y=\begin{pmatrix}5.75\\2.25\\0.75\\4.25\end{pmatrix}=3.25\,\mathbf 1+2.5\,\mathbf P-1\,\mathbf Q.$$

The extra $+1$ in frame 1 was split: $\tfrac34$ of it went into $a,u,v$ (it was "absorbed"), and only the alternating remainder is visible in the residual.

---

## Step 4. What AIA does with a phase-step error

At one pixel, to first order and for now with exact $P_n,Q_n$, the data are

$$\mathbf y=A\begin{pmatrix}a\\u\\v\end{pmatrix}+\mathbf e,\qquad
\mathbf e=\begin{pmatrix}w_1\Delta_1\\\vdots\\w_N\Delta_N\end{pmatrix},\qquad w_n=P_nv-Q_nu.$$

The pixel step returns

$$\begin{pmatrix}\hat a\\\hat u\\\hat v\end{pmatrix}=A_p^{-1}A^\top\mathbf y
=\begin{pmatrix}a\\u\\v\end{pmatrix}+A_p^{-1}A^\top\mathbf e,$$

so the AIA estimates are **shifted** by $A_p^{-1}A^\top\mathbf e$. The residual is

$$\mathbf r=\mathbf y-A\begin{pmatrix}\hat a\\\hat u\\\hat v\end{pmatrix}=\mathbf e-F\mathbf e=\Pi\mathbf e.$$

The phase-step signal is split exactly as in Step 1:

$$\mathbf e=\underbrace{F\mathbf e}_{\text{hidden inside }\hat a,\hat u,\hat v}+\underbrace{\Pi\mathbf e}_{\text{visible in }\mathbf r}.$$

For $N=4$, the residual keeps only $(e_1-e_2+e_3-e_4)/4$; the other three numbers of $\mathbf e$ are hidden.

---

## Step 5. From Eq. (17) to Eq. (19)

Eq. (17) at one pixel, written as columns (here $a^{(1)},u^{(1)},v^{(1)},u,v$ are numbers at this pixel, and $\mathbf P^{(1)},\mathbf Q^{(1)}$ are columns over frames):

$$\mathbf r=A\begin{pmatrix}a^{(1)}\\u^{(1)}\\v^{(1)}\end{pmatrix}+u\,\mathbf P^{(1)}+v\,\mathbf Q^{(1)}+\mathbf e,\qquad
e_n=w_n\sum_j\alpha_{nj}H_j.$$

Multiply both sides by $\Pi$:

- **Left side:** $\mathbf r$ comes from a pixel step, so $A^\top\mathbf r=0$ (property b) and $\Pi\mathbf r=\mathbf r$ (property c).
- **First term on the right:** $\Pi A=0$ (property a), so it disappears, whatever $a^{(1)},u^{(1)},v^{(1)}$ are.

Result:

$$\mathbf r=\Pi\big(u\,\mathbf P^{(1)}+v\,\mathbf Q^{(1)}+\mathbf e\big),$$

which is Eq. (19). The $3K$ pixel corrections are gone without solving for them. Once $\mathbf P^{(1)},\mathbf Q^{(1)},\alpha$ are known, they follow from a $3\times3$ pixel step on the remaining signal.

---

## Step 6. Why a fit with frozen parameters recovers only part of $\alpha$

For clarity, drop $\mathbf P^{(1)},\mathbf Q^{(1)}$ and keep one mode, $\Delta_n(x,y)=\alpha_nH(x,y)$. At pixel $k$:

$$e_n(k)=w_n(k)H(k)\,\alpha_n,\qquad r_n(k)=\sum_m\Pi_{nm}\,w_m(k)H(k)\,\alpha_m.$$

**Frozen fit** (per frame, pretending $\mathbf r=\mathbf e$): minimize $\sum_k\big(r_n(k)-w_n(k)H(k)\alpha_n\big)^2$ for each $n$:

$$\alpha_n^{\text{frozen}}=\frac{\sum_kw_nHr_n}{\sum_kw_n^2H^2}.$$

Substitute $r_n$ and write $S_{nm}=\sum_kw_n(k)w_m(k)H(k)^2$:

$$\alpha_n^{\text{frozen}}=\frac{1}{S_{nn}}\sum_m\Pi_{nm}S_{nm}\,\alpha_m.$$

If nothing were absorbed ($\Pi=I$), this would give $\alpha_n$. It does not, because $\Pi\ne I$.

**Size of the error.** Take uniform steps, $g_n=1$, and many fringes.

*Averaging over fringes.* $w_n=-b\sin(\Phi+\delta_n)$, so

$$w_nw_m=\frac{b^2}{2}\Big[\cos(\delta_n-\delta_m)-\cos(2\Phi+\delta_n+\delta_m)\Big].$$

The second term oscillates with the fringes and averages out over pixels, so $S_{nm}\approx S\cos(\delta_n-\delta_m)$ with $S=\sum_kb^2H^2/2$.

*Uniform steps.* $A_p=\operatorname{diag}(N,N/2,N/2)$, so

$$F_{nm}=\frac1N\big[1+2\cos(\delta_n-\delta_m)\big],\qquad
\Pi_{nm}=\delta_{nm}-\frac1N\big[1+2\cos(\delta_n-\delta_m)\big].$$

(For $N=4$ this reproduces the matrices of Step 3.)

*Combine.* Using $2\cos^2x=1+\cos2x$ and $\sum_m\alpha_m=0$:

$$\alpha_n^{\text{frozen}}=\alpha_n-\frac1N\sum_m\Big[\cos(\delta_n-\delta_m)+\cos\big(2(\delta_n-\delta_m)\big)\Big]\alpha_m.$$

*Evaluate by temporal harmonic.* The only fact needed is $\sum_{m}\cos(c-h\delta_m)=0$ unless $h$ is a multiple of $N$.

- **First harmonic**, $\alpha_m=\cos\delta_m$:
  $\frac1N\sum_m\cos(\delta_n-\delta_m)\cos\delta_m=\frac1{2N}\sum_m[\cos\delta_n+\cos(\delta_n-2\delta_m)]=\tfrac12\cos\delta_n$, and the $\cos2(\cdot)$ sum vanishes for $N\ge4$.
  So $\alpha_n^{\text{frozen}}=\tfrac12\alpha_n$.
- **Second harmonic**, $\alpha_m=\cos2\delta_m$: the $\cos(\cdot)$ sum vanishes, and
  $\frac1N\sum_m\cos(2\delta_n-2\delta_m)\cos2\delta_m=\tfrac12\cos2\delta_n$ for $N\ge5$.
  So $\alpha_n^{\text{frozen}}=\tfrac12\alpha_n$. For $N=4$ an extra $\tfrac12$ appears and $\alpha_n^{\text{frozen}}=0$.
- **Higher harmonics** (not equal to $\pm1,\pm2$ modulo $N$): both sums vanish, so $\alpha_n^{\text{frozen}}=\alpha_n$.

The first harmonic of the phase error is half hidden in $\hat a$, and the second in $\hat u,\hat v$. The simulation gave ratios $0.52$ and $0.51$ for $N=5$, and $0.48$, $0.50$, $0.99$ for $N=8$.

---

## Step 7. The projected fit recovers $\alpha$ exactly

Fit the residual with the **projected** model instead, as Eq. (19) prescribes. Minimize

$$\sum_k\big\|\mathbf r(k)-\Pi\,\mathbf e(k)\big\|^2,\qquad e_n(k)=w_n(k)H(k)\alpha_n.$$

Write $\mathbf e(k)=D(k)\,\boldsymbol\alpha$ with $D(k)=\operatorname{diag}\big(w_1(k)H(k),\dots,w_N(k)H(k)\big)$. Setting the derivative to zero and using $\Pi^\top\Pi=\Pi$ (property d):

$$\sum_kD\,\Pi\,D\;\boldsymbol\alpha^{\text{proj}}=\sum_kD\,\Pi\,\mathbf r=\sum_kD\,\mathbf r\qquad(\Pi\mathbf r=\mathbf r).$$

Entry by entry:

$$\sum_m\Pi_{nm}S_{nm}\,\alpha_m^{\text{proj}}=\sum_kw_nHr_n.$$

From Step 6, the right side is exactly $\sum_m\Pi_{nm}S_{nm}\,\alpha_m$ (true values). Therefore

$$\sum_m\Pi_{nm}S_{nm}\,\alpha_m^{\text{proj}}=\sum_m\Pi_{nm}S_{nm}\,\alpha_m\quad\Longrightarrow\quad\boldsymbol\alpha^{\text{proj}}=\boldsymbol\alpha$$

whenever the $N\times N$ matrix $[\Pi_{nm}S_{nm}]$ is invertible on zero-mean $\boldsymbol\alpha$. This step uses **no** many-fringe or uniform-step approximation.

**The whole difference in one line.** Both fits have the same right side, $\sum_kw_nHr_n$. The frozen fit divides by the diagonal $S_{nn}$; the projected fit solves with the full matrix $\Pi_{nm}S_{nm}$.

**Why $N=4$ fails.** Take $\boldsymbol\alpha=\mathbf c=(1,-1,1,-1)^\top$. Since $c_n^2=1$,

$$\mathbf c^\top\mathbf e=H\sum_nw_n=H\Big(v\sum_nP_n-u\sum_nQ_n\Big)=0,$$

so $\Pi\mathbf e=0$ at every pixel. This phase-error pattern leaves no trace in the residual, and $[\Pi_{nm}S_{nm}]$ is singular in that direction.

---

## Summary

1. $\Pi$ = "subtract what the pixel step can fit." It is the same $N\times N$ matrix at every pixel.
2. The residual is already a leftover: $\Pi\mathbf r=\mathbf r$.
3. The pixel corrections are something the pixel step can fit, so $\Pi$ removes them.
4. AIA hides $F\mathbf e$ inside $\hat a,\hat u,\hat v$; only $\Pi\mathbf e$ is visible. So the residual must be compared with $\Pi\mathbf e$, not with $\mathbf e$.
5. A frozen fit compares with $\mathbf e$ and loses about half of the first and second temporal harmonics of $\alpha$. The projected fit compares with $\Pi\mathbf e$ and recovers $\alpha$ exactly.
6. Each pixel keeps $N-3$ independent numbers. This is where $N-3$ in $J_{\max}$ comes from, and why $N=4$ (one number per pixel) is not enough.
