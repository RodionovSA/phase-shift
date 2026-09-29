# Plans

## Quadrature-frame gauge breaks the VP-AIA ordering

**Problem.** The piston model $a+P_nu+Q_nv$ is invariant under a shift ($P_n\to P_n+p$, $Q_n\to Q_n+q$, $a\to a-pu-qv$) and any linear map of $(u,v)$ with the inverse on $(P_n,Q_n)$. `docs/vp_aia.md` Appendix D fixes these freedoms by conventions. With $\Delta_n\ne0$ the model gains $(P_nv-Q_nu)\sin\Delta_n$, and $w_n=P_nv-Q_nu$ is invariant only under rotation and scale; shift, shear, and anisotropic scale change it. The physical parameters generally do not satisfy the conventions, so the AIA estimates differ from them by a gauge offset that does not scale with $\Delta_n$, contrary to the ordering of `vp_aia.md` Appendix B. $w_n$ is then off by a fixed amount, and $\alpha_{nj}$ is biased by the same relative amount at any $\Delta_n$.

**Evidence.** Synthetic test (30×40 field, $N=7$, 5 polynomial modes): normalizing the true parameters changes $w_n$ by 0.4%. With physical truth, both the first- and second-order passes floor at about $2\times10^{-3}$ relative error in $\alpha$. With truth generated in the convention gauge, the errors scale as $\Delta^2$ and $\Delta^3$ as derived.

**Potential fix.** Of the six directions fixed by `vp_aia.md` Eq. (14), keep only rotation and scale fixed, and fit the other four (shift, shear, anisotropic scale), which the data determine through $w_n\Delta_n$. These unknowns multiply $\Delta_n$, so the fit becomes bilinear. The mathematics must be settled before any change to `docs/vp_aia.md` or the code.

**Test of the fix.** Same synthetic case with physical truth. The four gauge parameters were applied to the normalized AIA baseline and chosen by Gauss–Newton on the full-model misfit after the second-order pass (an outer loop that tests the principle only, not the bilinear fit). Relative $\alpha$ error after the second-order pass:

| max $\Delta_n$ (rad) | conventions | fitted gauge |
|---|---|---|
| 0.2 | 1.8e-3 | 2.3e-3 |
| 0.1 | 1.8e-3 | 3.4e-4 |
| 0.05 | 2.0e-3 | 8.8e-5 |
| 0.025 | 2.0e-3 | 2.3e-5 |

With the fitted gauge the floor disappears and the misfit falls ×8 per halving of $\Delta_n$, the third-order rate of the convention-gauge truth. At 0.2 rad the third-order error dominates.
