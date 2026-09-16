"""Recover phase from a synthetic stack: uv run python examples/synthetic_stack.py."""

import numpy as np

from phase_shift import PhaseConfig, PhaseSolver
from phase_shift.utils import wrap


def main():
    y, x = np.mgrid[:48, :64]
    phase = 0.31 * x + 0.19 * y + 0.4 * np.sin(x / 11.0)
    steps = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    stack = 2.0 + np.cos(phase[None] + steps[:, None, None])
    solver = PhaseSolver(
        PhaseConfig(gain_mode="none", use_alpha=False), device="cpu"
    ).fit(stack.astype(np.float32))

    # Blind recovery has a global sign and piston ambiguity. Compare both
    # branches after aligning their constant offset to the known truth.
    errors = []
    for sign in (1, -1):
        difference = wrap(sign * solver.result.phi - phase)
        piston = np.angle(np.mean(np.exp(1j * difference)))
        errors.append(np.sqrt(np.mean(wrap(difference - piston) ** 2)))
    print(f"Phase shape: {solver.result.phi.shape}")
    print(f"Reconstruction RMSE: {solver.result.reconstruction_error:.6g}")
    print(f"Aligned phase RMS: {min(errors):.6g} rad")


if __name__ == "__main__":
    main()
