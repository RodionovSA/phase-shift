# tests/conftest.py
"""Run tests on NumPy unless marked ``gpu``; ``gpu`` tests need a working CuPy."""

import pytest

from phase_shift import backend


@pytest.fixture(autouse=True)
def _device(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("gpu") is None:
        monkeypatch.setattr(backend, "CUPY_AVAILABLE", False)
    elif not backend.CUPY_AVAILABLE:
        pytest.skip("cupy is not installed")
