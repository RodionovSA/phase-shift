import numpy as np
import pytest

from simulator import circle, rectangle


class TestMasks:
    def test_circle_centered(self):
        m = circle(5, 5, radius=1.0)
        assert m.shape == (1, 5, 5) and m.dtype == bool
        expected = np.zeros((5, 5), bool)
        expected[2, 1:4] = expected[1:4, 2] = True
        np.testing.assert_array_equal(m[0], expected)

    def test_circle_offset_center(self):
        m = circle(4, 6, radius=0.5, center=(5, 0))
        assert m.sum() == 1 and m[0, 0, 5]

    def test_rectangle(self):
        m = rectangle(6, 8, width=4, height=2, center=(2, 3))
        expected = np.zeros((6, 8), bool)
        expected[2:5, 0:5] = True
        np.testing.assert_array_equal(m[0], expected)

    def test_combine(self):
        ring = circle(9, 9, radius=4) & ~circle(9, 9, radius=2)
        assert ring[0, 4, 4] == False and ring[0, 4, 7] == True  # noqa: E712

    @pytest.mark.parametrize("build", [
        lambda: circle(4, 4, radius=0),
        lambda: rectangle(4, 4, width=-1, height=2),
        lambda: circle(0, 4, radius=1),
    ])
    def test_rejects_bad_parameters(self, build):
        with pytest.raises(ValueError, match="must be positive"):
            build()
