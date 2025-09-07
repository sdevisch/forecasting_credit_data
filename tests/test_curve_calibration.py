from __future__ import annotations

import numpy as np

from credit_data.curve_calibration import hazards_from_cumulative, compute_scalers


def test_hazards_from_cumulative_monotone():
    cum = [0.0, 0.02, 0.05, 0.05, 0.10]
    haz = hazards_from_cumulative(cum)
    assert len(haz) == len(cum)
    assert np.all(haz >= 0.0)
    assert np.all(haz <= 1.0)


def test_compute_scalers_shape_and_bounds():
    model_haz = np.array([0.01, 0.02, 0.03])
    target_cum = [0.01, 0.03, 0.06]
    s = compute_scalers(model_haz, target_cum)
    assert s.shape == model_haz.shape
    assert np.all(s >= 0.1) and np.all(s <= 10.0)
