import numpy as np
import pytest

from lob_mvp.cli import _blend_probabilities
from lob_mvp.core import MVPConfig, assign_labels, build_features, generate_synthetic_events
from lob_mvp.model import RegimeAwareLOB, make_windows


def test_feature_pipeline_and_label_balance():
    cfg = MVPConfig()
    frame = build_features(generate_synthetic_events(count=400), cfg)
    frame = assign_labels(frame, frame["return_h50"].abs().quantile(0.7))
    assert len(frame) > cfg.window
    assert set(frame["label_h50"]).issubset({0, 1, 2})
    assert frame[cfg.feature_columns + cfg.regime_columns].isna().sum().sum() == 0


def test_model_shapes():
    cfg = MVPConfig(short_events=20, medium_bars=1, sample_stride=1)
    frame = assign_labels(build_features(generate_synthetic_events(count=400), cfg), 0.00001)
    x, xm, r, p, dt, y = make_windows(
        frame, cfg.feature_columns, cfg.regime_columns, cfg
    )
    model = RegimeAwareLOB(len(cfg.feature_columns), len(cfg.regime_columns), max_window=cfg.window)
    logits, embedding, gate = model(x[:2], xm[:2], r[:2], p[:2], dt[:2])
    assert logits.shape == (2, 3)
    assert embedding.shape[0] == 2 and gate.shape == (2, cfg.window)


def test_probability_blend():
    logistic = np.array([[0.2, 0.5, 0.3]])
    transformer = np.array([[0.4, 0.4, 0.2]])
    np.testing.assert_allclose(
        _blend_probabilities(logistic, transformer, 0.9),
        [[0.22, 0.49, 0.29]],
    )
    with pytest.raises(ValueError):
        _blend_probabilities(logistic, transformer, 1.1)
