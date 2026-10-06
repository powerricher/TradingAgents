from dataclasses import FrozenInstanceError

import pytest

from prism.models import FrozenPrediction, PrismScoreComponents
from prism.scoring import PrismScoringEngine
from prism.store import FrozenPredictionStore


def sample_components(**overrides):
    values = {
        "catalyst_quality": 80,
        "event_probability": 80,
        "success_probability": 80,
        "fundamental_impact": 80,
        "surprise_edge": 80,
        "priced_in_edge": 80,
        "price_elasticity": 80,
        "positioning": 80,
        "market_regime": 80,
        "liquidity_risk": 80,
    }
    values.update(overrides)
    return PrismScoreComponents(**values)


def test_equal_components_produce_equal_weighted_score():
    engine = PrismScoringEngine()
    assert engine.score(sample_components()) == 80.0
    assert engine.decision(80.0) == "BUY"


def test_threshold_is_70():
    engine = PrismScoringEngine()
    assert engine.decision(70.0) == "BUY"
    assert engine.decision(69.99) == "PASS"


def test_component_range_is_enforced():
    with pytest.raises(ValueError):
        PrismScoringEngine().score(sample_components(catalyst_quality=101))


def test_prediction_is_immutable():
    prediction = FrozenPrediction(
        prediction_id="NVDA_2026-01-05_test",
        ticker="NVDA",
        analysis_date="2026-01-05",
        catalyst_type="test",
        catalyst_summary="test catalyst",
        components=sample_components(),
        prism_score=80.0,
        decision="BUY",
    )
    with pytest.raises(FrozenInstanceError):
        prediction.prism_score = 10.0


def test_store_is_write_once(tmp_path):
    prediction = FrozenPrediction(
        prediction_id="NVDA_2026-01-05_test",
        ticker="NVDA",
        analysis_date="2026-01-05",
        catalyst_type="test",
        catalyst_summary="test catalyst",
        components=sample_components(),
        prism_score=80.0,
        decision="BUY",
    )
    store = FrozenPredictionStore(tmp_path)
    store.freeze(prediction)
    with pytest.raises(FileExistsError):
        store.freeze(prediction)
