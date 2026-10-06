from prism.agent import PrismAssessment
from prism.pipeline import PrismM1Pipeline


class FakeAssessor:
    def __call__(self, state):
        return PrismAssessment(
            catalyst_type="earnings",
            catalyst_summary="Quarterly results",
            catalyst_quality=80,
            event_probability=80,
            success_probability=80,
            fundamental_impact=80,
            surprise_edge=80,
            priced_in_edge=80,
            price_elasticity=80,
            positioning=80,
            market_regime=80,
            liquidity_risk=80,
        )


def test_pipeline_scores_and_freezes(tmp_path):
    pipeline = object.__new__(PrismM1Pipeline)
    from prism.scoring import PrismScoringEngine
    from prism.store import FrozenPredictionStore
    pipeline.assess = FakeAssessor()
    pipeline.scoring = PrismScoringEngine()
    pipeline.store = FrozenPredictionStore(tmp_path)

    prediction = pipeline.freeze_from_state({
        "company_of_interest": "NVDA",
        "trade_date": "2026-01-05",
    })

    assert prediction.prism_score == 80.0
    assert prediction.decision == "BUY"
    assert (tmp_path / f"{prediction.prediction_id}.json").exists()
