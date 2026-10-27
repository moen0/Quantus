"""Strategy registry: experiments refer to strategies by name."""
from tiq.strategies.base import SIZING_DEFAULTS, Context, Strategy, size_positions
from tiq.strategies.benchmarks import BuyAndHold, VolTargetBuyAndHold
from tiq.strategies.meanrev import ZScoreReversion
from tiq.strategies.trend import DonchianEnsemble, MovingAverageCrossover, TimeSeriesMomentum

STRATEGIES: dict[str, Strategy] = {
    s.name: s for s in (
        DonchianEnsemble(), MovingAverageCrossover(), TimeSeriesMomentum(), ZScoreReversion(),
        BuyAndHold(), VolTargetBuyAndHold(),
    )
}


def get_strategy(name: str) -> Strategy:
    try:
        return STRATEGIES[name]
    except KeyError:
        raise KeyError(f"unknown strategy {name!r}; available: {sorted(STRATEGIES)}") from None


__all__ = ["STRATEGIES", "SIZING_DEFAULTS", "Context", "Strategy", "get_strategy", "size_positions"]
