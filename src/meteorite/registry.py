"""Model registry helpers: champion / challenger promotion."""


def should_promote(new_rmse: float, champion_rmse: float | None, min_improvement: float = 0.0) -> bool:
    """Promote when there is no champion yet or the challenger is not worse.

    ``min_improvement`` is a relative margin: 0.02 means the challenger needs a 2 % lower RMSE.
    """
    if champion_rmse is None:
        return True
    return new_rmse <= champion_rmse * (1.0 - min_improvement)
