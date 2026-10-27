"""tiq (thisIsMyQuant): research platform for systematic crypto strategies.

Layers, bottom to top:
    tiq.data        archive client, Parquet lake, quality checks, aligned panels, point-in-time universe
    tiq.strategies  causal indicators and strategies that turn a panel into target weights
    tiq.backtest    portfolio simulator with fees, spread and market impact; performance metrics
    tiq.stats       Sharpe inference, deflated Sharpe, PBO, bootstrap and SPA tests
    tiq.research    walk-forward validation, experiments driven by TOML files, reports, trial registry
    tiq.live        live trade collection into ClickHouse (optional extra)
"""

__version__ = "0.2.0"
