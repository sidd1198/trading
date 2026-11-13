# Algorithmic trading: learn by doing

Learning project. Everything stays in backtesting and paper trading. No real money.

## Rules of the project
- Data checks first, then a buy-and-hold baseline, then any strategy.
- Every strategy is compared against buy-and-hold, after transaction costs and slippage.
- Watch for look-ahead bias, survivorship bias, overfitting and tuning on the test period.
- Report negative and within-noise results honestly.

## Data conventions
- Daily bars. Price used is the ADJUSTED close (splits and dividends included).
- `^NSEI` is a price index (no dividends). `NIFTYBEES.NS` is an ETF you could actually trade.
- Downloaded data is cached in `data/raw/` and is NOT committed to git.

## Stages
- [x] Stage 0: setup and data checks (`scripts/00_check_data.py`). `^NSEI` passes. `NIFTYBEES.NS` FAILS
      (bad prices around 2010 and Dec 2019); investigate with `scripts/00b_crosscheck.py`.
- [ ] Stage 1: buy-and-hold baseline and metrics (`scripts/01_buy_and_hold.py`, runs only on data that passes)
- [ ] Stage 2: moving-average crossover with costs
- [ ] Stage 3: robustness (out-of-sample, walk-forward, parameter sensitivity)
- [ ] Stage 4: paper trading
- [ ] Stage 5: ML signal with time-ordered validation

## Run
    C:\ml\venv\Scripts\Activate.ps1
    python scripts\00_check_data.py
    python -m pytest -q
