## Project Status Board

- [x] Relaxed consolidation detection (bb_squeeze_factor, min_consolidation_bars, compliance_threshold)
- [x] Dynamic ATR multiplier for breakout confirmation
- [x] Tiered volume filter (1.1x/1.3x avg)
- [x] Hybrid retest logic (require_retest default False, retest_threshold_pct 0.3)
- [x] Entry timing adapts to volatility
- [x] Structural stop-loss logic
- [x] Adaptive take-profit logic
- [x] Liquidity check: volume z-score (skip signals if z-score < 1.0)
- [x] Dynamic minimum bars logic uses consolidation['dynamic_min_bars'] if present
- [x] BB squeeze lookback is now 20 by default
- [x] Volume confirmation is relaxed to 1.1x for BTCUSD/XAUUSD
- [ ] User to verify/test new behavior

## Executor's Feedback or Assistance Requests

- All requested relaxations and logic changes have been applied to breakout_trading_strategy.py as specified.
- Added a liquidity check using volume z-score (50-bar rolling window). If the z-score is less than 1.0, the signal is skipped and a debug message is logged.
- Dynamic minimum bars logic now uses the value from consolidation['dynamic_min_bars'] if present, otherwise falls back to the default.
- BB squeeze lookback is now 20 by default for faster squeeze detection.
- Volume confirmation is relaxed to 1.1x for BTCUSD/XAUUSD, making it easier to trigger signals for these volatile instruments.
- Please review and test these changes. If further tuning or bug fixes are needed, specify which aspect to address next.

### Summary of Changes
- Consolidation detection is less strict (bb_squeeze_factor=0.65, min_consolidation_bars=12, compliance_threshold=0.70).
- Breakout ATR threshold is now dynamic (0.10-0.15 ATR based on volatility).
- Volume filter is now tiered (1.1x for strong breakouts, 1.3x otherwise; 1.1x for BTCUSD/XAUUSD).
- Retest is not required for initial breakouts (require_retest=False by default, retest_threshold_pct=0.3).
- Entry timing (wait_for_confirmation_candle) is now adaptive to volatility.
- Stop-loss uses structural logic (min of breakout_low or entry-1.0*ATR for BUY, max of breakout_high or entry+1.0*ATR for SELL).
- Take-profit is adaptive to volatility (1.8x ATR for high, 2.5x ATR for low volatility).
- Liquidity check: signals require a tick_volume z-score >= 1.0 (50-bar window).
- BB squeeze lookback is now 20 by default.
- Dynamic minimum bars logic is now used for consolidation. 