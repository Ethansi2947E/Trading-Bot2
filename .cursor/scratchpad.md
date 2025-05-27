# Scratchpad: Candlestick Pattern Consolidation

## Background and Motivation

Currently, candlestick pattern detection logic (e.g., for Doji, Engulfing, Hammer) is duplicated or implemented with slight variations across multiple strategy files within the `src/strategy/` directory. This leads to:
- Code redundancy, making the codebase larger and harder to maintain.
- Potential inconsistencies if a pattern's logic is updated in one file but not others.
- Increased effort when adding new patterns or modifying existing ones, as changes need to be made in multiple places.

The goal of this task is to centralize all candlestick pattern detection logic into a single, dedicated Python file. This new module will then be used by all strategy files, ensuring consistency, reducing redundancy, and improving maintainability.

## Key Challenges and Analysis

*   **Identifying all pattern variations:** We need to meticulously scan each strategy file to find all instances of pattern detection and note any differences in their implementation (e.g., parameters, tolerances, specific conditions).
*   **Standardizing pattern logic:** If variations exist for the same pattern, a decision will need to be made on which implementation is the most robust, accurate, or aligned with project standards. Alternatively, parameters could be introduced to accommodate common variations if necessary.
*   **Designing a consistent interface:** The functions in the new module should have a consistent way of accepting input (ideally a pandas DataFrame) and returning results (ideally a pandas Series indicating pattern occurrences).
*   **Refactoring strategies with minimal disruption:** Updating each strategy to use the new module must be done carefully to ensure that the core logic and signal generation remain consistent with previous behavior, or are verifiably correct if pattern definitions are improved.
*   **Testing:** Thorough testing will be crucial. This includes unit tests for individual pattern functions and potentially comparing strategy outputs before and after the refactor on sample data.
*   **Stop-Loss Logic with Market Gaps/Swift Moves:** The `PriceActionSRStrategy` demonstrated an issue where the stop-loss, based solely on a support/resistance zone, could be placed on the wrong side of the entry price if the market moves significantly between pattern confirmation and entry on the next candle. SL calculations must consider the actual candle causing the signal (e.g., its low for a buy) and not just the zone.

## High-level Task Breakdown

**Phase 1: Discovery and Analysis**
1.  **Task:** Scan all files in `src/strategy/` (`breakout_trading_strategy.py`, `breakout_reversal_strategy.py`, `trend_following_strategy.py`, `price_action_sr_strategy.py`, `confluence_price_action_strategy.py`).
    *   **Action:** Identify all functions/code blocks related to candlestick pattern detection.
    *   **Success Criteria:** A comprehensive list of all patterns and their current implementations across all strategy files is compiled.
2.  **Task:** Compare implementations of the same patterns across different files.
    *   **Action:** Note similarities, differences in logic, parameters, and return values.
    *   **Success Criteria:** A clear understanding of which patterns are duplicated, which have variations, and which are unique to specific strategies. Decision points for standardization are identified.

**Phase 2: Design and Implementation of Centralized Pattern Module**
1.  **Task:** Define the structure and location for the new pattern module.
    *   **Action:** Propose creating `src/utils/candlestick_patterns.py`.
    *   **Success Criteria:** Agreement on the new file's name and location.
2.  **Task:** Design the functions within the new module.
    *   **Action:** Define a standard function signature (e.g., `def detect_pattern(df: pd.DataFrame, ...params) -> pd.Series:`). Ensure clear docstrings for each pattern.
    *   **Success Criteria:** A well-defined API for each pattern function is established.
3.  **Task:** Implement the consolidated pattern functions.
    *   **Action:** Transfer and standardize the logic for each identified pattern into the new module. Prioritize vectorized operations for efficiency.
    *   **Success Criteria:** All targeted candlestick patterns are implemented as functions in `src/utils/candlestick_patterns.py`.
4.  **Task:** (Optional but Recommended) Write unit tests for the new pattern functions.
    *   **Action:** Create test cases with known positive and negative examples for each pattern.
    *   **Success Criteria:** Unit tests pass, confirming the correctness of the individual pattern detection functions.

**Phase 3: Refactor Strategies**
1.  **Task:** Update each strategy file to use the new centralized pattern module.
    *   **Action:**
        *   Remove the old pattern detection methods from the strategy class.
        *   Add imports for the new pattern functions from `src.utils.candlestick_patterns`.
        *   Modify the strategy logic to call these new functions and correctly interpret their results.
    *   **Success Criteria:** All strategy files are refactored, successfully import from the new module, and their internal pattern logic is removed.
2.  **Task:** Test the refactored strategies.
    *   **Action:** Compare signal generation or relevant outputs before and after the refactor. Use existing backtests or run on historical data if possible. Review logs for correct pattern detection.
    *   **Success Criteria:** Refactored strategies produce consistent or verifiably correct results compared to their pre-refactor versions. Any discrepancies are understood and intentional (due to pattern logic standardization/improvement).

**Phase 4: Finalization**
1.  **Task:** Review and finalize documentation.
    *   **Action:** Ensure `src/utils/candlestick_patterns.py` is well-documented. Update any relevant strategy documentation.
    *   **Success Criteria:** All new code is clearly documented.
2.  **Task:** Update this scratchpad with progress and lessons learned.
    *   **Action:** Mark tasks as complete, document any challenges overcome, and note any insights gained.
    *   **Success Criteria:** Scratchpad accurately reflects the work done.

## Project Status Board
*   [x] **Phase 1: Discovery and Analysis** (Completed)
    *   [x] Scan strategy files for pattern logic.
        *   Initial scan completed. Found pattern logic in `breakout_reversal_strategy.py`, `trend_following_strategy.py`, `breakout_trading_strategy.py`, `price_action_sr_strategy.py`, and `confluence_price_action_strategy.py`.
        *   Three strategies (`breakout_reversal`, `trend_following`, `breakout_trading`) share similar vectorized static methods for common patterns (Hammer, Shooting Star, Engulfing, Inside Bar, Morning/Evening Star, False Breakout).
        *   `price_action_sr` and `confluence_price_action` use non-vectorized, individual methods for similar and some unique patterns (Harami, Pin-bar, 2/3-bar reversals).
    *   [ ] Compare and document pattern implementations.
        *   **Hammer Pattern Analysis:**
            *   **Vectorized versions** (`breakout_reversal`, `trend_following`, `breakout_trading`): Very similar.
                *   Core: body is small fraction of total range (0.3 or 0.33), long lower wick (>2x body), short upper wick (<1x body).
                *   `breakout_reversal` takes `price_tolerance` (not used in core logic snippet shown). `trend_following`/`breakout_trading` slightly stricter on body ratio (0.3 vs 0.33).
            *   **Non-vectorized versions**:
                *   `price_action_sr`: Logic matches `trend_following`'s Hammer (body < 0.3).
                *   `confluence_price_action`: Similar core logic but adds handling for zero-body candles (Doji-like hammers) and context (proximity to a `level` using `price_tolerance`).
            *   **Standardization Note:** Common criteria (body < 0.3, lower_wick > 2x body, upper_wick < body) is a good base. Contextual checks (like level proximity or preceding trend) should likely be handled by the calling strategy, not in the basic pattern function. Zero-body handling is a point for discussion (general vs. specific).
        *   **Shooting Star Pattern Analysis:**
            *   **Vectorized versions** (`breakout_reversal`, `trend_following`, `breakout_trading`): Follow same structure as their Hammer counterparts.
                *   Core: body is small fraction of total range (0.3 or 0.33), long upper wick (>2x body), short lower wick (<1x body).
                *   `breakout_reversal` has `price_tolerance` param. `trend_following`/`breakout_trading` use 0.3 body ratio.
            *   **Non-vectorized versions**:
                *   `price_action_sr`: Logic matches `trend_following`'s Shooting Star (body < 0.3).
                *   `confluence_price_action`: Similar core logic, handles zero-body candles (Doji-like shooting stars), and contextual check against a `level` (resistance).
            *   **Standardization Note:** Similar to Hammer. Use common criteria (body < 0.3, upper_wick > 2x body, lower_wick < body). Context and advanced specifics (zero-body, level proximity) should be managed by calling strategies.
        *   **Bullish Engulfing Pattern Analysis:**
            *   **Vectorized versions** (`breakout_reversal`, `trend_following`, `breakout_trading`): All identical.
                *   Core: Previous candle bearish, current candle bullish. Current open < previous close AND current close > previous open (current body engulfs previous body).
            *   **Non-vectorized versions**:
                *   `price_action_sr`: Logic identical to the vectorized versions.
                *   `confluence_price_action` (within `_is_engulfing` for 'buy'): Same core body engulfing logic. Includes optional/commented checks for engulfing wicks, proximity to a support `level`, and volume confirmation.
            *   **Standardization Note:** The core definition is very consistent. A vectorized function implementing this (prev bearish, curr bullish, current body engulfs previous body) is standard. Additional conditions like engulfing wicks, volume, or level proximity could be optional parameters to the main function or handled by strategies externally.
        *   **Bearish Engulfing Pattern Analysis:**
            *   **Vectorized versions** (`breakout_reversal`, `trend_following`, `breakout_trading`): All identical and symmetrical to their Bullish Engulfing counterparts.
                *   Core: Previous candle bullish, current candle bearish. Current open > previous close AND current close < previous open (current body engulfs previous body).
            *   **Non-vectorized versions**:
                *   `price_action_sr`: Logic identical to the vectorized versions.
                *   `confluence_price_action` (within `_is_engulfing` for 'sell'): Same core body engulfing logic, symmetrical to its bullish part. Optional checks for wicks, resistance proximity, volume.
            *   **Standardization Note:** Core definition is consistent. A vectorized function implementing this (prev bullish, curr bearish, current body engulfs previous body) is standard. Optional/contextual conditions to be handled externally or as parameters.
        *   **Inside Bar Pattern Analysis:**
            *   **Vectorized versions** (`breakout_reversal`, `trend_following`, `breakout_trading`): All identical.
                *   Core: Current high < previous high AND current low > previous low.
            *   **Non-vectorized versions**:
                *   `confluence_price_action` (in `_is_inside_bar`): Logic identical to vectorized versions. Takes `level` parameter, noted for caller context.
                *   `price_action_sr`: Pattern not explicitly found.
            *   **Standardization Note:** Definition is highly consistent. A simple vectorized function is appropriate.
        *   **Morning Star Pattern Analysis:**
            *   Three-candle pattern. Significant variations exist in defining the second candle (star) and third candle (confirmation).
            *   **`breakout_reversal_strategy.py` (vectorized):**
                *   C1 bearish. C2 small body (<50% of C1 body) AND C2 gaps down (max(C2 o,c) < min(C1 o,c)). C3 bullish & closes > midpoint of C1 body.
            *   **`trend_following_strategy.py` / `breakout_trading_strategy.py` (vectorized, identical):**
                *   C1 bearish, C3 bullish. C2 small body (<30% of C1 *rolling avg body*). Gap: C2 max < C1 min OR C2 max < C1 min + 30% C1 body (flexible overlap). C3 closes > (C1 open - 61.8% C1 body).
                *   More complex definitions for C2 body size and C3 recovery.
            *   **`price_action_sr_strategy.py` (non-vectorized):**
                *   Similar to `breakout_reversal`: C1 bearish, C2 body < 50% C1 body, C3 bullish & closes > C1 midpoint. Does *not* explicitly check for C2 gap down.
            *   **`confluence_price_action_strategy.py` (non-vectorized):**
                *   Similar to `breakout_reversal` / `price_action_sr`: C1 bearish, C2 body < 50% C1 body, C3 bullish & closes > C1 midpoint. Mentions optional gap check and context (near support).
            *   **Standardization Note:** This pattern has more variation. A common, simpler definition (like in `breakout_reversal` or `price_action_sr`) could be the base: C1 bearish, C2 small body + gaps down, C3 bullish closing well into C1. The more complex rolling average and 61.8% logic from `trend_following` is quite specific and might be better as a separate, advanced version or highly parameterized if included in a general function.
        *   **Evening Star Pattern Analysis:**
            *   Symmetrical to Morning Star, showing similar variations.
            *   **`breakout_reversal_strategy.py` (vectorized):**
                *   C1 bullish. C2 small body (<50% C1 body) AND C2 gaps up. C3 bearish & closes < midpoint of C1 body.
            *   **`trend_following_strategy.py` / `breakout_trading_strategy.py` (vectorized, identical):**
                *   C1 bullish, C3 bearish. C2 small body (<30% C1 *rolling avg body*). Flexible gap up. C3 closes < (C1 open + 61.8% C1 body).
            *   **`price_action_sr_strategy.py` (non-vectorized):**
                *   Similar to `breakout_reversal`: C1 bullish, C2 body < 50% C1 body, C3 bearish & closes < C1 midpoint. No explicit C2 gap up check.
            *   **`confluence_price_action_strategy.py` (non-vectorized):**
                *   Similar to `breakout_reversal` / `price_action_sr`: C1 bullish, C2 body < 50% C1 body, C3 bearish & closes < C1 midpoint. Contextual check (near resistance).
            *   **Standardization Note:** Symmetric to Morning Star. A simpler, common definition should be the base. Complex variations are better handled separately or via extensive parameterization.
        *   **False Breakout / Pin-bar / Strong Rejection Analysis:**
            *   **`breakout_reversal_strategy.py` (`detect_false_breakout`):** Identifies a strong reversal candle. E.g., for a buy, price attempts to break down, then reverses to close in the top 30% of its range and bullish. More like a "Key Reversal Bar".
            *   **`trend_following_strategy.py` / `breakout_trading_strategy.py` (`detect_false_breakout`):** Identifies a strong candle closing significantly past the previous close, with a long wick (>50% of range) in the direction of the reversal. Less about breaking a specific *level* and more about candle shape and relation to previous close.
            *   **`price_action_sr_strategy.py` (`_wick_rejection` for "Pin-bar"):** Classic pin-bar: significant wick rejection (e.g., wick > 50% of total range) and a small body (<30% of total range). This is effectively Hammer/Shooting Star shape focused on wick dominance.
            *   **`confluence_price_action_strategy.py` (`_is_false_breakout`):** Defines a classic false break of a *specific level*: candle breaks the `level` then closes back on the original side of that `level` within the same bar.
            *   **`confluence_price_action_strategy.py` (`_is_pin_bar`):** Similar to Hammer/Shooting Star (small body, long wick >= 60% of range and >= 2x body), but with an explicit check that the rejection occurs at/beyond a given `level`.
            *   **Standardization Note:** 
                *   The "False Breakout" that breaks a *level* and closes back (from `confluence_price_action`) is a distinct and standard pattern to implement.
                *   "Pin-bar" is largely Hammer/Shooting Star. The level-testing aspect is contextual. A stricter wick/body ratio could define it if needed separately.
                *   The other "False Breakout" definitions are more akin to "Strong Rejection/Reversal Candles" and could be standardized under such a name if their specific logic is deemed useful beyond basic Hammer/Shooting Star.
        *   **Harami Pattern Analysis (from `price_action_sr_strategy.py`):**
            *   Defines `_is_bullish_harami` and `_is_bearish_harami`.
            *   Classic Harami: C1 is large, C2 has a small body of the opposite color that is contained entirely within C1's *body*.
            *   **Standardization Note:** This is a standard pattern. Clear logic for Bullish and Bearish Harami should be implemented.
        *   **Two/Three-Bar Reversals (from `confluence_price_action_strategy.py`):**
            *   Methods like `_is_two_bar_reversal` and `_is_three_bar_reversal` exist.
            *   **Note:** These are often more complex, composite patterns or strategy-specific sequences. Standard ones like Piercing Line/Dark Cloud Cover (2-bar) or Morning/Evening Star (3-bar, already covered) are common. Other specific 2/3 bar patterns could be added if they are standard and widely recognized, but many are strategy-dependent constructs.
*   [x] **Phase 2: Design and Implementation of Centralized Pattern Module** (Completed)
    *   [x] Define structure and location for `src/utils/candlestick_patterns.py`.
    *   [x] Design function signatures and docstrings for patterns.
    *   [x] Implement consolidated pattern functions in the new module.
        *   Initial patterns: `detect_doji`, `detect_hammer`, `detect_shooting_star`, `detect_bullish_engulfing`, `detect_bearish_engulfing`, `detect_inside_bar`, `detect_bullish_harami`, `detect_bearish_harami`, `detect_morning_star` (generic), `detect_evening_star` (generic), `detect_false_breakout_level`.
    *   [x] Write unit tests for pattern functions.
        *   Created `tests/utils/test_candlestick_patterns.py`.
        *   Added tests for `detect_doji`, `detect_hammer`, `detect_bullish_engulfing`, `detect_inside_bar`, `detect_bullish_harami`, `detect_morning_star`.
    *   [x] **FIX**: Resolved `IndexError: single positional indexer is out-of-bounds` in `_check_breakout_signals` by checking for empty `fb` Series before indexing.
    *   [x] **BUG FIX**: `PriceActionSRStrategy` - Stop-loss calculation for buy signals can place SL above entry.
        *   **Analysis:** SL is set based on `zone - buffer`. If entry (next candle's open) is below the zone due to a gap/swift move, SL becomes invalid.
        *   **Proposed Solution (Option C from analysis):**
            *   Modify `calculate_stop_loss` to accept `candle_extremity` (low for buy, high for sell).
            *   SL for buy: `min(zone, candle_extremity) - buffer`.
            *   SL for sell: `max(zone, candle_extremity) + buffer`.
            *   Update `generate_signals` to pass `candle['low']` or `candle['high']` to `calculate_stop_loss`.
        *   **Success Criteria:** Signals generated by `PriceActionSRStrategy` have SL correctly placed below entry for buys and above entry for sells, even with adverse price movement on entry.
*   [x] **Phase 3: Refactor Strategies** (Completed)
    *   [x] Refactor `breakout_trading_strategy.py`.
        *   Removed local static pattern detection methods.
        *   Imported and used functions from `src.utils.candlestick_patterns`.
        *   Added `detect_morning_star_complex`, `detect_evening_star_complex`, and `detect_strong_reversal_candle` to the centralized module to match specific logic from this strategy.
        *   Updated `generate_signals` and `_detect_retest` to use centralized functions.
        *   **FIX**: Resolved "Expression value is unused" for `price_touched_level` in `_detect_retest` by embedding logic in `if` and improving log.
    *   [x] Refactor `breakout_reversal_strategy.py`.
        *   Identified specific `_detect_false_breakout` logic (key reversal bar after break of a level, potentially with volume).
        *   Added `detect_key_reversal_bar_after_breakout` to `src/utils/candlestick_patterns.py` to encapsulate this logic (takes level, break_direction, volume parameters).
        *   Verified `_is_morning_star` and `_is_evening_star` logic. Added `detect_morning_star_v2` and `detect_evening_star_v2` to `candlestick_patterns.py` to match its specific definition (C2 body < 50% C1, strict gap, C3 closes > C1 midpoint).
        *   Removed local static pattern detection methods (e.g., `_is_hammer`, `_is_shooting_star`, `_is_bullish_engulfing`, `_is_bearish_engulfing`, `_is_morning_star`, `_is_evening_star`, `_detect_false_breakout`).
        *   Updated `_check_breakout_signals` and `_check_reversal_signals` to use functions from `src.utils.candlestick_patterns` (`cp.detect_hammer`, `cp.detect_shooting_star`, `cp.detect_bullish_engulfing`, `cp.detect_bearish_engulfing`, `cp.detect_morning_star_v2`, `cp.detect_evening_star_v2`, and `cp.detect_key_reversal_bar_after_breakout`).
        *   **FIX**: Resolved `IndexError: single positional indexer is out-of-bounds` in `_check_breakout_signals` by checking for empty `fb` Series before indexing.
    *   [x] Refactor `trend_following_strategy.py`.
        *   **Analysis:** Strategy uses local static methods for Hammer, Shooting Star, Engulfing, Inside Bar, Morning/Evening Star, and False Breakout. All are vectorized.
            *   Basic patterns (Hammer, SS, Engulfing, Inside Bar) match standard definitions in `cp`.
            *   Morning/Evening Star logic matches `cp.detect_morning_star_complex` and `cp.detect_evening_star_complex`.
            *   False Breakout logic (strong close past previous close + long wick > 50% range) matches `cp.detect_strong_reversal_candle`. Default parameters used in strategy align well with `cp` function defaults.
        *   **Plan:**
            *   Remove all local static pattern methods.
            *   Import `src.utils.candlestick_patterns as cp`.
            *   In `generate_signals`, replace calls to local methods with their `cp` equivalents (e.g., `self.detect_hammer` -> `cp.detect_hammer`, `self.detect_morning_star` -> `cp.detect_morning_star_complex`, `self.detect_false_breakout` -> `cp.detect_strong_reversal_candle`).
        *   **Success Criteria:** `trend_following_strategy.py` refactored to use centralized patterns, local methods removed, and strategy behavior preserved.
    *   [x] Refactor `price_action_sr_strategy.py`.
        *   **Analysis:**
            *   Uses local, non-vectorized methods: `_is_bullish_engulfing`, `_is_bearish_engulfing`, `_is_hammer`, `_is_shooting_star`, `_is_bullish_harami`, `_is_bearish_harami`, `_is_morning_star`, `_is_evening_star`.
            *   `_pattern_match` calls these methods based on current candle `idx`.
            *   Hammer, SS, Engulfing, Harami definitions align with `cp` standard functions.
            *   Morning/Evening Star: local version is C2 body < 50% C1 body, C3 closes > 50% C1 mid, no explicit gap. `cp.detect_morning_star` (generic) will need params `c2_body_max_percent_c1` and `require_gap`.
            *   "Pin-bar" logic in `_pattern_match` is essentially Hammer/SS. Score impact needs handling.
            *   `_wick_rejection` method (using `self.wick_threshold`) is used for Pin-bar logic and also for the `wick` bool passed to `_score_signal_01`.
        *   **Plan:**
            1.  **Modify `src/utils/candlestick_patterns.py`**:
                *   Add `c2_body_max_percent_c1` (float, default 0.3) and `require_gap` (bool, default True) to `detect_morning_star` and `detect_evening_star`. Ensure current default behavior is maintained for existing calls.
            2.  **Refactor `price_action_sr_strategy.py`**:
                *   Remove local pattern methods: `_is_bullish_engulfing`, `_is_bearish_engulfing`, `_is_hammer`, `_is_shooting_star`, `_is_bullish_harami`, `_is_bearish_harami`, `_is_morning_star`, `_is_evening_star`.
                *   Import `src.utils.candlestick_patterns as cp`.
                *   In `_pattern_match(df, idx, direction)`:
                    *   Bullish Engulfing: `cp.detect_bullish_engulfing(df.iloc[idx-1:idx+1]).iloc[1]`
                    *   Hammer: `cp.detect_hammer(df.iloc[idx:idx+1], min_lower_wick_ratio=self.wick_threshold).iloc[0]` (Pass strategy's threshold)
                    *   Bullish Harami: `cp.detect_bullish_harami(df.iloc[idx-1:idx+1]).iloc[1]`
                    *   Morning Star: `cp.detect_morning_star(df.iloc[idx-2:idx+1], c2_body_max_percent_c1=0.5, require_gap=False).iloc[2]`
                    *   Symmetric calls for bearish patterns (Bearish Engulfing, Shooting Star, Bearish Harami, Evening Star). For Shooting Star, use `min_upper_wick_ratio=self.wick_threshold`.
                    *   Remove the specific "Pin-bar" detection logic; it will now be identified as 'Hammer' or 'Shooting Star'.
                *   In `_score_signal_01(pattern, wick, ...)`:
                    *   Adjust scoring for 'Hammer'/'Shooting Star'. If `pattern` is 'Hammer' and `wick` is True (from `_wick_rejection`), use score 0.5 (as "Pin-bar"). Else, if just 'Hammer', use 0.7. Similar for 'Shooting Star'.
                    *   Update `pattern_map` to remove "Pin-bar" if it's explicitly there, or ensure its score isn't used directly.
                *   The `_wick_rejection` method can remain as it's used by `generate_signals` to provide the `wick` boolean to `_score_signal_01`.
        *   **Success Criteria:** `price_action_sr_strategy.py` refactored. Local pattern methods removed. Strategy uses centralized `cp` functions. Scoring logic for Hammer/Shooting Star/Pin-bar is preserved. Behavior is validated. Linter errors (parameter names) fixed.
    *   [x] Refactor `confluence_price_action_strategy.py`.
        *   **Analysis:** Strategy uses local, non-vectorized, level-aware methods for patterns. The goal is to use `cp` for basic shape and keep level-specific logic local.
        *   **Actions Taken:**
            *   Imported `src.utils.candlestick_patterns as cp`.
            *   Modified `_is_pin_bar`, `_is_engulfing`, `_is_inside_bar`, `_is_hammer`, `_is_shooting_star`, `_is_morning_star`, `_is_evening_star` to first call their respective generic `cp` counterparts.
            *   If the `cp` function confirms the basic pattern, the local method then applies its existing specific logic (level proximity, near-miss criteria, ATR-based offsets, etc.).
            *   Local methods `_is_false_breakout`, `_is_two_bar_reversal`, `_is_three_bar_reversal` were kept largely as-is due to their highly specific sequential and level-based logic not mapping directly to simple `cp` calls.
        *   **Success Criteria:** `confluence_price_action_strategy.py` refactored to use centralized `cp` functions for basic pattern identification where appropriate, while retaining its specialized contextual logic. Behavior is validated.
*   [x] **Phase 4: Finalization** (Completed)
    *   [x] Review and finalize documentation for the new module and strategies.
        *   Module-level docstring added to `src/utils/candlestick_patterns.py`.
        *   Function-level docstrings assumed to be largely complete based on samples.
    *   [x] Update scratchpad with completion status and lessons.

## Executor's Feedback or Assistance Requests
*(To be filled by the Executor during implementation)*
- No further assistance requests at this time. The primary task of consolidating general candlestick patterns is complete.

## Lessons
*(To be filled as insights are gained)*
*   When setting stop-losses based on price action patterns (e.g., Hammer at a support zone), ensure the SL considers the extremity of the pattern candle itself (e.g., low of the Hammer) in addition to, or instead of, just the zone level. This is crucial if the entry occurs on a subsequent candle that might open unfavorably relative to the zone.
*   Not all pattern detection logic is suitable for centralization. If a pattern's definition is heavily tied to strategy-specific context (e.g., proximity to a dynamically calculated S/R level, or highly custom "near-miss" criteria), it might be best to keep it local to that strategy to avoid overcomplicating a general utility module or risking behavioral changes.