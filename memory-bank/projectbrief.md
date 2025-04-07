# Trading Bot Project Brief

## Project Overview
The Trading Bot is an advanced algorithmic trading system designed to implement Smart Money Concepts (SMC), multi-timeframe analysis, order block detection, and sophisticated risk management for automated forex trading via MetaTrader 5. The system follows institutional trading patterns to identify high-probability trade setups and executes them with disciplined risk management.

## Core Objectives
1. **Implement Smart Money Concepts (SMC)** - Apply institutional trading patterns and market structure analysis
2. **Multi-Timeframe Analysis** - Allow strategies to define their own timeframe requirements and analyze price across multiple timeframes
3. **Order Block Detection** - Identify zones of liquidity where large orders have been executed
4. **Risk Management** - Implement dynamic position sizing and risk controls to protect capital
5. **Real-Time Trading** - Execute trades directly through MetaTrader 5
6. **Monitoring & Notifications** - Provide trade alerts and performance updates via Telegram
7. **Backtesting** - Test strategies against historical data before live deployment

## Core Features
- Strategy-defined timeframe selection
- Advanced order block detection
- Dynamic position sizing and risk management
- Real-time trading through MT5
- Telegram notifications and command handling
- Multi-strategy support with isolation
- Trade management with partial take-profits and trailing stops
- Market structure identification
- Divergence pattern detection
- Volume analysis

## Target Users
1. **Algorithmic Traders** - Those who want to implement automated trading strategies based on Smart Money Concepts
2. **Institutional Trading Enthusiasts** - Traders who follow institutional trading patterns and market structure
3. **Forex Traders** - Specifically targeting forex markets through MetaTrader 5

## Success Criteria
1. **Stability** - System runs reliably without interruptions
2. **Profitability** - Generates positive returns over time
3. **Risk Control** - Maintains drawdowns within defined parameters
4. **Ease of Extension** - Allows for easy addition of new strategies
5. **Performance Tracking** - Provides detailed metrics on trading performance

## Constraints
1. **MetaTrader 5 Dependency** - Relies on MT5 for market data and trade execution
2. **Python Environment** - Requires Python 3.10+ ecosystem
3. **Network Requirements** - Needs stable internet connection for MT5 communication and Telegram notifications 