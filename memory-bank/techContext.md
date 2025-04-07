# Technical Context

## Technology Stack

### Core Technologies
- **Python 3.10+** - Primary programming language
- **MetaTrader 5** - Trading platform and API
- **Pandas** - Data analysis and manipulation
- **NumPy** - Numerical computations
- **Asyncio** - Asynchronous I/O for responsive operation
- **Telegram Bot API** - For notifications and command handling

### Key Libraries
- **MetaTrader5** - Python library for MT5 integration
- **python-telegram-bot** - Telegram bot framework
- **pandas-ta** - Technical analysis extensions for pandas
- **loguru** - Enhanced logging capabilities
- **python-dotenv** - Environment variable management
- **pytest** - Testing framework
- **pyright** - Static type checking tool

## Development Environment

### Prerequisites
1. **Python 3.10+** - Required for modern language features
2. **MetaTrader 5 Terminal** - Installed and configured with a demo/live account
3. **Telegram Bot Token** - For notifications and command handling

### Environment Setup
1. **Virtual Environment**
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

2. **Dependencies Installation**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables Configuration**
   The following variables should be set in the `.env` file:
   ```
   MT5_SERVER=your_broker_server
   MT5_LOGIN=your_login
   MT5_PASSWORD=your_password
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   ADMIN_CHAT_ID=your_chat_id
   ```

## Technical Constraints

### MetaTrader 5 Limitations
1. **API Connection** - MT5 must be running for the bot to connect
2. **Rate Limits** - MT5 may impose rate limits on data requests
3. **Windows Dependency** - Native MT5 runs primarily on Windows
4. **Account Types** - Different brokers may have varying MT5 implementations

### Execution Environment
1. **Continuous Operation** - System designed to run 24/5 during forex market hours
2. **Network Dependency** - Requires stable internet connection
3. **Memory Usage** - Data caching increases with the number of symbols and timeframes
4. **CPU Usage** - Strategy computation increases with complexity and number of symbols

## Project Structure

```
├── config/               # Configuration files
│   ├── config.py         # Main configuration
│   └── ...
├── src/                  # Source code
│   ├── mt5_handler.py    # MetaTrader 5 interface
│   ├── risk_manager.py   # Risk management
│   ├── trading_bot.py    # Main trading bot orchestrator
│   ├── strategy/         # Strategy implementations
│   │   ├── base.py       # Base strategy class
│   │   └── ...           # Individual strategies
│   ├── telegram/         # Telegram integration
│   │   ├── bot.py        # Telegram bot implementation
│   │   └── commands.py   # Command handlers
│   └── utils/            # Utility modules
│       ├── indicators.py # Technical indicators
│       ├── data_manager.py # Market data management
│       └── ...
├── scripts/              # Utility scripts
├── main.py               # Main entry point
├── .env                  # Environment variables
└── requirements.txt      # Dependencies
```

## Technical Decisions

### Asynchronous Architecture
The system uses Python's asyncio for concurrent operation, allowing it to:
- Handle multiple market data streams simultaneously
- Respond to Telegram commands while running the main trading loop
- Monitor and manage positions without blocking other operations

### Data Caching Strategy
To minimize MT5 API calls and improve performance:
- Market data is cached in memory with configurable update frequencies
- Each timeframe has a separate update schedule based on its period
- Strategies access cached data rather than making direct MT5 requests

### Risk Management Implementation
Risk management is implemented as a separate module to ensure:
- Consistent application of risk rules across all strategies
- Centralized position sizing calculations
- Independent validation of trade signals before execution

### Telegram Integration
Telegram was chosen for notifications and control because it offers:
- Secure, encrypted communication
- Mobile and desktop access
- Easy command parsing and response handling
- Media sharing capabilities for charts and reports

### Type Checking Strategy
To improve code quality and prevent runtime errors:
- Python type hints are used extensively throughout the codebase
- Pyright is used for static type checking
- A hybrid approach is used for dynamic data structures:
  - Strong typing for internal components
  - Strategic use of `Any` and type ignores for external interfaces
  - File-level type ignore directives when needed
- Common patterns for handling type issues:
  - Explicit type conversions when working with NumPy and Pandas
  - Type guards for runtime type checking
  - Type ignore comments only where necessary and with clear rationale

## Development Workflow

### Code Style and Standards
- **PEP 8** - Python style guide compliance
- **Type Hints** - Used throughout the codebase for better IDE support and static analysis
- **Docstrings** - Google style docstrings on all public methods and classes
- **Logging** - Comprehensive logging using loguru

### Type Checking Workflow
1. **Development** - Add type hints during initial development
2. **Verification** - Run pyright to catch type errors early
3. **Fixing** - Address type errors through:
   - Improved type annotations
   - Code refactoring for better type safety
   - Strategic use of type ignore comments when necessary
4. **Documentation** - Document type-related decisions in code comments

### Testing Approach
1. **Unit Tests** - For individual components like indicators and risk calculations
2. **Integration Tests** - For interaction between components
3. **Strategy Backtests** - For validating strategy performance
4. **Type Checking** - Static analysis with pyright to prevent type errors

### Deployment Process
1. **Development** on a local environment with MT5 demo account
2. **Testing** with historical data and paper trading
3. **Production** deployment on a stable server with MT5 live account

## Dependencies and External Systems

### External Dependencies
1. **MetaTrader 5** - Trading platform for execution and data
2. **Telegram** - For notifications and commands
3. **Broker API** - Through MT5 for market access

### Integration Points
1. **MT5Handler** - Primary integration with MetaTrader 5
2. **TelegramBot** - Integration with Telegram messaging
3. **ConfigSystem** - Configuration loading and management 