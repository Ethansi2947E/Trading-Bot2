# Technical Context

## Development Environment

### Core Technologies
1. Python 3.8+
   - Async/await support
   - Type hints
   - Modern features
   - Performance improvements

2. MetaTrader 5
   - Trading platform
   - Market data source
   - Order execution
   - Position management

3. Development Tools
   - VS Code
   - Git
   - Python debugger
   - Code profiler

### Dependencies

1. Core Libraries
   ```python
   # requirements.txt
   MetaTrader5>=5.0.34
   pandas>=1.3.0
   numpy>=1.20.0
   asyncio>=3.4.3
   aiohttp>=3.8.0
   python-dotenv>=0.19.0
   loguru>=0.5.3
   ```

2. Development Libraries
   ```python
   # dev-requirements.txt
   pytest>=6.2.5
   pytest-asyncio>=0.15.1
   pytest-cov>=2.12.1
   black>=21.7b0
   flake8>=3.9.2
   mypy>=0.910
   ```

## Technical Constraints

### MetaTrader 5
1. Connection
   - Single connection per instance
   - Connection timeouts
   - Reconnection handling
   - Status monitoring

2. Data Access
   - Rate limits
   - Data availability
   - Timeframe restrictions
   - Symbol limitations

3. Trading
   - Order types
   - Position limits
   - Execution speed
   - Price slippage

### System Resources
1. Memory
   - 8GB minimum
   - Data caching
   - Memory management
   - Garbage collection

2. CPU
   - 4+ cores
   - Processing power
   - Task scheduling
   - Load balancing

3. Network
   - 100Mbps minimum
   - Low latency
   - Stable connection
   - Error handling

### Performance
1. Data Processing
   - Real-time handling
   - Efficient algorithms
   - Memory optimization
   - CPU utilization

2. Trade Execution
   - Quick response
   - Accurate timing
   - Error recovery
   - State management

## Development Setup

### Environment Variables
```bash
# .env
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_server
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Configuration
```python
# config.py
config = {
    "symbols": ["EURUSD", "GBPUSD"],
    "timeframes": ["M5", "H1"],
    "risk_percent": 2.0,
    "max_positions": 5,
    "data_fetch_interval": 60
}
```

### Directory Structure
```
trading_bot/
├── src/
│   ├── mt5_handler.py
│   ├── trading_bot.py
│   ├── risk_manager.py
│   └── strategy/
├── tests/
│   ├── test_mt5_handler.py
│   ├── test_trading_bot.py
│   └── test_risk_manager.py
├── config/
│   ├── config.py
│   └── .env
└── docs/
    ├── setup.md
    └── api.md
```

## Testing Environment

### Unit Tests
```python
# test_mt5_handler.py
import pytest
import asyncio
from src.mt5_handler import MT5Handler

@pytest.mark.asyncio
async def test_connection():
    handler = MT5Handler()
    assert await handler.connect()
```

### Integration Tests
```python
# test_trading_bot.py
import pytest
from src.trading_bot import TradingBot

@pytest.mark.integration
def test_full_cycle():
    bot = TradingBot()
    # Test complete trading cycle
```

### Performance Tests
```python
# test_performance.py
import cProfile
import pstats

def profile_data_processing():
    profiler = cProfile.Profile()
    # Profile critical operations
```

## Deployment

### Requirements
1. Hardware
   - CPU: 4+ cores
   - RAM: 8GB+
   - Storage: 50GB+
   - Network: 100Mbps+

2. Software
   - Python 3.8+
   - MetaTrader 5
   - Required packages
   - System utilities

3. Network
   - Stable connection
   - Low latency
   - Sufficient bandwidth
   - Reliable DNS

### Setup Steps
1. System Setup
   ```bash
   # Install Python
   sudo apt update
   sudo apt install python3.8
   
   # Install pip
   sudo apt install python3-pip
   
   # Install virtualenv
   pip3 install virtualenv
   ```

2. Project Setup
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate environment
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. Configuration
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit configuration
   nano config/config.py
   ```

## Monitoring

### System Metrics
1. Resource Usage
   - CPU utilization
   - Memory consumption
   - Disk usage
   - Network traffic

2. Performance
   - Response times
   - Processing speed
   - Data throughput
   - Error rates

3. Trading Metrics
   - Order execution
   - Position management
   - Risk compliance
   - Profit/loss

### Logging
```python
# logger.py
from loguru import logger

logger.add("logs/trading.log",
    rotation="1 day",
    retention="30 days",
    level="INFO")
```

### Alerts
1. System Alerts
   - Resource limits
   - Performance issues
   - Error thresholds
   - Connection problems

2. Trading Alerts
   - Order execution
   - Position changes
   - Risk violations
   - Profit/loss

## Security

### Access Control
1. API Keys
   - Secure storage
   - Regular rotation
   - Access monitoring
   - Usage limits

2. Authentication
   - Strong passwords
   - Two-factor auth
   - Session management
   - Access logs

### Data Protection
1. Sensitive Data
   - Encryption
   - Secure storage
   - Access control
   - Data backup

2. Communication
   - Secure protocols
   - Data encryption
   - Channel security
   - Error handling

## Maintenance

### Regular Tasks
1. Daily
   - Log review
   - Error check
   - Backup verify
   - Performance check

2. Weekly
   - System update
   - Code review
   - Test run
   - Documentation update

3. Monthly
   - Performance audit
   - Security review
   - Feature planning
   - System optimization

### Procedures
1. Backup
   ```bash
   # Backup script
   #!/bin/bash
   rsync -av --delete /trading_bot /backup
   ```

2. Update
   ```bash
   # Update script
   git pull origin main
   pip install -r requirements.txt
   ```

3. Monitoring
   ```python
   # monitor.py
   async def check_system():
       # System health check
       pass
   ```

## Development Workflow

### Version Control
1. Git Flow
   - Feature branches
   - Pull requests
   - Code review
   - Merge strategy

2. Commits
   - Clear messages
   - Atomic changes
   - Documentation
   - Tests included

### Code Quality
1. Style Guide
   - PEP 8
   - Type hints
   - Documentation
   - Comments

2. Testing
   - Unit tests
   - Integration tests
   - Performance tests
   - Coverage reports

### Documentation
1. Code
   - Docstrings
   - Comments
   - Type hints
   - Examples

2. Project
   - README
   - API docs
   - Setup guide
   - User manual

## Future Considerations

### Scalability
1. Multi-Account
   - Account management
   - Risk distribution
   - Performance tracking
   - Reporting

2. Performance
   - Optimization
   - Caching
   - Distribution
   - Load balancing

### Features
1. Analysis
   - Advanced indicators
   - Pattern recognition
   - Market analysis
   - Risk assessment

2. Integration
   - External data
   - Other platforms
   - APIs
   - Services

### Intelligence
1. Machine Learning
   - Pattern recognition
   - Prediction models
   - Risk analysis
   - Optimization

2. Automation
   - Strategy selection
   - Parameter tuning
   - Risk adjustment
   - Performance optimization

### Trading Strategy Parameters

#### BreakoutReversalStrategy Configuration
```python
# Default initialization
strategy = BreakoutReversalStrategy(
    primary_timeframe="M15",  # Changed from M5 to M15
    higher_timeframe="H1"
)
```

#### Timeframe Profiles
The strategy now dynamically adjusts parameters based on the primary timeframe:

1. **M1 (1-minute)** - "scalping" profile
   - Update intervals: very short (30min-1hr)
   - Pattern recognition: ultra-short term
   - Retest time: 2-4 hours maximum

2. **M5/M15 (5/15-minute)** - "intraday" profile
   - Update intervals: short (1-2 hours)
   - Level update: every 1-2 hours
   - Trend line update: every 1-2 hours
   - Consolidation range update: every 30min-1hr
   - Retest time: 4-8 hours maximum

3. **H1/H4 (1/4-hour)** - "intraday_swing" profile
   - Update intervals: medium (2-4 hours)
   - Level update: every 4 hours
   - Trend line update: every 4 hours
   - Consolidation range update: every 2 hours
   - Retest time: 12 hours maximum

4. **D1+ (Daily+)** - "swing" profile
   - Update intervals: long (8+ hours)
   - Level update: every 8 hours
   - Trend line update: every 8 hours
   - Consolidation range update: every 4 hours
   - Retest time: 24 hours maximum

This adaptive approach ensures parameters are appropriately scaled to each timeframe's characteristics:
- Lower timeframes get more frequent updates
- Higher timeframes maintain longer-term stability
- Retest windows match the typical price action speed of each timeframe 