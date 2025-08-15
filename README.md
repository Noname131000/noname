# Telegram Signal Bot for Futures Trading

A sophisticated, AI-powered trading signal bot that analyzes cryptocurrency futures markets and sends professional trading signals to Telegram. The bot implements a comprehensive 4-gate filtering system combining technical analysis, AI verification, and risk management.

## 🚀 Features

### Core Functionality
- **Real-time Market Analysis**: Monitors multiple cryptocurrency pairs across 5m, 15m, and 1h timeframes
- **4-Gate Signal Filtering System**: Rigorous validation through technical analysis, scoring, AI verification, and notifications
- **AI-Powered Verification**: Gemini AI integration for signal quality assessment
- **Professional Telegram Notifications**: Clean, formatted signals with tracking codes
- **Complete Signal Lifecycle Management**: Entry, TP1/TP2, SL, and trailing stop notifications
- **Performance Analytics**: Comprehensive tracking of signal performance and bot metrics

### Technical Analysis Components
- **EMAs**: 20, 50, 200 with alignment and slope analysis
- **Momentum**: RSI(14), StochRSI(3,3,14,14) with crossover detection
- **Trend Strength**: ADX(14) with configurable thresholds
- **Volatility**: ATR(14), Bollinger Bands(20,2), Keltner Channels(20,1.5)
- **Volume**: VWAP with standard deviation bands, relative volume analysis
- **Squeeze Detection**: Bollinger Bands inside Keltner Channels
- **Multi-timeframe Analysis**: H1 bias validation

### Risk Management
- **Dynamic Stop Losses**: ATR-based with mode-specific multipliers
- **Take Profit Levels**: TP1 (50% position) and TP2 (remaining position)
- **Trailing Stops**: Automatic trailing after TP1
- **Risk/Reward Validation**: Minimum 1.5:1 ratio requirement
- **Position Sizing**: Configurable risk per trade
- **Cooldown System**: Pair-specific cooldowns after stop loss hits

## 🏗️ Architecture

### Gate System Overview

#### Gate 1: Candle Analysis
- Receives kline close data from Binance webhook/websocket
- Calculates all technical indicators
- Data normalization and validation
- Spam control and debouncing

#### Gate 2: Indicator Combination & Scoring
- **Trend Analysis** (0-25 points): EMA alignment, slope, ADX, MTF bias
- **Momentum Analysis** (0-20 points): RSI positioning, StochRSI crosses
- **Volatility Analysis** (0-20 points): ATR levels, squeeze conditions, VWAP distance
- **Volume Analysis** (0-20 points): Relative volume, efficiency ratios
- **Structure Analysis** (0-10 points): BB positioning, entry quality
- **Risk/Reward** (0-5 points): Final R:R ratio scoring
- **Minimum Score**: 60/100 to proceed

#### Gate 3: AI Verification
- Structured prompt with complete signal context
- Gemini AI returns: PROCESS/HOLD/REJECT + confidence + reasons
- Minimum 60% confidence required
- Fallback logic for API failures

#### Gate 4: Telegram Notification
- Professional message formatting with monospace numbers
- Unique tracking codes (DDHHM-PAIR-TF-MODE-HASH4)
- Real-time signal updates (entry fills, TP/SL hits)
- Performance summaries and system notifications

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- Binance API access (for webhook setup)
- Gemini AI API key
- Telegram bot token and chat ID

### Quick Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd telegram-signal-bot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Environment Configuration**
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

4. **Create data directory**
```bash
mkdir -p app/data
```

## ⚙️ Configuration

### Required Environment Variables

```env
# Exchange Configuration
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret

# AI Configuration  
GEMINI_API_KEY=your_gemini_api_key

# Telegram Configuration
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Webhook Security
WEBHOOK_SECRET=your_secure_webhook_secret
```

### Trading Configuration

```env
# Trading Pairs (comma-separated)
PAIR_WHITELIST=BTCUSDT,ETHUSDT,ADAUSDT,SOLUSDT

# Timeframes to monitor
TIMEFRAMES=5m,15m,1h

# Signal Thresholds
MIN_SCORE_GATE2=60
MIN_AI_CONFIDENCE=60
MIN_RR=1.5

# Risk Management
RISK_PER_TRADE_PCT=0.5
MAX_ACTIVE_SIGNALS=5
COOLDOWN_BARS_AFTER_SL=7
```

### Technical Indicator Settings

```env
# ADX Thresholds by Trading Mode
ADX_MIN_SCALP=18
ADX_MIN_INTRADAY=20  
ADX_MIN_SWING=25

# ATR Multipliers
ATR_MULT_SL_SCALP=1.0
ATR_MULT_SL_INTRADAY=1.5
TP1_ATR=1.0
TP2_ATR=2.0

# VWAP Deviation
VWAP_DEV_MAX=2.0
```

## 🚀 Usage

### Starting the Bot

```bash
# Run the bot
python -m app.main

# With custom config
python -m app.main --config /path/to/config

# Run health check
python -m app.main --health

# Run pipeline test
python -m app.main --test
```

### Webhook Setup

If using Binance webhooks (recommended for production):

1. Configure webhook endpoint in Binance
2. Set `WEBHOOK_SECRET` for signature validation
3. Point webhook to: `http://your-server:8000/webhook/binance`

### WebSocket Alternative

To use WebSocket instead of webhooks, modify in `main.py`:
```python
self.data_source = create_data_source(
    signal_processor_callback=self.pipeline.process_kline_data,
    use_webhook=False  # Set to False for WebSocket
)
```

## 📊 Signal Format

### New Signal Example
```
🟢 NEW SIGNAL

PAIR          : BTCUSDT
SIGNAL        : BUY  
ENTRY LIMIT   : 47250.00000 (valid 20 bars)
TP/SL         : TP1=47750.00000 | TP2=48250.00000 | SL=46750.00000
VOL/ATR       : 1.18 (STRONG)
AI NOTE       : Strong bullish confluence; RSI>55; Volume 1.4x average
CODE          : 15142-BTCUSDT-15M-INTRA-7F2D

📊 TECHNICAL DETAILS
• Score: 78/100 | R:R 2.1
• ADX: 24.5 | RSI: 58.2 | Vol: 1.4x  
• 15m INTRA | MTF: BULL
• AI: PROCESS (85%)
```

### Signal Update Example
```
🎯 SIGNAL UPDATE

CODE          : 15142-BTCUSDT-15M-INTRA-7F2D
PAIR          : BTCUSDT
ENTRY         : 47250.00000
SIGNAL        : BUY
INFO          : TP1 HIT (+1.06%) | REMAINING POSITION TRAILING
```

## 📈 Performance Tracking

The bot automatically tracks:
- **Signal Performance**: Win rate, P&L, duration
- **Pipeline Metrics**: Gate conversion rates, processing volume
- **Technical Quality**: Score distributions, AI agreement rates
- **System Health**: Error rates, response times

Performance summaries are sent daily via Telegram.

## 🔧 Monitoring & Maintenance

### Log Files
- Application logs: `app/data/bot.log`
- Structured logging with different levels
- Automatic log rotation (implement as needed)

### Database
- SQLite database: `app/data/signals.db`
- Signal lifecycle tracking
- Performance analytics
- Automatic cleanup of old records

### Health Checks
```bash
# System health check
python -m app.main --health

# Check active signals
# (via Telegram commands - implement as needed)
```

## 🛠️ Development

### Project Structure
```
app/
├── main.py                 # Application entry point
├── config/
│   └── settings.py         # Configuration management
├── core/
│   ├── webhook_binance.py  # Data source handling
│   ├── indicators.py       # Technical analysis
│   ├── scoring.py          # Signal scoring system
│   ├── ai_gemini.py        # AI verification
│   ├── notifier.py         # Telegram notifications
│   ├── storage.py          # Database and tracking
│   └── signal_pipeline.py  # Main orchestrator
├── utils/
│   └── logger.py           # Logging utilities
└── data/                   # Runtime data storage
```

### Running Tests
```bash
# Run pipeline test
python -m app.main --test

# Run with pytest (if tests are added)
pytest tests/
```

### Code Formatting
```bash
black app/
flake8 app/
```

## 🚨 Important Notes

### Risk Disclaimer
This bot is for educational and research purposes. Trading cryptocurrencies involves significant risk. Always:
- Test thoroughly with paper trading first
- Start with small position sizes
- Monitor signal performance closely
- Implement proper risk management

### API Limits
- **Gemini AI**: Be mindful of rate limits and costs
- **Binance**: Respect API rate limits
- **Telegram**: Bot API has message rate limits

### Security
- Keep API keys secure and never commit them
- Use webhook signatures for validation
- Regularly rotate API keys
- Monitor for unusual activity

## 📞 Support

### Common Issues

1. **Configuration Errors**: Check `.env` file settings
2. **API Connection Issues**: Verify API keys and network connectivity
3. **Database Errors**: Ensure write permissions to `app/data/`
4. **Memory Usage**: Monitor for memory leaks in long-running instances

### Debugging
- Set `LOG_LEVEL=DEBUG` for detailed logging
- Use `--test` flag to test pipeline components
- Check database contents for signal tracking issues

## 📄 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]

---

**Disclaimer**: This software is provided "as is" without warranty. Trading involves risk of loss. Use at your own discretion.
