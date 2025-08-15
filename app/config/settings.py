import os
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    # Exchange settings
    EXCHANGE = os.getenv("EXCHANGE", "BINANCE")
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
    BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
    
    # Gemini AI
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
    
    # Trading pairs and timeframes
    PAIR_WHITELIST = os.getenv("PAIR_WHITELIST", "BTCUSDT,ETHUSDT,ADAUSDT,SOLUSDT").split(",")
    TIMEFRAMES = os.getenv("TIMEFRAMES", "5m,15m,1h").split(",")
    
    # Technical indicators
    EMA_FAST = int(os.getenv("EMA_FAST", "20"))
    EMA_SLOW = int(os.getenv("EMA_SLOW", "50"))
    EMA_BASE = int(os.getenv("EMA_BASE", "200"))
    RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
    STOCHRSI_K = int(os.getenv("STOCHRSI_K", "3"))
    STOCHRSI_D = int(os.getenv("STOCHRSI_D", "3"))
    STOCHRSI_RSI_PERIOD = int(os.getenv("STOCHRSI_RSI_PERIOD", "14"))
    STOCHRSI_STOCH_PERIOD = int(os.getenv("STOCHRSI_STOCH_PERIOD", "14"))
    ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
    ADX_PERIOD = int(os.getenv("ADX_PERIOD", "14"))
    BB_PERIOD = int(os.getenv("BB_PERIOD", "20"))
    BB_STD = float(os.getenv("BB_STD", "2.0"))
    KC_PERIOD = int(os.getenv("KC_PERIOD", "20"))
    KC_MULT = float(os.getenv("KC_MULT", "1.5"))
    
    # ADX thresholds
    ADX_MIN_SCALP = float(os.getenv("ADX_MIN_SCALP", "18"))
    ADX_MIN_INTRADAY = float(os.getenv("ADX_MIN_INTRADAY", "20"))
    ADX_MIN_SWING = float(os.getenv("ADX_MIN_SWING", "25"))
    
    # Risk management
    ATR_MULT_SL_SCALP = float(os.getenv("ATR_MULT_SL_SCALP", "1.0"))
    ATR_MULT_SL_INTRADAY = float(os.getenv("ATR_MULT_SL_INTRADAY", "1.5"))
    TP1_ATR = float(os.getenv("TP1_ATR", "1.0"))
    TP2_ATR = float(os.getenv("TP2_ATR", "2.0"))
    TRAILING_ATR = float(os.getenv("TRAILING_ATR", "1.5"))
    VWAP_DEV_MAX = float(os.getenv("VWAP_DEV_MAX", "2.0"))
    MIN_RR = float(os.getenv("MIN_RR", "1.5"))
    
    # Signal scoring
    MIN_SCORE_GATE2 = int(os.getenv("MIN_SCORE_GATE2", "60"))
    MIN_AI_CONFIDENCE = int(os.getenv("MIN_AI_CONFIDENCE", "60"))
    
    # Order management
    LIMIT_OFFSET_ATR = float(os.getenv("LIMIT_OFFSET_ATR", "0.25"))
    ORDER_TIF = os.getenv("ORDER_TIF", "GTC")
    RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.5"))
    MAX_ACTIVE_SIGNALS = int(os.getenv("MAX_ACTIVE_SIGNALS", "5"))
    COOLDOWN_BARS_AFTER_SL = int(os.getenv("COOLDOWN_BARS_AFTER_SL", "7"))
    
    # Volume and flow
    VOLUME_THRESHOLD = float(os.getenv("VOLUME_THRESHOLD", "1.2"))
    VOLUME_LOOKBACK = int(os.getenv("VOLUME_LOOKBACK", "20"))
    
    # Squeeze settings
    MIN_SQUEEZE_BARS = int(os.getenv("MIN_SQUEEZE_BARS", "10"))
    
    # Webhook settings
    WEBHOOK_HOST = os.getenv("WEBHOOK_HOST", "0.0.0.0")
    WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "8000"))
    WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "your_webhook_secret")
    
    # Database
    DB_PATH = os.getenv("DB_PATH", "/workspace/app/data/signals.db")
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "/workspace/app/data/bot.log")

    @classmethod
    def validate(cls) -> List[str]:
        """Validate required settings and return list of errors"""
        errors = []
        
        if not cls.BINANCE_API_KEY:
            errors.append("BINANCE_API_KEY is required")
        if not cls.BINANCE_API_SECRET:
            errors.append("BINANCE_API_SECRET is required")
        if not cls.GEMINI_API_KEY:
            errors.append("GEMINI_API_KEY is required")
        if not cls.TELEGRAM_BOT_TOKEN:
            errors.append("TELEGRAM_BOT_TOKEN is required")
        if not cls.TELEGRAM_CHAT_ID:
            errors.append("TELEGRAM_CHAT_ID is required")
        if not cls.WEBHOOK_SECRET:
            errors.append("WEBHOOK_SECRET is required")
            
        return errors

settings = Settings()