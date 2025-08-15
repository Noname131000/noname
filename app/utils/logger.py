import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

class CustomFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: grey + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

def setup_logger(
    name: str = "signal_bot",
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    console_output: bool = True
) -> logging.Logger:
    """
    Set up structured logger for the application
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        log_level: Logging level
        console_output: Whether to output to console
    
    Returns:
        Configured logger instance
    """
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(CustomFormatter())
        logger.addHandler(console_handler)
    
    return logger

def log_signal_data(logger: logging.Logger, signal_data: dict, stage: str):
    """Log signal data at different pipeline stages"""
    logger.info(f"[{stage}] Processing signal for {signal_data.get('pair', 'UNKNOWN')} "
               f"on {signal_data.get('timeframe', 'UNKNOWN')} timeframe")
    
    if stage == "GATE1":
        logger.debug(f"[{stage}] OHLC: {signal_data.get('ohlc', {})}")
        logger.debug(f"[{stage}] Volume: {signal_data.get('volume', 0)}")
        
    elif stage == "GATE2":
        logger.info(f"[{stage}] Signal score: {signal_data.get('score', 0)}")
        logger.info(f"[{stage}] Trend filters: {signal_data.get('trend_filters', {})}")
        logger.info(f"[{stage}] Risk/Reward: {signal_data.get('risk_reward', 0)}")
        
    elif stage == "GATE3":
        logger.info(f"[{stage}] AI decision: {signal_data.get('ai_decision', 'UNKNOWN')}")
        logger.info(f"[{stage}] AI confidence: {signal_data.get('ai_confidence', 0)}")
        logger.info(f"[{stage}] AI reasons: {signal_data.get('ai_reasons', [])}")
        
    elif stage == "GATE4":
        logger.info(f"[{stage}] Signal sent to Telegram")
        logger.info(f"[{stage}] Signal code: {signal_data.get('code', 'UNKNOWN')}")

def log_trade_update(logger: logging.Logger, code: str, update_type: str, details: dict):
    """Log trade execution updates"""
    logger.info(f"[TRADE_UPDATE] Code: {code} | Type: {update_type} | Details: {details}")

def log_error(logger: logging.Logger, error: Exception, context: str = ""):
    """Log errors with context"""
    logger.error(f"[ERROR] {context}: {str(error)}", exc_info=True)

def log_performance_metrics(logger: logging.Logger, metrics: dict):
    """Log performance metrics"""
    logger.info(f"[METRICS] {metrics}")

# Global logger instance
main_logger = None

def get_logger() -> logging.Logger:
    """Get the main application logger"""
    global main_logger
    if main_logger is None:
        from ..config.settings import settings
        main_logger = setup_logger(
            name="signal_bot",
            log_file=settings.LOG_FILE,
            log_level=settings.LOG_LEVEL
        )
    return main_logger