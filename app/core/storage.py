import sqlite3
import asyncio
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import os

from .scoring import SignalData, SignalDirection, TradingMode
from .ai_gemini import AIDecision
from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger()

@dataclass
class SignalRecord:
    """Complete signal record for database storage"""
    code: str
    symbol: str
    timeframe: str
    direction: str
    mode: str
    
    # Entry details
    entry_limit: float
    entry_filled_price: Optional[float]
    entry_timestamp: str
    entry_filled_timestamp: Optional[str]
    
    # Exit details
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    trailing_stop: float
    
    # Risk management
    risk_reward: float
    position_size_pct: float
    valid_bars: int
    
    # Scoring and AI
    total_score: float
    score_breakdown: str  # JSON
    ai_decision: str
    ai_confidence: int
    ai_reasons: str  # JSON
    
    # Technical context
    indicators_snapshot: str  # JSON
    mtf_bias: str
    vol_atr_ratio: float
    squeeze_status: str  # JSON
    
    # Status tracking
    status: str  # PENDING, FILLED, TP1_HIT, TP2_HIT, SL_HIT, EXPIRED, CANCELLED
    fill_percentage: float  # 0-100
    
    # Results
    pnl_pct: Optional[float]
    pnl_absolute: Optional[float]
    exit_timestamp: Optional[str]
    exit_reason: Optional[str]
    duration_minutes: Optional[int]
    
    # Metadata
    created_at: str
    updated_at: str

class SignalDatabase:
    """SQLite database for signal tracking and analytics"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or settings.DB_PATH
        self.connection = None
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Ensure database directory exists"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """Initialize database connection and tables"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            
            await self._create_tables()
            await self._create_indexes()
            
            logger.info(f"Database initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Database connection closed")
    
    async def _create_tables(self):
        """Create database tables"""
        
        signals_table = """
        CREATE TABLE IF NOT EXISTS signals (
            code TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            direction TEXT NOT NULL,
            mode TEXT NOT NULL,
            
            entry_limit REAL NOT NULL,
            entry_filled_price REAL,
            entry_timestamp TEXT NOT NULL,
            entry_filled_timestamp TEXT,
            
            stop_loss REAL NOT NULL,
            take_profit_1 REAL NOT NULL,
            take_profit_2 REAL NOT NULL,
            trailing_stop REAL NOT NULL,
            
            risk_reward REAL NOT NULL,
            position_size_pct REAL NOT NULL,
            valid_bars INTEGER NOT NULL,
            
            total_score REAL NOT NULL,
            score_breakdown TEXT NOT NULL,
            ai_decision TEXT NOT NULL,
            ai_confidence INTEGER NOT NULL,
            ai_reasons TEXT NOT NULL,
            
            indicators_snapshot TEXT NOT NULL,
            mtf_bias TEXT NOT NULL,
            vol_atr_ratio REAL NOT NULL,
            squeeze_status TEXT NOT NULL,
            
            status TEXT NOT NULL DEFAULT 'PENDING',
            fill_percentage REAL NOT NULL DEFAULT 0.0,
            
            pnl_pct REAL,
            pnl_absolute REAL,
            exit_timestamp TEXT,
            exit_reason TEXT,
            duration_minutes INTEGER,
            
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
        
        updates_table = """
        CREATE TABLE IF NOT EXISTS signal_updates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_code TEXT NOT NULL,
            update_type TEXT NOT NULL,
            update_details TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (signal_code) REFERENCES signals (code)
        )
        """
        
        performance_table = """
        CREATE TABLE IF NOT EXISTS performance_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            symbol TEXT,
            timeframe TEXT,
            total_signals INTEGER NOT NULL DEFAULT 0,
            wins INTEGER NOT NULL DEFAULT 0,
            losses INTEGER NOT NULL DEFAULT 0,
            win_rate REAL NOT NULL DEFAULT 0.0,
            total_pnl_pct REAL NOT NULL DEFAULT 0.0,
            avg_pnl_pct REAL NOT NULL DEFAULT 0.0,
            max_win_pct REAL NOT NULL DEFAULT 0.0,
            max_loss_pct REAL NOT NULL DEFAULT 0.0,
            avg_duration_minutes REAL NOT NULL DEFAULT 0.0,
            created_at TEXT NOT NULL
        )
        """
        
        cursor = self.connection.cursor()
        cursor.execute(signals_table)
        cursor.execute(updates_table)
        cursor.execute(performance_table)
        self.connection.commit()
    
    async def _create_indexes(self):
        """Create database indexes for performance"""
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals (symbol)",
            "CREATE INDEX IF NOT EXISTS idx_signals_timeframe ON signals (timeframe)",
            "CREATE INDEX IF NOT EXISTS idx_signals_status ON signals (status)",
            "CREATE INDEX IF NOT EXISTS idx_signals_created_at ON signals (created_at)",
            "CREATE INDEX IF NOT EXISTS idx_updates_signal_code ON signal_updates (signal_code)",
            "CREATE INDEX IF NOT EXISTS idx_updates_timestamp ON signal_updates (timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_performance_date ON performance_summary (date)"
        ]
        
        cursor = self.connection.cursor()
        for index in indexes:
            cursor.execute(index)
        self.connection.commit()
    
    async def save_signal(self, signal_record: SignalRecord) -> bool:
        """Save new signal to database"""
        
        try:
            cursor = self.connection.cursor()
            
            # Convert record to dict for insertion
            data = asdict(signal_record)
            
            # Prepare SQL
            columns = list(data.keys())
            placeholders = ['?' for _ in columns]
            values = list(data.values())
            
            sql = f"""
            INSERT INTO signals ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            """
            
            cursor.execute(sql, values)
            self.connection.commit()
            
            logger.info(f"Saved signal to database: {signal_record.code}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving signal {signal_record.code}: {e}")
            return False
    
    async def update_signal_status(self, code: str, status: str, 
                                 update_details: Dict = None) -> bool:
        """Update signal status and details"""
        
        try:
            cursor = self.connection.cursor()
            
            # Prepare update data
            update_data = {
                'status': status,
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Add specific update details
            if update_details:
                update_data.update(update_details)
            
            # Build SQL
            set_clause = ', '.join([f"{k} = ?" for k in update_data.keys()])
            values = list(update_data.values()) + [code]
            
            sql = f"UPDATE signals SET {set_clause} WHERE code = ?"
            
            cursor.execute(sql, values)
            self.connection.commit()
            
            # Log update
            await self._log_signal_update(code, status, update_details or {})
            
            logger.info(f"Updated signal {code}: {status}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating signal {code}: {e}")
            return False
    
    async def _log_signal_update(self, code: str, update_type: str, details: Dict):
        """Log signal update to updates table"""
        
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT INTO signal_updates (signal_code, update_type, update_details, timestamp)
                VALUES (?, ?, ?, ?)
            """, (
                code,
                update_type,
                json.dumps(details),
                datetime.now(timezone.utc).isoformat()
            ))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error logging update for signal {code}: {e}")
    
    async def get_signal(self, code: str) -> Optional[SignalRecord]:
        """Get signal by code"""
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM signals WHERE code = ?", (code,))
            row = cursor.fetchone()
            
            if row:
                return SignalRecord(**dict(row))
            return None
            
        except Exception as e:
            logger.error(f"Error getting signal {code}: {e}")
            return None
    
    async def get_active_signals(self) -> List[SignalRecord]:
        """Get all active signals (PENDING or FILLED)"""
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT * FROM signals 
                WHERE status IN ('PENDING', 'FILLED', 'TP1_HIT')
                ORDER BY created_at DESC
            """)
            
            rows = cursor.fetchall()
            return [SignalRecord(**dict(row)) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting active signals: {e}")
            return []
    
    async def get_signals_by_status(self, status: str, limit: int = 100) -> List[SignalRecord]:
        """Get signals by status"""
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT * FROM signals 
                WHERE status = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (status, limit))
            
            rows = cursor.fetchall()
            return [SignalRecord(**dict(row)) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting signals by status {status}: {e}")
            return []
    
    async def get_performance_stats(self, days: int = 30) -> Dict:
        """Get performance statistics for the last N days"""
        
        try:
            cursor = self.connection.cursor()
            
            # Get date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
            start_date = start_date.replace(day=start_date.day - days)
            
            # Query completed signals
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_signals,
                    SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl_pct < 0 THEN 1 ELSE 0 END) as losses,
                    AVG(CASE WHEN pnl_pct IS NOT NULL THEN pnl_pct ELSE 0 END) as avg_pnl,
                    SUM(CASE WHEN pnl_pct IS NOT NULL THEN pnl_pct ELSE 0 END) as total_pnl,
                    MAX(pnl_pct) as max_win,
                    MIN(pnl_pct) as max_loss,
                    AVG(duration_minutes) as avg_duration
                FROM signals 
                WHERE status IN ('TP1_HIT', 'TP2_HIT', 'SL_HIT')
                AND created_at >= ?
            """, (start_date.isoformat(),))
            
            row = cursor.fetchone()
            
            total = row['total_signals'] or 0
            wins = row['wins'] or 0
            losses = row['losses'] or 0
            
            stats = {
                'total_signals': total,
                'wins': wins,
                'losses': losses,
                'win_rate': (wins / total * 100) if total > 0 else 0,
                'total_pnl_pct': row['total_pnl'] or 0,
                'avg_pnl_pct': row['avg_pnl'] or 0,
                'max_win_pct': row['max_win'] or 0,
                'max_loss_pct': row['max_loss'] or 0,
                'avg_duration_minutes': row['avg_duration'] or 0,
                'period': f"Last {days} days"
            }
            
            # Get best performing pair
            cursor.execute("""
                SELECT symbol, SUM(pnl_pct) as total_pnl
                FROM signals 
                WHERE status IN ('TP1_HIT', 'TP2_HIT', 'SL_HIT')
                AND created_at >= ?
                AND pnl_pct IS NOT NULL
                GROUP BY symbol
                ORDER BY total_pnl DESC
                LIMIT 1
            """, (start_date.isoformat(),))
            
            best_row = cursor.fetchone()
            if best_row:
                stats['best_pair'] = best_row['symbol']
                stats['best_pair_pnl'] = best_row['total_pnl']
            else:
                stats['best_pair'] = 'N/A'
                stats['best_pair_pnl'] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {
                'total_signals': 0, 'wins': 0, 'losses': 0, 'win_rate': 0,
                'total_pnl_pct': 0, 'avg_pnl_pct': 0, 'max_win_pct': 0, 'max_loss_pct': 0,
                'avg_duration_minutes': 0, 'best_pair': 'N/A', 'period': f"Last {days} days"
            }
    
    async def cleanup_old_signals(self, days: int = 90):
        """Clean up old completed signals to manage database size"""
        
        try:
            cursor = self.connection.cursor()
            
            cutoff_date = datetime.now(timezone.utc)
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)
            
            # Delete old completed signals
            cursor.execute("""
                DELETE FROM signals 
                WHERE status IN ('TP2_HIT', 'SL_HIT', 'EXPIRED', 'CANCELLED')
                AND created_at < ?
            """, (cutoff_date.isoformat(),))
            
            deleted_count = cursor.rowcount
            
            # Delete related updates
            cursor.execute("""
                DELETE FROM signal_updates 
                WHERE signal_code NOT IN (SELECT code FROM signals)
            """, )
            
            self.connection.commit()
            
            logger.info(f"Cleaned up {deleted_count} old signals")
            
        except Exception as e:
            logger.error(f"Error cleaning up old signals: {e}")

class SignalTracker:
    """High-level signal tracking and management"""
    
    def __init__(self):
        self.db = SignalDatabase()
        self.active_signals = {}  # code -> SignalRecord
    
    async def start(self):
        """Start signal tracker"""
        await self.db.initialize()
        
        # Load active signals
        active = await self.db.get_active_signals()
        for signal in active:
            self.active_signals[signal.code] = signal
        
        logger.info(f"Signal tracker started with {len(active)} active signals")
    
    async def stop(self):
        """Stop signal tracker"""
        await self.db.close()
        logger.info("Signal tracker stopped")
    
    def generate_signal_code(self, symbol: str, timeframe: str, mode: str) -> str:
        """Generate unique signal code"""
        
        # Format: DDHHM-SYMBOL-TF-MODE-HASH4
        now = datetime.now(timezone.utc)
        date_part = now.strftime("%d%H%M")
        
        # Create hash from timestamp and symbol for uniqueness
        hash_input = f"{now.isoformat()}-{symbol}-{timeframe}"
        hash_hex = hashlib.md5(hash_input.encode()).hexdigest()[:4].upper()
        
        return f"{date_part}-{symbol}-{timeframe}-{mode}-{hash_hex}"
    
    async def create_signal_record(self, signal_data: SignalData, ai_decision: AIDecision,
                                 signal_code: str) -> SignalRecord:
        """Create signal record from signal data"""
        
        now = datetime.now(timezone.utc).isoformat()
        entry_plan = signal_data.entry_plan
        indicators = signal_data.indicators
        
        # Prepare JSON data
        score_breakdown = {
            'trend': signal_data.score.trend_score,
            'momentum': signal_data.score.momentum_score,
            'volatility': signal_data.score.volatility_score,
            'volume': signal_data.score.volume_score,
            'structure': signal_data.score.structure_score,
            'risk_reward': signal_data.score.rr_score
        }
        
        indicators_snapshot = {
            'close': indicators.ohlc['close'],
            'ema20': indicators.ema20,
            'ema50': indicators.ema50,
            'ema200': indicators.ema200,
            'rsi': indicators.rsi,
            'adx': indicators.adx,
            'atr': indicators.atr,
            'vwap': indicators.vwap,
            'relative_volume': indicators.relative_volume
        }
        
        return SignalRecord(
            code=signal_code,
            symbol=signal_data.symbol,
            timeframe=signal_data.timeframe,
            direction=entry_plan.direction.value,
            mode=entry_plan.mode.value,
            
            entry_limit=entry_plan.entry_limit,
            entry_filled_price=None,
            entry_timestamp=now,
            entry_filled_timestamp=None,
            
            stop_loss=entry_plan.stop_loss,
            take_profit_1=entry_plan.take_profit_1,
            take_profit_2=entry_plan.take_profit_2,
            trailing_stop=entry_plan.trailing_stop,
            
            risk_reward=entry_plan.risk_reward,
            position_size_pct=entry_plan.position_size_pct,
            valid_bars=entry_plan.valid_bars,
            
            total_score=signal_data.score.total_score,
            score_breakdown=json.dumps(score_breakdown),
            ai_decision=ai_decision.decision,
            ai_confidence=ai_decision.confidence,
            ai_reasons=json.dumps(ai_decision.reasons),
            
            indicators_snapshot=json.dumps(indicators_snapshot),
            mtf_bias=signal_data.mtf_bias,
            vol_atr_ratio=signal_data.vol_atr_ratio,
            squeeze_status=json.dumps(signal_data.squeeze_status),
            
            status='PENDING',
            fill_percentage=0.0,
            
            pnl_pct=None,
            pnl_absolute=None,
            exit_timestamp=None,
            exit_reason=None,
            duration_minutes=None,
            
            created_at=now,
            updated_at=now
        )
    
    async def track_new_signal(self, signal_data: SignalData, ai_decision: AIDecision) -> str:
        """Track new signal and return signal code"""
        
        # Generate code
        signal_code = self.generate_signal_code(
            signal_data.symbol,
            signal_data.timeframe,
            signal_data.entry_plan.mode.value
        )
        
        # Create record
        record = await self.create_signal_record(signal_data, ai_decision, signal_code)
        
        # Save to database
        if await self.db.save_signal(record):
            self.active_signals[signal_code] = record
            logger.info(f"Tracking new signal: {signal_code}")
            return signal_code
        else:
            logger.error(f"Failed to save signal: {signal_code}")
            return None
    
    async def update_signal(self, code: str, update_type: str, details: Dict) -> bool:
        """Update signal status and details"""
        
        if code not in self.active_signals:
            logger.warning(f"Signal {code} not found in active signals")
            return False
        
        # Update database
        success = await self.db.update_signal_status(code, update_type, details)
        
        if success:
            # Update local cache
            signal = self.active_signals[code]
            signal.status = update_type
            signal.updated_at = datetime.now(timezone.utc).isoformat()
            
            # Update specific fields based on update type
            if update_type == 'FILLED':
                signal.entry_filled_price = details.get('entry_price')
                signal.entry_filled_timestamp = details.get('timestamp')
                signal.fill_percentage = 100.0
                
            elif update_type in ['TP1_HIT', 'TP2_HIT', 'SL_HIT']:
                signal.pnl_pct = details.get('pnl_pct')
                signal.pnl_absolute = details.get('pnl_absolute')
                signal.exit_timestamp = details.get('timestamp')
                signal.exit_reason = update_type
                
                if signal.entry_filled_timestamp:
                    entry_time = datetime.fromisoformat(signal.entry_filled_timestamp)
                    exit_time = datetime.fromisoformat(details.get('timestamp'))
                    signal.duration_minutes = int((exit_time - entry_time).total_seconds() / 60)
                
                # Remove from active if fully closed
                if update_type in ['TP2_HIT', 'SL_HIT']:
                    del self.active_signals[code]
            
            elif update_type in ['EXPIRED', 'CANCELLED']:
                signal.exit_timestamp = details.get('timestamp')
                signal.exit_reason = update_type
                del self.active_signals[code]
        
        return success
    
    async def get_performance_summary(self, days: int = 30) -> Dict:
        """Get performance summary"""
        return await self.db.get_performance_stats(days)
    
    async def cleanup_old_data(self):
        """Cleanup old data"""
        await self.db.cleanup_old_signals()
    
    def get_active_signal_codes(self) -> List[str]:
        """Get list of active signal codes"""
        return list(self.active_signals.keys())
    
    def get_active_signal(self, code: str) -> Optional[SignalRecord]:
        """Get active signal by code"""
        return self.active_signals.get(code)

# Factory function
def create_signal_tracker() -> SignalTracker:
    """Create signal tracker instance"""
    return SignalTracker()