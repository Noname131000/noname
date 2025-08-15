import asyncio
import aiohttp
from typing import Dict, List, Optional
from datetime import datetime
import html
import json

from .scoring import SignalData, EntryPlan, SignalDirection
from .ai_gemini import AIDecision
from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger()

class TelegramNotifier:
    """Professional Telegram notification system for trading signals"""
    
    def __init__(self):
        self.bot_token = settings.TELEGRAM_BOT_TOKEN
        self.chat_id = settings.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.session = None
        self.message_queue = asyncio.Queue()
        self.is_running = False
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def start_message_processor(self):
        """Start background message processor"""
        self.is_running = True
        while self.is_running:
            try:
                # Process messages from queue with rate limiting
                message, parse_mode = await asyncio.wait_for(
                    self.message_queue.get(), timeout=1.0
                )
                await self._send_message(message, parse_mode)
                await asyncio.sleep(1)  # Rate limiting
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Message processor error: {e}")
    
    def stop_message_processor(self):
        """Stop background message processor"""
        self.is_running = False
    
    async def send_new_signal(self, signal_data: SignalData, ai_decision: AIDecision, 
                            signal_code: str) -> bool:
        """Send new signal notification to Telegram"""
        
        if not signal_data.entry_plan:
            logger.warning("Cannot send signal without entry plan")
            return False
        
        try:
            message = self._format_new_signal(signal_data, ai_decision, signal_code)
            await self._queue_message(message, "HTML")
            
            logger.info(f"Queued new signal notification: {signal_code}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending new signal: {e}")
            return False
    
    async def send_signal_update(self, signal_code: str, pair: str, entry_price: float,
                               signal_direction: str, update_type: str, 
                               details: Dict) -> bool:
        """Send signal update notification"""
        
        try:
            message = self._format_signal_update(
                signal_code, pair, entry_price, signal_direction, update_type, details
            )
            await self._queue_message(message, "HTML")
            
            logger.info(f"Queued signal update: {signal_code} - {update_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending signal update: {e}")
            return False
    
    async def send_system_notification(self, title: str, message: str, 
                                     level: str = "INFO") -> bool:
        """Send system notification"""
        
        try:
            formatted_message = self._format_system_notification(title, message, level)
            await self._queue_message(formatted_message, "HTML")
            
            logger.info(f"Queued system notification: {title}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending system notification: {e}")
            return False
    
    def _format_new_signal(self, signal_data: SignalData, ai_decision: AIDecision, 
                          signal_code: str) -> str:
        """Format new signal message"""
        
        entry_plan = signal_data.entry_plan
        indicators = signal_data.indicators
        
        # Determine signal emoji
        direction_emoji = "🟢" if entry_plan.direction == SignalDirection.LONG else "🔴"
        
        # Format VOL/ATR with interpretation
        vol_atr = signal_data.vol_atr_ratio
        vol_atr_text = f"{vol_atr:.2f}"
        if vol_atr >= 1.0:
            vol_atr_text += " (STRONG)"
        elif vol_atr >= 0.7:
            vol_atr_text += " (MODERATE)"
        else:
            vol_atr_text += " (WEAK)"
        
        # Build AI note
        ai_note = ai_decision.overall_assessment
        if ai_decision.reasons:
            ai_note += f" • {'; '.join(ai_decision.reasons[:2])}"
        
        # Format message
        message = f"""
{direction_emoji} <b>NEW SIGNAL</b>

<b>PAIR</b>          : <code>{signal_data.symbol}</code>
<b>SIGNAL</b>        : <code>{entry_plan.direction.value}</code>
<b>ENTRY LIMIT</b>   : <code>{entry_plan.entry_limit:.5f}</code> (valid {entry_plan.valid_bars} bars)
<b>TP/SL</b>         : TP1=<code>{entry_plan.take_profit_1:.5f}</code> | TP2=<code>{entry_plan.take_profit_2:.5f}</code> | SL=<code>{entry_plan.stop_loss:.5f}</code>
<b>VOL/ATR</b>       : <code>{vol_atr_text}</code>
<b>AI NOTE</b>       : {html.escape(ai_note)}
<b>CODE</b>          : <code>{signal_code}</code>

📊 <b>TECHNICAL DETAILS</b>
• Score: {signal_data.score.total_score:.0f}/100 | R:R {entry_plan.risk_reward:.1f}
• ADX: {indicators.adx:.1f} | RSI: {indicators.rsi:.1f} | Vol: {indicators.relative_volume:.1f}x
• {signal_data.timeframe} {entry_plan.mode.value} | MTF: {signal_data.mtf_bias}
• AI: {ai_decision.decision} ({ai_decision.confidence}%)
"""
        
        # Add warnings if any
        if signal_data.warnings or ai_decision.concerns:
            warnings = signal_data.warnings + ai_decision.concerns
            message += f"\n⚠️ <b>WARNINGS</b>: {'; '.join(warnings[:2])}"
        
        return message.strip()
    
    def _format_signal_update(self, signal_code: str, pair: str, entry_price: float,
                            signal_direction: str, update_type: str, details: Dict) -> str:
        """Format signal update message"""
        
        # Determine update emoji
        update_emojis = {
            "TP1_HIT": "🎯",
            "TP2_HIT": "🎯🎯", 
            "SL_HIT": "🛑",
            "ENTRY_FILLED": "✅",
            "TRAILING": "📈",
            "EXPIRED": "⏰",
            "CANCELLED": "❌"
        }
        
        emoji = update_emojis.get(update_type, "📊")
        
        # Format info based on update type
        if update_type in ["TP1_HIT", "TP2_HIT"]:
            pnl = details.get('pnl_pct', 0)
            pnl_text = f"(+{pnl:.2f}%)" if pnl > 0 else f"({pnl:.2f}%)"
            info = f"{update_type.replace('_', ' ')} {pnl_text}"
            if update_type == "TP1_HIT":
                info += " | REMAINING POSITION TRAILING"
                
        elif update_type == "SL_HIT":
            pnl = details.get('pnl_pct', 0)
            pnl_text = f"({pnl:.2f}%)"
            info = f"STOP LOSS HIT {pnl_text}"
            
        elif update_type == "ENTRY_FILLED":
            info = f"ENTRY FILLED @ {entry_price:.5f}"
            
        elif update_type == "TRAILING":
            new_sl = details.get('new_stop_loss', 0)
            info = f"TRAILING STOP UPDATED TO {new_sl:.5f}"
            
        elif update_type == "EXPIRED":
            info = "ENTRY LIMIT EXPIRED"
            
        else:
            info = update_type.replace('_', ' ')
        
        message = f"""
{emoji} <b>SIGNAL UPDATE</b>

<b>CODE</b>          : <code>{signal_code}</code>
<b>PAIR</b>          : <code>{pair}</code>
<b>ENTRY</b>         : <code>{entry_price:.5f}</code>
<b>SIGNAL</b>        : <code>{signal_direction}</code>
<b>INFO</b>          : {info}
"""
        
        return message.strip()
    
    def _format_system_notification(self, title: str, message: str, level: str) -> str:
        """Format system notification message"""
        
        level_emojis = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "🚨",
            "SUCCESS": "✅"
        }
        
        emoji = level_emojis.get(level, "📢")
        
        formatted_message = f"""
{emoji} <b>{title}</b>

{html.escape(message)}

<i>Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        
        return formatted_message.strip()
    
    async def _queue_message(self, message: str, parse_mode: str = "HTML"):
        """Add message to queue for processing"""
        await self.message_queue.put((message, parse_mode))
    
    async def _send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send message directly to Telegram"""
        
        if not self.bot_token or not self.chat_id:
            logger.error("Telegram bot token or chat ID not configured")
            return False
        
        url = f"{self.base_url}/sendMessage"
        
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        }
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.post(url, json=payload, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('ok'):
                        return True
                    else:
                        logger.error(f"Telegram API error: {data}")
                        return False
                else:
                    error_text = await response.text()
                    logger.error(f"Telegram HTTP error {response.status}: {error_text}")
                    return False
                    
        except asyncio.TimeoutError:
            logger.error("Telegram request timeout")
            return False
        except Exception as e:
            logger.error(f"Telegram request failed: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test Telegram bot connection"""
        
        url = f"{self.base_url}/getMe"
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('ok'):
                        bot_info = data.get('result', {})
                        logger.info(f"Telegram bot connected: {bot_info.get('username', 'Unknown')}")
                        return True
                    else:
                        logger.error(f"Telegram bot test failed: {data}")
                        return False
                else:
                    logger.error(f"Telegram connection test failed: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Telegram connection test error: {e}")
            return False
    
    async def send_startup_message(self) -> bool:
        """Send bot startup notification"""
        
        message = f"""
🚀 <b>SIGNAL BOT STARTED</b>

<b>Timestamp</b>: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
<b>Pairs</b>: {', '.join(settings.PAIR_WHITELIST[:5])}{'...' if len(settings.PAIR_WHITELIST) > 5 else ''}
<b>Timeframes</b>: {', '.join(settings.TIMEFRAMES)}
<b>Min Score</b>: {settings.MIN_SCORE_GATE2}
<b>Min R:R</b>: {settings.MIN_RR}

Bot is now monitoring markets for signals...
"""
        
        return await self._send_message(message.strip())
    
    async def send_shutdown_message(self) -> bool:
        """Send bot shutdown notification"""
        
        message = f"""
🛑 <b>SIGNAL BOT STOPPED</b>

<b>Timestamp</b>: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Bot has been shut down.
"""
        
        return await self._send_message(message.strip())
    
    async def send_performance_summary(self, stats: Dict) -> bool:
        """Send performance summary"""
        
        total_signals = stats.get('total_signals', 0)
        wins = stats.get('wins', 0)
        losses = stats.get('losses', 0)
        win_rate = (wins / total_signals * 100) if total_signals > 0 else 0
        total_pnl = stats.get('total_pnl_pct', 0)
        
        message = f"""
📊 <b>PERFORMANCE SUMMARY</b>

<b>Total Signals</b>: {total_signals}
<b>Wins</b>: {wins} | <b>Losses</b>: {losses}
<b>Win Rate</b>: {win_rate:.1f}%
<b>Total P&L</b>: {total_pnl:+.2f}%
<b>Avg per Trade</b>: {(total_pnl/total_signals):+.2f}% if total_signals > 0 else 0

<b>Best Performer</b>: {stats.get('best_pair', 'N/A')}
<b>Period</b>: {stats.get('period', 'Unknown')}
"""
        
        return await self._send_message(message.strip())
    
    def format_signal_summary_for_ai(self, signal_data: SignalData) -> str:
        """Create formatted summary for AI context"""
        
        entry_plan = signal_data.entry_plan
        if not entry_plan:
            return "No valid signal"
        
        return (
            f"{signal_data.symbol} {signal_data.timeframe} "
            f"{entry_plan.direction.value} @ {entry_plan.entry_limit:.5f} "
            f"(Score: {signal_data.score.total_score:.0f}, R:R: {entry_plan.risk_reward:.1f})"
        )

class NotificationManager:
    """Manage all notification workflows"""
    
    def __init__(self):
        self.telegram = TelegramNotifier()
        self.message_processor_task = None
    
    async def start(self):
        """Start notification manager"""
        await self.telegram.__aenter__()
        
        # Test connection
        if not await self.telegram.test_connection():
            logger.error("Failed to connect to Telegram")
            return False
        
        # Start message processor
        self.message_processor_task = asyncio.create_task(
            self.telegram.start_message_processor()
        )
        
        # Send startup message
        await self.telegram.send_startup_message()
        
        logger.info("Notification manager started")
        return True
    
    async def stop(self):
        """Stop notification manager"""
        if self.message_processor_task:
            self.telegram.stop_message_processor()
            self.message_processor_task.cancel()
            try:
                await self.message_processor_task
            except asyncio.CancelledError:
                pass
        
        await self.telegram.send_shutdown_message()
        await asyncio.sleep(2)  # Allow final message to send
        
        await self.telegram.__aexit__(None, None, None)
        logger.info("Notification manager stopped")
    
    async def notify_new_signal(self, signal_data: SignalData, ai_decision: AIDecision,
                              signal_code: str) -> bool:
        """Send new signal notification"""
        return await self.telegram.send_new_signal(signal_data, ai_decision, signal_code)
    
    async def notify_signal_update(self, signal_code: str, pair: str, entry_price: float,
                                 signal_direction: str, update_type: str, 
                                 details: Dict) -> bool:
        """Send signal update notification"""
        return await self.telegram.send_signal_update(
            signal_code, pair, entry_price, signal_direction, update_type, details
        )
    
    async def notify_system_event(self, title: str, message: str, level: str = "INFO") -> bool:
        """Send system notification"""
        return await self.telegram.send_system_notification(title, message, level)
    
    async def send_daily_summary(self, stats: Dict) -> bool:
        """Send daily performance summary"""
        return await self.telegram.send_performance_summary(stats)

# Factory function
def create_notification_manager() -> NotificationManager:
    """Create notification manager instance"""
    return NotificationManager()