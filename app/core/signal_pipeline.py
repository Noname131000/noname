import asyncio
from typing import Dict, Optional
from datetime import datetime, timezone

from .indicators import TechnicalIndicators, IndicatorData
from .scoring import SignalScorer, SignalData
from .ai_gemini import GeminiAIVerifier, AIDecision
from .notifier import NotificationManager
from .storage import SignalTracker
from ..config.settings import settings
from ..utils.logger import get_logger, log_signal_data

logger = get_logger()

class SignalPipeline:
    """Main signal processing pipeline orchestrating all gates"""
    
    def __init__(self):
        # Core components
        self.indicators = TechnicalIndicators()
        self.scorer = SignalScorer(self.indicators)
        self.ai_verifier = None  # Initialized async
        self.notifier = NotificationManager()
        self.tracker = SignalTracker()
        
        # State tracking
        self.is_running = False
        self.processed_count = 0
        self.signals_sent = 0
        
        # Performance tracking
        self.gate_stats = {
            'gate1_processed': 0,
            'gate2_passed': 0,
            'gate3_passed': 0,
            'gate4_sent': 0,
            'gate2_failed': 0,
            'gate3_failed': 0
        }
    
    async def start(self):
        """Start the signal pipeline"""
        try:
            logger.info("Starting signal pipeline...")
            
            # Start AI verifier
            self.ai_verifier = GeminiAIVerifier()
            await self.ai_verifier.__aenter__()
            
            # Start notification manager
            await self.notifier.start()
            
            # Start signal tracker
            await self.tracker.start()
            
            self.is_running = True
            
            logger.info("Signal pipeline started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start signal pipeline: {e}")
            return False
    
    async def stop(self):
        """Stop the signal pipeline"""
        logger.info("Stopping signal pipeline...")
        
        self.is_running = False
        
        # Stop components
        if self.ai_verifier:
            await self.ai_verifier.__aexit__(None, None, None)
        
        await self.notifier.stop()
        await self.tracker.stop()
        
        # Log final stats
        logger.info(f"Pipeline stopped. Stats: {self.gate_stats}")
    
    async def process_kline_data(self, symbol: str, timeframe: str, kline_data: Dict):
        """
        Main entry point for processing kline data through all gates
        
        Args:
            symbol: Trading pair symbol
            timeframe: Chart timeframe
            kline_data: Normalized kline data from webhook/websocket
        """
        
        if not self.is_running:
            return
        
        try:
            logger.debug(f"Processing kline data: {symbol} {timeframe}")
            
            # GATE 1: Check Candle & Calculate Indicators
            indicators = await self._gate1_process_candle(symbol, timeframe, kline_data)
            if not indicators:
                return
            
            # GATE 2: Indicator Combination & Scoring
            signal_data = await self._gate2_score_signal(symbol, timeframe, indicators)
            if not signal_data or not self.scorer.passes_gate2(signal_data):
                self.gate_stats['gate2_failed'] += 1
                log_signal_data(logger, signal_data.__dict__ if signal_data else {}, "GATE2_FAILED")
                return
            
            self.gate_stats['gate2_passed'] += 1
            log_signal_data(logger, signal_data.__dict__, "GATE2_PASSED")
            
            # GATE 3: AI Verification
            ai_decision = await self._gate3_ai_verification(signal_data)
            if not self.ai_verifier.passes_gate3(ai_decision):
                self.gate_stats['gate3_failed'] += 1
                log_signal_data(logger, signal_data.__dict__, "GATE3_FAILED")
                await self._notify_ai_rejection(signal_data, ai_decision)
                return
            
            self.gate_stats['gate3_passed'] += 1
            log_signal_data(logger, signal_data.__dict__, "GATE3_PASSED")
            
            # GATE 4: Send to Telegram & Track
            await self._gate4_send_signal(signal_data, ai_decision)
            self.gate_stats['gate4_sent'] += 1
            self.signals_sent += 1
            
            # Update cooldowns
            self.scorer.update_cooldowns()
            
        except Exception as e:
            logger.error(f"Error in signal pipeline for {symbol} {timeframe}: {e}")
    
    async def _gate1_process_candle(self, symbol: str, timeframe: str, 
                                  kline_data: Dict) -> Optional[IndicatorData]:
        """
        GATE 1: Process candle data and calculate technical indicators
        
        Returns:
            IndicatorData if successful, None if insufficient data
        """
        
        try:
            # Update indicators with new kline data
            indicators = self.indicators.update_data(symbol, timeframe, kline_data)
            
            self.gate_stats['gate1_processed'] += 1
            self.processed_count += 1
            
            log_signal_data(logger, {
                'symbol': symbol,
                'timeframe': timeframe,
                'ohlc': indicators.ohlc,
                'volume': indicators.volume
            }, "GATE1")
            
            return indicators
            
        except Exception as e:
            logger.error(f"Gate 1 error for {symbol} {timeframe}: {e}")
            return None
    
    async def _gate2_score_signal(self, symbol: str, timeframe: str, 
                                indicators: IndicatorData) -> Optional[SignalData]:
        """
        GATE 2: Score signal using indicator combination
        
        Returns:
            SignalData if valid signal found, None otherwise
        """
        
        try:
            # Evaluate signal
            signal_data = self.scorer.evaluate_signal(symbol, timeframe, indicators)
            
            log_signal_data(logger, {
                'pair': symbol,
                'timeframe': timeframe,
                'score': signal_data.score.total_score,
                'trend_filters': signal_data.score.trend_details,
                'risk_reward': signal_data.entry_plan.risk_reward if signal_data.entry_plan else 0
            }, "GATE2")
            
            return signal_data
            
        except Exception as e:
            logger.error(f"Gate 2 error for {symbol} {timeframe}: {e}")
            return None
    
    async def _gate3_ai_verification(self, signal_data: SignalData) -> AIDecision:
        """
        GATE 3: AI verification of signal
        
        Returns:
            AIDecision with verification result
        """
        
        try:
            # Get AI verification
            ai_decision = await self.ai_verifier.verify_signal(signal_data)
            
            log_signal_data(logger, {
                'pair': signal_data.symbol,
                'timeframe': signal_data.timeframe,
                'ai_decision': ai_decision.decision,
                'ai_confidence': ai_decision.confidence,
                'ai_reasons': ai_decision.reasons
            }, "GATE3")
            
            return ai_decision
            
        except Exception as e:
            logger.error(f"Gate 3 error for {signal_data.symbol} {signal_data.timeframe}: {e}")
            # Return fallback decision on error
            return AIDecision(
                decision="REJECT",
                confidence=0,
                reasons=["AI verification failed"],
                concerns=["System error"],
                overall_assessment="Unable to verify signal"
            )
    
    async def _gate4_send_signal(self, signal_data: SignalData, ai_decision: AIDecision):
        """
        GATE 4: Send signal to Telegram and start tracking
        """
        
        try:
            # Track signal first to get code
            signal_code = await self.tracker.track_new_signal(signal_data, ai_decision)
            
            if not signal_code:
                logger.error("Failed to generate signal code")
                return
            
            # Send to Telegram
            success = await self.notifier.notify_new_signal(signal_data, ai_decision, signal_code)
            
            if success:
                log_signal_data(logger, {
                    'pair': signal_data.symbol,
                    'timeframe': signal_data.timeframe,
                    'code': signal_code
                }, "GATE4")
                
                logger.info(f"Signal sent successfully: {signal_code}")
            else:
                logger.error(f"Failed to send signal notification: {signal_code}")
                
                # Update tracker with failed notification
                await self.tracker.update_signal(signal_code, 'CANCELLED', {
                    'reason': 'Failed to send notification',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
            
        except Exception as e:
            logger.error(f"Gate 4 error: {e}")
    
    async def _notify_ai_rejection(self, signal_data: SignalData, ai_decision: AIDecision):
        """Send notification about AI rejection if it's a high-scoring signal"""
        
        # Only notify for high-scoring signals that were rejected by AI
        if (signal_data.score.total_score >= 75 and 
            ai_decision.decision == "REJECT" and 
            ai_decision.confidence < 30):
            
            await self.notifier.notify_system_event(
                "High Score Signal Rejected by AI",
                f"Signal for {signal_data.symbol} {signal_data.timeframe} "
                f"scored {signal_data.score.total_score:.0f}/100 but was rejected by AI. "
                f"Reasons: {'; '.join(ai_decision.reasons[:2])}",
                "WARNING"
            )
    
    async def process_signal_update(self, signal_code: str, update_type: str, details: Dict):
        """Process signal update (TP/SL hits, entry fills, etc.)"""
        
        try:
            # Get signal from tracker
            signal = self.tracker.get_active_signal(signal_code)
            if not signal:
                logger.warning(f"Signal {signal_code} not found for update")
                return False
            
            # Update tracker
            success = await self.tracker.update_signal(signal_code, update_type, details)
            
            if success:
                # Send update notification
                await self.notifier.notify_signal_update(
                    signal_code,
                    signal.symbol,
                    signal.entry_filled_price or signal.entry_limit,
                    signal.direction,
                    update_type,
                    details
                )
                
                # Handle stop loss for cooldown
                if update_type == 'SL_HIT':
                    self.scorer.add_cooldown(signal.symbol, signal.timeframe)
                
                logger.info(f"Processed signal update: {signal_code} - {update_type}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error processing signal update {signal_code}: {e}")
            return False
    
    async def send_daily_summary(self):
        """Send daily performance summary"""
        
        try:
            # Get performance stats
            stats = await self.tracker.get_performance_summary(1)  # Last 24 hours
            
            # Add pipeline stats
            stats.update({
                'total_processed': self.processed_count,
                'signals_generated': self.signals_sent,
                'gate2_pass_rate': (self.gate_stats['gate2_passed'] / 
                                  max(1, self.gate_stats['gate1_processed'])) * 100,
                'gate3_pass_rate': (self.gate_stats['gate3_passed'] / 
                                  max(1, self.gate_stats['gate2_passed'])) * 100
            })
            
            await self.notifier.send_daily_summary(stats)
            
            # Reset daily stats
            self.gate_stats = {key: 0 for key in self.gate_stats}
            self.processed_count = 0
            self.signals_sent = 0
            
        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")
    
    async def cleanup_old_data(self):
        """Periodic cleanup of old data"""
        
        try:
            await self.tracker.cleanup_old_data()
            logger.info("Completed periodic data cleanup")
            
        except Exception as e:
            logger.error(f"Error in data cleanup: {e}")
    
    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status"""
        
        return {
            'is_running': self.is_running,
            'processed_count': self.processed_count,
            'signals_sent': self.signals_sent,
            'gate_stats': self.gate_stats.copy(),
            'active_signals': self.tracker.get_active_signal_codes(),
            'indicators_cache_size': len(self.indicators.price_history),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def handle_emergency_stop(self, reason: str):
        """Emergency stop with notification"""
        
        logger.critical(f"Emergency stop triggered: {reason}")
        
        await self.notifier.notify_system_event(
            "EMERGENCY STOP",
            f"Signal pipeline stopped due to: {reason}",
            "ERROR"
        )
        
        await self.stop()
    
    async def validate_system_health(self) -> bool:
        """Validate system health and dependencies"""
        
        try:
            # Test AI connection
            if not self.ai_verifier:
                return False
            
            # Test Telegram connection
            # (This would be called during startup)
            
            # Check database connection
            if not self.tracker.db.connection:
                return False
            
            # Validate settings
            validation_errors = settings.validate()
            if validation_errors:
                logger.error(f"Configuration errors: {validation_errors}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"System health check failed: {e}")
            return False
    
    async def process_market_hours_check(self) -> bool:
        """Check if markets are open (optional feature)"""
        
        # For crypto, markets are always open
        # For forex, you could implement trading hours logic here
        return True
    
    def get_performance_metrics(self) -> Dict:
        """Get detailed performance metrics"""
        
        total_processed = max(1, self.gate_stats['gate1_processed'])
        
        return {
            'pipeline_efficiency': {
                'gate1_to_gate2': (self.gate_stats['gate2_passed'] / total_processed) * 100,
                'gate2_to_gate3': (self.gate_stats['gate3_passed'] / 
                                 max(1, self.gate_stats['gate2_passed'])) * 100,
                'gate3_to_gate4': (self.gate_stats['gate4_sent'] / 
                                 max(1, self.gate_stats['gate3_passed'])) * 100,
                'overall_conversion': (self.gate_stats['gate4_sent'] / total_processed) * 100
            },
            'volume_metrics': {
                'total_processed': total_processed,
                'signals_generated': self.gate_stats['gate4_sent'],
                'avg_processing_rate': total_processed / max(1, 
                    (datetime.now(timezone.utc).hour + 1))  # Rough hourly rate
            },
            'quality_metrics': {
                'gate2_rejection_rate': (self.gate_stats['gate2_failed'] / total_processed) * 100,
                'gate3_rejection_rate': (self.gate_stats['gate3_failed'] / 
                                       max(1, self.gate_stats['gate2_passed'])) * 100
            }
        }

# Factory function
def create_signal_pipeline() -> SignalPipeline:
    """Create signal pipeline instance"""
    return SignalPipeline()