#!/usr/bin/env python3
"""
Telegram Signal Bot for Futures Trading

Main application entry point that orchestrates the complete signal generation system
including webhook handling, technical analysis, AI verification, and notifications.
"""

import asyncio
import signal
import sys
from typing import Optional
from datetime import datetime
import traceback

from .config.settings import settings
from .core.webhook_binance import create_data_source
from .core.signal_pipeline import create_signal_pipeline
from .utils.logger import get_logger, setup_logger

# Initialize logger
logger = setup_logger(
    name="signal_bot_main",
    log_file=settings.LOG_FILE,
    log_level=settings.LOG_LEVEL
)

class SignalBotApplication:
    """Main application class orchestrating all components"""
    
    def __init__(self):
        self.pipeline = None
        self.data_source = None
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Background tasks
        self.data_source_task = None
        self.daily_summary_task = None
        self.cleanup_task = None
        
    async def start(self):
        """Start the complete signal bot application"""
        
        try:
            logger.info("=" * 60)
            logger.info("STARTING TELEGRAM SIGNAL BOT")
            logger.info("=" * 60)
            
            # Validate configuration
            await self._validate_configuration()
            
            # Initialize signal pipeline
            logger.info("Initializing signal pipeline...")
            self.pipeline = create_signal_pipeline()
            
            if not await self.pipeline.start():
                raise Exception("Failed to start signal pipeline")
            
            # Initialize data source
            logger.info("Initializing data source...")
            self.data_source = create_data_source(
                signal_processor_callback=self.pipeline.process_kline_data,
                use_webhook=True  # Change to False to use WebSocket
            )
            
            # Start data source
            if hasattr(self.data_source, 'start_server'):
                # Webhook mode
                self.data_source_task = asyncio.create_task(
                    self.data_source.start_server()
                )
                logger.info("Webhook server started")
            else:
                # WebSocket mode
                self.data_source_task = asyncio.create_task(
                    self.data_source.subscribe_to_streams()
                )
                logger.info("WebSocket streams started")
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.is_running = True
            
            logger.info("=" * 60)
            logger.info("SIGNAL BOT STARTED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"Monitoring {len(settings.PAIR_WHITELIST)} pairs on {len(settings.TIMEFRAMES)} timeframes")
            logger.info(f"Minimum signal score: {settings.MIN_SCORE_GATE2}")
            logger.info(f"Minimum AI confidence: {settings.MIN_AI_CONFIDENCE}")
            logger.info("Bot is ready to process signals...")
            
            return True
            
        except Exception as e:
            logger.critical(f"Failed to start application: {e}")
            logger.debug(traceback.format_exc())
            await self._emergency_cleanup()
            return False
    
    async def stop(self):
        """Stop the application gracefully"""
        
        logger.info("Shutting down signal bot...")
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Stop data source
        if self.data_source_task:
            self.data_source_task.cancel()
            try:
                await self.data_source_task
            except asyncio.CancelledError:
                pass
        
        if hasattr(self.data_source, 'stop'):
            self.data_source.stop()
        
        # Stop background tasks
        await self._stop_background_tasks()
        
        # Stop pipeline
        if self.pipeline:
            await self.pipeline.stop()
        
        logger.info("Signal bot stopped gracefully")
    
    async def run(self):
        """Run the application until shutdown signal"""
        
        if not await self.start():
            return False
        
        try:
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
        finally:
            await self.stop()
        
        return True
    
    async def _validate_configuration(self):
        """Validate application configuration"""
        
        logger.info("Validating configuration...")
        
        # Check required settings
        validation_errors = settings.validate()
        if validation_errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in validation_errors)
            raise Exception(error_msg)
        
        # Check pairs and timeframes
        if not settings.PAIR_WHITELIST:
            raise Exception("No trading pairs configured")
        
        if not settings.TIMEFRAMES:
            raise Exception("No timeframes configured")
        
        logger.info("Configuration validation passed")
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        
        # Daily summary task (runs every 24 hours)
        self.daily_summary_task = asyncio.create_task(
            self._daily_summary_loop()
        )
        
        # Cleanup task (runs every 6 hours)
        self.cleanup_task = asyncio.create_task(
            self._cleanup_loop()
        )
        
        logger.info("Background tasks started")
    
    async def _stop_background_tasks(self):
        """Stop background tasks"""
        
        tasks = [self.daily_summary_task, self.cleanup_task]
        
        for task in tasks:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Background tasks stopped")
    
    async def _daily_summary_loop(self):
        """Background task for sending daily summaries"""
        
        try:
            while self.is_running:
                # Wait for 24 hours or shutdown
                try:
                    await asyncio.wait_for(
                        self.shutdown_event.wait(),
                        timeout=24 * 60 * 60  # 24 hours
                    )
                    break  # Shutdown event received
                except asyncio.TimeoutError:
                    pass  # Continue with daily summary
                
                if self.is_running and self.pipeline:
                    try:
                        await self.pipeline.send_daily_summary()
                        logger.info("Daily summary sent")
                    except Exception as e:
                        logger.error(f"Error sending daily summary: {e}")
                
        except asyncio.CancelledError:
            logger.debug("Daily summary task cancelled")
        except Exception as e:
            logger.error(f"Error in daily summary loop: {e}")
    
    async def _cleanup_loop(self):
        """Background task for periodic cleanup"""
        
        try:
            while self.is_running:
                # Wait for 6 hours or shutdown
                try:
                    await asyncio.wait_for(
                        self.shutdown_event.wait(),
                        timeout=6 * 60 * 60  # 6 hours
                    )
                    break  # Shutdown event received
                except asyncio.TimeoutError:
                    pass  # Continue with cleanup
                
                if self.is_running and self.pipeline:
                    try:
                        await self.pipeline.cleanup_old_data()
                        logger.info("Periodic cleanup completed")
                    except Exception as e:
                        logger.error(f"Error in periodic cleanup: {e}")
                
        except asyncio.CancelledError:
            logger.debug("Cleanup task cancelled")
        except Exception as e:
            logger.error(f"Error in cleanup loop: {e}")
    
    async def _emergency_cleanup(self):
        """Emergency cleanup on startup failure"""
        
        try:
            if self.pipeline:
                await self.pipeline.stop()
        except Exception as e:
            logger.error(f"Error in emergency cleanup: {e}")
    
    def signal_handler(self, sig, frame):
        """Handle system signals for graceful shutdown"""
        
        logger.info(f"Received signal {sig}")
        self.shutdown_event.set()

# Global application instance
app: Optional[SignalBotApplication] = None

async def main():
    """Main entry point"""
    
    global app
    
    try:
        # Create application instance
        app = SignalBotApplication()
        
        # Set up signal handlers for graceful shutdown
        if sys.platform != 'win32':
            loop = asyncio.get_event_loop()
            for sig in [signal.SIGINT, signal.SIGTERM]:
                loop.add_signal_handler(
                    sig, 
                    lambda: app.shutdown_event.set()
                )
        
        # Run application
        success = await app.run()
        
        if success:
            logger.info("Application completed successfully")
            return 0
        else:
            logger.error("Application failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 0
    except Exception as e:
        logger.critical(f"Critical error in main: {e}")
        logger.debug(traceback.format_exc())
        return 1
    finally:
        if app:
            try:
                await app.stop()
            except Exception as e:
                logger.error(f"Error in final cleanup: {e}")

def run_bot():
    """Synchronous entry point for running the bot"""
    
    try:
        # Create new event loop for clean startup
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the main coroutine
        exit_code = loop.run_until_complete(main())
        
        return exit_code
        
    except KeyboardInterrupt:
        logger.info("Bot interrupted by user")
        return 0
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        return 1
    finally:
        try:
            # Clean up the loop
            loop.close()
        except Exception:
            pass

# Development/testing utilities
async def test_pipeline():
    """Test the signal pipeline with sample data"""
    
    logger.info("Testing signal pipeline...")
    
    # Sample kline data for testing
    sample_kline = {
        'symbol': 'BTCUSDT',
        'interval': '15m',
        'open_time': 1640995200000,
        'close_time': 1640996100000,
        'open': '47000.00',
        'high': '47500.00',
        'low': '46800.00',
        'close': '47200.00',
        'volume': '150.5',
        'quote_volume': '7100000.0',
        'trades': 1250,
        'taker_buy_base_volume': '75.2',
        'taker_buy_quote_volume': '3550000.0',
        'is_closed': True
    }
    
    try:
        pipeline = create_signal_pipeline()
        await pipeline.start()
        
        # Process sample data
        await pipeline.process_kline_data('BTCUSDT', '15m', sample_kline)
        
        # Get status
        status = pipeline.get_pipeline_status()
        logger.info(f"Pipeline status: {status}")
        
        await pipeline.stop()
        
        logger.info("Pipeline test completed")
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        raise

async def health_check():
    """Perform system health check"""
    
    logger.info("Performing health check...")
    
    try:
        pipeline = create_signal_pipeline()
        
        # Basic health check
        healthy = await pipeline.validate_system_health()
        
        if healthy:
            logger.info("✓ System health check passed")
        else:
            logger.error("✗ System health check failed")
        
        return healthy
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return False

if __name__ == "__main__":
    # Command line interface
    import argparse
    
    parser = argparse.ArgumentParser(description="Telegram Signal Bot")
    parser.add_argument('--test', action='store_true', help='Run pipeline test')
    parser.add_argument('--health', action='store_true', help='Run health check')
    parser.add_argument('--config', help='Path to configuration file')
    
    args = parser.parse_args()
    
    if args.test:
        asyncio.run(test_pipeline())
    elif args.health:
        asyncio.run(health_check())
    else:
        exit_code = run_bot()
        sys.exit(exit_code)