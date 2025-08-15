import asyncio
import json
import hmac
import hashlib
import time
from typing import Dict, Optional, Callable
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime, timezone

from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger()

class BinanceWebhookHandler:
    """Handle Binance webhook events for kline close data"""
    
    def __init__(self, signal_processor_callback: Callable):
        self.app = FastAPI(title="Binance Webhook Handler")
        self.signal_processor = signal_processor_callback
        self.debounce_cache = {}  # Prevent duplicate processing
        self.setup_routes()
    
    def setup_routes(self):
        """Setup webhook routes"""
        
        @self.app.post("/webhook/binance")
        async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
            """Receive and process Binance webhook data"""
            try:
                # Get raw body and headers
                body = await request.body()
                headers = dict(request.headers)
                
                # Validate webhook signature
                if not self._validate_signature(body, headers):
                    logger.warning("Invalid webhook signature received")
                    raise HTTPException(status_code=401, detail="Invalid signature")
                
                # Parse JSON data
                try:
                    data = json.loads(body.decode('utf-8'))
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in webhook: {e}")
                    raise HTTPException(status_code=400, detail="Invalid JSON")
                
                # Validate webhook data structure
                if not self._validate_webhook_data(data):
                    logger.warning("Invalid webhook data structure")
                    raise HTTPException(status_code=400, detail="Invalid data structure")
                
                # Process in background to avoid blocking
                background_tasks.add_task(self._process_kline_data, data)
                
                return JSONResponse({"status": "received", "timestamp": time.time()})
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Webhook processing error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": time.time()}
        
        @self.app.get("/webhook/status")
        async def webhook_status():
            """Webhook status endpoint"""
            return {
                "status": "active",
                "pairs": settings.PAIR_WHITELIST,
                "timeframes": settings.TIMEFRAMES,
                "debounce_cache_size": len(self.debounce_cache)
            }
    
    def _validate_signature(self, body: bytes, headers: Dict[str, str]) -> bool:
        """Validate webhook signature"""
        if not settings.WEBHOOK_SECRET:
            logger.warning("No webhook secret configured - skipping validation")
            return True
        
        signature = headers.get('x-signature') or headers.get('signature')
        if not signature:
            return False
        
        expected_signature = hmac.new(
            settings.WEBHOOK_SECRET.encode('utf-8'),
            body,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
    
    def _validate_webhook_data(self, data: Dict) -> bool:
        """Validate webhook data structure"""
        required_fields = ['e', 's', 'k']  # event type, symbol, kline
        
        # Check main structure
        for field in required_fields:
            if field not in data:
                return False
        
        # Check if it's a kline event
        if data['e'] != 'kline':
            return False
        
        # Check kline data structure
        kline = data['k']
        required_kline_fields = ['s', 't', 'T', 'o', 'h', 'l', 'c', 'v', 'x']
        for field in required_kline_fields:
            if field not in kline:
                return False
        
        # Check if kline is closed
        if not kline['x']:  # x = is kline closed
            return False
        
        return True
    
    async def _process_kline_data(self, webhook_data: Dict):
        """Process kline close data"""
        try:
            kline = webhook_data['k']
            symbol = kline['s']
            
            # Check if symbol is in whitelist
            if symbol not in settings.PAIR_WHITELIST:
                logger.debug(f"Symbol {symbol} not in whitelist, ignoring")
                return
            
            # Extract timeframe from interval
            interval = kline['i']
            if interval not in settings.TIMEFRAMES:
                logger.debug(f"Timeframe {interval} not monitored, ignoring")
                return
            
            # Create debounce key
            close_time = kline['T']
            debounce_key = f"{symbol}_{interval}_{close_time}"
            
            # Check debounce cache
            current_time = time.time()
            if debounce_key in self.debounce_cache:
                if current_time - self.debounce_cache[debounce_key] < 5:  # 5 second debounce
                    logger.debug(f"Debouncing duplicate webhook for {debounce_key}")
                    return
            
            self.debounce_cache[debounce_key] = current_time
            
            # Clean old debounce entries (older than 1 minute)
            self._clean_debounce_cache(current_time)
            
            # Normalize kline data
            normalized_data = self._normalize_kline_data(kline)
            
            logger.info(f"Processing kline close: {symbol} {interval} at {normalized_data['close_time']}")
            
            # Send to signal processor
            try:
                await self.signal_processor(symbol, interval, normalized_data)
            except Exception as e:
                logger.error(f"Signal processor error for {symbol} {interval}: {e}")
            
        except Exception as e:
            logger.error(f"Error processing kline data: {e}")
    
    def _normalize_kline_data(self, kline: Dict) -> Dict:
        """Normalize kline data to standard format"""
        return {
            'symbol': kline['s'],
            'interval': kline['i'],
            'open_time': int(kline['t']),
            'close_time': int(kline['T']),
            'open': kline['o'],
            'high': kline['h'],
            'low': kline['l'],
            'close': kline['c'],
            'volume': kline['v'],
            'quote_volume': kline['q'],
            'trades': int(kline['n']),
            'taker_buy_base_volume': kline['V'],
            'taker_buy_quote_volume': kline['Q'],
            'is_closed': kline['x']
        }
    
    def _clean_debounce_cache(self, current_time: float):
        """Clean old entries from debounce cache"""
        cutoff_time = current_time - 60  # 1 minute
        keys_to_remove = [
            key for key, timestamp in self.debounce_cache.items()
            if timestamp < cutoff_time
        ]
        
        for key in keys_to_remove:
            del self.debounce_cache[key]
    
    async def start_server(self):
        """Start the webhook server"""
        config = uvicorn.Config(
            self.app,
            host=settings.WEBHOOK_HOST,
            port=settings.WEBHOOK_PORT,
            log_level="info",
            access_log=False
        )
        
        server = uvicorn.Server(config)
        logger.info(f"Starting webhook server on {settings.WEBHOOK_HOST}:{settings.WEBHOOK_PORT}")
        
        try:
            await server.serve()
        except Exception as e:
            logger.error(f"Webhook server error: {e}")
            raise

class BinanceWebhookSubscriber:
    """Subscribe to Binance websocket streams for kline data"""
    
    def __init__(self, signal_processor_callback: Callable):
        self.signal_processor = signal_processor_callback
        self.connections = {}
        self.running = False
    
    async def subscribe_to_streams(self):
        """Subscribe to Binance websocket streams for all pairs/timeframes"""
        import websockets
        
        # Build stream names
        streams = []
        for symbol in settings.PAIR_WHITELIST:
            for tf in settings.TIMEFRAMES:
                stream = f"{symbol.lower()}@kline_{tf}"
                streams.append(stream)
        
        if not streams:
            logger.warning("No streams to subscribe to")
            return
        
        # Binance allows max 1024 streams per connection
        # Split into chunks if needed
        chunk_size = 200  # Conservative limit
        stream_chunks = [streams[i:i + chunk_size] for i in range(0, len(streams), chunk_size)]
        
        logger.info(f"Subscribing to {len(streams)} streams in {len(stream_chunks)} connections")
        
        # Create tasks for each connection
        tasks = []
        for i, chunk in enumerate(stream_chunks):
            task = asyncio.create_task(self._connect_to_stream_chunk(chunk, i))
            tasks.append(task)
        
        self.running = True
        
        try:
            # Wait for all connections
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Stream subscription error: {e}")
            self.running = False
            raise
    
    async def _connect_to_stream_chunk(self, streams: list, connection_id: int):
        """Connect to a chunk of streams"""
        stream_names = "/".join(streams)
        ws_url = f"wss://stream.binance.com:9443/ws/{stream_names}"
        
        logger.info(f"Connection {connection_id}: Connecting to {len(streams)} streams")
        
        while self.running:
            try:
                async with websockets.connect(ws_url) as websocket:
                    logger.info(f"Connection {connection_id}: Connected successfully")
                    self.connections[connection_id] = websocket
                    
                    async for message in websocket:
                        if not self.running:
                            break
                        
                        try:
                            data = json.loads(message)
                            
                            # Handle stream data
                            if 'stream' in data and 'data' in data:
                                stream_data = data['data']
                                if stream_data.get('e') == 'kline' and stream_data['k']['x']:  # kline closed
                                    await self._process_websocket_kline(stream_data)
                        
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON from websocket connection {connection_id}")
                        except Exception as e:
                            logger.error(f"Error processing websocket message: {e}")
            
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"Connection {connection_id}: Websocket closed, reconnecting...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Connection {connection_id}: Websocket error: {e}")
                await asyncio.sleep(10)
    
    async def _process_websocket_kline(self, data: Dict):
        """Process kline data from websocket"""
        try:
            kline = data['k']
            symbol = kline['s']
            interval = kline['i']
            
            # Check whitelist
            if symbol not in settings.PAIR_WHITELIST or interval not in settings.TIMEFRAMES:
                return
            
            # Normalize and process
            normalized_data = {
                'symbol': symbol,
                'interval': interval,
                'open_time': int(kline['t']),
                'close_time': int(kline['T']),
                'open': kline['o'],
                'high': kline['h'],
                'low': kline['l'],
                'close': kline['c'],
                'volume': kline['v'],
                'quote_volume': kline['q'],
                'trades': int(kline['n']),
                'taker_buy_base_volume': kline['V'],
                'taker_buy_quote_volume': kline['Q'],
                'is_closed': kline['x']
            }
            
            logger.debug(f"Websocket kline close: {symbol} {interval}")
            await self.signal_processor(symbol, interval, normalized_data)
            
        except Exception as e:
            logger.error(f"Error processing websocket kline: {e}")
    
    def stop(self):
        """Stop websocket connections"""
        self.running = False
        for connection in self.connections.values():
            asyncio.create_task(connection.close())

# Factory function to create appropriate data source
def create_data_source(signal_processor_callback: Callable, use_webhook: bool = True):
    """
    Create data source (webhook or websocket)
    
    Args:
        signal_processor_callback: Callback function for processing signals
        use_webhook: If True, use webhook handler; if False, use websocket subscriber
    
    Returns:
        BinanceWebhookHandler or BinanceWebhookSubscriber instance
    """
    if use_webhook:
        return BinanceWebhookHandler(signal_processor_callback)
    else:
        return BinanceWebhookSubscriber(signal_processor_callback)