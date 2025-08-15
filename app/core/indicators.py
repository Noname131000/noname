import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timezone

@dataclass
class IndicatorData:
    """Container for all technical indicators"""
    # Price data
    ohlc: Dict[str, float]
    volume: float
    timestamp: datetime
    
    # EMAs
    ema20: float
    ema50: float
    ema200: float
    ema_slope_50: float
    
    # Momentum
    rsi: float
    stochrsi_k: float
    stochrsi_d: float
    stochrsi_cross: str  # "BULL_CROSS", "BEAR_CROSS", "NONE"
    
    # Trend
    adx: float
    
    # Volatility
    atr: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float
    kc_upper: float
    kc_middle: float
    kc_lower: float
    kc_width: float
    squeeze_active: bool
    squeeze_bars: int
    
    # Volume/Flow
    vwap: float
    vwap_std: float
    vwap_distance: float  # distance from VWAP in std deviations
    relative_volume: float
    
    # Derived metrics
    ema_alignment: str  # "BULL", "BEAR", "MIXED"
    trend_strength: str  # "STRONG", "MODERATE", "WEAK"

class TechnicalIndicators:
    """Technical indicators calculator for trading signals"""
    
    def __init__(self):
        self.price_history = {}  # Store price history per symbol
        self.indicator_history = {}  # Store calculated indicators
    
    def update_data(self, symbol: str, timeframe: str, kline_data: dict) -> IndicatorData:
        """
        Update price data and calculate all indicators
        
        Args:
            symbol: Trading pair symbol
            timeframe: Chart timeframe (5m, 15m, 1h)
            kline_data: Raw kline data from exchange
            
        Returns:
            IndicatorData object with all calculated indicators
        """
        key = f"{symbol}_{timeframe}"
        
        # Initialize history if needed
        if key not in self.price_history:
            self.price_history[key] = []
            self.indicator_history[key] = []
        
        # Convert kline data to OHLCV format
        ohlcv = {
            'timestamp': datetime.fromtimestamp(kline_data['close_time'] / 1000, tz=timezone.utc),
            'open': float(kline_data['open']),
            'high': float(kline_data['high']),
            'low': float(kline_data['low']),
            'close': float(kline_data['close']),
            'volume': float(kline_data['volume'])
        }
        
        # Add to history
        self.price_history[key].append(ohlcv)
        
        # Keep only last 500 candles for memory efficiency
        if len(self.price_history[key]) > 500:
            self.price_history[key] = self.price_history[key][-500:]
        
        # Calculate indicators
        indicators = self._calculate_all_indicators(key)
        
        # Store indicators
        self.indicator_history[key].append(indicators)
        if len(self.indicator_history[key]) > 500:
            self.indicator_history[key] = self.indicator_history[key][-500:]
        
        return indicators
    
    def _calculate_all_indicators(self, key: str) -> IndicatorData:
        """Calculate all technical indicators for the given symbol/timeframe"""
        history = self.price_history[key]
        
        if len(history) < 200:  # Need enough data for EMA200
            # Return minimal data if not enough history
            latest = history[-1]
            return IndicatorData(
                ohlc={'open': latest['open'], 'high': latest['high'], 
                     'low': latest['low'], 'close': latest['close']},
                volume=latest['volume'],
                timestamp=latest['timestamp'],
                ema20=latest['close'], ema50=latest['close'], ema200=latest['close'],
                ema_slope_50=0, rsi=50, stochrsi_k=50, stochrsi_d=50, stochrsi_cross="NONE",
                adx=0, atr=0, bb_upper=latest['close'], bb_middle=latest['close'],
                bb_lower=latest['close'], bb_width=0, kc_upper=latest['close'],
                kc_middle=latest['close'], kc_lower=latest['close'], kc_width=0,
                squeeze_active=False, squeeze_bars=0, vwap=latest['close'],
                vwap_std=0, vwap_distance=0, relative_volume=1.0,
                ema_alignment="MIXED", trend_strength="WEAK"
            )
        
        # Convert to pandas for calculations
        df = pd.DataFrame(history)
        df.set_index('timestamp', inplace=True)
        
        # Calculate EMAs
        ema20 = self._ema(df['close'], 20).iloc[-1]
        ema50 = self._ema(df['close'], 50).iloc[-1]
        ema200 = self._ema(df['close'], 200).iloc[-1]
        
        # EMA slope (last 5 bars)
        ema50_series = self._ema(df['close'], 50)
        ema_slope_50 = (ema50_series.iloc[-1] - ema50_series.iloc[-6]) / 5 if len(ema50_series) >= 6 else 0
        
        # RSI
        rsi = self._rsi(df['close'], 14).iloc[-1]
        
        # Stochastic RSI
        stochrsi_k, stochrsi_d, stochrsi_cross = self._stochastic_rsi(df['close'], 3, 3, 14, 14)
        
        # ADX
        adx = self._adx(df['high'], df['low'], df['close'], 14).iloc[-1]
        
        # ATR
        atr = self._atr(df['high'], df['low'], df['close'], 14).iloc[-1]
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._bollinger_bands(df['close'], 20, 2)
        bb_width = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1] * 100
        
        # Keltner Channels
        kc_upper, kc_middle, kc_lower = self._keltner_channels(df['high'], df['low'], df['close'], 20, 1.5)
        kc_width = (kc_upper.iloc[-1] - kc_lower.iloc[-1]) / kc_middle.iloc[-1] * 100
        
        # Squeeze detection
        squeeze_active, squeeze_bars = self._detect_squeeze(bb_upper, bb_lower, kc_upper, kc_lower)
        
        # VWAP
        vwap, vwap_std = self._vwap(df['high'], df['low'], df['close'], df['volume'])
        vwap_distance = (df['close'].iloc[-1] - vwap.iloc[-1]) / vwap_std.iloc[-1] if vwap_std.iloc[-1] > 0 else 0
        
        # Relative volume
        relative_volume = self._relative_volume(df['volume'], 20)
        
        # Derived indicators
        ema_alignment = self._get_ema_alignment(ema20, ema50, ema200)
        trend_strength = self._get_trend_strength(adx)
        
        latest = history[-1]
        
        return IndicatorData(
            ohlc={'open': latest['open'], 'high': latest['high'], 
                 'low': latest['low'], 'close': latest['close']},
            volume=latest['volume'],
            timestamp=latest['timestamp'],
            ema20=ema20,
            ema50=ema50,
            ema200=ema200,
            ema_slope_50=ema_slope_50,
            rsi=rsi,
            stochrsi_k=stochrsi_k,
            stochrsi_d=stochrsi_d,
            stochrsi_cross=stochrsi_cross,
            adx=adx,
            atr=atr,
            bb_upper=bb_upper.iloc[-1],
            bb_middle=bb_middle.iloc[-1],
            bb_lower=bb_lower.iloc[-1],
            bb_width=bb_width,
            kc_upper=kc_upper.iloc[-1],
            kc_middle=kc_middle.iloc[-1],
            kc_lower=kc_lower.iloc[-1],
            kc_width=kc_width,
            squeeze_active=squeeze_active,
            squeeze_bars=squeeze_bars,
            vwap=vwap.iloc[-1],
            vwap_std=vwap_std.iloc[-1],
            vwap_distance=vwap_distance,
            relative_volume=relative_volume,
            ema_alignment=ema_alignment,
            trend_strength=trend_strength
        )
    
    def _ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return series.ewm(span=period).mean()
    
    def _rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _stochastic_rsi(self, series: pd.Series, k_period: int, d_period: int, 
                       rsi_period: int, stoch_period: int) -> Tuple[float, float, str]:
        """Calculate Stochastic RSI"""
        rsi_series = self._rsi(series, rsi_period)
        
        if len(rsi_series) < stoch_period + d_period:
            return 50.0, 50.0, "NONE"
        
        # Calculate %K
        rsi_min = rsi_series.rolling(window=stoch_period).min()
        rsi_max = rsi_series.rolling(window=stoch_period).max()
        stoch_k = 100 * ((rsi_series - rsi_min) / (rsi_max - rsi_min))
        
        # Smooth %K to get final %K
        k_smoothed = stoch_k.rolling(window=k_period).mean()
        
        # Calculate %D
        d_smoothed = k_smoothed.rolling(window=d_period).mean()
        
        # Detect crossovers
        cross = "NONE"
        if len(k_smoothed) >= 2 and len(d_smoothed) >= 2:
            k_curr, k_prev = k_smoothed.iloc[-1], k_smoothed.iloc[-2]
            d_curr, d_prev = d_smoothed.iloc[-1], d_smoothed.iloc[-2]
            
            if k_prev <= d_prev and k_curr > d_curr and k_curr < 20:
                cross = "BULL_CROSS"
            elif k_prev >= d_prev and k_curr < d_curr and k_curr > 80:
                cross = "BEAR_CROSS"
        
        return k_smoothed.iloc[-1], d_smoothed.iloc[-1], cross
    
    def _adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate Average Directional Index"""
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        dm_plus = (high - high.shift()).where((high - high.shift()) > (low.shift() - low), 0)
        dm_minus = (low.shift() - low).where((low.shift() - low) > (high - high.shift()), 0)
        
        # Smooth the values
        tr_smooth = tr.rolling(window=period).mean()
        dm_plus_smooth = dm_plus.rolling(window=period).mean()
        dm_minus_smooth = dm_minus.rolling(window=period).mean()
        
        # Directional Indicators
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        
        # ADX calculation
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=period).mean()
        
        return adx.fillna(0)
    
    def _atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def _bollinger_bands(self, series: pd.Series, period: int, std_dev: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def _keltner_channels(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                         period: int, multiplier: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Keltner Channels"""
        middle = self._ema(close, period)
        atr = self._atr(high, low, close, period)
        upper = middle + (atr * multiplier)
        lower = middle - (atr * multiplier)
        return upper, middle, lower
    
    def _detect_squeeze(self, bb_upper: pd.Series, bb_lower: pd.Series, 
                       kc_upper: pd.Series, kc_lower: pd.Series) -> Tuple[bool, int]:
        """Detect Bollinger Band squeeze inside Keltner Channels"""
        squeeze_condition = (bb_upper <= kc_upper) & (bb_lower >= kc_lower)
        
        # Count consecutive squeeze bars
        squeeze_bars = 0
        for i in range(len(squeeze_condition) - 1, -1, -1):
            if squeeze_condition.iloc[i]:
                squeeze_bars += 1
            else:
                break
        
        return squeeze_condition.iloc[-1], squeeze_bars
    
    def _vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, 
             volume: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate Volume Weighted Average Price and standard deviation"""
        typical_price = (high + low + close) / 3
        
        # Reset VWAP daily (assuming intraday timeframes)
        # For simplicity, calculate rolling VWAP over the session
        cumulative_volume = volume.cumsum()
        cumulative_tp_volume = (typical_price * volume).cumsum()
        
        vwap = cumulative_tp_volume / cumulative_volume
        
        # Calculate VWAP standard deviation
        vwap_variance = ((typical_price - vwap) ** 2 * volume).cumsum() / cumulative_volume
        vwap_std = np.sqrt(vwap_variance)
        
        return vwap.fillna(close), vwap_std.fillna(0)
    
    def _relative_volume(self, volume: pd.Series, lookback: int) -> float:
        """Calculate relative volume compared to average"""
        if len(volume) < lookback + 1:
            return 1.0
        
        current_volume = volume.iloc[-1]
        avg_volume = volume.iloc[-lookback-1:-1].mean()
        
        return current_volume / avg_volume if avg_volume > 0 else 1.0
    
    def _get_ema_alignment(self, ema20: float, ema50: float, ema200: float) -> str:
        """Determine EMA alignment trend"""
        if ema20 > ema50 > ema200:
            return "BULL"
        elif ema20 < ema50 < ema200:
            return "BEAR"
        else:
            return "MIXED"
    
    def _get_trend_strength(self, adx: float) -> str:
        """Classify trend strength based on ADX"""
        if adx >= 25:
            return "STRONG"
        elif adx >= 20:
            return "MODERATE"
        else:
            return "WEAK"
    
    def get_mtf_bias(self, symbol: str, primary_tf: str = "1h") -> str:
        """Get multi-timeframe bias from higher timeframe"""
        key = f"{symbol}_{primary_tf}"
        
        if key not in self.indicator_history or not self.indicator_history[key]:
            return "NEUTRAL"
        
        latest_indicators = self.indicator_history[key][-1]
        
        if latest_indicators.ohlc['close'] > latest_indicators.ema200:
            return "BULL"
        elif latest_indicators.ohlc['close'] < latest_indicators.ema200:
            return "BEAR"
        else:
            return "NEUTRAL"
    
    def calculate_vol_atr_ratio(self, indicators: IndicatorData) -> float:
        """Calculate VOL/ATR efficiency ratio"""
        if indicators.atr == 0:
            return 0.0
        
        return (indicators.relative_volume) / (indicators.atr / indicators.ohlc['close'])
    
    def get_squeeze_status(self, symbol: str, timeframe: str) -> Dict[str, any]:
        """Get detailed squeeze status information"""
        key = f"{symbol}_{timeframe}"
        
        if key not in self.indicator_history or not self.indicator_history[key]:
            return {"active": False, "bars": 0, "type": "NONE"}
        
        indicators = self.indicator_history[key][-1]
        
        return {
            "active": indicators.squeeze_active,
            "bars": indicators.squeeze_bars,
            "bb_width": indicators.bb_width,
            "kc_width": indicators.kc_width,
            "type": "SQUEEZE" if indicators.squeeze_active else "EXPANSION"
        }