from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math

from .indicators import IndicatorData, TechnicalIndicators
from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger()

class SignalDirection(Enum):
    LONG = "BUY"
    SHORT = "SELL"
    NONE = "NONE"

class TradingMode(Enum):
    SCALP = "SCALP"
    INTRADAY = "INTRA"
    SWING = "SWING"

@dataclass
class SignalScore:
    """Container for signal scoring breakdown"""
    trend_score: float  # 0-25
    momentum_score: float  # 0-20
    volatility_score: float  # 0-20
    volume_score: float  # 0-20
    structure_score: float  # 0-10
    rr_score: float  # 0-5
    total_score: float  # 0-100
    
    # Detailed breakdowns
    trend_details: Dict
    momentum_details: Dict
    volatility_details: Dict
    volume_details: Dict
    structure_details: Dict

@dataclass
class EntryPlan:
    """Container for entry plan details"""
    direction: SignalDirection
    entry_limit: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    trailing_stop: float
    risk_reward: float
    position_size_pct: float
    valid_bars: int
    mode: TradingMode

@dataclass
class SignalData:
    """Complete signal data package"""
    symbol: str
    timeframe: str
    timestamp: str
    indicators: IndicatorData
    score: SignalScore
    entry_plan: Optional[EntryPlan]
    mtf_bias: str
    vol_atr_ratio: float
    squeeze_status: Dict
    reasons: List[str]
    warnings: List[str]

class SignalScorer:
    """Score and validate trading signals based on technical analysis"""
    
    def __init__(self, indicators_engine: TechnicalIndicators):
        self.indicators = indicators_engine
        self.cooldown_tracker = {}  # Track pairs on cooldown after SL
    
    def evaluate_signal(self, symbol: str, timeframe: str, indicators: IndicatorData) -> SignalData:
        """
        Evaluate and score a potential trading signal
        
        Args:
            symbol: Trading pair
            timeframe: Chart timeframe
            indicators: Technical indicator data
            
        Returns:
            Complete SignalData package
        """
        logger.debug(f"Evaluating signal for {symbol} {timeframe}")
        
        # Check cooldown
        if self._is_on_cooldown(symbol, timeframe):
            logger.info(f"{symbol} {timeframe} is on cooldown after SL")
            return self._create_no_signal(symbol, timeframe, indicators, "On cooldown after stop loss")
        
        # Get multi-timeframe bias
        mtf_bias = self.indicators.get_mtf_bias(symbol, "1h")
        
        # Calculate VOL/ATR ratio
        vol_atr_ratio = self.indicators.calculate_vol_atr_ratio(indicators)
        
        # Get squeeze status
        squeeze_status = self.indicators.get_squeeze_status(symbol, timeframe)
        
        # Determine trading mode based on timeframe
        mode = self._get_trading_mode(timeframe)
        
        # Check basic requirements first
        basic_checks = self._validate_basic_requirements(indicators, mode)
        if not basic_checks['valid']:
            return self._create_no_signal(symbol, timeframe, indicators, basic_checks['reason'])
        
        # Score all components
        trend_score, trend_details = self._score_trend(indicators, mtf_bias, mode)
        momentum_score, momentum_details = self._score_momentum(indicators)
        volatility_score, volatility_details = self._score_volatility(indicators, squeeze_status)
        volume_score, volume_details = self._score_volume(indicators)
        
        # Determine signal direction
        direction = self._determine_direction(indicators, mtf_bias, trend_details, momentum_details)
        
        if direction == SignalDirection.NONE:
            return self._create_no_signal(symbol, timeframe, indicators, "No clear directional bias")
        
        # Create entry plan
        entry_plan = self._create_entry_plan(direction, indicators, mode)
        
        if not entry_plan:
            return self._create_no_signal(symbol, timeframe, indicators, "Unable to create valid entry plan")
        
        # Score structure and R:R
        structure_score, structure_details = self._score_structure(indicators, entry_plan)
        rr_score = self._score_risk_reward(entry_plan.risk_reward)
        
        # Calculate total score
        total_score = trend_score + momentum_score + volatility_score + volume_score + structure_score + rr_score
        
        # Create score object
        score = SignalScore(
            trend_score=trend_score,
            momentum_score=momentum_score,
            volatility_score=volatility_score,
            volume_score=volume_score,
            structure_score=structure_score,
            rr_score=rr_score,
            total_score=total_score,
            trend_details=trend_details,
            momentum_details=momentum_details,
            volatility_details=volatility_details,
            volume_details=volume_details,
            structure_details=structure_details
        )
        
        # Generate reasons and warnings
        reasons, warnings = self._generate_analysis_text(score, indicators, entry_plan, mtf_bias)
        
        return SignalData(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=indicators.timestamp.isoformat(),
            indicators=indicators,
            score=score,
            entry_plan=entry_plan,
            mtf_bias=mtf_bias,
            vol_atr_ratio=vol_atr_ratio,
            squeeze_status=squeeze_status,
            reasons=reasons,
            warnings=warnings
        )
    
    def _validate_basic_requirements(self, indicators: IndicatorData, mode: TradingMode) -> Dict:
        """Validate basic signal requirements"""
        
        # Check ADX threshold
        adx_threshold = {
            TradingMode.SCALP: settings.ADX_MIN_SCALP,
            TradingMode.INTRADAY: settings.ADX_MIN_INTRADAY,
            TradingMode.SWING: settings.ADX_MIN_SWING
        }[mode]
        
        if indicators.adx < adx_threshold:
            return {'valid': False, 'reason': f'ADX {indicators.adx:.1f} below threshold {adx_threshold}'}
        
        # Check VWAP deviation
        if abs(indicators.vwap_distance) > settings.VWAP_DEV_MAX:
            return {'valid': False, 'reason': f'Price too far from VWAP ({indicators.vwap_distance:.1f}σ)'}
        
        # Check ATR validity
        if indicators.atr <= 0:
            return {'valid': False, 'reason': 'Invalid ATR value'}
        
        return {'valid': True, 'reason': ''}
    
    def _score_trend(self, indicators: IndicatorData, mtf_bias: str, mode: TradingMode) -> Tuple[float, Dict]:
        """Score trend strength and alignment (0-25 points)"""
        score = 0
        details = {}
        
        # EMA alignment (0-8 points)
        if indicators.ema_alignment == "BULL":
            score += 8
            details['ema_alignment'] = 'Bullish alignment'
        elif indicators.ema_alignment == "BEAR":
            score += 8
            details['ema_alignment'] = 'Bearish alignment'
        else:
            score += 2
            details['ema_alignment'] = 'Mixed alignment'
        
        # EMA slope (0-5 points)
        slope_strength = abs(indicators.ema_slope_50) / indicators.ohlc['close'] * 1000
        if slope_strength > 0.5:
            score += 5
            details['ema_slope'] = 'Strong slope'
        elif slope_strength > 0.2:
            score += 3
            details['ema_slope'] = 'Moderate slope'
        else:
            score += 1
            details['ema_slope'] = 'Weak slope'
        
        # ADX strength (0-7 points)
        if indicators.adx >= 30:
            score += 7
            details['adx_strength'] = 'Very strong trend'
        elif indicators.adx >= 25:
            score += 5
            details['adx_strength'] = 'Strong trend'
        elif indicators.adx >= 20:
            score += 3
            details['adx_strength'] = 'Moderate trend'
        else:
            score += 1
            details['adx_strength'] = 'Weak trend'
        
        # MTF alignment (0-5 points)
        current_direction = "BULL" if indicators.ohlc['close'] > indicators.ema200 else "BEAR"
        if mtf_bias == current_direction:
            score += 5
            details['mtf_alignment'] = 'Aligned with higher timeframe'
        elif mtf_bias == "NEUTRAL":
            score += 2
            details['mtf_alignment'] = 'Neutral higher timeframe'
        else:
            score += 0
            details['mtf_alignment'] = 'Against higher timeframe'
        
        return min(score, 25), details
    
    def _score_momentum(self, indicators: IndicatorData) -> Tuple[float, Dict]:
        """Score momentum indicators (0-20 points)"""
        score = 0
        details = {}
        
        # RSI positioning (0-10 points)
        rsi = indicators.rsi
        if 45 <= rsi <= 55:
            score += 5
            details['rsi_position'] = 'Neutral zone'
        elif 55 < rsi <= 65:
            score += 8
            details['rsi_position'] = 'Bullish zone'
        elif 35 <= rsi < 45:
            score += 8
            details['rsi_position'] = 'Bearish zone'
        elif rsi > 65:
            score += 6
            details['rsi_position'] = 'Overbought zone'
        elif rsi < 35:
            score += 6
            details['rsi_position'] = 'Oversold zone'
        else:
            score += 3
            details['rsi_position'] = 'Extreme zone'
        
        # StochRSI cross (0-10 points)
        if indicators.stochrsi_cross == "BULL_CROSS":
            score += 10
            details['stochrsi'] = 'Bullish crossover from oversold'
        elif indicators.stochrsi_cross == "BEAR_CROSS":
            score += 10
            details['stochrsi'] = 'Bearish crossover from overbought'
        elif indicators.stochrsi_k < 20:
            score += 5
            details['stochrsi'] = 'Oversold condition'
        elif indicators.stochrsi_k > 80:
            score += 5
            details['stochrsi'] = 'Overbought condition'
        else:
            score += 2
            details['stochrsi'] = 'Neutral momentum'
        
        return min(score, 20), details
    
    def _score_volatility(self, indicators: IndicatorData, squeeze_status: Dict) -> Tuple[float, Dict]:
        """Score volatility and squeeze conditions (0-20 points)"""
        score = 0
        details = {}
        
        # ATR relative to price (0-5 points)
        atr_pct = (indicators.atr / indicators.ohlc['close']) * 100
        if 0.5 <= atr_pct <= 2.0:
            score += 5
            details['atr_level'] = 'Optimal volatility'
        elif 0.2 <= atr_pct < 0.5 or 2.0 < atr_pct <= 3.0:
            score += 3
            details['atr_level'] = 'Acceptable volatility'
        else:
            score += 1
            details['atr_level'] = 'Suboptimal volatility'
        
        # Squeeze conditions (0-10 points)
        if squeeze_status['active'] and squeeze_status['bars'] >= settings.MIN_SQUEEZE_BARS:
            score += 10
            details['squeeze'] = f'Active squeeze for {squeeze_status["bars"]} bars'
        elif not squeeze_status['active'] and squeeze_status['bars'] == 0:
            score += 7
            details['squeeze'] = 'Fresh breakout from squeeze'
        elif not squeeze_status['active']:
            score += 5
            details['squeeze'] = 'Expansion phase'
        else:
            score += 2
            details['squeeze'] = 'Early squeeze formation'
        
        # VWAP distance (0-5 points)
        vwap_dist = abs(indicators.vwap_distance)
        if vwap_dist <= 1.0:
            score += 5
            details['vwap_distance'] = 'Close to fair value'
        elif vwap_dist <= 1.5:
            score += 3
            details['vwap_distance'] = 'Reasonable distance from fair value'
        else:
            score += 1
            details['vwap_distance'] = 'Extended from fair value'
        
        return min(score, 20), details
    
    def _score_volume(self, indicators: IndicatorData) -> Tuple[float, Dict]:
        """Score volume and flow indicators (0-20 points)"""
        score = 0
        details = {}
        
        # Relative volume (0-15 points)
        rel_vol = indicators.relative_volume
        if rel_vol >= 2.0:
            score += 15
            details['relative_volume'] = 'Very high volume'
        elif rel_vol >= 1.5:
            score += 12
            details['relative_volume'] = 'High volume'
        elif rel_vol >= 1.2:
            score += 8
            details['relative_volume'] = 'Above average volume'
        elif rel_vol >= 0.8:
            score += 5
            details['relative_volume'] = 'Normal volume'
        else:
            score += 2
            details['relative_volume'] = 'Low volume'
        
        # Volume efficiency (0-5 points)
        vol_atr = self.indicators.calculate_vol_atr_ratio(indicators)
        if vol_atr >= 1.0:
            score += 5
            details['volume_efficiency'] = 'Efficient price movement'
        elif vol_atr >= 0.7:
            score += 3
            details['volume_efficiency'] = 'Moderate efficiency'
        else:
            score += 1
            details['volume_efficiency'] = 'Inefficient movement'
        
        return min(score, 20), details
    
    def _score_structure(self, indicators: IndicatorData, entry_plan: EntryPlan) -> Tuple[float, Dict]:
        """Score structural levels and positioning (0-10 points)"""
        score = 0
        details = {}
        
        # Position relative to Bollinger Bands (0-5 points)
        close = indicators.ohlc['close']
        bb_position = (close - indicators.bb_lower) / (indicators.bb_upper - indicators.bb_lower)
        
        if entry_plan.direction == SignalDirection.LONG:
            if bb_position <= 0.3:
                score += 5
                details['bb_position'] = 'Near lower band (good for long)'
            elif bb_position <= 0.5:
                score += 3
                details['bb_position'] = 'Below middle (acceptable for long)'
            else:
                score += 1
                details['bb_position'] = 'Upper half (poor for long)'
        else:  # SHORT
            if bb_position >= 0.7:
                score += 5
                details['bb_position'] = 'Near upper band (good for short)'
            elif bb_position >= 0.5:
                score += 3
                details['bb_position'] = 'Above middle (acceptable for short)'
            else:
                score += 1
                details['bb_position'] = 'Lower half (poor for short)'
        
        # Entry limit quality (0-5 points)
        entry_distance = abs(entry_plan.entry_limit - close) / indicators.atr
        if entry_distance <= 0.5:
            score += 5
            details['entry_limit'] = 'Tight entry limit'
        elif entry_distance <= 1.0:
            score += 3
            details['entry_limit'] = 'Reasonable entry limit'
        else:
            score += 1
            details['entry_limit'] = 'Wide entry limit'
        
        return min(score, 10), details
    
    def _score_risk_reward(self, risk_reward: float) -> float:
        """Score risk/reward ratio (0-5 points)"""
        if risk_reward >= 3.0:
            return 5
        elif risk_reward >= 2.5:
            return 4
        elif risk_reward >= 2.0:
            return 3
        elif risk_reward >= 1.5:
            return 2
        else:
            return 0
    
    def _determine_direction(self, indicators: IndicatorData, mtf_bias: str, 
                           trend_details: Dict, momentum_details: Dict) -> SignalDirection:
        """Determine signal direction based on all factors"""
        
        # Count bullish vs bearish factors
        bull_factors = 0
        bear_factors = 0
        
        # Trend factors
        if indicators.ema_alignment == "BULL":
            bull_factors += 1
        elif indicators.ema_alignment == "BEAR":
            bear_factors += 1
        
        if indicators.ema_slope_50 > 0:
            bull_factors += 1
        else:
            bear_factors += 1
        
        if mtf_bias == "BULL":
            bull_factors += 1
        elif mtf_bias == "BEAR":
            bear_factors += 1
        
        # Momentum factors
        if indicators.rsi > 50:
            bull_factors += 1
        else:
            bear_factors += 1
        
        if indicators.stochrsi_cross == "BULL_CROSS":
            bull_factors += 2
        elif indicators.stochrsi_cross == "BEAR_CROSS":
            bear_factors += 2
        
        # Price position
        if indicators.ohlc['close'] > indicators.vwap:
            bull_factors += 1
        else:
            bear_factors += 1
        
        # Determine direction
        if bull_factors >= bear_factors + 2:
            return SignalDirection.LONG
        elif bear_factors >= bull_factors + 2:
            return SignalDirection.SHORT
        else:
            return SignalDirection.NONE
    
    def _create_entry_plan(self, direction: SignalDirection, indicators: IndicatorData, 
                          mode: TradingMode) -> Optional[EntryPlan]:
        """Create detailed entry plan with levels"""
        
        close = indicators.ohlc['close']
        atr = indicators.atr
        
        # Calculate entry limit based on direction and current levels
        if direction == SignalDirection.LONG:
            # Entry on pullback to support level
            entry_limit = max(
                indicators.ema20,
                indicators.vwap - (indicators.vwap_std * 0.5),
                indicators.bb_middle
            )
            entry_limit -= (settings.LIMIT_OFFSET_ATR * atr)
            
            # Risk management
            if mode == TradingMode.SCALP:
                sl_distance = settings.ATR_MULT_SL_SCALP * atr
            else:
                sl_distance = settings.ATR_MULT_SL_INTRADAY * atr
            
            stop_loss = entry_limit - sl_distance
            take_profit_1 = entry_limit + (settings.TP1_ATR * atr)
            take_profit_2 = entry_limit + (settings.TP2_ATR * atr)
            trailing_stop = entry_limit + (settings.TRAILING_ATR * atr)
            
        else:  # SHORT
            # Entry on pullback to resistance level
            entry_limit = min(
                indicators.ema20,
                indicators.vwap + (indicators.vwap_std * 0.5),
                indicators.bb_middle
            )
            entry_limit += (settings.LIMIT_OFFSET_ATR * atr)
            
            # Risk management
            if mode == TradingMode.SCALP:
                sl_distance = settings.ATR_MULT_SL_SCALP * atr
            else:
                sl_distance = settings.ATR_MULT_SL_INTRADAY * atr
            
            stop_loss = entry_limit + sl_distance
            take_profit_1 = entry_limit - (settings.TP1_ATR * atr)
            take_profit_2 = entry_limit - (settings.TP2_ATR * atr)
            trailing_stop = entry_limit - (settings.TRAILING_ATR * atr)
        
        # Calculate risk/reward ratio
        risk = abs(entry_limit - stop_loss)
        reward = abs(take_profit_1 - entry_limit)
        risk_reward = reward / risk if risk > 0 else 0
        
        # Validate minimum R:R
        if risk_reward < settings.MIN_RR:
            return None
        
        # Entry validity period
        valid_bars = 10 if mode == TradingMode.SCALP else 20
        
        return EntryPlan(
            direction=direction,
            entry_limit=entry_limit,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            trailing_stop=trailing_stop,
            risk_reward=risk_reward,
            position_size_pct=settings.RISK_PER_TRADE_PCT,
            valid_bars=valid_bars,
            mode=mode
        )
    
    def _get_trading_mode(self, timeframe: str) -> TradingMode:
        """Determine trading mode from timeframe"""
        if timeframe in ["1m", "3m", "5m"]:
            return TradingMode.SCALP
        elif timeframe in ["15m", "30m", "1h"]:
            return TradingMode.INTRADAY
        else:
            return TradingMode.SWING
    
    def _generate_analysis_text(self, score: SignalScore, indicators: IndicatorData, 
                              entry_plan: EntryPlan, mtf_bias: str) -> Tuple[List[str], List[str]]:
        """Generate human-readable analysis"""
        reasons = []
        warnings = []
        
        # Add strong points
        if score.trend_score >= 20:
            reasons.append(f"{mtf_bias} trend alignment")
        if score.momentum_score >= 15:
            reasons.append("Strong momentum signals")
        if score.volatility_score >= 15:
            reasons.append("Favorable volatility conditions")
        if score.volume_score >= 15:
            reasons.append("High volume confirmation")
        
        # Add specific details
        if indicators.squeeze_active:
            reasons.append(f"Squeeze active for {indicators.squeeze_bars} bars")
        if indicators.relative_volume > 1.5:
            reasons.append(f"Volume {indicators.relative_volume:.1f}x average")
        if entry_plan.risk_reward >= 2.5:
            reasons.append(f"Excellent R:R {entry_plan.risk_reward:.1f}")
        
        # Add warnings
        if abs(indicators.vwap_distance) > 1.5:
            warnings.append("Extended from VWAP")
        if indicators.adx < 25:
            warnings.append("Moderate trend strength")
        if indicators.relative_volume < 1.0:
            warnings.append("Below average volume")
        
        return reasons, warnings
    
    def _create_no_signal(self, symbol: str, timeframe: str, indicators: IndicatorData, 
                         reason: str) -> SignalData:
        """Create a no-signal response"""
        return SignalData(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=indicators.timestamp.isoformat(),
            indicators=indicators,
            score=SignalScore(0, 0, 0, 0, 0, 0, 0, {}, {}, {}, {}, {}),
            entry_plan=None,
            mtf_bias="NEUTRAL",
            vol_atr_ratio=0,
            squeeze_status={},
            reasons=[],
            warnings=[reason]
        )
    
    def _is_on_cooldown(self, symbol: str, timeframe: str) -> bool:
        """Check if pair is on cooldown after stop loss"""
        key = f"{symbol}_{timeframe}"
        return key in self.cooldown_tracker
    
    def add_cooldown(self, symbol: str, timeframe: str):
        """Add pair to cooldown after stop loss"""
        key = f"{symbol}_{timeframe}"
        self.cooldown_tracker[key] = settings.COOLDOWN_BARS_AFTER_SL
        logger.info(f"Added cooldown for {symbol} {timeframe}")
    
    def update_cooldowns(self):
        """Update cooldown counters (call on each new bar)"""
        expired_keys = []
        for key in self.cooldown_tracker:
            self.cooldown_tracker[key] -= 1
            if self.cooldown_tracker[key] <= 0:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cooldown_tracker[key]
            logger.info(f"Cooldown expired for {key}")
    
    def passes_gate2(self, signal_data: SignalData) -> bool:
        """Check if signal passes Gate 2 requirements"""
        if signal_data.entry_plan is None:
            return False
        
        if signal_data.score.total_score < settings.MIN_SCORE_GATE2:
            return False
        
        # Must have minimum risk/reward
        if signal_data.entry_plan.risk_reward < settings.MIN_RR:
            return False
        
        # Must meet ADX threshold
        mode = signal_data.entry_plan.mode
        adx_threshold = {
            TradingMode.SCALP: settings.ADX_MIN_SCALP,
            TradingMode.INTRADAY: settings.ADX_MIN_INTRADAY,
            TradingMode.SWING: settings.ADX_MIN_SWING
        }[mode]
        
        if signal_data.indicators.adx < adx_threshold:
            return False
        
        # Must not be too far from VWAP
        if abs(signal_data.indicators.vwap_distance) > settings.VWAP_DEV_MAX:
            return False
        
        return True