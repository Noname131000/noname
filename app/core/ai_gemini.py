import json
import asyncio
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import aiohttp
import time

from .scoring import SignalData
from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger()

@dataclass
class AIDecision:
    """Container for AI verification decision"""
    decision: str  # "HOLD", "PROCESS", "REJECT"
    confidence: int  # 0-100
    reasons: List[str]
    concerns: List[str]
    overall_assessment: str

class GeminiAIVerifier:
    """Gemini AI verification for trading signals (Gate 3)"""
    
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.model = "gemini-1.5-flash"
        self.session = None
        self.request_count = 0
        self.last_request_time = 0
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def verify_signal(self, signal_data: SignalData) -> AIDecision:
        """
        Verify trading signal using Gemini AI
        
        Args:
            signal_data: Complete signal data package
            
        Returns:
            AIDecision with verification result
        """
        logger.info(f"Verifying signal with AI: {signal_data.symbol} {signal_data.timeframe}")
        
        try:
            # Rate limiting
            await self._rate_limit()
            
            # Create structured prompt
            prompt = self._create_verification_prompt(signal_data)
            
            # Send request to Gemini
            response = await self._send_gemini_request(prompt)
            
            if not response:
                logger.error("No response from Gemini AI")
                return self._create_fallback_decision("AI_ERROR")
            
            # Parse AI response
            decision = self._parse_ai_response(response)
            
            logger.info(f"AI Decision: {decision.decision} (confidence: {decision.confidence}%)")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in AI verification: {e}")
            return self._create_fallback_decision("ERROR")
    
    def _create_verification_prompt(self, signal_data: SignalData) -> str:
        """Create structured prompt for Gemini AI"""
        
        indicators = signal_data.indicators
        entry_plan = signal_data.entry_plan
        score = signal_data.score
        
        # Build context summary
        context = {
            "metadata": {
                "pair": signal_data.symbol,
                "timeframe": signal_data.timeframe,
                "timestamp": signal_data.timestamp,
                "mode": entry_plan.mode.value if entry_plan else "UNKNOWN"
            },
            "trend_analysis": {
                "ema_alignment": indicators.ema_alignment,
                "ema_slope_direction": "UP" if indicators.ema_slope_50 > 0 else "DOWN",
                "adx_strength": indicators.adx,
                "mtf_bias": signal_data.mtf_bias,
                "trend_score": score.trend_score
            },
            "momentum_analysis": {
                "rsi": indicators.rsi,
                "stochrsi_k": indicators.stochrsi_k,
                "stochrsi_d": indicators.stochrsi_d,
                "stochrsi_cross": indicators.stochrsi_cross,
                "momentum_score": score.momentum_score
            },
            "volatility_analysis": {
                "atr_percent": (indicators.atr / indicators.ohlc['close']) * 100,
                "squeeze_active": indicators.squeeze_active,
                "squeeze_bars": indicators.squeeze_bars,
                "vwap_distance_sigma": indicators.vwap_distance,
                "volatility_score": score.volatility_score
            },
            "volume_flow": {
                "relative_volume": indicators.relative_volume,
                "vol_atr_ratio": signal_data.vol_atr_ratio,
                "volume_score": score.volume_score
            },
            "trading_plan": {
                "signal_direction": entry_plan.direction.value if entry_plan else "NONE",
                "entry_limit": entry_plan.entry_limit if entry_plan else 0,
                "stop_loss": entry_plan.stop_loss if entry_plan else 0,
                "take_profit_1": entry_plan.take_profit_1 if entry_plan else 0,
                "take_profit_2": entry_plan.take_profit_2 if entry_plan else 0,
                "risk_reward_ratio": entry_plan.risk_reward if entry_plan else 0,
                "position_size_pct": entry_plan.position_size_pct if entry_plan else 0
            },
            "scoring": {
                "total_score": score.total_score,
                "breakdown": {
                    "trend": score.trend_score,
                    "momentum": score.momentum_score,
                    "volatility": score.volatility_score,
                    "volume": score.volume_score,
                    "structure": score.structure_score,
                    "risk_reward": score.rr_score
                }
            },
            "system_analysis": {
                "reasons": signal_data.reasons,
                "warnings": signal_data.warnings
            }
        }
        
        prompt = f"""
You are an expert trading signal analyst. Analyze the following trading signal and provide a verification decision.

SIGNAL CONTEXT:
{json.dumps(context, indent=2)}

TASK:
Based on the technical analysis provided, determine if this signal should be:
1. PROCESS - Strong signal with good confluence, recommend proceeding
2. HOLD - Moderate signal with some concerns, suggest caution but can proceed
3. REJECT - Weak signal or significant risks, recommend against

ANALYSIS REQUIREMENTS:
- Evaluate the confluence of technical indicators
- Assess risk/reward profile and position sizing
- Consider market structure and timing
- Identify any red flags or concerns
- Provide confidence level (0-100%)

RESPONSE FORMAT (JSON only):
{{
    "decision": "PROCESS|HOLD|REJECT",
    "confidence": 85,
    "reasons": [
        "Primary reason supporting the decision",
        "Secondary confluence factor",
        "Risk management consideration"
    ],
    "concerns": [
        "Potential risk or weakness",
        "Market condition to monitor"
    ],
    "overall_assessment": "Brief 1-2 sentence summary of the signal quality"
}}

GUIDELINES:
- PROCESS: Score >75, strong confluence, clear R:R >2.0, minimal concerns
- HOLD: Score 60-75, moderate confluence, acceptable R:R >1.5, some concerns manageable  
- REJECT: Score <60, weak confluence, poor R:R <1.5, significant risks
- Consider multi-timeframe alignment, volume confirmation, and volatility conditions
- Be conservative with high-risk setups or conflicting signals
- Provide clear, actionable reasoning

Analyze the signal and respond with JSON only:
"""
        return prompt
    
    async def _send_gemini_request(self, prompt: str) -> Optional[str]:
        """Send request to Gemini API"""
        
        if not self.api_key:
            logger.error("Gemini API key not configured")
            return None
            
        url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.1,  # Low temperature for consistent analysis
                "topK": 1,
                "topP": 0.8,
                "maxOutputTokens": 1000,
                "candidateCount": 1
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            async with self.session.post(url, json=payload, headers=headers, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'candidates' in data and len(data['candidates']) > 0:
                        candidate = data['candidates'][0]
                        if 'content' in candidate and 'parts' in candidate['content']:
                            return candidate['content']['parts'][0]['text']
                    
                    logger.error(f"Unexpected Gemini response structure: {data}")
                    return None
                    
                else:
                    error_text = await response.text()
                    logger.error(f"Gemini API error {response.status}: {error_text}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error("Gemini API request timeout")
            return None
        except Exception as e:
            logger.error(f"Gemini API request failed: {e}")
            return None
    
    def _parse_ai_response(self, response: str) -> AIDecision:
        """Parse AI response and extract decision"""
        
        try:
            # Clean response - sometimes AI adds markdown formatting
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            # Parse JSON
            data = json.loads(cleaned_response)
            
            # Validate required fields
            decision = data.get('decision', 'REJECT').upper()
            if decision not in ['PROCESS', 'HOLD', 'REJECT']:
                decision = 'REJECT'
            
            confidence = max(0, min(100, int(data.get('confidence', 0))))
            reasons = data.get('reasons', ['AI analysis completed'])
            concerns = data.get('concerns', [])
            assessment = data.get('overall_assessment', 'Signal analyzed by AI')
            
            return AIDecision(
                decision=decision,
                confidence=confidence,
                reasons=reasons[:5],  # Limit to 5 reasons
                concerns=concerns[:3],  # Limit to 3 concerns
                overall_assessment=assessment
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.debug(f"Raw response: {response}")
            
            # Try to extract decision from text
            response_upper = response.upper()
            if 'PROCESS' in response_upper:
                decision = 'PROCESS'
                confidence = 70
            elif 'HOLD' in response_upper:
                decision = 'HOLD'
                confidence = 50
            else:
                decision = 'REJECT'
                confidence = 30
            
            return AIDecision(
                decision=decision,
                confidence=confidence,
                reasons=['Fallback analysis from unparseable response'],
                concerns=['AI response format issue'],
                overall_assessment='Signal requires manual review'
            )
        
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return self._create_fallback_decision("PARSE_ERROR")
    
    def _create_fallback_decision(self, error_type: str) -> AIDecision:
        """Create fallback decision when AI fails"""
        
        fallback_decisions = {
            "API_ERROR": ("REJECT", 0, ["AI service unavailable"], ["Cannot verify signal"], "AI verification failed"),
            "PARSE_ERROR": ("REJECT", 0, ["Response parsing failed"], ["AI output unreadable"], "AI analysis unclear"),
            "TIMEOUT": ("REJECT", 0, ["AI request timeout"], ["Service delay"], "AI verification timeout"),
            "ERROR": ("REJECT", 0, ["General AI error"], ["Unknown issue"], "AI verification error")
        }
        
        decision, confidence, reasons, concerns, assessment = fallback_decisions.get(
            error_type, 
            ("REJECT", 0, ["Unknown error"], ["System issue"], "Signal verification failed")
        )
        
        return AIDecision(
            decision=decision,
            confidence=confidence,
            reasons=reasons,
            concerns=concerns,
            overall_assessment=assessment
        )
    
    async def _rate_limit(self):
        """Basic rate limiting for API requests"""
        current_time = time.time()
        
        # Limit to 1 request per second
        if current_time - self.last_request_time < 1.0:
            await asyncio.sleep(1.0 - (current_time - self.last_request_time))
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def passes_gate3(self, ai_decision: AIDecision) -> bool:
        """Check if signal passes Gate 3 requirements"""
        
        # Must be PROCESS or HOLD
        if ai_decision.decision == "REJECT":
            return False
        
        # Must meet minimum confidence threshold
        if ai_decision.confidence < settings.MIN_AI_CONFIDENCE:
            return False
        
        return True
    
    def get_verification_summary(self, ai_decision: AIDecision) -> str:
        """Get formatted summary of AI verification"""
        
        decision_emoji = {
            "PROCESS": "🟢",
            "HOLD": "🟡", 
            "REJECT": "🔴"
        }
        
        emoji = decision_emoji.get(ai_decision.decision, "⚪")
        
        summary = f"{emoji} {ai_decision.decision} ({ai_decision.confidence}%)"
        
        if ai_decision.reasons:
            summary += f"\n• {'; '.join(ai_decision.reasons[:2])}"
        
        if ai_decision.concerns:
            summary += f"\n⚠️ {'; '.join(ai_decision.concerns[:2])}"
        
        return summary
    
    async def batch_verify_signals(self, signals: List[SignalData]) -> List[Tuple[SignalData, AIDecision]]:
        """Verify multiple signals with rate limiting"""
        
        results = []
        
        for signal in signals:
            decision = await self.verify_signal(signal)
            results.append((signal, decision))
            
            # Small delay between requests
            await asyncio.sleep(0.5)
        
        return results
    
    def create_ai_context_summary(self, signal_data: SignalData) -> str:
        """Create concise context summary for logging"""
        
        entry_plan = signal_data.entry_plan
        if not entry_plan:
            return "No entry plan available"
        
        return (
            f"{signal_data.symbol} {signal_data.timeframe} | "
            f"{entry_plan.direction.value} @ {entry_plan.entry_limit:.5f} | "
            f"SL: {entry_plan.stop_loss:.5f} | "
            f"TP1: {entry_plan.take_profit_1:.5f} | "
            f"R:R {entry_plan.risk_reward:.1f} | "
            f"Score: {signal_data.score.total_score:.0f}/100"
        )

# Context manager for easy usage
class AIVerifier:
    """Context manager wrapper for GeminiAIVerifier"""
    
    def __init__(self):
        self.verifier = None
    
    async def __aenter__(self):
        self.verifier = GeminiAIVerifier()
        await self.verifier.__aenter__()
        return self.verifier
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.verifier:
            await self.verifier.__aexit__(exc_type, exc_val, exc_tb)

# Utility function for one-off verification
async def verify_signal_with_ai(signal_data: SignalData) -> AIDecision:
    """Utility function to verify a single signal"""
    async with AIVerifier() as verifier:
        return await verifier.verify_signal(signal_data)