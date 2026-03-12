"""
EmoMem 自适应参数框架 (Adaptive Parameter Framework)
=====================================================
Safety-Constrained Adaptive Parameter Manager

学术依据:
  - Constrained Thompson Sampling (Deb et al., RLC 2025)
  - SPC-based adaptive emotion baselines (Schreuder et al., Mental Health Science 2024)
  - LLM-as-parameter-estimator (Masadome & Harada, IEEJ 2025)
  - Safety envelopes for AI agents (Lu et al., 2025)

参数分类:
  - Class A (Immutable): 安全关键参数, frozen=True, 永不自适应
  - Class B (LLM-computed per-turn): 上下文依赖阈值, 由 LLM 每轮计算
  - Class C (Learned over time): 用户自适应参数, 从交互历史中学习
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from .config import (
    # Class A (frozen) — listed for documentation, never modified
    CRISIS_THRESHOLD,
    CRISIS_STEP_THRESHOLD,
    URGENCY_ALPHA1,
    URGENCY_ALPHA2,
    URGENCY_ALPHA4,
    # Class C defaults
    BASELINE_LR,
    BASELINE_SIGMA_LR,
    EMOTION_INERTIA_ALPHA,
    BETA_DECAY_HALFLIFE_DAYS,
    LLM_OVERRIDE_CONFIDENCE_THRESHOLD,
    LLM_OPENNESS_SILENCE_THRESHOLD,
    COLD_START_MU,
    # Adaptive formula constants (§6.3)
    ADAPTIVE_LR_VARIANCE_SENSITIVITY,
    ADAPTIVE_INERTIA_BASE,
    ADAPTIVE_INERTIA_SCALE,
    ADAPTIVE_TS_BLEND_HORIZON,
    ADAPTIVE_SEVERITY_MIN_SAMPLES,
    ADAPTIVE_SEVERITY_MIN_SPREAD,
    ADAPTIVE_SEVERITY_PERCENTILE_LOW,
    ADAPTIVE_SEVERITY_PERCENTILE_HIGH,
    ADAPTIVE_SEVERITY_LOW_BOUNDS,
    ADAPTIVE_SEVERITY_HIGH_BOUNDS,
    ADAPTIVE_TS_MIN_INTERACTIONS,
    ADAPTIVE_TS_PRIOR_MIN,
    ADAPTIVE_TS_PRIOR_MAX,
)


# ══════════════════════════════════════════════════════════════
# 安全包络 (Safety Envelope)
# ══════════════════════════════════════════════════════════════

@dataclass
class ParameterBounds:
    """
    Safety envelope for one adaptive parameter.

    Every adaptive computation is clamped within [min_val, max_val].
    Class A parameters have frozen=True and are never computed adaptively.
    """
    default: float
    min_val: float
    max_val: float
    frozen: bool = False


# Registry of all adaptive parameters with their safety bounds.
# Class A parameters are included with frozen=True for documentation completeness.
PARAMETER_BOUNDS: Dict[str, ParameterBounds] = {
    # ── Class A: Immutable safety-critical (frozen=True) ──
    "crisis_threshold": ParameterBounds(
        default=CRISIS_THRESHOLD, min_val=CRISIS_THRESHOLD,
        max_val=CRISIS_THRESHOLD, frozen=True,
    ),
    "crisis_step_threshold": ParameterBounds(
        default=CRISIS_STEP_THRESHOLD, min_val=CRISIS_STEP_THRESHOLD,
        max_val=CRISIS_STEP_THRESHOLD, frozen=True,
    ),
    "urgency_alpha1": ParameterBounds(
        default=URGENCY_ALPHA1, min_val=URGENCY_ALPHA1,
        max_val=URGENCY_ALPHA1, frozen=True,
    ),
    "urgency_alpha2": ParameterBounds(
        default=URGENCY_ALPHA2, min_val=URGENCY_ALPHA2,
        max_val=URGENCY_ALPHA2, frozen=True,
    ),
    "urgency_alpha4": ParameterBounds(
        default=URGENCY_ALPHA4, min_val=URGENCY_ALPHA4,
        max_val=URGENCY_ALPHA4, frozen=True,
    ),

    # ── Class C: User-adaptive parameters ──
    "baseline_lr": ParameterBounds(
        default=BASELINE_LR, min_val=0.02, max_val=0.15,
    ),
    "baseline_sigma_lr": ParameterBounds(
        default=BASELINE_SIGMA_LR, min_val=0.02, max_val=0.15,
    ),
    "emotion_inertia_alpha": ParameterBounds(
        default=EMOTION_INERTIA_ALPHA, min_val=0.2, max_val=0.6,
    ),
    "severity_low_threshold": ParameterBounds(
        default=0.3, min_val=ADAPTIVE_SEVERITY_LOW_BOUNDS[0],
        max_val=ADAPTIVE_SEVERITY_LOW_BOUNDS[1],
    ),
    "severity_high_threshold": ParameterBounds(
        default=0.7, min_val=ADAPTIVE_SEVERITY_HIGH_BOUNDS[0],
        max_val=ADAPTIVE_SEVERITY_HIGH_BOUNDS[1],
    ),
    "beta_decay_halflife_days": ParameterBounds(
        default=float(BETA_DECAY_HALFLIFE_DAYS), min_val=30.0, max_val=120.0,
    ),
    "llm_override_confidence_threshold": ParameterBounds(
        default=LLM_OVERRIDE_CONFIDENCE_THRESHOLD, min_val=0.4, max_val=0.8,
    ),
    "openness_silence_threshold": ParameterBounds(
        default=LLM_OPENNESS_SILENCE_THRESHOLD, min_val=0.15, max_val=0.35,
    ),
}


def _clamp(value: float, bounds: ParameterBounds) -> float:
    """Clamp value within safety envelope. Returns default for NaN/Inf."""
    if math.isnan(value) or math.isinf(value):
        return bounds.default
    return max(bounds.min_val, min(bounds.max_val, value))


# ══════════════════════════════════════════════════════════════
# 用户统计数据 (User Statistics)
# ══════════════════════════════════════════════════════════════

@dataclass
class UserStats:
    """
    Aggregated user statistics for adaptive parameter computation.
    Computed by MemoryModule from interaction history.
    """
    emotion_variance: float = 0.0
    emotion_autocorrelation: Optional[float] = None
    severity_history: List[float] = field(default_factory=list)
    strategy_feedback: Dict[str, List[float]] = field(default_factory=dict)
    n_interactions: int = 0


# ══════════════════════════════════════════════════════════════
# 自适应参数输出 (Adaptive Parameter Output)
# ══════════════════════════════════════════════════════════════

@dataclass
class AdaptiveParams:
    """
    Per-turn adaptive parameter overrides computed from user history + LLM context.

    All values are clamped within ParameterBounds safety envelopes.
    When insufficient data is available, defaults from config.py are used.
    """
    # Emotion baseline
    baseline_lr: float = BASELINE_LR
    baseline_sigma_lr: float = BASELINE_SIGMA_LR
    # Emotion inertia
    emotion_inertia_alpha: float = EMOTION_INERTIA_ALPHA
    # Severity bins
    severity_low_threshold: float = 0.3
    severity_high_threshold: float = 0.7
    # TS priors
    cold_start_mu: Dict[str, float] = field(default_factory=dict)
    beta_decay_halflife_days: float = float(BETA_DECAY_HALFLIFE_DAYS)
    # LLM override
    llm_override_confidence_threshold: float = LLM_OVERRIDE_CONFIDENCE_THRESHOLD
    openness_silence_threshold: float = LLM_OPENNESS_SILENCE_THRESHOLD


# ══════════════════════════════════════════════════════════════
# 自适应参数管理器 (Adaptive Parameter Manager)
# ══════════════════════════════════════════════════════════════

class AdaptiveParameterManager:
    """
    Computes per-turn parameter overrides from user history + LLM context.

    Three adaptation sources:
      1. User emotional variance → baseline_lr, emotion_inertia_alpha
      2. User interaction history → cold_start_mu, severity thresholds
      3. LLM confidence signal → override thresholds

    All outputs clamped within ParameterBounds (safety envelope).
    Class A parameters are never modified.

    Academic grounding:
      - baseline_lr: Schreuder et al. SPC approach (inverse-variance scaling)
      - emotion_inertia_alpha: autocorrelation-based smoothing
      - severity bins: user-specific quantile adaptation
      - cold_start_mu: empirical mean gradual transition (Deb et al.)
    """

    def __init__(self) -> None:
        self.bounds = PARAMETER_BOUNDS

    def compute(
        self,
        user_stats: UserStats,
        llm_confidence: float = 1.0,
    ) -> AdaptiveParams:
        """
        Compute adaptive parameter overrides for current turn.

        Args:
            user_stats: Aggregated user statistics from memory
            llm_confidence: LLM self-reported analysis confidence [0, 1]

        Returns:
            AdaptiveParams with all values clamped within safety bounds
        """
        params = AdaptiveParams()

        # Source 1: Emotional variance → baseline LR and inertia
        params.baseline_lr = self.compute_adaptive_baseline_lr(
            user_stats.emotion_variance
        )
        params.baseline_sigma_lr = params.baseline_lr  # Coupled to baseline LR

        params.emotion_inertia_alpha = self.compute_adaptive_inertia(
            user_stats.emotion_autocorrelation
        )

        # Source 2: Interaction history → severity bins and TS priors
        low_th, high_th = self.compute_severity_percentiles(
            user_stats.severity_history
        )
        params.severity_low_threshold = low_th
        params.severity_high_threshold = high_th

        params.cold_start_mu = self.compute_adaptive_ts_priors(
            user_stats.strategy_feedback, user_stats.n_interactions
        )

        # Source 3: LLM confidence — no adaptive override needed,
        # confidence flows directly into planning.py Step 4.6

        return params

    def compute_adaptive_baseline_lr(self, emotion_variance: float) -> float:
        """
        Adaptive baseline learning rate (Schreuder et al. SPC approach).

        Formula: lr = 1 / (2 + emotion_variance * 20)
          - High-variance users → lower LR (0.02), slower baseline adaptation
          - Stable users → higher LR (0.15), faster calibration

        Args:
            emotion_variance: Variance of recent valence observations

        Returns:
            Clamped learning rate within [0.02, 0.15]
        """
        if emotion_variance <= 0.0:
            return self.bounds["baseline_lr"].default
        raw_lr = 1.0 / (2.0 + emotion_variance * ADAPTIVE_LR_VARIANCE_SENSITIVITY)
        return _clamp(raw_lr, self.bounds["baseline_lr"])

    def compute_adaptive_inertia(self, emotion_autocorrelation: Optional[float]) -> float:
        """
        Adaptive emotion inertia alpha (autocorrelation-based).

        Formula: alpha = 0.3 + 0.2 * autocorr(valence, lag=1)
          - High emotional persistence → higher alpha (more smoothing)
          - Volatile users → lower alpha (faster response)

        Args:
            emotion_autocorrelation: Lag-1 autocorrelation of valence [-1, 1],
                                     or None when insufficient data

        Returns:
            Clamped alpha within [0.2, 0.6]
        """
        # Default when autocorrelation not available (None = insufficient data)
        if emotion_autocorrelation is None:
            return self.bounds["emotion_inertia_alpha"].default
        raw_alpha = ADAPTIVE_INERTIA_BASE + ADAPTIVE_INERTIA_SCALE * emotion_autocorrelation
        return _clamp(raw_alpha, self.bounds["emotion_inertia_alpha"])

    def compute_severity_percentiles(
        self,
        severity_history: List[float],
    ) -> Tuple[float, float]:
        """
        Adaptive severity bins (user-specific quantiles).

        Replaces fixed [0.3, 0.7] with personal distribution quartiles.
        Falls back to static values when insufficient data (<10 observations).

        Minimum spread enforced: high_th >= low_th + 0.2

        Args:
            severity_history: Historical severity scores

        Returns:
            (low_threshold, high_threshold) both clamped within bounds
        """
        low_bounds = self.bounds["severity_low_threshold"]
        high_bounds = self.bounds["severity_high_threshold"]

        if len(severity_history) < ADAPTIVE_SEVERITY_MIN_SAMPLES:
            return low_bounds.default, high_bounds.default

        arr = np.array(severity_history, dtype=np.float64)
        raw_low = float(np.percentile(arr, ADAPTIVE_SEVERITY_PERCENTILE_LOW))
        raw_high = float(np.percentile(arr, ADAPTIVE_SEVERITY_PERCENTILE_HIGH))

        low = _clamp(raw_low, low_bounds)
        high = _clamp(raw_high, high_bounds)

        # Enforce minimum spread
        half_spread = ADAPTIVE_SEVERITY_MIN_SPREAD / 2.0
        if high - low < ADAPTIVE_SEVERITY_MIN_SPREAD - 1e-9:
            mid = (low + high) / 2.0
            low = _clamp(mid - half_spread, low_bounds)
            high = _clamp(mid + half_spread, high_bounds)

        return low, high

    def compute_adaptive_ts_priors(
        self,
        strategy_feedback: Dict[str, List[float]],
        n_interactions: int,
    ) -> Dict[str, float]:
        """
        Adaptive Thompson Sampling cold-start priors (empirical mean update).

        Formula: mu_si = (1-w) * STRATEGY_COLD_START_MU[s] + w * empirical_mean[s]
          w = min(1.0, n_feedback / ADAPTIVE_TS_BLEND_HORIZON)

        Prior beliefs shift toward observed effectiveness as data accumulates.

        Args:
            strategy_feedback: {strategy: [feedback_scores]} from memory
            n_interactions: Total interaction count

        Returns:
            Dict of strategy → adaptive prior mu values.
            Empty dict when insufficient data (caller falls back to static).
        """
        from .planning import STRATEGY_COLD_START_MU

        if n_interactions < ADAPTIVE_TS_MIN_INTERACTIONS:
            return {}  # Not enough data, use static priors

        result: Dict[str, float] = {}
        for strategy, scores in strategy_feedback.items():
            if not scores:
                continue
            n_feedback = len(scores)
            empirical_mean = float(np.mean(scores))
            w = min(1.0, n_feedback / ADAPTIVE_TS_BLEND_HORIZON)
            static_mu = STRATEGY_COLD_START_MU.get(strategy, COLD_START_MU)
            adaptive_mu = (1.0 - w) * static_mu + w * empirical_mean
            # Clamp to reasonable range to prevent extreme priors
            result[strategy] = max(ADAPTIVE_TS_PRIOR_MIN, min(ADAPTIVE_TS_PRIOR_MAX, adaptive_mu))

        return result
