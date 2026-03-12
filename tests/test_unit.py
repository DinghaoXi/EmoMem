"""
EmoMem 精品单元测试 (Offline Unit Tests — No LLM API Required)
================================================================
覆盖范围 (18 tests):
  1. 数据模型 (models.py):     EmotionVector 归一化/效价, StateVector 构造, WorkingMemory 默认值与 session reset
  2. 自适应框架 (adaptation.py): baseline_lr, inertia, severity_bins, ts_priors 计算, NaN/Inf 安全性
  3. 记忆模块 (memory.py):     ring buffer 容量上限, 情感记忆基线更新, episodic memory 存储
  4. 策略选择 (planning.py):    Thompson Sampling 基本功能, crisis override, acute phase 约束, 强观测偏移
  5. 配置参数 (config.py):      Class A 安全参数不可变, 参数范围合理性, 具体安全值

所有测试纯算法逻辑，不调用 LLM API。
"""

import math
import numpy as np
import pytest

from src.config import (
    NUM_EMOTIONS, VALENCE_SIGNS, RING_BUFFER_SIZE,
    CRISIS_THRESHOLD, CRISIS_STEP_THRESHOLD,
    URGENCY_ALPHA1, URGENCY_ALPHA2, URGENCY_ALPHA4,
    BASELINE_LR, EMOTION_INERTIA_ALPHA,
    STRATEGIES, ACUTE_ALLOWED_STRATEGIES,
)
from src.models import (
    EmotionVector, StateVector, WorkingMemory, IntentDistribution,
    AffectiveMemory, EmotionBaseline, BetaParams,
    TrajectoryResult, RecoveryPhase, TrajectoryDirection,
    EpisodeUnit, ContextVector, PlanningOutput, RawSignal,
)
from src.adaptation import (
    AdaptiveParameterManager, UserStats, AdaptiveParams,
    PARAMETER_BOUNDS, ParameterBounds,
)
from src.memory import MemoryModule
from src.planning import (
    select_strategy, analyze_trajectory, discretize_context,
    assess_situation, _ols_slope,
)


# ══════════════════════════════════════════════════════════════
# 1. 数据模型 (Data Models) — 3 tests
# ══════════════════════════════════════════════════════════════

class TestDataModels:
    """数据模型测试: EmotionVector, StateVector, WorkingMemory"""

    def test_emotion_vector_normalization_and_valence(self):
        """
        Test 1: EmotionVector normalization and valence computation.
        Verify: (1) default e sums to 1 (on simplex Delta^7);
                (2) valence = sigma . e with known values;
                (3) pure joy -> +1, pure sadness -> -1, uniform -> -0.125.
        """
        # Default: uniform distribution [1/8]*8 on the simplex
        ev_default = EmotionVector()
        assert ev_default.e.shape == (NUM_EMOTIONS,), "Emotion vector should have 8 dimensions"
        assert abs(ev_default.e.sum() - 1.0) < 1e-9, "Default distribution should sum to 1"

        # Uniform valence: sigma = [+1,+1,-1,0,-1,-1,-1,+1] -> sum = -1, mean = -1/8 = -0.125
        expected_uniform_valence = float(np.dot(VALENCE_SIGNS, ev_default.e))
        assert abs(ev_default.valence() - expected_uniform_valence) < 1e-9
        assert abs(ev_default.valence() - (-0.125)) < 1e-9, "Uniform valence should be -0.125"

        # Pure joy: e = [1,0,...,0], sigma[0] = +1 -> valence = +1
        joy_e = np.zeros(NUM_EMOTIONS)
        joy_e[0] = 1.0
        ev_joy = EmotionVector(e=joy_e, intensity=0.9, confidence=0.95)
        assert abs(ev_joy.valence() - 1.0) < 1e-9, "Pure joy valence should be +1"
        assert ev_joy.intensity == 0.9
        assert ev_joy.confidence == 0.95

        # Pure sadness: e[4] = 1.0, sigma[4] = -1 -> valence = -1
        sad_e = np.zeros(NUM_EMOTIONS)
        sad_e[4] = 1.0
        ev_sad = EmotionVector(e=sad_e, intensity=0.8)
        assert abs(ev_sad.valence() - (-1.0)) < 1e-9, "Pure sadness valence should be -1"

        # Pure surprise: e[3] = 1.0, sigma[3] = 0 -> valence = 0
        surprise_e = np.zeros(NUM_EMOTIONS)
        surprise_e[3] = 1.0
        ev_surprise = EmotionVector(e=surprise_e)
        assert abs(ev_surprise.valence()) < 1e-9, "Pure surprise valence should be 0"

    def test_state_vector_construction(self):
        """
        Test 2: StateVector default construction and custom construction.
        Verify: all fields have sensible defaults; custom values propagate correctly.
        """
        sv = StateVector()
        # Default values
        assert sv.urgency_level == 0.0, "Default urgency should be 0"
        assert sv.relationship_depth == 0.0
        assert sv.topic_keywords == []
        assert sv.controllability is None, "controllability should default to None (no LLM)"
        assert sv.life_impact is None
        assert sv.llm_confidence is None
        assert sv.recommended_approach is None
        assert sv.llm_urgency is None
        assert sv.recovery_phase is None
        assert sv.planning_intent is None

        # Custom construction
        ev = EmotionVector(intensity=0.7, confidence=0.8)
        intent = IntentDistribution(probabilities={
            "VENT": 0.8, "COMFORT": 0.1, "CRISIS": 0.0,
            "ADVICE": 0.0, "SHARE_JOY": 0.0,
            "CHAT": 0.1, "REFLECT": 0.0,
        })
        sv_custom = StateVector(
            emotion=ev,
            intent=intent,
            urgency_level=0.75,
            relationship_depth=0.4,
        )
        assert sv_custom.emotion.intensity == 0.7
        assert sv_custom.intent.top_intent() == "VENT"
        assert sv_custom.urgency_level == 0.75

        # IntentDistribution.top_intent() with empty probabilities
        intent_empty = IntentDistribution(probabilities={})
        assert intent_empty.top_intent() == "CHAT", "Empty probabilities should return CHAT"

    def test_working_memory_defaults_and_session_reset(self):
        """
        Test 3: WorkingMemory defaults and MemoryModule.reset_working_memory().
        Verify: (1) all fields have correct defaults;
                (2) reset_working_memory clears all fields to initial state.
        """
        wm = WorkingMemory()
        assert wm.turn_count == 0, "New session turn_count should be 0"
        assert wm.crisis_cooldown_counter == 0
        assert wm.crisis_episode_count == 0
        assert wm.prev_emotion is None
        assert wm.prev_intent is None
        assert wm.active_emotion is None
        assert wm.last_strategy == ""
        assert wm.user_name == "用户"
        assert wm.session_buffer == ""
        assert wm.active_topic == ""
        assert wm.active_intent == ""

        # Simulate usage then reset
        mem_module = MemoryModule()
        mem_module.working_memory.turn_count = 15
        mem_module.working_memory.crisis_cooldown_counter = 3
        mem_module.working_memory.crisis_episode_count = 2
        mem_module.working_memory.last_strategy = "active_listening"
        mem_module.working_memory.active_emotion = EmotionVector(intensity=0.9)

        # Reset should restore all fields
        mem_module.reset_working_memory()
        assert mem_module.working_memory.turn_count == 0
        assert mem_module.working_memory.crisis_cooldown_counter == 0
        assert mem_module.working_memory.crisis_episode_count == 0
        assert mem_module.working_memory.last_strategy == ""
        assert mem_module.working_memory.active_emotion is None


# ══════════════════════════════════════════════════════════════
# 2. 自适应参数框架 (Adaptive Parameter Framework) — 5 tests
# ══════════════════════════════════════════════════════════════

class TestAdaptiveFramework:
    """自适应参数框架测试: 四种计算公式 + NaN/Inf 安全性"""

    def setup_method(self):
        """Initialize AdaptiveParameterManager before each test"""
        self.manager = AdaptiveParameterManager()

    def test_adaptive_baseline_lr(self):
        """
        Test 4: Baseline LR formula lr = 1/(2 + variance*20), clamped [0.02, 0.15].
        Verify: (1) zero variance -> default; (2) high variance -> low lr;
                (3) low variance -> high lr (clamped); (4) negative -> default.
        """
        # Zero variance -> default
        lr_zero = self.manager.compute_adaptive_baseline_lr(0.0)
        assert lr_zero == BASELINE_LR, "Zero variance should return default LR"

        # Negative variance -> default
        lr_neg = self.manager.compute_adaptive_baseline_lr(-0.1)
        assert lr_neg == BASELINE_LR, "Negative variance should return default LR"

        # High variance (1.0) -> lr = 1/(2+20) ~= 0.0455, within [0.02, 0.15]
        lr_high = self.manager.compute_adaptive_baseline_lr(1.0)
        expected = 1.0 / (2.0 + 1.0 * 20.0)
        assert abs(lr_high - expected) < 1e-9
        assert 0.02 <= lr_high <= 0.15, f"High variance LR={lr_high} should be in safe range"

        # Very low variance -> lr ~ 1/2.02 = 0.495 -> clamped to 0.15
        lr_tiny = self.manager.compute_adaptive_baseline_lr(0.001)
        assert lr_tiny == 0.15, "Very low variance LR should be clamped to 0.15"

        # Very high variance (100.0) -> lr = 1/(2+2000) ~= 0.0005 -> clamped to 0.02
        lr_extreme = self.manager.compute_adaptive_baseline_lr(100.0)
        assert lr_extreme == 0.02, "Extreme variance should be clamped to min 0.02"

    def test_adaptive_inertia(self):
        """
        Test 5: Inertia formula alpha = 0.3 + 0.2 * autocorrelation, clamped [0.2, 0.6].
        Verify: (1) None -> default; (2) zero autocorr -> 0.3;
                (3) positive -> higher; (4) negative -> lower; (5) clamped.
        """
        # None (insufficient data) -> default
        alpha_none = self.manager.compute_adaptive_inertia(None)
        assert alpha_none == EMOTION_INERTIA_ALPHA, "None autocorr should return default inertia"

        # Zero autocorrelation (valid statistical value) -> 0.3 + 0.2*0.0 = 0.3
        alpha_zero = self.manager.compute_adaptive_inertia(0.0)
        assert abs(alpha_zero - 0.3) < 1e-9, "Zero autocorr should yield formula value 0.3"

        # Positive autocorrelation (0.8) -> 0.3 + 0.2*0.8 = 0.46
        alpha_pos = self.manager.compute_adaptive_inertia(0.8)
        assert abs(alpha_pos - 0.46) < 1e-9

        # Full positive autocorrelation (1.0) -> 0.3 + 0.2*1.0 = 0.5
        alpha_full = self.manager.compute_adaptive_inertia(1.0)
        assert abs(alpha_full - 0.5) < 1e-9

        # Strong negative autocorrelation (-1.0) -> 0.3 + 0.2*(-1.0) = 0.1 -> clamped to 0.2
        alpha_neg = self.manager.compute_adaptive_inertia(-1.0)
        assert alpha_neg == 0.2, "Strong negative autocorr should be clamped to lower bound 0.2"

        # Extreme positive value (2.0) -> 0.3 + 0.4 = 0.7 -> clamped to 0.6
        alpha_extreme = self.manager.compute_adaptive_inertia(2.0)
        assert alpha_extreme == 0.6, "Extreme positive should be clamped to upper bound 0.6"

    def test_severity_percentiles(self):
        """
        Test 6: Severity percentiles from P25/P75.
        Verify: (1) <10 samples -> defaults (0.3, 0.7);
                (2) sufficient data -> quantiles within bounds;
                (3) uniform values -> minimum spread 0.2 enforced.
        """
        # Insufficient samples -> defaults
        low, high = self.manager.compute_severity_percentiles([0.1, 0.5, 0.9])
        assert low == 0.3 and high == 0.7, "Insufficient data should return defaults"

        # Empty list -> defaults
        low_empty, high_empty = self.manager.compute_severity_percentiles([])
        assert low_empty == 0.3 and high_empty == 0.7

        # 9 samples (just under threshold) -> defaults
        low_9, high_9 = self.manager.compute_severity_percentiles([0.5] * 9)
        assert low_9 == 0.3 and high_9 == 0.7

        # Uniformly distributed 0-1 with 20 samples
        history = [i / 19.0 for i in range(20)]
        low, high = self.manager.compute_severity_percentiles(history)
        assert 0.15 <= low <= 0.45, f"P25 low threshold {low} should be in bounds"
        assert 0.55 <= high <= 0.85, f"P75 high threshold {high} should be in bounds"
        assert high - low >= 0.2 - 1e-6, "Minimum spread should be enforced"

        # All identical values -> minimum spread should be enforced
        uniform = [0.4] * 20
        low_u, high_u = self.manager.compute_severity_percentiles(uniform)
        assert high_u - low_u >= 0.2 - 1e-6, "Identical values should enforce min spread 0.2"

    def test_adaptive_ts_priors(self):
        """
        Test 7: TS priors with empirical mean blending.
        Verify: (1) <5 interactions -> empty dict;
                (2) 20 feedbacks -> w=1.0 -> clamped to [0.2, 0.8];
                (3) partial feedback -> blended values.
        """
        # Insufficient interactions
        result_cold = self.manager.compute_adaptive_ts_priors(
            {"active_listening": [0.7, 0.8]}, n_interactions=3
        )
        assert result_cold == {}, "Insufficient interactions should return empty dict"

        # Exactly at threshold (5) -> should compute
        result_at = self.manager.compute_adaptive_ts_priors(
            {"active_listening": [0.6]}, n_interactions=5
        )
        assert "active_listening" in result_at, "At threshold should compute priors"

        # 20 feedbacks -> w = min(1.0, 20/20) = 1.0 -> mu = 0*static + 1.0*0.9 = 0.9 -> clamped to 0.8
        feedback_high = {"active_listening": [0.9] * 20}
        result = self.manager.compute_adaptive_ts_priors(feedback_high, n_interactions=25)
        assert "active_listening" in result
        assert result["active_listening"] == 0.8, "Very high empirical mean should be clamped to 0.8"

        # 20 feedbacks with very low scores -> clamped to 0.2
        feedback_low = {"problem_solving": [0.05] * 20}
        result_low = self.manager.compute_adaptive_ts_priors(feedback_low, n_interactions=25)
        assert "problem_solving" in result_low
        assert result_low["problem_solving"] == 0.2, "Very low empirical mean should be clamped to 0.2"

        # Partial feedback: 5 feedbacks -> w = 5/20 = 0.25 -> blended
        feedback_few = {"emotional_validation": [0.3, 0.3, 0.3, 0.3, 0.3]}
        result_few = self.manager.compute_adaptive_ts_priors(feedback_few, n_interactions=10)
        assert "emotional_validation" in result_few
        mu = result_few["emotional_validation"]
        assert 0.2 <= mu <= 0.8, f"Blended prior {mu} should be in safe range"

        # Empty feedback for a strategy -> should not appear
        result_empty = self.manager.compute_adaptive_ts_priors(
            {"active_listening": []}, n_interactions=10
        )
        assert "active_listening" not in result_empty, "Empty feedback list should not produce a prior"

    def test_nan_inf_safety(self):
        """
        Test 8: NaN/Inf safety across all adaptive methods.
        Verify: no NaN or Inf values propagate to output.
        """
        # NaN variance -> should not crash, returns safe value
        lr_nan = self.manager.compute_adaptive_baseline_lr(float('nan'))
        assert not math.isnan(lr_nan) and not math.isinf(lr_nan), "NaN input should not produce NaN LR"

        # Inf variance -> lr = 1/(2+inf*20) = 0 -> clamped to 0.02
        lr_inf = self.manager.compute_adaptive_baseline_lr(float('inf'))
        assert lr_inf == 0.02, "Inf variance should yield minimum LR"

        # -Inf variance -> treated as <= 0, returns default
        lr_neginf = self.manager.compute_adaptive_baseline_lr(float('-inf'))
        assert lr_neginf == BASELINE_LR, "-Inf variance should return default LR"

        # NaN autocorrelation -> should not crash
        alpha_nan = self.manager.compute_adaptive_inertia(float('nan'))
        assert not math.isnan(alpha_nan) and not math.isinf(alpha_nan)

        # Inf autocorrelation -> _clamp returns default
        alpha_inf = self.manager.compute_adaptive_inertia(float('inf'))
        assert alpha_inf == EMOTION_INERTIA_ALPHA, "Inf autocorr should return default"

        # -Inf autocorrelation -> _clamp returns default
        alpha_neginf = self.manager.compute_adaptive_inertia(float('-inf'))
        assert alpha_neginf == EMOTION_INERTIA_ALPHA, "-Inf autocorr should return default"

        # Full compute() with pathological inputs should not crash
        stats = UserStats(
            emotion_variance=float('inf'),
            emotion_autocorrelation=float('nan'),
            severity_history=[float('nan')] * 5,
            n_interactions=3,
        )
        params = self.manager.compute(stats, llm_confidence=0.5)
        assert isinstance(params, AdaptiveParams), "compute() should return valid AdaptiveParams"
        assert not math.isnan(params.baseline_lr)
        assert not math.isnan(params.emotion_inertia_alpha)
        assert not math.isnan(params.severity_low_threshold)
        assert not math.isnan(params.severity_high_threshold)


# ══════════════════════════════════════════════════════════════
# 3. 记忆模块 (Memory Module) — 3 tests
# ══════════════════════════════════════════════════════════════

class TestMemoryModule:
    """记忆模块测试: ring buffer, 情感基线更新, episodic memory"""

    def test_ring_buffer_capacity_limit(self):
        """
        Test 9: Ring buffer capacity enforced at maxlen=RING_BUFFER_SIZE=20.
        Verify: (1) buffer never exceeds K_buf=20 after 30 writes;
                (2) oldest entries are evicted FIFO.
        """
        mem = MemoryModule()
        baseline = mem.affective_memory.emotion_baseline

        # Verify initial state
        assert len(baseline.ring_buffer) == 0, "Ring buffer should start empty"
        assert baseline.ring_buffer.maxlen == RING_BUFFER_SIZE, \
            f"Ring buffer maxlen should be {RING_BUFFER_SIZE}"

        # Write 30 observations, verify buffer caps at 20
        for i in range(30):
            e = np.ones(NUM_EMOTIONS) / NUM_EMOTIONS
            e[0] = 0.5 + i * 0.01  # Slight variation
            e = e / e.sum()  # Re-normalize to simplex
            ev = EmotionVector(e=e, intensity=0.5, confidence=0.7)
            mem.update_emotion_baseline(ev)

        assert len(baseline.ring_buffer) == RING_BUFFER_SIZE, \
            f"Ring buffer size should be {RING_BUFFER_SIZE}, got {len(baseline.ring_buffer)}"

        # Verify FIFO eviction: oldest 10 entries (i=0..9) evicted, newest (i=20..29) retained
        oldest_in_buf = baseline.ring_buffer[0]
        newest_in_buf = baseline.ring_buffer[-1]
        assert newest_in_buf[0] > oldest_in_buf[0], \
            "Newest entry should have higher e[0] than oldest remaining"

    def test_emotion_baseline_update(self):
        """
        Test 10: Emotion baseline EMA update with 2-sigma outlier exclusion.
        Verify: (1) normal observations update baseline and increase confidence;
                (2) outlier (>2 sigma) is excluded from baseline update;
                (3) confidence decays on outlier observation.

        Note: initial confidence=0.2 < BASELINE_RESET_THRESHOLD=0.3, so
        _reset_baseline_from_buffer() triggers until buffer has >= 3 entries.
        We feed 4 normal observations first to stabilize.
        """
        mem = MemoryModule()
        baseline = mem.affective_memory.emotion_baseline

        # Warm up: 4 normal observations to stabilize confidence above 0.3
        for i in range(4):
            warm_e = np.ones(NUM_EMOTIONS) / NUM_EMOTIONS
            warm_e[0] += 0.01 * (i + 1)
            warm_e = warm_e / warm_e.sum()
            mem.update_emotion_baseline(EmotionVector(e=warm_e, intensity=0.5))

        # After warm-up, _reset_baseline_from_buffer sets confidence to 0.5
        confidence_before = baseline.confidence
        assert confidence_before >= 0.3, \
            f"Post-warmup confidence should be >= 0.3, got {confidence_before}"

        # Normal observation: should update baseline and increase confidence
        normal_e = np.ones(NUM_EMOTIONS) / NUM_EMOTIONS
        normal_e[0] += 0.02
        normal_e = normal_e / normal_e.sum()
        mem.update_emotion_baseline(EmotionVector(e=normal_e, intensity=0.5))
        assert baseline.confidence >= confidence_before, \
            "Normal observation should increase or maintain confidence"

        # Save state before outlier
        baseline_before_outlier = baseline.normal_state.copy()
        confidence_before_outlier = baseline.confidence

        # Outlier observation: extreme shift (pure anger), should be excluded by 2-sigma rule
        outlier_e = np.zeros(NUM_EMOTIONS)
        outlier_e[6] = 1.0  # anger = 1.0
        ev_outlier = EmotionVector(e=outlier_e, intensity=0.9)
        mem.update_emotion_baseline(ev_outlier)

        # Baseline should NOT be updated by outlier
        assert np.allclose(baseline.normal_state, baseline_before_outlier, atol=1e-9), \
            "Outlier observation should not update baseline"
        # Confidence should decay on outlier
        assert baseline.confidence < confidence_before_outlier, \
            "Outlier should decrease confidence"

    def test_episodic_memory_storage(self):
        """
        Test 11: Episodic memory selective write with importance threshold tau_write=0.3.
        Verify: (1) high importance (intensity=0.9) -> stored;
                (2) low importance (intensity=0.0) -> not stored;
                (3) deferred update queue always populated regardless of storage.
        """
        mem = MemoryModule()

        # High importance state vector (high emotional intensity)
        ev_high = EmotionVector(intensity=0.9, confidence=0.8)
        sv_high = StateVector(
            emotion=ev_high,
            intent=IntentDistribution(probabilities={
                "VENT": 0.8, "COMFORT": 0.1, "ADVICE": 0.0,
                "SHARE_JOY": 0.0, "CHAT": 0.1, "REFLECT": 0.0, "CRISIS": 0.0,
            }),
            raw_signal=RawSignal(text_content="I am really sad today"),
        )
        planning_out = PlanningOutput(
            context_vector=ContextVector(phase=RecoveryPhase.EARLY_RECOVERY),
        )

        episode = mem.write(sv_high, "emotional_validation", planning_out)

        # importance: EmotionalArousal=0.9*0.3=0.27 + LifeImpact=0.5*0.3=0.15 = 0.42 > 0.3
        assert episode is not None, "High importance episode should be stored"
        assert len(mem.episodic_memory) == 1
        assert episode.agent_strategy == "emotional_validation"

        # Deferred update queue should be populated
        assert len(mem.deferred_update_queue) == 1
        assert mem.deferred_update_queue[0].strategy == "emotional_validation"

        # Low importance: intensity=0.0 -> importance ~ 0.15 (only LifeImpact mock=0.5*0.3) < 0.3
        ev_low = EmotionVector(intensity=0.0, confidence=0.5)
        sv_low = StateVector(
            emotion=ev_low,
            raw_signal=RawSignal(text_content="ok"),
        )
        episode_low = mem.write(sv_low, "active_listening", planning_out)

        # importance = 0.3*0 + 0.3*0.5 + 0.25*0 + 0.15*0 = 0.15 < 0.3 -> not stored
        assert episode_low is None, "Low importance episode should not be stored"
        assert len(mem.episodic_memory) == 1, "Episodic memory should not grow"
        # Deferred update queue should still grow (strategy learning is independent of storage)
        assert len(mem.deferred_update_queue) == 2


# ══════════════════════════════════════════════════════════════
# 4. 策略选择 (Strategy Selection) — 4 tests
# ══════════════════════════════════════════════════════════════

class TestStrategySelection:
    """策略选择测试: Thompson Sampling, crisis override, acute phase constraint, strong observation"""

    def test_thompson_sampling_diversity(self):
        """
        Test 12: Thompson Sampling returns valid strategy names and shows diversity on cold start.
        Verify: (1) all returned strategies are in STRATEGIES;
                (2) cold-start Beta(2,2) priors produce diverse selections over many runs.
        """
        affective = AffectiveMemory()
        sv = StateVector(urgency_level=0.3)
        traj = TrajectoryResult(current_phase=RecoveryPhase.STABLE_STATE)
        ctx = ContextVector(
            phase=RecoveryPhase.STABLE_STATE,
            severity_level="low",
            intent_group="neutral",
        )

        selected_strategies = set()
        for _ in range(100):
            strategy = select_strategy(ctx, affective, sv, traj)
            selected_strategies.add(strategy)
            assert strategy in STRATEGIES, f"Strategy '{strategy}' not in valid strategy list"

        # Cold start should produce diversity (Beta(2,2) sampling has high variance)
        assert len(selected_strategies) >= 3, \
            f"TS cold start should produce diversity, got {len(selected_strategies)} unique strategies"

    def test_crisis_override(self):
        """
        Test 13: Crisis override when urgency > 0.9 forces CRISIS_PROTOCOL.
        Verify: (1) urgency > 0.9 -> always CRISIS_PROTOCOL;
                (2) urgency = 0.9 (at boundary, not exceeding) -> no trigger.
        """
        affective = AffectiveMemory()
        traj = TrajectoryResult(current_phase=RecoveryPhase.STABLE_STATE)
        ctx = ContextVector(
            phase=RecoveryPhase.STABLE_STATE,
            severity_level="high",
            intent_group="insight",
        )

        # High urgency > CRISIS_THRESHOLD=0.9
        sv_crisis = StateVector(urgency_level=0.95)
        for _ in range(20):
            strategy = select_strategy(ctx, affective, sv_crisis, traj)
            assert strategy == "CRISIS_PROTOCOL", \
                f"urgency>0.9 should always return CRISIS_PROTOCOL, got: {strategy}"

        # Extreme urgency = 1.0
        sv_extreme = StateVector(urgency_level=1.0)
        assert select_strategy(ctx, affective, sv_extreme, traj) == "CRISIS_PROTOCOL"

        # Boundary: urgency = 0.9 (equals threshold, does NOT exceed) -> should NOT trigger
        sv_boundary = StateVector(urgency_level=0.9)
        strategies = set()
        for _ in range(30):
            s = select_strategy(ctx, affective, sv_boundary, traj)
            strategies.add(s)
        assert "CRISIS_PROTOCOL" not in strategies, \
            "urgency=0.9 (at but not exceeding threshold) should not trigger CRISIS_PROTOCOL"

    def test_acute_phase_strategy_constraint(self):
        """
        Test 14: Acute distress phase constrains strategies to ACUTE_ALLOWED_STRATEGIES.
        Verify: in acute_distress phase, only {active_listening, emotional_validation,
                empathic_reflection, companionable_silence} are returned.
        """
        affective = AffectiveMemory()
        sv = StateVector(urgency_level=0.5)  # Below crisis threshold
        traj = TrajectoryResult(current_phase=RecoveryPhase.ACUTE_DISTRESS)
        ctx = ContextVector(
            phase=RecoveryPhase.ACUTE_DISTRESS,
            severity_level="high",
            intent_group="insight",
        )

        # Run enough iterations to verify constraint holds
        returned_strategies = set()
        for _ in range(100):
            strategy = select_strategy(ctx, affective, sv, traj)
            returned_strategies.add(strategy)
            assert strategy in ACUTE_ALLOWED_STRATEGIES, \
                f"Acute distress should not select '{strategy}', only {ACUTE_ALLOWED_STRATEGIES}"

        # Should still show some diversity within the allowed set
        assert len(returned_strategies) >= 2, \
            "Should have diversity within acute allowed strategies"

    def test_strong_observation_shifts_preference(self):
        """
        Test 15: Injecting a high alpha for one strategy makes it dominate TS selection.
        Verify: a strategy with Beta(100, 2) is selected significantly more often
                than cold-start strategies with Beta(2, 2).
        """
        ctx = ContextVector(
            phase=RecoveryPhase.STABLE_STATE,
            severity_level="low",
            intent_group="neutral",
        )
        sv = StateVector(urgency_level=0.3)
        traj = TrajectoryResult(current_phase=RecoveryPhase.STABLE_STATE)

        # Inject strong positive observation for active_listening
        affective = AffectiveMemory()
        ctx_key = ctx.to_key()
        affective.strategy_matrix[("active_listening", ctx_key)] = BetaParams(
            alpha=100.0, beta=2.0, observation_count=50,
        )

        al_count = 0
        n_trials = 100
        for _ in range(n_trials):
            s = select_strategy(ctx, affective, sv, traj)
            if s == "active_listening":
                al_count += 1

        # With Beta(100,2) mean ~= 0.98, this strategy should dominate
        assert al_count >= 60, \
            f"Strong observation should make active_listening dominate ({al_count}/{n_trials})"

        # Compare: inject strong negative observation for problem_solving
        affective.strategy_matrix[("problem_solving", ctx_key)] = BetaParams(
            alpha=1.0, beta=100.0, observation_count=50,
        )
        ps_count = 0
        for _ in range(n_trials):
            s = select_strategy(ctx, affective, sv, traj)
            if s == "problem_solving":
                ps_count += 1

        # With Beta(1,100), problem_solving should almost never be selected
        assert ps_count <= 5, \
            f"Strong negative observation should suppress problem_solving ({ps_count}/{n_trials})"


# ══════════════════════════════════════════════════════════════
# 5. 配置参数 (Configuration Parameters) — 3 tests
# ══════════════════════════════════════════════════════════════

class TestConfigParameters:
    """配置参数测试: Class A frozen, parameter ranges, specific safety values"""

    def test_class_a_parameters_frozen(self):
        """
        Test 16: Class A safety parameters have frozen=True and min==max==default.
        Verify: all Class A params are immutable in the safety envelope.
        """
        class_a_params = [
            "crisis_threshold",
            "crisis_step_threshold",
            "urgency_alpha1",
            "urgency_alpha2",
            "urgency_alpha4",
        ]
        for name in class_a_params:
            bounds = PARAMETER_BOUNDS[name]
            assert bounds.frozen is True, f"Class A param '{name}' should be frozen=True"
            assert bounds.min_val == bounds.max_val == bounds.default, \
                f"Class A param '{name}' min/max/default should all be equal (immutable)"

        # Confirm compute() does not modify Class A parameters
        manager = AdaptiveParameterManager()
        params = manager.compute(UserStats(), llm_confidence=1.0)
        # AdaptiveParams does not contain Class A fields — they are constants in config.py
        assert CRISIS_THRESHOLD == 0.9, "compute() must not modify CRISIS_THRESHOLD"
        assert URGENCY_ALPHA1 == 1.0, "compute() must not modify URGENCY_ALPHA1"

    def test_parameter_ranges_reasonable(self):
        """
        Test 17: All parameter bounds are reasonable.
        Verify: (1) defaults are within [min, max];
                (2) non-frozen params have min < max;
                (3) no degenerate ranges.
        """
        for name, bounds in PARAMETER_BOUNDS.items():
            # All params: default within bounds
            assert bounds.min_val <= bounds.default <= bounds.max_val, \
                f"Param '{name}' default {bounds.default} not in [{bounds.min_val}, {bounds.max_val}]"

            if not bounds.frozen:
                # Class C: must have adjustable range
                assert bounds.min_val < bounds.max_val, \
                    f"Class C param '{name}' should have min < max"

                # Range should not be excessively large
                spread = bounds.max_val - bounds.min_val
                assert spread <= 100.0, f"Param '{name}' range {spread} is too large"

        # Specific range checks for key parameters
        lr_bounds = PARAMETER_BOUNDS["baseline_lr"]
        assert lr_bounds.min_val >= 0.01, "baseline_lr lower bound should not be too low"
        assert lr_bounds.max_val <= 0.3, "baseline_lr upper bound should not be too high"

        inertia_bounds = PARAMETER_BOUNDS["emotion_inertia_alpha"]
        assert inertia_bounds.min_val >= 0.1, "Inertia alpha lower bound should be reasonable"
        assert inertia_bounds.max_val <= 0.8, "Inertia alpha upper bound should be reasonable"

        # Severity bins constraint: high bounds must be above low bounds
        low_th_bounds = PARAMETER_BOUNDS["severity_low_threshold"]
        high_th_bounds = PARAMETER_BOUNDS["severity_high_threshold"]
        assert high_th_bounds.min_val > low_th_bounds.min_val, \
            "severity_high min should exceed severity_low min"

    def test_specific_safety_values(self):
        """
        Test 18: Specific safety-critical constant values.
        Verify: CRISIS_THRESHOLD=0.9, CRISIS_STEP_THRESHOLD=0.5,
                and other safety invariants hold exactly.
        """
        # Core safety thresholds
        assert CRISIS_THRESHOLD == 0.9, "CRISIS_THRESHOLD must be exactly 0.9"
        assert CRISIS_STEP_THRESHOLD == 0.5, "CRISIS_STEP_THRESHOLD must be exactly 0.5"

        # Urgency weights should match spec
        assert URGENCY_ALPHA1 == 1.0, "URGENCY_ALPHA1 (crisis intent weight) must be 1.0"
        assert URGENCY_ALPHA2 == 0.8, "URGENCY_ALPHA2 (negative emotion weight) must be 0.8"
        assert URGENCY_ALPHA4 == 0.7, "URGENCY_ALPHA4 (deterioration trend weight) must be 0.7"

        # Emotion dimensions
        assert NUM_EMOTIONS == 8, "Plutchik emotion space must have 8 dimensions"
        assert len(VALENCE_SIGNS) == NUM_EMOTIONS, "VALENCE_SIGNS length must match NUM_EMOTIONS"
        assert RING_BUFFER_SIZE == 20, "Ring buffer K_buf must be 20"

        # Strategy space
        assert len(STRATEGIES) == 10, "Strategy space must have 10 strategies"
        assert ACUTE_ALLOWED_STRATEGIES == {
            "active_listening", "emotional_validation",
            "empathic_reflection", "companionable_silence",
        }, "ACUTE_ALLOWED_STRATEGIES must match spec"

        # ACUTE_ALLOWED_STRATEGIES must be a subset of STRATEGIES
        for s in ACUTE_ALLOWED_STRATEGIES:
            assert s in STRATEGIES, f"Acute allowed strategy '{s}' must be in STRATEGIES"

        # Baseline learning rate defaults
        assert BASELINE_LR == 0.05, "BASELINE_LR default must be 0.05"
        assert EMOTION_INERTIA_ALPHA == 0.4, "EMOTION_INERTIA_ALPHA default must be 0.4"
