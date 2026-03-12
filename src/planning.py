"""
EmoMem 规划模块 (Planning Module) — §4 完整实现
==================================================
来源: emomem.md §4.1–§4.7

执行顺序 (按数据依赖):
  (1) 情绪轨迹分析 (§4.4) → (2) 情境评估 (§4.2) → (3) 目标推断 (§4.3) → (4) Thompson 采样策略选择 (§4.5)

子组件执行顺序说明:
  尽管文档节编号§4.2在§4.4之前，实际执行以数据依赖关系为准:
  - 情境评估需要轨迹分析的 direction 来计算 f_trend
  - Thompson 采样需要轨迹分析的 current_phase 和情境评估的 severity
"""

from __future__ import annotations

import logging
import math
import numpy as np

logger = logging.getLogger(__name__)
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .config import (
    # §4.4 轨迹分析参数
    TRAJECTORY_WINDOW_K, TAU_D,
    ACUTE_DISTRESS_DELTA_B, EARLY_RECOVERY_DELTA_B, CONSOLIDATION_DELTA_B,
    T_HOLD, EPSILON_H, EPSILON_H_EXIT, T_MAX_ACUTE,
    RECOVERY_PHASES, ACUTE_ALLOWED_STRATEGIES,
    # §4.2 情境评估参数
    SEVERITY_W1, SEVERITY_W2, SEVERITY_W3, SEVERITY_W4,
    F_TREND_MAP,
    # §4.5 Thompson Sampling 参数
    STRATEGIES, NUM_STRATEGIES,
    BETA_ALPHA0, BETA_BETA0, BETA_DECAY_HALFLIFE_DAYS,
    N_BLEND, N_PRIOR, COLD_START_MU,
    RELATIONSHIP_DEPTH_THRESHOLD,  # P_REL_ROLLBACK no longer needed (deterministic rollback)
    CRISIS_THRESHOLD, EXPLORATION_STRATEGIES, INSIGHT_STRATEGIES,
    # §3.6a planning intent 映射
    PLANNING_INTENT_PHI,
    # 通用
    EPSILON,
    VALENCE_SIGNS,
    # §4.5 LLM 语义信号阈值
    LLM_OPENNESS_SILENCE_THRESHOLD, LLM_OVERRIDE_CONFIDENCE_THRESHOLD,
    # §4.5 策略禁止阈值
    POSITIVE_REINFORCEMENT_VALENCE_BLOCK,
)

from .models import (
    StateVector, EmotionVector, EmotionBaseline,
    AffectiveMemory, RetrievedContext, BetaParams,
    TrajectoryResult, TrajectoryDirection, RecoveryPhase,
    SituationAppraisal, GoalSet, ContextVector, PlanningOutput,
)


# §4.5 LLM recommended_approach → 策略映射
# 将 LLM 的语义理解映射为 Thompson Sampling 的策略约束集
# 理论依据: 让具备上下文理解能力的 LLM 判断用户的沟通状态和支持需求,
# 替代硬编码关键词匹配, 提升跨场景泛化性 (Vinyals et al., 2015; Brown et al., 2020)
APPROACH_TO_STRATEGIES = {
    "passive_support": {"companionable_silence", "emotional_validation", "active_listening"},
    "active_validation": {"emotional_validation", "empathic_reflection", "active_listening",
                          "strength_recognition", "companionable_silence"},
    "gentle_exploration": {"gentle_guidance", "empathic_reflection", "active_listening",
                           "emotional_validation", "cognitive_reframing"},
    "solution_oriented": {"information_providing", "cognitive_reframing", "problem_solving",
                          "gentle_guidance", "empathic_reflection"},
    "positive_engagement": {"positive_reinforcement", "active_listening", "empathic_reflection",
                            "strength_recognition", "gentle_guidance"},
}


# ═══════════════════════════════════════════════════════════
# 辅助函数 / Helper Functions
# ═══════════════════════════════════════════════════════════

def _ols_slope(values: np.ndarray) -> float:
    """
    OLS 线性回归斜率 β̂ (最小二乘法)
    ─────────────────────────────────
    对序列 {v_1, ..., v_k} 计算:
      j̄ = (k+1)/2
      v̄ = mean(v)
      β̂ = Σ (j - j̄)(v_j - v̄) / Σ (j - j̄)²

    Parameters:
        values: 长度 k 的效价序列 (v_1 最旧, v_k 最新)
    Returns:
        float: 回归斜率 β̂
    """
    k = len(values)
    if k < 2:
        return 0.0
    # j = [1, 2, ..., k]
    j = np.arange(1, k + 1, dtype=np.float64)
    j_bar = (k + 1) / 2.0
    v_bar = np.mean(values)
    numerator = np.sum((j - j_bar) * (values - v_bar))
    denominator = np.sum((j - j_bar) ** 2)
    if abs(denominator) < EPSILON:
        return 0.0
    return float(numerator / denominator)


def _compute_volatility(values: np.ndarray) -> float:
    """
    §4.4 波动性计算 / Volatility Calculation
    ─────────────────────────────────────────
    volatility = min(1.0, mean(|v_{i+1} - v_i|) / 0.5)

    归一化常数 0.5 对应 valence 全范围 [-1, +1] 中"剧烈波动"的参考值。
    当 k <= 2 时返回 0。

    Parameters:
        values: 效价序列
    Returns:
        float: 波动性 ∈ [0, 1]
    """
    k = len(values)
    if k <= 1:
        return 0.0
    diffs = np.abs(np.diff(values))  # |v_{i+1} - v_i|, 长度 k-1
    mean_diff = float(np.nanmean(diffs))  # nanmean guards against NaN inputs
    if np.isnan(mean_diff):
        return 0.0
    return min(1.0, mean_diff / 0.5)


def _determine_candidate_phase(delta_b: float, beta_hat: float) -> RecoveryPhase:
    """
    §4.4 候选阶段判定函数 / Candidate Phase Mapping
    ─────────────────────────────────────────────────
    按优先级表判定候选阶段:
      1. δ_b < -0.3 且 β̂ ≤ 0     → acute_distress   (情绪显著低于基线且未改善)
      2. δ_b < -0.1 或 (δ_b < 0 且 β̂ > 0) → early_recovery (低于基线但有改善趋势)
      3. |δ_b| ≤ 0.1 且 β̂ ≥ -τ_d → consolidation    (接近基线且趋势稳定/上升)
      4. δ_b > -0.1 且 β̂ ≥ 0     → stable_state     (基线附近或以上)
      5. catch-all                → early_recovery    (安全回退)

    Parameters:
        delta_b: 基线偏差 δ_b(t) = valence(t) - valence_baseline
        beta_hat: OLS 回归斜率
    Returns:
        RecoveryPhase: 候选阶段
    """
    # 优先级 1: acute_distress
    # 条件: δ_b < -0.3 且 β̂ ≤ 0 (情绪显著低于基线且未改善)
    if delta_b < ACUTE_DISTRESS_DELTA_B and beta_hat <= 0:
        return RecoveryPhase.ACUTE_DISTRESS

    # 优先级 2: early_recovery
    # 条件: δ_b < -0.1 或 (δ_b < 0 且 β̂ > 0) (低于基线但有改善趋势)
    if delta_b < EARLY_RECOVERY_DELTA_B or (ACUTE_DISTRESS_DELTA_B <= delta_b < 0 and beta_hat > 0):
        return RecoveryPhase.EARLY_RECOVERY

    # 优先级 3: consolidation
    # 条件: |δ_b| ≤ 0.1 且 β̂ ≥ -τ_d (接近基线且趋势稳定/上升)
    if abs(delta_b) <= CONSOLIDATION_DELTA_B and beta_hat >= -TAU_D:
        return RecoveryPhase.CONSOLIDATION

    # 优先级 4: stable_state
    # 条件: δ_b > -0.1 且 β̂ ≥ 0 (基线附近或以上)
    if delta_b > EARLY_RECOVERY_DELTA_B and beta_hat >= 0:
        return RecoveryPhase.STABLE_STATE

    # 优先级 5: catch-all → early_recovery (安全回退)
    # 涵盖"接近/高于基线但快速下降"等边界情况
    return RecoveryPhase.EARLY_RECOVERY


def _get_boundary_delta_b(candidate: RecoveryPhase, prev_phase: RecoveryPhase) -> float:
    """
    获取当前阶段与候选阶段之间的分界阈值 δ_b^boundary
    ───────────────────────────────────────────────────
    该阈值用于滞后带(hysteresis band)判定:
      仅当 |δ_b(t) - δ_b^boundary| > ε_h 时允许阶段转换。

    分界阈值取候选阶段判定表中对应条件的 δ_b 临界值:
      - acute_distress ↔ early_recovery: boundary = -0.3
      - early_recovery ↔ consolidation: boundary = -0.1
      - consolidation ↔ stable_state:   boundary = -0.1 (取 |δ_b| ≤ 0.1 的边界)

    Parameters:
        candidate: 候选阶段
        prev_phase: 当前(前一)阶段
    Returns:
        float: 分界阈值 δ_b^boundary
    """
    # 根据转换方向确定分界阈值
    phases_ordered = [
        RecoveryPhase.ACUTE_DISTRESS,
        RecoveryPhase.EARLY_RECOVERY,
        RecoveryPhase.CONSOLIDATION,
        RecoveryPhase.STABLE_STATE,
    ]
    # 使用候选阶段与当前阶段之间的最近边界阈值
    pair = frozenset({candidate, prev_phase})

    if RecoveryPhase.ACUTE_DISTRESS in pair and RecoveryPhase.EARLY_RECOVERY in pair:
        return ACUTE_DISTRESS_DELTA_B    # -0.3
    elif RecoveryPhase.EARLY_RECOVERY in pair and RecoveryPhase.CONSOLIDATION in pair:
        return EARLY_RECOVERY_DELTA_B    # -0.1
    elif RecoveryPhase.CONSOLIDATION in pair and RecoveryPhase.STABLE_STATE in pair:
        return EARLY_RECOVERY_DELTA_B    # -0.1 (|δ_b| ≤ 0.1 边界)
    else:
        # 跳跃式转换(如 acute_distress → stable_state), 取最近跨越的边界
        # 使用候选阶段的主要判定阈值
        if candidate == RecoveryPhase.ACUTE_DISTRESS:
            return ACUTE_DISTRESS_DELTA_B    # -0.3
        elif candidate == RecoveryPhase.EARLY_RECOVERY:
            return EARLY_RECOVERY_DELTA_B    # -0.1
        elif candidate == RecoveryPhase.CONSOLIDATION:
            return EARLY_RECOVERY_DELTA_B    # -0.1 (|δ_b| = 0.1 边界)
        else:  # stable_state
            return EARLY_RECOVERY_DELTA_B    # -0.1 (|δ_b| ≤ 0.1 边界, 与 consolidation↔stable_state 一致)


# ═══════════════════════════════════════════════════════════
# 子组件 1: 情绪轨迹分析 (§4.4)
# Emotion Trajectory Analysis
# ═══════════════════════════════════════════════════════════

def analyze_trajectory(
    valence_history: List[float],
    emotion_baseline: EmotionBaseline,
    prev_result: Optional[TrajectoryResult] = None,
) -> TrajectoryResult:
    """
    §4.4 情绪轨迹分析 / Emotion Trajectory Analysis
    ════════════════════════════════════════════════════
    从情感记忆中提取用户近期情绪时间序列，分析趋势和模式。

    算法步骤:
      1. 收集最近 k=5 个会话级效价采样点
      2. 若 k ≤ 2: 冷启动安全模式 (direction=stable, β̂=0, volatility=0)
      3. 若 k ≥ 3: OLS 回归计算斜率 β̂, 判定方向
      4. 计算 δ_b(t) = valence(t) - valence_baseline
      5. 候选阶段判定 (优先级表)
      6. 滞后保护 (hysteresis band + T_hold)
      7. 急性困扰特殊处理 (T_max_acute 强制重估)

    Parameters:
        valence_history: 最近若干个会话级效价值列表 (最旧在前, 最新在后)
        emotion_baseline: 情绪基线对象
        prev_result: 上一轮轨迹分析结果 (用于滞后保护中的 phase_duration 和 current_phase)
    Returns:
        TrajectoryResult: 包含 direction, beta_hat, delta_b, volatility, current_phase, phase_duration
    """
    # ── Step 1: 收集最近 k 个会话级效价值 ──
    # 取最近 TRAJECTORY_WINDOW_K=5 个采样点
    k = min(len(valence_history), TRAJECTORY_WINDOW_K)
    values = np.array(valence_history[-k:], dtype=np.float64) if k > 0 else np.array([], dtype=np.float64)

    # ── Step 2: 冷启动安全模式 ──
    # 当 k ≤ 2 时数据不足以推断趋势，强制 direction=stable, β̂=0, volatility=0
    if k <= 2:
        beta_hat = 0.0
        volatility = 0.0
        direction = TrajectoryDirection.STABLE
    else:
        # ── Step 3: OLS 线性回归计算斜率 β̂ ──
        # β̂ = Σ(j - j̄)(v_j - v̄) / Σ(j - j̄)²
        beta_hat = _ols_slope(values)
        volatility = _compute_volatility(values)

        # 方向判定:
        #   β̂ > τ_d (0.05) → improving (改善中)
        #   β̂ < -τ_d      → declining (恶化中)
        #   否则            → stable (稳定)
        # 额外判定: 高波动性时标记为 volatile
        if beta_hat > TAU_D:
            direction = TrajectoryDirection.IMPROVING
        elif beta_hat < -TAU_D:
            direction = TrajectoryDirection.DECLINING
        else:
            direction = TrajectoryDirection.STABLE

        # 高波动性覆盖: 当 volatility > 0.6 且方向为 stable 时标记为 volatile
        # (volatile 仅在 stable 区间内触发，declining/improving 的明确趋势不被覆盖)
        if volatility > 0.6:
            if direction == TrajectoryDirection.STABLE:
                direction = TrajectoryDirection.VOLATILE
            elif direction in (TrajectoryDirection.IMPROVING, TrajectoryDirection.DECLINING):
                # Log high volatility even when there's a clear trend
                logger.info(f"High volatility ({volatility:.2f}) with {direction.value} trend")

    # ── Step 4: 计算基线偏差 δ_b(t) ──
    # δ_b(t) = valence(t) - valence_baseline
    # 当前效价 = 最新采样点; 基线效价 = σ^T · e_baseline
    current_valence = float(values[-1]) if k > 0 else 0.0
    baseline_valence = float(np.dot(VALENCE_SIGNS, emotion_baseline.normal_state))
    delta_b = current_valence - baseline_valence

    # ── Step 5: 候选阶段判定 (Candidate Phase Mapping) ──
    candidate_phase = _determine_candidate_phase(delta_b, beta_hat)

    # ── Step 6: 滞后保护 (Hysteresis Protection) ──
    # 获取上一轮状态
    if prev_result is not None:
        prev_phase = prev_result.current_phase
        prev_duration = prev_result.phase_duration
    else:
        # 首次调用: 无历史状态，直接采用候选阶段
        prev_phase = candidate_phase
        prev_duration = 0

    # 默认: 维持当前阶段, 持续轮次+1
    current_phase = prev_phase
    phase_duration = prev_duration + 1

    if candidate_phase != prev_phase:
        # ── 阶段转换判定 ──

        # 特殊规则 1: 进入 acute_distress 不受 T_hold 约束 (安全优先)
        # "从任何阶段跳入 acute_distress 不受 T_hold 约束" — 规约 §4.4
        # 例外: 刚从 acute_distress 经 T_max_acute 强制退出后, 新阶段须
        # 满足 T_hold=3 才可重入, 防止"强制退出→立即重入"的振荡
        if candidate_phase == RecoveryPhase.ACUTE_DISTRESS:
            # 判断是否为"T_max_acute 强制退出后立即重入"情况:
            # 使用 TrajectoryResult.forced_exit_from_acute 布尔标记 (由 Step 7 设置)
            is_reentry_after_forced_exit = (
                prev_result is not None
                and prev_result.forced_exit_from_acute
                and prev_duration < T_HOLD
            )
            if is_reentry_after_forced_exit:
                # 刚从 acute_distress 强制退出, 新阶段尚未满足 T_hold
                # 维持当前阶段, 防止振荡
                pass
            else:
                # 非重入场景: 直接进入 acute_distress, 仅需滞后带检查
                # (安全优先, 不受 T_hold 约束)
                boundary = _get_boundary_delta_b(candidate_phase, prev_phase)
                if abs(delta_b - boundary) > EPSILON_H:
                    current_phase = candidate_phase
                    phase_duration = 1

        # 特殊规则 2: 从 acute_distress 退出时使用降低的滞后阈值 ε_h_exit=0.02
        elif prev_phase == RecoveryPhase.ACUTE_DISTRESS:
            # 不受 T_hold 约束(使用 ε_h_exit 替代 ε_h)
            # "从 acute_distress 退出时，采用降低的滞后阈值 ε_{h,exit} = 0.02"
            boundary = _get_boundary_delta_b(candidate_phase, prev_phase)
            if abs(delta_b - boundary) > EPSILON_H_EXIT:
                current_phase = candidate_phase
                phase_duration = 1
            # 否则维持 acute_distress

        else:
            # 一般转换: 需同时满足 T_hold 和 ε_h
            if prev_duration >= T_HOLD:
                boundary = _get_boundary_delta_b(candidate_phase, prev_phase)
                if abs(delta_b - boundary) > EPSILON_H:
                    current_phase = candidate_phase
                    phase_duration = 1
            # 否则维持当前阶段

    # ── Step 7: T_max_acute 强制重估 ──
    # 当 acute_distress 持续超过 T_max_acute=10 轮时:
    #   - 若 candidate_phase != acute_distress → 无条件转入 candidate_phase
    #   - 若 candidate_phase == acute_distress → 计数器重置, 继续维持
    forced_exit = False
    if current_phase == RecoveryPhase.ACUTE_DISTRESS and phase_duration >= T_MAX_ACUTE:
        if candidate_phase != RecoveryPhase.ACUTE_DISTRESS:
            # 强制退出: 转入候选阶段
            current_phase = candidate_phase
            phase_duration = 1  # 新阶段从1开始, 需满足 T_hold=3 才可再入 acute_distress
            forced_exit = True  # 标记: 下一轮检查此标记以抑制立即重入
        else:
            # 用户状态确实持续恶化: 重置计数器, 开启下一个10轮周期
            phase_duration = 1

    return TrajectoryResult(
        direction=direction,
        beta_hat=beta_hat,
        delta_b=delta_b,
        volatility=volatility,
        current_phase=current_phase,
        phase_duration=phase_duration,
        forced_exit_from_acute=forced_exit,
    )


# ═══════════════════════════════════════════════════════════
# 子组件 2: 情境评估 (§4.2)
# Situation Appraisal
# ═══════════════════════════════════════════════════════════

def assess_situation(
    state_vector: StateVector,
    trajectory: TrajectoryResult,
    emotion_baseline: Optional[EmotionBaseline] = None,
) -> SituationAppraisal:
    """
    §4.2 情境评估 / Situation Appraisal
    ════════════════════════════════════
    基于认知评价理论 (Lazarus & Folkman, 1984) 计算综合严重度。

    定义 4.1 (综合严重度评分):
      severity = w1·ι + w2·‖e(t) - e_baseline‖₂/√2 + w3·f_trend + w4·(1 - controllability)

    其中:
      - ι(t): 情绪总体唤醒水平 (intensity)
      - ‖e(t) - e_baseline‖₂/√2: 归一化情绪偏差 (理论最大 L2 距离 = √2)
      - f_trend: 趋势因子, 由 direction 映射
      - controllability: 用户对情境的控制感 (Mock 为 0.5)

    Parameters:
        state_vector: 感知模块输出的状态向量
        trajectory: 轨迹分析结果
        emotion_baseline: 情绪基线 (用于 L2 偏差计算, 无值时回退至 |delta_b|)
    Returns:
        SituationAppraisal: 包含 severity, controllability, f_trend
    """
    # ── 分量 1: 情绪强度 ι(t) ──
    intensity = state_vector.emotion.intensity

    # ── 分量 2: 情绪偏差 ‖e(t) - e_baseline‖₂ / √2 ──
    # √2 为8维单纯形 Δ⁷ 上两点的理论最大L2距离
    # (两个不同的one-hot分布之间的欧氏距离 = √((1-0)² + (0-1)²) = √2)
    e_current = state_vector.emotion.e
    if emotion_baseline is not None:
        # 严格按规约 §4.2 定义 4.1 计算: ‖e(t) - e_baseline‖₂ / √2
        e_base = emotion_baseline.normal_state
        diff_norm = float(np.linalg.norm(e_current - e_base))
        baseline_deviation = min(1.0, diff_norm / np.sqrt(2))
    else:
        # 回退: 无基线时使用 delta_b 标量近似
        baseline_deviation = min(1.0, abs(trajectory.delta_b))

    # ── 分量 3: 趋势因子 f_trend ──
    # f_trend 映射: declining=1.0, volatile=0.75, stable=0.5, improving=0.0
    f_trend = F_TREND_MAP.get(trajectory.direction.value, 0.5)

    # ── 分量 4: 可控性 (controllability) ──
    # 使用 LLM 评估值, 无值时回退至 0.5
    controllability = (
        state_vector.controllability
        if state_vector.controllability is not None
        else 0.5
    )

    # ── 综合严重度 ──
    # 当 LLM 提供 life_impact 时, 使用增强的5分量公式
    # life_impact 捕获 L2 距离无法检测的语义严重度 (失业、健康危机等)
    life_impact = state_vector.life_impact
    if life_impact is not None:
        # 5 components: redistribute weight from baseline_deviation to life_impact
        severity = (
            SEVERITY_W1 * intensity                  # 0.30
            + 0.15 * baseline_deviation              # reduced from 0.25
            + SEVERITY_W3 * f_trend                  # 0.25
            + SEVERITY_W4 * (1.0 - controllability)  # 0.20
            + 0.10 * life_impact                     # new: LLM life impact
        )
    else:
        # Fallback: original 4-component formula (no LLM)
        severity = (
            SEVERITY_W1 * intensity
            + SEVERITY_W2 * baseline_deviation
            + SEVERITY_W3 * f_trend
            + SEVERITY_W4 * (1.0 - controllability)
        )
    # 裁剪到 [0, 1]
    severity = float(np.clip(severity, 0.0, 1.0))

    # Shadow mode safety override: when LLM detects crisis but formula misses,
    # escalate severity to ensure safety. LLM semantic understanding can catch
    # crisis signals (metaphor, indirection) that keyword formulas miss.
    llm_urgency = getattr(state_vector, 'llm_urgency', None)
    if llm_urgency is not None:
        logger.info(
            f"[Shadow] severity: formula={severity:.3f}, llm_urgency={llm_urgency:.3f}"
        )
        if llm_urgency > 0.9 and severity < 0.7:
            logger.warning(
                f"[Shadow Override] LLM urgency ({llm_urgency:.3f}) >> formula severity "
                f"({severity:.3f}). Escalating severity to {llm_urgency:.3f} for safety."
            )
            severity = float(llm_urgency)

    return SituationAppraisal(
        severity=severity,
        controllability=controllability,
        f_trend=f_trend,
    )


# ═══════════════════════════════════════════════════════════
# 子组件 3: 目标推断 (§4.3)
# Goal Inference
# ═══════════════════════════════════════════════════════════

def infer_goals(
    state_vector: StateVector,
    trajectory: TrajectoryResult,
    appraisal: SituationAppraisal,
) -> GoalSet:
    """
    §4.3 目标推断 / Goal Inference
    ══════════════════════════════
    基于恢复阶段(phase)和用户意图(intent)推断多层级目标。

    多层目标架构:
      - immediate:    即时目标 — 缓解当前情绪
      - session:      会话目标 — 帮助用户识别问题根源
      - long_term:    长期目标 — 建立健康应对模式
      - relationship: 关系目标 — 深化信任、促进自我披露 (★ 始终存在)

    关键设计原则 (Social Penetration Theory; Altman & Taylor, 1973):
      关系目标始终存在于每次交互中。关系深化来自持续的、递进的自我披露过程。

    Parameters:
        state_vector: 感知模块输出的状态向量
        trajectory: 轨迹分析结果
        appraisal: 情境评估结果
    Returns:
        GoalSet: 多层级目标集合
    """
    phase = trajectory.current_phase
    top_intent = state_vector.intent.top_intent()

    # ── 即时目标 (immediate goal) ──
    # 根据阶段和意图映射
    immediate_goals_by_phase = {
        RecoveryPhase.ACUTE_DISTRESS: "提供情感安全感，稳定用户当前情绪状态",
        RecoveryPhase.EARLY_RECOVERY: "温和陪伴，支持用户初步情绪恢复",
        RecoveryPhase.CONSOLIDATION: "帮助用户巩固情绪改善成果",
        RecoveryPhase.STABLE_STATE: "维护积极情绪状态，强化用户内在力量",
    }
    # 意图特化覆盖
    if top_intent == "CRISIS":
        immediate = "确保用户安全，提供即时危机支持与专业资源引导"
    elif top_intent == "VENT":
        immediate = "倾听与接纳，为用户提供安全的情绪宣泄空间"
    elif top_intent == "SHARE_JOY":
        immediate = "共同分享与庆祝用户的积极体验"
    else:
        immediate = immediate_goals_by_phase.get(
            phase, "理解并回应用户当前的情感需求"
        )

    # ── 会话目标 (session goal) ──
    session_goals_by_phase = {
        RecoveryPhase.ACUTE_DISTRESS: "帮助用户表达和识别当前困扰的核心情绪",
        RecoveryPhase.EARLY_RECOVERY: "引导用户理解情绪触发因素，建立初步认知框架",
        RecoveryPhase.CONSOLIDATION: "促进用户对情绪模式的反思，识别应对策略",
        RecoveryPhase.STABLE_STATE: "探讨用户的成长与进步，规划未来方向",
    }
    if top_intent == "ADVICE":
        session = "帮助用户梳理问题，共同探索可行的解决方案"
    elif top_intent == "REFLECT":
        session = "引导用户深度反思，发现情绪行为的内在模式"
    else:
        session = session_goals_by_phase.get(
            phase, "帮助用户获得对当前情境的更清晰认知"
        )

    # ── 长期目标 (long-term goal) ──
    long_term_goals_by_phase = {
        RecoveryPhase.ACUTE_DISTRESS: "发展健康的情绪调节能力，建立危机应对机制",
        RecoveryPhase.EARLY_RECOVERY: "培养自我觉察能力，建立可持续的情绪恢复路径",
        RecoveryPhase.CONSOLIDATION: "强化适应性应对策略，提升心理韧性",
        RecoveryPhase.STABLE_STATE: "促进持续的个人成长与心理幸福感",
    }
    long_term = long_term_goals_by_phase.get(
        phase, "支持用户建立健康的情绪应对模式"
    )

    # ── 关系目标 (relationship goal) — ★ 始终存在 ──
    # 基于关系深度 d(t) 调整关系目标的具体内容
    d = state_vector.relationship_depth
    if d < 0.2:
        relationship = "建立基础信任与安全感，营造无压力的对话氛围"
    elif d < 0.5:
        relationship = "适度促进自我披露，深化情感联结与相互理解"
    else:
        relationship = "维护深层信任关系，支持开放性的情感探索与成长"

    return GoalSet(
        immediate_goal=immediate,
        session_goal=session,
        long_term_goal=long_term,
        relationship_goal=relationship,
    )


# ═══════════════════════════════════════════════════════════
# 子组件 4: Thompson 采样策略选择 (§4.5)
# Context-Stratified Thompson Sampling
# ═══════════════════════════════════════════════════════════

def determine_planning_intent(state_vector: StateVector) -> str:
    """
    定义 3.6a: 预规划意图推断 / Pre-Planning Intent Estimation
    ═══════════════════════════════════════════════════════════
    基于 StateVector 中的意图分布和紧急度，通过确定性规则快速推断 planning_intent。
    此映射仅使用感知模块已输出的 IntentDistribution 和 urgency，不依赖任何后续模块输出。

    映射规则 (按优先级排序):
      1. urgency > 0.9              → crisis_response (危机响应)
      2. P(REFLECT) > 0.4 或 P(ADVICE) > 0.4 → pattern_reflection (模式反思)
      3. P(SHARE_JOY) > 0.5         → positive_reframing (正向重构, Φ=0.1)
      4. P(VENT) + P(COMFORT) > 0.6 → emotional_validation (共情验证)
      5. 否则                       → default (默认)

    Parameters:
        state_vector: 感知模块输出的状态向量
    Returns:
        str: planning_intent 字符串
    """
    urgency = state_vector.urgency_level
    intent = state_vector.intent

    # 优先级 1: 危机信号最高优先
    if urgency > CRISIS_THRESHOLD:
        return "crisis_response"

    # 优先级 2: 用户主动寻求深度理解
    if intent.p("REFLECT") > 0.4 or intent.p("ADVICE") > 0.4:
        return "pattern_reflection"

    # 优先级 3: 积极情境下引导正向 (Φ=0.1, 抑制负面情绪一致检索)
    if intent.p("SHARE_JOY") > 0.5:
        return "positive_reframing"

    # 优先级 4: 情感支持需求
    if intent.p("VENT") + intent.p("COMFORT") > 0.6:
        return "emotional_validation"

    # 优先级 5: 标准检索模式
    return "default"


def discretize_context(
    state_vector: StateVector,
    trajectory: TrajectoryResult,
    appraisal: SituationAppraisal,
    adaptive_params: Optional['AdaptiveParams'] = None,
) -> ContextVector:
    """
    定义 4.2: 情境向量离散化 / Context Discretization
    ══════════════════════════════════════════════════
    将连续状态映射到有限类别 (4×3×3=36 类) 以避免数据稀疏。

    离散化维度:
      - phase:          4 个恢复阶段 (直接使用 trajectory.current_phase)
      - severity_level: 3 个级别 (low < 0.3, 0.3 ≤ medium < 0.7, high ≥ 0.7)
      - intent_group:   3 个意图组:
          * exploration: ADVICE + REFLECT (用户主动探索)
          * insight:     VENT + COMFORT + CRISIS (需要情感支持 → 对应被动探索策略)
            [注: 此处 "insight" 命名沿用 config.py 的 intent_group 定义,
             实际含义为 "support_seeking"]
          * neutral:     SHARE_JOY + CHAT (分享与闲聊)

    Parameters:
        state_vector: 感知模块输出的状态向量
        trajectory: 轨迹分析结果
        appraisal: 情境评估结果
    Returns:
        ContextVector: 离散化的情境向量
    """
    # ── phase: 直接使用轨迹分析结果 ──
    phase = trajectory.current_phase

    # ── severity_level: 严重度分箱 ──
    # Use adaptive thresholds when available (user-specific quantiles)
    # Fallback: static [0.3, 0.7] when no adaptive params
    if adaptive_params is not None:
        low_th = adaptive_params.severity_low_threshold
        high_th = adaptive_params.severity_high_threshold
    else:
        low_th = 0.3
        high_th = 0.7
    severity = appraisal.severity
    if severity < low_th:
        severity_level = "low"
    elif severity < high_th:
        severity_level = "medium"
    else:
        severity_level = "high"

    # ── intent_group: 意图分组 ──
    # support_seeking (VENT+COMFORT+CRISIS) → "insight" (config naming)
    # sharing (SHARE_JOY+CHAT) → "neutral"
    # exploration (ADVICE+REFLECT) → "exploration"
    top_intent = state_vector.intent.top_intent()
    support_intents = {"VENT", "COMFORT", "CRISIS"}
    sharing_intents = {"SHARE_JOY", "CHAT"}
    exploration_intents = {"ADVICE", "REFLECT"}

    if top_intent in support_intents:
        intent_group = "insight"
    elif top_intent in exploration_intents:
        intent_group = "exploration"
    else:
        intent_group = "neutral"

    return ContextVector(
        phase=phase,
        severity_level=severity_level,
        intent_group=intent_group,
        turn_in_session=0,  # 由调用方填充
        relationship_depth=state_vector.relationship_depth,
    )


# Strategy-specific cold-start priors (§4.5.2)
# Source: Hill (2009) Helping Skills model — general effectiveness rankings
# across exploration/insight/action phases, calibrated to [0.4, 0.7] range.
STRATEGY_COLD_START_MU = {
    # Exploration phase skills — broadly effective across phases (Hill 2009, Ch.5)
    "active_listening": 0.65,        # Highest: universally safe, core attending skill
    "emotional_validation": 0.60,    # High: affirms experience, builds rapport (Rogers, 1957)
    "empathic_reflection": 0.60,     # High: deepens emotional awareness (Elliott et al., 2004)
    "companionable_silence": 0.55,   # Moderate-high: effective for withdrawn/overwhelmed users
    # Insight phase skills — require timing and user readiness
    "gentle_guidance": 0.50,         # Moderate: bridges exploration→insight, context-dependent
    "cognitive_reframing": 0.45,     # Below-average: premature use risks invalidation (Hill, 2009 Ch.9)
    "strength_recognition": 0.50,    # Moderate: timing-sensitive, effective in consolidation (§4.5.2)
    # Action phase skills — need established trust and readiness
    "problem_solving": 0.40,         # Lowest non-crisis: requires user readiness (Hill 2009, Ch.13)
    "information_providing": 0.45,   # Below-average: useful only when user seeks it explicitly
    # Safety protocol — highest prior by design
    "CRISIS_PROTOCOL": 0.70,         # Safety-first: must win TS when crisis signals present
}


def _compute_cross_context_mean(
    strategy: str,
    affective_memory: AffectiveMemory,
    now: datetime,
) -> Optional[float]:
    """
    §4.5.2 层次贝叶斯先验: 计算策略 s_i 的跨情境平均有效率 μ_si
    ═══════════════════════════════════════════════════════════════
    μ_si = 衰减后所有已观测情境中该策略 Beta 分布均值的平均

    对每个 (s_i, c_k) 对:
      1. 先按 Algorithm 4.1 Step 1 的时间衰减公式调整参数
      2. 计算衰减后的 Beta 均值 α'/(α'+β')
      3. 取所有已观测情境的均值

    Parameters:
        strategy: 策略名称
        affective_memory: 情感记忆
        now: 当前时间
    Returns:
        Optional[float]: 跨情境平均有效率; 无任何观测时返回 None
    """
    means = []
    for (s, c), params in affective_memory.strategy_matrix.items():
        if s != strategy or params.observation_count == 0:
            continue
        # 时间衰减: γ_β = exp(-ln2 · Δt / τ_β)
        delta_days = max(0.0, (now - params.last_updated).total_seconds() / 86400.0)
        gamma_beta = math.exp(-math.log(2) * delta_days / BETA_DECAY_HALFLIFE_DAYS)
        # 衰减后参数
        alpha_decayed = BETA_ALPHA0 + gamma_beta * (params.alpha - BETA_ALPHA0)
        beta_decayed = BETA_BETA0 + gamma_beta * (params.beta - BETA_BETA0)
        # Beta 均值
        denom = alpha_decayed + beta_decayed
        if denom < 1e-10:
            continue  # skip degenerate context
        mean_val = alpha_decayed / denom
        means.append(mean_val)

    if not means:
        return None
    return float(np.mean(means))


def select_strategy(
    context_vector: ContextVector,
    affective_memory: AffectiveMemory,
    state_vector: StateVector,
    trajectory: TrajectoryResult,
    adaptive_params: Optional['AdaptiveParams'] = None,
) -> str:
    """
    算法 4.1: 上下文 Thompson 采样策略选择 ★★★ 核心创新三
    ══════════════════════════════════════════════════════════
    Context-Stratified Thompson Sampling with Affective Memory Prior

    将心理咨询理论的策略分类与贝叶斯在线学习结合，实现个性化支持。

    Steps:
      1.   加载 Beta 先验 + 时间衰减
      1.5  层次贝叶斯先验混合 (Hierarchical Prior Blending)
      2.   Thompson 采样: θ_i ~ Beta(α_i, β_i)
      3.   选择: selected = argmax(θ)
      4.   安全约束: urgency > 0.9 → CRISIS_PROTOCOL
      4.5  Hill 阶段匹配约束 (phase constraint)
      4.6  低确定度覆盖 (low confidence override)
      5.   关系深度软约束 (relationship depth soft constraint)

    Parameters:
        context_vector: 离散化情境向量
        affective_memory: 情感记忆 (包含策略有效性矩阵)
        state_vector: 感知模块状态向量
        trajectory: 轨迹分析结果 (用于 phase constraint)
    Returns:
        str: 选中的策略名称
    """
    now = datetime.now()
    context_key = context_vector.to_key()
    phase = trajectory.current_phase

    # ═══════════════════════════════════════════
    # Step 1 & 1.5: 遍历所有策略, 计算衰减后 Beta 参数
    # ═══════════════════════════════════════════
    sampled_values: Dict[str, float] = {}

    for strategy in STRATEGIES:
        key = (strategy, context_key)

        # ── Step 1: 加载先验参数 + 时间衰减 ──
        if key in affective_memory.strategy_matrix:
            params = affective_memory.strategy_matrix[key]
            # 时间衰减: γ_β = exp(-ln2 · Δt / τ_β), τ_β = 60 天
            # 参数向先验指数回归, 防止长期累积导致策略冻结
            delta_days = max(0.0, (now - params.last_updated).total_seconds() / 86400.0)
            gamma_beta = math.exp(-math.log(2) * delta_days / BETA_DECAY_HALFLIFE_DAYS)
            # α' = α₀ + γ_β · (α - α₀)  (衰减后不低于先验)
            # β' = β₀ + γ_β · (β - β₀)
            alpha = BETA_ALPHA0 + gamma_beta * (params.alpha - BETA_ALPHA0)
            beta = BETA_BETA0 + gamma_beta * (params.beta - BETA_BETA0)
            n_obs = params.observation_count
        else:
            # 无历史数据: 弱先验 Beta(2, 2)
            alpha = BETA_ALPHA0
            beta = BETA_BETA0
            n_obs = 0

        # ── Step 1.5: 层次贝叶斯先验混合 (§4.5.2) ──
        # 当 N_obs < N_blend=8 时, 利用跨情境信息缓解冷启动
        if n_obs < N_BLEND:
            # μ_si = 策略 s_i 在所有已观测情境上的衰减后平均有效率
            mu_si = _compute_cross_context_mean(strategy, affective_memory, now)
            if mu_si is None:
                # Use adaptive prior if available, else static cold-start
                if (adaptive_params is not None
                        and adaptive_params.cold_start_mu
                        and strategy in adaptive_params.cold_start_mu):
                    mu_si = adaptive_params.cold_start_mu[strategy]
                else:
                    mu_si = STRATEGY_COLD_START_MU.get(strategy, 0.5)

            # 映射为固定强度伪先验: α_prior = N_prior · μ_si, β_prior = N_prior · (1-μ_si)
            # N_prior = 4 (与初始先验强度 α₀+β₀=4 一致)
            alpha_prior = N_PRIOR * mu_si
            beta_prior = N_PRIOR * (1.0 - mu_si)

            # 渐进混合权重: λ_h = max(0, 1 - N_obs / N_blend)
            # N_obs=0 时 λ_h=1 (完全继承先验), N_obs≥N_blend 时 λ_h=0 (完全使用观测)
            lambda_h = max(0.0, 1.0 - n_obs / N_BLEND)

            # 混合: α_blend = λ_h·α_prior + (1-λ_h)·α_obs
            alpha = lambda_h * alpha_prior + (1.0 - lambda_h) * alpha
            beta = lambda_h * beta_prior + (1.0 - lambda_h) * beta

        # ── Step 2: Thompson 采样 ──
        # θ_i ~ Beta(α_i, β_i): 从衰减后Beta分布随机采样
        # 确保参数合法 (> 0)
        alpha = max(alpha, EPSILON)
        beta = max(beta, EPSILON)
        theta_i = float(np.random.beta(alpha, beta))
        sampled_values[strategy] = theta_i

    # ── Step 3: 选择采样值最高的策略 ──
    # selected = argmax_i(θ_i)
    if not sampled_values:
        selected = "active_listening"  # Safe default
    else:
        selected = max(sampled_values, key=sampled_values.get)  # type: ignore[arg-type]

    # ═══════════════════════════════════════════
    # Step 4–5: 安全约束与策略覆盖
    # ═══════════════════════════════════════════

    # ── Step 4: 安全约束覆盖 ──
    # 防御性冗余设计 (defense-in-depth):
    # 正常流程下 urgency>0.9 在 Phase 1/6.2 已触发危机快速通道(绕过Planning)
    # 此检查作为备份安全层, 防止上游模块在边缘情况下漏检
    if state_vector.urgency_level > CRISIS_THRESHOLD:
        logger.info("[StrategyOverride:crisis] Urgency %.2f > %.2f — forcing CRISIS_PROTOCOL",
                    state_vector.urgency_level, CRISIS_THRESHOLD)
        return "CRISIS_PROTOCOL"  # 危机协议: 绕过 Beta 更新, 避免污染策略学习

    # ── Step 4.5: Hill 阶段匹配约束 ──
    # 防止在不适当的恢复阶段使用过于"前进"的策略
    # 理论依据: Hill (2009) 三阶段模型 — 探索阶段先于洞察/行动阶段
    if phase == RecoveryPhase.ACUTE_DISTRESS:
        # 急性困扰期: 仅允许被动Exploration策略(s1,s2,s3) + 陪伴沉默(s9)
        # gentle_guidance虽归属Exploration, 但引导提问属认知工作, 急性期不适用
        if selected not in ACUTE_ALLOWED_STRATEGIES:
            logger.info("[StrategyOverride:phase] Acute distress constraint: %s -> active_listening", selected)
            # Fallback: active_listening is the safest attending skill — pure listening
            # without probing or reframing, which could overwhelm in acute phase
            # (Hill 2009, Ch.5: attending skills as foundation)
            selected = "active_listening"
    elif phase == RecoveryPhase.EARLY_RECOVERY:
        # 初步恢复期: 允许Exploration + 温和Insight, 禁止Action阶段策略
        if selected in {"problem_solving", "positive_reinforcement"}:
            logger.info("[StrategyOverride:phase] Early recovery constraint: %s -> gentle_guidance", selected)
            # Fallback: gentle_guidance chosen over problem_solving because user is
            # not yet ready for directive strategies (Hill 2009, exploration phase).
            # Chosen over active_listening because early recovery permits mild guidance.
            selected = "gentle_guidance"
    # consolidation 和 stable_state 阶段无策略限制

    # ── Step 4.5b: 强负面情绪禁止积极强化 ──
    # 当 valence < POSITIVE_REINFORCEMENT_VALENCE_BLOCK 时,
    # positive_reinforcement 不适合 — 避免在用户深度负面情绪时给出正面鼓励
    current_valence = state_vector.emotion.valence()
    if current_valence < POSITIVE_REINFORCEMENT_VALENCE_BLOCK:
        if selected == "positive_reinforcement":
            logger.info("[StrategyOverride:valence] Valence %.2f < %.2f, blocking positive_reinforcement -> empathic_reflection",
                        current_valence, POSITIVE_REINFORCEMENT_VALENCE_BLOCK)
            # Fallback: empathic_reflection chosen because when valence is strongly negative,
            # the user needs their feelings acknowledged before any positive framing.
            # empathic_reflection mirrors emotional content without redirecting (Elliott et al., 2004).
            selected = "empathic_reflection"

    # ── Step 4.6: LLM 语义信号引导 (替代硬编码低确定度覆盖) ──
    # 使用 LLM 的 recommended_approach 语义信号约束 TS 选择
    # 保守设计: 仅在 LLM 置信度足够高 且 TS 选择落在推荐集之外时干预
    # 防止约束级联: 如果 LLM 已覆盖, 跳过后续关系/开放度约束
    llm_overrode = False
    recommended = state_vector.recommended_approach
    if recommended and recommended in APPROACH_TO_STRATEGIES:
        approach_set = APPROACH_TO_STRATEGIES[recommended]
        if selected not in approach_set:
            # 仅在 LLM 置信度足够时覆盖 TS 选择
            # llm_confidence is now a proper field on StateVector
            confidence = state_vector.llm_confidence if state_vector.llm_confidence is not None else 1.0
            if confidence > LLM_OVERRIDE_CONFIDENCE_THRESHOLD:
                best_in_set = max(
                    approach_set,
                    key=lambda s: sampled_values.get(s, 0.0),
                )
                logger.info(
                    "[StrategyOverride:llm] approach '%s': %s -> %s "
                    "(TS value: %.3f, confidence: %.2f)",
                    recommended, selected, best_in_set,
                    sampled_values.get(best_in_set, 0), confidence,
                )
                selected = best_in_set
                llm_overrode = True

    # ── Step 5: 关系深度约束 ──
    # Deterministic rollback for very low relationship depth.
    # Rationale: with d < RELATIONSHIP_DEPTH_THRESHOLD, advanced strategies
    # (cognitive_reframing, problem_solving) require established trust
    # (Altman & Taylor, 1973 — Social Penetration Theory).
    # If trust is insufficient, the rollback should always apply.
    # 防止级联: 如果 LLM 已在 Step 4.6 覆盖, 跳过关系深度约束
    if (not llm_overrode
            and context_vector.relationship_depth < RELATIONSHIP_DEPTH_THRESHOLD
            and selected in ("cognitive_reframing", "problem_solving")):
        logger.info("[StrategyOverride:depth] Low relationship depth (%.2f < %.2f) — "
                    "rollback %s -> empathic_reflection",
                    context_vector.relationship_depth, RELATIONSHIP_DEPTH_THRESHOLD, selected)
        # Fallback: empathic_reflection builds rapport through mirroring, which is
        # a prerequisite before attempting cognitive or problem-solving work
        # with a user who hasn't yet developed trust (Hill 2009, Ch.5-6).
        selected = "empathic_reflection"

    # ── Step 6: 沟通开放度引导 (始终生效, 用户状态优先于 LLM 推荐) ──
    # communication_openness 反映用户当前沟通意愿, 属于用户状态而非推荐,
    # 应始终尊重, 不受 LLM 覆盖级联影响
    openness = state_vector.communication_openness
    if openness is not None and openness < LLM_OPENNESS_SILENCE_THRESHOLD:
        logger.info("[StrategyOverride:openness] Openness %.2f < %.2f — override %s -> companionable_silence",
                    openness, LLM_OPENNESS_SILENCE_THRESHOLD, selected)
        selected = "companionable_silence"

    return selected


# ═══════════════════════════════════════════════════════════
# 主类: PlanningModule
# ═══════════════════════════════════════════════════════════

class PlanningModule:
    """
    §4 规划模块 / Planning Module
    ═════════════════════════════
    基于感知模块的 StateVector 和记忆模块的检索结果，制定结构化响应计划。
    规划模块不生成文字，只生成决策方案。

    执行流程:
      (1) 情绪轨迹分析 (§4.4) — 分析用户情绪趋势和恢复阶段
      (2) 情境评估 (§4.2)     — 综合评估当前情境的严重度
      (3) 目标推断 (§4.3)     — 制定多层级支持目标
      (4) Thompson 采样 (§4.5) — 个性化策略选择

    将规划独立为模块的学术理由:
      可解释性和可审计性。在情感陪伴的伦理敏感场景中 (Kirk et al., 2025),
      能够审查"Agent为什么选择了这个策略"比最终回复更重要。
      独立的规划模块使每个决策点可追溯、可干预。

    Attributes:
        _prev_trajectory: 上一轮轨迹分析结果 (用于滞后保护)
    """

    def __init__(self) -> None:
        # 持久化上一轮轨迹分析结果, 用于阶段滞后保护 (hysteresis)
        self._prev_trajectory: Optional[TrajectoryResult] = None

    def reset_session(self) -> None:
        """
        重置会话状态 — 清除轨迹缓存, 防止跨会话滞后状态污染。
        Reset session state — clear trajectory cache to prevent stale
        hysteresis across sessions.
        """
        self._prev_trajectory = None

    def plan(
        self,
        state_vector: StateVector,
        retrieved_context: RetrievedContext,
        affective_memory: AffectiveMemory,
        prev_trajectory: Optional[TrajectoryResult] = None,
        adaptive_params: Optional['AdaptiveParams'] = None,
        use_thompson_sampling: bool = True,
    ) -> PlanningOutput:
        """
        主入口: 执行完整的规划流程
        ═══════════════════════════
        Parameters:
            state_vector: 感知模块输出的状态向量 (§2.7)
            retrieved_context: MAR 检索返回的上下文 (§3.7)
            affective_memory: Tier 4 情感记忆 (§3.5)
            prev_trajectory: 外部传入的上一轮轨迹结果 (可选; 若为 None 则使用内部缓存)
            adaptive_params: 自适应参数覆盖 (§6.3, 可选; 若为 None 则使用静态默认值)
            use_thompson_sampling: 是否启用 Thompson Sampling (默认 True; False 时退回规则策略)
        Returns:
            PlanningOutput: 规划模块完整输出 (§4.7)
        """
        # 使用外部传入的 prev_trajectory 或内部缓存
        trajectory_prev = prev_trajectory if prev_trajectory is not None else self._prev_trajectory

        # ── Step 0: 预规划意图推断 (定义 3.6a) ──
        # 确定性规则映射, O(1) 复杂度, 不依赖后续模块
        planning_intent = determine_planning_intent(state_vector)

        # Shadow mode: compare formula planning_intent with LLM planning_intent
        llm_planning_intent = getattr(state_vector, 'planning_intent', None)
        if llm_planning_intent is not None:
            logger.info(
                f"[Shadow] planning_intent: formula={planning_intent}, llm={llm_planning_intent}, "
                f"match={planning_intent == llm_planning_intent}"
            )

        # ── Step 1: 情绪轨迹分析 (§4.4) ──
        # 从情感记忆中提取会话级效价历史
        valence_history = self._extract_valence_history(
            affective_memory, retrieved_context
        )
        trajectory = analyze_trajectory(
            valence_history=valence_history,
            emotion_baseline=affective_memory.emotion_baseline,
            prev_result=trajectory_prev,
        )
        # 缓存本轮轨迹结果, 供下一轮使用
        self._prev_trajectory = trajectory

        # Shadow mode: compare formula phase with LLM phase
        llm_phase = getattr(state_vector, 'recovery_phase', None)
        if llm_phase is not None:
            logger.info(
                f"[Shadow] phase: formula={trajectory.current_phase}, llm={llm_phase}, "
                f"match={trajectory.current_phase.value == llm_phase}"
            )

        # ── Step 2: 情境评估 (§4.2) ──
        appraisal = assess_situation(
            state_vector, trajectory,
            emotion_baseline=affective_memory.emotion_baseline,
        )

        # ── Step 3: 目标推断 (§4.3) ──
        goals = infer_goals(state_vector, trajectory, appraisal)

        # ── Step 4: 情境离散化 + 策略选择 (§4.5) ──
        context_vector = discretize_context(
            state_vector, trajectory, appraisal,
            adaptive_params=adaptive_params,
        )
        if use_thompson_sampling:
            selected_strategy = select_strategy(
                context_vector=context_vector,
                affective_memory=affective_memory,
                state_vector=state_vector,
                trajectory=trajectory,
                adaptive_params=adaptive_params,
            )
        else:
            # 消融基线: 规则映射策略选择
            selected_strategy = self._rule_based_strategy(state_vector, planning_intent, trajectory)

        return PlanningOutput(
            trajectory=trajectory,
            appraisal=appraisal,
            goals=goals,
            selected_strategy=selected_strategy,
            context_vector=context_vector,
            planning_intent=planning_intent,
        )

    def _rule_based_strategy(
        self,
        state_vector: StateVector,
        planning_intent: str,
        trajectory: TrajectoryResult,
    ) -> str:
        """
        规则映射策略选择 (消融基线).
        Deterministic intent-to-strategy mapping without Thompson Sampling.
        Phase constraints are enforced to maintain safety invariants (Hill, 2009).
        """
        intent = state_vector.intent.top_intent()
        INTENT_STRATEGY_MAP = {
            "CRISIS": "CRISIS_PROTOCOL",
            "VENT": "active_listening",
            "COMFORT": "empathic_reflection",
            "ADVICE": "gentle_guidance",
            "REFLECT": "cognitive_reframing",
            "SHARE_JOY": "positive_reinforcement",
            "CHAT": "emotional_validation",
        }
        selected = INTENT_STRATEGY_MAP.get(intent, "active_listening")

        # Phase constraint enforcement (mirrors select_strategy §4.5 constraints)
        phase = trajectory.current_phase
        if phase == RecoveryPhase.ACUTE_DISTRESS:
            if selected not in ACUTE_ALLOWED_STRATEGIES:
                logger.info("[RuleStrategy:phase] Acute constraint: %s -> active_listening", selected)
                selected = "active_listening"
        elif phase == RecoveryPhase.EARLY_RECOVERY:
            if selected in {"problem_solving", "positive_reinforcement"}:
                logger.info("[RuleStrategy:phase] Early recovery constraint: %s -> gentle_guidance", selected)
                selected = "gentle_guidance"

        return selected

    def _extract_valence_history(
        self,
        affective_memory: AffectiveMemory,
        retrieved_context: RetrievedContext,
    ) -> List[float]:
        """
        从记忆系统中提取会话级效价历史
        ─────────────────────────────────
        优先使用情感记忆的环形缓冲区 (ring_buffer) 中存储的情绪快照;
        若不可用, 则从检索上下文中的 EpisodeUnit 提取 emotional_valence。

        Parameters:
            affective_memory: Tier 4 情感记忆
            retrieved_context: MAR 检索结果
        Returns:
            List[float]: 效价历史 (最旧在前, 最新在后)
        """
        # 方案 1: 从环形缓冲区提取 (情绪基线的 ring_buffer 存储了最近 K_buf=20 次情绪快照)
        ring = affective_memory.emotion_baseline.ring_buffer
        if len(ring) > 0:
            # ring_buffer 中存储的是 EmotionVector 或 np.ndarray
            # 计算每个快照的 valence
            history = []
            for item in ring:
                if isinstance(item, EmotionVector):
                    history.append(item.valence())
                elif isinstance(item, np.ndarray):
                    history.append(float(np.dot(VALENCE_SIGNS, item)))
                else:
                    # 兜底: 假设是标量
                    history.append(float(item))
            return history

        # 方案 2: 从检索到的 EpisodeUnit 提取
        if retrieved_context.episodes:
            episodes = sorted(retrieved_context.episodes, key=lambda ep: ep.timestamp)
            return [ep.emotional_valence for ep in episodes]

        # 方案 3: 无历史数据
        return []
