"""
EmoMem 数据模型 — 所有核心数据结构
=====================================
来源: emomem.md 定义 2.1–3.7
"""

from __future__ import annotations
import uuid
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
from collections import deque

from .config import (
    NUM_EMOTIONS, NUM_INTENTS, EMOTIONS, INTENT_TYPES,
    BASELINE_COLD_START, BASELINE_CONFIDENCE_INIT, BASELINE_SIGMA_INIT,
    BETA_ALPHA0, BETA_BETA0, BEHAVIOR_COLD_START_DEFAULTS, BEHAVIOR_SIGMA_MIN,
    BEHAVIOR_SIGMA_INIT_MULTIPLIER,
    RING_BUFFER_SIZE, STRATEGIES, NUM_STRATEGIES, RECOVERY_PHASES,
)


# ──────────────────────────────────────────────
# 枚举类型
# ──────────────────────────────────────────────

class IntentType(Enum):
    """§2.4 七种情感陪伴意图"""
    VENT = "VENT"              # 倾诉宣泄
    COMFORT = "COMFORT"        # 寻求安慰
    ADVICE = "ADVICE"          # 寻求建议
    SHARE_JOY = "SHARE_JOY"    # 分享喜悦
    CHAT = "CHAT"              # 陪伴闲聊
    REFLECT = "REFLECT"        # 深度自省
    CRISIS = "CRISIS"          # 危机信号


class RecoveryPhase(Enum):
    """§4.4 / §3.5.3 情绪恢复阶段"""
    ACUTE_DISTRESS = "acute_distress"      # 谷底期
    EARLY_RECOVERY = "early_recovery"      # 恢复上升期
    CONSOLIDATION = "consolidation"        # 巩固期
    STABLE_STATE = "stable_state"          # 稳定期


class ResolutionStatus(Enum):
    """情景记忆事件解决状态"""
    UNRESOLVED = "unresolved"
    IMPROVING = "improving"
    RESOLVED = "resolved"


class TrajectoryDirection(Enum):
    """§4.4 轨迹方向"""
    DECLINING = "declining"
    VOLATILE = "volatile"
    STABLE = "stable"
    IMPROVING = "improving"



# ──────────────────────────────────────────────
# 定义 2.2: 情绪向量
# ──────────────────────────────────────────────

@dataclass
class EmotionVector:
    """
    定义 2.2: EmotionVector(t) = ⟨e(t), ι(t), κ(t)⟩
    - e: 8维归一化情绪分布 (Plutchik, 位于 Δ⁷ 单纯形上)
    - intensity: 情绪总体唤醒水平 ι ∈ [0, 1]
    - confidence: 识别确定度 κ ∈ [0, 1]
    """
    e: np.ndarray = field(default_factory=lambda: np.ones(NUM_EMOTIONS) / NUM_EMOTIONS)
    intensity: float = 0.0    # ι(t)
    confidence: float = 0.5   # κ(t)

    def valence(self) -> float:
        """
        定义 2.3: valence(t) = Σ σ_i · e_i(t)
        σ = [+1, +1, -1, 0, -1, -1, -1, +1]
        返回值 ∈ [-1, +1]
        """
        from .config import VALENCE_SIGNS
        return float(np.dot(VALENCE_SIGNS, self.e))

    def copy(self) -> EmotionVector:
        return EmotionVector(e=self.e.copy(), intensity=self.intensity, confidence=self.confidence)


# ──────────────────────────────────────────────
# §2.2 原始信号
# ──────────────────────────────────────────────

@dataclass
class PunctuationFeatures:
    ellipsis_count: int = 0
    exclamation_count: int = 0
    question_marks: int = 0


@dataclass
class ParalinguisticFeatures:
    emoji_pattern: List[str] = field(default_factory=list)
    punctuation_features: PunctuationFeatures = field(default_factory=PunctuationFeatures)
    message_segments: int = 1
    character_count: int = 0


@dataclass
class TemporalFeatures:
    timestamp: datetime = field(default_factory=datetime.now)
    since_last_message: Optional[timedelta] = None
    session_duration: timedelta = field(default_factory=lambda: timedelta(0))


@dataclass
class RawSignal:
    """§2.2 多模态输入解析器输出"""
    text_content: str = ""
    paralinguistic: ParalinguisticFeatures = field(default_factory=ParalinguisticFeatures)
    temporal: TemporalFeatures = field(default_factory=TemporalFeatures)


# ──────────────────────────────────────────────
# §2.4 意图分布
# ──────────────────────────────────────────────

@dataclass
class IntentDistribution:
    """7类意图的概率分布 (非互斥)"""
    probabilities: Dict[str, float] = field(
        default_factory=lambda: {it: 0.0 for it in INTENT_TYPES}
    )

    def top_intent(self) -> str:
        """返回概率最高的意图"""
        if not self.probabilities:
            return "CHAT"
        return max(self.probabilities, key=self.probabilities.get)

    def p(self, intent: str) -> float:
        """获取指定意图的概率"""
        return self.probabilities.get(intent, 0.0)


# ──────────────────────────────────────────────
# §2.5 隐式信号
# ──────────────────────────────────────────────

@dataclass
class ImplicitSignals:
    bas_score: float = 0.0                          # 行为异常度评分
    anomaly_dimensions: List[str] = field(default_factory=list)
    temporal_context: str = "normal_hours"           # "late_night" | "normal_hours"


# ──────────────────────────────────────────────
# §2.7 状态向量
# ──────────────────────────────────────────────

@dataclass
class StateVector:
    """
    定义 2.6: 状态向量 — 感知模块最终输出
    融合六个子组件的输出
    """
    emotion: EmotionVector = field(default_factory=EmotionVector)
    intent: IntentDistribution = field(default_factory=IntentDistribution)
    implicit_signals: ImplicitSignals = field(default_factory=ImplicitSignals)
    topic_keywords: List[str] = field(default_factory=list)
    topic_category: str = ""
    urgency_level: float = 0.0            # ∈ [0, 1]
    relationship_depth: float = 0.0       # d(t) ∈ [0, 1]
    raw_signal: Optional[RawSignal] = None  # 保留原始输入
    # §2.7 LLM 上下文评估结果 (Context Assessment from LLM)
    # 当 LLM 可用时由感知模块填充, 否则为 None (降级为关键词匹配)
    controllability: Optional[float] = None     # 可控性 [0, 1]
    life_impact: Optional[float] = None         # 生活影响 [0, 1]
    explicit_feedback: Optional[float] = None   # 显式反馈 [0, 1]
    # LLM 语义信号
    communication_openness: Optional[float] = None   # 沟通开放度 [0, 1]
    recommended_approach: Optional[str] = None       # LLM 建议的支持方式
    # LLM 分析置信度 (自适应参数框架)
    llm_confidence: Optional[float] = None           # LLM 自报告分析置信度 [0, 1]
    # LLM-Primary fields (shadow mode: LLM-assessed, compared against formula-computed values)
    llm_urgency: Optional[float] = None              # LLM-assessed urgency [0, 1], shadow of urgency_level
    recovery_phase: Optional[str] = None             # LLM-assessed recovery phase (one of RECOVERY_PHASES)
    planning_intent: Optional[str] = None            # LLM-assessed planning intent


# ──────────────────────────────────────────────
# §3.1 情景记忆单元
# ──────────────────────────────────────────────

@dataclass
class EpisodeUnit:
    """
    定义 3.1: 情景记忆单元
    每次交互的完整记录
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    event: str = ""                                    # 事件描述
    emotion_snapshot: EmotionVector = field(default_factory=EmotionVector)
    people_involved: List[str] = field(default_factory=list)
    topic_category: str = ""
    user_coping_style: str = ""
    agent_strategy: str = ""                           # Agent使用的策略
    user_feedback_score: Optional[float] = None        # ★ 延迟一轮填充
    resolution_status: ResolutionStatus = ResolutionStatus.UNRESOLVED
    emotional_valence: float = 0.0                     # ∈ [-1, 1]
    importance: float = 0.0                            # ∈ [0, 1]
    mention_count: int = 0                             # 用户提及次数
    similar_event_count: int = 0                       # 相似事件次数


# ──────────────────────────────────────────────
# §3.5.1 情绪基线
# ──────────────────────────────────────────────

@dataclass
class EmotionBaseline:
    """
    §3.5.1 情绪基线 — Tier 4 子结构
    记录用户的"正常"情绪分布
    """
    normal_state: np.ndarray = field(
        default_factory=lambda: BASELINE_COLD_START.copy()
    )
    confidence: float = BASELINE_CONFIDENCE_INIT      # 初始 0.2 (低置信)
    sigma_baseline: float = BASELINE_SIGMA_INIT        # σ_baseline 初始 0.3
    last_updated: datetime = field(default_factory=datetime.now)
    sample_count: int = 0
    # 环形缓冲区: 存储最近 K_buf=20 次情绪向量, 用于重估
    ring_buffer: deque = field(
        default_factory=lambda: deque(maxlen=RING_BUFFER_SIZE)
    )


# ──────────────────────────────────────────────
# §3.5.2 触发-情绪关联图
# ──────────────────────────────────────────────

@dataclass
class EmotionPattern:
    """触发事件的典型情绪模式"""
    triggered_emotions: np.ndarray = field(
        default_factory=lambda: np.ones(NUM_EMOTIONS) / NUM_EMOTIONS
    )
    intensity_range: Tuple[float, float] = (0.0, 1.0)
    observation_count: int = 0
    confidence: float = 0.0


# ──────────────────────────────────────────────
# §3.5.3 情绪恢复曲线
# ──────────────────────────────────────────────

@dataclass
class RecoveryPattern:
    """情绪恢复模式"""
    emotion_type: str = ""
    avg_recovery_duration: timedelta = field(default_factory=lambda: timedelta(days=3))
    effective_coping: List[str] = field(default_factory=list)
    effective_agent_aids: List[str] = field(default_factory=list)
    recovery_trajectory: str = "linear"  # "linear" | "oscillating" | "plateau_then_resolution"


# ──────────────────────────────────────────────
# §3.5.4 策略有效性矩阵
# ──────────────────────────────────────────────

@dataclass
class BetaParams:
    """
    定义 3.3: 单个 (strategy, context) 对的 Beta 分布参数
    初始先验 Beta(2, 2)
    """
    alpha: float = BETA_ALPHA0
    beta: float = BETA_BETA0
    last_updated: datetime = field(default_factory=datetime.now)
    observation_count: int = 0

    def mean(self) -> float:
        """Beta分布均值 = α / (α + β), 防御负值和除零"""
        a = max(self.alpha, 1e-6)
        b = max(self.beta, 1e-6)
        return a / (a + b)

    def sample(self) -> float:
        """Thompson Sampling: 从 Beta(α, β) 采样, 防御 α/β ≤ 0"""
        a = max(self.alpha, 1e-6)
        b = max(self.beta, 1e-6)
        return float(np.random.beta(a, b))


# ──────────────────────────────────────────────
# §3.4 语义记忆
# ──────────────────────────────────────────────

@dataclass
class PersonalityTraits:
    """Big Five 人格特质"""
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5


@dataclass
class RelationshipInfo:
    """关系图谱中的单个人物信息"""
    role: str = ""
    closeness: float = 0.5
    conflict_patterns: List[str] = field(default_factory=list)
    positive_associations: List[str] = field(default_factory=list)
    recent_dynamics: str = ""


@dataclass
class CommunicationPreference:
    """沟通偏好"""
    preferred_strategies: List[str] = field(default_factory=list)
    disliked_strategies: List[str] = field(default_factory=list)
    humor_receptivity: float = 0.5
    preferred_depth: str = "moderate"   # surface | moderate | deep
    language_style: str = "口语化"       # 正式 | 口语化 | 网络用语


@dataclass
class BehavioralBaseline:
    """行为基线 (§2.5)"""
    mean: Dict[str, float] = field(
        default_factory=lambda: dict(BEHAVIOR_COLD_START_DEFAULTS)
    )
    std: Dict[str, float] = field(
        default_factory=lambda: {k: v * BEHAVIOR_SIGMA_INIT_MULTIPLIER for k, v in BEHAVIOR_SIGMA_MIN.items()}
    )


@dataclass
class UserProfile:
    """用户画像"""
    personality: PersonalityTraits = field(default_factory=PersonalityTraits)
    core_values: List[str] = field(default_factory=list)
    recurring_themes: List[str] = field(default_factory=list)
    life_stage: str = ""
    behavioral_baseline: BehavioralBaseline = field(default_factory=BehavioralBaseline)


@dataclass
class SemanticMemory:
    """
    §3.4 Tier 3: 语义记忆
    持久化知识图谱
    """
    user_profile: UserProfile = field(default_factory=UserProfile)
    relationship_graph: Dict[str, RelationshipInfo] = field(default_factory=dict)
    communication_preference: CommunicationPreference = field(default_factory=CommunicationPreference)


# ──────────────────────────────────────────────
# §3.5 情感记忆 (Tier 4)
# ──────────────────────────────────────────────

@dataclass
class AffectiveMemory:
    """
    §3.5 Tier 4: 情感记忆 ★ 核心创新一
    包含四个子结构
    """
    # 3.5.1 情绪基线
    emotion_baseline: EmotionBaseline = field(default_factory=EmotionBaseline)

    # 3.5.2 触发-情绪关联图
    trigger_map: Dict[str, EmotionPattern] = field(default_factory=dict)

    # 3.5.3 情绪恢复曲线
    recovery_patterns: Dict[str, RecoveryPattern] = field(default_factory=dict)

    # 3.5.4 策略有效性矩阵: (strategy_name, context_key) -> BetaParams
    strategy_matrix: Dict[Tuple[str, str], BetaParams] = field(default_factory=dict)


# ──────────────────────────────────────────────
# §3.2 工作记忆
# ──────────────────────────────────────────────

@dataclass
class WorkingMemory:
    """
    §3.2 Tier 1: 工作记忆
    当前会话的即时上下文
    """
    session_buffer: str = ""
    active_emotion: Optional[EmotionVector] = None
    active_topic: str = ""
    active_intent: str = ""
    turn_count: int = 0
    user_name: str = "用户"
    last_strategy: str = ""
    # 上一轮情绪快照 (用于 trend_local 计算)
    prev_emotion: Optional[EmotionVector] = None
    # §2.7b 危机冷却计数器 (Emotion Inertia)
    crisis_cooldown_counter: int = 0
    # §2.7b 上一轮意图快照 (用于意图连续性判断)
    prev_intent: Optional[IntentDistribution] = None
    # §2.7c 危机复发敏感期: 会话内危机发生次数 (用于 post-crisis urgency boost)
    crisis_episode_count: int = 0


# ──────────────────────────────────────────────
# §4.4 轨迹分析结果
# ──────────────────────────────────────────────

@dataclass
class TrajectoryResult:
    """情绪轨迹分析输出"""
    direction: TrajectoryDirection = TrajectoryDirection.STABLE
    beta_hat: float = 0.0                # 回归斜率
    delta_b: float = 0.0                 # 基线偏差
    volatility: float = 0.0              # 波动性
    current_phase: RecoveryPhase = RecoveryPhase.STABLE_STATE
    phase_duration: int = 0              # 当前阶段已持续轮次
    forced_exit_from_acute: bool = False  # T_max_acute 强制退出标记


# ──────────────────────────────────────────────
# §4.2 情境评估结果
# ──────────────────────────────────────────────

@dataclass
class SituationAppraisal:
    """情境评估输出"""
    severity: float = 0.0                # ∈ [0, 1]
    controllability: float = 0.5         # ∈ [0, 1] (Mock LLM 评估)
    f_trend: float = 0.5                 # 趋势因子


# ──────────────────────────────────────────────
# §4.3 目标推断结果
# ──────────────────────────────────────────────

@dataclass
class GoalSet:
    """多层级目标集合"""
    immediate_goal: str = ""       # 即时目标
    session_goal: str = ""         # 会话目标
    long_term_goal: str = ""       # 长期目标
    relationship_goal: str = ""    # 关系目标 (始终存在)


# ──────────────────────────────────────────────
# §4.5 规划输出
# ──────────────────────────────────────────────

@dataclass
class ContextVector:
    """
    定义 4.2: 情境向量 (用于 Thompson Sampling)
    离散化为 context_key 字符串
    """
    phase: RecoveryPhase = RecoveryPhase.STABLE_STATE
    severity_level: str = "low"          # low | medium | high
    intent_group: str = "neutral"        # exploration | insight | neutral
    turn_in_session: int = 0
    relationship_depth: float = 0.0

    def to_key(self) -> str:
        """离散化为字典键"""
        return f"{self.phase.value}|{self.severity_level}|{self.intent_group}"


@dataclass
class PlanningOutput:
    """规划模块完整输出"""
    trajectory: TrajectoryResult = field(default_factory=TrajectoryResult)
    appraisal: SituationAppraisal = field(default_factory=SituationAppraisal)
    goals: GoalSet = field(default_factory=GoalSet)
    selected_strategy: str = "active_listening"
    context_vector: ContextVector = field(default_factory=ContextVector)
    planning_intent: str = "default"


# ──────────────────────────────────────────────
# 延迟更新队列项
# ──────────────────────────────────────────────

@dataclass
class DeferredUpdate:
    """
    算法 3.1 步骤③: 延迟更新队列项
    在 t+1 轮回溯执行 Beta 更新
    """
    strategy: str = ""
    context_key: str = ""
    episode_id: str = ""
    emotion_snapshot: Optional[EmotionVector] = None
    intent: Optional[IntentDistribution] = None
    timestamp: datetime = field(default_factory=datetime.now)


# ──────────────────────────────────────────────
# 检索结果
# ──────────────────────────────────────────────

@dataclass
class RetrievedContext:
    """MAR 检索返回的上下文"""
    episodes: List[EpisodeUnit] = field(default_factory=list)
    semantic: Optional[SemanticMemory] = None
    affective: Optional[AffectiveMemory] = None
    grounding_facts: List[str] = field(default_factory=list)
