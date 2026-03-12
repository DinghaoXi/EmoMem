"""
EmoMem 框架全局配置 — 所有超参数与常量
=============================================
来源: emomem.md 技术规约文档
每个参数均标注了对应的文档章节号，方便交叉核验。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ═══════════════════════════════════════════════════════════
# 参数分类说明 (Parameter Classification)
# ═══════════════════════════════════════════════════════════
# Class A (安全关键，不可修改): CRISIS_THRESHOLD, STRATEGIES,
#   NUM_EMOTIONS, NUM_INTENTS, CRISIS_COOLDOWN_TURNS,
#   EMOTION_CROSS_VALIDATION_*, ACUTE_ALLOWED_STRATEGIES
# Class C (用户自适应): BASELINE_LR, EMOTION_INERTIA_ALPHA,
#   severity bins, TS priors — 由 AdaptiveParameterManager 运行时覆盖
# ═══════════════════════════════════════════════════════════


# ──────────────────────────────────────────────
# §2.1 Plutchik 情绪空间
# ──────────────────────────────────────────────
EMOTIONS = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
NUM_EMOTIONS = 8

# 定义 2.3: 效价符号向量 σ
# joy(+1), trust(+1), fear(-1), surprise(0), sadness(-1), disgust(-1), anger(-1), anticipation(+1)
VALENCE_SIGNS = np.array([+1, +1, -1, 0, -1, -1, -1, +1], dtype=np.float64)

# Plutchik 正负不对称: 均匀分布时 valence ≈ -0.125
UNIFORM_VALENCE_OFFSET = -0.125


# ──────────────────────────────────────────────
# §2.4 意图分类
# ──────────────────────────────────────────────
INTENT_TYPES = ["VENT", "COMFORT", "ADVICE", "SHARE_JOY", "CHAT", "REFLECT", "CRISIS"]
NUM_INTENTS = 7


# ──────────────────────────────────────────────
# §2.5 隐式信号检测
# ──────────────────────────────────────────────
# 行为特征维度名称
BEHAVIOR_FEATURES = ["msg_time", "reply_interval", "msg_frequency", "msg_length", "emoji_rate", "session_init"]

# 表4: BAS 权重
BAS_WEIGHTS = np.array([0.20, 0.15, 0.20, 0.15, 0.10, 0.20], dtype=np.float64)

# BAS_max: BAS 理论最大值 (所有维度偏离 5σ)
BAS_MAX = 5.0

# 行为基线更新率 (EMA)
BEHAVIOR_BASELINE_LR = 0.1          # λ = 0.1
BEHAVIOR_BASELINE_SIGMA_LR = 0.1    # λ_σ = 0.1

# 冷启动默认值 (§2.5)
BEHAVIOR_COLD_START_DEFAULTS = {
    "msg_time": 14.0,          # 下午2时 (24小时制)
    "reply_interval": 120.0,   # 秒
    "msg_frequency": 3.0,      # 条/天
    "msg_length": 50.0,        # 字符
    "emoji_rate": 0.15,        # 比率
    "session_init": 0.5,       # 次/天
}

# σ_i,min: 各维度标准差下界
BEHAVIOR_SIGMA_MIN = {
    "msg_time": 1.0,           # 小时
    "reply_interval": 10.0,    # 秒
    "msg_frequency": 0.5,      # 条/天
    "msg_length": 5.0,         # 字符
    "emoji_rate": 0.01,        # 比率
    "session_init": 0.1,       # 次/天
}

BEHAVIOR_SIGMA_INIT_MULTIPLIER = 3.0  # 初始标准差 = σ_min × 此倍数

# α_3 冷启动渐进: α_3(n_cum) = min(0.6, 0.6 * (n_cum - 1) / 5)
ALPHA3_FULL = 0.6              # 完全恢复值
ALPHA3_RAMP_INTERACTIONS = 5   # 5次交互后完全恢复

# 防除零常量
EPSILON = 1e-6


# ──────────────────────────────────────────────
# §2.7 紧急度评分
# ──────────────────────────────────────────────
URGENCY_ALPHA1 = 1.0    # 危机意图权重
URGENCY_ALPHA2 = 0.8    # 高强度负面情绪权重
URGENCY_ALPHA3 = 0.6    # 行为异常度权重 (冷启动期为函数)
URGENCY_ALPHA4 = 0.7    # 近期恶化趋势权重
CRISIS_THRESHOLD = 0.9  # urgency > 0.9 时触发危机快速通道
CRISIS_STEP_THRESHOLD = 0.5          # §2.7 危机指标阶跃阈值: P(CRISIS) > 0.5 → 1, else 0

# ──────────────────────────────────────────────
# §2.7b 情绪惯性 (Emotion Inertia)
# ──────────────────────────────────────────────
EMOTION_INERTIA_ALPHA = 0.4          # EMA 混合权重: 40% 前一轮, 60% 当前
CRISIS_COOLDOWN_TURNS = 3            # 危机后最少保持的高紧急度轮数 (显式危机)
CRISIS_COOLDOWN_IMPLICIT_TURNS = 1   # 隐含危机冷却轮数 (较短, 避免 over-persistence)
CRISIS_COOLDOWN_DECAY = 0.15         # 冷却期每轮紧急度下限衰减量
# §2.7c 危机复发敏感期 (Post-Crisis Relapse Sensitivity)
# 冷却期结束后, 若会话内曾发生过危机且用户再现负面情绪, 提供 urgency boost.
# 两层保护:
#   1. 乘数 boost: 0.85 * intensity (高于 α₂=0.8 的标准负面情绪权重)
#   2. 最低下限: 0.45 (确保即使情绪惯性稀释了强度, 仍保持有意义的紧急度)
# 安全设计: max(boost) < CRISIS_THRESHOLD=0.9, 不会误触发危机快速通道
POST_CRISIS_SENSITIVITY_ALPHA = 0.85
POST_CRISIS_URGENCY_FLOOR = 0.45     # 危机后负面情绪的最低紧急度下限
CONTEXT_INTENT_WEIGHT = 0.2          # 意图连续性权重 (低置信度时)
INTENT_CONTINUITY_THRESHOLD = 0.5    # 意图置信度低于此值时启用连续性

# §2 感知模块阈值 (从 perception.py 中集中)
EMOTION_CROSS_VALIDATION_ALPHA = 0.6           # 交叉验证中关键词权重
EMOTION_CROSS_VALIDATION_VALENCE = -0.15       # 触发交叉验证的效价阈值
EMOTION_CROSS_VALIDATION_CONFIDENCE_PENALTY = 0.7  # 混合结果置信度乘数
KEYWORD_CRISIS_FLOOR_THRESHOLD = 0.3           # 关键词危机评分最低触发线
KEYWORD_CRISIS_DAMPENING = 0.6                 # 关键词危机下限抑制因子
KL_UNIFORM_THRESHOLD = 0.1                     # KL 散度均匀分布检测阈值
MAX_KEYWORD_SCORE_PER_INTENT = 10.0            # 每意图关键词分数上限 (防 softmax 饱和)
POSITIVE_CRISIS_SUPPRESSION_STRONG = 0.2       # 2+ 正面指标时危机抑制因子
POSITIVE_CRISIS_SUPPRESSION_MILD = 0.5         # 1 个正面指标时危机抑制因子
MAX_CONVERSATION_HISTORY = 200                 # 感知模块对话历史最大轮数

# §4.5 LLM 语义信号阈值 (从 planning.py 中集中)
LLM_OPENNESS_SILENCE_THRESHOLD = 0.25         # 沟通开放度低于此值 → companionable_silence
LLM_OVERRIDE_CONFIDENCE_THRESHOLD = 0.6       # LLM approach 覆盖 TS 选择的最低置信度


# ──────────────────────────────────────────────
# §3.2 工作记忆
# ──────────────────────────────────────────────
WORKING_MEMORY_TOKEN_LIMIT = 2000


# ──────────────────────────────────────────────
# §3.3 情景记忆
# ──────────────────────────────────────────────
# 定义 3.2: 重要性评分权重
IMPORTANCE_WEIGHTS = np.array([0.30, 0.30, 0.25, 0.15], dtype=np.float64)
#                              EmotionalArousal, LifeImpact, MentionFreq, Recurrence
N_FREQ = 10          # MentionFreq 归一化上限
N_REC = 5            # Recurrence 归一化上限
TAU_WRITE = 0.3      # 写入重要性阈值


# ──────────────────────────────────────────────
# §3.5.1 情绪基线
# ──────────────────────────────────────────────
BASELINE_LR = 0.05              # η = 0.05 (基线学习率)
BASELINE_SIGMA_LR = 0.05        # η_σ = 0.05 (σ学习率)
BASELINE_CONFIDENCE_INCREMENT = 0.01   # Δc = 0.01
BASELINE_CONFIDENCE_DECAY = 0.05       # Δc_decay = 0.05
BASELINE_SIGMA_INIT = 0.3       # σ_baseline 初始值
BASELINE_SIGMA_MIN = 0.05       # σ_min 下界
BASELINE_CONFIDENCE_INIT = 0.2  # 初始置信度
BASELINE_RESET_THRESHOLD = 0.3  # confidence < 0.3 触发重估
RING_BUFFER_SIZE = 20           # K_buf = 20

# 冷启动: 均匀分布 [1/8, ..., 1/8]
BASELINE_COLD_START = np.ones(NUM_EMOTIONS, dtype=np.float64) / NUM_EMOTIONS


# ──────────────────────────────────────────────
# §3.5.4 策略有效性矩阵 & Beta 分布
# ──────────────────────────────────────────────
# 初始先验
BETA_ALPHA0 = 2.0               # α₀
BETA_BETA0 = 2.0                # β₀

# 时间衰减
BETA_DECAY_HALFLIFE_DAYS = 60   # τ_β = 60 天

# 会话末轮衰减系数
SESSION_END_DECAY = 0.5         # ξ = 0.5

# VENT 负向信号衰减因子
GAMMA_VENT = 0.3                # γ_vent = 0.3


# ──────────────────────────────────────────────
# §3.7 MAR (Mood-Adaptive Retrieval)
# ──────────────────────────────────────────────
# MAR 基础权重 (定义 3.5)
MAR_ALPHA_BASE = 0.35    # 语义相似度
MAR_BETA_BASE = 0.25     # 情绪一致性
MAR_GAMMA_BASE = 0.20    # 时间近因性
MAR_OMEGA_BASE = 0.20    # 重要性

# 时间衰减参数
MAR_LAMBDA_R = np.log(2)         # λ_r = ln(2)
MAR_TAU_R_DAYS = 30              # τ_r = 30 天

# 检索返回数量
MAR_TOP_K = 10

# 定义 3.6b: Planning intent 到 Φ 映射 (规约 §3.7)
PLANNING_INTENT_PHI = {
    "positive_reframing": 0.1,     # 正向重构: 抑制负面情绪一致检索
    "emotional_validation": 0.3,   # 共情验证: 适度情绪匹配
    "crisis_response": 0.5,        # 危机响应: 中等情绪一致性
    "default": 1.0,                # 默认
    "pattern_reflection": 1.5,     # 模式反思: 强化情绪一致检索
}


# ──────────────────────────────────────────────
# §3.9 遗忘 (Forget)
# ──────────────────────────────────────────────
TAU_FORGET_DAYS = 90        # τ_f = 90 天 (遗忘半衰期)
RETENTION_THRESHOLD = 0.1   # Retention < 0.1 时归档
EMOTION_PROTECTION_RHO = 3.0   # ρ = 3.0 (情感保护系数)


# ──────────────────────────────────────────────
# §4.1 策略空间 (10 种帮助技能)
# ──────────────────────────────────────────────
STRATEGIES = [
    "active_listening",         # s1: 积极倾听
    "emotional_validation",     # s2: 情感验证
    "empathic_reflection",      # s3: 共情反射
    "gentle_guidance",          # s4: 温和引导
    "cognitive_reframing",      # s5: 认知重构
    "problem_solving",          # s6: 问题解决
    "information_providing",    # s7: 信息提供
    "strength_recognition",     # s8: 优势识别
    "companionable_silence",    # s9: 陪伴性沉默
    "positive_reinforcement",   # s10: 积极强化
]
NUM_STRATEGIES = 10

# 策略分组: Exploration (被动探索) vs Insight (主动引导)
EXPLORATION_STRATEGIES = {"active_listening", "emotional_validation", "empathic_reflection", "companionable_silence"}
INSIGHT_STRATEGIES = {"gentle_guidance", "cognitive_reframing", "problem_solving", "information_providing", "strength_recognition", "positive_reinforcement"}


# ──────────────────────────────────────────────
# §4.4 情绪轨迹分析
# ──────────────────────────────────────────────
TRAJECTORY_WINDOW_K = 5         # k = 5 (回归窗口会话数)
TAU_D = 0.05                    # τ_d = 0.05 (方向稳定判定阈值)

# 阶段判定阈值
ACUTE_DISTRESS_DELTA_B = -0.3    # δ_b < -0.3
EARLY_RECOVERY_DELTA_B = -0.1    # -0.3 ≤ δ_b ≤ -0.1
CONSOLIDATION_DELTA_B = 0.1      # |δ_b| ≤ 0.1

# 滞后带 (Hysteresis)
T_HOLD = 3                      # 最小保持轮次
EPSILON_H = 0.05                 # 滞后带宽度
EPSILON_H_EXIT = 0.02            # acute_distress 退出滞后阈值
T_MAX_ACUTE = 10                 # acute_distress 最大保持轮次

# 恢复阶段枚举
RECOVERY_PHASES = ["acute_distress", "early_recovery", "consolidation", "stable_state"]

# acute_distress 阶段允许的策略
ACUTE_ALLOWED_STRATEGIES = {"active_listening", "emotional_validation", "empathic_reflection", "companionable_silence"}


# ──────────────────────────────────────────────
# §4.5 Thompson Sampling
# ──────────────────────────────────────────────
# 情境离散化维度 (定义 4.2)
CONTEXT_DIMENSIONS = {
    "phase": RECOVERY_PHASES,  # 4 个阶段
    "severity": ["low", "medium", "high"],  # 3 个级别
    "intent_group": ["exploration", "insight", "neutral"],  # 3 个意图组
}
# 总情境数 = 4 × 3 × 3 = 36
NUM_CONTEXTS = 36

# 层次贝叶斯先验 (§4.5.2)
N_BLEND = 8                 # 渐进混合观测数
N_PRIOR = 4                 # 固定伪观测数上限
COLD_START_MU = 0.5         # 全冷启动回退 μ_si

# §4.5 情绪适配约束
POSITIVE_REINFORCEMENT_VALENCE_BLOCK = -0.3  # 强负面情绪时禁止积极强化

# 关系深度软约束 (§4.5.2)
RELATIONSHIP_DEPTH_THRESHOLD = 0.3   # d < 0.3 时触发
P_REL_ROLLBACK = 0.3                 # 回退概率 30%
RELATIONSHIP_DEPTH_LR = 0.1          # λ_d = 0.1


# ──────────────────────────────────────────────
# §4.2 情境评估 (Situation Appraisal)
# ──────────────────────────────────────────────
# severity 权重 (§4.2 定义 4.1)
SEVERITY_W1 = 0.30   # 情绪强度
SEVERITY_W2 = 0.25   # 基线偏差
SEVERITY_W3 = 0.25   # 趋势因子
SEVERITY_W4 = 0.20   # 不可控性

# f_trend 映射
F_TREND_MAP = {
    "declining": 1.0,
    "volatile": 0.75,
    "stable": 0.5,
    "improving": 0.0,
}


# ──────────────────────────────────────────────
# §3.4 反馈评分 (Feedback Score)
# ──────────────────────────────────────────────
# feedback_score 权重
FEEDBACK_DELTA_EMOTION_W = 0.35
FEEDBACK_ENGAGEMENT_W = 0.25
FEEDBACK_CONTINUATION_W = 0.20
FEEDBACK_EXPLICIT_W = 0.20

# 会话级代理评分权重 (Session-Level Surrogate)
SURROGATE_DELTA_VALENCE_W = 0.40
SURROGATE_F_BAR_W = 0.30
SURROGATE_EXPLICIT_LAST_W = 0.30


# ──────────────────────────────────────────────
# §4.5.2 自我披露深度等级
# ──────────────────────────────────────────────
DISCLOSURE_LEVELS = {
    "L0_factual": 0.0,
    "L1_opinion": 0.33,        # 规约 §4.5.2: 探索性披露
    "L2_emotion": 0.67,        # 规约 §4.5.2: 情感性披露
    "L3_vulnerability": 1.0,
}


# ──────────────────────────────────────────────
# §5.3 人格一致性守卫参数 (Persona Guard)
# ──────────────────────────────────────────────
PERSONA_GUARD_HUMAN_CLAIM_PHRASES = [
    "我是人类", "我是真人", "我不是AI", "我不是机器"
]
PERSONA_GUARD_HUMAN_CLAIM_REPLACEMENT = "我是你的AI伙伴"
PERSONA_GUARD_MEDICAL_PHRASES = [
    "你得了", "你患有", "诊断为", "你的病是"
]
PERSONA_GUARD_MEDICAL_REPLACEMENT = "建议你咨询专业医生了解"
PERSONA_GUARD_EMPTY_FALLBACK = "我在这里，你想聊什么都可以。"

# ──────────────────────────────────────────────
# §5.4 语调校准参数 (Tone Calibration)
# ──────────────────────────────────────────────
TONE_ACUTE_MAX_LENGTH = 100
TONE_ACUTE_SEARCH_WINDOW = 120
TONE_ACUTE_MIN_LENGTH = 20
TONE_SENTENCE_SEPARATORS = ["。", "！", "？", "…", "，", "；"]
TONE_TRUNCATION_ELLIPSIS = "……"

# ──────────────────────────────────────────────
# §5.5 危机关切参数 (Crisis Concern)
# ──────────────────────────────────────────────
CRISIS_CONCERN_FAREWELL_SIGNALS = [
    "想通了", "整理好了", "该还的", "人情", "再见",
    "最后一次", "不会再麻烦", "不用担心我了", "以后不会了", "添麻烦"
]
CRISIS_CONCERN_EXISTENTIAL_SIGNALS = [
    "活着", "为了什么", "意义", "没有意义", "人生"
]
CRISIS_CONCERN_GRATITUDE_SIGNALS = [
    "谢谢你", "你是个好", "谢谢", "再见"
]
CRISIS_CONCERN_HARM_SIGNALS = [
    "不想活", "想死", "自杀", "结束", "跳楼", "割腕",
    "安眠药", "了结"
]
CRISIS_CONCERN_MIN_LENGTH = 5

# ──────────────────────────────────────────────
# §6.3 自适应框架公式常数 (Adaptive Formula Constants)
# ──────────────────────────────────────────────
ADAPTIVE_LR_VARIANCE_SENSITIVITY = 20.0
ADAPTIVE_INERTIA_BASE = 0.3
ADAPTIVE_INERTIA_SCALE = 0.2
ADAPTIVE_TS_BLEND_HORIZON = 20.0
ADAPTIVE_SEVERITY_MIN_SAMPLES = 10
ADAPTIVE_SEVERITY_MIN_SPREAD = 0.2
ADAPTIVE_SEVERITY_PERCENTILE_LOW = 25          # 低严重度分位数
ADAPTIVE_SEVERITY_PERCENTILE_HIGH = 75         # 高严重度分位数
ADAPTIVE_SEVERITY_LOW_BOUNDS: Tuple[float, float] = (0.15, 0.45)
ADAPTIVE_SEVERITY_HIGH_BOUNDS: Tuple[float, float] = (0.55, 0.85)
ADAPTIVE_TS_MIN_INTERACTIONS = 5               # TS 自适应先验最低交互次数
ADAPTIVE_TS_PRIOR_MIN = 0.2                    # TS 自适应先验下限
ADAPTIVE_TS_PRIOR_MAX = 0.8                    # TS 自适应先验上限


@dataclass
class EmoMemConfig:
    """聚合配置对象，便于依赖注入"""
    # 可在此处覆盖任何默认值
    debug: bool = False
    max_history_turns: int = 100
    mar_top_k: int = MAR_TOP_K
