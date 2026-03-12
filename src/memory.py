"""
EmoMem 记忆模块 (Memory Module) — 四层记忆架构与五大操作
===========================================================
来源: emomem.md §3 Memory Module 规约

四层记忆架构 (4-Tier Memory Architecture):
  - Tier 1: Working Memory    工作记忆 (当前会话即时上下文)
  - Tier 2: Episodic Memory   情景记忆 (带时间戳的事件记录)
  - Tier 3: Semantic Memory   语义记忆 (持久化知识图谱)
  - Tier 4: Affective Memory  情感记忆 ★核心创新一 (情感模式库)

五大记忆操作 (5 Memory Operations):
  1. Write   — 选择性写入 (Algorithm 3.1)
  2. Read    — 心境自适应检索 MAR (Definition 3.5/3.6) ★核心创新二
  3. Reflect — 反思整合 (Algorithm 3.2)
  4. Forget  — 情感保护遗忘 (Definition 3.7)
  5. Update  — 修订式更新 (§3.10)
"""

from __future__ import annotations

import logging
import math
import re
import uuid
import numpy as np

logger = logging.getLogger(__name__)
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    # §3.2 工作记忆
    WORKING_MEMORY_TOKEN_LIMIT,
    # §3.3 情景记忆 — 重要性评分
    IMPORTANCE_WEIGHTS, N_FREQ, N_REC, TAU_WRITE,
    # §3.5.1 情绪基线
    BASELINE_LR, BASELINE_SIGMA_LR,
    BASELINE_CONFIDENCE_INCREMENT, BASELINE_CONFIDENCE_DECAY,
    BASELINE_SIGMA_MIN, BASELINE_CONFIDENCE_INIT,
    BASELINE_COLD_START, BASELINE_RESET_THRESHOLD,
    RING_BUFFER_SIZE, NUM_EMOTIONS,
    # §3.5.4 策略有效性矩阵 & Beta 分布
    BETA_ALPHA0, BETA_BETA0,
    BETA_DECAY_HALFLIFE_DAYS, SESSION_END_DECAY, GAMMA_VENT,
    # §3.7 MAR 检索
    MAR_ALPHA_BASE, MAR_BETA_BASE, MAR_GAMMA_BASE, MAR_OMEGA_BASE,
    MAR_LAMBDA_R, MAR_TAU_R_DAYS, MAR_TOP_K,
    PLANNING_INTENT_PHI,
    # §3.9 遗忘
    TAU_FORGET_DAYS, RETENTION_THRESHOLD, EMOTION_PROTECTION_RHO,
    # §3.4 反馈评分
    FEEDBACK_DELTA_EMOTION_W, FEEDBACK_ENGAGEMENT_W,
    FEEDBACK_CONTINUATION_W, FEEDBACK_EXPLICIT_W,
    SURROGATE_DELTA_VALENCE_W, SURROGATE_F_BAR_W, SURROGATE_EXPLICIT_LAST_W,
    # §2.5 行为基线
    BEHAVIOR_BASELINE_LR, BEHAVIOR_BASELINE_SIGMA_LR, BEHAVIOR_SIGMA_MIN,
    # 其他
    EPSILON, VALENCE_SIGNS,
)
from .models import (
    # 数据结构
    EmotionVector, StateVector, IntentDistribution,
    EpisodeUnit, WorkingMemory, SemanticMemory, AffectiveMemory,
    BetaParams, EmotionBaseline, DeferredUpdate, RetrievedContext,
    BehavioralBaseline, EmotionPattern, PlanningOutput,
    # 枚举
    ResolutionStatus,
)


# ══════════════════════════════════════════════════════════════
# 辅助函数 (Helper Functions)
# ══════════════════════════════════════════════════════════════

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """将值裁剪到 [lo, hi] 区间 / Clamp value to [lo, hi].
    使用 np.clip 以正确传播 NaN (IEEE 754 compliant).
    """
    return float(np.clip(value, lo, hi))


def _keyword_overlap(text_a: str, text_b: str) -> float:
    """
    简易文本相似度: 关键词重叠率 (Mock, 替代真实 embedding 余弦相似度)
    Simple keyword overlap ratio as a mock for cosine similarity of embeddings.
    对两段文本提取去重关键词, 返回 Jaccard 相似度 ∈ [0, 1].
    """
    # 分词: 中文按双字 bigram 切分 (近似分词), 英文按空格/标点
    # Tokenization: Chinese bigrams (approximate segmentation), English by whitespace
    def _tokenize(text: str) -> set:
        tokens = set()
        # 英文单词
        tokens.update(re.findall(r'[a-zA-Z]+', text.lower()))
        # 中文: 提取连续汉字串, 再按 bigram 切分
        for seg in re.findall(r'[\u4e00-\u9fff]+', text):
            if len(seg) == 1:
                tokens.add(seg)
            else:
                for i in range(len(seg) - 1):
                    tokens.add(seg[i:i+2])
        return tokens
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union) if union else 0.0


def _llm_semantic_similarity(
    text_a: str,
    text_b: str,
    llm_provider=None,
) -> float:
    """
    LLM 驱动的语义相似度 (替代关键词重叠)
    LLM-driven semantic similarity (replaces keyword overlap).

    当 llm_provider 可用时, 使用 LLM 评估两段文本的语义相似度;
    否则降级为关键词重叠。
    """
    if llm_provider is None:
        return _keyword_overlap(text_a, text_b)

    try:
        response = llm_provider.client.chat.completions.create(
            model=llm_provider.model,
            messages=[
                {"role": "system", "content": "你是语义相似度评估器。评估两段文本的语义相似度，输出一个0.0到1.0的浮点数。0.0=完全无关，1.0=语义完全相同。只输出数字，不要其他内容。"},
                {"role": "user", "content": f"文本A: {text_a[:200]}\n\n文本B: {text_b[:200]}"},
            ],
            max_tokens=10,
        )
        text = (response.choices[0].message.content or "").strip()
        import re as _re
        match = _re.search(r'[\d.]+', text)
        if match:
            return float(min(1.0, max(0.0, float(match.group()))))
        return _keyword_overlap(text_a, text_b)
    except Exception:
        return _keyword_overlap(text_a, text_b)


def _emotion_congruence(e_query: np.ndarray, e_memory: np.ndarray) -> float:
    """
    情绪一致性 EmoCon: 1 - ||e_query - e_memory||_2 / sqrt(2)
    Emotional Congruence score.
    使用 L2 距离归一化到 [0, 1], 因为两个概率分布之差的 L2 范数最大为 sqrt(2).

    注: 规约文档定义3.5中使用 JSD, 但此处简化为 L2 距离的归一化版本
    作为 Mock 实现. 两者均衡量分布相似度, L2 实现更轻量.
    """
    # Validate simplex constraint for e_query
    q_sum = float(e_query.sum())
    if abs(q_sum - 1.0) > 0.05:
        logger.warning("e_query off-simplex (sum=%.4f), renormalizing", q_sum)
        if q_sum > 1e-8:
            e_query = e_query / q_sum
        else:
            e_query = np.ones_like(e_query) / len(e_query)

    # Validate simplex constraint for e_memory
    m_sum = float(e_memory.sum())
    if abs(m_sum - 1.0) > 0.05:
        logger.warning("e_memory off-simplex (sum=%.4f), renormalizing", m_sum)
        if m_sum > 1e-8:
            e_memory = e_memory / m_sum
        else:
            e_memory = np.ones_like(e_memory) / len(e_memory)

    diff_norm = np.linalg.norm(e_query - e_memory)
    # 两个单纯形上分布的 L2 距离上界为 sqrt(2)
    # 裁剪到 [0, 1] 防止未归一化输入导致负值
    return float(max(0.0, 1.0 - diff_norm / math.sqrt(2)))


def _detect_explicit_feedback(text: str) -> float:
    """
    显式反馈关键词检测 (Mock) — 检测用户文本中的正/负向反馈信号
    Detect explicit feedback keywords in user text.

    正向关键词 → 高值 (0.8-1.0)
    负向关键词 → 低值 (0.0-0.2)
    无明确反馈 → 中性值 (0.5)
    """
    positive_keywords = [
        "谢谢", "感谢", "说得对", "有道理", "好的", "嗯嗯", "对对",
        "明白了", "懂了", "有帮助", "很好", "不错",
        "thank", "thanks", "helpful", "agree", "right", "exactly", "yes",
    ]
    negative_keywords = [
        "算了", "你不懂", "没用", "不想说了", "烦", "无语", "别说了",
        "不是这样", "你不理解", "没有帮助",
        "useless", "whatever", "forget it", "you don't understand", "stop",
    ]
    text_lower = text.lower()
    pos_count = sum(1 for kw in positive_keywords if kw in text_lower)
    neg_count = sum(1 for kw in negative_keywords if kw in text_lower)

    if pos_count > 0 and neg_count == 0:
        return min(1.0, 0.7 + 0.1 * pos_count)
    elif neg_count > 0 and pos_count == 0:
        return max(0.0, 0.3 - 0.1 * neg_count)
    elif pos_count > 0 and neg_count > 0:
        # 混合信号 → 略偏中性 / Mixed signals → slightly neutral
        return 0.5
    else:
        # 无明确反馈 → 默认 0.5 / No explicit feedback → default 0.5
        return 0.5


def _mock_engagement(text: str) -> float:
    """
    Mock 参与度信号: 基于文本长度估算
    Mock engagement signal based on text length.

    较长的回复通常意味着更高的参与度.
    engagement = min(1.0, 0.5 * length_ratio + 0.5 * speed_ratio)
    此处简化: 仅使用 length_ratio (字符数 / 200), speed_ratio 取 0.5 (中性).
    """
    char_count = len(text.strip())
    length_ratio = min(1.0, char_count / 200.0)
    speed_ratio = 0.5  # Mock: 无法获取真实回复速度
    return min(1.0, 0.5 * length_ratio + 0.5 * speed_ratio)


def _mock_continuation(prev_topics: List[str], current_topics: List[str]) -> float:
    """
    Mock 话题延续信号: 检测用户是否继续讨论同一话题
    Mock continuation signal: detect if user continues the same topic.
    返回 0.0 (话题转换) 或 1.0 (话题延续).
    """
    if not prev_topics or not current_topics:
        return 0.5  # 无法判断 / Cannot determine
    overlap = set(prev_topics) & set(current_topics)
    return 1.0 if overlap else 0.0


# ══════════════════════════════════════════════════════════════
# MemoryModule 主类
# ══════════════════════════════════════════════════════════════

class MemoryModule:
    """
    EmoMem 记忆模块 — 管理四层记忆架构与五大记忆操作.

    Memory Module — manages the 4-tier memory architecture and 5 memory operations.

    四层记忆架构:
      - working_memory:   Tier 1 工作记忆 (当前会话即时上下文)
      - episodic_memory:  Tier 2 情景记忆 (时间戳标注的事件列表)
      - semantic_memory:  Tier 3 语义记忆 (持久化知识图谱)
      - affective_memory: Tier 4 情感记忆 (情感模式库 ★核心创新)

    延迟更新队列:
      - deferred_update_queue: 存储待执行的 Beta 更新项 (Algorithm 3.1 Step ③)

    归档区:
      - archived_episodes: 被遗忘但未删除的情景记忆 (Definition 3.7)
    """

    def __init__(self, llm_provider=None):
        """
        初始化四层空记忆结构.
        Initialize the 4-tier empty memory structure.

        Args:
            llm_provider: 可选的 LLM 提供者实例, 用于语义相似度等增强功能.
                          Optional LLM provider instance for semantic similarity and other enhancements.
        """
        # ── LLM 提供者 (可选) ──
        self.llm_provider = llm_provider

        # ── Tier 1: 工作记忆 (Working Memory) ──
        # 当前会话的即时上下文, 每轮更新
        self.working_memory = WorkingMemory()

        # ── Tier 2: 情景记忆 (Episodic Memory) ──
        # 带时间戳的事件记录列表
        self.episodic_memory: List[EpisodeUnit] = []

        # ── Tier 3: 语义记忆 (Semantic Memory) ──
        # 持久化知识图谱: 用户画像 + 关系图谱 + 沟通偏好
        self.semantic_memory = SemanticMemory()

        # ── Tier 4: 情感记忆 (Affective Memory) ★核心创新 ──
        # 情感模式库: 情绪基线 + 触发图 + 恢复曲线 + 策略有效性矩阵
        self.affective_memory = AffectiveMemory()

        # ── 延迟更新队列 (Deferred Update Queue) ──
        # Algorithm 3.1 Step ③: 存储在 t+1 轮才执行的 Beta 更新项
        self.deferred_update_queue: List[DeferredUpdate] = []

        # ── 归档区 (Archive) ──
        # Definition 3.7: 低保留度记忆的归档, 不删除
        self.archived_episodes: List[EpisodeUnit] = []

    # ══════════════════════════════════════════════
    # 记忆操作一: 写入 (Write) — Algorithm 3.1
    # ══════════════════════════════════════════════

    def write(
        self,
        state_vector: StateVector,
        agent_strategy: str,
        planning_output: PlanningOutput,
    ) -> Optional[EpisodeUnit]:
        """
        算法 3.1: 选择性写入 (Selective Write).

        Algorithm 3.1: Selective Write.

        步骤:
          ① 始终更新 Working Memory (每轮实时更新)
          ② 条件性写入 Episodic Memory (importance > τ_write = 0.3)
          ③ 入队延迟更新项, 在 t+1 轮回溯执行 Beta 更新

        Args:
            state_vector: 感知模块输出的状态向量 (Definition 2.6)
            agent_strategy: Agent 本轮使用的策略名称
            planning_output: 规划模块输出 (包含 context_vector, planning_intent 等)

        Returns:
            写入的 EpisodeUnit (如果通过重要性阈值), 否则 None
        """
        # ── Step ① 始终更新工作记忆 ──
        # Always update working memory with current turn information
        self._update_working_memory(state_vector, agent_strategy)

        # ── Step ② 条件性写入情景记忆 ──
        # Construct episode unit from current turn data
        episode = self._construct_episode(state_vector, agent_strategy)
        episode.importance = self.compute_importance(episode)

        stored_episode: Optional[EpisodeUnit] = None

        if episode.importance > TAU_WRITE:
            # 重要性 > τ_write = 0.3, 写入情景记忆
            # Importance exceeds threshold, store in episodic memory
            # feedback_score 暂为 None, 延迟至 t+1 轮填充 (Definition 3.4 时序说明)
            self.episodic_memory.append(episode)
            stored_episode = episode

        # ── Step ③ 入队延迟更新项 ──
        # ★ Beta 更新独立于 Episodic Memory 存储:
        # 即使 episode 未入库 (importance ≤ τ_write), 策略学习信号仍被保留
        # deferred_update_queue 自包含所有必要信息
        if agent_strategy:
            context_key = planning_output.context_vector.to_key()
            deferred = DeferredUpdate(
                strategy=agent_strategy,
                context_key=context_key,
                episode_id=episode.id,
                emotion_snapshot=state_vector.emotion.copy(),
                intent=IntentDistribution(probabilities=state_vector.intent.probabilities.copy()),
                timestamp=datetime.now(),
            )
            self.deferred_update_queue.append(deferred)

        return stored_episode

    def _update_working_memory(self, state_vector: StateVector, agent_strategy: str):
        """
        更新工作记忆: 每轮实时刷新即时上下文.
        Update working memory with current turn context.

        包括: 当前情绪快照, 话题, 意图, 轮次计数, 上一轮策略.
        """
        # prev_emotion 由 main.py _post_turn_update() 统一更新, 此处不再重复赋值
        # prev_emotion is updated by main.py _post_turn_update() — single source of truth

        self.working_memory.active_emotion = state_vector.emotion.copy()
        self.working_memory.active_topic = state_vector.topic_category
        self.working_memory.active_intent = state_vector.intent.top_intent()
        # turn_count 由 main.py process_turn() 统一设置, 此处不再自增
        # turn_count is set by main.py process_turn() — single source of truth
        self.working_memory.last_strategy = agent_strategy

        # 更新 session_buffer 摘要 (简化版: 追加关键信息)
        # Update session buffer summary (simplified: append key info)
        turn_summary = (
            f"[Turn {self.working_memory.turn_count}] "
            f"topic={state_vector.topic_category}, "
            f"intent={state_vector.intent.top_intent()}, "
            f"strategy={agent_strategy}"
        )
        self.working_memory.session_buffer += f"\n{turn_summary}"

        # 容量管理: 超过 token 限制时截断保留最近内容
        # Capacity management: truncate when exceeding token limit
        # (简化: 以字符数近似 token 数, 中文约 1 字 ≈ 1-2 tokens)
        if len(self.working_memory.session_buffer) > WORKING_MEMORY_TOKEN_LIMIT:
            # 保留后半部分 (最近的对话记录)
            prefix = "...[truncated]..."
            max_content = WORKING_MEMORY_TOKEN_LIMIT - len(prefix)
            self.working_memory.session_buffer = prefix + self.working_memory.session_buffer[-max_content:]

    def _construct_episode(self, state_vector: StateVector, agent_strategy: str) -> EpisodeUnit:
        """
        从状态向量构造情景记忆单元 (Definition 3.1).
        Construct an EpisodeUnit from the current state vector.
        """
        episode = EpisodeUnit(
            id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(),
            event=state_vector.raw_signal.text_content if state_vector.raw_signal else "",
            emotion_snapshot=state_vector.emotion.copy(),
            people_involved=[],  # 可由 NER 模块提取 / Can be extracted by NER
            topic_category=state_vector.topic_category,
            user_coping_style="",  # 可由 LLM 推断 / Can be inferred by LLM
            agent_strategy=agent_strategy,
            user_feedback_score=None,  # ★ 延迟一轮填充 / Delayed by one turn
            resolution_status=ResolutionStatus.UNRESOLVED,
            emotional_valence=state_vector.emotion.valence(),
            importance=0.0,  # 将在 compute_importance 中计算
            mention_count=0,
            similar_event_count=0,
        )
        return episode

    # ══════════════════════════════════════════════
    # 重要性评分 (Importance Score) — Definition 3.2
    # ══════════════════════════════════════════════

    def compute_importance(self, episode: EpisodeUnit) -> float:
        """
        定义 3.2: 重要性评分.
        Definition 3.2: Importance Score.

        Importance(e) = w1·EmotionalArousal + w2·LifeImpact + w3·MentionFreq + w4·Recurrence

        各分量均归一化至 [0, 1]:
          - EmotionalArousal: 情绪快照的强度 ι(e) ∈ [0, 1]
          - LifeImpact: LLM 评估 (Mock = 0.5)
          - MentionFreq: min(mention_count / N_freq, 1.0), N_freq = 10
          - Recurrence: min(similar_event_count / N_rec, 1.0), N_rec = 5

        权重: w = [0.30, 0.30, 0.25, 0.15]
        """
        # 分量计算 / Component calculation
        emotional_arousal = episode.emotion_snapshot.intensity
        life_impact = 0.5  # Mock: LLM 评估不可用 / LLM assessment unavailable
        mention_freq = min(episode.mention_count / N_FREQ, 1.0)
        recurrence = min(episode.similar_event_count / N_REC, 1.0)

        components = np.array([emotional_arousal, life_impact, mention_freq, recurrence])
        importance = float(np.dot(IMPORTANCE_WEIGHTS, components))

        return _clamp(importance, 0.0, 1.0)

    # ══════════════════════════════════════════════
    # 延迟更新处理 (Deferred Update Processing)
    # Algorithm 3.1 Step ③ 的回溯执行
    # ══════════════════════════════════════════════

    def process_deferred_updates(self, current_state_vector: StateVector):
        """
        在 t+1 轮开始时回溯执行延迟更新.
        Execute deferred updates at the start of turn t+1.

        流程:
          1. 遍历 deferred_update_queue 中的每一项
          2. 使用当前轮 (t+1) 的信号计算 feedback_score (Definition 3.4)
          3. 更新 Beta 参数 (Definition 3.3)
          4. 如果对应 episode 存在于 episodic_memory, 填充其 feedback_score
          5. 清空队列

        Args:
            current_state_vector: 当前轮 (t+1) 的状态向量, 包含用户对上一轮的反馈信号
        """
        computed_scores: List[float] = []
        for deferred in self.deferred_update_queue:
            # 计算反馈评分 (Definition 3.4)
            feedback_score = self.compute_feedback_score(current_state_vector, deferred)
            computed_scores.append(feedback_score)

            # 更新 Beta 参数 (Definition 3.3)
            self.update_beta(
                strategy=deferred.strategy,
                context_key=deferred.context_key,
                feedback_score=feedback_score,
                is_session_end=False,
            )

            # 如果 episode 存在于 episodic_memory, 填充 feedback_score
            # If the episode exists in episodic memory, fill in feedback_score
            for ep in self.episodic_memory:
                if ep.id == deferred.episode_id:
                    ep.user_feedback_score = feedback_score
                    break

        # 清空队列 / Clear the queue
        self.deferred_update_queue.clear()
        return computed_scores

    # ══════════════════════════════════════════════
    # 反馈评分 (Feedback Score) — Definition 3.4
    # ══════════════════════════════════════════════

    def compute_feedback_score(
        self,
        current_sv: StateVector,
        deferred: DeferredUpdate,
    ) -> float:
        """
        定义 3.4: 多信号反馈评分.
        Definition 3.4: Multi-signal feedback score.

        feedback_score = clamp(
            0.35 · Δemotion_adj + 0.25 · engagement + 0.20 · continuation + 0.20 · explicit,
            0, 1
        )

        VENT 宣泄衰减:
          当 P(VENT) > 0.5 且 engagement > 0.6 时:
            Δemotion_adj = max(Δemotion, γ_vent · Δemotion)
          其中 γ_vent = 0.3

        Args:
            current_sv: 当前轮 (t+1) 的状态向量
            deferred: 上一轮 (t) 的延迟更新项 (包含上一轮的情绪快照和意图)

        Returns:
            feedback_score ∈ [0, 1]
        """
        # ── Δemotion: 情绪效价变化 ──
        # 当前轮效价 - 上一轮效价, 归一化到 [0, 1]
        # (原始范围 [-1,+1] 直接加权会导致 raw_score 可为负, 使 Thompson Sampling 系统性低估)
        current_valence = current_sv.emotion.valence()
        prev_valence = deferred.emotion_snapshot.valence() if deferred.emotion_snapshot else 0.0
        delta_emotion_raw = _clamp(current_valence - prev_valence, -1.0, 1.0)
        delta_emotion = (delta_emotion_raw + 1.0) / 2.0  # [-1,1] → [0,1], 中性变化=0.5

        # ── VENT 宣泄衰减 ──
        # 当意图为 VENT (概率 > 0.5) 且参与度高 (> 0.6) 时,
        # 负向情绪信号衰减至 γ_vent = 0.3 倍 (避免宣泄场景的系统性误判)
        delta_emotion_adj = delta_emotion  # 默认不调整
        current_text = current_sv.raw_signal.text_content if current_sv.raw_signal else ""
        engagement = _mock_engagement(current_text)

        if deferred.intent is not None:
            p_vent = deferred.intent.p("VENT")

            if p_vent > 0.5 and engagement > 0.6:
                # VENT 衰减: 负向偏移(< 0.5)向中性拉回, 正向信号保留
                # 0.5 为中性点, 偏移量衰减至 γ_vent=0.3 倍
                dampened = 0.5 + GAMMA_VENT * (delta_emotion - 0.5)
                delta_emotion_adj = max(delta_emotion, dampened)

        # ── engagement: 参与度 (已在上面计算) ──

        # ── continuation: 话题延续 ──
        prev_topics = [deferred.episode_id]  # 简化 fallback
        # 从 working memory 获取上一轮话题关键词
        prev_topic_kw = (
            self.working_memory.active_topic.split()
            if self.working_memory.active_topic
            else []
        )
        current_topic_kw = current_sv.topic_keywords if current_sv.topic_keywords else []
        continuation = _mock_continuation(prev_topic_kw, current_topic_kw)

        # ── explicit: 显式反馈 ──
        # 优先使用 LLM 感知模块提供的显式反馈值 (当可用时)
        # Prefer LLM-derived explicit feedback from perception when available
        if hasattr(current_sv, 'explicit_feedback') and current_sv.explicit_feedback is not None:
            explicit = current_sv.explicit_feedback
        else:
            explicit = _detect_explicit_feedback(current_text)

        # ── 加权求和 + clamp ──
        raw_score = (
            FEEDBACK_DELTA_EMOTION_W * delta_emotion_adj    # 0.35
            + FEEDBACK_ENGAGEMENT_W * engagement            # 0.25
            + FEEDBACK_CONTINUATION_W * continuation        # 0.20
            + FEEDBACK_EXPLICIT_W * explicit                # 0.20
        )
        return _clamp(raw_score, 0.0, 1.0)

    # ══════════════════════════════════════════════
    # 会话级代理评分 (Session-Level Surrogate)
    # ══════════════════════════════════════════════

    def compute_session_surrogate(
        self,
        session_start_valence: float,
        final_valence: float,
        feedback_history: List[float],
    ) -> float:
        """
        会话级代理评分 — 用于会话结束时无 t+1 轮输入的 fallback.
        Session-Level Surrogate — fallback when no t+1 turn input is available.

        feedback_fallback = clamp(
            0.40 · Δvalence_session + 0.30 · f̄_session + 0.30 · explicit_last,
            0, 1
        )

        其中:
          - Δvalence_session = clamp(valence(t) - valence(t0), -1, 1)
          - f̄_session = 该会话内已计算的 feedback_score 均值 (单轮会话取 0.5)
          - explicit_last = 末轮显式反馈 (此处简化取 0.5)

        Args:
            session_start_valence: 会话首轮效价 valence(t0)
            final_valence: 会话末轮效价 valence(t)
            feedback_history: 本会话内已计算的 feedback_score 列表

        Returns:
            session surrogate score ∈ [0, 1]
        """
        # Δvalence_session ∈ [-1, 1] — raw directional signal per spec §3.5.2
        # Negative = user worsened, positive = improved. Final clamp handles [0,1] output.
        delta_valence = _clamp(final_valence - session_start_valence, -1.0, 1.0)

        # f̄_session: 会话内 feedback_score 均值
        # 单轮会话无可用值 → 默认 0.5
        if feedback_history:
            f_bar = float(np.mean(feedback_history))
        else:
            f_bar = 0.5

        # explicit_last: 末轮显式反馈 (简化取 0.5)
        explicit_last = 0.5

        raw_score = (
            SURROGATE_DELTA_VALENCE_W * delta_valence    # 0.40
            + SURROGATE_F_BAR_W * f_bar                  # 0.30
            + SURROGATE_EXPLICIT_LAST_W * explicit_last  # 0.30
        )
        return _clamp(raw_score, 0.0, 1.0)

    # ══════════════════════════════════════════════
    # 会话结束处理 (Session End Processing)
    # ══════════════════════════════════════════════

    def process_session_end(
        self,
        session_start_valence: float,
        final_valence: float,
        feedback_history: List[float],
    ):
        """
        处理会话结束时剩余的延迟更新.
        Process remaining deferred updates at session end.

        使用 session-level surrogate 替代缺失的 t+1 轮反馈,
        并以 ξ = 0.5 衰减系数标记为会话末更新.

        Args:
            session_start_valence: 会话首轮效价
            final_valence: 会话末轮效价
            feedback_history: 本会话内已计算的 feedback_score 列表
        """
        # 计算会话级代理评分
        surrogate_score = self.compute_session_surrogate(
            session_start_valence, final_valence, feedback_history
        )

        for deferred in self.deferred_update_queue:
            # 使用 surrogate 分数, 并标记为 session_end (ξ = 0.5 衰减)
            self.update_beta(
                strategy=deferred.strategy,
                context_key=deferred.context_key,
                feedback_score=surrogate_score,
                is_session_end=True,
            )
            # 填充 episode 的 feedback_score
            for ep in self.episodic_memory:
                if ep.id == deferred.episode_id:
                    ep.user_feedback_score = surrogate_score
                    break

        self.deferred_update_queue.clear()

    # ══════════════════════════════════════════════
    # Beta 分布更新 (Beta Update) — Definition 3.3
    # ══════════════════════════════════════════════

    def update_beta(
        self,
        strategy: str,
        context_key: str,
        feedback_score: float,
        is_session_end: bool = False,
    ):
        """
        定义 3.3: Beta 分布后验更新 (策略有效性矩阵).
        Definition 3.3: Beta posterior update for the strategy effectiveness matrix.

        更新规则 (启发式连续奖励扩展):
          if fs > 0.5:  α += 2·(fs - 0.5)
          if fs ≤ 0.5:  β += 2·(0.5 - fs)

        会话末轮衰减 (ξ = 0.5):
          当 is_session_end=True 时, 更新幅度乘以 SESSION_END_DECAY = 0.5

        时间衰减机制 (在加载时执行):
          α' = α₀ + γ_β(Δt) · (α - α₀)
          β' = β₀ + γ_β(Δt) · (β - β₀)
          γ_β(Δt) = exp(-ln2 · Δt / τ_β), τ_β = 60 天

        Args:
            strategy: 策略名称
            context_key: 情境键 (phase|severity|intent_group)
            feedback_score: 反馈评分 ∈ [0, 1]
            is_session_end: 是否为会话末更新 (应用 ξ 衰减)
        """
        key = (strategy, context_key)

        # 获取或创建 Beta 参数
        # Get or create Beta parameters for this (strategy, context) pair
        if key not in self.affective_memory.strategy_matrix:
            self.affective_memory.strategy_matrix[key] = BetaParams(
                alpha=BETA_ALPHA0,
                beta=BETA_BETA0,
                last_updated=datetime.now(),
                observation_count=0,
            )

        bp = self.affective_memory.strategy_matrix[key]

        # NOTE (H2 fix): Time decay removed from update_beta().
        # Time decay is applied ONLY at sampling time in planning.py select_strategy()
        # (Step 1, lines 776-783). Applying decay here AND at sampling caused double
        # erosion of learned Beta parameters, preventing strategy convergence.
        # update_beta() should apply reward update to CURRENT parameters, not re-decay.
        now = datetime.now()

        # ── 计算更新幅度 ──
        # Compute update magnitude
        fs = feedback_score
        decay_factor = SESSION_END_DECAY if is_session_end else 1.0

        if fs > 0.5:
            # 正向反馈 → 增加 α (增加有效性估计)
            increment = decay_factor * 2.0 * (fs - 0.5)
            bp.alpha += increment
        else:
            # 负向反馈 → 增加 β (降低有效性估计)
            increment = decay_factor * 2.0 * (0.5 - fs)
            bp.beta += increment

        bp.last_updated = now
        bp.observation_count += 1

    # ══════════════════════════════════════════════
    # 情绪基线更新 — §3.5.1
    # ══════════════════════════════════════════════

    def update_emotion_baseline(self, emotion_vector: EmotionVector, adaptive_baseline_lr: Optional[float] = None):
        """
        §3.5.1: 情绪基线更新 (EMA + 2σ 异常值排除).
        §3.5.1: Emotion baseline update (EMA with 2σ outlier exclusion).

        流程:
          1. 计算当前观测偏差距离 δ(t) = ||e(t) - e_baseline||_2
          2. 无条件更新 σ_baseline (EMA, 下界保护 σ_min = 0.05)
          3. 无条件将 e(t) 写入环形缓冲区
          4. 如果 δ(t) < 2σ: 更新基线 (EMA, η = adaptive or 0.05), 增加置信度 (+Δc)
             否则: 不更新基线, 降低置信度 (-Δc_decay)
          5. 如果 confidence < 0.3: 触发基线重估 (从环形缓冲区重建)

        Args:
            emotion_vector: 当前轮的情绪向量
            adaptive_baseline_lr: 自适应学习率 (None = use static BASELINE_LR)
        """
        baseline = self.affective_memory.emotion_baseline
        e_current = emotion_vector.e  # 8 维情绪分布
        e_baseline = baseline.normal_state

        # ── Step 1: 计算偏差距离 δ(t) ──
        delta = float(np.linalg.norm(e_current - e_baseline))

        # ── Step 2: 无条件更新 σ_baseline ──
        # σ(t+1) = max(σ_min, sqrt(η_σ · δ² + (1-η_σ) · σ²))
        # 使用 EMA 更新, 避免存储完整历史
        # Unconditional σ update via EMA to avoid storing full history
        new_sigma_sq = BASELINE_SIGMA_LR * (delta ** 2) + (1 - BASELINE_SIGMA_LR) * (baseline.sigma_baseline ** 2)
        baseline.sigma_baseline = max(BASELINE_SIGMA_MIN, math.sqrt(new_sigma_sq))

        # ── Step 3: 无条件写入环形缓冲区 ──
        # Ring buffer: 最近 K_buf=20 次情绪向量, 用于重估
        baseline.ring_buffer.append(e_current.copy())

        # ── Step 4: 条件更新基线 ──
        threshold = 2.0 * baseline.sigma_baseline

        if delta < threshold:
            # NaN 防护: 防止损坏的情绪向量永久污染基线
            if np.isnan(e_current).any():
                logger.warning("NaN in e_current, skipping baseline update")
                return
            # 正常观测: 更新基线 (EMA) + 增加置信度
            # Normal observation: update baseline via EMA and increment confidence
            # Use adaptive LR when available
            lr = adaptive_baseline_lr if adaptive_baseline_lr is not None else BASELINE_LR
            baseline.normal_state = (
                lr * e_current + (1 - lr) * e_baseline
            )
            # 重新归一化到单纯形 (确保分布合法)
            # Re-normalize to simplex (ensure valid distribution)
            total = baseline.normal_state.sum()
            if total > EPSILON:
                baseline.normal_state /= total

            baseline.confidence = min(1.0, baseline.confidence + BASELINE_CONFIDENCE_INCREMENT)
            baseline.sample_count += 1
        else:
            # 异常观测: 不更新基线, 衰减置信度
            # Outlier: do not update baseline, decay confidence
            baseline.confidence = max(0.0, baseline.confidence - BASELINE_CONFIDENCE_DECAY)

        baseline.last_updated = datetime.now()

        # ── Step 5: 置信度过低时触发基线重估 ──
        # When confidence drops below threshold, reset baseline from ring buffer
        if baseline.confidence < BASELINE_RESET_THRESHOLD:
            self._reset_baseline_from_buffer()

    def _reset_baseline_from_buffer(self):
        """
        基线重估: 从环形缓冲区的逐维中位数重建基线.
        Baseline reset: rebuild from per-dimension median of ring buffer.

        重置项:
          - normal_state = 缓冲区逐维中位数 (归一化)
          - confidence = 0.5
          - σ_baseline = 缓冲区偏差距离的经验标准差
        """
        baseline = self.affective_memory.emotion_baseline
        if len(baseline.ring_buffer) < 3:
            # 缓冲区样本不足, 无法重估 → 回退到冷启动
            # Insufficient samples in buffer → fallback to cold start
            baseline.normal_state = BASELINE_COLD_START.copy()
            baseline.confidence = BASELINE_CONFIDENCE_INIT
            return

        # 逐维中位数 / Per-dimension median
        buffer_array = np.array(list(baseline.ring_buffer))  # shape: (N, 8)
        median_state = np.median(buffer_array, axis=0)

        # 归一化到单纯形 / Normalize to simplex
        total = median_state.sum()
        if total > EPSILON:
            median_state /= total
        else:
            # All-zero buffer: fallback to cold start
            median_state = BASELINE_COLD_START.copy()

        baseline.normal_state = median_state

        # σ 重置为缓冲区偏差距离的经验标准差
        # Reset σ to empirical std of deviations in buffer
        deviations = [
            float(np.linalg.norm(obs - median_state))
            for obs in buffer_array
        ]
        if len(deviations) > 1:
            baseline.sigma_baseline = max(BASELINE_SIGMA_MIN, float(np.std(deviations)))
        else:
            baseline.sigma_baseline = BASELINE_SIGMA_MIN

        # 置信度重置 / Confidence reset
        baseline.confidence = 0.5
        baseline.last_updated = datetime.now()

    # ══════════════════════════════════════════════
    # 行为基线更新 — §2.5
    # ══════════════════════════════════════════════

    def update_behavioral_baseline(self, behavior_features: Dict[str, float]):
        """
        §2.5: 行为基线更新 (EMA 更新均值和标准差).
        §2.5: Behavioral baseline update (EMA for mean and std).

        对每个行为维度:
          mean_new = λ · x + (1-λ) · mean_old,  λ = 0.1
          std_new  = max(σ_min, sqrt(λ_σ · (x - mean)² + (1-λ_σ) · std²)),  λ_σ = 0.1

        Args:
            behavior_features: 当前轮的行为特征字典
                例: {"msg_time": 14.0, "reply_interval": 120.0, ...}
        """
        bb = self.semantic_memory.user_profile.behavioral_baseline

        for dim, value in behavior_features.items():
            if dim not in bb.mean:
                continue

            old_mean = bb.mean[dim]
            old_std = bb.std[dim]
            sigma_min = BEHAVIOR_SIGMA_MIN.get(dim, 0.01)

            # 均值 EMA 更新 / Mean EMA update
            new_mean = BEHAVIOR_BASELINE_LR * value + (1 - BEHAVIOR_BASELINE_LR) * old_mean
            bb.mean[dim] = new_mean

            # 标准差 EMA 更新 (下界保护) / Std EMA update with floor
            deviation_sq = (value - old_mean) ** 2
            new_std_sq = BEHAVIOR_BASELINE_SIGMA_LR * deviation_sq + (1 - BEHAVIOR_BASELINE_SIGMA_LR) * (old_std ** 2)
            bb.std[dim] = max(sigma_min, math.sqrt(new_std_sq))

    # ══════════════════════════════════════════════
    # 记忆操作二: 心境自适应检索 MAR — Definition 3.5/3.6
    # ══════════════════════════════════════════════

    def read_mar(
        self,
        state_vector: StateVector,
        planning_intent: str = "default",
    ) -> RetrievedContext:
        """
        定义 3.5/3.6: 心境自适应检索 (Mood-Adaptive Retrieval) ★核心创新二.
        Definition 3.5/3.6: Mood-Adaptive Retrieval (MAR).

        检索评分:
          Score(m, q, t) = α'·Sim + β'·EmoCon + γ'·Rec + ω'·Imp

        动态 β 调整 (Definition 3.6):
          β(t) = β_base · Φ(planning_intent)
          权重归一化以保证概率一致性

        各分量:
          - Sim: 文本语义相似度 (Mock: 关键词重叠率)
          - EmoCon: 情绪一致性 = 1 - ||e_query - e_memory||_2 / √2
          - Rec: 时间近因性 = exp(-λ_r · Δt / τ_r)
          - Imp: 记忆重要性 = episode.importance

        返回 top-K 个最相关的 episode + 语义记忆 + 情感记忆.

        Args:
            state_vector: 当前状态向量 (查询)
            planning_intent: 规划意图 (用于动态 β 调整)

        Returns:
            RetrievedContext: 包含 episodes, semantic, affective 数据
        """
        if not self.episodic_memory:
            # 情景记忆为空 → 返回空上下文 + 语义/情感记忆
            return RetrievedContext(
                episodes=[],
                semantic=self.semantic_memory,
                affective=self.affective_memory,
                grounding_facts=[],
            )

        # ── 动态 β 调整 (Definition 3.6b) ──
        # β(t) = β_base · Φ(planning_intent)
        phi = PLANNING_INTENT_PHI.get(planning_intent, 1.0)
        beta_adjusted = MAR_BETA_BASE * phi

        # 权重归一化 / Weight normalization
        raw_sum = MAR_ALPHA_BASE + beta_adjusted + MAR_GAMMA_BASE + MAR_OMEGA_BASE
        if raw_sum < 1e-8:
            alpha_n = beta_n = gamma_n = omega_n = 0.25
        else:
            alpha_n = MAR_ALPHA_BASE / raw_sum
            beta_n = beta_adjusted / raw_sum
            gamma_n = MAR_GAMMA_BASE / raw_sum
            omega_n = MAR_OMEGA_BASE / raw_sum

        # ── 构造查询 ──
        query_text = state_vector.raw_signal.text_content if state_vector.raw_signal else ""
        query_emotion = state_vector.emotion.e
        now = datetime.now()

        # ── 对每个 episode 计算 MAR 评分 ──
        scored_episodes: List[Tuple[float, EpisodeUnit]] = []

        for episode in self.episodic_memory:
            # Sim: 语义相似度 (Mock: 关键词重叠)
            # TODO: 在检索循环中逐条调用 LLM 语义相似度代价过高,
            # 未来应替换为 embedding 向量余弦相似度 (e.g. text-embedding-3-small)
            # 可使用 _llm_semantic_similarity() 对 top-K 候选做精排 (re-ranking)
            sim = _keyword_overlap(query_text, episode.event)

            # EmoCon: 情绪一致性
            emo_con = _emotion_congruence(query_emotion, episode.emotion_snapshot.e)

            # Rec: 时间近因性
            # Rec = exp(-λ_r · Δt / τ_r), λ_r = ln2, τ_r = 30 天
            delta_t_seconds = (now - episode.timestamp).total_seconds()
            delta_t_days = max(0.0, delta_t_seconds / 86400.0)
            rec = math.exp(-MAR_LAMBDA_R * delta_t_days / MAR_TAU_R_DAYS)

            # Imp: 重要性
            imp = episode.importance

            # 加权评分 / Weighted score
            score = alpha_n * sim + beta_n * emo_con + gamma_n * rec + omega_n * imp

            scored_episodes.append((score, episode))

        # ── 排序并取 top-K ──
        scored_episodes.sort(key=lambda x: x[0], reverse=True)
        top_k = [ep for _, ep in scored_episodes[:MAR_TOP_K]]

        # ── 构造 grounding facts ──
        # 提取可用于 Agent 回应的锚定事实
        grounding_facts = []
        for ep in top_k:
            if ep.event:
                grounding_facts.append(
                    f"[{ep.timestamp.strftime('%Y-%m-%d')}] {ep.event} "
                    f"(emotion_valence={ep.emotional_valence:.2f})"
                )

        return RetrievedContext(
            episodes=top_k,
            semantic=self.semantic_memory,
            affective=self.affective_memory,
            grounding_facts=grounding_facts,
        )

    def read_mar_fixed(self, state_vector: StateVector) -> RetrievedContext:
        """
        固定权重记忆检索 (消融基线).
        Uses constant base weights instead of mood-adaptive retrieval.
        No planning_intent-based beta adjustment — weights are static.
        """
        if not self.episodic_memory:
            return RetrievedContext(
                episodes=[],
                semantic=self.semantic_memory,
                affective=self.affective_memory,
                grounding_facts=[],
            )

        # 固定权重归一化 (无 Φ 调整)
        raw_sum = MAR_ALPHA_BASE + MAR_BETA_BASE + MAR_GAMMA_BASE + MAR_OMEGA_BASE
        if raw_sum < 1e-8:
            alpha_n = beta_n = gamma_n = omega_n = 0.25
        else:
            alpha_n = MAR_ALPHA_BASE / raw_sum
            beta_n = MAR_BETA_BASE / raw_sum
            gamma_n = MAR_GAMMA_BASE / raw_sum
            omega_n = MAR_OMEGA_BASE / raw_sum

        query_text = state_vector.raw_signal.text_content if state_vector.raw_signal else ""
        query_emotion = state_vector.emotion.e
        now = datetime.now()

        scored_episodes: List[Tuple[float, EpisodeUnit]] = []
        for episode in self.episodic_memory:
            sim = _keyword_overlap(query_text, episode.event)
            emo_con = _emotion_congruence(query_emotion, episode.emotion_snapshot.e)
            delta_t_seconds = (now - episode.timestamp).total_seconds()
            delta_t_days = max(0.0, delta_t_seconds / 86400.0)
            rec = math.exp(-MAR_LAMBDA_R * delta_t_days / MAR_TAU_R_DAYS)
            imp = episode.importance
            score = alpha_n * sim + beta_n * emo_con + gamma_n * rec + omega_n * imp
            scored_episodes.append((score, episode))

        scored_episodes.sort(key=lambda x: x[0], reverse=True)
        top_k = [ep for _, ep in scored_episodes[:MAR_TOP_K]]

        grounding_facts = []
        for ep in top_k:
            if ep.event:
                grounding_facts.append(
                    f"[{ep.timestamp.strftime('%Y-%m-%d')}] {ep.event} "
                    f"(emotion_valence={ep.emotional_valence:.2f})"
                )

        return RetrievedContext(
            episodes=top_k,
            semantic=self.semantic_memory,
            affective=self.affective_memory,
            grounding_facts=grounding_facts,
        )

    # ══════════════════════════════════════════════
    # 记忆操作三: 反思整合 (Reflect) — Algorithm 3.2
    # ══════════════════════════════════════════════

    def reflect(self):
        """
        算法 3.2: 反思整合 (Reflective Consolidation) — 简化 Mock 版.
        Algorithm 3.2: Reflective Consolidation (simplified mock).

        触发时机: 会话结束后异步执行 (类似人类睡眠期间的记忆巩固).

        三个步骤:
          R-Step 1: 情景 → 语义 整合
            从近期 episodes 中提取模式, 更新语义记忆
          R-Step 2: 情景 → 情感 整合
            更新触发-情绪关联图和恢复模式
          R-Step 3: 前瞻性规划 (受 RMM 启发)
            提取未来事件, 设置主动跟进触发 (Mock)
        """
        if not self.episodic_memory:
            return

        # 取最近 10 个 episodes 进行反思
        # Use the most recent 10 episodes for reflection
        recent = self.episodic_memory[-10:]

        # ── R-Step 1: 情景 → 语义 整合 ──
        # Extract patterns into semantic memory
        self._reflect_semantic(recent)

        # ── R-Step 2: 情景 → 情感 整合 ──
        # Update trigger map and recovery patterns
        self._reflect_affective(recent)

        # ── R-Step 3: 前瞻性规划 (Mock) ──
        # Proactive follow-up triggers (not implemented in mock)

    def _reflect_semantic(self, recent_episodes: List[EpisodeUnit]):
        """
        R-Step 1: 从近期 episodes 提取主题模式到语义记忆.
        Extract topic patterns from recent episodes into semantic memory.

        简化 Mock:
          - 提取反复出现的 topic_category → recurring_themes
          - 提取涉及的人物 → relationship_graph (仅添加条目, 不深入分析)
        """
        # 统计主题频率 / Count topic frequencies
        topic_counts: Dict[str, int] = {}
        for ep in recent_episodes:
            if ep.topic_category:
                topic_counts[ep.topic_category] = topic_counts.get(ep.topic_category, 0) + 1

        # 高频主题加入 recurring_themes / Add frequent topics to recurring_themes
        for topic, count in topic_counts.items():
            if count >= 2 and topic not in self.semantic_memory.user_profile.recurring_themes:
                self.semantic_memory.user_profile.recurring_themes.append(topic)

        # 提取涉及的人物 / Extract people involved
        for ep in recent_episodes:
            for person in ep.people_involved:
                if person and person not in self.semantic_memory.relationship_graph:
                    from .models import RelationshipInfo
                    self.semantic_memory.relationship_graph[person] = RelationshipInfo(
                        role="",
                        closeness=0.5,
                    )

    def _reflect_affective(self, recent_episodes: List[EpisodeUnit]):
        """
        R-Step 2: 从近期 episodes 更新触发-情绪关联图.
        Update trigger-emotion association map from recent episodes.

        对每个 episode:
          - 如果 topic_category 已在 trigger_map 中: 贝叶斯更新情绪分布
          - 否则: 初始化新的 EmotionPattern
        """
        for ep in recent_episodes:
            topic = ep.topic_category
            if not topic:
                continue

            if topic in self.affective_memory.trigger_map:
                # 贝叶斯更新: 用当前观测的情绪分布更新已有模式
                # Bayesian update: blend current emotion with existing pattern
                pattern = self.affective_memory.trigger_map[topic]
                n = pattern.observation_count
                # 增量均值更新: p_new = (n · p_old + e_current) / (n + 1)
                new_emotions = (
                    (n * pattern.triggered_emotions + ep.emotion_snapshot.e) / (n + 1)
                )
                if np.isnan(new_emotions).any():
                    logger.warning("NaN detected in affective memory emotion blending, skipping update")
                else:
                    # 重归一化到概率单纯形 (防止浮点累积漂移)
                    em_total = new_emotions.sum()
                    if em_total > 1e-8:
                        new_emotions = new_emotions / em_total
                    pattern.triggered_emotions = new_emotions
                # 更新强度范围 / Update intensity range
                lo, hi = pattern.intensity_range
                ι = ep.emotion_snapshot.intensity
                pattern.intensity_range = (min(lo, ι), max(hi, ι))
                pattern.observation_count = n + 1
                # 置信度随观测增加 / Confidence grows with observations
                pattern.confidence = min(1.0, 1.0 - 1.0 / (pattern.observation_count + 1))
            else:
                # 初始化新模式 / Initialize new pattern
                self.affective_memory.trigger_map[topic] = EmotionPattern(
                    triggered_emotions=ep.emotion_snapshot.e.copy(),
                    intensity_range=(ep.emotion_snapshot.intensity, ep.emotion_snapshot.intensity),
                    observation_count=1,
                    confidence=0.3,
                )

    # ══════════════════════════════════════════════
    # 记忆操作四: 遗忘 (Forget) — Definition 3.7
    # ══════════════════════════════════════════════

    def forget_check(self):
        """
        定义 3.7: 记忆保留度检查与归档.
        Definition 3.7: Memory retention check and archival.

        Retention(m, t) = Importance(m) · exp_decay · EmoProt(m)

        其中:
          exp_decay = exp(-ln2 · (t - m.timestamp) / τ_f), τ_f = 90 天
          EmoProt(m) = 1.0                              if arousal < 0.5
                     = 1.0 + ρ · (arousal - 0.5)        if arousal ≥ 0.5
          ρ = 3.0, 最大 EmoProt = 2.5 (arousal=1.0 时)

        当 Retention < 0.1 时, 移入归档区 (不删除).
        归档记忆在用户主动提及时仍可被召回.
        """
        now = datetime.now()
        surviving: List[EpisodeUnit] = []

        for episode in self.episodic_memory:
            # ── 计算时间衰减 exp_decay ──
            delta_t_days = (now - episode.timestamp).total_seconds() / 86400.0
            exp_decay = math.exp(-math.log(2) * delta_t_days / TAU_FORGET_DAYS)

            # ── 计算情感保护 EmoProt ──
            arousal = episode.emotion_snapshot.intensity
            if arousal < 0.5:
                emo_prot = 1.0
            else:
                emo_prot = 1.0 + EMOTION_PROTECTION_RHO * (arousal - 0.5)

            # ── 计算保留度 Retention ──
            retention = episode.importance * exp_decay * emo_prot

            if retention < RETENTION_THRESHOLD:
                # 低保留度 → 移入归档区 / Low retention → archive
                self.archived_episodes.append(episode)
            else:
                surviving.append(episode)

        self.episodic_memory = surviving

    # ══════════════════════════════════════════════
    # 记忆操作五: 更新 (Update) — §3.10
    # ══════════════════════════════════════════════

    def update_episode(
        self,
        existing_episode_id: str,
        new_info: StateVector,
        update_type: str = "revision",
    ) -> Optional[EpisodeUnit]:
        """
        §3.10: 修订式更新 — 不删除旧记忆, 标记为历史版本.
        §3.10: Revision-style update — mark old memory as superseded, not deleted.

        保留记忆演化轨迹: Agent 需要知道完整叙事.
        例如 "用户之前分手过, 后来复合了" 而非仅当前状态.

        Args:
            existing_episode_id: 被修订的旧 episode ID
            new_info: 新信息的状态向量
            update_type: 更新类型 ("revision" | "resolution" | "escalation")

        Returns:
            新创建的 EpisodeUnit, 如果找到旧 episode; 否则 None
        """
        # 查找旧 episode / Find the old episode
        old_episode: Optional[EpisodeUnit] = None
        for ep in self.episodic_memory:
            if ep.id == existing_episode_id:
                old_episode = ep
                break

        if old_episode is None:
            return None

        # 标记旧记忆为历史版本 (使用 resolution_status 字段标记)
        # Mark old memory as superseded
        old_episode.resolution_status = ResolutionStatus.RESOLVED

        # 创建新版本 / Create new version
        new_episode = self._construct_episode(new_info, agent_strategy="")
        new_episode.importance = self.compute_importance(new_episode)
        new_episode.topic_category = old_episode.topic_category  # 保留话题分类

        self.episodic_memory.append(new_episode)
        return new_episode

    # ══════════════════════════════════════════════
    # Pre-Planning Intent Estimation — Definition 3.6a
    # ══════════════════════════════════════════════

    @staticmethod
    def estimate_planning_intent(state_vector: StateVector) -> str:
        """
        定义 3.6a: 预规划意图推断 (Pre-Planning Intent Estimation).
        Definition 3.6a: Pre-Planning Intent Estimation.

        基于 StateVector 中的意图分布和紧急度, 通过确定性规则快速推断 planning_intent.
        此映射仅使用感知模块已输出的 IntentDistribution 和 urgency,
        不依赖任何后续模块输出, 从而打破循环依赖.
        计算复杂度 O(1).

        映射规则 (按优先级排序):
          1. urgency > 0.9                        → crisis_response
          2. P(REFLECT) > 0.4 或 P(ADVICE) > 0.4 → pattern_reflection
          3. P(SHARE_JOY) > 0.5                   → positive_reframing (映射为 default 中的特殊值)
          4. P(VENT) + P(COMFORT) > 0.6           → emotional_validation
          5. 否则                                   → default

        Args:
            state_vector: 感知模块输出的状态向量

        Returns:
            planning_intent 字符串
        """
        intent = state_vector.intent
        urgency = state_vector.urgency_level

        # 优先级 1: 危机信号
        if urgency > 0.9:
            return "crisis_response"

        # 优先级 2: 用户主动寻求深度理解
        if intent.p("REFLECT") > 0.4 or intent.p("ADVICE") > 0.4:
            return "pattern_reflection"

        # 优先级 3: 积极情境 (Φ=0.1, 抑制负面情绪一致检索)
        if intent.p("SHARE_JOY") > 0.5:
            return "positive_reframing"

        # 优先级 4: 情感支持需求
        if intent.p("VENT") + intent.p("COMFORT") > 0.6:
            return "emotional_validation"

        # 默认
        return "default"

    # ══════════════════════════════════════════════
    # 工具方法 (Utility Methods)
    # ══════════════════════════════════════════════

    def get_episode_by_id(self, episode_id: str) -> Optional[EpisodeUnit]:
        """
        按 ID 查找情景记忆 (含归档区).
        Find episode by ID (including archive).
        """
        for ep in self.episodic_memory:
            if ep.id == episode_id:
                return ep
        for ep in self.archived_episodes:
            if ep.id == episode_id:
                return ep
        return None

    def get_recent_episodes(self, n: int = 5) -> List[EpisodeUnit]:
        """
        获取最近 n 个情景记忆.
        Get the most recent n episodes.
        """
        return self.episodic_memory[-n:]

    def get_beta_params(self, strategy: str, context_key: str) -> BetaParams:
        """
        获取指定 (策略, 情境) 对的 Beta 参数 (不存在则创建).
        Get Beta parameters for (strategy, context) pair (create if not exists).
        """
        key = (strategy, context_key)
        if key not in self.affective_memory.strategy_matrix:
            self.affective_memory.strategy_matrix[key] = BetaParams(
                alpha=BETA_ALPHA0,
                beta=BETA_BETA0,
                last_updated=datetime.now(),
                observation_count=0,
            )
        return self.affective_memory.strategy_matrix[key]

    def reset_working_memory(self):
        """
        重置工作记忆 (新会话开始时调用).
        Reset working memory (called at the start of a new session).
        """
        self.working_memory = WorkingMemory()

    @property
    def episode_count(self) -> int:
        """情景记忆总数 (不含归档) / Episodic memory count (excluding archived)."""
        return len(self.episodic_memory)

    @property
    def archived_count(self) -> int:
        """归档记忆总数 / Archived memory count."""
        return len(self.archived_episodes)

    def summary(self) -> Dict[str, Any]:
        """
        记忆模块状态摘要 (用于调试).
        Memory module status summary (for debugging).
        """
        baseline = self.affective_memory.emotion_baseline
        return {
            "working_memory_turns": self.working_memory.turn_count,
            "episodic_count": self.episode_count,
            "archived_count": self.archived_count,
            "semantic_themes": self.semantic_memory.user_profile.recurring_themes,
            "semantic_relationships": list(self.semantic_memory.relationship_graph.keys()),
            "affective_baseline_confidence": baseline.confidence,
            "affective_baseline_sigma": baseline.sigma_baseline,
            "affective_trigger_map_size": len(self.affective_memory.trigger_map),
            "affective_strategy_matrix_size": len(self.affective_memory.strategy_matrix),
            "deferred_queue_size": len(self.deferred_update_queue),
            "ring_buffer_size": len(baseline.ring_buffer),
        }

    # ══════════════════════════════════════════════
    # 自适应参数辅助方法 (Adaptive Parameter Helpers)
    # ══════════════════════════════════════════════

    def compute_emotion_variance(self) -> float:
        """
        Compute emotional variance from ring buffer for adaptive baseline LR.

        Uses valence (σ^T · e) of recent emotion observations.
        Returns 0.0 when insufficient data (<3 observations).
        """
        buf = self.affective_memory.emotion_baseline.ring_buffer
        if len(buf) < 3:
            return 0.0
        valences = []
        for item in buf:
            if isinstance(item, np.ndarray):
                valences.append(float(np.dot(VALENCE_SIGNS, item)))
            else:
                valences.append(float(item))
        return float(np.var(valences))

    def compute_emotion_autocorrelation(self) -> Optional[float]:
        """
        Compute lag-1 autocorrelation of valence for adaptive inertia.

        Returns None when insufficient data (<4 observations) or zero variance,
        so that the adaptive framework can distinguish "no data" from
        "genuinely zero autocorrelation".
        """
        buf = self.affective_memory.emotion_baseline.ring_buffer
        if len(buf) < 4:
            return None
        valences = []
        for item in buf:
            if isinstance(item, np.ndarray):
                valences.append(float(np.dot(VALENCE_SIGNS, item)))
            else:
                valences.append(float(item))
        arr = np.array(valences)
        mean_v = np.mean(arr)
        var_v = np.var(arr)
        if var_v < 1e-10:
            return None
        # Lag-1 autocorrelation: r = E[(v_t - μ)(v_{t-1} - μ)] / var
        n = len(arr)
        cov = np.mean((arr[1:] - mean_v) * (arr[:-1] - mean_v))
        return float(np.clip(cov / var_v, -1.0, 1.0))

    def get_severity_history(self) -> List[float]:
        """
        Extract severity history from episodic memory for adaptive severity bins.

        Uses emotional intensity as proxy for severity (consistent with §4.2).
        """
        history = []
        for ep in self.episodic_memory:
            # Use emotion intensity as severity proxy
            history.append(ep.emotion_snapshot.intensity)
        return history

    def compute_strategy_effectiveness(self) -> Dict[str, List[float]]:
        """
        Compute empirical strategy effectiveness for adaptive TS priors.

        Returns {strategy: [feedback_scores]} from all episodes with feedback.
        """
        result: Dict[str, List[float]] = {}
        for ep in self.episodic_memory:
            if ep.agent_strategy and ep.user_feedback_score is not None:
                if ep.agent_strategy not in result:
                    result[ep.agent_strategy] = []
                result[ep.agent_strategy].append(ep.user_feedback_score)
        return result

    def get_interaction_count(self) -> int:
        """Total interaction count (episodic memory size)."""
        return len(self.episodic_memory)
