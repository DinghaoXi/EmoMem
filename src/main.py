"""
EmoMem 主对话循环 — PPAM 管线编排器
======================================
EmoMem Main Conversation Loop — PPAM Pipeline Orchestrator

本模块实现 EmoMem Agent 的完整 PPAM (Perception-Planning-Action-Memory) 管线：
  Phase 1:   感知 (Perception)      — 解析用户输入为 StateVector
  Phase 1.5: 预规划意图估计          — 确定 planning_intent (打破循环依赖)
  Phase 2:   记忆读取 (Memory Read)  — MAR 心境自适应检索
  Phase 3:   规划 (Planning)         — 轨迹分析 → 情境评估 → 目标推断 → Thompson Sampling
  Phase 4:   行动 (Action)           — 生成自然语言回复
  Phase 5:   记忆写入 (Memory Write) — 写入情景记忆 + 延迟 Beta 更新入队

特殊路径 (§6.2):
  - 危机快速通道: urgency > 0.9 → 跳过 Phase 1.5/3, 直接 Crisis Handler
  - 首轮处理: 无 deferred updates, 冷启动情绪基线

运行方式:
  cd /Users/dinghao/Downloads/emomem
  python -m src.main
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

from .config import (
    CRISIS_THRESHOLD,
    STRATEGIES,
    EMOTIONS,
    VALENCE_SIGNS,
    RELATIONSHIP_DEPTH_LR,
    DISCLOSURE_LEVELS,
)
from .models import (
    StateVector,
    EmotionVector,
    WorkingMemory,
    AffectiveMemory,
    SemanticMemory,
    RetrievedContext,
    PlanningOutput,
    RecoveryPhase,
    TrajectoryResult,
    DeferredUpdate,
    BehavioralBaseline,
    IntentDistribution,
)
from .perception import PerceptionModule
from .memory import MemoryModule
from .planning import PlanningModule, determine_planning_intent
from .action import ActionModule
from .mock_llm import MockLLM
from .llm_provider import create_provider, BaseLLMProvider, MockProvider
from .adaptation import AdaptiveParameterManager, AdaptiveParams, UserStats


# ──────────────────────────────────────────────
# LLM 输出质量检测 (Garbled output detection)
# ──────────────────────────────────────────────

# 检测 LLM 生成的乱码/重复文本 (如 "我我我我我我...")
# 理论背景: Holtzman et al. (2019) "The Curious Case of Neural Text Degeneration"
# 重复退化是 autoregressive LM 的已知失效模式, 需在应用层拦截
_REPEAT_PATTERN = re.compile(r'(.)\1{9,}')  # 同一字符连续出现≥10次


def _is_garbled_response(text: str) -> bool:
    """
    检测 LLM 输出是否为乱码/退化文本.

    判定条件 (满足任一即为乱码):
      1. 存在单字符连续重复≥10次 (如 "我我我我我我我我我我")
      2. 文本≥20字且去重后字符种类 < 文本长度的10% (极低多样性)

    Returns:
        True if the response is garbled and should be retried/replaced.
    """
    if not text or len(text) < 5:
        return False
    # 条件1: 单字符重复退化
    if _REPEAT_PATTERN.search(text):
        return True
    # 条件2: 字符多样性过低 (排除标点和空格)
    content_chars = [c for c in text if not c.isspace() and c not in '，。！？、；：""''（）…—·']
    if len(content_chars) >= 20:
        unique_ratio = len(set(content_chars)) / len(content_chars)
        if unique_ratio < 0.10:
            return True
    return False


# ──────────────────────────────────────────────
# 诊断信息格式化 (Diagnostic display helpers)
# ──────────────────────────────────────────────

def _format_emotion_bar(emotion: EmotionVector) -> str:
    """
    格式化情绪向量为可读的诊断字符串。
    Format the emotion vector into a readable diagnostic string.
    """
    lines = []
    for i, emo_name in enumerate(EMOTIONS):
        val = emotion.e[i]
        # 防御 NaN/Inf: 确保 bar_len 为合法非负整数
        if np.isnan(val) or np.isinf(val):
            val = 0.0
        bar_len = max(0, min(20, int(val * 20)))
        bar = "█" * bar_len + "░" * (20 - bar_len)
        lines.append(f"  {emo_name:14s} {bar} {val:.3f}")
    return "\n".join(lines)


def _format_diagnostics(
    state_vector: StateVector,
    planning_output: Optional[PlanningOutput],
    turn: int,
    n_cum: int,
) -> str:
    """
    格式化完整的诊断信息面板。
    Format the full diagnostic panel for debugging.
    """
    sv = state_vector
    emo = sv.emotion
    valence = emo.valence()

    parts = [
        f"\n{'='*60}",
        f"  Turn #{turn} (cumulative: {n_cum})",
        f"{'='*60}",
        f"  Valence:     {valence:+.3f}",
        f"  Intensity:   {emo.intensity:.3f}",
        f"  Confidence:  {emo.confidence:.3f}",
        f"  Urgency:     {sv.urgency_level:.3f}",
        f"  Top Intent:  {sv.intent.top_intent()}",
        f"  Topic:       {sv.topic_category or 'N/A'} ({', '.join(sv.topic_keywords) if sv.topic_keywords else 'none'})",
    ]

    # 情绪分布 / Emotion distribution
    parts.append(f"\n  Emotion Distribution:")
    parts.append(_format_emotion_bar(emo))

    # 规划信息 / Planning info
    if planning_output:
        po = planning_output
        if po.trajectory:
            parts.extend([
                f"\n  Recovery Phase:  {po.trajectory.current_phase.value}",
                f"  Phase Duration:  {po.trajectory.phase_duration}",
                f"  Direction:       {po.trajectory.direction.value}",
                f"  Delta_b:         {po.trajectory.delta_b:+.3f}",
            ])
        if po.appraisal:
            parts.append(f"  Severity:        {po.appraisal.severity:.3f}")
        parts.extend([
            f"  Strategy:        {po.selected_strategy}",
            f"  Planning Intent: {po.planning_intent}",
        ])
        if po.goals:
            parts.append(f"  Immediate Goal:  {po.goals.immediate_goal}")
    elif sv.urgency_level > CRISIS_THRESHOLD:
        parts.append(f"\n  *** CRISIS FAST-TRACK ACTIVATED ***")

    parts.append(f"{'='*60}")
    return "\n".join(parts)


# ──────────────────────────────────────────────
# 退出命令集 (Exit command set)
# ──────────────────────────────────────────────
EXIT_COMMANDS = {"exit", "quit", "bye", "再见", "退出", "结束"}

# 危机状态下的告别回复 (Crisis-aware session exit)
CRISIS_EXIT_RESPONSE = (
    "我注意到你现在可能正在经历很艰难的时刻。在你离开之前，我想让你知道：\n\n"
    "如果你需要帮助，请随时联系专业支持：\n"
    "  · 全国24小时心理援助热线：400-161-9995\n"
    "  · 北京心理危机研究与干预中心：010-82951332\n"
    "  · 生命热线：400-821-1215\n\n"
    "你随时可以回来找我聊天。照顾好自己。"
)


class EmoMemAgent:
    """
    EmoMem 情感陪伴 Agent — 主对话管理器。
    EmoMem Emotional Companion Agent — Main Conversation Manager.

    编排 PPAM 四模块协同工作，管理会话状态，
    处理特殊路径（危机快速通道、首轮冷启动）。

    Orchestrates the four PPAM modules, manages session state,
    and handles special paths (crisis fast-track, cold-start).

    Attributes:
        perception: 感知模块 (§2)
        memory:     记忆模块 (§3)
        planning:   规划模块 (§4)
        action:     行动模块 (§5)
        llm:        Mock LLM 服务
        n_cum:      累计交互轮次 (跨会话)
        turn:       当前会话轮次
        session_start_valence: 会话首轮效价 (用于会话级代理评分)
        feedback_history: 本会话已计算的 feedback_score 列表
    """

    def __init__(self):
        """
        初始化 EmoMem Agent — 创建所有模块实例并设置初始状态。
        Initialize EmoMem Agent — instantiate all modules and set initial state.
        """
        # ── 初始化 LLM 服务 ──
        # Initialize LLM services (auto-detect API availability)
        self.llm_provider = create_provider()  # LLM provider (自动检测)
        self._use_llm = not isinstance(self.llm_provider, MockProvider)
        self.llm = MockLLM(
            llm_provider=self.llm_provider if self._use_llm else None,
        )  # 规则降级方案 (LLM 可用时自动升级)

        # ── 初始化 PPAM 四模块 ──
        # Initialize PPAM modules
        self.perception = PerceptionModule(
            llm_provider=self.llm_provider if self._use_llm else None,
        )
        self.memory = MemoryModule(
            llm_provider=self.llm_provider if self._use_llm else None,
        )
        self.planning = PlanningModule()
        self.action = ActionModule(
            self.llm,
            llm_provider=self.llm_provider if self._use_llm else None,
        )
        self.adaptive_manager = AdaptiveParameterManager()

        # ── 会话状态跟踪 ──
        # Session state tracking
        self.n_cum: int = 0               # 累计交互轮次 (跨会话) / cumulative turn count
        self.turn: int = 0                 # 当前会话轮次 / current session turn
        self.session_start: datetime = datetime.now()
        self.session_start_valence: float = 0.0   # 会话首轮效价 / first-turn valence
        self.last_message_time: Optional[datetime] = None  # 上一条消息时间
        self.feedback_history: List[float] = []   # 本会话 feedback_score 列表
        self.last_state_vector: Optional[StateVector] = None  # 上一轮状态向量
        self.last_planning_output: Optional[PlanningOutput] = None  # 上一轮规划输出

        # ── 诊断模式 ──
        # Diagnostic mode flag (always on for prototype)
        self.debug: bool = True

        # ── 消融实验开关 (Ablation Experiment Toggles) ──
        # 用于论文消融实验，关闭后使用基线替代方案
        self.ablation = {
            "affective_memory": True,     # 情感记忆层 (§3.5)
            "mar": True,                  # 情绪自适应检索 (§3.6)
            "thompson_sampling": True,    # Thompson Sampling (§4.5)
        }

    # ──────────────────────────────────────────
    # 主处理方法: process_turn
    # ──────────────────────────────────────────

    def process_turn(self, user_input: str) -> str:
        """
        处理一轮对话 — 执行完整 PPAM 管线。
        Process one conversation turn — execute the full PPAM pipeline.

        管线流程 (Pipeline, §6.1):
          Phase 1:   感知 → StateVector
          Phase 1.5: 预规划意图推断 → planning_intent
          Phase 2:   记忆读取 → RetrievedContext (+ 延迟更新处理)
          Phase 3:   规划 → PlanningOutput (策略选择)
          Phase 4:   行动 → 自然语言回复
          Phase 5:   记忆写入 → EpisodeUnit + DeferredUpdate

        特殊路径:
          - 空输入: 温和提示用户
          - 危机快速通道 (urgency > 0.9): 跳过 Phase 1.5/3

        Args:
            user_input: 用户输入文本

        Returns:
            Agent 回复文本 (Chinese)
        """
        # ── 边界条件: 空输入 ──
        # Edge case: empty input
        if not user_input or not user_input.strip():
            return "我在这里，你想说什么都可以。不着急，慢慢来。"

        # ── 更新轮次计数 (单一来源: self.turn 为权威计数) ──
        # Update turn counters (single source of truth: self.turn)
        self.turn += 1
        self.n_cum += 1
        # 同步工作记忆轮次计数 (防止与 _update_working_memory 中的 +=1 产生分歧)
        self.memory.working_memory.turn_count = self.turn
        now = datetime.now()

        # ── 管线计时开始 ──
        t_start = time.time()

        # ════════════════════════════════════════
        # Phase 1: PERCEPTION (感知)
        # ════════════════════════════════════════
        # 解析用户输入 → StateVector
        # Parse user input → StateVector
        state_vector = self.perception.process(
            text=user_input,
            timestamp=now,
            working_memory=self.memory.working_memory,
            behavioral_baseline=(
                self.memory.semantic_memory.user_profile.behavioral_baseline
            ),
            n_cum=self.n_cum,
            relationship_depth=self._get_relationship_depth(),
            last_message_time=self.last_message_time,
            session_start=self.session_start,
        )
        t_perception = time.time()
        logger.info(f"[PPAM Timing] Phase 1 (Perception): {t_perception - t_start:.2f}s")

        # 记录会话首轮效价 (用于 session-level surrogate)
        # Record first-turn valence for session-level surrogate
        if self.turn == 1:
            self.session_start_valence = state_vector.emotion.valence()

        # ════════════════════════════════════════
        # Phase 2 前置: 延迟更新处理
        # Pre-Phase 2: Process deferred updates from previous turn
        # ════════════════════════════════════════
        # 算法 3.1 步骤③: 在 t+1 轮回溯执行 Beta 更新
        # Algorithm 3.1 Step 3: retroactively execute Beta updates at turn t+1
        if self.turn > 1 and self.memory.deferred_update_queue:
            scores = self.memory.process_deferred_updates(state_vector)
            self.feedback_history.extend(scores)

        # ════════════════════════════════════════
        # 危机快速通道检查 (§6.2)
        # Crisis fast-track check
        # ════════════════════════════════════════
        if state_vector.urgency_level > CRISIS_THRESHOLD:
            # 危机路径: Phase 1 → 跳过 Phase 1.5 → 简化 Phase 2 → 跳过 Phase 3 → Phase 4 (Crisis Handler)
            # Crisis path: skip planning, go directly to crisis handler

            if self.debug:
                diag = _format_diagnostics(state_vector, None, self.turn, self.n_cum)
                print(diag)

            # 简化 Memory Read (固定 Φ=0.5) — 消融: MAR关闭时使用固定权重检索
            if self.ablation.get("mar", True):
                retrieved = self.memory.read_mar(state_vector, "crisis_response")
            else:
                retrieved = self.memory.read_mar_fixed(state_vector)

            # Crisis Handler (§5.5)
            response = self.action.handle_crisis(state_vector, user_message=user_input)

            # Phase 5 (受限): 写入 EpisodeUnit, 但不执行 Beta 更新
            # Restricted Phase 5: write episode, but skip Beta update
            crisis_planning = PlanningOutput(
                selected_strategy="CRISIS_PROTOCOL",
                planning_intent="crisis_response",
                trajectory=TrajectoryResult(current_phase=RecoveryPhase.ACUTE_DISTRESS),
            )
            self.last_planning_output = crisis_planning
            self.memory.write(
                state_vector=state_vector,
                agent_strategy="CRISIS_PROTOCOL",
                planning_output=crisis_planning,
            )
            # 危机回复不入队 deferred update (不影响策略后验)
            # Crisis response does NOT enqueue deferred updates
            # Remove all CRISIS_PROTOCOL deferred updates
            self.memory.deferred_update_queue = [
                item for item in self.memory.deferred_update_queue
                if item.strategy != "CRISIS_PROTOCOL"
            ]

            # 追加危机回复到对话历史 (确保后续轮次有完整上下文)
            self.perception.add_assistant_message(response)

            # 更新情绪基线 (§3.5.1) — 消融: 关闭时跳过
            if self.ablation.get("affective_memory", True):
                e_range = float(state_vector.emotion.e.max() - state_vector.emotion.e.min())
                if e_range > 0.05:  # 非均匀分布才更新基线
                    self.memory.update_emotion_baseline(state_vector.emotion)

            # 更新关系深度 (§4.5.2) — 危机披露通常是高深度信号
            self._update_relationship_depth(state_vector)

            # 更新状态 / Update state
            self._post_turn_update(state_vector, now)
            return response

        # ════════════════════════════════════════
        # Phase 1.5: PRE-PLANNING INTENT ESTIMATION
        # 预规划意图推断 (定义 3.6a)
        # ════════════════════════════════════════
        # 确定性规则映射 → planning_intent
        # 用于 MAR 的动态 β 权重调整
        planning_intent = determine_planning_intent(state_vector)

        # ════════════════════════════════════════
        # Phase 2: MEMORY READ (记忆读取)
        # ════════════════════════════════════════
        # MAR 心境自适应检索 (定义 3.5/3.6) — 消融: MAR关闭时使用固定权重检索
        if self.ablation.get("mar", True):
            retrieved = self.memory.read_mar(state_vector, planning_intent)
        else:
            retrieved = self.memory.read_mar_fixed(state_vector)
        t_memory = time.time()
        logger.info(f"[PPAM Timing] Phase 1.5+2 (Intent+Memory Read): {t_memory - t_perception:.2f}s")

        # ════════════════════════════════════════
        # Phase 2.5: ADAPTIVE PARAMETER COMPUTATION
        # ════════════════════════════════════════
        # 消融: affective_memory关闭时使用默认参数
        if self.ablation.get("affective_memory", True):
            # Compute per-turn adaptive parameter overrides from user history
            user_stats = UserStats(
                emotion_variance=self.memory.compute_emotion_variance(),
                emotion_autocorrelation=self.memory.compute_emotion_autocorrelation(),
                severity_history=self.memory.get_severity_history(),
                strategy_feedback=self.memory.compute_strategy_effectiveness(),
                n_interactions=self.memory.get_interaction_count(),
            )
            llm_conf = state_vector.llm_confidence if state_vector.llm_confidence is not None else 1.0
            adaptive_params = self.adaptive_manager.compute(user_stats, llm_conf)
        else:
            adaptive_params = AdaptiveParams()  # 使用默认静态参数

        # ════════════════════════════════════════
        # Phase 3: PLANNING (规划)
        # ════════════════════════════════════════
        # 轨迹分析 → 情境评估 → 目标推断 → Thompson Sampling
        # Trajectory → Appraisal → Goals → Strategy Selection
        # 消融: TS关闭时使用规则映射
        planning_output = self.planning.plan(
            state_vector=state_vector,
            retrieved_context=retrieved,
            affective_memory=self.memory.affective_memory,
            adaptive_params=adaptive_params,
            use_thompson_sampling=self.ablation.get("thompson_sampling", True),
        )
        self.last_planning_output = planning_output
        t_planning = time.time()
        logger.info(f"[PPAM Timing] Phase 3 (Planning): {t_planning - t_memory:.2f}s")

        # 打印诊断信息 / Print diagnostics
        if self.debug:
            diag = _format_diagnostics(
                state_vector, planning_output, self.turn, self.n_cum
            )
            print(diag)

        # ════════════════════════════════════════
        # Phase 4: ACTION (行动)
        # ════════════════════════════════════════
        # 生成自然语言回复
        # Generate natural language response
        if self._use_llm:
            try:
                # LLM 模式: 使用大模型生成自然回复
                emo = state_vector.emotion
                if len(emo.e) == 0:
                    dominant_emotion = "sadness"  # safe default
                else:
                    # 均匀分布检测: argmax(uniform) = joy (idx 0) — 会误导 LLM
                    e_range = float(emo.e.max() - emo.e.min())
                    if e_range < 0.05:
                        dominant_emotion = "sadness"  # 安全默认 (避免 joy 误导)
                    else:
                        dominant_idx = int(emo.e.argmax())
                        dominant_emotion = EMOTIONS[dominant_idx] if dominant_idx < len(EMOTIONS) else "sadness"
                memory_ctx = ""
                if retrieved and retrieved.episodes:
                    mem_parts = []
                    for ep in retrieved.episodes[:3]:
                        if ep.event:
                            mem_parts.append(f"- {ep.event[:80]}")
                    memory_ctx = "\n".join(mem_parts)

                state_info = {
                    "dominant_emotion": dominant_emotion,
                    "intensity": emo.intensity,
                    "valence": float(emo.valence()),
                    "intent": state_vector.intent.top_intent(),
                    "topic": state_vector.topic_category or "日常对话",
                    "phase": planning_output.trajectory.current_phase.value if planning_output.trajectory else "stable_state",
                    "urgency": state_vector.urgency_level,
                    "controllability": state_vector.controllability if state_vector.controllability is not None else 0.5,
                    "life_impact": state_vector.life_impact if state_vector.life_impact is not None else 0.5,
                    "direction": planning_output.trajectory.direction.value if (hasattr(planning_output, 'trajectory') and planning_output.trajectory and hasattr(planning_output.trajectory, 'direction')) else "stable",
                    "immediate_goal": planning_output.goals.immediate_goal if hasattr(planning_output, 'goals') and planning_output.goals else "",
                }
                # 传递历史时排除当前用户消息 (已通过 user_message 参数单独传入)
                # 避免当前消息在 prompt 中重复出现
                # 获取最近13条以便在末尾为user消息时可以排除后仍有12条
                hist = self.perception.get_recent_history(13)
                if hist and hist[-1].get("role") == "user":
                    response_history = hist[:-1][-12:]
                else:
                    response_history = hist[-12:]
                response = self.llm_provider.generate_response(
                    user_message=user_input,
                    strategy=planning_output.selected_strategy,
                    state_info=state_info,
                    history=response_history,
                    memory_context=memory_ctx,
                )
                # 乱码/退化检测: LLM 可能产生重复退化文本 (Holtzman et al., 2019)
                # 检测到后重试一次; 仍失败则降级为模板生成
                if _is_garbled_response(response):
                    logger.warning(f"Garbled LLM output detected (len={len(response)}), retrying once")
                    response = self.llm_provider.generate_response(
                        user_message=user_input,
                        strategy=planning_output.selected_strategy,
                        state_info=state_info,
                        history=response_history,
                        memory_context=memory_ctx,
                    )
                    if _is_garbled_response(response):
                        logger.warning("Garbled output persists after retry, falling back to template")
                        response = self.action.generate(
                            planning_output=planning_output,
                            state_vector=state_vector,
                            retrieved_context=retrieved,
                            user_message=user_input,
                        )
                # 记录 Agent 回复到对话历史
                self.perception.add_assistant_message(response)
            except Exception as e:
                # LLM 失败时降级为模板
                logger.warning(f"LLM generate_response 失败, 降级为模板: {e}")
                response = self.action.generate(
                    planning_output=planning_output,
                    state_vector=state_vector,
                    retrieved_context=retrieved,
                    user_message=user_input,
                )
                # 降级路径也需要追加 assistant 消息到历史
                self.perception.add_assistant_message(response)
        else:
            response = self.action.generate(
                planning_output=planning_output,
                state_vector=state_vector,
                retrieved_context=retrieved,
            )
            # 非 LLM 路径也需要追加 assistant 消息到历史
            self.perception.add_assistant_message(response)

        t_action = time.time()
        logger.info(f"[PPAM Timing] Phase 4 (Action): {t_action - t_planning:.2f}s")

        # ════════════════════════════════════════
        # Phase 5: MEMORY WRITE (记忆写入)
        # ════════════════════════════════════════
        # 算法 3.1: 选择性写入
        # Algorithm 3.1: Selective Write
        # ① 更新工作记忆
        # ② 条件性写入情景记忆 (importance > τ_write)
        # ③ 入队延迟更新 (Beta 更新在 t+1 轮执行)
        self.memory.write(
            state_vector=state_vector,
            agent_strategy=planning_output.selected_strategy,
            planning_output=planning_output,
        )

        t_write = time.time()
        logger.info(f"[PPAM Timing] Phase 5 (Memory Write): {t_write - t_action:.2f}s")

        # 更新情绪基线 (§3.5.1) — 消融: 关闭时跳过
        if self.ablation.get("affective_memory", True):
            self.memory.update_emotion_baseline(
                state_vector.emotion,
                adaptive_baseline_lr=adaptive_params.baseline_lr,
            )

        # 更新关系深度 (§4.5.2)
        # Update relationship depth
        self._update_relationship_depth(state_vector)

        # 更新状态 / Update state
        self._post_turn_update(state_vector, now)

        t_end = time.time()
        logger.info(f"[PPAM Timing] Total: {t_end - t_start:.2f}s")

        return response

    # ──────────────────────────────────────────
    # 会话结束处理 (Session End)
    # ──────────────────────────────────────────

    def end_session(self):
        """
        会话结束处理 — 执行清理和巩固操作。
        End session — perform cleanup and consolidation.

        流程:
        1. 为最后一轮计算会话级代理评分 (Session-Level Surrogate)
        2. 执行反思整合 (Reflective Consolidation, 算法 3.2)
        3. 执行遗忘检查 (Forget Check, 定义 3.7)
        """
        if self.turn == 0:
            print("[Session] 本次会话无实质内容。/ No turns in this session.")
            return

        # ── 1. 会话级代理评分 ──
        # Session-Level Surrogate for last turn's deferred update
        if self.memory.deferred_update_queue:
            # 计算 fallback feedback score
            final_valence = (
                self.last_state_vector.emotion.valence()
                if self.last_state_vector else 0.0
            )
            surrogate = self.memory.compute_session_surrogate(
                session_start_valence=self.session_start_valence,
                final_valence=final_valence,
                feedback_history=self.feedback_history,
            )
            # 处理最后的 deferred updates with surrogate score
            # 使用 SESSION_END_DECAY = 0.5 衰减系数
            for deferred in self.memory.deferred_update_queue:
                self.memory.update_beta(
                    strategy=deferred.strategy,
                    context_key=deferred.context_key,
                    feedback_score=surrogate,
                    is_session_end=True,
                )
                # 回填 episode feedback (如果存在)
                for ep in self.memory.episodic_memory:
                    if ep.id == deferred.episode_id:
                        ep.user_feedback_score = surrogate
                        break
            self.memory.deferred_update_queue.clear()

        # ── 2. 反思整合 ──
        # Reflective Consolidation (Algorithm 3.2)
        print("[Session] 执行反思整合... / Running reflective consolidation...")
        self.memory.reflect()

        # ── 3. 遗忘检查 ──
        # Forget Check (Definition 3.7)
        print("[Session] 执行遗忘检查... / Running forget check...")
        self.memory.forget_check()

        # ── 会话统计 ──
        # Session statistics
        print(f"\n[Session Summary]")
        print(f"  总轮次 / Total turns: {self.turn}")
        print(f"  累计交互 / Cumulative: {self.n_cum}")
        print(f"  情景记忆数 / Episodic memories: {len(self.memory.episodic_memory)}")
        print(f"  归档记忆数 / Archived memories: {len(self.memory.archived_episodes)}")

        # 重置会话状态 (保留跨会话状态)
        # Reset session state (keep cross-session state)
        self.turn = 0
        self.session_start = datetime.now()
        self.session_start_valence = 0.0
        self.last_message_time = None
        self.feedback_history = []
        self.last_state_vector = None
        self.last_planning_output = None

        # 重置工作记忆 / Reset working memory
        self.memory.working_memory = WorkingMemory()

        # 重置规划模块的轨迹缓存 (防止跨会话滞后状态污染)
        # Reset planning trajectory cache to prevent stale hysteresis across sessions
        self.planning.reset_session()

        # 清除感知模块的对话历史
        self.perception.clear_history()

    # ──────────────────────────────────────────
    # 交互式主循环 (Interactive Main Loop)
    # ──────────────────────────────────────────

    def run(self):
        """
        运行交互式对话循环。
        Run the interactive conversation loop.

        输入 "exit", "quit", "bye", "再见", "退出", "结束" 退出。
        Type one of the exit commands to end the session.
        """
        print("=" * 60)
        print("  EmoMem — 记忆增强 AI 情感陪伴 Agent")
        print("  Memory-Augmented AI Emotional Companion")
        print("=" * 60)
        print("  输入你想说的话，我会在这里陪着你。")
        print("  Type anything to chat. Type 'exit' or '再见' to end.")
        print("=" * 60)
        print()

        try:
            while True:
                # 获取用户输入 / Get user input
                try:
                    user_input = input("你: ").strip()
                except EOFError:
                    # 处理管道输入结束 / Handle piped input EOF
                    break

                # 检查退出命令 / Check exit commands
                if user_input.lower() in EXIT_COMMANDS:
                    # 危机状态下不使用轻松告别语
                    if self._is_in_crisis_state():
                        print(f"\nAgent: {CRISIS_EXIT_RESPONSE}")
                    else:
                        print("\nAgent: 好的，期待下次和你聊天。照顾好自己。再见！")
                    break

                # 处理一轮对话 / Process one turn
                response = self.process_turn(user_input)

                # 输出回复 / Print response
                print(f"\nAgent: {response}\n")

        except KeyboardInterrupt:
            print("\n\n[中断] 收到退出信号。/ Interrupted.")

        finally:
            # 会话结束处理 / End session cleanup
            self.end_session()
            print("\n[EmoMem] 会话已结束。感谢使用！/ Session ended. Thank you!")

    # ──────────────────────────────────────────
    # 内部辅助方法 (Internal helpers)
    # ──────────────────────────────────────────

    def _post_turn_update(
        self,
        state_vector: StateVector,
        now: datetime,
    ):
        """
        轮次后更新 — 更新各种追踪状态。
        Post-turn update — update various tracking state.
        """
        # 更新上一条消息时间 / Update last message time
        self.last_message_time = now

        # 缓存状态向量 / Cache state vector
        self.last_state_vector = state_vector

        # 更新工作记忆中的上一轮情绪快照 (供下一轮 trend_local 计算)
        # Update prev_emotion in working memory for next turn's trend_local
        self.memory.working_memory.prev_emotion = state_vector.emotion.copy()

        # 更新上一轮意图快照 (供下一轮意图连续性计算, §2.7b)
        # 深拷贝避免后续 intent blending 修改已存储的 prev_intent
        self.memory.working_memory.prev_intent = IntentDistribution(
            probabilities=dict(state_vector.intent.probabilities)
        )

    def _get_relationship_depth(self) -> float:
        """
        获取当前关系深度 d(t)。
        Get current relationship depth d(t).

        从工作记忆或默认值获取。
        """
        # 简化: 使用基于累计轮次的渐进增长
        # Simplified: gradual growth based on cumulative turns
        # d(t) 的正式更新在 _update_relationship_depth 中执行
        if hasattr(self, '_relationship_depth'):
            return self._relationship_depth
        return 0.0

    def _is_in_crisis_state(self) -> bool:
        """
        检查当前是否处于危机状态。
        用于会话退出时决定是否使用危机感知告别语。

        判定条件 (任一为真):
        - 上一轮 urgency > CRISIS_THRESHOLD
        - 当前处于 acute_distress 阶段
        - 危机冷却期内 (crisis_cooldown_counter > 0)
        """
        # 最近一轮紧急度高
        if self.last_state_vector and self.last_state_vector.urgency_level > CRISIS_THRESHOLD:
            return True
        # 处于急性困扰阶段
        if (self.last_planning_output and self.last_planning_output.trajectory
                and self.last_planning_output.trajectory.current_phase == RecoveryPhase.ACUTE_DISTRESS):
            return True
        # 危机冷却期内
        if self.memory.working_memory.crisis_cooldown_counter > 0:
            return True
        return False

    def _update_relationship_depth(self, state_vector: StateVector):
        """
        §4.5.2: 更新关系深度 d(t)。
        Update relationship depth d(t).

        d(t+1) = d(t) + λ_d · (disclosure_level - d(t))
        其中 λ_d = 0.1 (关系深度学习率)

        披露等级判定 (简化):
        - L0_factual (0.00): 陈述事实
        - L1_opinion (0.25): 表达观点
        - L2_emotion (0.50): 分享情绪
        - L3_vulnerability (1.00): 暴露脆弱面
        """
        # 获取当前深度 / Get current depth
        current_d = self._get_relationship_depth()

        # 简化的披露等级判定 / Simplified disclosure level assessment
        # 基于情绪强度和意图类型推断
        intensity = state_vector.emotion.intensity
        top_intent = state_vector.intent.top_intent()

        # 高情绪强度 + 倾诉/安慰意图 → 高披露等级
        if top_intent in ("CRISIS", "VENT", "COMFORT") and intensity > 0.5:
            disclosure_value = DISCLOSURE_LEVELS["L3_vulnerability"]
        elif top_intent in ("VENT", "COMFORT", "REFLECT"):
            disclosure_value = DISCLOSURE_LEVELS["L2_emotion"]
        elif top_intent in ("ADVICE", "SHARE_JOY"):
            disclosure_value = DISCLOSURE_LEVELS["L1_opinion"]
        else:
            disclosure_value = DISCLOSURE_LEVELS["L0_factual"]

        # EMA 更新: d(t+1) = d(t) + λ_d · (disclosure - d(t))
        new_d = current_d + RELATIONSHIP_DEPTH_LR * (disclosure_value - current_d)
        self._relationship_depth = float(np.clip(new_d, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════
# 入口点 (Entry point)
# ══════════════════════════════════════════════════════════════

def main():
    """EmoMem Agent 入口函数 / Entry point function."""
    agent = EmoMemAgent()
    agent.run()


if __name__ == "__main__":
    main()
