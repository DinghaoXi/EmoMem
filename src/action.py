"""
EmoMem 行动模块 (Action Module)
=================================
§5: 将规划模块的结构化策略方案转化为自然、有温度、个性化的语言输出。

本模块包含四个子组件：
1. 记忆锚定生成器 (Memory-Grounded Generator) — §5.2
2. 人格一致性守卫 (Persona Consistency Guard) — §5.3
3. 语调校准器 (Tone Calibrator) — §5.4
4. 危机升级处理器 (Crisis Escalation Handler) — §5.5

数据流:
- 输入: PlanningOutput (F5), StateVector, RetrievedContext (F3/F4)
- 输出: 自然语言回复 → User
"""

from __future__ import annotations

import logging
from typing import Optional

from .config import (
    CRISIS_THRESHOLD,
    STRATEGIES,
    PERSONA_GUARD_HUMAN_CLAIM_PHRASES,
    PERSONA_GUARD_HUMAN_CLAIM_REPLACEMENT,
    PERSONA_GUARD_MEDICAL_PHRASES,
    PERSONA_GUARD_MEDICAL_REPLACEMENT,
    PERSONA_GUARD_EMPTY_FALLBACK,
    TONE_ACUTE_MAX_LENGTH,
    TONE_ACUTE_SEARCH_WINDOW,
    TONE_ACUTE_MIN_LENGTH,
    TONE_SENTENCE_SEPARATORS,
    TONE_TRUNCATION_ELLIPSIS,
    CRISIS_CONCERN_FAREWELL_SIGNALS,
    CRISIS_CONCERN_EXISTENTIAL_SIGNALS,
    CRISIS_CONCERN_GRATITUDE_SIGNALS,
    CRISIS_CONCERN_HARM_SIGNALS,
    CRISIS_CONCERN_MIN_LENGTH,
)
from .models import (
    StateVector,
    PlanningOutput,
    RetrievedContext,
    RecoveryPhase,
)
from .mock_llm import MockLLM

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# §5.4 策略-语调映射 (表9)
# ──────────────────────────────────────────────
# 每种策略对应的语调特征描述和修饰前缀/后缀
# Tone characteristics and prefix/suffix modifiers per strategy

TONE_MAP = {
    "active_listening": {
        "characteristics": "温和、不评判、反射性",
        "prefix": "",
        "suffix": "",
    },
    "emotional_validation": {
        "characteristics": "接纳、肯定",
        "prefix": "",
        "suffix": "",
    },
    "empathic_reflection": {
        "characteristics": "温暖、共情",
        "prefix": "",
        "suffix": "",
    },
    "gentle_guidance": {
        "characteristics": "温和、开放式",
        "prefix": "",
        "suffix": "",
    },
    "cognitive_reframing": {
        "characteristics": "稍正式、引导思考",
        "prefix": "",
        "suffix": "",
    },
    "problem_solving": {
        "characteristics": "务实、具体",
        "prefix": "",
        "suffix": "",
    },
    "information_providing": {
        "characteristics": "知识性、参考性",
        "prefix": "",
        "suffix": "",
    },
    "strength_recognition": {
        "characteristics": "热情、肯定",
        "prefix": "",
        "suffix": "",
    },
    "companionable_silence": {
        "characteristics": "简短、不施压",
        "prefix": "",
        "suffix": "",
    },
    "positive_reinforcement": {
        "characteristics": "热情、具体",
        "prefix": "",
        "suffix": "",
    },
}

# ──────────────────────────────────────────────
# §5.4 恢复阶段 → 语调修饰 (Recovery phase tone modifiers)
# ──────────────────────────────────────────────
# 不同恢复阶段对回复语调的额外调整
PHASE_TONE_MODIFIERS = {
    RecoveryPhase.ACUTE_DISTRESS: {
        "energy": "low",       # 低能量——使用简短句
        "warmth": "high",      # 高温暖——更多共情词
        "directness": "low",   # 低直接——不施压
        "suffix": "",
    },
    RecoveryPhase.EARLY_RECOVERY: {
        "energy": "medium",
        "warmth": "high",
        "directness": "low",
        "suffix": "",
    },
    RecoveryPhase.CONSOLIDATION: {
        "energy": "medium",
        "warmth": "medium",
        "directness": "medium",
        "suffix": "",
    },
    RecoveryPhase.STABLE_STATE: {
        "energy": "high",
        "warmth": "medium",
        "directness": "high",
        "suffix": "",
    },
}


class ActionModule:
    """
    行动模块 — 将规划结果转化为自然语言回复。
    Action Module — transforms planning output into natural language responses.

    职责:
    1. 调用 Mock LLM 基于策略生成回复 (memory-grounded generation)
    2. 基本的人格一致性检查 (persona consistency guard)
    3. 基于恢复阶段的语调校准 (tone calibration by recovery phase)
    4. 危机快速通道处理 (crisis fast-track)
    """

    def __init__(self, mock_llm: MockLLM, llm_provider=None):
        """
        初始化行动模块。
        Initialize the Action module.

        Args:
            mock_llm: Mock LLM 服务实例，用于回复生成
            llm_provider: Optional BaseLLMProvider instance for LLM-powered
                          crisis concern generation. None = keyword-only mode.
        """
        self.llm = mock_llm
        self.llm_provider = llm_provider

    # ──────────────────────────────────────────
    # 主入口: 生成回复 (Main entry: generate response)
    # ──────────────────────────────────────────

    def generate(
        self,
        planning_output: PlanningOutput,
        state_vector: StateVector,
        retrieved_context: Optional[RetrievedContext] = None,
        user_message: str = "",
    ) -> str:
        """
        行动模块主方法 — 生成最终回复。
        Main action method — generates the final response.

        流程 (Pipeline):
        1. 检查是否触发危机快速通道 (crisis fast-track check)
        2. 调用 Mock LLM 生成基础回复 (base response via Mock LLM)
        3. 人格一致性守卫 (persona consistency guard)
        4. 语调校准 (tone calibration by recovery phase)

        Args:
            planning_output: 规划模块输出（含策略选择、轨迹分析等）
            state_vector: 感知模块输出的状态向量
            retrieved_context: MAR 检索的上下文

        Returns:
            最终回复文本 (final response text)
        """
        # ── Step 0: 危机快速通道 (§5.5, §6.2) ──
        # urgency > 0.9 时绕过常规流程
        if state_vector.urgency_level > CRISIS_THRESHOLD:
            return self.handle_crisis(state_vector, user_message=user_message)

        # ── Step 1: 获取选定策略 ──
        strategy = planning_output.selected_strategy
        if strategy not in STRATEGIES:
            strategy = "active_listening"  # 安全回退 / safe fallback

        # ── Step 2: 调用 Mock LLM 生成基础回复 ──
        # Memory-Grounded Generator (§5.2)
        base_response = self.llm.generate_response(
            strategy=strategy,
            state_vector=state_vector,
            retrieved_context=retrieved_context,
            goals=planning_output.goals,
        )

        # ── Step 3: 人格一致性守卫 (§5.3) ──
        # 基本检查：确保回复不违背 Agent 的核心人格设定
        response = self._apply_persona_guard(base_response, strategy)

        # ── Step 4: 语调校准 (§5.4) ──
        # 根据恢复阶段微调回复的语调
        current_phase = planning_output.trajectory.current_phase
        response = self._apply_tone_calibration(response, strategy, current_phase)

        return response

    # ──────────────────────────────────────────
    # 危机快速通道 (Crisis Fast-Track Handler)
    # ──────────────────────────────────────────

    def handle_crisis(self, state_vector: StateVector, user_message: str = "") -> str:
        """
        危机快速通道 — 当 urgency > 0.9 时绕过常规流程。
        Crisis fast-track — bypasses normal pipeline when urgency > 0.9.

        §5.5 安全协议:
        1. 表达关切（不评判，引用用户具体内容）
        2. 评估风险
        3. 提供专业资源（心理热线等）
        4. 持续陪伴
        5. 策略约束（禁用认知重构、问题解决等施压策略）

        §6.2 特殊路径:
        - agent_strategy = "CRISIS_PROTOCOL"
        - 不执行 Thompson Sampling Beta 更新

        Args:
            state_vector: 当前状态向量
            user_message: 用户原始消息（用于个性化关切表达）

        Returns:
            危机回复文本 (crisis response text)
        """
        # ── 1. 个性化关切 (Personalized concern, no judgment) ──
        # 根据用户具体内容生成1句话acknowledgment，保持"被听见"的感觉
        concern = self._build_crisis_concern(user_message)

        # ── 2. 安全确认 (Safety check) ──
        safety_check = "你现在安全吗？"

        # ── 3. 专业资源 (Professional resources — fixed, never vary) ──
        resources = (
            "如果你现在正处于危机中，请联系专业帮助：\n"
            "  · 全国24小时心理援助热线：400-161-9995\n"
            "  · 北京心理危机研究与干预中心：010-82951332\n"
            "  · 生命热线：400-821-1215\n"
            "  · 如果情况紧急，请拨打120或前往最近的医院急诊。"
        )

        # ── 4. 持续陪伴承诺 (Companionship) ──
        companionship = "你不是一个人。"

        # 组合危机回复 / Compose crisis response
        crisis_response = (
            f"{concern}{safety_check}\n\n"
            f"{resources}\n\n"
            f"{companionship}"
        )

        return crisis_response

    def _build_crisis_concern(self, user_message: str) -> str:
        """
        根据用户消息内容生成个性化关切表达。
        在保证安全的前提下，让用户感到"被听见"。

        Three-tier fallback:
          Tier 1: LLM personalized concern (if llm_provider available)
          Tier 2: Keyword template matching (hardcoded patterns)
          Tier 3: Generic concern message
        """
        if not user_message:
            return "你说的这些让我很担心你的状况。"

        # ── Tier 1: LLM personalized concern (if available) ──
        if self.llm_provider is not None:
            try:
                concern = self.llm_provider.generate_crisis_concern(
                    user_message,
                )
                if concern and len(concern) > CRISIS_CONCERN_MIN_LENGTH:
                    return concern
            except Exception as e:
                logger.warning(
                    f"LLM crisis concern generation failed: {e}"
                )

        # ── Tier 2: Keyword template matching (existing logic) ──
        msg = user_message.strip()

        if any(s in msg for s in CRISIS_CONCERN_FAREWELL_SIGNALS):
            return "听到你说这些，我很担心你现在的状况。"
        elif any(s in msg for s in CRISIS_CONCERN_EXISTENTIAL_SIGNALS):
            return "你提到的这些问题让我很担心。"
        elif any(s in msg for s in CRISIS_CONCERN_GRATITUDE_SIGNALS):
            return "谢谢你愿意和我说这些，但我现在非常担心你。"
        elif any(s in msg for s in CRISIS_CONCERN_HARM_SIGNALS):
            return "你说的这些让我非常担心你的安全。"
        else:
            return "你说的这些让我很担心你的状况。"

    # ──────────────────────────────────────────
    # 人格一致性守卫 (Persona Consistency Guard)
    # ──────────────────────────────────────────

    def _apply_persona_guard(self, response: str, strategy: str) -> str:
        """
        §5.3 人格一致性守卫 — 基本检查确保回复符合 Agent 人格设定。
        Persona consistency guard — basic checks for agent personality alignment.

        检查项 (Checks):
        1. 不应包含暴力/攻击性语言
        2. 不应声称是人类
        3. 不应给出医疗诊断
        4. 不应使用冷漠/敷衍的表达

        真实系统中使用 LLM-as-judge 进行一致性评分（1-5分），
        低于3分触发重新生成。此处为简化的规则检查。

        Args:
            response: 待检查的回复文本
            strategy: 当前使用的策略

        Returns:
            通过检查的回复文本（或修正后的文本）
        """
        # ── Check 1: 确保不声称是人类 ──
        # Ensure the agent doesn't claim to be human
        for phrase in PERSONA_GUARD_HUMAN_CLAIM_PHRASES:
            if phrase in response:
                response = response.replace(phrase, PERSONA_GUARD_HUMAN_CLAIM_REPLACEMENT)

        # ── Check 2: 确保不给出医疗诊断 ──
        # Ensure no medical diagnoses are made
        for phrase in PERSONA_GUARD_MEDICAL_PHRASES:
            if phrase in response:
                response = response.replace(
                    phrase,
                    PERSONA_GUARD_MEDICAL_REPLACEMENT,
                )

        # ── Check 3: 敷衍表达由 LLM prompt 的"绝对禁止"规则保证 ──
        # RESPONSE_SYSTEM_PROMPT 的"绝对禁止"部分已包含此约束

        # ── Check 4: 确保回复不为空 ──
        # Ensure response is not empty
        if not response.strip():
            response = PERSONA_GUARD_EMPTY_FALLBACK

        return response

    # ──────────────────────────────────────────
    # 语调校准器 (Tone Calibrator)
    # ──────────────────────────────────────────

    def _apply_tone_calibration(
        self,
        response: str,
        strategy: str,
        current_phase: RecoveryPhase,
    ) -> str:
        """
        §5.4 语调校准 — 根据恢复阶段和策略微调回复语调。
        Tone calibration based on recovery phase and strategy.

        调整规则 (Adjustment rules):
        - acute_distress: 缩短回复，增加温暖用语，添加陪伴后缀
        - early_recovery: 保持温暖，适度引导
        - consolidation: 平衡温暖与直接
        - stable_state: 允许更直接和具活力的表达

        Args:
            response: 基础回复文本
            strategy: 当前策略
            current_phase: 当前恢复阶段

        Returns:
            语调校准后的回复文本
        """
        # 获取阶段修饰器 / Get phase modifier
        modifier = PHASE_TONE_MODIFIERS.get(current_phase)
        if not modifier:
            return response

        # ── acute_distress 阶段特殊处理 ──
        # 在急性困扰期，回复应更简短温暖
        if current_phase == RecoveryPhase.ACUTE_DISTRESS:
            # 如果回复过长，截取到最优句号处
            # Truncate long responses at best sentence boundary
            if len(response) > TONE_ACUTE_MAX_LENGTH:
                # Search for sentence boundary in a slightly larger window
                search_window = response[:TONE_ACUTE_SEARCH_WINDOW]
                best_idx = -1
                for sep in TONE_SENTENCE_SEPARATORS:
                    idx = search_window.rfind(sep)
                    if idx > TONE_ACUTE_MIN_LENGTH and idx > best_idx:
                        best_idx = idx
                if best_idx > 0:
                    response = search_window[:best_idx + 1]
                else:
                    # Fallback: truncate at max length but add ellipsis
                    response = response[:TONE_ACUTE_MAX_LENGTH] + TONE_TRUNCATION_ELLIPSIS

        # ── 添加阶段性后缀 ──
        # Append phase-specific suffix
        suffix = modifier.get("suffix", "")
        if suffix and suffix not in response:
            response = response.rstrip() + suffix

        return response
