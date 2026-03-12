"""
EmoMem Mock LLM 服务 — 模拟大语言模型能力
=============================================
Mock LLM Service — Simulates LLM capabilities without an actual API.

本模块为 EmoMem 框架提供模拟的 LLM 服务，包括：
- 基于策略模板的中文回复生成 (template-based response generation)
- 可控性评估 (controllability assessment)
- 生活影响评估 (life impact assessment)
- 话题提取 (topic extraction)
- 会话摘要 (session summarization)

所有方法均为 Mock 实现，无需实际 LLM API 调用。
"""

from __future__ import annotations

import random
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import (
    STRATEGIES,
    EMOTIONS,
    INTENT_TYPES,
)
from .models import (
    EmotionVector,
    StateVector,
    GoalSet,
    RetrievedContext,
    EpisodeUnit,
)


# ──────────────────────────────────────────────
# 策略模板库 (Strategy Template Library)
# ──────────────────────────────────────────────
# 每种策略对应多条中文模板，运行时随机选择一条。
# 模板中可包含占位符 {topic}、{memory_ref}、{emotion_word} 等，
# 由 generate_response 方法填充。

STRATEGY_TEMPLATES: Dict[str, List[str]] = {
    # s1: 积极倾听 — 反射用户所说，温和不评判
    "active_listening": [
        "嗯，我听到你说的了。{reflection}",
        "你是说{reflection}，对吗？我在认真听。",
        "我注意到你提到了{topic}，能多跟我说说吗？",
        "嗯嗯，我在这里听着。{reflection}",
        "听起来{topic}对你来说很重要，你想继续聊聊吗？",
    ],

    # s2: 情感验证 — 肯定用户的感受
    "emotional_validation": [
        "你的感受完全可以理解，任何人遇到这样的事情都会{emotion_word}。",
        "有这样的感觉是很正常的，你不需要为此感到抱歉。",
        "换做是谁，面对{topic}这样的事情，都会有类似的感受。",
        "你的{emotion_word}是完全合理的，我理解你现在的心情。",
        "这种感受很真实，你不必压抑它。",
    ],

    # s3: 共情反射 — 深层情绪共鸣
    "empathic_reflection": [
        "听到你说这些，我也觉得很{emotion_word}。你承受了很多。",
        "我能感受到你现在有多{emotion_word}。这真的不容易。",
        "你一个人扛着这些，真的很不容易。",
        "我能体会到{topic}给你带来的{emotion_word}，这种感受一定很沉重。",
        "谢谢你愿意跟我分享这些。你的{emotion_word}我感同身受。",
    ],

    # s4: 温和引导 — 开放性问题探索
    "gentle_guidance": [
        "你觉得是什么让你最{emotion_word}的呢？",
        "能多说说{topic}的情况吗？我想更好地理解你。",
        "在这件事里，什么是你最在意的部分？",
        "你现在最希望发生什么变化呢？",
        "如果给自己一些时间想想，你觉得{topic}背后最深层的感受是什么？",
    ],

    # s5: 认知重构 — 提供新视角
    "cognitive_reframing": [
        "我理解你的感受。你有没有想过，从另一个角度来看，{topic}也许意味着一些新的可能？",
        "虽然现在很{emotion_word}，但也许这件事也在告诉你一些重要的东西。",
        "换一个角度想想，也许{topic}的发生，恰好给了你一个重新审视的机会。",
        "我注意到你一直在关注{topic}的消极面，如果我们也看看其中积极的部分呢？",
        "有时候困难本身也是成长的契机。你觉得从{topic}中，你学到了什么？",
    ],

    # s6: 问题解决 — 具体行动建议
    "problem_solving": [
        "关于{topic}，我们可以一起想想有什么具体的应对方法。",
        "或许可以试试把{topic}拆分成几个小步骤，一步一步来？",
        "你觉得在{topic}这件事上，有什么是你现在就可以着手做的吗？",
        "让我们一起列出几个可能的方案，看看哪个最适合你现在的情况。",
        "面对{topic}，也许可以先从最小的一步开始。你觉得第一步可以是什么？",
    ],

    # s7: 信息提供 — 分享相关知识或资源
    "information_providing": [
        "关于{topic}，我了解到一些可能对你有帮助的信息。",
        "在面对类似{topic}的情况时，很多人发现这些方法有所帮助……",
        "有一些关于{topic}的建议可能对你有参考价值。",
        "根据我的了解，{topic}相关的资源可能对你有所帮助。",
        "你知道吗？面对{topic}这样的情况，有一些经过验证的方法可以参考。",
    ],

    # s8: 优势识别 — 肯定用户的力量
    "strength_recognition": [
        "我注意到你在面对{topic}时展现出了很大的勇气。",
        "你能够坦诚地面对自己的感受，这本身就是一种力量。",
        "虽然现在很{emotion_word}，但你一直在坚持，这很了不起。",
        "你的韧性比你自己意识到的要强得多。从你处理{topic}的方式就能看出来。",
        "能够主动寻求帮助和倾诉，说明你是一个非常有自我觉察力的人。",
    ],

    # s9: 陪伴性沉默 — 温暖简短的存在感
    "companionable_silence": [
        "我在这里。不急，想说的时候随时都可以。",
        "我一直在。",
        "嗯，你的节奏就好。",
        "不需要着急，你的节奏就好。我在。",
        "我会一直在这里的。",
    ],

    # s10: 积极强化 — 肯定和庆祝
    "positive_reinforcement": [
        "你做到了！这真的很了不起！",
        "太棒了！{topic}的进展让我也替你感到开心。",
        "你的努力没有白费，这个成果值得庆祝！",
        "我为你感到骄傲！你一步步走到了这里。",
        "这是一个很重要的进步！你应该为自己感到自豪。",
    ],
}

# ──────────────────────────────────────────────
# 情绪词映射 (Emotion → Chinese expression)
# ──────────────────────────────────────────────
EMOTION_WORDS: Dict[str, str] = {
    "joy": "开心",
    "trust": "信任",
    "fear": "害怕",
    "surprise": "惊讶",
    "sadness": "难过",
    "disgust": "反感",
    "anger": "生气",
    "anticipation": "期待",
}

# ──────────────────────────────────────────────
# 显式反馈关键词 (Explicit feedback keywords)
# ──────────────────────────────────────────────
POSITIVE_FEEDBACK_KW = ["谢谢", "感谢", "说得对", "有道理", "好的", "嗯嗯",
                        "是的", "对", "你说得对", "谢", "有帮助", "舒服多了",
                        "好多了", "心情好了", "开心"]
NEGATIVE_FEEDBACK_KW = ["算了", "你不懂", "没用", "别说了", "烦", "无语",
                        "不想聊", "闭嘴", "够了", "不是这样"]

# ──────────────────────────────────────────────
# 话题关键词库 (Topic keyword dictionaries)
# ──────────────────────────────────────────────
TOPIC_CATEGORIES: Dict[str, List[str]] = {
    "work": ["工作", "上班", "加班", "同事", "老板", "领导", "公司", "项目",
             "任务", "绩效", "晋升", "裁员", "辞职", "面试", "简历", "薪资",
             "职场", "业务", "客户", "会议", "deadline"],
    "relationship": ["男朋友", "女朋友", "男友", "女友", "恋人", "对象", "爱人",
                     "老公", "老婆", "丈夫", "妻子", "分手", "吵架", "复合",
                     "暧昧", "表白", "约会", "相亲", "感情", "恋爱"],
    "family": ["爸爸", "妈妈", "父母", "家人", "爸", "妈", "哥哥", "姐姐",
               "弟弟", "妹妹", "爷爷", "奶奶", "外公", "外婆", "家里",
               "亲人", "儿子", "女儿", "孩子"],
    "health": ["身体", "健康", "生病", "医院", "医生", "看病", "吃药", "疼",
               "痛", "失眠", "睡不着", "焦虑", "抑郁", "心理", "压力"],
    "study": ["考试", "学习", "论文", "作业", "毕业", "学校", "老师", "课",
              "成绩", "GPA", "大学", "研究生", "实验", "答辩"],
    "social": ["朋友", "社交", "聚会", "孤独", "寂寞", "独处", "合群",
               "社恐", "尴尬"],
    "self": ["自己", "自我", "意义", "价值", "未来", "迷茫", "方向",
             "选择", "人生", "目标", "梦想"],
}

# ──────────────────────────────────────────────
# 可控性关键词 (Controllability keywords)
# ──────────────────────────────────────────────
HIGH_CONTROL_KW = ["我可以", "我决定", "我选择", "我打算", "也许我该",
                   "我想试试", "我能", "我要", "我计划"]
LOW_CONTROL_KW = ["没办法", "不得不", "被迫", "无能为力", "没有选择",
                  "控制不了", "无法", "不可能", "被", "只能"]

# ──────────────────────────────────────────────
# 生活影响关键词 (Life impact keywords)
# ──────────────────────────────────────────────
HIGH_IMPACT_KW = ["人生", "改变", "永远", "再也", "完了", "崩溃",
                  "死", "活不下去", "结束", "一辈子", "毁了", "绝望"]
LOW_IMPACT_KW = ["小事", "没关系", "还好", "没什么", "一般",
                 "暂时", "偶尔"]


class MockLLM:
    """
    Mock LLM 服务 — 模拟大语言模型的各项能力。

    使用模板匹配和规则系统替代真实 LLM 调用，
    确保 EmoMem 框架可在无 API key 的环境下完整运行和测试。

    Mock LLM service — simulates LLM capabilities using templates and rules.
    Enables the full EmoMem framework to run without an actual LLM API.
    """

    def __init__(self, llm_provider=None):
        """初始化 Mock LLM，加载模板库 / Initialize with template libraries.

        Args:
            llm_provider: 可选的 LLM 提供者实例，支持 assess_context / analyze_message
                          方法。若提供，则优先使用 LLM 语义理解；若调用失败，
                          降级为关键词匹配。
                          Optional LLM provider instance with assess_context /
                          analyze_message methods. If provided, LLM semantic
                          understanding is tried first; falls back to keyword
                          matching on failure.
        """
        self.templates = STRATEGY_TEMPLATES
        self.emotion_words = EMOTION_WORDS
        self.llm_provider = llm_provider

    # ──────────────────────────────────────────
    # 核心方法：生成回复 (Core: generate response)
    # ──────────────────────────────────────────

    def generate_response(
        self,
        strategy: str,
        state_vector: StateVector,
        retrieved_context: Optional[RetrievedContext] = None,
        goals: Optional[GoalSet] = None,
    ) -> str:
        """
        根据选定策略生成中文模板回复。
        Generate a template-based Chinese response based on the selected strategy.

        Args:
            strategy: 从 Thompson Sampling 选出的策略名 (e.g., "active_listening")
            state_vector: 感知模块输出的状态向量
            retrieved_context: MAR 检索的上下文（可含情景记忆）
            goals: 规划模块推断的多层级目标

        Returns:
            中文回复文本 (Chinese response text)
        """
        # 均匀分布检测: 情绪不可靠时覆盖到安全策略
        # When emotion is unreliable (uniform distribution from LLM failure),
        # override to safe strategies that don't reference specific emotions
        e = state_vector.emotion.e
        e_max = float(np.max(e)) if len(e) > 0 else 0.0
        e_min = float(np.min(e)) if len(e) > 0 else 0.0
        if (e_max - e_min) < 0.05 and strategy in (
            "emotional_validation", "positive_reinforcement",
            "cognitive_reframing", "strength_recognition",
        ):
            # 这些策略的模板会用 {emotion_word}，均匀分布下可能错配
            strategy = "active_listening"

        # 获取该策略的模板列表，若策略未知则回退到 active_listening
        # Fallback to active_listening if strategy is unknown
        templates = self.templates.get(strategy, self.templates["active_listening"])

        # 随机选择一条模板 / Randomly select a template
        template = random.choice(templates)

        # ── 填充占位符 (Fill placeholders) ──

        # {topic}: 从 state_vector 提取话题关键词
        topic = "这件事"  # 默认占位 (default placeholder)
        if state_vector.topic_keywords:
            topic = "、".join(state_vector.topic_keywords[:2])
        elif state_vector.topic_category:
            topic = state_vector.topic_category

        # {emotion_word}: 从 EmotionVector 中选择主导情绪对应的中文词
        # 检测均匀分布 (LLM 失败兜底时 max ≈ 0.125): 使用安全的中性词
        e = state_vector.emotion.e
        e_max = float(np.max(e)) if len(e) > 0 else 0.0
        e_min = float(np.min(e)) if len(e) > 0 else 0.0
        is_uniform = (e_max - e_min) < 0.05  # 均匀分布检测
        if is_uniform:
            emotion_name = "unknown"
            emotion_word = "不好受"
        else:
            dominant_idx = int(np.argmax(e))
            emotion_name = EMOTIONS[dominant_idx]
            emotion_word = self.emotion_words.get(emotion_name, "不舒服")

        # {reflection}: 基于用户原始输入构造反射性表述
        reflection = self._build_reflection(state_vector)

        # {memory_ref}: 从检索上下文中提取可引用的记忆片段
        memory_ref = self._build_memory_reference(retrieved_context)

        # 执行模板替换 / Perform template substitution
        response = template.format(
            topic=topic,
            emotion_word=emotion_word,
            reflection=reflection,
            memory_ref=memory_ref,
        )

        # 如果有可用的记忆引用且策略适合引用，则追加记忆锚定
        # Append memory-grounded content if relevant memories are available
        if memory_ref and strategy in (
            "empathic_reflection", "strength_recognition",
            "cognitive_reframing", "gentle_guidance",
        ):
            response += f" {memory_ref}"

        return response

    def _build_reflection(self, state_vector: StateVector) -> str:
        """
        构建反射性表述 — 从原始输入中提炼关键内容回映给用户。
        Build reflective statement from user's raw input.
        """
        if state_vector.raw_signal and state_vector.raw_signal.text_content:
            text = state_vector.raw_signal.text_content
            # 截取前30字作为反射内容 (use first 30 chars for reflection)
            if len(text) > 30:
                return "你提到了\"" + text[:30] + "...\""
            else:
                return "你提到了\"" + text + "\""
        return "你提到的这些"

    def _build_memory_reference(
        self, retrieved_context: Optional[RetrievedContext]
    ) -> str:
        """
        从检索上下文中构造自然的记忆引用。
        Build natural memory reference from retrieved context.

        遵循 §5.2 记忆引用原则：
        - 自然引用（"上次你说到…" 而非 "根据我的记录…"）
        - 选择性引用（仅在相关时引用）
        - 情感连贯（匹配当前情绪）
        """
        if not retrieved_context or not retrieved_context.episodes:
            return ""

        # 选择最相关的一条记忆 (pick the most relevant episode)
        episode = retrieved_context.episodes[0]
        if not episode.event:
            return ""

        # 用自然语言引用 (natural language reference)
        event_brief = episode.event[:40] if len(episode.event) > 40 else episode.event
        return f"我记得你之前提到过{event_brief}。"

    # ──────────────────────────────────────────
    # 可控性评估 (Controllability Assessment)
    # ──────────────────────────────────────────

    def assess_controllability(self, text: str) -> float:
        """
        可控性评估 — 优先使用 LLM 语义理解，降级为关键词匹配。
        Controllability assessment — LLM semantic understanding first,
        falls back to keyword matching.

        §4.2: controllability ∈ [0, 1]
        高可控 = 用户感到有能力影响结果
        低可控 = 用户感到无力、被动

        Args:
            text: 用户输入文本

        Returns:
            controllability score ∈ [0.3, 0.7]
        """
        # LLM 模式: 语义理解可控性
        if self.llm_provider is not None:
            try:
                result = self.llm_provider.assess_context(text)
                return result.get("controllability", 0.5)
            except Exception:
                pass  # 降级为关键词匹配

        # 关键词匹配 (降级方案)
        score = 0.5  # 基准分 (baseline)

        # 检测高可控关键词 / Check for high-control keywords
        high_hits = sum(1 for kw in HIGH_CONTROL_KW if kw in text)
        # 检测低可控关键词 / Check for low-control keywords
        low_hits = sum(1 for kw in LOW_CONTROL_KW if kw in text)

        # 调整得分 / Adjust score
        score += high_hits * 0.08
        score -= low_hits * 0.08

        # 限制在 [0.3, 0.7] 范围 / Clamp to [0.3, 0.7]
        return float(np.clip(score, 0.3, 0.7))

    # ──────────────────────────────────────────
    # 生活影响评估 (Life Impact Assessment)
    # ──────────────────────────────────────────

    def assess_life_impact(self, text: str) -> float:
        """
        生活影响评估 — 优先使用 LLM 语义理解，降级为关键词匹配。
        Life impact assessment — LLM semantic understanding first,
        falls back to keyword matching.

        用于 定义 3.2 重要性评分的 LifeImpact 分量。

        Args:
            text: 用户输入文本

        Returns:
            life_impact score ∈ [0.3, 0.8]
        """
        # LLM 模式: 语义理解生活影响
        if self.llm_provider is not None:
            try:
                result = self.llm_provider.assess_context(text)
                return result.get("life_impact", 0.5)
            except Exception:
                pass  # 降级为关键词匹配

        # 关键词匹配 (降级方案)
        score = 0.5  # 基准分 (baseline)

        # 检测高影响关键词 / Check for high-impact keywords
        high_hits = sum(1 for kw in HIGH_IMPACT_KW if kw in text)
        # 检测低影响关键词 / Check for low-impact keywords
        low_hits = sum(1 for kw in LOW_IMPACT_KW if kw in text)

        score += high_hits * 0.1
        score -= low_hits * 0.08

        return float(np.clip(score, 0.3, 0.8))

    # ──────────────────────────────────────────
    # 话题提取 (Topic Extraction)
    # ──────────────────────────────────────────

    def extract_topics(self, text: str) -> Tuple[List[str], str]:
        """
        话题提取 — 优先使用 LLM 语义理解，降级为关键词匹配。
        Topic extraction — LLM semantic understanding first,
        falls back to keyword matching.

        Args:
            text: 用户输入文本

        Returns:
            (keywords, category) — 关键词列表 + 话题类别字符串
            keywords: 匹配到的关键词列表（最多5个）
            category: 最匹配的话题类别（如 "work", "relationship" 等）
        """
        # LLM 模式: 语义话题提取
        if self.llm_provider is not None:
            try:
                # Use analyze_message for topic extraction
                result = self.llm_provider.analyze_message(text)
                return result.topic_keywords, result.topic_category
            except Exception:
                pass  # 降级为关键词匹配

        # 关键词匹配 (降级方案)
        matched_keywords: List[str] = []
        category_scores: Dict[str, int] = {}

        # 遍历所有话题类别，统计关键词命中
        # Scan all topic categories and count keyword hits
        for cat, keywords in TOPIC_CATEGORIES.items():
            hits = [kw for kw in keywords if kw in text]
            if hits:
                matched_keywords.extend(hits)
                category_scores[cat] = len(hits)

        # 去重并限制数量 / Deduplicate and limit count
        seen = set()
        unique_keywords = []
        for kw in matched_keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        unique_keywords = unique_keywords[:5]

        # 选择命中最多的类别 / Select category with most hits
        if category_scores:
            category = max(category_scores, key=category_scores.get)
        else:
            category = "general"

        return unique_keywords, category

    # ──────────────────────────────────────────
    # 显式反馈检测 (Explicit Feedback Detection)
    # ──────────────────────────────────────────

    def detect_explicit_feedback(self, text: str) -> float:
        """
        显式反馈检测 — 优先使用 LLM 语义理解，降级为关键词匹配。
        Explicit feedback detection — LLM semantic understanding first,
        falls back to keyword matching.

        §3.4 定义:
        - "谢谢"、"说得对" → 高值
        - "算了"、"你不懂" → 低值
        - 无明确反馈 → 0.5

        Args:
            text: 用户输入文本

        Returns:
            explicit feedback score ∈ [0, 1]
        """
        # LLM 模式: 语义理解反馈信号
        if self.llm_provider is not None:
            try:
                result = self.llm_provider.assess_context(text)
                return result.get("explicit_feedback", 0.5)
            except Exception:
                pass  # 降级为关键词匹配

        # 关键词匹配 (降级方案)
        pos_hits = sum(1 for kw in POSITIVE_FEEDBACK_KW if kw in text)
        neg_hits = sum(1 for kw in NEGATIVE_FEEDBACK_KW if kw in text)

        if pos_hits > 0 and neg_hits == 0:
            return min(1.0, 0.6 + pos_hits * 0.1)
        elif neg_hits > 0 and pos_hits == 0:
            return max(0.0, 0.4 - neg_hits * 0.1)
        elif pos_hits > 0 and neg_hits > 0:
            # 混合信号：偏向中性 / Mixed signals: lean neutral
            return 0.5
        else:
            # 无显式反馈 / No explicit feedback
            return 0.5

    # ──────────────────────────────────────────
    # 会话摘要 (Session Summarization)
    # ──────────────────────────────────────────

    def summarize_session(self, buffer: str) -> str:
        """
        Mock 会话摘要 — 从会话缓冲区生成简要摘要。
        Mock session summarization from session buffer.

        真实场景下由 LLM 完成，此处提取前200字并添加固定前缀。

        Args:
            buffer: 会话历史文本缓冲区

        Returns:
            摘要文本 (summary text)
        """
        if not buffer or not buffer.strip():
            return "本次会话无实质内容。"

        # 截取关键部分 / Extract key portion
        lines = buffer.strip().split("\n")
        # 保留最多前10行 / Keep at most first 10 lines
        key_lines = lines[:10]
        summary_body = " ".join(line.strip() for line in key_lines if line.strip())

        # 限制长度 / Limit length
        if len(summary_body) > 200:
            summary_body = summary_body[:200] + "…"

        return f"会话摘要：{summary_body}"

    # ──────────────────────────────────────────
    # 参与度评估 (Engagement Assessment)
    # ──────────────────────────────────────────

    def assess_engagement(self, text: str, avg_length: float = 50.0) -> float:
        """
        评估用户参与度 — 基于回复长度和速度。
        Assess user engagement based on reply length.

        §3.4: engagement = min(1, w_e1 * length_ratio + w_e2 * speed_ratio)
        简化版仅使用长度比 (simplified: length ratio only).

        Args:
            text: 用户当前输入
            avg_length: 历史平均消息长度

        Returns:
            engagement score ∈ [0, 1]
        """
        if not text:
            return 0.0

        current_length = len(text)
        length_ratio = min(2.0, current_length / max(1.0, avg_length))

        # w_e1=0.5 用于长度, w_e2=0.5 用于速度（此处速度默认0.5）
        # w_e1=0.5 for length, w_e2=0.5 for speed (speed defaults to 0.5 here)
        engagement = min(1.0, 0.5 * length_ratio + 0.5 * 0.5)

        return engagement

    # ──────────────────────────────────────────
    # 话题延续判断 (Topic Continuation Detection)
    # ──────────────────────────────────────────

    def detect_topic_continuation(
        self, current_text: str, prev_keywords: List[str]
    ) -> float:
        """
        话题延续判断 — 优先使用 LLM 语义理解，降级为关键词匹配。
        Topic continuation detection — LLM semantic understanding first,
        falls back to keyword matching.

        §3.4: continuation ∈ {0, 1} — 二值指标。

        Args:
            current_text: 当前用户输入
            prev_keywords: 上一轮的话题关键词

        Returns:
            continuation score (0.0 or 1.0)
        """
        # LLM 模式: 语义理解话题延续
        if self.llm_provider is not None:
            try:
                result = self.llm_provider.assess_context(
                    current_text, prev_topics=prev_keywords
                )
                return result.get("topic_continuation", 0.5)
            except Exception:
                pass  # 降级为关键词匹配

        # 关键词匹配 (降级方案)
        if not prev_keywords:
            return 0.5  # 首轮无参照 / No reference for first turn

        # 检查上一轮关键词是否出现在当前输入中
        # Check if previous keywords appear in current input
        hits = sum(1 for kw in prev_keywords if kw in current_text)
        return 1.0 if hits > 0 else 0.0
