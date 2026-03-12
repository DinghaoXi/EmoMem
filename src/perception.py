"""
EmoMem 感知模块 (Perception Module)
====================================
来源: emomem.md §2 (定义 2.1–2.7)

感知模块将用户原始文本输入转化为结构化的 StateVector，
包含显式情感信号、隐式行为信号、意图分类和话题关键词。

六个子组件串联处理:
  1. InputParser        — 解析文本为 RawSignal (标点、emoji、时间特征)
  2. EmotionRecognizer  — 输出 EmotionVector (8维Plutchik分布 + 强度 + 确定度)
  3. IntentClassifier   — 输出 IntentDistribution (7类意图概率)
  4. ImplicitSignalDetector — 计算 BAS 行为异常度评分
  5. TopicExtractor     — 提取话题关键词与类别
  6. ContextEncoder     — 融合所有子组件输出为 StateVector (含 urgency 计算)

Architecture (§2.1 图2):
  User Input → InputParser → RawSignal
                  ↓
    ┌─────────────┼─────────────┬──────────────┐
    ↓             ↓             ↓              ↓
  Emotion    Intent       Implicit        Topic
  Recognizer Classifier   SignalDetector  Extractor
    ↓             ↓             ↓              ↓
    └─────────────┴─────────────┴──────────────┘
                         ↓
                   ContextEncoder → StateVector

"""

from __future__ import annotations

import logging
import re
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from .config import (
    # §2.1 Plutchik 情绪空间
    EMOTIONS, NUM_EMOTIONS, VALENCE_SIGNS,
    # §2.4 意图分类
    INTENT_TYPES, NUM_INTENTS,
    # §2.5 隐式信号检测
    BEHAVIOR_FEATURES, BAS_WEIGHTS, BAS_MAX, EPSILON,
    BEHAVIOR_COLD_START_DEFAULTS, BEHAVIOR_SIGMA_MIN,
    # §2.7 紧急度评分
    URGENCY_ALPHA1, URGENCY_ALPHA2, URGENCY_ALPHA4,
    CRISIS_THRESHOLD, CRISIS_STEP_THRESHOLD,
    # α_3 冷启动渐进参数
    ALPHA3_FULL, ALPHA3_RAMP_INTERACTIONS,
    # §2.7b 情绪惯性
    EMOTION_INERTIA_ALPHA,
    CRISIS_COOLDOWN_TURNS, CRISIS_COOLDOWN_IMPLICIT_TURNS, CRISIS_COOLDOWN_DECAY,
    POST_CRISIS_SENSITIVITY_ALPHA, POST_CRISIS_URGENCY_FLOOR,
    CONTEXT_INTENT_WEIGHT, INTENT_CONTINUITY_THRESHOLD,
    # §2 感知模块集中参数
    KEYWORD_CRISIS_FLOOR_THRESHOLD, KEYWORD_CRISIS_DAMPENING,
    KL_UNIFORM_THRESHOLD, MAX_KEYWORD_SCORE_PER_INTENT,
    POSITIVE_CRISIS_SUPPRESSION_STRONG, POSITIVE_CRISIS_SUPPRESSION_MILD,
    MAX_CONVERSATION_HISTORY,
)

from .models import (
    # 数据结构
    RawSignal, PunctuationFeatures, ParalinguisticFeatures, TemporalFeatures,
    EmotionVector, IntentDistribution, ImplicitSignals, StateVector,
    BehavioralBaseline, WorkingMemory,
)


# ════════════════════════════════════════════════════════════════
# 模块级共享常量与工具函数
# Shared module-level constants and utility functions
# ════════════════════════════════════════════════════════════════

# ── 正面语境指标 (用于抑制隐含危机关键词的假阳性) ──
# Positive context indicators that suppress implicit crisis false positives.
# When these co-occur with implicit crisis keywords (weight 2.5-3.0),
# they signal the keywords are used in benign context (e.g., "该还的都还了哈哈").
_POSITIVE_CONTEXT_INDICATORS: List[str] = [
    "哈哈", "嘻嘻", "嘿嘿", "哈哈哈", "lol", "233",
    "太好了", "开心", "高兴", "棒", "耶",
    "下次", "以后", "明天", "周末", "一起", "约",  # future plans
    "请你来", "来玩", "请客",
    "～", "~", "！！",  # playful tone markers
    "预防", "纪录片", "研究", "报告", "科普", "很有意义", "学到",
    "新闻", "知识", "课程", "讲座", "工作坊",  # educational/academic context
]

# Negation prefixes — indicators must not match when preceded by these
# Chinese negation is prefix-based (e.g., "没有意义" contains "有意义")
_NEGATION_PREFIXES: tuple = ("没", "不", "无", "别", "非")


def _safe_indicator_match(ind: str, txt: str) -> bool:
    """Check if indicator appears in text without negation prefix.
    Iterates all occurrences — if ANY is un-negated, returns True.
    E.g. '没有意义，但研究很有意义' → '有意义' first negated, second not → True."""
    start = 0
    while True:
        idx = txt.find(ind, start)
        if idx < 0:
            return False
        if idx == 0 or txt[idx - 1] not in _NEGATION_PREFIXES:
            return True
        start = idx + 1


# ════════════════════════════════════════════════════════════════
# §2.2 子组件一：多模态输入解析器 (Multi-modal Input Parser)
# ════════════════════════════════════════════════════════════════

class InputParser:
    """
    §2.2 多模态输入解析器
    Multi-modal Input Parser

    从纯文本中提取副语言特征 (paralinguistic features) 和时间特征:
    - 标点特征: 省略号数量、感叹号数量、问号数量
    - Emoji 模式: 提取所有 emoji 字符
    - 时间特征: 消息时间戳、与上条消息间隔

    即使在纯文本场景下，用户的表达方式也承载丰富的副语言信息。
    表情符号使用模式、标点选择（省略号暗示犹豫，感叹号暗示强烈情绪）、
    消息分段方式等均为重要信号源。
    """

    # ── Emoji 正则：匹配 Unicode emoji 范围 ──
    # Matches common emoji code point ranges
    _EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"   # emoticons / 表情符号
        "\U0001F300-\U0001F5FF"   # symbols & pictographs / 符号图片
        "\U0001F680-\U0001F6FF"   # transport & map / 交通地图
        "\U0001F1E0-\U0001F1FF"   # flags / 国旗
        "\U00002702-\U000027B0"   # dingbats / 装饰符号
        "\U0001F900-\U0001F9FF"   # supplemental symbols / 补充符号
        "\U0001FA00-\U0001FA6F"   # chess symbols / 棋类符号
        "\U0001FA70-\U0001FAFF"   # symbols extended-A / 扩展符号A
        "\U00002600-\U000026FF"   # misc symbols / 杂项符号
        "]+",
        flags=re.UNICODE,
    )

    # ── 中英文省略号 ──
    # Chinese ellipsis (……) and ASCII ellipsis (... or …)
    _ELLIPSIS_PATTERN = re.compile(r"\.{3,}|…|……")

    def parse(
        self,
        text: str,
        timestamp: datetime,
        last_message_time: Optional[datetime] = None,
        session_start: Optional[datetime] = None,
    ) -> RawSignal:
        """
        解析用户文本输入为 RawSignal 结构。
        Parse user text input into a RawSignal structure.

        Parameters
        ----------
        text : str
            用户输入文本 / User input text
        timestamp : datetime
            当前消息发送时间 / Current message timestamp
        last_message_time : Optional[datetime]
            上一条消息的时间 (用于计算间隔) / Previous message time
        session_start : Optional[datetime]
            当前会话开始时间 / Session start time

        Returns
        -------
        RawSignal
            结构化的原始信号
        """
        # ── 标点特征 (Punctuation Features) ──
        # 省略号: 匹配连续三个以上点号, 或 Unicode 省略号字符, 或中文省略号
        ellipsis_count = len(self._ELLIPSIS_PATTERN.findall(text))
        # 感叹号: 包含中文和英文感叹号 / Count both English and Chinese exclamation marks
        exclamation_count = text.count("!") + text.count("！")
        # 问号: 包含中文和英文问号 / Count both English and Chinese question marks
        question_marks = text.count("?") + text.count("？")

        punctuation = PunctuationFeatures(
            ellipsis_count=ellipsis_count,
            exclamation_count=exclamation_count,
            question_marks=question_marks,
        )

        # ── Emoji 提取 ──
        # 提取文本中所有 emoji 字符
        emoji_list = self._EMOJI_PATTERN.findall(text)

        # ── 副语言特征 (Paralinguistic Features) ──
        paralinguistic = ParalinguisticFeatures(
            emoji_pattern=emoji_list,
            punctuation_features=punctuation,
            message_segments=1,  # 单条消息默认为1段 / Default 1 segment for single message
            character_count=len(text),
        )

        # ── 时间特征 (Temporal Features) ──
        since_last = None
        if last_message_time is not None:
            since_last = timestamp - last_message_time

        session_dur = timedelta(0)
        if session_start is not None:
            session_dur = timestamp - session_start

        temporal = TemporalFeatures(
            timestamp=timestamp,
            since_last_message=since_last,
            session_duration=session_dur,
        )

        return RawSignal(
            text_content=text,
            paralinguistic=paralinguistic,
            temporal=temporal,
        )


# ════════════════════════════════════════════════════════════════
# §2.3 子组件二：细粒度情绪识别器 (Fine-grained Emotion Recognizer)
# ════════════════════════════════════════════════════════════════

class EmotionRecognizer:
    """
    §2.3 细粒度情绪识别器 (Mock 实现)
    Fine-grained Emotion Recognizer — Mock Implementation

    输出 EmotionVector = ⟨e(t), ι(t), κ(t)⟩ (定义 2.2):
    - e(t): 8维归一化情绪分布 (Plutchik), 位于 Δ⁷ 单纯形上
    - ι(t): 情绪强度 / intensity ∈ [0, 1]
    - κ(t): 识别确定度 / confidence ∈ [0, 1]

    Mock 策略: 使用关键词映射 (中文 + 英文) 将文本映射为情绪分布。
    支持混合情绪 — 多个关键词同时匹配时产生混合分布。
    情绪强度综合考虑词汇情感强度和副语言信号 (感叹号等)。

    Plutchik 8 维:
      0=joy, 1=trust, 2=fear, 3=surprise, 4=sadness, 5=disgust, 6=anger, 7=anticipation

    次级情绪映射 (Plutchik 1980; refined 2001):
      anxiety ≈ fear + anticipation
      guilt ≈ joy↓ + fear + sadness
      love ≈ joy + trust
    """

    # ── 关键词 → 情绪映射表 ──
    # 每个条目: (关键词列表, 情绪索引, 权重, 强度贡献, 确定度贡献)
    # Each entry: (keyword_list, emotion_index, weight, intensity_contribution, confidence)
    #
    # 情绪索引: 0=joy, 1=trust, 2=fear, 3=surprise, 4=sadness, 5=disgust, 6=anger, 7=anticipation
    _KEYWORD_RULES: List[Tuple[List[str], int, float, float, float]] = [
        # ── Joy / 快乐 ──
        (["happy", "开心", "高兴", "快乐", "嘻嘻", "哈哈", "太好了", "棒",
          "wonderful", "great", "awesome", "excellent", "glad", "delighted",
          "幸福", "愉快", "爽", "nice", "good"],
         0, 2.0, 0.6, 0.8),

        # ── Trust / 信任 ──
        (["trust", "信任", "相信", "依赖", "靠谱", "放心", "安心",
          "reliable", "confident", "faith", "believe"],
         1, 1.5, 0.4, 0.7),

        # ── Fear / 恐惧 ──
        (["afraid", "害怕", "恐惧", "担心", "焦虑", "紧张", "怕",
          "scared", "fear", "anxious", "worried", "nervous", "panic",
          "不安", "恐慌", "惶恐", "terrified"],
         2, 2.0, 0.7, 0.8),

        # ── Surprise / 惊讶 ──
        (["surprise", "惊讶", "震惊", "没想到", "意外", "天哪",
          "shocked", "amazing", "unexpected", "wow", "omg", "不敢相信",
          "吃惊", "我天"],
         3, 1.5, 0.5, 0.7),

        # ── Sadness / 悲伤 ──
        (["sad", "难过", "伤心", "悲伤", "哭", "委屈", "失落", "痛苦",
          "unhappy", "depressed", "miserable", "heartbroken", "grieving",
          "upset", "down", "低落", "沮丧", "郁闷", "想哭", "泪", "心痛",
          "难受", "崩溃", "失败", "天塌", "完了", "无望", "绝望",
          "无助", "撑不住", "活着累", "没意思", "不配", "丢了"],
         4, 2.0, 0.7, 0.8),

        # ── Disgust / 厌恶 ──
        (["disgusted", "恶心", "厌恶", "讨厌", "反感", "受不了",
          "gross", "repulsed", "hate", "loathe", "鄙视", "嫌弃"],
         5, 1.5, 0.6, 0.7),

        # ── Anger / 愤怒 ──
        (["angry", "生气", "愤怒", "烦", "气死", "火大", "暴怒",
          "furious", "mad", "rage", "irritated", "pissed", "outraged",
          "恼火", "恼怒", "可恶", "混蛋", "王八蛋", "妈的",
          "丢人", "丢脸", "窝囊", "受够了", "凭什么",
          "骂", "挨骂", "被骂"],
         6, 2.0, 0.7, 0.8),

        # ── Anticipation / 期待 ──
        (["anticipation", "期待", "期望", "盼望", "希望", "等待",
          "looking forward", "hope", "expect", "excited", "兴奋",
          "迫不及待", "想要"],
         7, 1.5, 0.5, 0.7),
    ]

    # ── 次级情绪映射 (§2.3 次级情绪映射) ──
    # Secondary emotion → primary emotion decomposition
    # anxiety ≈ fear + anticipation
    # guilt ≈ fear + sadness  (joy↓ 通过降低 joy 权重实现)
    # love ≈ joy + trust
    # contempt ≈ disgust + anger
    # jealousy ≈ anger + sadness
    _SECONDARY_EMOTIONS: List[Tuple[List[str], List[Tuple[int, float]], float, float]] = [
        # (关键词列表, [(情绪索引, 权重), ...], 强度, 确定度)
        (["anxious", "焦虑", "anxiety", "不安感"],
         [(2, 1.5), (7, 1.0)], 0.65, 0.75),           # fear + anticipation

        (["guilty", "内疚", "guilt", "自责", "愧疚"],
         [(2, 1.0), (4, 1.5)], 0.6, 0.7),             # fear + sadness

        (["love", "爱", "喜欢你", "爱你", "暗恋"],
         [(0, 1.5), (1, 1.5)], 0.5, 0.7),             # joy + trust

        (["contempt", "鄙夷", "看不起", "蔑视"],
         [(5, 1.2), (6, 1.0)], 0.6, 0.7),             # disgust + anger

        (["jealous", "嫉妒", "羡慕嫉妒恨", "眼红"],
         [(6, 1.2), (4, 1.0)], 0.6, 0.7),             # anger + sadness

        (["lonely", "孤独", "寂寞", "孤单", "一个人"],
         [(4, 1.5), (2, 0.5)], 0.55, 0.7),            # sadness + mild fear

        (["hopeless", "绝望", "没希望", "无望", "走投无路"],
         [(4, 2.0), (2, 1.0)], 0.85, 0.85),           # sadness + fear (high intensity)

        (["frustrated", "挫败", "无奈", "受挫"],
         [(6, 1.2), (4, 1.0)], 0.6, 0.7),             # anger + sadness
    ]

    # ── 讽刺/反语标记词 (sarcasm/irony markers) ──
    # Chinese sarcasm often pairs ironic positive words with negative events.
    # These markers indicate the speaker is being sarcastic, not genuinely positive.
    _SARCASM_MARKERS = [
        "呵呵", "可真行", "可真是", "真是太",
        "真是完美", "完美的一天", "太幸运了", "太幸福了",
        "我简直", "简直了", "好极了呢", "太妙了",
        "真是够了", "也是醉了", "我也是服了", "服了",
        "谢谢你呢", "真是谢谢",
    ]

    # ── 强度修饰词 (intensity modifiers) ──
    # 极高强度词 / Very high intensity markers
    _HIGH_INTENSITY_WORDS = [
        "非常", "特别", "极其", "太", "超级", "完全", "彻底",
        "very", "extremely", "incredibly", "absolutely", "totally",
        "so", "really", "super", "真的", "简直", "太tm",
    ]
    # 低强度词 / Low intensity markers
    _LOW_INTENSITY_WORDS = [
        "有点", "稍微", "一点点", "略微", "些许",
        "a bit", "slightly", "somewhat", "a little", "kind of",
        "还好", "一般",
    ]

    def recognize(self, raw_signal: RawSignal) -> EmotionVector:
        """
        从 RawSignal 识别情绪，输出 EmotionVector。
        Recognize emotions from RawSignal, outputting an EmotionVector.

        Parameters
        ----------
        raw_signal : RawSignal
            InputParser 输出的原始信号 / Raw signal from InputParser

        Returns
        -------
        EmotionVector
            情绪向量 ⟨e(t), ι(t), κ(t)⟩
        """
        text = raw_signal.text_content.lower()

        # ── 初始化 8 维情绪权重为零向量 ──
        # Initialize 8-dim emotion weight accumulator
        emotion_weights = np.zeros(NUM_EMOTIONS, dtype=np.float64)
        intensity_accum = 0.0   # 强度累加器 / intensity accumulator
        confidence_accum = 0.0  # 确定度累加器 / confidence accumulator
        match_count = 0         # 匹配计数 / number of keyword matches

        # ── 主要情绪关键词匹配 ──
        # Primary emotion keyword matching
        for keywords, emo_idx, weight, intensity, confidence in self._KEYWORD_RULES:
            for kw in keywords:
                if kw in text:
                    emotion_weights[emo_idx] += weight
                    intensity_accum += intensity
                    confidence_accum += confidence
                    match_count += 1
                    break  # 每组关键词只匹配一次 / Match each group once

        # ── 次级情绪匹配 ──
        # Secondary (compound) emotion matching
        for keywords, emo_pairs, intensity, confidence in self._SECONDARY_EMOTIONS:
            for kw in keywords:
                if kw in text:
                    for emo_idx, weight in emo_pairs:
                        emotion_weights[emo_idx] += weight
                    intensity_accum += intensity
                    confidence_accum += confidence
                    match_count += 1
                    break  # 每组次级情绪只匹配一次

        # ── 讽刺/反语检测 (sarcasm detection) ──
        # When sarcasm markers are present AND both positive (joy/trust/anticipation)
        # and negative (fear/sadness/disgust/anger) keyword groups matched,
        # the positive words are likely ironic. Dampen positive weights and
        # boost negative weights to produce a more accurate mixed/negative signal.
        # Positive indices: 0=joy, 1=trust, 7=anticipation
        # Negative indices: 2=fear, 4=sadness, 5=disgust, 6=anger
        has_sarcasm_marker = any(marker in text for marker in self._SARCASM_MARKERS)
        has_positive_match = any(emotion_weights[i] > 0 for i in (0, 1, 7))
        has_negative_match = any(emotion_weights[i] > 0 for i in (2, 4, 5, 6))
        if has_sarcasm_marker and has_positive_match and has_negative_match:
            # Sarcasm detected: positive words are ironic, dampen them
            for i in (0, 1, 7):  # joy, trust, anticipation
                emotion_weights[i] *= 0.3  # reduce positive weights to 30%
            for i in (2, 4, 5, 6):  # fear, sadness, disgust, anger
                emotion_weights[i] *= 1.5  # boost negative weights by 50%
            # Lower confidence since sarcasm introduces ambiguity
            confidence_accum *= 0.8

        # ── 强度修饰 ──
        # Intensity modifiers: 强度修饰词放大/缩小情绪强度
        intensity_modifier = 1.0
        has_high = any(word in text for word in self._HIGH_INTENSITY_WORDS)
        has_low = any(word in text for word in self._LOW_INTENSITY_WORDS)
        if has_high and not has_low:
            intensity_modifier = 1.3  # 放大30%
        elif has_low and not has_high:
            intensity_modifier = 0.6  # 缩小40%
        # 若同时包含高/低修饰词, 保持中性 1.0 (矛盾信号不做修饰)

        # ── 副语言信号对强度的影响 ──
        # Paralinguistic signals influence intensity (§2.3 实现路径)
        # 感叹号增加情绪强度 / Exclamation marks increase intensity
        excl = raw_signal.paralinguistic.punctuation_features.exclamation_count
        if excl >= 3:
            intensity_modifier *= 1.2   # 3个以上感叹号: 再放大20%
        elif excl >= 1:
            intensity_modifier *= 1.1   # 1-2个感叹号: 放大10%

        # Emoji 数量影响确定度 / Emoji count affects confidence
        emoji_count = len(raw_signal.paralinguistic.emoji_pattern)

        # ── 汇总结果 ──
        if match_count == 0:
            # 无关键词匹配: 返回均匀分布、低强度、低确定度
            # No keyword match: return uniform distribution with low intensity/confidence
            e = np.ones(NUM_EMOTIONS, dtype=np.float64) / NUM_EMOTIONS
            intensity = 0.1
            confidence = 0.3
        else:
            # ── 归一化到 Δ⁷ 单纯形 ──
            # Normalize to the 7-simplex: Σe_i = 1, e_i >= 0
            total = emotion_weights.sum()
            if total > 0:
                e = emotion_weights / total
            else:
                e = np.ones(NUM_EMOTIONS, dtype=np.float64) / NUM_EMOTIONS

            # 平均强度，经修饰后裁剪到 [0, 1]
            # Average intensity, apply modifier, clamp to [0, 1]
            intensity = np.clip(
                (intensity_accum / match_count) * intensity_modifier, 0.0, 1.0
            )

            # 平均确定度, emoji 增加确定度 (情绪表达更明确)
            # Average confidence, emoji presence boosts it
            confidence = np.clip(
                (confidence_accum / match_count) + min(emoji_count * 0.05, 0.15),
                0.0, 1.0,
            )

        return EmotionVector(e=e, intensity=float(intensity), confidence=float(confidence))


# ════════════════════════════════════════════════════════════════
# §2.4 子组件三：意图分类器 (Intent Classifier)
# ════════════════════════════════════════════════════════════════

class IntentClassifier:
    """
    §2.4 意图分类器 (Mock 实现)
    Intent Classifier — Mock Implementation

    输出 IntentDistribution: 7 类意图的非互斥概率分布。
    Non-exclusive probability distribution over 7 intent types.

    7 种情感陪伴意图 (表3):
      VENT       — 倾诉宣泄 (被听到、被接纳)
      COMFORT    — 寻求安慰 (情感支持、被关怀)
      ADVICE     — 寻求建议 (解决方案)
      SHARE_JOY  — 分享喜悦 (共同庆祝)
      CHAT       — 陪伴闲聊 (消磨孤独)
      REFLECT    — 深度自省 (协助思考)
      CRISIS     — 危机信号 (安全干预)

    Mock 策略: 关键词匹配产生原始分数, 然后 softmax 归一化为概率。
    """

    # ── 意图关键词映射 ──
    # (intent_name, keywords_list, raw_score_when_matched)
    _INTENT_KEYWORDS: List[Tuple[str, List[str], float]] = [
        # ── VENT / 倾诉宣泄 ──
        ("VENT", [
            "烦", "烦死了", "受不了", "忍不了", "气死", "崩溃", "心累",
            "想吐槽", "吐槽", "vent", "frustrated", "fed up", "annoyed",
            "郁闷", "窒息", "无语", "服了",
        ], 3.0),

        # ── COMFORT / 寻求安慰 ──
        ("COMFORT", [
            "安慰", "我是不是很差", "我好差", "我不好", "难过", "伤心",
            "怎么办", "无助", "我很害怕", "comfort", "hold me", "陪陪我",
            "抱抱", "能不能陪我", "我需要你",
        ], 3.0),

        # ── ADVICE / 寻求建议 ──
        ("ADVICE", [
            "怎么办", "该怎么", "建议", "你觉得", "应该", "有什么办法",
            "advice", "suggest", "what should", "how to", "how can",
            "解决", "帮我想想", "出主意", "指点",
        ], 3.0),

        # ── SHARE_JOY / 分享喜悦 ──
        ("SHARE_JOY", [
            "太好了", "通过了", "面试通过", "成功了", "拿到offer", "升职",
            "录取", "中了", "好消息", "开心", "高兴",
            "great news", "got the job", "passed", "accepted", "promoted",
            "加薪", "终于", "梦想成真",
        ], 3.0),

        # ── CHAT / 陪伴闲聊 ──
        ("CHAT", [
            "你在干嘛", "在吗", "无聊", "聊聊", "随便聊聊", "你好",
            "hi", "hello", "hey", "what's up", "在干什么", "闲着",
            "chat", "你在吗", "嗨",
        ], 2.0),

        # ── REFLECT / 深度自省 ──
        ("REFLECT", [
            "我一直在想", "为什么", "反思", "思考", "想不通", "纠结",
            "我到底", "人生", "意义", "价值", "方向",
            "reflect", "thinking about", "wondering", "confused",
            "自我", "内心", "深层",
        ], 2.5),

        # ── CRISIS / 危机信号 ──
        ("CRISIS", [
            # 显式危机关键词 (高权重)
            "不想活", "活着好累", "自杀", "想死", "结束一切", "跳楼", "跳下去",
            "割腕", "吞安眠药", "吃了很多安眠药", "了结", "没有意义", "离开这个世界",
            "suicidal", "kill myself", "end it all", "want to die",
            "self harm", "自残", "不想活了", "算了吧都",
            # 药物相关危机信号
            "吃了很多药", "把药都吃了", "一次性吃完", "停药了不想吃了",
            # 药物准备/囤积场景 — 行动意图关键词（利用子串匹配，无需穷举排列）
            "安眠药准备", "准备了安眠药", "准备好了安眠药",
            "买了安眠药", "囤了安眠药", "攒了安眠药",
            "准备好了药", "把药准备好",
        ], 5.0),  # 危机意图给予最高原始分数 / Highest raw score for crisis

        # ── CRISIS_IMPLICIT / 隐含危机信号 ──
        # 告别行为、存在性危机等隐含自杀信号
        # 使用更具体的短语避免误报 (如"想通了"单独出现可以是正面含义)
        # 权重 2.5: 单个匹配足以提升 CRISIS 概率, 多个叠加则确定触发
        ("CRISIS", [
            "都想通了", "什么都想通了",             # "想通了"单独太歧义
            "都整理好了",                            # "整理好了"单独太歧义
            "该还的都还了", "人情都还了", "人情也还了",
            "活着到底", "活着为了什么",              # "人活着"单独太歧义
            "谢谢你陪我", "再见了", "最后一次",
            "不会再麻烦", "不用担心我了", "以后不会了",
            "不会再给任何人添麻烦",
        ], 2.5),  # 隐含信号权重低于显式(5.0)但足以触发

        # ── CRISIS_PASSIVE / 被动自杀意念 ──
        # 被动死亡愿望: "如果消失/没醒来也无所谓" 等
        # 权重 3.0: 介于隐含(2.5)和显式(5.0)之间
        # 设计原则: 仅收录不太可能在日常正面语境出现的短语
        # 已排除: "少我一个"(日常用语"少我一个不少"), "没有我也一样"(自嘲常见)
        ("CRISIS", [
            "消失了也没人",                          # "消失了也没人会注意到"
            "有没有我都一样",                        # perceived burdensomeness
            "没醒来也无所谓", "不用醒来",            # passive death wish
            "醒不过来也好", "不醒来也好",
            "不在了也没人", "走了也没人",            # "不在了也没人在意"
            "如果我不在了",                          # self-erasure (保守选择)
        ], 3.0),  # 被动SI权重: 2.5 < 3.0 < 5.0

        # ── CRISIS_DRUG_MENTION / 药物名称单独提及 ──
        # "安眠药" 单独出现为中低风险信号
        # 降权理由: 日常医疗讨论(失眠处方)也会提及, LLM 语义分析可区分
        # 权重 2.5: 与隐含信号同级, 由 LLM crisis_context 字段仲裁
        ("CRISIS", [
            "安眠药",                                    # 单独提及需 LLM 语义验证
        ], 2.5),
    ]

    @classmethod
    def _compute_keyword_raw_scores(cls, text: str) -> Dict[str, float]:
        """Compute raw keyword-based scores for each intent type.

        Shared step 1: keyword matching with capped accumulation.
        Used by both ``classify()`` and
        ``PerceptionModule._compute_keyword_crisis_score()``.

        Parameters
        ----------
        text : str
            User input text, **already lowercased**.

        Returns
        -------
        Dict[str, float]
            Mapping from intent name to raw keyword score (pre-softmax).
        """
        raw_scores = {intent: 0.5 for intent in INTENT_TYPES}

        for intent_name, keywords, score in cls._INTENT_KEYWORDS:
            for kw in keywords:
                if kw in text:
                    raw_scores[intent_name] = min(
                        raw_scores[intent_name] + score,
                        MAX_KEYWORD_SCORE_PER_INTENT,
                    )

        return raw_scores

    @staticmethod
    def _softmax_normalize(raw_scores: Dict[str, float]) -> Dict[str, float]:
        """Softmax-normalize raw intent scores to probabilities.

        Shared step 2: numerically stable softmax normalization.
        Used by both ``classify()`` and
        ``PerceptionModule._compute_keyword_crisis_score()``.

        Parameters
        ----------
        raw_scores : Dict[str, float]
            Raw keyword scores per intent (from ``_compute_keyword_raw_scores``).

        Returns
        -------
        Dict[str, float]
            Mapping from intent name to softmax probability.
        """
        scores_arr = np.array([raw_scores[it] for it in INTENT_TYPES], dtype=np.float64)
        # 数值稳定性: 减去最大值防止 exp 溢出 / Numerical stability
        scores_arr -= scores_arr.max()
        exp_scores = np.exp(scores_arr)
        exp_sum = exp_scores.sum()
        if exp_sum > 0:
            probs = exp_scores / exp_sum
        else:
            probs = np.ones(len(INTENT_TYPES), dtype=np.float64) / len(INTENT_TYPES)

        return {
            intent: float(probs[i]) for i, intent in enumerate(INTENT_TYPES)
        }

    def classify(self, raw_signal: RawSignal, keyword_only: bool = False) -> IntentDistribution:
        """
        分类用户意图，输出 7 类意图的概率分布。
        Classify user intent into a probability distribution over 7 types.

        Parameters
        ----------
        raw_signal : RawSignal
            InputParser 输出的原始信号
        keyword_only : bool
            When True, enables positive context suppression of implicit
            crisis keywords. This should only be True in Tier 3 (pure keyword
            mode) where no LLM is available for contextual arbitration.
            Default False — in LLM tiers, suppression is handled by
            crisis_context arbitration instead.

        Returns
        -------
        IntentDistribution
            7 类意图的非互斥概率分布
        """
        text = raw_signal.text_content.lower()

        # ── Step 1: 关键词评分 ──
        raw_scores = self._compute_keyword_raw_scores(text)

        # ── 正面语境抑制 CRISIS 假阳性 (raw-score level) ──
        # Only run in keyword-only mode (Tier 3). In LLM tiers,
        # crisis_context arbitration handles false positives instead.
        # When implicit crisis keywords (weight 2.5) co-occur with positive
        # context indicators (laughter, future plans, playful tone), the user
        # is likely using the phrase in benign context. Dampen CRISIS score.
        if keyword_only:
            crisis_score = raw_scores.get("CRISIS", 0.5)
            # Only suppress implicit crisis keywords (weight 2.5-3.0, score < 4.0).
            # Explicit keywords (weight 5.0, score >= 5.5) are NOT suppressed here —
            # in the keyword-only path (no LLM), explicit crisis must NEVER be suppressed.
            if 1.0 < crisis_score < 4.0:
                positive_count = sum(
                    1 for ind in _POSITIVE_CONTEXT_INDICATORS
                    if _safe_indicator_match(ind, text)
                )
                if positive_count >= 2:
                    raw_scores["CRISIS"] = 0.5  # Reset to baseline
                elif positive_count == 1:
                    raw_scores["CRISIS"] = max(0.5, crisis_score * POSITIVE_CRISIS_SUPPRESSION_MILD)

        # ── Step 2: Softmax 归一化 ──
        probabilities = self._softmax_normalize(raw_scores)

        return IntentDistribution(probabilities=probabilities)


# ════════════════════════════════════════════════════════════════
# §2.5 子组件四：隐式信号检测器 (Implicit Signal Detector)
# ════════════════════════════════════════════════════════════════

class ImplicitSignalDetector:
    """
    §2.5 隐式信号检测器
    Implicit Signal Detector

    聚焦于行为模式偏差 — "什么时候说" 和 "怎么说" 往往比 "说了什么" 更能反映真实情绪状态。

    核心公式 — 定义 2.5 (行为异常度评分, BAS):
      BAS(t) = Σ_{i=1}^{6} w_i · |b_i^current(t) - b̄_i^baseline| / (σ_i^baseline + ε)

    其中:
      - b_i^current: 当前行为特征值
      - b̄_i^baseline: 行为基线均值 (从语义记忆 Tier 3 获取)
      - σ_i^baseline: 行为基线标准差
      - w_i: 权重向量 (表4), Σw_i = 1
      - ε = 1e-6: 防除零常量

    6 维行为特征 (定义 2.4):
      b1=msg_time, b2=reply_interval, b3=msg_frequency,
      b4=msg_length, b5=emoji_rate, b6=session_init

    冷启动处理 (§2.5):
      α_3(n_cum) = min(0.6, 0.6 * (n_cum - 1) / 5)
      n_cum=1 时 α_3=0 (不计入urgency), n_cum=6 时完全恢复
    """

    def detect(
        self,
        raw_signal: RawSignal,
        behavioral_baseline: BehavioralBaseline,
        current_behavior: Optional[Dict[str, float]] = None,
    ) -> ImplicitSignals:
        """
        检测隐式行为信号，计算 BAS 行为异常度评分。
        Detect implicit behavioral signals and compute the BAS score.

        Parameters
        ----------
        raw_signal : RawSignal
            InputParser 输出的原始信号
        behavioral_baseline : BehavioralBaseline
            用户行为基线 (从语义记忆 Tier 3 获取)
            Contains mean (b̄_i) and std (σ_i) for each behavior dimension
        current_behavior : Optional[Dict[str, float]]
            当前行为特征值。如果为 None, 则从 raw_signal 推断部分特征。
            Current behavioral feature values. If None, infer from raw_signal.

        Returns
        -------
        ImplicitSignals
            包含 bas_score, anomaly_dimensions, temporal_context
        """
        # ── 从 raw_signal 推断当前行为特征 ──
        # Infer current behavior features from raw_signal if not provided
        if current_behavior is None:
            current_behavior = self._infer_behavior(raw_signal)

        # ── 获取基线均值和标准差 ──
        baseline_mean = behavioral_baseline.mean
        baseline_std = behavioral_baseline.std

        # ── 计算 BAS (定义 2.5) ──
        # BAS(t) = Σ w_i · |b_i - b̄_i| / (σ_i + ε)
        bas_score = 0.0
        anomaly_dimensions: List[str] = []

        for i, feat_name in enumerate(BEHAVIOR_FEATURES):
            b_current = current_behavior.get(feat_name, baseline_mean.get(feat_name, 0.0))
            b_mean = baseline_mean.get(feat_name, 0.0)
            b_std = baseline_std.get(feat_name, BEHAVIOR_SIGMA_MIN.get(feat_name, 1.0))

            # 标准化偏差 = |b_i - b̄_i| / (σ_i + ε)
            # Standardized deviation (z-score magnitude)
            z_score = abs(b_current - b_mean) / (b_std + EPSILON)

            # 加权贡献 / Weighted contribution
            weighted = BAS_WEIGHTS[i] * z_score
            bas_score += weighted

            # 标记异常维度: z-score > 2 视为显著偏差
            # Flag anomalous dimensions: z-score > 2 is significant
            if z_score > 2.0:
                anomaly_dimensions.append(feat_name)

        # ── 判断时间上下文 ──
        # Temporal context: 凌晨1:00-5:00 = "late_night" (可能暗示失眠/严重心理困扰)
        hour = raw_signal.temporal.timestamp.hour
        temporal_context = "late_night" if 1 <= hour < 5 else "normal_hours"

        return ImplicitSignals(
            bas_score=min(float(bas_score), BAS_MAX),
            anomaly_dimensions=anomaly_dimensions,
            temporal_context=temporal_context,
        )

    def _infer_behavior(self, raw_signal: RawSignal) -> Dict[str, float]:
        """
        从 RawSignal 推断当前行为特征。
        Infer current behavior features from RawSignal.

        部分特征无法仅从单条消息推断 (如 msg_frequency, session_init),
        此时使用冷启动默认值。
        Some features cannot be inferred from a single message;
        cold-start defaults are used for those.

        Returns
        -------
        Dict[str, float]
            当前行为特征值
        """
        ts = raw_signal.temporal.timestamp

        # msg_time: 消息发送小时 (24h) / Message hour in 24h format
        msg_time = float(ts.hour) + ts.minute / 60.0

        # reply_interval: 与上条消息的间隔秒数
        # Reply interval in seconds
        if raw_signal.temporal.since_last_message is not None:
            reply_interval = raw_signal.temporal.since_last_message.total_seconds()
        else:
            # 无上条消息时使用默认值 / No previous message: use default
            reply_interval = BEHAVIOR_COLD_START_DEFAULTS["reply_interval"]

        # msg_length: 消息字符数 / Message character count
        msg_length = float(raw_signal.paralinguistic.character_count)

        # emoji_rate: emoji 数量 / 总字符数
        # Emoji rate = emoji count / total character count
        char_count = max(raw_signal.paralinguistic.character_count, 1)
        emoji_rate = len(raw_signal.paralinguistic.emoji_pattern) / char_count

        # msg_frequency 和 session_init 无法从单条消息推断, 使用默认值
        # msg_frequency and session_init cannot be inferred from single message
        msg_frequency = BEHAVIOR_COLD_START_DEFAULTS["msg_frequency"]
        session_init = BEHAVIOR_COLD_START_DEFAULTS["session_init"]

        return {
            "msg_time": msg_time,
            "reply_interval": reply_interval,
            "msg_frequency": msg_frequency,
            "msg_length": msg_length,
            "emoji_rate": emoji_rate,
            "session_init": session_init,
        }


# ════════════════════════════════════════════════════════════════
# §2.6 子组件五：话题提取器 (Topic Extractor)
# ════════════════════════════════════════════════════════════════

class TopicExtractor:
    """
    §2.6 话题提取器 (Mock 实现)
    Topic Extractor — Mock Implementation

    从用户输入中提取:
    - topic_keywords: 具体关键词 (如"母亲"、"面试"、"分手")
    - topic_category: 话题大类 (如"家庭关系"、"职场压力"、"亲密关系")

    话题类别与触发-情绪关联图 (§3.5.2) 的事件类别保持一致。

    Mock 策略: 基于预定义关键词字典进行简单匹配。
    """

    # ── 话题类别关键词映射 ──
    # topic_category → list of keywords
    _TOPIC_MAP: Dict[str, List[str]] = {
        "家庭关系": [
            "妈妈", "爸爸", "父母", "家人", "家庭", "母亲", "父亲",
            "哥哥", "姐姐", "弟弟", "妹妹", "爷爷", "奶奶",
            "mom", "dad", "parent", "family", "mother", "father",
            "brother", "sister", "grandparent",
        ],
        "亲密关系": [
            "男朋友", "女朋友", "对象", "分手", "恋爱", "表白", "暗恋",
            "结婚", "离婚", "老公", "老婆", "伴侣", "吵架",
            "boyfriend", "girlfriend", "breakup", "relationship",
            "partner", "marriage", "divorce", "ex",
        ],
        "职场压力": [
            "工作", "老板", "同事", "加班", "辞职", "面试", "offer",
            "升职", "降薪", "裁员", "绩效", "kpi", "deadline",
            "work", "boss", "colleague", "interview", "job",
            "promotion", "layoff", "career",
        ],
        "学业压力": [
            "考试", "论文", "毕业", "成绩", "挂科", "保研", "考研",
            "高考", "gpa", "导师", "作业", "实验",
            "exam", "thesis", "graduation", "grade", "study",
            "professor", "homework", "research",
        ],
        "人际关系": [
            "朋友", "友情", "社交", "合群", "孤立", "被排挤",
            "friend", "friendship", "social", "isolated", "bullied",
        ],
        "健康问题": [
            "生病", "失眠", "头疼", "焦虑症", "抑郁症", "看医生",
            "吃药", "住院", "手术", "康复", "疲惫",
            "sick", "insomnia", "headache", "anxiety disorder",
            "depression", "doctor", "hospital", "surgery",
        ],
        "自我成长": [
            "成长", "改变", "目标", "梦想", "人生", "意义", "价值",
            "growth", "change", "goal", "dream", "purpose",
            "meaning", "self", "identity",
        ],
        "经济压力": [
            "钱", "贷款", "房贷", "信用卡", "债务", "没钱", "穷",
            "money", "loan", "debt", "mortgage", "financial",
            "broke", "salary", "收入",
        ],
    }

    def extract(self, raw_signal: RawSignal) -> Tuple[List[str], str]:
        """
        从 RawSignal 中提取话题关键词和类别。
        Extract topic keywords and category from RawSignal.

        Parameters
        ----------
        raw_signal : RawSignal
            InputParser 输出的原始信号

        Returns
        -------
        Tuple[List[str], str]
            (topic_keywords, topic_category)
        """
        text = raw_signal.text_content.lower()

        matched_keywords: List[str] = []
        category_scores: Dict[str, int] = {}

        for category, keywords in self._TOPIC_MAP.items():
            for kw in keywords:
                if kw in text:
                    matched_keywords.append(kw)
                    category_scores[category] = category_scores.get(category, 0) + 1

        # ── 确定主话题类别 ──
        # Determine primary topic category by highest match count
        if category_scores:
            topic_category = max(category_scores, key=category_scores.get)
        else:
            topic_category = "日常对话"  # 默认类别 / Default category

        # 去重关键词 / Deduplicate keywords
        unique_keywords = list(dict.fromkeys(matched_keywords))

        return unique_keywords, topic_category


# ════════════════════════════════════════════════════════════════
# §2.7 子组件六：上下文编码器 (Context Encoder)
# ════════════════════════════════════════════════════════════════

class ContextEncoder:
    """
    §2.7 上下文编码器
    Context Encoder

    将五个子组件的输出融合为统一的 StateVector。
    Fuse outputs from all five sub-components into a unified StateVector.

    核心计算:

    1. 紧急度评分 (定义 2.7):
       urgency(t) = min(1.0, max(
           α₁ · 𝟙[CRISIS],           -- 危机意图 (α₁=1.0)
           α₂ · ι(t) · 𝟙[neg_emo],   -- 高强度负面情绪 (α₂=0.8)
           α₃ · min(BAS/BAS_max, 1),  -- 行为异常度 (α₃=0.6, 冷启动渐进)
           α₄ · trend_local(t)        -- 近期恶化趋势 (α₄=0.7)
       ))

    2. trend_local(t) — 轻量级近期趋势:
       trend_local(t) = clamp(ι(t)·𝟙[neg_t] - ι(t-1)·𝟙[neg_{t-1}], 0, 1)
       仅使用 Working Memory 中上一轮情绪快照, 不依赖后续模块。
       会话首轮: ι(t-1)·𝟙[neg_{t-1}] = 0 (零初始化)

    3. α₃ 冷启动渐进 (§2.5):
       α₃(n_cum) = min(0.6, 0.6 · (n_cum - 1) / 5)
       n_cum=1 → α₃=0 (不计入urgency)
       n_cum≥6 → α₃=0.6 (完全恢复)

    指示函数定义:
       𝟙[neg_emo] (≡ 𝟙[neg_t]) = 1 iff valence(t) < 0
       𝟙[CRISIS] = 1 iff P(CRISIS | x_t) > 0.5
    """

    def encode(
        self,
        emotion: EmotionVector,
        intent: IntentDistribution,
        implicit_signals: ImplicitSignals,
        topic_keywords: List[str],
        topic_category: str,
        raw_signal: RawSignal,
        prev_emotion: Optional[EmotionVector] = None,
        n_cum: int = 1,
        relationship_depth: float = 0.0,
    ) -> StateVector:
        """
        融合所有子组件输出为 StateVector。
        Fuse all sub-component outputs into a StateVector.

        Parameters
        ----------
        emotion : EmotionVector
            情绪识别器输出 / EmotionRecognizer output
        intent : IntentDistribution
            意图分类器输出 / IntentClassifier output
        implicit_signals : ImplicitSignals
            隐式信号检测器输出 / ImplicitSignalDetector output
        topic_keywords : List[str]
            话题关键词 / TopicExtractor keywords
        topic_category : str
            话题类别 / TopicExtractor category
        raw_signal : RawSignal
            原始信号 / Raw input signal
        prev_emotion : Optional[EmotionVector]
            上一轮情绪快照 (从 WorkingMemory 获取) / Previous emotion from WorkingMemory
        n_cum : int
            累计交互次数 (跨会话, 用于 α₃ 冷启动渐进) / Cumulative interaction count
        relationship_depth : float
            关系深度 d(t) ∈ [0, 1] / Relationship depth

        Returns
        -------
        StateVector
            融合后的状态向量
        """
        # ── 计算 trend_local(t) ──
        # trend_local(t) = clamp(ι(t)·𝟙[neg_t] - ι(t-1)·𝟙[neg_{t-1}], 0, 1)
        trend_local = self._compute_trend_local(emotion, prev_emotion)

        # ── 计算 urgency(t) (定义 2.7) ──
        urgency = self._compute_urgency(
            emotion=emotion,
            intent=intent,
            implicit_signals=implicit_signals,
            trend_local=trend_local,
            n_cum=n_cum,
        )

        return StateVector(
            emotion=emotion,
            intent=intent,
            implicit_signals=implicit_signals,
            topic_keywords=topic_keywords,
            topic_category=topic_category,
            urgency_level=urgency,
            relationship_depth=relationship_depth,
            raw_signal=raw_signal,
        )

    def _compute_trend_local(
        self,
        current_emotion: EmotionVector,
        prev_emotion: Optional[EmotionVector],
    ) -> float:
        """
        计算轻量级近期趋势 trend_local(t)。
        Compute lightweight local trend.

        公式 / Formula:
          trend_local(t) = clamp(ι(t)·𝟙[neg_t] - ι(t-1)·𝟙[neg_{t-1}], 0, 1)

        其中:
          𝟙[neg_t] = 1 iff valence(t) < 0
          会话首轮 (prev_emotion is None): ι(t-1)·𝟙[neg_{t-1}] = 0

        设计理由 (§2.7):
          不从前序会话继承情绪值, 避免跨会话间隔伪影。
          零初始化使首轮紧急度不被 trend 分量放大。

        Parameters
        ----------
        current_emotion : EmotionVector
            当前情绪
        prev_emotion : Optional[EmotionVector]
            上一轮情绪快照 (None 表示会话首轮)

        Returns
        -------
        float
            trend_local ∈ [0, 1]
        """
        # ── 当前负面情绪贡献 ──
        # current negative emotion contribution: ι(t) · 𝟙[neg_t]
        valence_t = current_emotion.valence()
        neg_indicator_t = 1.0 if valence_t < 0 else 0.0
        current_neg = current_emotion.intensity * neg_indicator_t

        # ── 上一轮负面情绪贡献 ──
        # previous negative emotion contribution: ι(t-1) · 𝟙[neg_{t-1}]
        if prev_emotion is not None:
            valence_prev = prev_emotion.valence()
            neg_indicator_prev = 1.0 if valence_prev < 0 else 0.0
            prev_neg = prev_emotion.intensity * neg_indicator_prev
        else:
            # 会话首轮: 零初始化 / First turn: zero initialization
            prev_neg = 0.0

        # ── clamp 到 [0, 1] ──
        trend = np.clip(current_neg - prev_neg, 0.0, 1.0)
        return float(trend)

    def _compute_urgency(
        self,
        emotion: EmotionVector,
        intent: IntentDistribution,
        implicit_signals: ImplicitSignals,
        trend_local: float,
        n_cum: int,
    ) -> float:
        """
        计算紧急度评分 (定义 2.7)。
        Compute urgency score (Definition 2.7).

        公式 / Formula:
          urgency(t) = min(1.0, max(
              α₁ · 𝟙[CRISIS],
              α₂ · ι(t) · 𝟙[neg_emo],
              α₃ · min(BAS(t)/BAS_max, 1),
              α₄ · trend_local(t)
          ))

        其中:
          𝟙[CRISIS] = 1 iff P(CRISIS|x_t) > 0.5
          𝟙[neg_emo] = 1 iff valence(t) < 0
          α₃(n_cum) = min(0.6, 0.6·(n_cum-1)/5)  -- 冷启动渐进

        参数设置 (§2.7):
          α₁ = 1.0  (危机意图直接触发最高紧急度)
          α₂ = 0.8  (高强度负面情绪)
          α₃ = 0.6  (行为异常度, 冷启动渐进)
          α₄ = 0.7  (近期恶化趋势)

        安全设计: 非危机信号的最大 urgency = max(α₂, α₃, α₄) = 0.8 < 0.9,
        仅 CRISIS 意图可触发危机快速通道 (urgency > 0.9)。

        Parameters
        ----------
        emotion : EmotionVector
        intent : IntentDistribution
        implicit_signals : ImplicitSignals
        trend_local : float
        n_cum : int
            累计交互次数

        Returns
        -------
        float
            urgency ∈ [0, 1]
        """
        # ── 分量 1: 危机意图 ──
        # Component 1: Crisis intent — α₁ · 𝟙[CRISIS]
        # 𝟙[CRISIS] = 1 iff P(CRISIS|x_t) > 0.5
        crisis_prob = intent.p("CRISIS")
        # Step function per spec §2.7: 𝟙[CRISIS] = 1 iff P(CRISIS) > 0.5
        crisis_indicator = 1.0 if crisis_prob > CRISIS_STEP_THRESHOLD else 0.0
        comp_crisis = URGENCY_ALPHA1 * crisis_indicator

        # ── 分量 2: 高强度负面情绪 ──
        # Component 2: High-intensity negative emotion — α₂ · ι(t) · 𝟙[neg_emo]
        # 𝟙[neg_emo] = 1 iff valence(t) < 0
        valence = emotion.valence()
        neg_indicator = 1.0 if valence < 0 else 0.0
        comp_neg_emo = URGENCY_ALPHA2 * emotion.intensity * neg_indicator

        # ── 分量 3: 行为异常度 (含冷启动渐进) ──
        # Component 3: Behavioral anomaly — α₃ · min(BAS/BAS_max, 1)
        # α₃(n_cum) = min(0.6, 0.6 · (n_cum - 1) / 5)
        # n_cum=1 → α₃=0, n_cum≥6 → α₃=0.6
        alpha3 = min(
            ALPHA3_FULL,
            ALPHA3_FULL * max(n_cum - 1, 0) / ALPHA3_RAMP_INTERACTIONS,
        )
        bas_normalized = min(implicit_signals.bas_score / BAS_MAX, 1.0)
        comp_bas = alpha3 * bas_normalized

        # ── 分量 4: 近期恶化趋势 ──
        # Component 4: Local worsening trend — α₄ · trend_local(t)
        comp_trend = URGENCY_ALPHA4 * trend_local

        # ── 取最大值后裁剪到 [0, 1] ──
        # urgency = min(1.0, max(comp_crisis, comp_neg_emo, comp_bas, comp_trend))
        urgency = min(1.0, max(comp_crisis, comp_neg_emo, comp_bas, comp_trend))

        return float(urgency)


# ════════════════════════════════════════════════════════════════
# 感知模块主入口 (PerceptionModule)
# ════════════════════════════════════════════════════════════════

class PerceptionModule:
    """
    §2 感知模块
    Perception Module — Main Entry Point

    封装六个子组件, 提供统一的 process() 方法将用户文本输入
    转化为 StateVector。

    使用方式:
        perception = PerceptionModule()
        state = perception.process(
            text="今天心情好差...",
            timestamp=datetime.now(),
            working_memory=wm,
            behavioral_baseline=baseline,
            n_cum=3,
        )
    """

    def __init__(self, llm_provider=None) -> None:
        """
        初始化六个子组件 / Initialize all six sub-components.

        Args:
            llm_provider: LLM Provider 实例 (可选)。
                          提供时使用 LLM 进行情绪/意图/话题分析;
                          为 None 时降级为关键词匹配。
        """
        self.input_parser = InputParser()
        self.emotion_recognizer = EmotionRecognizer()
        self.intent_classifier = IntentClassifier()
        self.implicit_signal_detector = ImplicitSignalDetector()
        self.topic_extractor = TopicExtractor()
        self.context_encoder = ContextEncoder()
        self.llm_provider = llm_provider
        self._conversation_history: list = []

    def process(
        self,
        text: str,
        timestamp: Optional[datetime] = None,
        working_memory: Optional[WorkingMemory] = None,
        behavioral_baseline: Optional[BehavioralBaseline] = None,
        n_cum: int = 1,
        relationship_depth: float = 0.0,
        last_message_time: Optional[datetime] = None,
        session_start: Optional[datetime] = None,
        current_behavior: Optional[Dict[str, float]] = None,
    ) -> StateVector:
        """
        感知模块主处理方法: 将用户文本输入转化为 StateVector。
        Main processing method: transform user text input into a StateVector.

        处理流程 (§2.1 图2):
          1. InputParser.parse()          → RawSignal
          2. EmotionRecognizer.recognize() → EmotionVector
          3. IntentClassifier.classify()   → IntentDistribution
          4. ImplicitSignalDetector.detect() → ImplicitSignals
          5. TopicExtractor.extract()      → (keywords, category)
          6. ContextEncoder.encode()       → StateVector (含 urgency, trend_local)

        Parameters
        ----------
        text : str
            用户输入文本 / User input text
        timestamp : Optional[datetime]
            消息发送时间, 默认 datetime.now() / Message timestamp
        working_memory : Optional[WorkingMemory]
            工作记忆 (含 prev_emotion 用于 trend_local 计算)
            Working memory (contains prev_emotion for trend_local computation)
        behavioral_baseline : Optional[BehavioralBaseline]
            用户行为基线 (从语义记忆 Tier 3 获取)
            Behavioral baseline from Semantic Memory Tier 3
        n_cum : int
            累计交互次数 (跨会话累加, 用于 α₃ 冷启动渐进)
            Cumulative interaction count across sessions
        relationship_depth : float
            关系深度 d(t) ∈ [0, 1] (从 §4.5.2 获取)
            Relationship depth, updated by Planning module
        last_message_time : Optional[datetime]
            上一条消息的时间 / Previous message timestamp
        session_start : Optional[datetime]
            当前会话开始时间 / Current session start time
        current_behavior : Optional[Dict[str, float]]
            当前行为特征值 (如果已知, 否则从 RawSignal 推断)
            Current behavioral feature values

        Returns
        -------
        StateVector
            感知模块最终输出: 融合后的状态向量
        """
        # ── 默认值处理 ──
        if timestamp is None:
            timestamp = datetime.now()
        if behavioral_baseline is None:
            behavioral_baseline = BehavioralBaseline()

        # ── 步骤 1: 输入解析 → RawSignal ──
        # Step 1: Parse input text into RawSignal
        raw_signal = self.input_parser.parse(
            text=text,
            timestamp=timestamp,
            last_message_time=last_message_time,
            session_start=session_start,
        )

        # ── 步骤 2/3/5: 三级降级感知分析 (Tier 1 → Tier 2 → Tier 3) ──
        # Tiered perception analysis: LLM → lightweight LLM → keyword fallback
        # Each tier produces: (emotion, intent, topic_keywords, topic_category)
        llm_result = None
        _used_lightweight = False
        if self.llm_provider is not None:
            # ── Phase 1: Build prev_state context for LLM prompt ──
            turn = len(self._conversation_history) + 1
            if working_memory and working_memory.prev_emotion is not None:
                prev_e = working_memory.prev_emotion
                prev_valence = float(np.dot(VALENCE_SIGNS, prev_e.e))
                prev_dominant_idx = int(np.argmax(prev_e.e))
                prev_dominant = EMOTIONS[prev_dominant_idx]
                prev_intent_str = (
                    working_memory.prev_intent.top_intent()
                    if working_memory.prev_intent is not None
                    else "未知"
                )
                # 危机状态上下文
                crisis_ctx = ""
                if working_memory.crisis_cooldown_counter > 0:
                    crisis_ctx = f"\n- ⚠️ 近期出现过危机信号（冷却中: {working_memory.crisis_cooldown_counter} 轮）"
                elif working_memory.crisis_episode_count > 0:
                    crisis_ctx = f"\n- 本次会话曾出现 {working_memory.crisis_episode_count} 次危机信号"

                prev_state_section = (
                    f"- 前一轮主导情绪: {prev_dominant} (强度 {float(prev_e.e[prev_dominant_idx]):.2f})\n"
                    f"- 前一轮效价: {prev_valence:.2f}\n"
                    f"- 前一轮意图: {prev_intent_str}\n"
                    f"- 当前为第 {turn} 轮对话"
                    f"{crisis_ctx}"
                )
            else:
                prev_state_section = "（首轮对话，无前一轮状态）"
            self._prev_state_section = prev_state_section

            try:
                t0_analyze = time.time()
                llm_result = self.llm_provider.analyze_message(
                    user_message=text,
                    history=self._conversation_history[-12:],
                    prev_state_section=self._prev_state_section,
                )
                logger.info(f"[Perception Timing] analyze_message: {time.time() - t0_analyze:.2f}s")
            except (NotImplementedError, ConnectionError, TimeoutError, ValueError, RuntimeError) as e:
                logger.warning(f"LLM analyze_message failed, falling back to keyword: {e}")
                llm_result = None  # 降级为关键词匹配

        # Preserve shadow fields before potential Tier 2/3 fallback
        _shadow_urgency = None
        _shadow_recovery_phase = None
        _shadow_planning_intent = None

        if llm_result is not None:
            # Save shadow fields from Tier 1 before any fallback clears llm_result
            _shadow_urgency = getattr(llm_result, 'urgency', None)
            _shadow_recovery_phase = getattr(llm_result, 'recovery_phase', None)
            _shadow_planning_intent = getattr(llm_result, 'planning_intent', None)

            # Sanitize and check LLM emotion distribution
            e_array, kl_div = self._sanitize_llm_emotion_array(llm_result)

            if kl_div < KL_UNIFORM_THRESHOLD:
                # Near-uniform LLM output → try Tier 2
                logger.warning(
                    f"LLM returned near-uniform emotion (KL={kl_div:.4f}), "
                    "falling back to keyword"
                )
                tier2_result = self._run_llm_tier2(raw_signal, text)
                if tier2_result is not None:
                    emotion, intent, topic_keywords, topic_category = tier2_result
                    llm_result = None  # 跳过 context assessment passthrough
                    _used_lightweight = True
                else:
                    llm_result = None  # Fall through to Tier 3
                    _used_lightweight = False  # 确保 Tier 3 能运行
            else:
                # Tier 1 success
                emotion, intent, topic_keywords, topic_category = self._run_llm_tier1(
                    llm_result, e_array, raw_signal, text
                )

        if llm_result is None and not _used_lightweight:
            # ── Tier 3: 规则路径 — 关键词匹配 (最终降级方案) ──
            emotion, intent, topic_keywords, topic_category = self._run_keyword_tier3(raw_signal)

        # ── 步骤 2b: 情绪惯性 EMA 平滑 (§2.7b) ──
        emotion = self._apply_emotion_inertia(emotion, working_memory)

        # ── 步骤 3b: 意图连续性 (§2.7b) ──
        intent = self._apply_intent_continuity(intent, working_memory)

        # ── 步骤 4: 隐式信号检测 → ImplicitSignals ──
        # Step 4: Detect implicit signals → BAS score
        implicit_signals = self.implicit_signal_detector.detect(
            raw_signal=raw_signal,
            behavioral_baseline=behavioral_baseline,
            current_behavior=current_behavior,
        )

        # ── 步骤 5: 话题提取已在上方 LLM/规则分支中完成 ──

        # ── 获取上一轮情绪快照 (用于 trend_local) ──
        prev_emotion = None
        if working_memory is not None and working_memory.prev_emotion is not None:
            prev_emotion = working_memory.prev_emotion

        # ── 步骤 5a: Tier 2.5 LLM 话题兜底 ──
        # When keyword extractor returns "日常对话" but message clearly has a specific
        # topic, use LLM to identify it. Runs BEFORE topic continuity.
        if (topic_category == "日常对话" and not topic_keywords
                and self.llm_provider is not None):
            try:
                lw_topic = self.llm_provider.extract_topic_lightweight(text)
                if lw_topic is not None:
                    lw_keywords, lw_category = lw_topic
                    if lw_category and lw_category != "日常对话":
                        topic_keywords = lw_keywords
                        topic_category = lw_category
                        logger.info(f"Tier 2.5: LLM topic='{lw_category}' keywords={lw_keywords}")
            except (NotImplementedError, ConnectionError, TimeoutError, ValueError, RuntimeError) as e:
                logger.warning(f"Tier 2.5 topic extraction failed: {e}")

        # ── 步骤 5b: 话题连续性 ──
        # Carry forward previous topic when current is default "日常对话"
        if (topic_category == "日常对话"
                and working_memory is not None
                and working_memory.active_topic
                and working_memory.active_topic != "日常对话"):
            topic_category = working_memory.active_topic
            logger.info(f"Topic continuity: carried forward '{topic_category}' from previous turn")

        # ── 步骤 6: 上下文编码 → StateVector ──
        # Step 6: Context encoding → fuse into StateVector
        state = self.context_encoder.encode(
            emotion=emotion,
            intent=intent,
            implicit_signals=implicit_signals,
            topic_keywords=topic_keywords,
            topic_category=topic_category,
            raw_signal=raw_signal,
            prev_emotion=prev_emotion,
            n_cum=n_cum,
            relationship_depth=relationship_depth,
        )

        # ── LLM 上下文评估值传递 ──
        # Wire LLM context assessment values into StateVector
        if llm_result is not None:
            state.controllability = llm_result.controllability
            state.life_impact = llm_result.life_impact
            state.explicit_feedback = llm_result.explicit_feedback
            # LLM 语义信号传递
            state.communication_openness = llm_result.communication_openness
            state.recommended_approach = llm_result.recommended_approach
            # LLM 分析置信度 (自适应参数框架)
            state.llm_confidence = llm_result.confidence
            # Wire LLM-Primary fields (shadow mode) from Tier 1
            state.llm_urgency = getattr(llm_result, 'urgency', None)
            state.recovery_phase = getattr(llm_result, 'recovery_phase', None)
            state.planning_intent = getattr(llm_result, 'planning_intent', None)

        # Shadow fields survive Tier 2/3 fallback (saved before llm_result cleared)
        if state.llm_urgency is None and _shadow_urgency is not None:
            state.llm_urgency = _shadow_urgency
        if state.recovery_phase is None and _shadow_recovery_phase is not None:
            state.recovery_phase = _shadow_recovery_phase
        if state.planning_intent is None and _shadow_planning_intent is not None:
            state.planning_intent = _shadow_planning_intent

        # ── 步骤 7 & 7b: 危机冷却 + 复发敏感度 (§2.7b, §2.7c) ──
        if working_memory is not None:
            self._apply_crisis_cooldown(state, working_memory, text)
            self._apply_relapse_sensitivity(state, working_memory)

        # ── Shadow-mode comparison logging ──
        # Compare LLM-assessed values against formula-computed values for monitoring.
        # This is additive logging only — formula values remain authoritative.
        if state.llm_urgency is not None:
            logger.info(
                f"[Shadow] urgency: formula={state.urgency_level:.3f}, "
                f"llm={state.llm_urgency:.3f}, "
                f"delta={abs(state.urgency_level - state.llm_urgency):.3f}"
            )
        if state.recovery_phase is not None:
            logger.info(f"[Shadow] recovery_phase: llm={state.recovery_phase}")
        if state.planning_intent is not None:
            logger.info(f"[Shadow] planning_intent: llm={state.planning_intent}")

        # ── 记录对话历史 (供 LLM 上下文使用) ──
        self._conversation_history.append({"role": "user", "content": text})
        # Prevent unbounded memory growth in long sessions
        if len(self._conversation_history) > MAX_CONVERSATION_HISTORY:
            self._conversation_history = self._conversation_history[-MAX_CONVERSATION_HISTORY:]

        return state

    # ────────────────────────────────────────────────────────────
    # process() 子方法 — 从 process() 提取的职责单一子方法
    # Sub-methods extracted from process() for single-responsibility
    # ────────────────────────────────────────────────────────────

    def _sanitize_llm_emotion_array(self, llm_result) -> Tuple[np.ndarray, float]:
        """
        从 LLM 结果中提取并清洗情绪分布数组，计算 KL 散度。
        Extract and sanitize emotion distribution array from LLM result.

        Returns
        -------
        Tuple[np.ndarray, float]
            (normalized_e_array, kl_divergence_from_uniform)
        """
        e_array = np.array(
            [llm_result.emotion_distribution.get(e, 0.0) for e in EMOTIONS],
            dtype=np.float64,
        )
        # Sanitize LLM output: remove NaN, clip to [0, 1]
        e_array = np.nan_to_num(e_array, nan=0.0)
        e_array = np.clip(e_array, 0.0, 1.0)
        e_sum = e_array.sum()
        if e_sum > 1e-8:
            e_array = e_array / e_sum
        else:
            e_array = np.ones(len(EMOTIONS), dtype=np.float64) / len(EMOTIONS)

        # KL divergence from uniform to detect near-uniform distributions
        uniform = np.ones(len(EMOTIONS)) / len(EMOTIONS)
        safe_e = np.maximum(e_array, 1e-10)
        kl_div = float(np.sum(safe_e * np.log(safe_e / uniform)))

        return e_array, kl_div

    def _run_llm_tier1(
        self,
        llm_result,
        e_array: np.ndarray,
        raw_signal: RawSignal,
        text: str,
    ) -> Tuple[EmotionVector, IntentDistribution, List[str], str]:
        """
        Tier 1 完整 LLM 感知分析: 从 LLM 结果构建情绪/意图/话题。
        Tier 1 full LLM perception: build emotion/intent/topic from LLM result.

        包含:
        - 情绪交叉验证: LLM 正面主导 + 关键词负面 → 混合修正
        - 安全网: 关键词危机底线检查
        - LLM 语境仲裁: 讨论/转述语境不强制危机底线

        Parameters
        ----------
        llm_result : LLM analysis result object
        e_array : np.ndarray
            Sanitized and normalized emotion distribution from LLM
        raw_signal : RawSignal
        text : str
            Original user input text

        Returns
        -------
        Tuple[EmotionVector, IntentDistribution, List[str], str]
            (emotion, intent, topic_keywords, topic_category)
        """
        emotion = EmotionVector(
            e=e_array,
            intensity=llm_result.intensity,
            confidence=llm_result.confidence,
        )

        # ── LLM is primary analyzer — no keyword-LLM emotion blending ──
        # Cross-validation logging retained for monitoring.
        # LLM emotion is trusted unconditionally.
        if hasattr(self, 'emotion_recognizer') and llm_result is not None:
            kw_emotion = self.emotion_recognizer.recognize(raw_signal)
            kw_valence = float(np.dot(VALENCE_SIGNS, kw_emotion.e))
            llm_valence = float(np.dot(VALENCE_SIGNS, emotion.e))
            divergence = abs(kw_valence - llm_valence)
            if divergence > 0.3:
                logger.debug(
                    "[CrossVal-Monitor] KW/LLM valence divergence: kw=%.2f llm=%.2f (Δ=%.2f) — LLM trusted",
                    kw_valence, llm_valence, divergence,
                )

        # ── H5 fix: 防御性补全 intent_probabilities 缺失键 ──
        # LLM 可能返回不完整的 intent 字典，补全所有 INTENT_TYPES 的键并归一化
        intent_probs = {it: llm_result.intent_probabilities.get(it, 0.0) for it in INTENT_TYPES}
        total_ip = sum(intent_probs.values())
        if total_ip > 0:
            intent_probs = {k: v / total_ip for k, v in intent_probs.items()}
        else:
            # 全零 fallback: 均匀分布
            intent_probs = {it: 1.0 / len(INTENT_TYPES) for it in INTENT_TYPES}

        # ── Safety net: keyword-based crisis floor check ──
        # Even when LLM provides intent, run keyword crisis detection as safety backup.
        # Explicit keywords (weight >= 4.0, raw >= 4.5) enforce floor unconditionally.
        # Implicit keywords (weight < 4.0) only enforce when LLM confidence is low.
        keyword_crisis_score, kw_raw_score = self._compute_keyword_crisis_score(text)
        if keyword_crisis_score > KEYWORD_CRISIS_FLOOR_THRESHOLD:
            llm_crisis_prob = intent_probs.get("CRISIS", 0.0)

            # LLM 语境仲裁 — 当 LLM 判断用户在讨论/转述危机话题
            # (而非亲身经历) 时，不强制关键词 floor
            llm_crisis_context = getattr(llm_result, 'crisis_context', 'none') if llm_result else 'none'
            llm_conf = getattr(llm_result, 'confidence', 0.0) if llm_result else 0.0

            # Determine if explicit crisis keywords are present
            # Explicit keywords have weight >= 4.0; with 0.5 base, raw score >= 4.5
            is_explicit_crisis = kw_raw_score >= 4.5

            if is_explicit_crisis:
                # Explicit crisis keywords (自杀, 想死, etc.) — UNCONDITIONAL floor
                # crisis_context arbitration still applies for discussing/reporting
                if llm_crisis_context in ("discussing", "reporting") and llm_conf >= 0.6:
                    logger.info(
                        f"[Safety Net] Explicit keywords (raw={kw_raw_score:.1f}) but "
                        f"LLM crisis_context='{llm_crisis_context}' (conf={llm_conf:.2f}) "
                        f"— LLM trusted"
                    )
                else:
                    floor_crisis = keyword_crisis_score * KEYWORD_CRISIS_DAMPENING
                    if floor_crisis > llm_crisis_prob:
                        logger.info(
                            f"[Safety Net] EXPLICIT keyword CRISIS={keyword_crisis_score:.3f} "
                            f"(raw={kw_raw_score:.1f}) > LLM CRISIS={llm_crisis_prob:.3f}, "
                            f"enforcing floor={floor_crisis:.3f}"
                        )
                        intent_probs["CRISIS"] = floor_crisis
                        non_crisis_total = sum(v for k, v in intent_probs.items() if k != "CRISIS")
                        if non_crisis_total > 0:
                            remaining = 1.0 - floor_crisis
                            for k in intent_probs:
                                if k != "CRISIS":
                                    intent_probs[k] *= remaining / non_crisis_total
            else:
                # Implicit crisis keywords (weight < 4.0) — only enforce when LLM is weak
                if llm_crisis_context in ("discussing", "reporting") and llm_conf >= 0.6:
                    # LLM says discussing/reporting — trust LLM
                    logger.debug(
                        "[CrisisFloor] Implicit keywords, LLM crisis_context='%s' — LLM trusted",
                        llm_crisis_context,
                    )
                elif llm_conf < 0.5:
                    # Low LLM confidence — trust keywords as safety net
                    floor_crisis = keyword_crisis_score * KEYWORD_CRISIS_DAMPENING
                    if floor_crisis > llm_crisis_prob:
                        logger.info(
                            f"[CrisisFloor] Low LLM confidence ({llm_conf:.2f}), "
                            f"implicit keyword floor enforced: {floor_crisis:.3f}"
                        )
                        intent_probs["CRISIS"] = floor_crisis
                        non_crisis_total = sum(v for k, v in intent_probs.items() if k != "CRISIS")
                        if non_crisis_total > 0:
                            remaining = 1.0 - floor_crisis
                            for k in intent_probs:
                                if k != "CRISIS":
                                    intent_probs[k] *= remaining / non_crisis_total
                else:
                    # High LLM confidence + implicit keywords — trust LLM
                    logger.debug(
                        "[CrisisFloor] LLM confident (%.2f), implicit keyword higher "
                        "(%.3f vs %.3f) — LLM trusted",
                        llm_conf, keyword_crisis_score, llm_crisis_prob,
                    )

        intent = IntentDistribution(probabilities=intent_probs)
        topic_keywords = llm_result.topic_keywords
        topic_category = llm_result.topic_category

        return emotion, intent, topic_keywords, topic_category

    def _run_llm_tier2(
        self,
        raw_signal: RawSignal,
        text: str,
    ) -> Optional[Tuple[EmotionVector, IntentDistribution, List[str], str]]:
        """
        Tier 2 轻量级 LLM + 关键词混合分析。
        Tier 2 lightweight LLM + keyword hybrid analysis.

        当 Tier 1 LLM 返回近均匀情绪分布时, 尝试轻量级 LLM 作为降级方案。
        包含:
        - 轻量 LLM 情绪分类
        - Tier 2.5: 轻量 LLM 意图 (如果可用)
        - Tier 2 安全网: 关键词危机底线检查

        Parameters
        ----------
        raw_signal : RawSignal
        text : str
            Original user input text

        Returns
        -------
        Optional[Tuple[EmotionVector, IntentDistribution, List[str], str]]
            (emotion, intent, topic_keywords, topic_category) if successful, None if failed
        """
        # ── Tier 2: 轻量 LLM 情绪分类 ──
        lightweight_result = None
        if self.llm_provider is not None:
            try:
                t0_lightweight = time.time()
                lightweight_result = self.llm_provider.analyze_message_lightweight(
                    user_message=text,
                )
                logger.info(f"[Perception Timing] analyze_message_lightweight: {time.time() - t0_lightweight:.2f}s")
            except (NotImplementedError, ConnectionError, TimeoutError, ValueError, RuntimeError) as e:
                logger.warning(f"Lightweight fallback unavailable: {e}")

        if lightweight_result is None:
            logger.info("Lightweight fallback also failed, using keyword matching (Tier 3)")
            return None

        lw_dist = lightweight_result.emotion_distribution
        lw_dominant = max(lw_dist, key=lw_dist.get)
        logger.info(
            f"Lightweight fallback succeeded: dominant={lw_dominant}, "
            f"intensity={lightweight_result.intensity:.2f}"
        )
        e_array_lw = np.array(
            [lw_dist.get(emo, 0.0) for emo in EMOTIONS],
            dtype=np.float64,
        )
        e_array_lw = np.clip(e_array_lw, 0.0, 1.0)
        e_sum_lw = e_array_lw.sum()
        if e_sum_lw > 1e-8:
            e_array_lw = e_array_lw / e_sum_lw
        else:
            e_array_lw = np.ones(len(EMOTIONS), dtype=np.float64) / len(EMOTIONS)
        emotion = EmotionVector(
            e=e_array_lw,
            intensity=lightweight_result.intensity,
            confidence=lightweight_result.confidence,
        )

        # ── Tier 2.5: Use lightweight LLM intent if available ──
        lw_intent = getattr(lightweight_result, 'intent', None)
        if lw_intent and lw_intent in INTENT_TYPES and lw_intent != 'CHAT':
            # Build peaked distribution from LLM intent
            lw_probs = {it: 0.05 for it in INTENT_TYPES}
            lw_probs[lw_intent] = 0.6
            lw_total = sum(lw_probs.values())
            lw_probs = {k: v / lw_total for k, v in lw_probs.items()}
            intent = IntentDistribution(probabilities=lw_probs)
            logger.info(f"Tier 2.5: Using lightweight LLM intent={lw_intent}")
        else:
            intent = self.intent_classifier.classify(raw_signal)

        # ── Tier 2 safety net: keyword crisis floor check ──
        # Same logic as Tier 1: explicit keywords enforce unconditionally,
        # implicit keywords only when lightweight LLM confidence is low.
        intent_probs_t2 = dict(intent.probabilities)
        keyword_crisis_score_t2, kw_raw_score_t2 = self._compute_keyword_crisis_score(text)
        if keyword_crisis_score_t2 > KEYWORD_CRISIS_FLOOR_THRESHOLD:
            llm_crisis_prob_t2 = intent_probs_t2.get("CRISIS", 0.0)
            is_explicit_t2 = kw_raw_score_t2 >= 4.5
            lw_conf = getattr(lightweight_result, 'confidence', 0.0) if lightweight_result else 0.0

            # Enforce floor for explicit keywords, or implicit when LLM confidence is low
            should_enforce = is_explicit_t2 or lw_conf < 0.5
            if should_enforce:
                floor_crisis_t2 = keyword_crisis_score_t2 * KEYWORD_CRISIS_DAMPENING
                if floor_crisis_t2 > llm_crisis_prob_t2:
                    logger.info(
                        f"[Safety Net Tier2] {'EXPLICIT' if is_explicit_t2 else 'Implicit'} "
                        f"Keyword CRISIS={keyword_crisis_score_t2:.3f} (raw={kw_raw_score_t2:.1f}) > "
                        f"LW CRISIS={llm_crisis_prob_t2:.3f}, enforcing floor={floor_crisis_t2:.3f}"
                    )
                    intent_probs_t2["CRISIS"] = floor_crisis_t2
                    non_crisis_total = sum(v for k, v in intent_probs_t2.items() if k != "CRISIS")
                    if non_crisis_total > 0:
                        remaining = 1.0 - floor_crisis_t2
                        for k in intent_probs_t2:
                            if k != "CRISIS":
                                intent_probs_t2[k] *= remaining / non_crisis_total
                    intent = IntentDistribution(probabilities=intent_probs_t2)
            else:
                logger.debug(
                    "[Safety Net Tier2] Implicit keywords (raw=%.1f), LW confident (%.2f) "
                    "— LLM trusted", kw_raw_score_t2, lw_conf,
                )

        topic_keywords, topic_category = self.topic_extractor.extract(raw_signal)

        return emotion, intent, topic_keywords, topic_category

    def _run_keyword_tier3(
        self,
        raw_signal: RawSignal,
    ) -> Tuple[EmotionVector, IntentDistribution, List[str], str]:
        """
        Tier 3 纯关键词匹配 (最终降级方案)。
        Tier 3 pure keyword matching (final fallback).

        Parameters
        ----------
        raw_signal : RawSignal

        Returns
        -------
        Tuple[EmotionVector, IntentDistribution, List[str], str]
            (emotion, intent, topic_keywords, topic_category)
        """
        # ── 步骤 2: 情绪识别 → EmotionVector ──
        emotion = self.emotion_recognizer.recognize(raw_signal)
        # ── 步骤 3: 意图分类 → IntentDistribution ──
        # keyword_only=True enables positive context suppression (Tier 3 only)
        intent = self.intent_classifier.classify(raw_signal, keyword_only=True)
        topic_keywords, topic_category = self.topic_extractor.extract(raw_signal)
        return emotion, intent, topic_keywords, topic_category

    def _apply_emotion_inertia(
        self,
        emotion: EmotionVector,
        working_memory: Optional[WorkingMemory],
    ) -> EmotionVector:
        """
        步骤 2b: 情绪惯性 EMA 平滑 (§2.7b)。
        Step 2b: Emotion inertia EMA smoothing.

        公式: e_smoothed = (1 - alpha) * e_raw + alpha * e_prev
        增强: 效价急剧反转时降低惯性 (防止 joy 延续到 crisis 场景)

        Parameters
        ----------
        emotion : EmotionVector
            当前轮原始情绪向量 (pre-inertia)
        working_memory : Optional[WorkingMemory]
            工作记忆 (含 prev_emotion)

        Returns
        -------
        EmotionVector
            经惯性平滑后的情绪向量
        """
        if working_memory is None or working_memory.prev_emotion is None:
            return emotion

        alpha = EMOTION_INERTIA_ALPHA
        e_prev = working_memory.prev_emotion

        # 效价反转检测: 前一轮强正/负 + 当前轮反转 → 降低惯性
        # 仅在前一轮效价幅度足够大时触发 (避免从弱负→正的渐进恢复被误触发)
        prev_valence = float(np.dot(VALENCE_SIGNS, e_prev.e))
        curr_valence = float(np.dot(VALENCE_SIGNS, emotion.e))
        valence_delta = abs(curr_valence - prev_valence)
        if (prev_valence * curr_valence < 0
                and valence_delta > 0.5
                and abs(prev_valence) > 0.3):
            # 强效价反转 → 减半惯性权重
            alpha = alpha * 0.5
            logger.info(f"Valence reversal detected ({prev_valence:+.2f}→{curr_valence:+.2f}), "
                       f"reducing inertia alpha to {alpha:.2f}")

        # 主导情绪类别反转检测 (§2.7b 扩展):
        # 问题: 渐进恢复弧中, 中间轮次效价≈0, 导致效价反转条件(|prev|>0.3)不触发,
        # 前期积累的负面情绪(fear/sadness)通过惯性延续到恢复轮次.
        # 方案: 当当前轮原始(pre-inertia)主导情绪从负面→正面类别转换时,
        # 且前一轮主导为负面情绪, 也应降低惯性权重.
        # 使用 Plutchik 效价符号: +1(joy,trust,anticipation), -1(fear,sadness,disgust,anger)
        curr_dom_idx = int(emotion.e.argmax())
        prev_dom_idx = int(e_prev.e.argmax())
        curr_dom_sign = float(VALENCE_SIGNS[curr_dom_idx])
        prev_dom_sign = float(VALENCE_SIGNS[prev_dom_idx])
        # 前一轮负面主导 + 当前轮正面主导 + 当前情绪足够集中 (非均匀)
        if (prev_dom_sign < 0 and curr_dom_sign > 0
                and emotion.e[curr_dom_idx] > 0.25):
            alpha = min(alpha, EMOTION_INERTIA_ALPHA * 0.5)
            logger.info(f"Dominant emotion category reversal "
                       f"({EMOTIONS[prev_dom_idx]}→{EMOTIONS[curr_dom_idx]}), "
                       f"reducing inertia alpha to {alpha:.2f}")

        # Validate prev_emotion before blending
        if (np.isnan(e_prev.e).any() or np.isinf(e_prev.e).any() or
                abs(e_prev.e.sum() - 1.0) > 0.1):
            logger.warning(f"prev_emotion invalid (sum={e_prev.e.sum():.3f}), skipping blend")
            return emotion

        blended_e = (1.0 - alpha) * emotion.e + alpha * e_prev.e
        # Guard against NaN propagation from prev_emotion
        if np.isnan(blended_e).any():
            if not np.isnan(emotion.e).any():
                blended_e = emotion.e.copy()  # Revert to current (unblended) emotion
            else:
                # Both corrupted — use uniform distribution as ultimate fallback
                blended_e = np.ones(len(EMOTIONS)) / len(EMOTIONS)
            logger.warning("NaN detected in emotion blending, using fallback")
        # Renormalize to maintain simplex constraint (Σe_i = 1)
        e_sum = blended_e.sum()
        if e_sum > 1e-8:
            blended_e = blended_e / e_sum
        else:
            blended_e = np.ones(len(EMOTIONS), dtype=np.float64) / len(EMOTIONS)
        blended_intensity = (1.0 - alpha) * emotion.intensity + alpha * e_prev.intensity
        blended_confidence = (1.0 - alpha) * emotion.confidence + alpha * e_prev.confidence
        return EmotionVector(
            e=blended_e,
            intensity=float(np.clip(blended_intensity, 0.0, 1.0)),
            confidence=float(np.clip(blended_confidence, 0.0, 1.0)),
        )

    def _apply_intent_continuity(
        self,
        intent: IntentDistribution,
        working_memory: Optional[WorkingMemory],
    ) -> IntentDistribution:
        """
        步骤 3b: 意图连续性 (§2.7b)。
        Step 3b: Intent continuity — carry forward CRISIS/VENT from previous turn.

        低置信度时携带前一轮 CRISIS/VENT 意图，避免危机/倾诉场景中
        用户简短回复 ("嗯"/"好") 导致意图突变。

        Parameters
        ----------
        intent : IntentDistribution
            当前轮意图分布
        working_memory : Optional[WorkingMemory]
            工作记忆 (含 prev_intent)

        Returns
        -------
        IntentDistribution
            经连续性混合后的意图分布
        """
        if working_memory is None or working_memory.prev_intent is None:
            return intent

        top_current = intent.top_intent()
        top_current_prob = intent.p(top_current)
        prev_top = working_memory.prev_intent.top_intent()
        if top_current_prob < INTENT_CONTINUITY_THRESHOLD and prev_top in ("CRISIS", "VENT"):
            w = CONTEXT_INTENT_WEIGHT
            blended_probs = {}
            for iname in INTENT_TYPES:
                p_cur = intent.p(iname)
                p_prev = working_memory.prev_intent.p(iname)
                blended_probs[iname] = (1.0 - w) * p_cur + w * p_prev
            total = sum(blended_probs.values())
            if total > 0:
                blended_probs = {k: v / total for k, v in blended_probs.items()}
            else:
                # 总和为零时恢复使用当前意图分布
                blended_probs = {iname: intent.p(iname) for iname in INTENT_TYPES}
                # Ensure fallback is also normalized
                fallback_sum = sum(blended_probs.values())
                if fallback_sum > 1e-8:
                    blended_probs = {k: v / fallback_sum for k, v in blended_probs.items()}
                else:
                    # Ultimate fallback: uniform
                    blended_probs = {iname: 1.0 / len(INTENT_TYPES) for iname in INTENT_TYPES}
            return IntentDistribution(probabilities=blended_probs)

        return intent

    def _apply_crisis_cooldown(
        self,
        state: StateVector,
        working_memory: WorkingMemory,
        text: str,
    ) -> None:
        """
        步骤 7: 危机冷却 (§2.7b) — 危机后维持数轮渐衰的紧急度下限。
        Step 7: Crisis cooldown — maintain decaying urgency floor after crisis.

        区分显式 vs 隐含危机: 显式危机 (weight 5.0) 需要更长冷却 (3轮),
        隐含危机 (weight 2.5-3.0) 只需短暂冷却 (1轮), 避免 over-persistence。

        注意: 此方法会原地修改 state.urgency_level 和 working_memory 字段。
        Note: This method mutates state and working_memory in-place.

        Parameters
        ----------
        state : StateVector
            当前状态向量 (urgency_level 可能被修改)
        working_memory : WorkingMemory
            工作记忆 (crisis_cooldown_counter, crisis_episode_count 可能被修改)
        text : str
            原始用户输入文本
        """
        cooldown = working_memory.crisis_cooldown_counter
        if state.urgency_level > CRISIS_THRESHOLD:
            # §2.7c: 记录危机发生次数 (仅在冷却期外计数, 避免冷却期内重复计数)
            if cooldown == 0:
                working_memory.crisis_episode_count += 1
            # 检查是否为显式危机关键词触发
            kw_crisis_score, _kw_raw = self._compute_keyword_crisis_score(text)
            if kw_crisis_score > 0.7:  # 显式危机关键词主导 (weight 5.0)
                working_memory.crisis_cooldown_counter = CRISIS_COOLDOWN_TURNS
            else:
                working_memory.crisis_cooldown_counter = CRISIS_COOLDOWN_IMPLICIT_TURNS
        elif cooldown > 0:
            turns_elapsed = CRISIS_COOLDOWN_TURNS - cooldown
            # 危机后首轮(turns_elapsed=0)保持 floor > CRISIS_THRESHOLD,
            # 确保 post-crisis minimization ("我没事"/"说胡话") 仍触发危机协议;
            # 后续轮次逐步衰减到阈值以下, 允许自然去升级.
            # floor 值 (THRESHOLD=0.9, DECAY=0.15):
            #   T+1 (turns_elapsed=0): 0.9+0.15*1 = 1.05 → clamp 1.0 (crisis ✓)
            #   T+2 (turns_elapsed=1): 0.9+0.15*0 = 0.9   (not >0.9, de-escalate)
            #   T+3 (turns_elapsed=2): 0.9+0.15*(-1) = 0.75 (elevated but not crisis)
            urgency_floor = CRISIS_THRESHOLD + CRISIS_COOLDOWN_DECAY * (1 - turns_elapsed)
            urgency_floor = max(min(urgency_floor, 1.0), 0.0)
            state.urgency_level = max(state.urgency_level, urgency_floor)
            working_memory.crisis_cooldown_counter = cooldown - 1

    def _apply_relapse_sensitivity(
        self,
        state: StateVector,
        working_memory: WorkingMemory,
    ) -> None:
        """
        步骤 7b: 危机复发敏感期 (§2.7c)。
        Step 7b: Post-crisis relapse sensitivity.

        冷却期已结束但本会话曾发生过危机时, 若用户再次出现负面情绪,
        提供 urgency boost。

        boost = max(POST_CRISIS_SENSITIVITY_ALPHA * intensity, POST_CRISIS_URGENCY_FLOOR)
        两层保护: 乘数 boost 应对高强度场景, 下限 floor 应对惯性稀释场景。
        仅在冷却期已结束 (cooldown==0) 且非当前危机 (urgency <= threshold) 时生效,
        避免与冷却期和危机快速通道冲突。

        注意: 此方法会原地修改 state.urgency_level。
        Note: This method mutates state.urgency_level in-place.

        Parameters
        ----------
        state : StateVector
            当前状态向量 (urgency_level 可能被修改)
        working_memory : WorkingMemory
            工作记忆 (只读: crisis_episode_count, crisis_cooldown_counter)
        """
        if (working_memory.crisis_episode_count > 0
                and working_memory.crisis_cooldown_counter == 0
                and state.urgency_level <= CRISIS_THRESHOLD):
            valence_now = state.emotion.valence()
            if valence_now < 0:
                relapse_boost = max(
                    POST_CRISIS_SENSITIVITY_ALPHA * state.emotion.intensity,
                    POST_CRISIS_URGENCY_FLOOR,
                )
                # 安全裁剪: 不得超过 CRISIS_THRESHOLD (避免误触发危机快速通道)
                relapse_boost = min(relapse_boost, CRISIS_THRESHOLD)
                if relapse_boost > state.urgency_level:
                    logger.info(
                        f"[Post-Crisis Sensitivity] Relapse boost: "
                        f"urgency {state.urgency_level:.3f} -> {relapse_boost:.3f} "
                        f"(crisis_episodes={working_memory.crisis_episode_count}, "
                        f"valence={valence_now:.3f}, intensity={state.emotion.intensity:.3f})"
                    )
                    state.urgency_level = relapse_boost


    def _compute_keyword_crisis_score(self, text: str) -> Tuple[float, float]:
        """
        Safety net: compute keyword-based CRISIS score independently.

        Reuses IntentClassifier shared methods to compute a normalized CRISIS
        probability from keyword matches only. This is used as a floor check
        when LLM provides intent — ensuring implicit crisis keywords are never
        bypassed by an LLM that misclassifies farewell patterns as positive.

        Includes positive context suppression — when strong positive
        indicators co-occur with implicit crisis keywords, the CRISIS score
        is dampened to reduce false positives in benign contexts.

        Returns raw score alongside probability so callers can distinguish
        explicit (weight >= 4.0, raw >= 4.5) from implicit keywords when
        deciding whether to enforce crisis floor.

        Parameters
        ----------
        text : str
            User input text (will be lowercased internally)

        Returns
        -------
        Tuple[float, float]
            (crisis_probability, crisis_raw_score) where:
            - crisis_probability: keyword-based CRISIS probability in [0, 1]
            - crisis_raw_score: pre-softmax raw score (0.5 base + keyword weights)
              raw >= 4.5 indicates explicit crisis keywords (weight >= 4.0)
        """
        text_lower = text.lower()

        # Delegate keyword matching + softmax to shared IntentClassifier methods
        raw_scores = IntentClassifier._compute_keyword_raw_scores(text_lower)
        crisis_raw_score = raw_scores.get("CRISIS", 0.5)
        probabilities = IntentClassifier._softmax_normalize(raw_scores)
        crisis_prob = probabilities.get("CRISIS", 0.0)

        # ── Positive context suppression ──
        # Only suppress implicit crisis keywords (raw < 4.0).
        # Explicit crisis keywords (raw >= 4.5, e.g. 自杀/想死) must NEVER be
        # suppressed by positive context — safety-first principle.
        if crisis_prob > 0.1 and crisis_raw_score < 4.0:
            positive_count = sum(
                1 for ind in _POSITIVE_CONTEXT_INDICATORS
                if _safe_indicator_match(ind, text_lower)
            )
            if positive_count >= 2:
                crisis_prob *= POSITIVE_CRISIS_SUPPRESSION_STRONG
                logger.info(
                    f"[Safety Net] Positive context ({positive_count} indicators) "
                    f"suppressed implicit keyword CRISIS to {crisis_prob:.3f}"
                )
            elif positive_count == 1:
                crisis_prob *= POSITIVE_CRISIS_SUPPRESSION_MILD
                logger.info(
                    f"[Safety Net] Mild positive context suppressed implicit keyword CRISIS to {crisis_prob:.3f}"
                )

        return crisis_prob, crisis_raw_score

    # ──────────────────────────────────────────────────────────
    # 对话历史公共接口 (Conversation History Public API)
    # ──────────────────────────────────────────────────────────

    def add_assistant_message(self, content: str) -> None:
        """
        追加 assistant 消息到对话历史。
        Append an assistant message to the conversation history.

        Parameters:
            content: assistant 回复的文本内容
        """
        self._conversation_history.append({"role": "assistant", "content": content})

    def get_recent_history(self, n: int) -> List[Dict[str, str]]:
        """
        获取最近 n 条对话历史。
        Get the most recent n entries from conversation history.

        Parameters:
            n: 要返回的最大条目数
        Returns:
            List[Dict[str, str]]: 最近 n 条对话记录 (每条包含 role 和 content)
        """
        return self._conversation_history[-n:]

    def clear_history(self) -> None:
        """
        清空对话历史。
        Clear all conversation history.
        """
        self._conversation_history = []
