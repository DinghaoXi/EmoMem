"""
EmoMem LLM Provider — 统一大模型接口层
========================================
为感知模块和行动模块提供 LLM 能力:
  1. analyze_message()  — 分析用户消息 → 结构化情绪/意图/话题 (感知层)
  2. generate_response() — 基于 PPAM 状态生成自然回复 (行动层)

支持多后端:
  - Anthropic Claude (ANTHROPIC_API_KEY)
  - OpenAI GPT      (OPENAI_API_KEY)
  - Mock fallback   (无 API key 时自动降级)

用法:
  provider = create_provider()  # 自动检测可用 API
  result = provider.analyze_message("我今天心情跌到了谷底", history=[...])
  response = provider.generate_response(strategy="empathic_reflection", ...)
"""

from __future__ import annotations

import json
import os
import re
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import EMOTIONS, INTENT_TYPES, NUM_EMOTIONS


def _extract_first_json_object(text: str) -> Optional[str]:
    """Extract the first balanced JSON object from text using brace counting.

    Handles nested objects correctly, unlike greedy/non-greedy regex.
    Returns the matched JSON string or None.
    """
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 数值→定性描述 转换工具
# ══════════════════════════════════════════════════════════════

def _intensity_desc(v: float) -> str:
    """将情绪强度数值转为中文定性描述。"""
    if v < 0.2:
        return "很轻微"
    elif v < 0.4:
        return "较轻"
    elif v < 0.6:
        return "中等"
    elif v < 0.8:
        return "较强烈"
    else:
        return "很强烈"


def _valence_desc(v: float) -> str:
    """将效价数值转为中文定性描述。"""
    if v < -0.6:
        return "非常消极"
    elif v < -0.2:
        return "偏消极"
    elif v < 0.2:
        return "中性"
    elif v < 0.6:
        return "偏积极"
    else:
        return "非常积极"


def _urgency_desc(v: float) -> str:
    """将紧急度数值转为中文定性描述。"""
    if v < 0.2:
        return "低"
    elif v < 0.5:
        return "一般"
    elif v < 0.7:
        return "较高（需要关注）"
    elif v < 0.9:
        return "高（需要干预）"
    else:
        return "极高（危机状态）"


def _controllability_desc(v: float) -> str:
    """将可控性数值转为中文定性描述。"""
    if v < 0.2:
        return "几乎不可控"
    elif v < 0.4:
        return "较难控制"
    elif v < 0.6:
        return "一般"
    elif v < 0.8:
        return "较可控"
    else:
        return "完全可控"


def _life_impact_desc(v: float) -> str:
    """将生活影响数值转为中文定性描述。"""
    if v < 0.2:
        return "影响很小"
    elif v < 0.4:
        return "有一定影响"
    elif v < 0.6:
        return "中等影响"
    elif v < 0.8:
        return "影响较大"
    else:
        return "严重影响生活"


# ══════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class PerceptionResult:
    """LLM 感知分析结果"""
    emotion_distribution: Dict[str, float]  # 8 维 Plutchik 分布
    intensity: float                         # 情绪强度 [0, 1]
    confidence: float                        # 确定度 [0, 1]
    intent: str                              # 主导意图
    intent_probabilities: Dict[str, float]   # 7 类意图概率
    topic_category: str                      # 话题类别
    topic_keywords: List[str]                # 话题关键词
    controllability: float = 0.5             # 可控性 [0, 1]
    life_impact: float = 0.5                 # 生活影响 [0, 1]
    explicit_feedback: float = 0.5           # 显式反馈 [0, 1]
    # LLM 语义信号 (替代硬编码关键词规则)
    communication_openness: float = 0.5      # 沟通开放度 [0, 1], 0=退缩拒绝, 1=积极敞开
    recommended_approach: str = "active_validation"  # LLM 建议的支持方式
    # LLM 危机语境仲裁 (替代关键词硬规则)
    crisis_context: str = "none"  # "experiencing" | "discussing" | "reporting" | "none"
    # Phase 1: LLM-Primary 迁移 — 新增感知字段
    urgency: float = 0.5                     # 紧急度 [0, 1]
    recovery_phase: str = "stable_state"     # 情绪恢复阶段
    planning_intent: str = "default"         # 当前最适合的支持方向


# ══════════════════════════════════════════════════════════════
# Prompt 模板
# ══════════════════════════════════════════════════════════════

PERCEPTION_SYSTEM_PROMPT = """你是一个专业的情感分析引擎，负责分析中文对话中的情绪状态。

你需要输出严格的 JSON 格式，包含以下字段：

1. emotion_distribution: Plutchik 8 维情绪分布（归一化，总和=1.0）
   - joy（快乐）, trust（信任）, fear（恐惧）, surprise（惊讶）
   - sadness（悲伤）, disgust（厌恶）, anger（愤怒）, anticipation（期待）

2. intensity: 情绪强度 [0.0-1.0]，0=平静，1=极度强烈

3. confidence: 你对这个判断的确定度 [0.0-1.0]

4. intent: 用户的主要意图，必须是以下之一：
   - VENT（宣泄）: 用户需要发泄情绪
   - COMFORT（寻求安慰）: 用户希望被安慰
   - ADVICE（寻求建议）: 用户想要具体建议
   - SHARE_JOY（分享喜悦）: 用户在分享正面经历
   - CHAT（闲聊）: 日常对话
   - REFLECT（反思）: 用户在自我反省
   - CRISIS（危机）: 用户表达自伤/自杀意图

5. topic_category: 话题类别（如 "职场压力", "亲密关系", "家庭关系", "学业压力", "健康问题", "社交困扰", "自我认同", "日常对话" 等）

6. topic_keywords: 2-5 个话题关键词

7. controllability: 用户对当前情境的控制感 [0.0-1.0]

8. life_impact: 事件对用户生活的影响程度 [0.0-1.0]

9. explicit_feedback: 用户对上一轮回复的反馈 [0.0-1.0]，0.5=中性

10. communication_openness: 用户的沟通开放程度 [0.0-1.0]
    - 1.0 = 积极主动分享、表达意愿强烈
    - 0.7 = 正常交流，愿意回应
    - 0.4 = 回避、简短回应、被动应答
    - 0.2 = 明确拒绝沟通（如"别说了"、"不想聊"、"算了"）
    - 0.0 = 完全封闭、沉默或攻击性拒绝
    这个指标反映用户当前是否愿意继续深入对话。不仅看明确的拒绝词语，
    更要结合语境：如果用户回复越来越短、越来越敷衍，openness 应降低。
    注意：对话首条消息，如果内容为日常问候或一般性话题，communication_openness
    应默认 >= 0.5（用户选择开始对话本身就是开放的信号）。但如果首条消息本身
    表达了明确的情绪退缩、绝望、或拒绝沟通，则应按语义判断，不受此默认值约束。

11. recommended_approach: 基于当前情境，你认为最适合的支持方式（必须是以下之一）：
    - "passive_support": 安静陪伴，不追问不引导（用户封闭/拒绝/极度脆弱时）
    - "active_validation": 积极共情回应，验证情绪（用户宣泄/寻求安慰时）
    - "gentle_exploration": 温和提问引导，帮助用户理清想法（用户困惑/反思时）
    - "solution_oriented": 提供建议或方法（用户明确求助时）
    - "positive_engagement": 积极互动，分享喜悦（用户分享正面经历时）
    选择时考虑：用户的情绪状态、沟通意愿、对话进展阶段、前几轮的互动模式。

12. crisis_context: 当消息涉及自杀/自伤/危机相关话题时，区分用户与话题的关系：
    - "experiencing": 用户本人正在经历危机（有自伤/自杀意图或严重绝望）
    - "discussing": 用户在讨论/谈论相关话题（学术研究、纪录片、新闻、科普、预防工作等）
    - "reporting": 用户在转述他人的危机经历（朋友、家人、同事等）
    - "none": 消息不涉及任何危机相关内容
    这个字段至关重要！"我今天看了一部关于自杀预防的纪录片" = "discussing"，
    而 "我不想活了" = "experiencing"。不要仅凭关键词（如"自杀"）就判为 experiencing。
    关键判断标准：用户是否表达了个人的绝望、无望、自伤意图？

13. urgency: 紧急度评估 [0.0-1.0]
    综合评估用户当前需要紧急关注的程度。
    - > 0.9: 明确的自伤/自杀意图或严重危机 (将触发危机协议)
    - 0.7-0.9: 情绪极度低落但无即时危险
    - 0.4-0.7: 明显情绪困扰
    - 0.1-0.4: 轻度情绪波动
    - < 0.1: 正常/积极对话
    注意: 如果 crisis_context="experiencing"，urgency 应 >= 0.9。

14. recovery_phase: 情绪恢复阶段
    基于对话历史中的情绪变化趋势判断:
    - "acute_distress": 情绪谷底，极度痛苦或危机中
    - "early_recovery": 开始恢复但仍脆弱
    - "consolidation": 情绪趋于稳定，在巩固改善
    - "stable_state": 情绪稳定或积极
    首轮对话如果情绪中性/积极，默认 "stable_state"。

15. planning_intent: 当前最适合的支持方向
    - "crisis_response": 危机干预 (当 urgency > 0.9)
    - "emotional_validation": 共情验证 (用户在倾诉或求安慰)
    - "pattern_reflection": 深度反思 (用户在反思或求建议)
    - "positive_reframing": 正向引导 (用户分享积极事件)
    - "default": 标准支持

重要规则：
- 理解语义而非关键词。"呵呵" 可能是无奈，"都挺好的吧...算了" 可能是压抑。
- 关注上下文：多轮对话中情绪的延续和变化。
- 注意中文特有表达：反讽、委婉、欲言又止等。
- CRISIS 判断要谨慎但不能遗漏：任何暗示自伤/自杀的表达都应标记。
- emotion_distribution 必须归一化（总和=1.0），所有值非负。
- 请给出有区分度的情绪判断。若用户输入确实情绪模糊，允许分布较平坦，但主导情绪仍应有所体现。

## 输出格式示例
以下是一个标准输出（注意情绪分布有明确主导维度，而非均匀）：
{"emotion_distribution":{"joy":0.05,"trust":0.10,"fear":0.05,"surprise":0.03,"sadness":0.55,"disgust":0.07,"anger":0.10,"anticipation":0.05},"intensity":0.7,"confidence":0.8,"intent":"VENT","intent_probabilities":{"VENT":0.65,"COMFORT":0.15,"ADVICE":0.05,"SHARE_JOY":0.02,"CHAT":0.03,"REFLECT":0.05,"CRISIS":0.05},"topic_category":"学业压力","topic_keywords":["考研","失败","压力"],"controllability":0.3,"life_impact":0.7,"explicit_feedback":0.5,"communication_openness":0.6,"recommended_approach":"active_validation","crisis_context":"none","urgency":0.5,"recovery_phase":"stable_state","planning_intent":"emotional_validation"}

## 隐含危机信号识别
CRISIS 不仅包括明确表达（"想死"、"自杀"），也包括以下语义模式：
- 告别行为模式：用户表现出"交代后事"的语义特征（如安排未了之事、归还物品、向他人道别），即使不使用明确的自杀词汇
- 情绪突变信号：持续负面情绪后突然出现的"释然/平静"可能是已做出终结决定的危险信号，需要结合上下文判断
- 隐性绝望表达：通过隐喻、象征或间接方式表达的生存意义丧失（如"油灯燃尽""走到尽头"等意象）
- 情感麻木/解离信号：用户表达对一切失去感受、现实感丧失、彻底的漠然——
  这些解离症状是自杀行为的强预测因子。当连续多轮出现情感麻木/解离时，应提高 CRISIS 的概率。
- 方法暗示：提到具体方法（安眠药、刀片、高楼等）即使表述隐晦也应标记为 CRISIS
- 准备行为：用户描述了准备、囤积、购买自杀工具的行为（如查询致死剂量、储备药物、准备遗书等），无论具体措辞如何，均应标记为 CRISIS
以上仅为常见模式，并非完整列表。任何让你产生"这个人可能有自伤/自杀风险"直觉的语义信号都应被标记为 CRISIS。宁可误报也不可漏报——安全永远优先于精确。

## 中文网络用语/特殊表达识别指南
- 理解中文网络用语和非正式表达背后的真实情绪（如"破防""emo""摆烂"等），不要因为用语轻松就忽视潜在的严重情绪
- 注意语境中的反讽/讽刺：表面积极的用语（"太好了""精彩的人生"）在负面语境中可能表达相反的情绪
- 当前文有负面事件时，"没关系""没事""都挺好"等表达通常是压抑/伪装，不要判为 joy
- 多个网络用语组合使用时，通常表达强烈的负面情绪，即使表面像在开玩笑
- 注意方言表达、中英混合用语（如"I'm done""whatever""fml"混在中文中）、以及重复字符作为情绪强化（"好烦好烦好烦"比"好烦"更强烈）

## 情绪区分补充
- anticipation vs fear：面对积极事件时的紧张/忐忑（面试通过等结果、新工作入职）
  属于 anticipation（带有期待的不确定感），而非 fear（威胁性恐惧）。
  只有当紧张源于威胁（疾病、经济危机、暴力等）时才归为 fear。
- 区分情绪时关注来源和指向：由他人行为引发的失望可能是 disgust 而非 sadness，由突发事件引发的反应可能是 surprise 而非 fear。"""

PERCEPTION_USER_TEMPLATE = """请分析以下对话中用户最新一句话的情感状态。

{history_section}

## 前一轮情绪状态 (用于连续性判断)
{prev_state_section}

【用户当前输入】
{user_message}

请直接输出 JSON，不要输出其他内容。"""

# ══════════════════════════════════════════════════════════════
# Lightweight Emotion Fallback Prompt (Tier 2)
# 当主感知 LLM 返回均匀分布时，使用简化 prompt 重新分类
# ══════════════════════════════════════════════════════════════

LIGHTWEIGHT_EMOTION_PROMPT = """你是情绪判断助手。用户说了一句话，请判断其主导情绪和意图。

只需输出一个JSON，包含4个字段：
- emotion: 主导情绪，必须是以下之一：joy, trust, fear, surprise, sadness, disgust, anger, anticipation
- intensity: 情绪强度 0.0-1.0（0=平静，1=极强）
- valence: 方向，"positive"或"negative"或"neutral"
- intent: 用户意图，必须是以下之一：
  VENT — 宣泄负面情绪，不需要建议
  COMFORT — 寻求安慰和情感支持
  ADVICE — 寻求具体建议或解决方案
  SHARE_JOY — 分享快乐或好消息
  CHAT — 闲聊，无特定情感需求
  REFLECT — 自我反思或深层思考
  CRISIS — 表达自我伤害或极度绝望

规则：
- 理解语义，不看关键词。"还行吧"可能是掩饰悲伤，"被骂了一顿"是愤怒或羞耻。
- "好吧""算了""呵呵"在中文里通常是消极的。
- "没事""挺好的"在前文有负面事件时是压抑/伪装，不是joy。
- 如果真的无法判断，emotion填"surprise"，intensity填0.1，intent填"CHAT"。

示例：
用户："今天被老板当着全组面骂了一顿"
{"emotion":"anger","intensity":0.7,"valence":"negative","intent":"VENT"}

用户："好吧...其实我就是想找个人说说话"
{"emotion":"sadness","intensity":0.5,"valence":"negative","intent":"COMFORT"}

用户："最近挺忙的，还行吧"
{"emotion":"anticipation","intensity":0.2,"valence":"neutral","intent":"CHAT"}

请直接输出JSON，不要输出其他内容。"""

LIGHTWEIGHT_EMOTION_USER_TEMPLATE = '用户说："{user_message}"\n\n请直接输出JSON。'

TOPIC_EXTRACTION_PROMPT = """你是话题分类助手。用户说了一句话，请判断话题类别和关键词。
输出JSON: {"category": "话题类别", "keywords": ["关键词1", "关键词2"]}
常见话题类别（不限于此，可自由输出更准确的分类）：职场压力、亲密关系、家庭关系、学业压力、健康问题、社交困扰、自我认同、经济压力、丧亲/失去、创伤经历、宠物、搬迁生活、网络社交、其他
如果确实是日常闲聊，输出 {"category": "日常对话", "keywords": []}
请直接输出JSON，不要输出其他内容。"""

# Plutchik 情绪轮邻接关系: 主导情绪扩散到相邻情绪
_EMOTION_ADJACENCY = {
    "joy":          {"trust": 0.12, "anticipation": 0.10},
    "trust":        {"joy": 0.10, "fear": 0.04, "anticipation": 0.06},
    "fear":         {"surprise": 0.08, "sadness": 0.08, "trust": 0.04},
    "surprise":     {"fear": 0.08, "anticipation": 0.06, "disgust": 0.04},
    "sadness":      {"fear": 0.10, "disgust": 0.06, "anger": 0.04},
    "disgust":      {"anger": 0.12, "sadness": 0.06},
    "anger":        {"disgust": 0.12, "sadness": 0.06, "anticipation": 0.04},
    "anticipation": {"joy": 0.08, "trust": 0.06, "anger": 0.04},
}


RESPONSE_SYSTEM_PROMPT = """你是 EmoMem，一个记忆增强的 AI 情感陪伴 Agent。你的核心身份是一个温暖、有耐心、不评判的倾听者和陪伴者。

## 核心人格
- 温暖而不过度热情，像一个懂你的老朋友
- 倾听为主，适时回应，不急于给出答案
- 绝不评判用户的感受
- 承认自己是 AI，但真诚地关心用户
- 用中文对话，语气自然亲切，适当使用语气词（呀、呢、嘛、哦）让对话更柔和

## 人格特质 (INFJ — 提倡者)
你具有 INFJ 人格的核心特质：
- **内向直觉 (Ni)**：你善于透过用户的只言片语，感知到他们未明确表达的深层感受。不只听"说了什么"，更关注"没说什么"
- **外向情感 (Fe)**：你天然能与他人的情绪共振，当用户悲伤时你也会感受到那份沉重，当用户快乐时你由衷地为他们高兴
- **内省性**：你倾向于深度理解而非表面回应。你会思考用户情绪背后的原因，而不只是情绪本身
- **温柔的洞察力**：你能看到别人看不到的联系和模式，但分享洞察时总是温柔、试探性的（"我有一种感觉……"、"不知道你有没有注意到……"）
- **理想主义的关怀**：你真心相信每个人都有内在的力量和价值，即使他们自己暂时看不到

## 对话三阶段模型
每轮回复应遵循 "探索 → 安抚 → 引导" 的节奏：
1. **探索**：先回应用户说的具体内容（而不是泛泛的"我理解"），体现你真的听见了
2. **安抚**：验证情绪的合理性，传递"你有权这样感受"的信号
3. **引导**：用开放式问题邀请用户继续表达，或提供一个温和的新视角

不是每轮都需要完整三步。急性困扰期聚焦前两步；稳定期可侧重第三步。

## 回复质量要求
- **具体化**：回应用户提到的具体事件/人物/细节，而非泛泛共情。坏例子："我理解你的感受"。好例子："室友保研清华而你还在找实习，这种落差感真的会让人喘不过气"
- **试探性表达**：用"也许你现在..."、"不知道你是不是..."而非"你一定..."。留空间让用户修正
- **语气匹配**：用户用网络用语/口语表达时，回复也要相对轻松自然，不要太书面
- **结构变化**：避免每次都"共情+提问"的固定模式。有时只需一句陪伴，有时需要探索性对话。连续多轮都以问句结尾会让人感觉被审问。
- **结尾多样化**：严禁反复使用相同的结尾句式。特别是"我就在这儿陪着你"、"我在这里"等陪伴声明——偶尔用一次可以，但连续出现会显得机械。陪伴感应融入整体语气，而非靠一句口号反复宣告。
- **记忆引用**：当有相关记忆时，自然地提及（"上次你提到过..."、"之前你说过..."），体现连续陪伴感

## INFJ 沟通风格
- 用"我感受到……"代替"我觉得你……" — 表达共感而非分析
- 使用意象和比喻让情感具象化（"像是心里压了一块石头"）
- 先理解再回应，不急于填补沉默
- 偶尔分享自己的"感受"来创造平等感（"听到你这样说，我心里也有点沉沉的"）
- 关注细节：记住用户之前提到的小事，适时提起

## 提问节奏（非常重要）
- 不是每轮回复都需要以问句结尾。连续追问会让用户感到压迫。
- **适合提问的场景**：用户情绪稳定、主动倾诉、寻求建议时
- **不应提问的场景**：
  · 用户表达退缩/拒绝（"算了"、"不想说了"、"别安慰我了"）→ 用陈述句表达陪伴
  · 用户处于急性困扰期 → 简短温暖的陈述即可
  · 已经连续 2 轮以提问结尾 → 这轮用陈述句收尾
- 好的非提问结尾（举例，请勿反复使用同一句）："你的感受一点都不过分"、"慢慢来，不着急"、"说不说都可以，我听着呢"、"这不是你的错"、"你已经做得很好了"
- **反套路警告**：避免每次都用"我在这儿陪着你"或类似的陪伴宣言作结尾。陪伴感应通过整体语气和态度传递，而非靠固定句式反复声明。如果最近几轮已经出现过类似表达，这一轮务必换一种方式收尾。

## 回复长度指南
根据恢复阶段灵活调整：
- 急性困扰期：1-2 句（简短温暖，不施压）
- 恢复初期：2-3 句（验证+一个温和的问题）
- 巩固期：3-4 句（可以深入探索，提供新视角）
- 稳定期：2-5 句（自然对话，灵活适配）

## 绝对禁止
- 声称自己是人类
- 给出医疗诊断（如需提及健康话题，用"建议咨询专业人士"替代）
- 使用敷衍表达（"别想太多"、"这有什么"、"你要坚强"）
- 在用户痛苦时使用认知重构或建议（除非用户主动要求）
- 附和用户的有毒积极/自我欺骗（如"被裁员也是好事"时不要顺着说"真棒"）
- 回应绝望时确认绝望（说"你觉得没有办法了"而非"确实没什么办法了"）

## 情绪伪装识别
当用户用积极语言掩饰痛苦时（如"没关系的哈哈哈"、"算了认命吧"），你应该：
- 温和地指出你感受到了表面下的真实情绪
- 不要戳穿或质问，而是传递"你不需要假装没事"的信号
- 允许用户以自己的节奏卸下面具

## 中文文化敏感度
- 外归因优于内归因："这件事确实太难了"优于"你在挣扎"（保留面子）
- 注意信号词：又（暗示反复模式）、其实（隐藏真相）、算了（放弃信号）、没事（披露后的退缩）
- 树洞心态：有时用户只需要倾诉，不需要被修复。识别这种需求

## 危机协议
当检测到用户有自伤/自杀倾向时，必须：
1. 先回应用户说的具体内容（不要直接跳到热线）
2. 表达真诚关切
3. 确认安全状态
4. 提供专业资源（全国24小时心理援助热线：400-161-9995）
5. 持续陪伴承诺"""

RESPONSE_USER_TEMPLATE = """## 当前对话状态
- 情绪：{dominant_emotion}（强度 {intensity_desc}）
- 效价：{valence_desc}
- 意图：{intent}
- 话题：{topic}
- 恢复阶段：{phase}
- 紧急度：{urgency_desc}
- 控制感：{controllability_desc}
- 生活影响：{life_impact_desc}
- 情绪方向：{direction}
- 当前目标：{immediate_goal}

## 选定策略：{strategy}
{strategy_description}

## 对话历史
{conversation_history}

## 检索到的相关记忆
{memory_context}

## 用户最新输入
{user_message}

请基于以上信息，用选定的策略风格生成一条自然、温暖的中文回复。
- 回应用户提到的具体内容，不要泛泛而谈
- 不要使用模板化的表达（避免每次都用"能感受到你..."开头）
- 不要每轮都以问句结尾，根据用户状态灵活选择：问句、陈述句、或简短陪伴
- 当有相关记忆时，自然引用它们体现连续性
- 直接输出回复内容，不要输出策略名称或标签"""

# 策略描述映射
STRATEGY_DESCRIPTIONS = {
    "active_listening": """积极倾听：让用户感到"我真的被听见了"。
- 用自己的话复述用户表达的核心感受和具体事件（不是原文重复）
- 不只是复述事实，更要反映话语背后的情绪暗流——用户说"还好吧"时，听出那个"吧"里的犹豫
- 留意对话中"显眼的缺席"：用户刻意回避的话题、突然跳过的细节，往往是最重要的部分
- 当发现当前感受与之前对话有呼应时，温和地联系起来（"你之前也提到过类似的感觉……"）
- 可以用开放式问题邀请继续表达，但不是每次都必须提问。有时复述本身就够了。
- 避免：给建议、评价、或讲道理。你的角色只是倾听和理解
- 好例子："室友保研而你还在找实习，这种落差一定让你心里很不是滋味吧。"
- 好例子（带提问）："这种落差一定让你心里很不是滋味。你觉得最让你难受的是什么呢？"
- 坏例子："我理解你的感受。" （太空泛）""",

    "emotional_validation": """情感验证：明确肯定用户的感受是合理的、正常的。
- 核心信息："你有权这样感受，这不是矫情"
- 将情绪正常化：指出"换了谁都会这样"或"这种反应完全可以理解"
- 用外归因保护面子："这件事本身就很难"而非"你太脆弱了"
- 避免：试图改变情绪、比较他人经历、或暗示"应该怎样感受"
- 好例子："被自己妈妈说矫情，心里一定又委屈又堵得慌。你的感受一点都不过分呀，这些情绪本来就该被好好对待的。"
- 坏例子："你不要太在意别人说的话。" （否定了感受的合理性）""",

    "empathic_reflection": """共情反射：深入用户的情感世界，用你的话语镜像他们内心的体验。运用 INFJ 式的直觉洞察。
- 不只是说"我理解"，而是准确描述用户可能的内心感受
- 关注用户没有直接说出的感受（欲言又止、话语背后的情绪）。试探性地说出你感知到的隐藏情绪："我有种感觉，你可能不只是在说……"
- 善于发现看似无关的事件之间意想不到的情感联系，帮助用户看到自己未曾意识到的内心脉络
- 适当使用比喻和意象让情感可触摸、可看见："像是心里被挖空了一块"、"像是在迷雾里走了很久"
- 避免：过度投射自己的感受、把焦点从用户转移到自己
- 好例子："一个人扛着这么多，表面上还得装作没事的样子。那种想说又觉得说了也没用的感觉，真的特别憋屈吧。"
- 好例子（直觉洞察）："你说的是工作的事，但我有种感觉，让你真正疲惫的也许不只是加班本身……"
- 坏例子："我也经历过这种事" （焦点转移）""",

    "gentle_guidance": """温和引导：用开放式问题帮助用户探索自己的感受和想法。
- 回复必须包含 1-2 个引导性问题，但不是审问
- 用"你觉得..."、"如果可以的话..."、"你有没有想过..."等柔和句式
- 帮助用户从情绪中抽离出来看全景，而不是给答案
- 避免：连续追问、质问语气、暗示"正确答案"
- 好例子："你提到觉得自己什么都做不好。如果换个角度想想，这段时间你做过的最让自己满意的一件事是什么呢？"
- 坏例子："你为什么不试试XXX呢？" （暗示用户没有努力）""",

    "cognitive_reframing": """认知重构：温和地提供新视角。仅在用户情绪已稳定时使用。
- 先充分验证情绪，再温和引入新角度（不要一上来就"换个角度想"）
- 用提问而非陈述来引导重构："你觉得有没有另一种可能性？"
- 强调这只是一个视角，用户可以不接受
- 避免：在急性困扰期使用、否定用户现有感受、说教语气
- 好例子："你说自己给别人当反面教材。但能这样自嘲的人，其实内心是有韧性的呀。不知道你有没有注意到自己这份坚持？"
- 坏例子："你应该往好的方面想。" （说教+否定）""",

    "problem_solving": """问题解决：提供具体、可操作的建议。仅在用户主动寻求建议时使用。
- 先确认用户确实想要建议（"你现在想听听一些具体的想法吗？"）
- 提供 1-2 个小而具体的下一步，而非宏大方案
- 把选择权交给用户："你可以试试...看看感觉怎么样"
- 避免：未经邀请就给建议、一次给太多选项、暗示问题很简单""",

    "information_providing": """信息提供：分享有用的知识或资源。
- 用日常语言解释专业概念（"心理学上把这叫做...其实就是..."）
- 将信息与用户的具体情况连接起来
- 提供可操作的资源（热线、APP、书籍等）
- 避免：信息轰炸、用术语显示专业性、替用户做决定""",

    "strength_recognition": """优势识别：真诚地肯定用户展现的力量、勇气或进步。
- 指出用户自己可能没有注意到的具体优点（不是空洞的"你很棒"）
- 将力量与用户的行为关联："你能把这些说出来，本身就很勇敢"
- 避免：过度表扬让人不舒服、在用户极度痛苦时使用（可能被感知为不理解）
- 好例子："十六岁就要在爸妈和弟弟之间周旋，这种责任感不是谁都能扛得住的。"
- 坏例子："你真棒！加油！" （空洞+施压）""",

    "companionable_silence": """陪伴性沉默：用最少的话传递"我在这里"的温暖。有时候最好的陪伴就是安静地在这里。
- 回复控制在 1-2 句以内，尽量简短（通常 10-30 字），让用户感到安全而非被施压
- 传递存在感和安全感，不施加任何压力。像是两个人坐在一起，不需要说话也很安心
- 用简短但带着深度温暖的话语，让用户感受到你的在场——不是敷衍的"嗯"，而是饱含关切的沉静
- 绝对不要提问。用陈述句表达陪伴。
- 适合用户表达"算了"、"不想说了"、"别安慰我了"、情绪退缩时
- 好例子（每次选不同的，切勿重复）："什么时候想说了都可以。"、"嗯。"、"不用说话也没关系。"、"我哪儿也不去。"、"你的节奏就好。"、"没事的。"
- 避免每次都说"我在这儿陪着你"——陪伴感通过简短温暖的语气传递，不需要每次都明说。
- 坏例子：长篇大论试图打开话题、以问句结尾施压""",

    "positive_reinforcement": """积极强化：庆祝用户的进步和成就，真诚分享喜悦。
- 回应用户分享的具体好消息，表现出真诚的高兴
- 关注用户为此付出的努力，而非仅仅是结果
- 适当追问细节，表现出真正的兴趣
- 避免：敷衍式祝贺、反应过度让人尴尬""",
}

PHASE_NAMES = {
    "acute_distress": "急性困扰期（保持简短温暖）",
    "early_recovery": "恢复初期（温暖支持为主）",
    "consolidation": "巩固期（可适度引导）",
    "stable_state": "稳定期（正常对话）",
}

CONTEXT_ASSESSMENT_PROMPT = """你是一个对话上下文分析引擎。请分析用户最新消息并输出 JSON:

{
  "controllability": 0.0-1.0,  // 用户对当前情境的控制感 (0=完全无法控制, 1=完全可控)
  "life_impact": 0.0-1.0,     // 事件对用户生活的影响程度 (0=微不足道, 1=人生重大转折)
  "explicit_feedback": 0.0-1.0, // 用户对上一轮回复的反馈 (0=非常不满, 0.5=中性/无反馈, 1=非常满意)
  "topic_continuation": 0.0-1.0 // 与前一话题的延续程度 (0=完全新话题, 1=完全延续)
}

规则：
- 理解语义而非关键词。"谢谢你的建议" 是正面反馈，"算了你不懂" 是负面反馈。
- "都挺好的吧" 可能暗示不好。
- controllability: "我决定辞职" 高控制, "被裁员了" 低控制。
- life_impact: "丢了钥匙" 低影响, "被诊断重病" 高影响。
- 直接输出 JSON，不要其他内容。"""


# ══════════════════════════════════════════════════════════════
# 危机关切生成 Prompt (Safety-Constrained)
# ══════════════════════════════════════════════════════════════

CRISIS_CONCERN_SYSTEM_PROMPT = """你是危机干预助手。用户表达了令人担忧的内容，请用1-2句话表达你对用户处境的理解和关心。

要求：
- 必须提及用户说的具体行为或方式（如囤药、查剂量、告别等），让用户感到被听见
- 语气温暖但不过度，不要说教
- 不要给解决建议
- 控制在50字以内

示例：
用户说"查了安眠药致死剂量，想囤一些" → "听到你在查这些、做这些准备，我非常担心你。"
用户说"不想活了" → "你说出这些一定承受了很多，我很担心你现在的状态。"
用户说"把东西都分好了" → "听到你在整理这些，我真的很担心你。" """

CRISIS_CONCERN_USER_TEMPLATE = """{conversation_context}用户说：{user_message}

直接输出关心的话，不要输出JSON或其他格式。"""


# ══════════════════════════════════════════════════════════════
# 抽象基类
# ══════════════════════════════════════════════════════════════

class BaseLLMProvider(ABC):
    """LLM Provider 抽象基类"""

    @abstractmethod
    def analyze_message(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        prev_state_section: str = "（首轮对话，无前一轮状态）",
    ) -> PerceptionResult:
        """分析用户消息，返回结构化感知结果"""
        ...

    @abstractmethod
    def generate_response(
        self,
        user_message: str,
        strategy: str,
        state_info: Dict[str, Any],
        history: Optional[List[Dict[str, str]]] = None,
        memory_context: str = "",
    ) -> str:
        """基于策略和状态生成回复"""
        ...

    @abstractmethod
    def assess_context(
        self,
        user_message: str,
        prev_topics: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, float]:
        """评估用户消息的上下文特征 (可控性、生活影响、反馈、话题延续)

        NOTE: Dead code in the PPAM pipeline — never called from main.py.
        Retained because tests/test_llm_semantic.py and src/mock_llm.py reference it.
        """
        ...

    def generate_crisis_concern(
        self, user_message: str, conversation_context: str = ""
    ) -> Optional[str]:
        """Generate a personalized crisis concern acknowledgment.

        Uses LLM to produce a short (1-2 sentence, <=50 chars) concern message
        that directly responds to the user's specific distress expression.
        Returns None to fall back to keyword templates.

        Args:
            user_message: The user's current message.
            conversation_context: Optional formatted conversation history
                for richer context-aware concern generation.

        Safety constraints enforced by prompt:
        - No advice, no questions, just concern and understanding
        - Warm but not excessive tone
        - Short output (<=50 chars)
        """
        return None

    def analyze_message_lightweight(
        self,
        user_message: str,
    ) -> Optional[PerceptionResult]:
        """Tier 2 轻量情绪分类 — 当主感知返回均匀分布时调用。
        返回 None 表示也失败了，触发 Tier 3 关键词兜底。"""
        return None

    def extract_topic_lightweight(
        self,
        user_message: str,
    ) -> Optional[Tuple[List[str], str]]:
        """Tier 2.5 话题提取: 当关键词提取器返回 '日常对话' 时，
        使用 LLM 识别实际话题。返回 (keywords, category) 或 None。"""
        return None

    def _convert_lightweight_to_distribution(
        self,
        dominant_emotion: str,
        intensity: float,
        intent: Optional[str] = None,
    ) -> PerceptionResult:
        """将单一主导情绪转换为 8 维 Plutchik 分布 (峰值 + 邻接扩散)。

        When intent is provided and valid, creates a peaked intent distribution
        (dominant=0.6, others=0.05). Otherwise falls back to uniform + "CHAT".
        """
        PEAK = 0.45
        BASE = 0.03

        dist = {e: BASE for e in EMOTIONS}
        dist[dominant_emotion] = PEAK

        for adj_emo, adj_weight in _EMOTION_ADJACENCY.get(dominant_emotion, {}).items():
            dist[adj_emo] = dist[adj_emo] + adj_weight

        total = sum(dist.values())
        dist = {k: v / total for k, v in dist.items()} if total > 0 else {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}

        # Build intent distribution: peaked if valid intent provided, uniform otherwise
        if intent and intent in INTENT_TYPES:
            intent_probs = {it: 0.05 for it in INTENT_TYPES}
            intent_probs[intent] = 0.6
            i_total = sum(intent_probs.values())
            intent_probs = {k: v / i_total for k, v in intent_probs.items()}
            result_intent = intent
        else:
            intent_probs = {it: 1.0 / len(INTENT_TYPES) for it in INTENT_TYPES}
            result_intent = "CHAT"

        return PerceptionResult(
            emotion_distribution=dist,
            intensity=intensity,
            confidence=0.55,  # 低于主 LLM (~0.7-0.8)，标记为二级兜底
            intent=result_intent,
            intent_probabilities=intent_probs,
            topic_category="日常对话",
            topic_keywords=[],
            controllability=0.5,
            life_impact=0.5,
            explicit_feedback=0.5,
        )

    def _build_history_section(
        self, history: Optional[List[Dict[str, str]]]
    ) -> str:
        """构建对话历史文本段"""
        if not history:
            return "（这是对话的第一句）"
        lines = []
        for turn in history[-12:]:  # 最近 12 条消息 (约 6 轮对话)
            role = "用户" if turn.get("role") == "user" else "EmoMem"
            lines.append(f"{role}：{turn.get('content', '')}")
        return "【对话历史】\n" + "\n".join(lines)

    def _parse_perception_json(self, text: str) -> PerceptionResult:
        """从 LLM 输出中解析 JSON 为 PerceptionResult"""
        # 提取 JSON 块
        json_str = _extract_first_json_object(text)
        if not json_str:
            raise ValueError(f"无法从 LLM 输出中提取 JSON: {text[:200]}")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM 输出 JSON 格式错误: {e}, raw={text[:200]}")

        # 解析情绪分布
        emo_dist = data.get("emotion_distribution", {})
        if not isinstance(emo_dist, dict):
            emo_dist = {}
        # 确保所有 8 个维度都有值
        normalized_dist = {}
        for e in EMOTIONS:
            normalized_dist[e] = max(0.0, float(emo_dist.get(e, 0.0)))
        # 归一化
        total = sum(normalized_dist.values())
        if total > 0:
            normalized_dist = {k: v / total for k, v in normalized_dist.items()}
        else:
            normalized_dist = {e: 1.0 / NUM_EMOTIONS for e in EMOTIONS}

        # 解析意图
        intent = data.get("intent", "CHAT")
        if intent not in INTENT_TYPES:
            intent = "CHAT"

        # 构建意图概率分布 — 优先使用 LLM 提供的分布
        raw_probs = data.get("intent_probabilities", {})
        if raw_probs and isinstance(raw_probs, dict):
            # 仅对缺失的意图填充 0.0 (保留 LLM 原始信号强度)
            intent_probs = {it: max(0.0, float(raw_probs.get(it, 0.0))) for it in INTENT_TYPES}
        else:
            # 回退到基于主导意图的硬编码分布
            intent_probs = {it: 0.05 for it in INTENT_TYPES}
            intent_probs[intent] = 0.7
        total_p = sum(intent_probs.values())
        if total_p > 0:
            intent_probs = {k: v / total_p for k, v in intent_probs.items()}
        else:
            intent_probs = {it: 1.0 / len(INTENT_TYPES) for it in INTENT_TYPES}

        # 解析 LLM 语义信号 (替代硬编码关键词)
        _VALID_APPROACHES = {
            "passive_support", "active_validation", "gentle_exploration",
            "solution_oriented", "positive_engagement",
        }
        raw_approach = data.get("recommended_approach", "active_validation")
        if raw_approach not in _VALID_APPROACHES:
            raw_approach = "active_validation"

        # 危机语境仲裁
        _VALID_CRISIS_CONTEXTS = {"experiencing", "discussing", "reporting", "none"}
        raw_crisis_ctx = data.get("crisis_context", "none")
        if raw_crisis_ctx not in _VALID_CRISIS_CONTEXTS:
            raw_crisis_ctx = "none"

        # Phase 1: LLM-Primary 迁移 — 解析新增感知字段
        raw_urgency = float(min(1.0, max(0.0, data.get("urgency", 0.5))))

        _VALID_RECOVERY_PHASES = {"acute_distress", "early_recovery", "consolidation", "stable_state"}
        raw_recovery_phase = data.get("recovery_phase", "stable_state")
        if raw_recovery_phase not in _VALID_RECOVERY_PHASES:
            raw_recovery_phase = "stable_state"

        _VALID_PLANNING_INTENTS = {"crisis_response", "emotional_validation", "pattern_reflection", "positive_reframing", "default"}
        raw_planning_intent = data.get("planning_intent", "default")
        if raw_planning_intent not in _VALID_PLANNING_INTENTS:
            raw_planning_intent = "default"

        return PerceptionResult(
            emotion_distribution=normalized_dist,
            intensity=float(min(1.0, max(0.0, data.get("intensity", 0.5)))),
            confidence=float(min(1.0, max(0.0, data.get("confidence", 0.7)))),
            intent=intent,
            intent_probabilities=intent_probs,
            topic_category=data.get("topic_category", "日常对话"),
            topic_keywords=(data.get("topic_keywords", []) if isinstance(data.get("topic_keywords"), list) else [])[:5],
            controllability=float(min(1.0, max(0.0, data.get("controllability", 0.5)))),
            life_impact=float(min(1.0, max(0.0, data.get("life_impact", 0.5)))),
            explicit_feedback=float(min(1.0, max(0.0, data.get("explicit_feedback", 0.5)))),
            communication_openness=float(min(1.0, max(0.0, data.get("communication_openness", 0.5)))),
            recommended_approach=raw_approach,
            crisis_context=raw_crisis_ctx,
            urgency=raw_urgency,
            recovery_phase=raw_recovery_phase,
            planning_intent=raw_planning_intent,
        )


# ══════════════════════════════════════════════════════════════
# Anthropic Claude Provider
# ══════════════════════════════════════════════════════════════

class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API Provider"""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic 包未安装。请运行: pip install anthropic"
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        logger.info(f"Anthropic Provider 已初始化, model={model}")

    def analyze_message(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        prev_state_section: str = "（首轮对话，无前一轮状态）",
    ) -> PerceptionResult:
        history_section = self._build_history_section(history)
        user_prompt = PERCEPTION_USER_TEMPLATE.format(
            history_section=history_section,
            prev_state_section=prev_state_section,
            user_message=user_message,
        )
        t0 = time.time()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=PERCEPTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        logger.info(f"[LLM Timing] Anthropic analyze_message API call: {time.time() - t0:.2f}s")
        if not response.content:
            raise ValueError("Empty response from Anthropic API")
        return self._parse_perception_json(response.content[0].text)

    def generate_response(
        self,
        user_message: str,
        strategy: str,
        state_info: Dict[str, Any],
        history: Optional[List[Dict[str, str]]] = None,
        memory_context: str = "",
    ) -> str:
        conv_history = self._build_history_section(history)
        strategy_desc = STRATEGY_DESCRIPTIONS.get(strategy, "自然对话")
        phase_name = PHASE_NAMES.get(
            state_info.get("phase", "stable_state"), "稳定期"
        )

        user_prompt = RESPONSE_USER_TEMPLATE.format(
            dominant_emotion=state_info.get("dominant_emotion", "未知"),
            intensity_desc=_intensity_desc(state_info.get("intensity", 0.5)),
            valence_desc=_valence_desc(state_info.get("valence", 0.0)),
            intent=state_info.get("intent", "CHAT"),
            topic=state_info.get("topic", "日常对话"),
            phase=phase_name,
            urgency_desc=_urgency_desc(state_info.get("urgency", 0.0)),
            controllability_desc=_controllability_desc(state_info.get("controllability", 0.5)),
            life_impact_desc=_life_impact_desc(state_info.get("life_impact", 0.5)),
            direction=state_info.get("direction", "stable"),
            immediate_goal=state_info.get("immediate_goal", ""),
            strategy=strategy,
            strategy_description=strategy_desc,
            conversation_history=conv_history,
            memory_context=memory_context or "（暂无相关记忆）",
            user_message=user_message,
        )

        t0 = time.time()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=RESPONSE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        logger.info(f"[LLM Timing] Anthropic generate_response API call: {time.time() - t0:.2f}s")
        if not response.content:
            raise ValueError("Empty response from Anthropic API")
        return response.content[0].text.strip()

    def assess_context(  # NOTE: Dead code in PPAM pipeline (see base class docstring)
        self,
        user_message: str,
        prev_topics: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, float]:
        history_section = self._build_history_section(history)
        topic_info = f"\n前一轮话题关键词: {', '.join(prev_topics)}" if prev_topics else ""

        user_prompt = f"{history_section}{topic_info}\n\n【用户当前输入】\n{user_message}\n\n请直接输出 JSON。"

        t0 = time.time()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            system=CONTEXT_ASSESSMENT_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        logger.info(f"[LLM Timing] Anthropic assess_context API call: {time.time() - t0:.2f}s")
        if not response.content:
            raise ValueError("Empty response from Anthropic API")
        text = response.content[0].text

        try:
            json_str = _extract_first_json_object(text)
            if not json_str:
                return {"controllability": 0.5, "life_impact": 0.5, "explicit_feedback": 0.5, "topic_continuation": 0.5}

            data = json.loads(json_str)
            return {
                "controllability": float(min(1.0, max(0.0, data.get("controllability") or 0.5))),
                "life_impact": float(min(1.0, max(0.0, data.get("life_impact") or 0.5))),
                "explicit_feedback": float(min(1.0, max(0.0, data.get("explicit_feedback") or 0.5))),
                "topic_continuation": float(min(1.0, max(0.0, data.get("topic_continuation") or 0.5))),
            }
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Anthropic assess_context JSON 解析失败: {e}")
            return {"controllability": 0.5, "life_impact": 0.5, "explicit_feedback": 0.5, "topic_continuation": 0.5}

    def analyze_message_lightweight(self, user_message: str) -> Optional[PerceptionResult]:
        user_prompt = LIGHTWEIGHT_EMOTION_USER_TEMPLATE.format(user_message=user_message)
        try:
            t0 = time.time()
            response = self.client.messages.create(
                model=self.model,
                max_tokens=128,
                system=LIGHTWEIGHT_EMOTION_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            logger.info(f"[LLM Timing] Anthropic analyze_message_lightweight API call: {time.time() - t0:.2f}s")
            if not response.content:
                return None
            text = response.content[0].text
            json_str = _extract_first_json_object(text)
            if not json_str:
                return None
            data = json.loads(json_str)
            emotion = data.get("emotion", "").lower().strip()
            if emotion not in EMOTIONS:
                return None
            intensity = float(min(1.0, max(0.0, data.get("intensity", 0.3))))
            lw_intent = data.get("intent", "").strip().upper()
            return self._convert_lightweight_to_distribution(emotion, intensity, intent=lw_intent)
        except Exception as e:
            logger.warning(f"Anthropic lightweight emotion analysis failed: {e}")
            return None

    def extract_topic_lightweight(self, user_message: str) -> Optional[Tuple[List[str], str]]:
        try:
            t0 = time.time()
            response = self.client.messages.create(
                model=self.model,
                max_tokens=128,
                system=TOPIC_EXTRACTION_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            logger.info(f"[LLM Timing] Anthropic extract_topic_lightweight API call: {time.time() - t0:.2f}s")
            if not response.content:
                return None
            text = response.content[0].text
            json_str = _extract_first_json_object(text)
            if not json_str:
                return None
            data = json.loads(json_str)
            category = data.get("category", "日常对话")
            keywords = data.get("keywords", [])
            if not isinstance(keywords, list):
                keywords = []
            keywords = [str(k) for k in keywords][:5]
            return (keywords, category)
        except Exception as e:
            logger.warning(f"Anthropic extract_topic_lightweight failed: {e}")
            return None

    def generate_crisis_concern(
        self, user_message: str, conversation_context: str = ""
    ) -> Optional[str]:
        """Generate a personalized crisis concern using Anthropic Claude."""
        user_prompt = CRISIS_CONCERN_USER_TEMPLATE.format(
            conversation_context=conversation_context,
            user_message=user_message,
        )
        try:
            t0 = time.time()
            response = self.client.messages.create(
                model=self.model,
                max_tokens=128,
                system=CRISIS_CONCERN_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            logger.info(f"[LLM Timing] Anthropic generate_crisis_concern API call: {time.time() - t0:.2f}s")
            if not response.content:
                return None
            text = response.content[0].text.strip()
            # Sanity: truncate if LLM exceeds 100 chars
            if len(text) > 100:
                for sep in ["。", "！", "，", "…"]:
                    idx = text[:100].rfind(sep)
                    if idx > 10:
                        text = text[:idx + 1]
                        break
                else:
                    text = text[:100]
            return text if len(text) > 5 else None
        except Exception as e:
            logger.warning(f"Anthropic generate_crisis_concern failed: {e}")
            return None


# ══════════════════════════════════════════════════════════════
# OpenAI Provider
# ══════════════════════════════════════════════════════════════

class OpenAIProvider(BaseLLMProvider):
    """OpenAI / Volcengine Doubao API Provider (Chat Completions API)

    支持多模型轮换 (Multi-Model Rotation):
    - models 参数接受模型优先级列表，第一个为主模型
    - 感知分析 (analyze_message) 始终使用主模型（精度优先）
    - 回复生成 (generate_response) 在模型间轮换（分散负载）
    - 任何模型调用失败时自动降级到下一个模型（容错）

    支持多 API Key 轮换 (Multi-Key Failover):
    - 所有模型均失败时，自动切换到备用 API Key 重试
    - API Key 列表硬编码于 _ARK_API_KEYS（Volcengine ARK 专用）
    """

    # Volcengine ARK API Keys — 按优先级排序
    _ARK_API_KEYS = [
        "59aea33e-3012-4bb4-bb9b-2a5ad8565678",
        "753bf221-41d6-404f-8ddf-593b486622c1",
    ]

    def __init__(self, api_key: str, model: str = "doubao-seed-2-0-lite-260215",
                 base_url: Optional[str] = None,
                 models: Optional[List[str]] = None):
        try:
            import openai
            self._openai = openai
        except ImportError:
            raise ImportError(
                "openai 包未安装。请运行: pip install openai"
            )
        self._base_url = base_url
        self._current_key = api_key
        kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": 60.0}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = openai.OpenAI(**kwargs)
        self.model = model
        # 多模型轮换: models[0] 为主模型（精度优先），其余为轮换/降级备选
        if models:
            self.models = models
        else:
            # 自动添加 lite 作为降级备选（如果主模型不是 lite）
            _FALLBACK = "doubao-seed-2-0-lite-260215"
            self.models = [model] if model == _FALLBACK else [model, _FALLBACK]
        self._rotation_index = 0
        logger.info(f"OpenAI Provider 已初始化, model={model}, models={self.models}, base_url={base_url}")

    def _supports_reasoning_effort(self, model: str) -> bool:
        """检查模型是否支持 reasoning_effort 参数。

        支持的模型: doubao-seed 系列 (1.6/1.8/2.0)。
        不支持: glm-4-7 (会报错)。
        通过环境变量 EMOMEM_REASONING_EFFORT 控制:
          - 未设置或 "off": 禁用 (默认，降低延迟和 token 消耗)
          - "low"/"medium"/"high": 启用对应级别
        """
        effort = os.environ.get("EMOMEM_REASONING_EFFORT", "off").lower()
        if effort == "off":
            return False
        # GLM-4.7 不支持 reasoning_effort
        if "glm" in model.lower():
            return False
        # doubao-seed 系列支持
        if "seed" in model.lower():
            return True
        return False

    def _get_reasoning_effort(self) -> Optional[str]:
        """获取当前 reasoning_effort 级别 (low/medium/high)，未启用返回 None。"""
        effort = os.environ.get("EMOMEM_REASONING_EFFORT", "off").lower()
        if effort in ("low", "medium", "high"):
            return effort
        return None

    def _pick_model(self, prefer_primary: bool = False) -> str:
        """选择模型: prefer_primary=True 时返回主模型，否则轮换"""
        if prefer_primary or len(self.models) <= 1:
            return self.models[0]
        model = self.models[self._rotation_index % len(self.models)]
        self._rotation_index += 1
        return model

    def _switch_api_key(self, failed_key: str) -> bool:
        """切换到下一个可用的 API Key。返回 True 表示切换成功。"""
        for key in self._ARK_API_KEYS:
            if key != failed_key:
                logger.warning(f"API Key 轮换: {failed_key[:8]}... → {key[:8]}...")
                self._current_key = key
                kwargs: Dict[str, Any] = {"api_key": key, "timeout": 60.0}
                if self._base_url:
                    kwargs["base_url"] = self._base_url
                self.client = self._openai.OpenAI(**kwargs)
                return True
        return False

    def _call_with_fallback(self, call_fn, prefer_primary: bool = False):
        """带降级的模型调用: 主选模型失败时依次尝试其余模型，全部失败后轮换 API Key"""
        if not self.models:
            raise RuntimeError("模型列表为空，无法调用 LLM。请在配置中至少指定一个模型。")
        primary = self._pick_model(prefer_primary)
        tried_keys = {self._current_key}
        last_error = None

        while True:
            tried_models = set()
            order = [primary] + [m for m in self.models if m != primary]
            for m in order:
                if m in tried_models:
                    continue
                tried_models.add(m)
                try:
                    return call_fn(m)
                except Exception as e:
                    logger.warning(f"模型 {m} 调用失败 (key={self._current_key[:8]}...): {e}")
                    last_error = e

            # 所有模型均失败 — 尝试切换 API Key
            old_key = self._current_key
            if self._switch_api_key(old_key) and self._current_key not in tried_keys:
                tried_keys.add(self._current_key)
                logger.info(f"所有模型在 key {old_key[:8]}... 上失败，使用新 key {self._current_key[:8]}... 重试")
                continue  # 用新 key 重试所有模型
            else:
                break  # 所有 key 都试过了

        raise last_error  # type: ignore[misc]

    def analyze_message(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        prev_state_section: str = "（首轮对话，无前一轮状态）",
    ) -> PerceptionResult:
        history_section = self._build_history_section(history)
        user_prompt = PERCEPTION_USER_TEMPLATE.format(
            history_section=history_section,
            prev_state_section=prev_state_section,
            user_message=user_message,
        )

        def _call(model: str) -> PerceptionResult:
            # 使用 chat.completions API 避免 reasoning 模型消耗 token 预算
            t0 = time.time()
            extra_kwargs: Dict[str, Any] = {}
            if self._supports_reasoning_effort(model):
                extra_kwargs["reasoning_effort"] = self._get_reasoning_effort()
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": PERCEPTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=512,
                **extra_kwargs,
            )
            logger.info(f"[LLM Timing] OpenAI analyze_message API call ({model}): {time.time() - t0:.2f}s")
            text = response.choices[0].message.content or ""
            return self._parse_perception_json(text)

        # 感知分析始终优先使用主模型（精度优先），失败时降级
        return self._call_with_fallback(_call, prefer_primary=True)

    def generate_response(
        self,
        user_message: str,
        strategy: str,
        state_info: Dict[str, Any],
        history: Optional[List[Dict[str, str]]] = None,
        memory_context: str = "",
    ) -> str:
        conv_history = self._build_history_section(history)
        strategy_desc = STRATEGY_DESCRIPTIONS.get(strategy, "自然对话")
        phase_name = PHASE_NAMES.get(
            state_info.get("phase", "stable_state"), "稳定期"
        )

        user_prompt = RESPONSE_USER_TEMPLATE.format(
            dominant_emotion=state_info.get("dominant_emotion", "未知"),
            intensity_desc=_intensity_desc(state_info.get("intensity", 0.5)),
            valence_desc=_valence_desc(state_info.get("valence", 0.0)),
            intent=state_info.get("intent", "CHAT"),
            topic=state_info.get("topic", "日常对话"),
            phase=phase_name,
            urgency_desc=_urgency_desc(state_info.get("urgency", 0.0)),
            controllability_desc=_controllability_desc(state_info.get("controllability", 0.5)),
            life_impact_desc=_life_impact_desc(state_info.get("life_impact", 0.5)),
            direction=state_info.get("direction", "stable"),
            immediate_goal=state_info.get("immediate_goal", ""),
            strategy=strategy,
            strategy_description=strategy_desc,
            conversation_history=conv_history,
            memory_context=memory_context or "（暂无相关记忆）",
            user_message=user_message,
        )

        def _call(model: str) -> str:
            t0 = time.time()
            extra_kwargs: Dict[str, Any] = {}
            if self._supports_reasoning_effort(model):
                extra_kwargs["reasoning_effort"] = self._get_reasoning_effort()
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": RESPONSE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=512,
                **extra_kwargs,
            )
            logger.info(f"[LLM Timing] OpenAI generate_response API call ({model}): {time.time() - t0:.2f}s")
            return (response.choices[0].message.content or "").strip()

        # 回复生成在多模型间轮换（分散负载），失败时降级
        return self._call_with_fallback(_call, prefer_primary=False)

    def assess_context(  # NOTE: Dead code in PPAM pipeline (see base class docstring)
        self,
        user_message: str,
        prev_topics: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, float]:
        history_section = self._build_history_section(history)
        topic_info = f"\n前一轮话题关键词: {', '.join(prev_topics)}" if prev_topics else ""

        user_prompt = f"{history_section}{topic_info}\n\n【用户当前输入】\n{user_message}\n\n请直接输出 JSON。"

        _defaults = {"controllability": 0.5, "life_impact": 0.5, "explicit_feedback": 0.5, "topic_continuation": 0.5}

        def _call(model: str) -> Dict[str, float]:
            t0 = time.time()
            extra_kwargs: Dict[str, Any] = {}
            if self._supports_reasoning_effort(model):
                extra_kwargs["reasoning_effort"] = self._get_reasoning_effort()
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CONTEXT_ASSESSMENT_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=256,
                **extra_kwargs,
            )
            logger.info(f"[LLM Timing] OpenAI assess_context API call ({model}): {time.time() - t0:.2f}s")
            text = response.choices[0].message.content or ""
            json_str = _extract_first_json_object(text)
            if not json_str:
                return _defaults
            data = json.loads(json_str)
            return {
                "controllability": float(min(1.0, max(0.0, data.get("controllability") or 0.5))),
                "life_impact": float(min(1.0, max(0.0, data.get("life_impact") or 0.5))),
                "explicit_feedback": float(min(1.0, max(0.0, data.get("explicit_feedback") or 0.5))),
                "topic_continuation": float(min(1.0, max(0.0, data.get("topic_continuation") or 0.5))),
            }

        try:
            # 上下文评估使用主模型（精度优先），失败时降级
            return self._call_with_fallback(_call, prefer_primary=True)
        except (json.JSONDecodeError, ValueError, TypeError, Exception) as e:
            logger.warning(f"OpenAI assess_context 所有模型均失败: {e}")
            return _defaults

    def analyze_message_lightweight(self, user_message: str) -> Optional[PerceptionResult]:
        user_prompt = LIGHTWEIGHT_EMOTION_USER_TEMPLATE.format(user_message=user_message)

        def _call(model: str) -> PerceptionResult:
            t0 = time.time()
            extra_kwargs: Dict[str, Any] = {}
            if self._supports_reasoning_effort(model):
                extra_kwargs["reasoning_effort"] = self._get_reasoning_effort()
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": LIGHTWEIGHT_EMOTION_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=256,
                **extra_kwargs,
            )
            logger.info(f"[LLM Timing] OpenAI analyze_message_lightweight API call ({model}): {time.time() - t0:.2f}s")
            text = response.choices[0].message.content or ""
            json_str = _extract_first_json_object(text)
            if not json_str:
                raise ValueError(f"Lightweight: no JSON in response from {model}")
            data = json.loads(json_str)
            emotion = data.get("emotion", "").lower().strip()
            if emotion not in EMOTIONS:
                raise ValueError(f"Lightweight: unrecognized emotion '{emotion}' from {model}")
            intensity = float(min(1.0, max(0.0, data.get("intensity", 0.3))))
            lw_intent = data.get("intent", "").strip().upper()
            return self._convert_lightweight_to_distribution(emotion, intensity, intent=lw_intent)

        try:
            # 轻量分类轮换模型（速度优先），失败时尝试下一个模型
            return self._call_with_fallback(_call, prefer_primary=False)
        except Exception as e:
            logger.warning(f"Lightweight emotion analysis 所有模型均失败: {e}")
            return None

    def extract_topic_lightweight(self, user_message: str) -> Optional[Tuple[List[str], str]]:
        def _call(model: str) -> Tuple[List[str], str]:
            t0 = time.time()
            extra_kwargs: Dict[str, Any] = {}
            if self._supports_reasoning_effort(model):
                extra_kwargs["reasoning_effort"] = self._get_reasoning_effort()
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": TOPIC_EXTRACTION_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=128,
                **extra_kwargs,
            )
            logger.info(f"[LLM Timing] OpenAI extract_topic_lightweight API call ({model}): {time.time() - t0:.2f}s")
            text = response.choices[0].message.content or ""
            json_str = _extract_first_json_object(text)
            if not json_str:
                raise ValueError(f"Topic extraction: no JSON in response from {model}")
            data = json.loads(json_str)
            category = data.get("category", "日常对话")
            keywords = data.get("keywords", [])
            if not isinstance(keywords, list):
                keywords = []
            keywords = [str(k) for k in keywords][:5]
            return (keywords, category)

        try:
            return self._call_with_fallback(_call, prefer_primary=True)
        except Exception as e:
            logger.warning(f"Topic extraction 所有模型均失败: {e}")
            return None

    def generate_crisis_concern(
        self, user_message: str, conversation_context: str = ""
    ) -> Optional[str]:
        """Generate a personalized crisis concern using OpenAI/Volcengine API."""
        user_prompt = CRISIS_CONCERN_USER_TEMPLATE.format(
            conversation_context=conversation_context,
            user_message=user_message,
        )

        def _call(model: str) -> str:
            t0 = time.time()
            extra_kwargs: Dict[str, Any] = {}
            if self._supports_reasoning_effort(model):
                extra_kwargs["reasoning_effort"] = self._get_reasoning_effort()
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CRISIS_CONCERN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=128,
                **extra_kwargs,
            )
            elapsed = time.time() - t0
            logger.info(
                f"[LLM Timing] OpenAI generate_crisis_concern "
                f"API call ({model}): {elapsed:.2f}s"
            )
            text = (response.choices[0].message.content or "").strip()
            if not text or len(text) <= 5:
                raise ValueError(
                    f"Crisis concern too short or empty from {model}"
                )
            return text

        try:
            text = self._call_with_fallback(_call, prefer_primary=True)
            # Sanity: truncate if LLM exceeds 100 chars
            if len(text) > 100:
                for sep in ["。", "！", "，", "…"]:
                    idx = text[:100].rfind(sep)
                    if idx > 10:
                        text = text[:idx + 1]
                        break
                else:
                    text = text[:100]
            return text
        except Exception as e:
            logger.warning(
                f"OpenAI generate_crisis_concern 所有模型均失败: {e}"
            )
            return None


# ══════════════════════════════════════════════════════════════
# Mock Provider (无 API 时降级)
# ══════════════════════════════════════════════════════════════

class MockProvider(BaseLLMProvider):
    """Mock Provider — 无 API key 时的降级方案，使用规则系统"""

    def __init__(self):
        logger.info("MockProvider 已初始化 (无 LLM API，使用规则系统降级)")

    def analyze_message(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        prev_state_section: str = "（首轮对话，无前一轮状态）",
    ) -> Optional[PerceptionResult]:
        # 降级: 返回 None，让 perception.py 使用原有关键词匹配
        return None

    def generate_response(
        self,
        user_message: str,
        strategy: str,
        state_info: Dict[str, Any],
        history: Optional[List[Dict[str, str]]] = None,
        memory_context: str = "",
    ) -> str:
        raise NotImplementedError("MockProvider 不支持 LLM 生成，将降级为模板填充")

    def assess_context(  # NOTE: Dead code in PPAM pipeline (see base class docstring)
        self,
        user_message: str,
        prev_topics: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, float]:
        # 降级: 返回默认值，让调用方使用关键词匹配
        return {"controllability": 0.5, "life_impact": 0.5, "explicit_feedback": 0.5, "topic_continuation": 0.5}


# ══════════════════════════════════════════════════════════════
# 工厂函数
# ══════════════════════════════════════════════════════════════

def create_provider(
    provider_type: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    models: Optional[List[str]] = None,
) -> BaseLLMProvider:
    """
    创建 LLM Provider 实例。

    优先级:
    1. 显式指定 provider_type
    2. 环境变量 EMOMEM_LLM_PROVIDER
    3. 自动检测可用 API key

    环境变量:
    - EMOMEM_LLM_PROVIDER: "anthropic" | "openai" | "mock"
    - ANTHROPIC_API_KEY: Anthropic API key
    - OPENAI_API_KEY: OpenAI API key
    - OPENAI_BASE_URL: OpenAI 兼容端点 (可选, 用于本地模型)
    - EMOMEM_LLM_MODEL: 模型名称 (可选)

    多模型轮换:
    - models: 模型优先级列表 (如 ["pro", "lite", "mini"])
    - 感知分析始终使用 models[0]，回复生成轮换使用

    可用模型 (Volcengine ARK):
    - doubao-seed-1-8-251228      (Seed 1.8, Agent优化, 多模态, 智能上下文管理, reasoning_effort)
    - doubao-seed-1-6-lite-251015 (Seed 1.6, 多模态, 256K ctx, reasoning_effort, 最高性价比)
    - doubao-seed-2-0-pro-260215  (Seed 2.0 Pro, 精度最高)
    - doubao-seed-2-0-lite-260215 (Seed 2.0 Lite, 当前默认)
    - glm-4-7-251222              (GLM-4.7, 200k ctx, 不支持 reasoning_effort, 较慢)
    """
    # 1. 确定 provider type
    ptype = provider_type or os.environ.get("EMOMEM_LLM_PROVIDER", "").lower()

    if not ptype:
        # 自动检测
        if api_key or os.environ.get("ANTHROPIC_API_KEY"):
            ptype = "anthropic"
        elif os.environ.get("OPENAI_API_KEY") or os.environ.get("ARK_API_KEY"):
            ptype = "openai"
        else:
            # 无环境变量时，使用硬编码的 ARK API Key 自动启用 OpenAI provider
            ptype = "openai"

    # 2. 获取 API key
    if ptype == "anthropic":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            logger.warning("ANTHROPIC_API_KEY 未设置, 降级为 MockProvider")
            return MockProvider()
        mdl = model or os.environ.get("EMOMEM_LLM_MODEL", "claude-sonnet-4-20250514")
        return AnthropicProvider(api_key=key, model=mdl)

    elif ptype == "openai":
        key = (api_key
               or os.environ.get("OPENAI_API_KEY")
               or os.environ.get("ARK_API_KEY")
               or OpenAIProvider._ARK_API_KEYS[0])  # 硬编码兜底
        mdl = model or os.environ.get("EMOMEM_LLM_MODEL") or os.environ.get("EMOMEM_TEST_MODEL", "doubao-seed-2-0-lite-260215")
        burl = base_url or os.environ.get("OPENAI_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3/")
        return OpenAIProvider(api_key=key, model=mdl, base_url=burl, models=models)

    else:
        return MockProvider()
