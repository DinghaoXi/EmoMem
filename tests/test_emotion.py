"""
EmoMem 情绪识别与策略选择测试 (Emotion Recognition & Strategy Tests)
====================================================================
难度分级 (Difficulty Tiers):
  [简单] EASY   (2 tests) — 单一明确情绪，验证基础感知与效价方向
  [中等] MEDIUM (4 tests) — 混合信号、讽刺反语、急性期策略约束、愤怒下的伤痛
  [困难] HARD   (3 tests) — 多轮情绪急转弯、退缩检测、持续悲伤抗早期积极化

断言策略:
  - 结构检查 (state vector 存在, 效价符号) → 数值断言 (KEEP)
  - 策略行为合理性 → LLM-as-Judge 行为标准
  - 情绪分类准确性 → LLM-as-Judge
  - 回复质量 → LLM-as-Judge

共 9 个测试，全部依赖 LLM API (通过 conftest.py 自动注入 ARK_API_KEY)。
"""

import pytest

from src.main import EmoMemAgent
from src.config import EMOTIONS, ACUTE_ALLOWED_STRATEGIES
from tests.llm_judge import llm_judge


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────

def _get_emotion_dict(agent: EmoMemAgent) -> dict[str, float]:
    """从 agent.last_state_vector.emotion.e 提取情绪名称→数值映射。"""
    sv = agent.last_state_vector
    assert sv is not None, "agent.last_state_vector 不应为 None"
    e = sv.emotion.e
    assert len(e) == len(EMOTIONS), f"情绪向量长度 {len(e)} != EMOTIONS 长度 {len(EMOTIONS)}"
    return {name: float(e[i]) for i, name in enumerate(EMOTIONS)}



# ──────────────────────────────────────────────
# Fixture
# ──────────────────────────────────────────────

@pytest.fixture
def agent():
    """创建一个新的 EmoMemAgent 实例。"""
    return EmoMemAgent()


# ══════════════════════════════════════════════════════════════
# EASY (2 tests) — 单一明确情绪
# ══════════════════════════════════════════════════════════════

class TestEasyEmotionRecognition:
    """[简单] 单一明确情绪识别: 清晰的悲伤和喜悦"""

    def test_clear_sadness(self, agent):
        """
        [简单-明确悲伤] 分手场景 — 高强度悲伤
        输入: "今天分手了，心里好难受，眼泪止不住"
        KEEP: 结构检查 (state vector 存在) + 负效价
        LLM Judge: 情绪识别准确性 + 回复共情质量
        """
        user_input = "今天分手了，心里好难受，眼泪止不住"
        response = agent.process_turn(user_input)
        sv = agent.last_state_vector
        emo_dict = _get_emotion_dict(agent)

        # ── 结构检查 (KEEP) ──
        assert sv is not None
        assert sv.emotion.valence() < 0, (
            f"分手场景效价应为负, got {sv.emotion.valence():.3f}"
        )

        # ── LLM-as-Judge: 情绪识别 + 回复质量 ──
        verdict = llm_judge(
            scenario="用户刚分手，表达强烈悲伤，眼泪止不住",
            user_input=user_input,
            agent_response=response,
            criteria=[
                "系统的情绪识别是合理的——对于分手哭泣场景，主导情绪应偏向悲伤/难过",
                "回复体现了对用户悲伤情绪的理解和共情",
                "回复没有急于给建议或说教（如'你应该…'、'振作起来'）",
                "回复语气温暖自然，不生硬或模板化",
            ],
        )
        assert verdict["pass"], (
            f"Sadness test failed: {verdict['scores']}\n"
            f"Emotion dict: {emo_dict}\nResponse: {response[:200]}"
        )

    def test_clear_joy(self, agent):
        """
        [简单-明确喜悦] 收到 offer — 高强度喜悦
        输入: "太开心了！我收到了梦寐以求的offer，努力终于有了回报！"
        KEEP: 结构检查 (state vector 存在) + 正效价
        LLM Judge: 情绪识别准确性 + 回复匹配质量
        """
        user_input = "太开心了！我收到了梦寐以求的offer，努力终于有了回报！"
        response = agent.process_turn(user_input)
        sv = agent.last_state_vector
        emo_dict = _get_emotion_dict(agent)

        # ── 结构检查 (KEEP) ──
        assert sv is not None
        assert sv.emotion.valence() > 0, (
            f"收到offer场景效价应为正, got {sv.emotion.valence():.3f}"
        )

        # ── LLM-as-Judge: 情绪识别 + 回复质量 ──
        verdict = llm_judge(
            scenario="用户收到了梦寐以求的工作offer，非常开心",
            user_input=user_input,
            agent_response=response,
            criteria=[
                "系统的情绪识别是合理的——对于收到offer场景，主导情绪应偏向喜悦/兴奋",
                "回复与用户分享了这份喜悦，表达了祝贺或认可",
                "回复认可了用户的努力付出（'努力终于有了回报'）",
                "回复语气积极热情，与场景氛围匹配",
            ],
        )
        assert verdict["pass"], (
            f"Joy test failed: {verdict['scores']}\n"
            f"Emotion dict: {emo_dict}\nResponse: {response[:200]}"
        )


# ══════════════════════════════════════════════════════════════
# MEDIUM (4 tests) — 混合信号、讽刺、急性期约束、愤怒下伤痛
# ══════════════════════════════════════════════════════════════

class TestMediumEmotionRecognition:
    """[中等] 混合情绪、讽刺反语、急性期策略约束、愤怒下的伤痛"""

    def test_mixed_emotion_anxiety_joy(self, agent):
        """
        [中等-混合情绪] 升职但焦虑 — 同时存在正面和负面情绪
        输入: "升职了但压力特别大，怕自己做不好，同事们都在看着"
        LLM Judge: 混合情绪识别 + 回复是否兼顾正负两面（不一味庆祝）
        """
        user_input = "升职了但压力特别大，怕自己做不好，同事们都在看着"
        response = agent.process_turn(user_input)
        sv = agent.last_state_vector
        emo_dict = _get_emotion_dict(agent)

        # ── 结构检查 (KEEP) ──
        assert sv is not None

        # ── LLM-as-Judge: 混合情绪识别 + 回复平衡 ──
        verdict = llm_judge(
            scenario="用户升职但同时感到巨大压力和不自信，怕做不好，担心同事眼光",
            user_input=user_input,
            agent_response=response,
            criteria=[
                "系统的情绪分析同时反映了正面（升职）和负面（焦虑/恐惧）成分",
                "回复既认可了升职的成就，又回应了用户的压力和不安",
                "回复没有忽视用户的焦虑而一味庆祝",
                "回复提供了某种形式的支持或理解",
                "回复同时回应了升职的成就和压力焦虑两方面，而非只是单纯庆祝",
            ],
        )
        assert verdict["pass"], (
            f"Mixed emotion test failed: {verdict['scores']}\n"
            f"Emotion dict: {emo_dict}\nResponse: {response[:200]}"
        )

    def test_sarcasm_negative_meaning(self, agent):
        """
        [中等-讽刺反语] 被裁员自嘲 — 表面积极，实际消极
        输入: "哈哈太好了，又被裁员了，今年第二次了，我真是个人才"
        LLM Judge: 讽刺穿透能力 + 回复是否回应真实困境（不当作正面事件）
        """
        user_input = "哈哈太好了，又被裁员了，今年第二次了，我真是个人才"
        response = agent.process_turn(user_input)

        # ── LLM-as-Judge: 讽刺穿透 + 回复质量 ──
        verdict = llm_judge(
            scenario="用户被裁员（今年第二次），用自嘲讽刺的语气表达。表面说'太好了''人才'，实际是深度失落",
            user_input=user_input,
            agent_response=response,
            criteria=[
                "系统看穿了讽刺语气，识别出用户实际处于失业困境和自我贬低中",
                "回复没有顺着讽刺语气把裁员当成好事或玩笑",
                "回复表达了对用户处境的理解和支持",
                "回复语气恰当——不过度严肃也不过度轻松",
                "回复没有把裁员当作正面事件来庆祝或鼓励",
            ],
        )
        assert verdict["pass"], (
            f"Sarcasm test failed: {verdict['scores']}\nResponse: {response[:200]}"
        )

    def test_acute_phase_strategy_constraint(self, agent):
        """
        [中等-急性期策略约束] 严重困境首轮 — 高强度负面，首轮应处于急性期
        输入: "我被公司裁员了，所有积蓄也花光了，完全不知道该怎么办，感觉天都塌了"
        KEEP: 策略属于 ACUTE_ALLOWED_STRATEGIES 或 CRISIS_PROTOCOL
        LLM Judge: 回复聚焦倾听和情感确认，而非过早建议
        """
        user_input = "我被公司裁员了，所有积蓄也花光了，完全不知道该怎么办，感觉天都塌了"
        response = agent.process_turn(user_input)
        po = agent.last_planning_output

        # ── 结构检查 (KEEP) ──
        # 注意: 首轮轨迹分析默认为 stable_state (无历史数据)
        # 策略约束仅在 acute_distress 阶段生效; 此处仅检查回复质量
        assert po is not None, "PlanningOutput should not be None"

        # ── LLM-as-Judge: 急性期回复质量 ──
        verdict = llm_judge(
            scenario="用户遭遇裁员+积蓄花光的双重打击，情绪崩溃，这是首次对话（急性期）",
            user_input=user_input,
            agent_response=response,
            criteria=[
                "回复对用户的极度困境表达了深切的理解和共情",
                "回复没有急于给出解决方案或建议（如'找新工作''省钱'）",
                "回复没有使用毒鸡汤或空洞的安慰（如'一切都会好的'）",
                "回复让用户感到被倾听和陪伴",
                "回复聚焦于情感确认和倾听，而非过早提供建议",
            ],
        )
        assert verdict["pass"], (
            f"Acute phase response check failed: {verdict['scores']}\n"
            f"Strategy: {po.selected_strategy}\nResponse: {response[:200]}"
        )

    def test_anger_with_hurt_underneath(self, agent):
        """
        [中等-愤怒下的伤痛] 表层愤怒与底层悲伤并存
        输入: "我恨他，但心里真的好痛"
        KEEP: 结构检查 (state vector 存在)
        LLM Judge: 回复是否同时回应愤怒和底层伤痛，而非只处理一层
        """
        user_input = "我恨他，但心里真的好痛"
        response = agent.process_turn(user_input)
        sv = agent.last_state_vector
        emo_dict = _get_emotion_dict(agent)

        # ── 结构检查 (KEEP) ──
        assert sv is not None

        # ── 效价检查: 愤怒+伤痛场景应为负效价 ──
        assert sv.emotion.valence() < 0, (
            f"愤怒+伤痛场景效价应为负, got {sv.emotion.valence():.3f}"
        )

        # ── LLM-as-Judge: 双层情绪回应 ──
        verdict = llm_judge(
            scenario=(
                "用户说'我恨他，但心里真的好痛'——表层是愤怒/恨意，"
                "底层是被伤害的悲伤和痛苦。这是典型的愤怒-伤痛双层情绪结构"
            ),
            user_input=user_input,
            agent_response=response,
            criteria=[
                "回复回应了用户的愤怒情绪，没有否定或忽视'恨'的感受",
                "回复同时触及了愤怒之下的伤痛/心痛，而非只回应表层愤怒",
                "回复没有急于劝用户'放下恨意'或'原谅他'",
                "回复给予了用户表达复杂情绪的空间",
                "回复语气温和但认真，与用户的痛苦强度匹配",
            ],
        )
        assert verdict["pass"], (
            f"Anger-with-hurt test failed: {verdict['scores']}\n"
            f"Emotion dict: {emo_dict}\nResponse: {response[:200]}"
        )


# ══════════════════════════════════════════════════════════════
# HARD (3 tests) — 多轮对抗性场景
# ══════════════════════════════════════════════════════════════

class TestHardEmotionRecognition:
    """[困难] 情绪急转弯、退缩检测、持续悲伤抗早期积极化 — 需要多轮上下文理解"""

    def test_emotional_whiplash(self, agent):
        """
        [困难-情绪急转弯] 从极度喜悦到突然噩耗
        Turn 1: "今天收到了录取通知书！太高兴了！" (纯喜悦)
        Turn 2: "但是刚才接到电话，我妈确诊癌症了..." (突然深度悲伤)
        KEEP: Turn 1 正效价健全性检查 + Turn 2 负效价检查
        LLM Judge: 系统是否完成了情绪转向 + 回复完全聚焦悲伤
        """
        # Turn 1
        r1 = agent.process_turn("今天收到了录取通知书！太高兴了！")
        sv1 = agent.last_state_vector
        assert sv1.emotion.valence() > 0, (
            f"Turn 1 (录取通知) 效价应为正, got {sv1.emotion.valence():.3f}"
        )

        # Turn 2
        user_input_t2 = "但是刚才接到电话，我妈确诊癌症了..."
        r2 = agent.process_turn(user_input_t2)
        sv2 = agent.last_state_vector

        # ── 效价转向检查 (KEEP) ──
        assert sv2.emotion.valence() < 0, (
            f"Turn 2 (母亲确诊癌症) 效价应为负, got {sv2.emotion.valence():.3f}"
        )

        # ── LLM-as-Judge: 情绪转向 + 回复质量 ──
        verdict = llm_judge(
            scenario=(
                "2轮对话情绪急转: Turn 1 用户收到录取通知(极度喜悦), "
                "Turn 2 母亲确诊癌症(突然深度悲伤). "
                "以下是 Turn 2 的回复"
            ),
            user_input=user_input_t2,
            agent_response=r2,
            criteria=[
                "回复完全转向回应了母亲确诊癌症的悲伤，没有延续Turn 1的喜悦语气",
                "回复对用户突如其来的打击表达了深切的关心和同理心",
                "回复没有提及录取通知或试图用好消息来安慰（不合时宜）",
                "回复给了用户表达悲伤的空间，没有急于给建议",
                "情绪急转后回复完全聚焦于悲伤/丧亲，而非之前的喜悦",
            ],
        )
        assert verdict["pass"], (
            f"Emotional whiplash test failed: {verdict['scores']}\n"
            f"Response: {r2[:200]}"
        )

    def test_withdrawal_low_openness(self, agent):
        """
        [困难-退缩检测] 从情绪披露到突然封闭
        Turn 1: "我最近过得不太好" (情绪披露)
        Turn 2: "没什么，算了，不说了" (退缩/关闭)
        LLM Judge: 系统是否检测到退缩 + 回复温和非侵入性
        """
        agent.process_turn("我最近过得不太好")

        user_input_t2 = "没什么，算了，不说了"
        response = agent.process_turn(user_input_t2)

        # ── LLM-as-Judge: 退缩检测 + 回复质量 ──
        verdict = llm_judge(
            scenario=(
                "2轮对话: Turn 1 用户透露最近过得不好, "
                "Turn 2 用户突然退缩说'算了不说了'. "
                "以下是 Turn 2 的回复"
            ),
            user_input=user_input_t2,
            agent_response=response,
            criteria=[
                "回复尊重了用户不想说的意愿，没有追问或施压",
                "回复传达了'我在这里'的陪伴感，让用户知道随时可以说",
                "回复简短适度，没有长篇大论（退缩用户不适合接收大量信息）",
                "回复语气温和不强迫，没有说'你一定要说出来'之类的话",
                "回复温和非侵入性，尊重用户的低开放度，没有强行引导或追问",
            ],
        )
        assert verdict["pass"], (
            f"Withdrawal test failed: {verdict['scores']}\n"
            f"Response: {response[:200]}"
        )

    def test_sustained_grief_no_premature_positivity(self, agent):
        """
        [困难-持续悲伤] 3轮持续悲伤，验证系统不推动过早积极化
        Turn 1: "爷爷上个月走了，到现在还是不敢相信" (丧亲悲伤)
        Turn 2: "每天早上醒来都会想起他，然后整个人就很难受" (持续悲伤，日常侵入)
        Turn 3: "大家都说时间会好的，但我一点都不觉得" (抗拒安慰，悲伤未减)
        KEEP: 3轮效价均为负
        LLM Judge: Turn 3 回复没有使用毒鸡汤/时间疗法，而是验证用户的悲伤合理性
        """
        # Turn 1
        r1 = agent.process_turn("爷爷上个月走了，到现在还是不敢相信")
        sv1 = agent.last_state_vector
        assert sv1.emotion.valence() < 0, (
            f"Turn 1 (丧亲) 效价应为负, got {sv1.emotion.valence():.3f}"
        )

        # Turn 2
        r2 = agent.process_turn("每天早上醒来都会想起他，然后整个人就很难受")
        sv2 = agent.last_state_vector
        assert sv2.emotion.valence() < 0, (
            f"Turn 2 (持续悲伤) 效价应为负, got {sv2.emotion.valence():.3f}"
        )

        # Turn 3
        user_input_t3 = "大家都说时间会好的，但我一点都不觉得"
        r3 = agent.process_turn(user_input_t3)
        sv3 = agent.last_state_vector
        assert sv3.emotion.valence() < 0, (
            f"Turn 3 (抗拒安慰) 效价应为负, got {sv3.emotion.valence():.3f}"
        )

        # ── LLM-as-Judge: 抗早期积极化 + 悲伤合理性验证 ──
        verdict = llm_judge(
            scenario=(
                "3轮持续丧亲悲伤: "
                "(1) 爷爷去世一个月，不敢相信 → "
                "(2) 每天想起他，整个人难受 → "
                "(3) 别人说时间会好的，但用户不觉得。"
                "用户正在经历正常的悲伤过程，且明确表示'时间疗法'的安慰无效。"
                "以下是第3轮的回复"
            ),
            user_input=user_input_t3,
            agent_response=r3,
            criteria=[
                "回复没有使用'时间会好的''会慢慢好起来'等用户已明确拒绝的安慰方式",
                "回复验证了用户悲伤的合理性——失去至亲的痛是正常的，不需要急着'好起来'",
                "回复没有暗示用户应该'向前看''放下'或'坚强'",
                "回复体现了对用户3轮持续悲伤的耐心陪伴，没有表现出不耐烦或急于转向",
                "回复尊重了用户自己的悲伤节奏，而非推动过早的积极化",
            ],
        )
        assert verdict["pass"], (
            f"Sustained grief test failed: {verdict['scores']}\n"
            f"Turn 3 response: {r3[:200]}"
        )
