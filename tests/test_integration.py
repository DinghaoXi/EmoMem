"""
EmoMem 集成测试 — PPAM 管线端到端验证
======================================
Integration tests for the full PPAM pipeline, organized by difficulty:

  [简单] 1 test  — single-turn pipeline validation
  [中等] 3 tests — multi-turn memory accumulation, session reset, empty input
  [困难] 3 tests — emotion arc tracking, diverse response quality, ablation toggle

断言策略:
  - 结构检查 (turn counter, state vector, 去重) -> 数值断言 (KEEP)
  - 回复质量 (长度、多样性、语境匹配) -> LLM-as-Judge 替代 (REPLACE)

All tests require a live LLM backend (ARK_API_KEY injected by conftest.py).
"""

from __future__ import annotations

import re
import pytest

from src.main import EmoMemAgent
from tests.llm_judge import llm_judge


# ──────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────

def _has_chinese(text: str) -> bool:
    """Return True if *text* contains at least one CJK Unified Ideograph."""
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _is_structurally_valid(text: str) -> bool:
    """
    验证回复的结构有效性 (非语义质量):
      - 非空字符串
      - 包含中文字符
      - 无乱码输出 (单字符重复 8+ 次)
    """
    if not isinstance(text, str) or not text.strip():
        return False
    if not _has_chinese(text):
        return False
    if re.search(r"(.)\1{7,}", text):
        return False
    return True


# ══════════════════════════════════════════════
# EASY — basic pipeline validation
# ══════════════════════════════════════════════

class TestEasyPipeline:
    """[简单] 单轮管线基础验证"""

    def test_single_turn_pipeline(self):
        """
        单轮中性输入 — 验证管线基本功能
        KEEP: 结构检查 (类型、中文、去乱码、turn counter、state vector)
        LLM Judge: 回复是否语境匹配
        """
        user_input = "今天天气不错，心情还行"
        agent = EmoMemAgent()
        response = agent.process_turn(user_input)

        # ── 结构检查 (KEEP) ──
        assert isinstance(response, str)
        assert _has_chinese(response), "Response should contain Chinese characters"
        assert not re.search(r"(.)\1{7,}", response), "Garbled output detected"
        assert agent.turn == 1
        sv = agent.last_state_vector
        assert sv is not None
        assert sv.emotion is not None
        assert sv.intent is not None

        # ── LLM-as-Judge: 回复语境匹配 ──
        verdict = llm_judge(
            scenario="用户随意聊天，说天气不错心情还行，情绪中性偏正",
            user_input=user_input,
            agent_response=response,
            criteria=[
                "回复是自然的中文对话，语气与用户的轻松语境匹配",
                "回复没有过度解读为负面情绪或危机",
                "回复让人感觉像是在和一个温和的朋友聊天",
            ],
        )
        assert verdict["pass"], (
            f"Single turn response check failed: {verdict['scores']}"
        )


# ══════════════════════════════════════════════
# MEDIUM — multi-turn, state management
# ══════════════════════════════════════════════

class TestMediumMultiTurn:
    """[中等] 多轮对话状态管理验证"""

    def test_memory_accumulation_3turns(self):
        """
        三轮对话 (压力->自我否定->好消息) — 验证状态积累和回复变化
        KEEP: turn counter, working memory, 回复去重
        LLM Judge: 三轮回复是否语境匹配且体现情绪连续性
        """
        agent = EmoMemAgent()
        inputs = [
            "最近工作压力好大",
            "老板总是批评我，觉得自己什么都做不好",
            "今天终于有个好消息，项目通过评审了",
        ]

        responses = []
        for text in inputs:
            responses.append(agent.process_turn(text))

        r1, r2, r3 = responses

        # ── 结构检查 (KEEP) ──
        assert agent.turn == 3
        wm = agent.memory.working_memory
        assert wm.prev_emotion is not None, "prev_emotion should be set after 3 turns"

        # 结构有效性
        for idx, resp in enumerate(responses, start=1):
            assert _is_structurally_valid(resp), (
                f"Turn {idx} response failed structural check: {resp!r:.120}"
            )

        # 回复不应完全相同 (管线 bug 检测)
        assert r1 != r2 and r2 != r3 and r1 != r3, "Responses should not be identical"

        # ── LLM-as-Judge: 三轮回复语境匹配 ──
        verdict = llm_judge(
            scenario=(
                "3轮对话情绪弧线: "
                "(1) 工作压力大 -> (2) 被老板批评，自我否定 -> (3) 好消息，项目通过评审。"
                "情绪从负面逐步到正面转折"
            ),
            user_input=" -> ".join(inputs),
            agent_response=f"Turn1: {r1[:100]}\nTurn2: {r2[:100]}\nTurn3: {r3[:100]}",
            criteria=[
                "Turn1 回复对工作压力表示理解和支持",
                "Turn2 回复对被批评和自我否定表达共情，没有说教",
                "Turn3 回复对好消息表达了开心或认可，体现了情绪转变",
                "三轮回复之间有连续性，不像三个独立对话",
            ],
        )
        assert verdict["pass"], (
            f"3-turn memory test failed: {verdict['scores']}"
        )

    def test_session_reset(self):
        """
        会话重置后状态隔离验证
        KEEP: turn counter reset, working memory reset
        LLM Judge: 新会话回复是否正常 (不强制检查是否引用上一会话)
        """
        agent = EmoMemAgent()

        # --- First session ---
        r1 = agent.process_turn("我今天很难过")
        assert agent.turn == 1
        assert _is_structurally_valid(r1)

        # End session
        agent.end_session()
        assert agent.turn == 0, "turn should reset to 0 after end_session()"

        # Working memory should be freshly reset
        wm = agent.memory.working_memory
        assert wm.turn_count == 0, "working memory turn_count should be 0 after reset"
        assert wm.prev_emotion is None, "prev_emotion should be None after reset"
        assert wm.crisis_cooldown_counter == 0, "crisis_cooldown should be 0 after reset"

        # --- Second session ---
        user_input_s2 = "你好"
        r2 = agent.process_turn(user_input_s2)
        assert agent.turn == 1, "turn should be 1 after first message in new session"
        assert _is_structurally_valid(r2)

        # Cumulative counter should reflect both sessions
        assert agent.n_cum == 2, "n_cum should be 2 (1 from each session)"

        # ── LLM-as-Judge: 新会话回复质量 ──
        verdict = llm_judge(
            scenario=(
                "用户在新会话中简单打招呼'你好'。"
                "注意：系统有跨会话情景记忆，上一会话用户曾说'我今天很难过'，"
                "因此回复中可能带有对上次情绪的关心，这是系统设计的正常行为"
            ),
            user_input=user_input_s2,
            agent_response=r2,
            criteria=[
                "回复包含了打招呼/问候的元素，或对用户表达了关心",
                "回复语气友好自然",
            ],
        )
        assert verdict["pass"], (
            f"Session reset response check failed: {verdict['scores']}"
        )

    def test_empty_input_handling(self):
        """
        空输入/纯空白输入 — 验证不崩溃并返回温和提示
        KEEP: 返回值类型、包含中文、turn counter 不递增 (空输入不计轮)
        """
        agent = EmoMemAgent()

        # ── 空字符串 ──
        response_empty = agent.process_turn("")
        assert isinstance(response_empty, str)
        assert _has_chinese(response_empty), "Empty input should return Chinese prompt"
        assert agent.turn == 0, "Empty input should not increment turn counter"

        # ── 纯空白 ──
        response_ws = agent.process_turn("   ")
        assert isinstance(response_ws, str)
        assert _has_chinese(response_ws), "Whitespace input should return Chinese prompt"
        assert agent.turn == 0, "Whitespace input should not increment turn counter"

        # ── 空输入后正常输入应正常工作 ──
        response_normal = agent.process_turn("你好呀")
        assert _is_structurally_valid(response_normal)
        assert agent.turn == 1, "Normal input after empty should set turn to 1"


# ══════════════════════════════════════════════
# HARD — complex multi-turn validation
# ══════════════════════════════════════════════

class TestHardIntegration:
    """[困难] 情绪弧线、回复多样性、消融开关验证"""

    def test_emotion_arc_direction(self):
        """
        跟踪情绪弧线: 悲伤->缓和->回落
        KEEP: Turn 1/3 负效价 (明确的负面刺激)
        LLM Judge: 三轮回复的情绪匹配度
        """
        agent = EmoMemAgent()
        inputs = [
            "最近被分手了，心里空落落的",
            "朋友们都在安慰我，感觉好了一些",
            "今天去了以前一起常去的咖啡店，又想哭了",
        ]

        responses = []
        svs = []
        for text in inputs:
            responses.append(agent.process_turn(text))
            svs.append(agent.last_state_vector)

        r1, r2, r3 = responses
        v1 = svs[0].emotion.valence()
        v3 = svs[2].emotion.valence()

        # ── 方向性检查 (KEEP — 明确的负面刺激) ──
        assert v1 < 0, f"Turn 1 (分手) valence should be negative, got {v1:.3f}"
        assert v3 < 0, f"Turn 3 (回落) valence should be negative, got {v3:.3f}"

        # ── LLM-as-Judge: 三轮回复情绪匹配 ──
        verdict = llm_judge(
            scenario=(
                "分手情绪弧线: (1) 刚分手，心里空落落 -> "
                "(2) 朋友安慰，感觉好了些 -> (3) 去了以前一起去的咖啡店，又想哭了"
            ),
            user_input=" -> ".join(inputs),
            agent_response=f"Turn1: {r1[:100]}\nTurn2: {r2[:100]}\nTurn3: {r3[:100]}",
            criteria=[
                "Turn1 对分手的失落表达了共情和陪伴",
                "Turn2 认可了用户感觉好了一些，同时保持温和支持",
                "Turn3 对情绪回落（去咖啡店触发回忆）表达了理解，没有说'你不是好了吗'",
                "三轮回复体现了对用户情绪波动轨迹的持续关注",
            ],
        )
        assert verdict["pass"], (
            f"Emotion arc test failed: {verdict['scores']}"
        )

    def test_response_quality_diverse(self):
        """
        四种不同情绪场景 — 验证回复多样性和语境匹配
        KEEP: 结构有效性 (中文、去乱码), 每个场景用 fresh agent
        LLM Judge: 每个场景的语境匹配度 + 整体多样性
        """
        scenarios = [
            ("今天考试考砸了", "考试失利，学业失望"),
            ("和好朋友吵架了", "友谊冲突，人际矛盾"),
            ("终于完成了毕业论文！", "学业成就，喜悦释放"),
            ("感觉生活好无聊，没什么意思", "无聊倦怠，生活缺乏意义感"),
        ]

        responses = []
        for text, _ in scenarios:
            a = EmoMemAgent()  # fresh agent per scenario
            responses.append(a.process_turn(text))

        # ── 结构检查 (KEEP) ──
        for i, resp in enumerate(responses):
            assert _is_structurally_valid(resp), (
                f"Scenario {i} failed structural check: {resp!r:.120}"
            )

        # ── LLM-as-Judge: 语境匹配 + 多样性 ──
        response_block = "\n".join(
            f"场景{i+1}({desc}): {resp[:120]}"
            for i, ((_, desc), resp) in enumerate(zip(scenarios, responses))
        )
        verdict = llm_judge(
            scenario=(
                "四种不同情绪场景的回复: "
                "(1) 考试考砸 (2) 和朋友吵架 (3) 完成毕业论文 (4) 生活无聊"
            ),
            user_input=" / ".join(text for text, _ in scenarios),
            agent_response=response_block,
            criteria=[
                "场景1(考试失利)的回复对学业挫折表达了理解",
                "场景2(朋友吵架)的回复对人际冲突表达了关心",
                "场景3(完成论文)的回复与用户分享了喜悦，语气积极",
                "场景4(无聊倦怠)的回复对生活无意义感有所回应",
                "四个回复之间有明显的语气和内容差异，不像模板化套话",
            ],
        )
        assert verdict["pass"], (
            f"Diverse response quality check failed: {verdict['scores']}"
        )

    def test_ablation_thompson_sampling_disabled(self):
        """
        消融实验: 关闭 Thompson Sampling — 验证系统仍能正常运行
        KEEP: 结构有效性, ablation 状态, 策略不为 None/空
        LLM Judge: 回复质量不因消融而严重退化
        """
        agent = EmoMemAgent()

        # ── 关闭 Thompson Sampling ──
        agent.ablation["thompson_sampling"] = False
        assert agent.ablation["thompson_sampling"] is False, (
            "thompson_sampling ablation toggle should be False"
        )

        user_input = "最近总是失眠，白天也提不起精神"
        response = agent.process_turn(user_input)

        # ── 结构检查 (KEEP) ──
        assert _is_structurally_valid(response), (
            f"Ablated response failed structural check: {response!r:.120}"
        )
        assert agent.turn == 1

        # 规划输出应存在且策略非空 (规则映射 fallback 应生效)
        po = agent.last_planning_output
        assert po is not None, "PlanningOutput should exist even with TS disabled"
        assert po.selected_strategy, (
            f"Strategy should not be empty with TS disabled, got: {po.selected_strategy!r}"
        )

        # StateVector 应正常填充
        sv = agent.last_state_vector
        assert sv is not None
        assert sv.emotion is not None
        assert sv.intent is not None

        # ── LLM-as-Judge: 回复质量 ──
        verdict = llm_judge(
            scenario=(
                "用户表达失眠和精力不振。系统的 Thompson Sampling 模块已关闭，"
                "使用规则映射 fallback 选择策略。验证回复质量不会严重退化"
            ),
            user_input=user_input,
            agent_response=response,
            criteria=[
                "回复对用户的失眠和疲惫表达了关心和理解",
                "回复是自然流畅的中文，没有出现乱码或无意义内容",
                "回复语气温和，与用户的低落状态匹配",
            ],
        )
        assert verdict["pass"], (
            f"Ablation response quality check failed: {verdict['scores']}\n"
            f"Strategy used: {po.selected_strategy}\nResponse: {response[:200]}"
        )
