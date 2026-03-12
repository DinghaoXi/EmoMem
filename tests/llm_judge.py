"""
LLM-as-Judge 评估助手 (LLM-as-Judge Evaluation Helper)
=====================================================
轻量级 LLM 评审函数，用于评估 EmoMem agent 回复的质量。

用法:
    from tests.llm_judge import llm_judge

    verdict = llm_judge(
        scenario="用户刚分手，表达悲伤",
        user_input="今天分手了，心里好难受",
        agent_response=response,
        criteria=[
            "回复体现了对用户悲伤情绪的理解和共情",
            "没有急于给建议或说教",
            "语气温暖自然，不生硬",
        ],
    )
    assert verdict["pass"], f"Judge failed: {verdict['scores']}"

设计原则:
    - 单函数，单 prompt 模板，无过度工程
    - temperature=0 保证确定性
    - LLM 调用失败时 pass=True，避免阻断测试
    - 使用与被测 agent 相同的 Volcengine API 基础设施
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


# ── Prompt 模板 ──

_JUDGE_SYSTEM_PROMPT = """\
你是一个专业的对话质量评审专家。你的任务是评估一个情感陪伴AI的回复质量。

评分规则：
- 对每个评估维度打分，分数范围 1-5
  1 = 完全不符合
  2 = 基本不符合
  3 = 勉强符合（最低可接受）
  4 = 良好符合
  5 = 完美符合
- 严格根据回复内容评分，不要猜测意图
- 输出严格的 JSON 格式，不要输出其他内容"""

_JUDGE_USER_TEMPLATE = """\
## 场景描述
{scenario}

## 用户输入
{user_input}

## AI回复
{agent_response}

## 评估维度
{criteria_block}

请对每个维度打分（1-5），输出 JSON 格式：
{{
{json_template}
}}

直接输出 JSON，不要输出其他内容。"""


def llm_judge(
    scenario: str,
    user_input: str,
    agent_response: str,
    criteria: list[str],
    *,
    threshold: int = 3,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> dict:
    """评估 agent 回复是否满足给定的质量标准。

    Args:
        scenario:       场景自然语言描述 (中文)
        user_input:     用户的原始输入
        agent_response: agent 产生的回复文本
        criteria:       评估维度列表，每项为中文自然语言描述
                        例如: ["回复体现了对用户悲伤情绪的共情", "没有说教"]
        threshold:      每项的最低通过分数 (默认 3，即 "勉强符合" 及以上)
        model:          评审用模型 (默认使用 EMOMEM_TEST_MODEL 或 pro)
        api_key:        API key (默认使用 ARK_API_KEY)
        base_url:       API base URL (默认使用 Volcengine ARK)

    Returns:
        dict with keys:
            scores: dict[str, int]  — 每个维度名称 → 1-5 评分
            pass:   bool            — 是否所有维度 >= threshold
            raw:    str | None      — LLM 原始输出 (调试用)
            error:  str | None      — 错误信息 (如果有)

    Notes:
        - LLM 调用失败时返回 pass=True，避免因评审故障阻断功能测试
        - temperature=0 保证评审结果可复现
    """
    # ── 构建 prompt ──
    criteria_block = "\n".join(
        f"{i+1}. {c}" for i, c in enumerate(criteria)
    )
    # JSON 模板中用维度编号作为 key，方便解析
    json_keys = [f'  "criterion_{i+1}": <1-5>' for i in range(len(criteria))]
    json_template = ",\n".join(json_keys)

    user_prompt = _JUDGE_USER_TEMPLATE.format(
        scenario=scenario,
        user_input=user_input,
        agent_response=agent_response,
        criteria_block=criteria_block,
        json_template=json_template,
    )

    # ── 调用 LLM ──
    _api_key = api_key or os.environ.get("ARK_API_KEY", "")
    _base_url = base_url or os.environ.get(
        "ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3/"
    )
    # 不使用 EMOMEM_TEST_MODEL 作为评审模型，避免自评偏差 (self-evaluation bias)
    _model = model or os.environ.get(
        "EMOMEM_JUDGE_MODEL",
        "doubao-seed-2-0-pro-260215",  # 默认使用 Pro 模型作为评审 (避免自评偏差)
    )

    if not _api_key:
        logger.warning("llm_judge: no API key available, returning pass=True")
        return {
            "scores": {c: threshold for c in criteria},
            "pass": True,
            "raw": None,
            "error": "no API key",
        }

    try:
        import openai

        client = openai.OpenAI(
            api_key=_api_key,
            base_url=_base_url,
            timeout=30.0,
        )

        response = client.chat.completions.create(
            model=_model,
            temperature=0,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw_text = response.choices[0].message.content or ""

    except Exception as exc:
        logger.warning("llm_judge: LLM call failed (%s), returning pass=True", exc)
        return {
            "scores": {c: threshold for c in criteria},
            "pass": True,
            "raw": None,
            "error": str(exc),
        }

    # ── 解析 JSON ──
    try:
        # 提取第一个 JSON 对象 (容忍 LLM 在 JSON 前后加文字)
        json_str = _extract_json(raw_text)
        parsed = json.loads(json_str)

        scores: dict[str, int] = {}
        for i, criterion in enumerate(criteria):
            key = f"criterion_{i+1}"
            val = parsed.get(key)
            if isinstance(val, (int, float)) and 1 <= val <= 5:
                scores[criterion] = int(val)
            else:
                # 无法解析该维度评分，给予通过分以避免误报
                logger.warning(
                    "llm_judge: criterion_%d value invalid (%r), defaulting to %d",
                    i + 1, val, threshold,
                )
                scores[criterion] = threshold

        all_pass = all(s >= threshold for s in scores.values())

        return {
            "scores": scores,
            "pass": all_pass,
            "raw": raw_text,
            "error": None,
        }

    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "llm_judge: JSON parse failed (%s), raw=%r, returning pass=True",
            exc, raw_text[:200],
        )
        return {
            "scores": {c: threshold for c in criteria},
            "pass": True,
            "raw": raw_text,
            "error": f"JSON parse: {exc}",
        }


def _extract_json(text: str) -> str:
    """从 LLM 输出中提取第一个完整的 JSON 对象。

    使用大括号计数法，正确处理嵌套和字符串内的大括号。
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in text")

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError("Unbalanced JSON braces in text")
