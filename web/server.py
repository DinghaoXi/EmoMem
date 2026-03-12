"""
EmoMem Flask 后端 API 服务器
===============================
包装 EmoMemAgent，为 HTML 聊天界面提供 REST 接口。

端点:
  POST /api/chat      — 发送消息并获取 Agent 回复 + 状态
  POST /api/reset     — 重置会话
  GET  /api/state     — 获取当前 Agent 状态（不发送消息）
  GET  /api/status    — 获取系统状态（LLM 模式、模型等）
  GET  /api/config    — 获取当前 LLM 配置信息
  POST /api/config    — 更新 LLM 配置（模型/API Key/Base URL）
  GET  /api/reasoning — 获取推理级别
  POST /api/reasoning — 设置推理级别（off/low/medium/high）
  POST /api/demo      — 运行预定义演示场景

运行方式:
  cd /Users/dinghao/Downloads/emomem
  python -m web.server
"""

from __future__ import annotations

import io
import logging
import os
import sys
import contextlib
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── 添加项目根目录到 sys.path ──
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from src.main import EmoMemAgent
from src.config import EMOTIONS, CRISIS_THRESHOLD, STRATEGIES
from src.llm_provider import MockProvider

# ──────────────────────────────────────────────
# Flask 应用初始化
# ──────────────────────────────────────────────

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app, origins=["http://127.0.0.1:8080", "http://localhost:8080"])  # 限制跨域源

# ── 全局 Agent 实例 (单用户原型) ──
agent: EmoMemAgent = EmoMemAgent()
agent.debug = False  # 关闭控制台诊断输出


# ──────────────────────────────────────────────
# 辅助函数: 从 Agent 提取状态
# ──────────────────────────────────────────────

def _extract_state() -> Dict[str, Any]:
    """
    从 agent.last_state_vector 提取完整状态字典。
    若 last_state_vector 为 None（首轮之前），返回空状态。
    """
    sv = agent.last_state_vector
    if sv is None:
        # 首轮之前，无状态可返回
        return {
            "turn": agent.turn,
            "n_cum": agent.n_cum,
            "emotion": None,
            "intent": None,
            "topic": None,
            "urgency": 0.0,
            "relationship_depth": 0.0,
            "is_crisis": False,
            "recovery": None,
            "strategy": None,
            "planning_intent": None,
            "severity": 0.0,
            "context_assessment": {
                "controllability": None,
                "life_impact": None,
                "explicit_feedback": None,
            },
            "llm_mode": agent._use_llm,
            "memory": {
                "episodic_count": len(agent.memory.episodic_memory),
                "archived_count": len(agent.memory.archived_episodes),
                "baseline_confidence": float(
                    agent.memory.affective_memory.emotion_baseline.confidence
                ) if hasattr(agent.memory, 'affective_memory') and agent.memory.affective_memory and agent.memory.affective_memory.emotion_baseline else 0.5,
                "working_memory_turns": agent.memory.working_memory.turn_count,
            },
            "feedback_history": list(agent.feedback_history),
            "llm_shadow": {
                "llm_urgency": None,
                "llm_recovery_phase": None,
                "llm_planning_intent": None,
            },
            "ablation": agent.ablation,
        }

    # ── 情绪分布: 使用 EMOTIONS 列表与 e 数组对应 ──
    emotion_distribution = {
        EMOTIONS[i]: float(sv.emotion.e[i]) for i in range(len(EMOTIONS))
    }
    dominant_idx = int(sv.emotion.e.argmax())
    # 均匀分布检测: max-min < 0.05 时情绪不可靠，标记为 uncertain
    e_range = float(sv.emotion.e.max() - sv.emotion.e.min())
    if e_range < 0.05:
        dominant_emotion = "uncertain"
        dominant_value = 0.0
    else:
        dominant_emotion = EMOTIONS[dominant_idx]
        dominant_value = float(sv.emotion.e[dominant_idx])

    # ── 意图概率 ──
    intent_probs = {k: float(v) for k, v in sv.intent.probabilities.items()}
    top_intent = sv.intent.top_intent()

    # ── 规划/轨迹信息: 从 agent.last_planning_output 获取当前轮结果 ──
    recovery_info = None
    strategy = None
    planning_intent = None
    severity = 0.0

    po = agent.last_planning_output
    if po is not None:
        # 直接从当前轮的 PlanningOutput 读取 (不再用 _prev_trajectory)
        traj = po.trajectory
        if traj is not None:
            recovery_info = {
                "phase": traj.current_phase.value,
                "phase_duration": traj.phase_duration,
                "direction": traj.direction.value if traj.direction else "unknown",
                "delta_b": float(traj.delta_b),
            }
        strategy = po.selected_strategy
        planning_intent = po.planning_intent
        # 从 appraisal 获取真实 severity
        if po.appraisal is not None:
            severity = float(po.appraisal.severity)

    # 兜底: 从工作记忆获取策略 (首轮 planning_output 可能为 None)
    if strategy is None:
        strategy = agent.memory.working_memory.last_strategy or None
    if strategy is None and agent.memory.episodic_memory:
        strategy = agent.memory.episodic_memory[-1].agent_strategy

    # ── 上下文评估: 优先使用 planning appraisal, 兜底 StateVector ──
    appraisal_ctrl = None
    if po is not None and po.appraisal is not None:
        appraisal_ctrl = float(po.appraisal.controllability)
    context_assessment = {
        "controllability": appraisal_ctrl if appraisal_ctrl is not None else (
            float(sv.controllability) if sv.controllability is not None else None
        ),
        "life_impact": float(sv.life_impact) if sv.life_impact is not None else None,
        "explicit_feedback": float(sv.explicit_feedback) if sv.explicit_feedback is not None else None,
    }

    return {
        "turn": agent.turn,
        "n_cum": agent.n_cum,
        "emotion": {
            "distribution": emotion_distribution,
            "dominant": dominant_emotion,
            "dominant_value": dominant_value,
            "intensity": float(sv.emotion.intensity),
            "confidence": float(sv.emotion.confidence),
            "valence": float(sv.emotion.valence()),
        },
        "intent": {
            "top": top_intent,
            "probabilities": intent_probs,
        },
        "topic": {
            "category": sv.topic_category or "",
            "keywords": list(sv.topic_keywords) if sv.topic_keywords else [],
        },
        "urgency": float(sv.urgency_level),
        "relationship_depth": float(sv.relationship_depth) if sv.relationship_depth is not None else None,
        "is_crisis": sv.urgency_level > CRISIS_THRESHOLD,
        "recovery": recovery_info,
        "strategy": strategy,
        "planning_intent": planning_intent,
        "severity": severity,
        "context_assessment": context_assessment,
        "llm_mode": agent._use_llm,
        "memory": {
            "episodic_count": len(agent.memory.episodic_memory),
            "archived_count": len(agent.memory.archived_episodes),
            "baseline_confidence": float(
                agent.memory.affective_memory.emotion_baseline.confidence
            ) if hasattr(agent.memory, 'affective_memory') and agent.memory.affective_memory and agent.memory.affective_memory.emotion_baseline else 0.5,
            "working_memory_turns": agent.memory.working_memory.turn_count,
        },
        "feedback_history": [float(f) for f in agent.feedback_history],
        # LLM Shadow Mode: LLM-assessed values for comparison with formula-computed values
        "llm_shadow": {
            "llm_urgency": float(sv.llm_urgency) if sv.llm_urgency is not None else None,
            "llm_recovery_phase": sv.recovery_phase if sv.recovery_phase is not None else None,
            "llm_planning_intent": sv.planning_intent if sv.planning_intent is not None else None,
        },
        "ablation": agent.ablation,
    }


def _suppress_stdout(func, *args, **kwargs):
    """
    重定向 stdout 以抑制 process_turn 中的诊断输出。
    即使 agent.debug=False, 某些模块仍可能打印信息。
    """
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        result = func(*args, **kwargs)
    return result


# ──────────────────────────────────────────────
# API 端点
# ──────────────────────────────────────────────

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    POST /api/chat
    请求: {"message": "用户输入文本"}
    响应: {"response": "Agent回复", "state": {...}}
    """
    data = request.get_json(silent=True) or {}
    message = data.get('message', '').strip()

    if not message:
        return jsonify({
            "error": "message 字段不能为空",
        }), 400

    if len(message) > 2000:
        return jsonify({
            "error": "消息过长，请限制在2000字以内",
        }), 400

    # 调用 Agent 处理一轮对话, 抑制 stdout 输出
    try:
        response = _suppress_stdout(agent.process_turn, message)
        return jsonify({
            "response": response,
            "state": _extract_state(),
        })
    except Exception as e:
        logger.exception("Chat processing failed")
        return jsonify({"error": "处理失败，请稍后重试"}), 500


@app.route('/api/reset', methods=['POST'])
def reset():
    """
    POST /api/reset
    重置会话 — 调用 agent.end_session() 并重新初始化。
    响应: {"status": "ok", "message": "会话已重置"}
    """
    global agent

    # 保存当前 LLM 配置 (跨 reset 保留)
    prev_provider = agent.llm_provider
    prev_use_llm = agent._use_llm

    # 抑制 end_session 的打印输出
    _suppress_stdout(agent.end_session)

    # 重新创建 Agent 实例（完全重置）
    agent = EmoMemAgent()
    agent.debug = False

    # 恢复 LLM 配置到 agent 及所有子模块
    if prev_use_llm and prev_provider is not None:
        agent.llm_provider = prev_provider
        agent._use_llm = True
        for module_name in ('perception', 'memory', 'llm'):
            module = getattr(agent, module_name, None)
            if module is not None:
                module.llm_provider = prev_provider

    return jsonify({
        "status": "ok",
        "message": "会话已重置",
    })


@app.route('/api/state', methods=['GET'])
def get_state():
    """
    GET /api/state
    获取当前 Agent 状态（不发送消息）。
    响应: 与 /api/chat 中的 state 对象相同。
    """
    return jsonify(_extract_state())


@app.route('/api/status', methods=['GET'])
def status():
    """
    GET /api/status
    获取系统状态信息（LLM 模式、模型、累计轮次等）。
    """
    return jsonify({
        'llm_mode': agent._use_llm,
        'provider': type(agent.llm_provider).__name__,
        'model': getattr(agent.llm_provider, 'model', 'N/A'),
        'total_turns': agent.n_cum,
        'reasoning_effort': os.environ.get("EMOMEM_REASONING_EFFORT", "off").lower(),
    })


@app.route('/api/config', methods=['GET'])
def get_config():
    """
    GET /api/config
    获取当前 LLM 配置信息。
    """
    provider = agent.llm_provider
    provider_name = type(provider).__name__
    model = getattr(provider, 'model', 'N/A')
    base_url = getattr(provider, 'client', None)
    if base_url:
        base_url = getattr(base_url, 'base_url', None)
        base_url = str(base_url) if base_url else None

    return jsonify({
        'provider': provider_name,
        'model': model,
        'base_url': base_url,
        'llm_mode': agent._use_llm,
        'reasoning_effort': os.environ.get("EMOMEM_REASONING_EFFORT", "off").lower(),
        'available_models': [
            {'id': 'doubao-seed-2-0-lite-260215', 'name': 'Seed 2.0 Lite (默认)', 'provider': 'volcengine'},
            {'id': 'doubao-seed-2-0-pro-260215', 'name': 'Seed 2.0 Pro (最高精度)', 'provider': 'volcengine'},
            {'id': 'doubao-seed-2-0-mini-260215', 'name': 'Seed 2.0 Mini (最快)', 'provider': 'volcengine'},
            {'id': 'doubao-seed-1-8-251228', 'name': 'Seed 1.8 (Agent优化)', 'provider': 'volcengine'},
            {'id': 'doubao-seed-1-6-lite-251015', 'name': 'Seed 1.6 Lite (高性价比)', 'provider': 'volcengine'},
            {'id': 'glm-4-7-251222', 'name': 'GLM-4.7 (200K上下文)', 'provider': 'volcengine'},
        ],
    })


@app.route('/api/config', methods=['POST'])
def update_config():
    """
    POST /api/config
    更新 LLM 配置（模型/API Key/Base URL），重建 Agent。
    请求: {"model": "...", "api_key": "...", "base_url": "...", "models": ["pro", "lite", "mini"]}
    支持多模型轮换: 提供 models 数组时，感知用 models[0]，回复生成轮换使用。
    """
    global agent

    data = request.get_json(silent=True) or {}
    new_model = data.get('model')
    new_api_key = data.get('api_key')
    new_base_url = data.get('base_url')
    new_models = data.get('models')  # 多模型轮换列表

    if not new_model and not new_api_key and not new_base_url and not new_models:
        return jsonify({'error': '至少需要提供 model, api_key, base_url, 或 models 之一'}), 400

    # 如果提供了 models 列表但没有 model，以 models[0] 为主模型
    if new_models and not new_model:
        new_model = new_models[0] if new_models else None

    # 保留当前配置作为默认值
    current_key = new_api_key or os.environ.get('OPENAI_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')
    current_model = new_model or getattr(agent.llm_provider, 'model', None)
    current_base_url = new_base_url
    if current_base_url is None:
        client = getattr(agent.llm_provider, 'client', None)
        if client:
            current_base_url = str(getattr(client, 'base_url', '')) or None

    # 需要模型或 Base URL 时必须有 API Key
    if (new_model or new_base_url) and not current_key:
        return jsonify({
            'error': '配置模型或 Base URL 时需要 API Key（通过请求或环境变量 OPENAI_API_KEY 提供）',
        }), 400

    try:
        # 使用新配置创建 provider（在 end_session 之前，确保失败时不影响旧 agent）
        from src.llm_provider import create_provider

        # 根据 base_url 判断 provider 类型
        provider_type = None
        if current_base_url and 'volces.com' in str(current_base_url):
            provider_type = 'openai'  # Volcengine uses OpenAI-compatible API
        elif current_base_url and 'openai' in str(current_base_url):
            provider_type = 'openai'

        new_provider = create_provider(
            provider_type=provider_type,
            api_key=current_key,
            model=current_model,
            base_url=current_base_url,
            models=new_models,
        )

        # 检查是否静默降级为 MockProvider
        if isinstance(new_provider, MockProvider) and current_key:
            return jsonify({
                'error': 'Provider 创建失败，请检查 API Key 和 Base URL 是否正确',
            }), 400

        # Provider 创建成功，安全地结束旧会话并重建 Agent
        _suppress_stdout(agent.end_session)

        agent = EmoMemAgent()
        agent.debug = False

        # 替换 provider 到 agent 及所有子模块
        use_llm = not isinstance(new_provider, MockProvider)
        agent.llm_provider = new_provider
        agent._use_llm = use_llm
        for module_name in ('perception', 'memory', 'llm'):
            module = getattr(agent, module_name, None)
            if module is not None:
                module.llm_provider = new_provider if use_llm else None

        models_info = getattr(new_provider, 'models', [getattr(new_provider, 'model', 'N/A')])
        return jsonify({
            'status': 'ok',
            'message': f'已切换到 {type(new_provider).__name__}（{current_model}）',
            'provider': type(new_provider).__name__,
            'model': getattr(new_provider, 'model', 'N/A'),
            'models': models_info,
            'llm_mode': agent._use_llm,
        })
    except Exception as e:
        # Provider 创建失败，旧 agent 完全保留
        return jsonify({
            'error': f'切换失败: {str(e)}',
        }), 500


@app.route('/api/demo', methods=['POST'])
def demo():
    """
    POST /api/demo
    运行预定义演示场景。
    请求: {"scenario": "work_stress" | "crisis" | "recovery"}
    响应: 数组形式的每轮结果
    """
    global agent

    data = request.get_json(silent=True) or {}
    scenario = data.get('scenario', 'work_stress')

    # ── 预定义演示场景 ──
    scenarios: Dict[str, List[str]] = {
        "work_stress": [
            "最近工作压力好大，每天都要加班到很晚",
            "老板总是临时加任务，感觉自己快撑不住了",
            "有时候真的很想辞职，但又不敢",
            "谢谢你听我说这些，感觉好一点了",
        ],
        "crisis": [
            "我觉得活着没有意义",
            "没有人在乎我",
            "我不想再继续了",
        ],
        "recovery": [
            "最近心情不太好，和男朋友分手了",
            "虽然很难过，但我知道这可能是对的",
            "今天约了朋友出去吃饭，感觉好多了",
            "我觉得自己在慢慢好起来",
            "谢谢你一直陪着我",
        ],
    }

    messages = scenarios.get(scenario)
    if messages is None:
        return jsonify({
            "error": f"未知场景: {scenario}",
            "available": list(scenarios.keys()),
        }), 400

    # 重置 Agent 后运行演示 (保留 LLM 配置)
    prev_provider = getattr(agent, 'llm_provider', None)
    prev_use_llm = getattr(agent, '_use_llm', False)
    _suppress_stdout(agent.end_session)
    agent = EmoMemAgent()
    agent.debug = False
    if prev_use_llm and prev_provider is not None:
        agent.llm_provider = prev_provider
        agent._use_llm = True
        for module_name in ('perception', 'memory', 'llm'):
            module = getattr(agent, module_name, None)
            if module is not None:
                module.llm_provider = prev_provider

    results = []
    for msg in messages:
        response = _suppress_stdout(agent.process_turn, msg)
        results.append({
            "user_message": msg,
            "response": response,
            "state": _extract_state(),
        })

    return jsonify(results)


@app.route('/api/model', methods=['GET'])
def get_model():
    """
    GET /api/model
    获取当前模型信息和可用模型列表。
    """
    provider = agent.llm_provider
    current_model = getattr(provider, 'model', 'mock')
    models_list = getattr(provider, 'models', [current_model])

    return jsonify({
        'model': current_model,
        'models': models_list,
        'provider': type(provider).__name__,
        'available': [
            {'id': 'doubao-seed-2-0-lite-260215', 'name': 'Seed 2.0 Lite (默认)'},
            {'id': 'doubao-seed-2-0-pro-260215', 'name': 'Seed 2.0 Pro (最高精度)'},
            {'id': 'doubao-seed-1-8-251228', 'name': 'Seed 1.8 (Agent优化)'},
            {'id': 'doubao-seed-1-6-lite-251015', 'name': 'Seed 1.6 Lite (高性价比)'},
            {'id': 'glm-4-7-251222', 'name': 'GLM-4.7 (200K上下文)'},
        ],
    })


@app.route('/api/model', methods=['POST'])
def set_model():
    """
    POST /api/model
    切换运行模型，自动重置会话。
    请求: {"model": "doubao-seed-1-8-251228"}
    """
    global agent

    data = request.get_json(silent=True) or {}
    new_model = data.get('model')

    if not new_model:
        return jsonify({'error': 'model 字段不能为空'}), 400

    valid_models = {
        'doubao-seed-2-0-lite-260215', 'doubao-seed-2-0-pro-260215',
        'doubao-seed-2-0-mini-260215',
        'doubao-seed-1-8-251228', 'doubao-seed-1-6-lite-251015',
        'glm-4-7-251222',
    }
    if new_model not in valid_models:
        return jsonify({'error': f'未知模型: {new_model}', 'available': list(valid_models)}), 400

    try:
        from src.llm_provider import create_provider, OpenAIProvider

        # 获取当前 API key 和 base_url
        old_provider = agent.llm_provider
        current_key = getattr(old_provider, '_current_key', None)
        current_base_url = getattr(old_provider, '_base_url', None)
        if current_base_url is None:
            client = getattr(old_provider, 'client', None)
            if client:
                current_base_url = str(getattr(client, 'base_url', '')) or None

        # 使用硬编码 key 兜底
        if not current_key:
            current_key = OpenAIProvider._ARK_API_KEYS[0]

        new_provider = create_provider(
            provider_type='openai',
            api_key=current_key,
            model=new_model,
            base_url=current_base_url,
        )

        # 重置会话
        prev_ablation = dict(agent.ablation)
        _suppress_stdout(agent.end_session)

        agent = EmoMemAgent()
        agent.debug = False
        agent.ablation = prev_ablation

        # 注入新 provider
        use_llm = not isinstance(new_provider, MockProvider)
        agent.llm_provider = new_provider
        agent._use_llm = use_llm
        for module_name in ('perception', 'memory', 'llm'):
            module = getattr(agent, module_name, None)
            if module is not None:
                module.llm_provider = new_provider if use_llm else None

        return jsonify({
            'status': 'ok',
            'model': new_model,
            'message': f'已切换到 {new_model}，会话已重置',
        })
    except Exception as e:
        logger.exception("Model switch failed")
        return jsonify({'error': f'切换失败: {str(e)}'}), 500


@app.route('/api/ablation', methods=['GET'])
def get_ablation():
    """GET /api/ablation — 获取消融实验开关状态"""
    return jsonify(agent.ablation)


@app.route('/api/ablation', methods=['POST'])
def set_ablation():
    """
    POST /api/ablation — 设置消融实验开关
    请求: {"affective_memory": true/false, "mar": true/false, "thompson_sampling": true/false}
    """
    global agent
    data = request.get_json(silent=True) or {}

    changed = []
    for key in ("affective_memory", "mar", "thompson_sampling"):
        if key in data:
            old_val = agent.ablation.get(key, True)
            new_val = bool(data[key])
            if old_val != new_val:
                agent.ablation[key] = new_val
                changed.append(f"{key}: {old_val} → {new_val}")

    # Reset session when ablation changes (clean comparison)
    if changed:
        _suppress_stdout(agent.end_session)
        # Preserve LLM config
        prev_provider = agent.llm_provider
        prev_use_llm = agent._use_llm
        prev_ablation = dict(agent.ablation)

        agent = EmoMemAgent()
        agent.debug = False
        agent.ablation = prev_ablation

        if prev_use_llm and prev_provider is not None:
            agent.llm_provider = prev_provider
            agent._use_llm = True
            for module_name in ('perception', 'memory', 'llm'):
                module = getattr(agent, module_name, None)
                if module is not None:
                    module.llm_provider = prev_provider

    return jsonify({
        "status": "ok",
        "ablation": agent.ablation,
        "changed": changed,
        "message": "会话已重置以应用消融设置" if changed else "无变更",
    })


@app.route('/api/reasoning', methods=['GET'])
def get_reasoning():
    """GET /api/reasoning — 获取当前推理级别"""
    effort = os.environ.get("EMOMEM_REASONING_EFFORT", "off").lower()
    model = getattr(agent.llm_provider, 'model', '')
    supports = 'glm' not in model.lower() and 'seed' in model.lower() if model else False
    return jsonify({
        'effort': effort,
        'supports_reasoning': supports,
        'available': ['off', 'low', 'medium', 'high'],
    })


@app.route('/api/reasoning', methods=['POST'])
def set_reasoning():
    """POST /api/reasoning — 设置推理级别"""
    data = request.get_json(silent=True) or {}
    effort = data.get('effort', '').lower()
    if effort not in ('off', 'low', 'medium', 'high'):
        return jsonify({'error': f'无效的推理级别: {effort}，可选: off, low, medium, high'}), 400
    os.environ["EMOMEM_REASONING_EFFORT"] = effort
    return jsonify({
        'status': 'ok',
        'effort': effort,
        'message': f'推理级别已设为 {effort}',
    })


# ──────────────────────────────────────────────
# 静态文件服务
# ──────────────────────────────────────────────

@app.route('/')
def serve_index():
    """提供 web/ 目录下的 index.html"""
    return send_from_directory('.', 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    """提供 web/ 目录下的其他静态文件"""
    # 安全检查: 禁止路径穿越
    if '..' in filename or filename.startswith('/'):
        return jsonify({"error": "forbidden"}), 403
    return send_from_directory('.', filename)


# ──────────────────────────────────────────────
# 入口点
# ──────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 50)
    print("  EmoMem API Server")
    print("  http://127.0.0.1:8080")
    print("=" * 50)
    app.run(host='127.0.0.1', port=8080, debug=False)
