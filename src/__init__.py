"""
EmoMem: Memory-Augmented AI Emotional Companion Agent
======================================================
PPAM (Perception-Planning-Action-Memory) 架构实现

模块结构:
  perception.py  — §2 感知: 用户输入 → StateVector (情绪/意图/话题/紧急度)
  planning.py    — §4 规划: 轨迹分析 → 情境评估 → Thompson Sampling 策略选择
  action.py      — §5 行动: 策略 → 自然语言回复 (LLM/模板双路径)
  memory.py      — §3 记忆: 四层架构 (Working/Episodic/Semantic/Affective) + MAR 检索
  adaptation.py  — 自适应参数框架: 安全约束下的用户个性化参数调整
  config.py      — 全局超参数 (标注对应 emomem.md 章节号)
  models.py      — 所有数据结构 (dataclasses)
  llm_provider.py — 统一 LLM 接口 (Anthropic/OpenAI/Mock)
  mock_llm.py    — 模板回复生成 (LLM 不可用时的降级方案)

运行: python -m src.main
"""
