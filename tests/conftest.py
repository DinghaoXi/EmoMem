"""
EmoMem 测试公共配置 (Shared Test Configuration)
================================================
- 动态 sys.path 设置
- 公共 helper 函数
"""

import sys
import os
import pytest
from pathlib import Path

# 动态添加项目根目录到 sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ── LLM 测试配置 ──
# 可用模型:
#   - doubao-seed-1-8-251228      (Seed 1.8, Agent优化, 多模态, 智能上下文管理)
#   - doubao-seed-1-6-lite-251015 (Seed 1.6, 多模态, 256K ctx, 最高性价比)
#   - doubao-seed-2-0-pro-260215  (Seed 2.0 Pro, 精度最高)
#   - doubao-seed-2-0-lite-260215 (Seed 2.0 Lite, 当前默认)
#   - glm-4-7-251222              (GLM-4.7, 200k ctx, 较慢)
# 切换模型: export EMOMEM_TEST_MODEL=doubao-seed-1-8-251228
LLM_API_KEY = os.environ.get("ARK_API_KEY", "59aea33e-3012-4bb4-bb9b-2a5ad8565678")
LLM_BASE_URL = os.environ.get(
    "ARK_BASE_URL",
    "https://ark.cn-beijing.volces.com/api/v3/",
)
LLM_MODEL = os.environ.get("EMOMEM_TEST_MODEL", "doubao-seed-2-0-pro-260215")

# ── 自动注入环境变量，确保 create_provider() 能检测到 API key ──
os.environ.setdefault("ARK_API_KEY", LLM_API_KEY)
os.environ.setdefault("EMOMEM_TEST_MODEL", LLM_MODEL)
