# EmoMem — 记忆增强的 AI 情感陪伴 Agent

> 🌐 [English Version →](README.md)

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/后端-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/协议-MIT-green.svg)](LICENSE)

![EmoMem 界面](docs/screenshot.png)

---

## 项目简介

EmoMem 是一个**记忆增强的 AI 情感陪伴 Agent** 研究原型。与把每一轮对话视为独立事件的传统聊天机器人不同，EmoMem 能够跨会话持续建构对用户情感世界的结构化理解——记住过去的情绪事件，学习哪种支持策略对这个特定用户最有效，并追踪用户长期的情绪恢复轨迹。

系统采用**感知 → 规划 → 行动 → 记忆（PPAM）**四模块架构，其中记忆不是被动的日志仓库，而是驱动其他三个模块运转的**核心引擎**。

---

## 三大核心创新

### 🔬 创新一 — 情感记忆层（Affective Memory Tier）

绝大多数对话系统在会话结束后就"忘掉"用户。EmoMem 引入专属的**情感记忆层**，在跨会话持久化存储以下内容：

| 组件 | 记录内容 |
|---|---|
| **情绪基线** | 用户在每个情绪维度上的典型水平（均值 + 方差） |
| **触发-情绪关联图** | 哪类事件（工作、人际关系、健康…）会引发该用户的哪种情绪 |
| **恢复曲线** | 该用户从负面情绪事件中恢复所需的典型时长与轨迹 |
| **策略有效性** | 过去哪些支持策略（情感验证、建议、认知重构…）对该用户有效 |

这使 EmoMem 从"无状态应答者"转变为**真正了解用户**的陪伴 Agent，并能据此自适应调整行为。

---

### 🔬 创新二 — 心境自适应检索（MAR, Mood-Adaptive Retrieval）

从记忆中检索历史信息时，朴素系统通常按时间近度或语义相似度检索。EmoMem 使用 **MAR**，根据当前情绪状态和规划目标**动态调整检索权重**：

- **目标为情感验证时** → 检索用户曾经有相似情绪的历史片段（心境一致检索，基于 Bower 1981 心境依存记忆理论）
- **目标为助力恢复时** → 检索用户曾成功克服类似困境的历史片段（心境不一致 / 积极唤醒检索）
- **目标为危机评估时** → 检索偏离情感基线的行为模式异常记录

检索权重函数：

$$\Phi = \alpha \cdot \text{sim}_{\text{情绪}} + \beta \cdot \text{sim}_{\text{语义}} + \gamma \cdot \text{时近度} + \delta \cdot \text{重要性}$$

其中 $(\alpha, \beta, \gamma, \delta)$ 由当前规划目标动态决定，而非固定参数。

---

### 🔬 创新三 — 上下文 Thompson 采样策略学习

EmoMem 通过**上下文多臂老虎机（Contextual Bandit）**框架，学习对特定用户最有效的支持策略：

- **动作空间**：9 种支持策略，源自 Hill（2009）帮助技能理论——包括积极倾听、情感验证、认知重构、问题解决、心理教育等
- **上下文特征**：情绪类型、紧急度、恢复阶段、关系深度、用户意图
- **学习机制**：Thompson 采样 + 每策略独立 Beta 分布 + 上下文聚类分层，在探索（尝试新策略）与利用（沿用有效策略）之间动态平衡

这使 EmoMem 从"泛化共情"进化为**个性化支持**——同一用户在第 1 天和第 30 天说"我感觉迷茫"，会因 Agent 已积累对其个人偏好的学习而获得本质不同的回应。

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                       EmoMem PPAM 架构                           │
│                                                                  │
│  用户输入                                                         │
│      │                                                           │
│      ▼                                                           │
│  ┌──────────┐   状态向量      ┌──────────────────────────┐      │
│  │  感知    │ ─────────────→  │          记忆            │      │
│  │  模块    │   记忆查询      │  ┌────────────────────┐  │      │
│  │ • 情绪  │ ─────────────→  │  │ 工作记忆           │  │      │
│  │   雷达  │                 │  │ 情景记忆           │  │      │
│  │ • 意图  │                 │  │ 情感记忆层 ★       │  │      │
│  │ • 紧急度│                 │  │ 语义记忆           │  │      │
│  └──────────┘                 │  └────────────────────┘  │      │
│                               └────────┬─────────────────┘      │
│                           检索上下文   │  锚定素材               │
│                                        ▼                         │
│                               ┌──────────────┐                  │
│                               │   规划模块   │                  │
│                               │ • 轨迹建模   │                  │
│                               │ • Thompson   │                  │
│                               │   采样 ★    │                  │
│                               └──────┬───────┘                  │
│                               策略计划│                          │
│                                        ▼                         │
│                               ┌──────────────┐                  │
│                               │   行动模块   │ → 回复           │
│                               └──────┬───────┘                  │
│                           情景记录   │（异步）                   │
│                                        ▼                         │
│                           反馈信号 → 记忆更新                    │
└─────────────────────────────────────────────────────────────────┘
```

**★** = EmoMem 独有的核心创新

---

## 情绪模型

EmoMem 使用 **Plutchik 8 维情绪模型**，而非简单的正/负分类：

```
喜悦(joy) · 信任(trust) · 恐惧(fear) · 惊讶(surprise)
悲伤(sadness) · 厌恶(disgust) · 愤怒(anger) · 期待(anticipation)
```

每一轮对话产生一个 `EmotionVector`：
- **e(t)** — 8 维情绪的概率分布（和为 1，支持混合情绪）
- **ι(t)** — 情绪强度 [0, 1]
- **κ(t)** — 识别确定度 [0, 1]——偏低时 Agent 会主动提问澄清，而不是主观断定用户的感受

标量**效价（valence）**由加权求和计算，用于追踪用户情绪随时间的变化轨迹。

---

## 情绪恢复轨迹

EmoMem 将用户的情绪变化建模为四个阶段：

```
谷底期 → 恢复期 → 巩固期 → 稳定期
```

规划模块根据当前阶段选择不同策略——谷底期优先情感验证和安全保障；巩固期着重强化积极应对模式。

---

## 项目结构

```
emomem_v1/
├── src/
│   ├── main.py          # EmoMemAgent — 顶层协调器
│   ├── perception.py    # 情绪识别、意图分类、紧急度评分
│   ├── memory.py        # 四层记忆：工作 / 情景 / 情感 / 语义
│   ├── planning.py      # 恢复轨迹建模 + Thompson 采样策略选择
│   ├── action.py        # 基于记忆锚定的回复生成
│   ├── adaptation.py    # 反馈处理与在线学习
│   ├── models.py        # 核心数据模型（StateVector、EpisodeRecord 等）
│   ├── llm_provider.py  # 多后端 LLM 抽象层
│   ├── mock_llm.py      # 规则兜底（无 API Key 时自动启用）
│   └── config.py        # 所有系统参数集中管理
├── web/
│   ├── index.html       # 含实时状态可视化的聊天界面
│   └── server.py        # Flask REST API 服务器
├── tests/
│   ├── test_unit.py         # 单元测试
│   ├── test_integration.py  # 端到端流水线测试
│   ├── test_emotion.py      # 情绪模型正确性测试
│   ├── test_crisis.py       # 危机检测与安全协议测试
│   └── llm_judge.py         # LLM-as-Judge 评估框架
└── docs/
    └── screenshot.png
```

---

## 快速上手

### 1. 安装依赖

```bash
pip install flask flask-cors anthropic openai
```

### 2. 配置 API Key

EmoMem 支持多种 LLM 后端，选择其一配置即可：

**方案 A — 火山引擎 ARK（豆包 Seed 系列，推荐）：**
```bash
export OPENAI_API_KEY=你的ARK_API_Key
export OPENAI_BASE_URL=https://ark.cn-beijing.volces.com/api/v3/
```

**方案 B — Anthropic Claude：**
```bash
export ANTHROPIC_API_KEY=你的Anthropic_Key
```

**方案 C — 无 API Key（Mock 模式）：**
系统自动降级到规则型 Mock Provider，所有功能正常运行，回复为模板化文本。

> 在 `src/llm_provider.py` 中，将 `_ARK_API_KEYS` 字段的 `"please enter your token"` 替换为你自己的 Key，或使用上述环境变量方式配置。

### 3. 启动 Web 界面

```bash
cd emomem_v1
python -m web.server
# 打开浏览器访问 http://127.0.0.1:8080
```

### 4. 运行测试

```bash
pytest tests/
# 或查看详细输出：
pytest tests/ -v
```

---

## Web 界面功能说明

界面实时展示 Agent 的内部状态：

| 面板 | 说明 |
|---|---|
| **情绪雷达** | 实时 8 轴 Plutchik 情绪雷达图 |
| **效价指数** | 连续情绪效价评分（−1 到 +1） |
| **紧急度仪表** | 危机风险分值，超过 0.9 触发安全协议 |
| **意图分布** | 检测到的前 3 个用户意图及其概率 |
| **恢复阶段** | 四阶段情绪恢复模型的当前位置 |
| **消融实验** | 独立开关情感记忆层 / MAR / Thompson 采样 |
| **LLM 影子对比** | LLM 评估值与公式计算值并排对比 |
| **策略历史** | 本次会话 Agent 使用的策略记录 |
| **记忆状态** | 情景记忆数量、已归档条目、基线置信度 |

---

## 环境变量

| 变量名 | 说明 | 默认值 |
|---|---|---|
| `OPENAI_API_KEY` | ARK 或 OpenAI 兼容接口的 Key | — |
| `ANTHROPIC_API_KEY` | Anthropic Claude Key | — |
| `OPENAI_BASE_URL` | OpenAI 兼容端点的 Base URL | `https://ark.cn-beijing.volces.com/api/v3/` |
| `EMOMEM_LLM_PROVIDER` | `anthropic` / `openai` / `mock` | 自动检测 |
| `EMOMEM_LLM_MODEL` | 覆盖默认模型名 | `doubao-seed-2-0-lite-260215` |
| `EMOMEM_REASONING_EFFORT` | 推理深度：`off` / `low` / `medium` / `high` | `off` |

---

## REST API

| 方法 | 端点 | 说明 |
|---|---|---|
| POST | `/api/chat` | 发送消息，返回回复 + 完整状态 |
| POST | `/api/reset` | 重置会话（保留 LLM 配置） |
| GET | `/api/state` | 获取当前 Agent 状态（不发送消息） |
| GET | `/api/status` | 系统状态（LLM 模式、模型、累计轮次） |
| GET/POST | `/api/config` | 获取或更新 LLM 配置 |
| GET/POST | `/api/reasoning` | 获取或设置推理深度 |
| POST | `/api/demo` | 运行预设演示场景（`work_stress` / `crisis` / `recovery`） |
| GET/POST | `/api/ablation` | 获取或切换消融实验开关 |

---

## 研究背景

EmoMem 处于三个活跃研究领域的交叉点：

- **AI 陪伴系统** — 针对孤独感和心理健康的长期情感支持
- **记忆增强 Agent** — 超越单会话上下文窗口的持久化用户建模
- **个性化策略学习** — 面向个体级支持优化的贝叶斯在线学习

主要理论基础：Plutchik 情绪轮模型（2001）、Hill 帮助技能理论（2009）、Bower 心境依存记忆理论（1981）、Gross 情绪调节过程模型（1998）、Thompson 采样（1933）、PPAM Agent 架构（Wang et al., 2024）。

---

## 许可证

MIT
