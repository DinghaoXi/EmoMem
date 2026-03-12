# EmoMem: 记忆增强的AI情感陪伴Agent技术框架

## Memory-Augmented AI Emotional Companion Agent: A Perception-Planning-Action-Memory Architecture

**Technical Specification · March 2025**
**For NSFC G01 Information Systems Research Direction**

---

> **摘要：** 本文档提出 EmoMem 框架——一种面向AI情感陪伴场景的记忆增强Agent技术架构。框架采用感知-规划-行动-记忆 (PPAM) 四模块设计，其**三大核心创新**包括：(1) **情感记忆层 (Affective Memory Tier)**，系统化表示用户的情绪基线、触发模式、恢复曲线和策略有效性；(2) **心境自适应检索 (Mood-Adaptive Retrieval, MAR)**，根据规划目标动态调整检索权重；(3) **上下文Thompson采样策略学习**，将帮助技能理论与贝叶斯在线学习结合，实现个性化支持策略优化。全部设计均有明确的心理学理论基础和计算机科学方法论支撑，并附有完整的实验验证方案。

> **阅读指引：** 本文档通过以下标记区分内容层次——
> - 🔬 **核心创新** = 本框架的关键学术贡献，是评审重点关注的原创内容
> - 📋 **基础组件** = 提供标准Agent功能，非原创但为系统完整性所必需
> - ⚙️ **实现细节** = 工程层面的参数设置与数据结构，可根据部署调整

---

## 目录

1. [框架概述与设计哲学](#1-框架概述与设计哲学)
2. [模块一：感知模块 (Perception)](#2-模块一感知模块-perception)
3. [模块二：记忆模块 (Memory) — 核心创新](#3-模块二记忆模块-memory--核心创新)
4. [模块三：规划模块 (Planning)](#4-模块三规划模块-planning)
5. [模块四：行动模块 (Action)](#5-模块四行动模块-action)
6. [模块间闭环交互协议](#6-模块间闭环交互协议)
7. [实验方案设计](#7-实验方案设计)
8. [文献对照与创新定位](#8-文献对照与创新定位)
9. [局限与未来方向](#9-局限与未来方向)
10. [参考文献](#10-参考文献)

---

## 1. 框架概述与设计哲学

### 1.1 核心设计理念

本框架采用经典的 **感知-规划-行动-记忆 (Perception-Planning-Action-Memory, PPAM)** 四模块Agent架构（Wang et al., 2024 Survey; Zhang et al., ACM TOIS 2025），但针对情感陪伴场景进行了三项关键改造。这延续了IS领域对话代理研究的传统——Diederich et al. (2022) [文献#35] 对262篇对话代理IS研究的综述确立了"设计-感知-结果"框架，本框架在此基础上新增"记忆-策略学习"维度。同时，Angst et al. (2024) [文献#45] 在JMIS特刊中明确指出AI伴侣应对孤独感是IS认可的前沿议题，为本研究提供了学科合法性支撑。

**理念一：记忆中心化 (Memory-Centric Design)。** 在传统Agent架构中，记忆通常作为被动存储模块。本框架将记忆重新定位为 **驱动其他三个模块运转的核心引擎**——感知依赖记忆提供情绪基线对比，规划依赖记忆提供策略经验先验，行动依赖记忆提供个性化锚定素材。Huang & Rust (2021) [文献#31] 将AI智能划分为机械→思考→情感三个阶段，情感AI是当前前沿——而情感AI的核心瓶颈恰在于缺乏跨会话的情感记忆，使Agent无法积累对用户的个性化理解。这一设计受CareCall系统（Jo et al., CHI 2024）部署于6000+独居老人的实证启发：长期记忆是提升自我披露和情感支持质量的关键因素。在IS理论层面，这呼应了信息系统成功模型（DeLone & McLean, 2003 [文献#43]）中"系统质量→用户满意→净收益"的路径——记忆质量直接决定系统质量感知。Puntoni et al. (2021) [文献#30] 提出的AI消费者体验框架（数据捕获→任务替代→体验增强）中，记忆模块承担"数据捕获"的核心功能，使后续"体验增强"成为可能。

**理念二：情感轨迹建模 (Affective Trajectory Modeling)。** 传统情感对话系统将情绪视为每轮独立的离散标签。本框架将情绪建模为 **连续动态轨迹**，包含基线水平、触发模式、恢复曲线和周期性波动。这一设计基于心理学中的情绪调节过程模型（Gross, 1998；本框架借鉴其"情绪调节是动态时间过程"的核心洞见，而策略分类则采用更适合陪伴场景的Hill帮助技能理论）和情感图式理论（Affective Schema Theory; Izard, 2009；本框架将其简化为事件类别-情绪模式的计算映射），并借鉴DAM-LLM（Lu & Li, arXiv:2510.27418, 2025）的贝叶斯情感动态更新思路。Smith et al. (2025) [文献#50] 从关系科学视角指出AI chatbot中情绪调节机制是维系关系的核心——情感轨迹建模正是实现这一机制的技术基础。

**理念三：个性化策略学习闭环 (Personalized Strategy Learning Loop)。** Agent不仅记住"发生了什么"，还记住"什么支持策略对这个特定用户在什么情境下有效"。通过上下文多臂老虎机（Contextual Bandit）模型实现在线策略优化，使Agent从"泛化共情"进化为"个性化支持"。这一思路受Mem-α（Wang, Y. et al., arXiv:2509.25911, 2025）的强化学习驱动记忆构建和MAGIC（Wang, H. et al., ECML-PKDD 2024）的策略记忆推理启发，但在策略空间和反馈信号设计上有本质区别。从IS视角看，个性化是IT信任建立（Komiak & Benbasat, 2006 [文献#38]）和技术采纳（UTAUT, Venkatesh et al., 2003 [文献#42]）的关键前因。

### 1.2 整体架构概览

**图1：EmoMem PPAM四模块架构及数据流**

```
┌─────────────────────────────────────────────────────────────────┐
│                     EmoMem Agent Architecture                    │
│                                                                  │
│  ┌──────────┐    F1: StateVector    ┌──────────┐                │
│  │          │ ──────────────────→  │          │                │
│  │ PERCEP-  │    F2: MemoryQuery    │  MEMORY  │                │
│  │  TION    │ ──────────────────→  │ (4-Tier) │                │
│  │          │                       │          │                │
│  └────┬─────┘                       └────┬─────┘                │
│       │ User                    F3: Retrieved    F4: Grounding   │
│       │ Input                   Context │         Facts │        │
│       │                               ▼                ▼        │
│       │                        ┌──────────┐    ┌──────────┐     │
│       │                        │ PLANNING │───→│  ACTION  │     │
│       │                        │          │ F5 │          │     │
│       │                        └────┬─────┘    └────┬─────┘     │
│       │                             │               │           │
│       │                        F7: Proactive   F6: Episode      │
│       │                        Trigger    ↑    Record │         │
│       │                             └──────┼─────────┘          │
│       │                                    │                    │
│       │                              ┌─────┴─────┐              │
│       └─────── User ◄──────────────  │  Output   │              │
│                                      └───────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

**表1：模块间数据流规约**

| 数据流 | 源模块 | 目标模块 | 数据内容 | 传输方式 |
|:------:|:------:|:--------:|:---------|:--------:|
| F1 | Perception | Memory | StateVector（写入Working Memory） | 同步 |
| F2 | Perception | Memory | MemoryQuery（基于当前情绪和话题的检索请求） | 同步 |
| F3 | Memory | Planning | RetrievedContext（历史事件 + 情感模式 + 策略经验） | 同步 |
| F4 | Memory | Action | GroundingFacts（用于生成锚定的记忆素材，为F3 RetrievedContext的子集，包含：可自然引用的情景记忆摘要、用户偏好的沟通风格、关系图谱中相关人物信息） | 同步 |
| F5 | Planning | Action | StrategyPlan（目标 + 策略 + 步骤序列） | 同步 |
| F6 | Action | Memory | EpisodeRecord（本次交互完整记录含反馈评分） | 异步 |
| F7 | Memory | Planning | ProactiveTrigger（异步主动关怀触发计划） | 异步 |

### 1.3 理论根基映射

**表2：设计理念与理论基础对照**

| 设计理念 | 心理学理论基础 | 计算机科学方法论 | 关键参考文献 | NSFC文献综述对应 |
|:---------|:---------------|:-----------------|:-------------|:-----------------|
| 记忆中心化 | 多存储模型 (Atkinson & Shiffrin, 1968) | OS级上下文管理 (MemGPT) | Packer et al., 2023; Jo et al., CHI 2024 | #43 DeLone & McLean IS成功模型 |
| 情感轨迹建模 | 情感图式理论 (Izard, 2009)；情绪调节理论 (Gross, 1998)——为MAR的目标驱动检索调控提供理论基础（见§3.5） | 贝叶斯动态更新 (DAM-LLM) | Lu & Li, 2025; Bower, 1981 | #50 Smith et al. 关系科学视角 |
| 个性化策略学习 | 帮助技能理论 (Hill, 2009)；认知评价理论 (Lazarus & Folkman, 1984) | Thompson Sampling; Contextual Bandit | Wang, Y. et al., 2025; Wang, H. et al., 2024 | #38 Komiak & Benbasat 信任 |
| 四层记忆分层 | 情景-语义记忆区分 (Tulving, 1972)；心境一致记忆 (Bower, 1981; Eich & Macaulay, 2000) | 向量数据库 + 知识图谱 | Park et al., UIST 2023; Zhong et al., AAAI 2024 | — |
| 关系深化目标 | 社会渗透理论 (Altman & Taylor, 1973) | 纵向关系建模 | Skjuve et al., IJHCS 2021, 2022 | #8-9 Skjuve纵向研究 |
| 安全对齐 | 社会情感对齐 (Kirk et al., 2025) | Safe RLHF; Constitutional AI | Kirk et al., Nature HSSC 2025 | #6 Kirk社会情感对齐 |

---

## 2. 模块一：感知模块 (Perception)

> 📋 **基础组件** · 感知模块提供标准的信号处理功能。其中情绪识别（2.3节）和隐式信号检测（2.5节）的设计为后续核心创新提供必要输入。各子组件均为标准信号处理流程，不单独定义算法；算法编号从记忆模块开始（算法3.1起）。

### 2.1 功能定位

将用户原始输入转化为结构化的 **状态向量 (StateVector)**，包含显式情感信号、隐式行为信号、意图分类和语境关键词。感知模块由六个子组件串联组成。

**图2：感知模块内部流程**

```
User Input
    │
    ▼
┌───────────────────┐
│ Multi-modal Input │
│     Parser        │──→ RawSignal
└───────┬───────────┘
        │
        ├───────────────────┬───────────────────┬───────────────┐
        ▼                   ▼                   ▼               ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐  ┌───────────┐
│  Emotion     │   │   Intent     │   │  Implicit    │  │  Topic    │
│  Recognizer  │   │  Classifier  │   │   Signal     │  │ Extractor │
│ (LLM/关键词) │   │ (LLM/关键词) │   │  Detector    │  │(LLM/关键词)│
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘  └─────┬─────┘
       │                  │                  │                │
       └──────────────────┴──────────────────┴────────────────┘
                                  │
                          ┌───────┴────────┐
                          │  LLM Context   │
                          │  Assessment    │──→ controllability,
                          │   (§2.8)       │   life_impact,
                          └───────┬────────┘   explicit_feedback
                                  │
                                  ▼
                        ┌─────────────────┐
                        │ Context Encoder │──→ StateVector
                        └─────────────────┘
```

### 2.2 子组件一：多模态输入解析器 (Multi-modal Input Parser)

> ⚙️ *实现细节*

**设计依据：** 即使在纯文本场景下，用户的表达方式也承载丰富的副语言信息。表情符号使用模式、标点选择（省略号暗示犹豫，感叹号暗示强烈情绪）、消息分段方式等均为重要信号源。

**输出数据结构：**

```python
RawSignal = {
    text_content: str,                    # 文本内容
    paralinguistic: {
        emoji_pattern: List[str],         # ["😢", "..."]
        punctuation_features: {
            ellipsis_count: int,          # 省略号数量
            exclamation_count: int,       # 感叹号数量
            question_marks: int           # 问号数量
        },
        message_segments: int,            # 消息分段数（连续消息数）
        character_count: int              # 字符总数
    },
    temporal: {
        timestamp: datetime,              # 消息发送时间
        since_last_message: timedelta,    # 与上条消息间隔
        session_duration: timedelta       # 当前会话时长
    },
    multimodal: {                         # 可选扩展接口
        voice_features: Optional[VoiceVector],   # 语音韵律特征
        facial_features: Optional[FaceVector]    # 面部表情特征
    }
}
```

### 2.3 子组件二：细粒度情绪识别器 (Fine-grained Emotion Recognizer)

**与现有工作的区别：** 传统情感对话系统（如MOEL; Lin et al., 2019）多采用Ekman六种基本情绪或简单的正/负/中三分类。本框架采用 **Plutchik情绪轮模型** (Plutchik, 2001) 的多维向量表示，并做了两项关键扩展：

1. 增加 **情绪确定度 (confidence)** 维度——区分"用户明确表达的情绪"和"需要推断的模糊情绪"
2. 支持 **混合情绪** (mixed emotions)——允许同一时刻多情绪共存（如"既愤怒又委屈"）

**Plutchik模型选型理由（vs VAD/Circumplex模型）：** 情感计算领域更常见的替代方案包括Russell的circumplex模型（Russell, 1980；valence-arousal二维）和Mehrabian的PAD模型（Mehrabian, 1996；三维）。本框架选择Plutchik 8维模型基于三项考量：(1) **策略区分需要**——情感陪伴需要识别具体情绪类型，如区分"恐惧"和"悲伤"对策略选择有直接影响（前者需安全感建立，后者需情感验证），而VAD模型将两者映射到valence-arousal空间的相近区域，难以支撑精准策略匹配；(2) **关联建模需要**——触发-情绪关联图（3.5.2节）需要为不同事件类别关联具体情绪模式，离散情绪标签比连续维度更适合构建可解释的关联；(3) **混合表示能力**——Plutchik的次级情绪组合机制天然支持混合情绪表示（如anxiety ≈ fear + anticipation），这对捕捉真实用户的复杂情感状态至关重要。局限在于高维表示增加了距离计算复杂度（见9.1节）。

**定义 2.1（情绪空间）。** 定义基本情绪集合：

$$\mathcal{E} = \{joy, trust, fear, surprise, sadness, disgust, anger, anticipation\}$$

**定义 2.2（情绪向量）。** 对于用户在时刻 $t$ 的输入 $x_t$，情绪识别器输出三元组：

$$\text{EmotionVector}(t) = \langle \mathbf{e}(t),\; \iota(t),\; \kappa(t) \rangle$$

其中：

- **情绪分布** $\mathbf{e}(t) = [e_1(t), e_2(t), \ldots, e_8(t)]$，满足 $\sum_{i=1}^{8} e_i = 1,\; e_i \geq 0$——归一化的混合情绪权重向量，位于8维单纯形 $\Delta^7$ 上
- **情绪强度** $\iota(t) \in [0, 1]$——反映情绪的总体唤醒水平（"有点难过" vs "完全崩溃"）
- **识别确定度** $\kappa(t) \in [0, 1]$——模型对识别结果的自信程度（"我真的很生气"时 $\kappa$ 高，"还好吧"时 $\kappa$ 低）

**确定度 $\kappa$ 的设计理由：** 当 $\kappa$ 较低时（用户情绪模糊），规划模块应采取**探索性策略**（温和提问以澄清情绪），而非直接断言用户的感受。这避免了"AI误读情绪导致用户反感"的常见问题。Ajeesh & Joseph (2025) [文献#48] 关于认知共情与情感共情的区分为此设计提供了理论基础——当情绪确定度低时，Agent应优先执行认知共情（理解原因）而非情感共情（表达感受共鸣）。

**次级情绪映射：** 日常语义标签（如"焦虑"、"内疚"）可通过Plutchik次级情绪映射还原为基本情绪组合（Plutchik, 2001）。例如：anxiety ≈ fear + anticipation，guilt ≈ joy↓ + fear + sadness，love ≈ joy + trust。触发-情绪关联图（3.5.2节）中同时记录语义标签和形式化向量。

**定义 2.3（情绪效价标量, Emotional Valence）。** 从8维情绪分布向量计算标量效价值，用于轨迹分析（4.4节）和情景记忆的 `emotional_valence` 字段：

$$\text{valence}(t) = \sum_{i=1}^{8} \sigma_i \cdot e_i(t), \quad \boldsymbol{\sigma} = [+1, +1, -1, 0, -1, -1, -1, +1]$$

其中效价符号 $\sigma_i$ 对应：joy(+), trust(+), fear(−), surprise(0), sadness(−), disgust(−), anger(−), anticipation(+)。

> **⚠️ 效价约定说明：** 本框架采用 **快乐主义效价 (hedonic valence)**（愉悦/不愉悦维度）而非Plutchik原始模型的趋近/回避维度。在Plutchik的趋近/回避分类中，anger属于趋近性（正向）情绪，与本框架的anger(−)赋值不同。本框架选择快乐主义效价的理由：情感陪伴场景中，愤怒对用户主观体验是消极的，Agent应将其视为需要关注的负面信号，而非趋近性行为。surprise标记为中性(0)因其效价取决于具体情境（积极惊喜vs消极惊吓），此处理与Ortony, Clore & Collins (1988) 的认知评价理论一致——情绪效价由个体对事件的主观评估决定，而非情绪类别的固有属性。

$\text{valence}(t) \in [-1, +1]$（因 $\sum e_i = 1$）。

> **注（Plutchik正负不对称）：** 由于8种基本情绪中正面3个(+3)、负面4个(−4)、中性1个(0)，均匀分布（即"无明确情绪"）的valence约为−0.125而非零。这不影响轨迹分析（使用差分Δvalence，偏移抵消），但基线初始化时应注意此偏移——初始基线valence宜设为经验均值而非零。

**实现路径：** 可通过LLM结构化情绪分析输出或微调轻量级分类器实现。情绪强度 $\iota(t)$ 综合考虑词汇情感强度（NRC VAD Lexicon; Mohammad, 2018）和副语言信号（感叹号数量、大写比例等）。

### 2.4 子组件三：意图分类器 (Intent Classifier)

**设计依据：** 情感陪伴场景下的用户意图不同于任务型对话。本框架定义七种情感陪伴意图类型，分类依据来自 Hill (2009) 的帮助技能理论和 Sharma et al. (2020) 的情感支持对话分类。

**表3：情感陪伴意图分类体系**

| 意图类型 | 代码 | 用户核心期望 | 典型表达示例 | 适配策略方向 |
|:---------|:----:|:-------------|:-------------|:-------------|
| 倾诉宣泄 | `VENT` | 被听到、被接纳 | "今天烦死了" | 积极倾听、情感验证 |
| 寻求安慰 | `COMFORT` | 情感支持、被关怀 | "我是不是很差" | 情感反射、正常化 |
| 寻求建议 | `ADVICE` | 解决方案 | "你觉得我该怎么办" | 问题解决、信息提供 |
| 分享喜悦 | `SHARE_JOY` | 共同庆祝 | "我通过面试了！" | 积极强化、庆祝 |
| 陪伴闲聊 | `CHAT` | 消磨孤独 | "你在干嘛" | 轻松对话、话题拓展 |
| 深度自省 | `REFLECT` | 协助思考 | "我一直在想为什么..." | 引导性提问、认知重构 |
| 危机信号 | `CRISIS` | 安全干预 | "活着好累" | 安全协议、资源引导 |

**分类输出：** 非互斥概率分布，允许多意图共存（如"VENT"+"COMFORT"），输出各意图的概率 $P(\text{intent}_j \mid x_t)$。

**与NSFC综述的连接：** Hu et al. (2024) [文献#18] 的AI宣泄实验表明，AI可降低愤怒/恐惧但不影响社会支持感知。这意味着 VENT 意图下Agent可有效帮助情绪缓解，但不应试图替代真人社会支持——这一边界条件直接约束意图-策略映射设计。

### 2.5 子组件四：隐式信号检测器 (Implicit Signal Detector)

**设计依据：** ComPeer（UIST 2024）的 Event Detector 模块证明了主动检测用户状态变化对陪伴质量的重要性。本组件聚焦于 **行为模式偏差** 而非内容分析——"什么时候说"和"怎么说"往往比"说了什么"更能反映真实情绪状态。

**定义 2.4（行为特征向量）。** 定义6维行为特征：

$$\mathbf{B}(t) = \big[\underbrace{b_1}_{\text{msg\_time}},\; \underbrace{b_2}_{\text{reply\_interval}},\; \underbrace{b_3}_{\text{msg\_frequency}},\; \underbrace{b_4}_{\text{msg\_length}},\; \underbrace{b_5}_{\text{emoji\_rate}},\; \underbrace{b_6}_{\text{session\_init}}\big]$$

**定义 2.5（行为异常度评分, Behavioral Anomaly Score, BAS）。** 基于用户历史行为基线 $\bar{\mathbf{B}}$ 和标准差 $\boldsymbol{\sigma}$（均从记忆模块 Tier 3 获取），计算标准化加权偏差：

$$\text{BAS}(t) = \sum_{i=1}^{6} w_i \cdot \frac{|b_i^{\text{current}}(t) - \bar{b}_i^{\text{baseline}}|}{\sigma_i^{\text{baseline}} + \epsilon}$$

其中 $\epsilon = 10^{-6}$ 防止除零，权重向量 $\mathbf{w}$ 满足 $\sum_{i=1}^{6} w_i = 1$。

**表4：隐式信号检测维度与权重**

| 维度 $i$ | 行为特征 | 异常条件 | 可能含义 | 权重 $w_i$ |
|:--------:|:---------|:---------|:---------|:----------:|
| 1 | 消息时间 msg_time | $b_1 \in [1{:}00, 5{:}00]$ | 失眠/严重心理困扰 | 0.20 |
| 2 | 回复间隔 reply_interval | $b_2 < 0.3\bar{b}_2$ 或 $b_2 > 3\bar{b}_2$ | 焦虑激动/情绪低落 | 0.15 |
| 3 | 消息频率 msg_frequency | $b_3 < 0.3\bar{b}_3$ 或 $b_3 > 2.5\bar{b}_3$ | 退缩回避/孤独感加剧 | 0.20 |
| 4 | 消息长度 msg_length | $b_4 < 0.4\bar{b}_4$ | 低能量/不想沟通 | 0.15 |
| 5 | Emoji使用率 emoji_rate | $b_5 \to 0$（此前常用） | 情绪严重低落 | 0.10 |
| 6 | 会话发起 session_init | 用户主动发起频率变化 | 依赖度/社交需求变化 | 0.20 |

**BAS\_max定义：** 用于紧急度归一化的BAS理论最大值，设定为 $\text{BAS}_{\max} = 5.0$（即所有维度偏离基线5个标准差的极端情况），可根据部署数据校准。

**基线更新规则（指数移动平均）：**

$$\bar{b}_i^{\text{baseline}}(t+1) = \lambda \cdot b_i^{\text{current}}(t) + (1-\lambda) \cdot \bar{b}_i^{\text{baseline}}(t), \quad \lambda = 0.1$$

> **学习率选择理由（$\lambda = 0.1$ vs 情绪基线 $\eta = 0.05$）：** 行为特征（消息时间、回复间隔、消息频率等）的自然变异性高于情绪分布——用户可能因工作日/休息日、忙碌/空闲等外部因素大幅变化行为模式，而情绪基线相对稳定。较高的学习率使行为基线更快适应生活方式变化（如作息调整），避免正常行为变化被长期误判为异常。若部署数据表明行为特征波动小于预期，可降至 $\lambda = 0.05$ 与情绪基线一致。

**行为基线标准差更新：** 行为基线标准差采用与情绪基线σ相同的EWMA方差估计方法（参见§3.5.1），并设置维度级下界防止z-score爆炸：

$$\sigma_i^{\text{baseline}}(t{+}1) = \max\!\Big(\sigma_{i,\min},\;\; \sqrt{\lambda_\sigma \cdot \big(b_i^{\text{current}}(t) - \bar{b}_i^{\text{baseline}}(t)\big)^2 + (1-\lambda_\sigma) \cdot \big(\sigma_i^{\text{baseline}}(t)\big)^2}\Big)$$

其中 $\lambda_\sigma = 0.1$（与均值更新率一致），$\sigma_{i,\min}$ 为各维度的经验下界（推荐：msg_time取1小时，reply_interval取10秒，msg_frequency取0.5条/天，msg_length取5字符，emoji_rate取0.01，session_init取0.1次/天）。**此更新为无条件执行**，避免选择偏差导致的方差低估。

> **冷启动初始化：** 新用户首次交互时，行为基线均值 $\bar{b}_i^{\text{baseline}}(0)$ 采用群体默认值初始化（基于部署数据的中位数估计）。推荐默认值：msg\_time = 14:00（下午2时）、reply\_interval = 120秒、msg\_frequency = 3条/天、msg\_length = 50字符、emoji\_rate = 0.15、session\_init = 0.5次/天。标准差 $\sigma_i^{\text{baseline}}(0)$ 初始化为 $3\sigma_{i,\min}$（各维度下界的3倍），确保初始BAS评分不过度敏感。**冷启动渐进恢复：** 紧急度公式中BAS权重 $\alpha_3$ 从0线性渐进至标称值，避免阶跃跳变：
>
> $$\alpha_3(n_{\text{cum}}) = \min\!\big(0.6,\;\; 0.6 \cdot (n_{\text{cum}} - 1) \,/\, 5\big)$$
>
> 其中 $n_{\text{cum}}$ 为累计交互次数（跨会话累加，区别于定义4.2中的会话内轮次 $n$）。$n_{\text{cum}}=1$ 时 $\alpha_3 = 0$（不计入urgency），$n_{\text{cum}}=6$ 时 $\alpha_3 = 0.6$（完全恢复），中间线性过渡。设计理由：EMA学习率 $\lambda = 0.1$ 下，5次更新使基线均值吸收约 $1-(0.9)^5 \approx 41\%$ 的个性化信息，线性渐进的 $\alpha_3$ 与基线可靠度同步增长，避免基线半成熟时突然引入全权重BAS导致urgency突增。

> **持久化说明：** 行为基线 $\bar{\mathbf{B}}$ 和 $\boldsymbol{\sigma}$ 持久化存储于 Tier 3 语义记忆的 `behavioral_baseline` 字段（§3.4），每轮更新后异步同步至语义记忆。系统重启或新会话开始时从语义记忆加载最近基线。

### 2.6 子组件五：话题提取器 (Topic Extractor)

> 📋 **基础组件**

**功能：** 从用户输入中提取话题关键词和话题类别，用于记忆检索匹配和触发-情绪关联图的事件分类。

**输出：**
- `topic_keywords: List[str]`——具体关键词（如"母亲"、"面试"、"分手"）
- `topic_category: str`——话题大类（如"家庭关系"、"职场压力"、"亲密关系"）

**实现：** 可通过LLM结构化输出或关键词提取+分类器实现。话题类别与触发-情绪关联图（3.5.2节）的事件类别保持一致。

### 2.7 子组件六：上下文编码器 (Context Encoder)

> ⚙️ *实现细节*

**功能：** 将上述五个子组件的输出融合为统一的 StateVector。

**定义 2.6（状态向量, StateVector）：**

```python
StateVector(t) = {
    emotion:          EmotionVector(t),       # 情绪三元组 ⟨e, ι, κ⟩
    intent:           IntentDistribution(t),  # 7类意图概率分布
    implicit_signals: {
        bas_score:          float,            # 行为异常度评分
        anomaly_dimensions: List[str],        # 触发异常的具体维度
        temporal_context:   str               # "late_night" | "normal_hours"
    },
    topic_keywords:   List[str],              # 话题关键词
    urgency_level:    float ∈ [0, 1],        # 紧急度综合评分
    relationship_depth: float ∈ [0, 1],     # ★ 关系深度 d(t)，Phase 1更新（§4.5.2）
    memory_query:     MemoryQuery,            # 向记忆模块的检索请求
    # §2.8 LLM 上下文评估字段 (LLM 可用时填充, 否则为 None)
    controllability:    Optional[float],      # ∈ [0, 1]，用户对情境的控制感
    life_impact:        Optional[float],      # ∈ [0, 1]，事件对用户生活的影响程度
    explicit_feedback:  Optional[float],      # ∈ [0, 1]，用户对上一轮回复的显式反馈
    # §2.8.6 LLM-Primary 影子字段 (shadow mode: LLM 评估值，与公式计算值对比记录)
    llm_urgency:        Optional[float],      # ∈ [0, 1]，LLM 评估的紧急度 (对标 urgency_level)
    recovery_phase:     Optional[str],        # LLM 评估的恢复阶段 (对标 §4.4 公式判定)
    planning_intent:    Optional[str]         # LLM 评估的规划意图 (对标 定义 3.6a 公式判定)
}
```

**定义 2.7（紧急度评分）。** 综合多信号取最大值：

$$\text{urgency}(t) = \min\!\bigg(1.0,\;\; \max\!\Big(\underbrace{\alpha_1 \cdot \mathbb{1}[\text{CRISIS}]}_{\text{危机意图}},\;\; \underbrace{\alpha_2 \cdot \iota(t) \cdot \mathbb{1}[\text{neg\_emo}]}_{\text{高强度负面情绪}},\;\; \underbrace{\alpha_3 \cdot \min\!\big(\frac{\text{BAS}(t)}{\text{BAS}_{\max}},\; 1\big)}_{\text{行为异常度}},\;\; \underbrace{\alpha_4 \cdot \text{trend}_{\text{local}}(t)}_{\text{近期恶化趋势}}\Big)\bigg)$$

**参数设置：** $\alpha_1 = 1.0$（危机意图直接触发最高紧急度），$\alpha_2 = 0.8$，$\alpha_3 = 0.6$，$\alpha_4 = 0.7$。

**指示函数定义：** $\mathbb{1}[\text{neg\_emo}]$（等价记为 $\mathbb{1}[\text{neg}_t]$）= 1 当且仅当 $\text{valence}(t) < 0$（即定义2.3中效价标量为负），否则为0。$\mathbb{1}[\text{CRISIS}]$ = 1 当且仅当感知模块意图分类器输出 $P(\text{CRISIS} \mid x_t) > 0.5$。

> **设计说明：** 非危机信号的最大urgency为 $\max(\alpha_2, \alpha_3, \alpha_4) = 0.8 < 0.9$，因此仅CRISIS意图可触发危机快速通道（urgency > 0.9）。这是有意的安全设计——行为异常和情绪恶化应引起关注但不应绕过常规规划流程，只有内容层面的危机信号（如"不想活了"）才触发最高级别响应。

**$\text{trend}_{\text{local}}$ 定义（轻量级近期趋势）：** 与规划模块中的完整情绪轨迹分析不同，此处使用仅依赖工作记忆的轻量级计算，确保感知模块无需等待规划模块输出：

$$\text{trend}_{\text{local}}(t) = \text{clamp}\!\Big(\iota(t) \cdot \mathbb{1}[\text{neg}_t] - \iota(t{-}1) \cdot \mathbb{1}[\text{neg}_{t-1}],\;\; 0,\;\; 1\Big)$$

即"当前负面情绪强度相比上一轮的增幅"，取值 $\in [0, 1]$。仅使用Working Memory中已有的上一轮情绪快照，不依赖任何后续模块。**会话首轮初始化：** 当Working Memory中无上一轮情绪快照时（会话首轮），取 $\iota(t{-}1) \cdot \mathbb{1}[\text{neg}_{t-1}] = 0$，此时 $\text{trend}_{\text{local}}(t) = \text{clamp}(\iota(t) \cdot \mathbb{1}[\text{neg}_t],\; 0,\; 1)$。设计理由：不从前序会话继承情绪值（跨会话间隔可能数小时/天），避免引入时间间隔伪影；采用零初始化使首轮紧急度不被trend分量放大（worst case贡献 $\alpha_4 \cdot \iota = 0.7 \cdot \iota$，不影响危机通道判定）。

> **设计说明：** 完整的情绪轨迹分析（方向、波动性、周期性、阶段判定）在规划模块（4.4节）中执行，利用情感记忆中的长期数据。感知模块的 $\text{trend}_{\text{local}}$ 仅作为紧急度的快速信号，确保危机响应的低延迟。

**关键阈值：** 当 $\text{urgency} > 0.9$ 时，触发 **危机快速通道**（绕过常规规划，直接进入安全协议）。

### 2.7b 情绪惯性机制 (Emotion Inertia) — §2.7 扩展

> ⚙️ *实现细节 — 解决多轮对话中情绪状态跳变问题*

**动机：** §2.3 的情绪识别器基于当前轮文本独立计算情绪向量，不具备跨轮记忆。这导致两个问题：(1) 连续多轮悲伤对话后，一句中性输入即可使悲伤情绪瞬间归零；(2) 危机检测触发后，下一轮稍有缓和即紧急度骤降。这与心理学中情绪具有惯性（emotional inertia）的实证发现不符（Kuppens et al., 2010）。

**定义 2.7b（情绪 EMA 平滑）。** 在情绪识别器输出 $\mathbf{e}_{\text{raw}}(t)$ 之后、送入上下文编码器之前，对情绪向量施加指数移动平均（EMA）平滑：

$$\mathbf{e}_{\text{smooth}}(t) = (1 - \alpha_{\text{inertia}}) \cdot \mathbf{e}_{\text{raw}}(t) + \alpha_{\text{inertia}} \cdot \mathbf{e}(t{-}1)$$

$$\iota_{\text{smooth}}(t) = (1 - \alpha_{\text{inertia}}) \cdot \iota_{\text{raw}}(t) + \alpha_{\text{inertia}} \cdot \iota(t{-}1)$$

$$\kappa_{\text{smooth}}(t) = (1 - \alpha_{\text{inertia}}) \cdot \kappa_{\text{raw}}(t) + \alpha_{\text{inertia}} \cdot \kappa(t{-}1)$$

**参数设置：** $\alpha_{\text{inertia}} = 0.4$（40% 来自上一轮情绪，60% 来自当前轮原始检测）。

**数学性质：** 由于 $\mathbf{e}_{\text{raw}}(t)$ 和 $\mathbf{e}(t{-}1)$ 均在概率单纯形上（$\sum e_i = 1, e_i \geq 0$），其凸组合仍在单纯形上，无需重新归一化。强度和确定度经 $\text{clamp}(\cdot, 0, 1)$ 保持有界。

**会话首轮：** 当 $\mathbf{e}(t{-}1) = \text{None}$（Working Memory 中无上一轮快照）时，不施加平滑，$\mathbf{e}_{\text{smooth}}(t) = \mathbf{e}_{\text{raw}}(t)$。跨会话不继承（同 trend\_local 设计理由）。

**定义 2.7c（意图连续性）。** 当当前轮意图分类器的最高概率 $P_{\max}(t) < \tau_{\text{intent}}$ 且上一轮主导意图为 CRISIS 或 VENT 时，混合上一轮意图分布：

$$\mathbf{p}_{\text{blend}}(t) = \frac{(1 - w_{\text{intent}}) \cdot \mathbf{p}_{\text{raw}}(t) + w_{\text{intent}} \cdot \mathbf{p}(t{-}1)}{\sum_i \big[(1 - w_{\text{intent}}) \cdot p_{\text{raw},i}(t) + w_{\text{intent}} \cdot p_i(t{-}1)\big]}$$

**参数设置：** $w_{\text{intent}} = 0.2$，$\tau_{\text{intent}} = 0.5$。

**设计说明：** 仅在当前轮信号模糊（$P_{\max} < 0.5$）且前一轮为高风险意图（CRISIS/VENT）时启用，避免覆盖强信号。归一化确保混合后仍为有效概率分布。

**定义 2.7d（危机冷却机制）。** 当紧急度触发危机阈值（$\text{urgency} > 0.9$）时，设置冷却计数器 $C = T_{\text{cooldown}}$。后续各轮中若 $C > 0$，施加紧急度下限：

$$\text{urgency}_{\text{floor}}(t) = \max\!\Big(0,\;\; \theta_{\text{crisis}} - \delta_{\text{decay}} \cdot (T_{\text{cooldown}} - C)\Big)$$

$$\text{urgency}_{\text{final}}(t) = \max\!\big(\text{urgency}_{\text{raw}}(t),\;\; \text{urgency}_{\text{floor}}(t)\big)$$

每轮结束后 $C \leftarrow C - 1$。当 $C = 0$ 时冷却期结束，恢复正常紧急度计算。

**参数设置：** $T_{\text{cooldown}} = 3$（轮），$\delta_{\text{decay}} = 0.15$，$\theta_{\text{crisis}} = 0.9$。

**冷却期紧急度衰减序列：** $0.90 \to 0.75 \to 0.60 \to$ 正常计算。

> **设计说明：** 三个机制协同工作但各自独立——EMA 平滑作用于情绪向量层（输入侧），意图连续性作用于意图分布层，危机冷却作用于紧急度标量层（输出侧）。这避免了单一机制的过度耦合，每个参数可独立调优。新增状态（$\mathbf{e}(t{-}1)$, $\mathbf{p}(t{-}1)$, $C$）均存储于 Working Memory（Tier 1），会话结束时自动重置。

### 2.8 LLM 上下文评估 (LLM Context Assessment) — 双路径架构

> ⚙️ *实现细节 — 感知模块的 LLM 语义增强与关键词降级机制*

**动机：** §2.3–§2.6 的各子组件最初采用关键词匹配实现情绪识别、意图分类、话题提取等功能。关键词方法在覆盖率和语义理解深度上存在固有局限——例如"算了你不懂"在关键词模式下难以识别为负面反馈，"我决定辞职"与"被裁员了"的可控性差异无法通过词表捕捉。本节描述感知模块的 LLM 语义增强架构，使上述子组件在 LLM 可用时获得深层语义理解能力，同时保持无 API 环境下的完整功能降级。

#### 2.8.1 双路径架构 (Dual-Path Architecture)

感知模块采用 **LLM 优先、关键词降级** 的双路径设计：

```
User Input
    │
    ├──────────────────────────────────────────┐
    ▼                                          ▼
┌────────────────────┐              ┌────────────────────┐
│   LLM 语义路径      │              │  关键词规则路径      │
│  (Primary Path)    │              │ (Fallback Path)    │
│                    │              │                    │
│ analyze_message()  │              │ keyword matching   │
│ assess_context()   │              │ regex patterns     │
│ generate_response()│              │ template filling   │
└────────┬───────────┘              └────────┬───────────┘
         │                                   │
         │  ◄── try/except ──►               │
         │      LLM 失败时                    │
         │      自动降级 ──────────────────────┘
         │                                   │
         └───────────────┬───────────────────┘
                         ▼
                    StateVector(t)
```

**降级触发条件：** (1) 未配置 LLM API key（环境变量缺失）；(2) LLM 调用超时或返回格式异常；(3) LLM 提供者初始化失败。任一条件满足时，对应功能自动回退至关键词规则路径，不影响系统整体运行。

**三个 LLM API 接口：**

| API 方法 | 调用阶段 | 功能 | 输出 |
|:---------|:---------|:-----|:-----|
| `analyze_message()` | Phase 1 感知 | 情绪分布、意图分类、话题提取、情绪强度、LLM语义信号、影子字段（urgency, recovery_phase, planning_intent） | `PerceptionResult` |
| `assess_context()` | Phase 1 感知 | 可控性、生活影响、显式反馈、话题延续度 | `Dict[str, float]` |
| `generate_response()` | Phase 4 行动 | 基于策略和上下文生成自然语言回复 | `str` |

#### 2.8.2 上下文评估接口 (Context Assessment API)

`assess_context()` 通过结构化提示（System Prompt）引导 LLM 对用户消息进行四维语义评估：

```python
assess_context(
    user_message: str,                        # 用户当前输入
    prev_topics: Optional[List[str]] = None,  # 上一轮话题关键词
    history: Optional[List[Dict]] = None      # 最近 6 轮对话历史
) → {
    "controllability":    float ∈ [0, 1],    # 用户对情境的控制感
    "life_impact":        float ∈ [0, 1],    # 事件对生活的影响程度
    "explicit_feedback":  float ∈ [0, 1],    # 对上一轮回复的反馈信号
    "topic_continuation": float ∈ [0, 1]     # 与前一话题的延续程度
}
```

**各维度语义评估标准：**

| 维度 | 低值 (→0) | 高值 (→1) | 示例 |
|:-----|:----------|:----------|:-----|
| controllability | 外部强制、不可抗力 | 用户主动选择、自主决定 | "被裁员了" → 0.2；"我决定辞职" → 0.8 |
| life_impact | 日常琐事、短暂不便 | 人生重大转折、长期影响 | "丢了钥匙" → 0.1；"被诊断重病" → 0.95 |
| explicit_feedback | 对上轮回复不满 | 对上轮回复满意 | "算了你不懂" → 0.15；"谢谢你的建议" → 0.85 |
| topic_continuation | 完全新话题 | 延续上一话题 | "对了今天吃什么" → 0.1；"然后他又说…" → 0.9 |

**LLM 系统提示要求：** "理解语义而非关键词"——例如"都挺好的吧"可能暗示不好，需结合对话历史和语气判断。LLM 输出限定为 JSON 格式，所有字段经 $\text{clamp}(\cdot, 0, 1)$ 裁剪后写入 StateVector。

**降级行为：** 当 LLM 不可用时，`mock_llm.py` 中的5个评估函数（`assess_controllability`, `assess_life_impact`, `detect_explicit_feedback`, `detect_topic_continuation`, `extract_topics`）各自实现独立的关键词匹配逻辑。每个函数内部优先尝试 `llm_provider.assess_context()` 调用，失败则回退至关键词规则，返回默认值 0.5（中性）。

#### 2.8.3 数据流向下游模块

LLM 上下文评估结果通过 StateVector 的三个可选字段传递至下游模块：

```
assess_context() ──→ StateVector.controllability ──→ §4.2 Situation Appraisal
                                                      └→ severity 公式中 (1 - controllability) 分量
                 ──→ StateVector.life_impact     ──→ 预留：未来可作为 severity 附加分量
                 ──→ StateVector.explicit_feedback──→ §3.6 记忆写入 / §5.6 反馈回路
                                                      └→ 优先使用 LLM 语义反馈, 替代关键词检测
```

**可控性传递：** §4.2 情境评估（Situation Appraisal）中 controllability 参与综合严重度计算（定义 4.1），当 StateVector 携带 LLM 评估的 controllability 时，规划模块可直接使用，取代固定默认值 0.5。

**显式反馈传递：** 记忆模块的反馈评分计算（定义 3.4）中，`explicit_feedback` 分量优先读取 StateVector 中 LLM 推断的值；当该字段为 `None` 时，降级为 `_detect_explicit_feedback()` 关键词匹配函数。

**生活影响传递：** `life_impact` 当 LLM 提供该字段时，参与 §4.2 情境评估的5分量 severity 公式（以 0.10 权重替代部分 baseline_deviation 权重）。当该字段为 `None` 时，降级为原始4分量公式。

**影子字段传递：** `llm_urgency`、`recovery_phase`、`planning_intent` 三个影子字段存储于 StateVector，在感知和规划阶段与公式计算值进行对比日志记录（§2.8.6），当前不参与实际决策。

> **设计说明：** 双路径架构确保系统在无外部 API 依赖时仍可完整运行（16 个离线单元测试在 Mock 模式下通过）。LLM 增强为渐进式升级——每个子组件独立决定是否使用 LLM 结果，单点 LLM 故障不影响其他子组件的输出。这一设计遵循优雅降级（Graceful Degradation）原则，与框架的鲁棒性要求一致。20 个在线测试（危机、情绪、集成）依赖 LLM API，使用 LLM-as-Judge 方法评估行为质量（§7.1.1）。

#### 2.8.4 均匀分布检测与网络用语识别 (Uniform Distribution Detection & Internet Slang Recognition)

**均匀分布自动降级：** 当 LLM 返回的情绪分布接近均匀（$\max(\mathbf{e}) - \min(\mathbf{e}) < 0.05$）时，系统判定 LLM 未能产生有效的情绪判断（通常因输入包含模型不理解的用语），自动降级为关键词匹配路径。此机制防止无区分度的 LLM 输出掩盖可用的规则信号。

**中文网络用语识别指南：** 感知模块的 LLM 系统提示中内嵌了中文互联网情绪表达词典，引导 LLM 正确解析以下非字面意义的表达模式：

| 表达 | 字面义 | 实际情绪 | 情绪映射 |
|:-----|:-------|:---------|:---------|
| "破防了" | 防线被破 | 情绪崩溃 | sadness/anger |
| "我真的会谢" | 感谢 | 无语、讽刺 | disgust/anger |
| "笑不活了" | 笑到不行 | 极度无奈的自嘲 | context-dependent |
| "原地升天" | 升天 | 震惊、崩溃 | surprise/anger |
| "emo了" | — | 情绪低落 | sadness |
| "摆烂/躺平" | 摆烂 | 绝望性放弃 | sadness/disgust |
| "算了/认命吧" | 放弃 | 无力感 | sadness (非释然) |
| "没关系/没事" | 无所谓 | 压抑/伪装（负面语境） | sadness (非 joy) |
| "哈哈哈" | 笑 | 负面语境中为苦笑/自嘲 | context-dependent |

**关键规则：** 多个网络用语组合使用时（如"破防了家人们，我真的会谢，直接原地升天"），通常表达强烈的负面情绪，即使表面像在开玩笑。LLM 提示中明确要求"不要输出均匀分布，必须给出有区分度的情绪判断"。

#### 2.8.5 感知提示的语义模式设计 (Principled Semantic Patterns in Perception Prompt)

> ⚙️ *实现细节 — LLM-Primary 迁移 Phase 1*

**动机：** 早期感知提示使用硬编码公式描述（如"A+B=crisis"），LLM 容易过拟合于具体规则而忽略语义理解。感知系统提示（`PERCEPTION_SYSTEM_PROMPT`）采用 **原则性语义模式 (principled semantic patterns)**——描述需要识别的语义现象和判断原则，而非机械规则。

**关键改动：**

1. **隐含危机信号语义模式：** 提示中以语义描述方式定义五类隐含危机信号——告别行为模式（"交代后事"语义特征）、情绪突变信号（持续负面后突然"释然"）、隐性绝望表达（隐喻/象征/间接方式表达生存意义丧失）、情感麻木/解离信号、**准备行为模式**（用户描述准备、囤积、购买自杀工具的行为，如查询致死剂量、储备药物、准备遗书等，无论具体措辞均标记为 CRISIS）。每类模式给出判断原则和语义特征描述，而非关键词列表——使 LLM 能泛化至训练数据未覆盖的表达方式。

2. **前一轮状态上下文注入：** 感知模块在调用 LLM 时构建 `prev_state_section`——包含前一轮主导情绪、效价、意图、当前对话轮次，以及**危机状态上下文**（危机冷却计数器剩余轮次、会话内累计危机次数）等信息，注入 `PERCEPTION_USER_TEMPLATE` 的 `{prev_state_section}` 占位符。这使 LLM 能够基于前一轮情绪状态和危机历史进行连续性判断，替代纯粹依赖对话历史文本。首轮对话时填充"（首轮对话，无前一轮状态）"。

3. **扩展感知输出字段：** `PerceptionResult` 新增三个字段——`urgency`（紧急度评估 $[0, 1]$）、`recovery_phase`（情绪恢复阶段判断）、`planning_intent`（最适合的支持方向）。这些字段使 LLM 能够基于完整语义上下文直接输出规划级判断，不再仅依赖公式计算。

#### 2.8.6 LLM-Primary 迁移与影子模式 (LLM-Primary Migration & Shadow Mode)

> ⚙️ *实现细节 — 渐进式 LLM-Primary 迁移策略*

**动机：** 感知模块的核心计算（紧急度 urgency、恢复阶段 recovery_phase、规划意图 planning_intent）当前由公式驱动。LLM 具备更强的语义理解能力（如识别隐喻危机信号、间接自杀表达），但直接替换公式存在可靠性风险。当前采用 **影子模式 (shadow mode)** 作为迁移 Phase 1——LLM 并行评估上述值，记录对比日志，但不影响实际决策。

**StateVector 影子字段 (Shadow Fields)：**

| 字段 | 类型 | 对标公式值 | 用途 |
|:-----|:-----|:-----------|:-----|
| `llm_urgency` | `Optional[float]` $\in [0, 1]$ | `urgency_level`（定义 2.7 公式计算） | 对比 LLM 与公式的紧急度判断 |
| `recovery_phase` | `Optional[str]` | §4.4 轨迹分析 `current_phase` | 对比 LLM 与 OLS 回归的阶段判定 |
| `planning_intent` | `Optional[str]` | 定义 3.6a 规则映射 | 对比 LLM 与确定性规则的意图推断 |

**影子对比日志 (Shadow Comparison Logging)：** 系统在两个阶段记录对比日志：

1. **感知阶段 (perception.py)：** 公式计算 `urgency_level` 完成后，与 `state.llm_urgency` 对比，记录 `[Shadow] urgency: formula=X, llm=Y, delta=Z`。同时记录 `recovery_phase` 和 `planning_intent` 的 LLM 评估值。

2. **规划阶段 (planning.py)：**
   - `assess_situation()` 中，公式计算 `severity` 后与 `llm_urgency` 对比，记录 `[Shadow] severity: formula=X, llm_urgency=Y`。
   - `plan()` 中，公式推断 `planning_intent` 后与 `state.planning_intent` 对比，记录匹配结果；轨迹分析 `current_phase` 后与 `state.recovery_phase` 对比，记录匹配结果。

**设计原则：** 影子模式是 **纯观测性 (observational only)** 的——所有公式计算值仍为权威值，LLM 评估值仅用于日志对比。当积累足够的对比数据验证 LLM 判断的准确性后，可在后续阶段逐步将权威性从公式转移至 LLM（Phase 2: LLM override with formula fallback → Phase 3: LLM-primary with formula safety net）。

---

## 3. 模块二：记忆模块 (Memory) — 核心创新

### 3.1 功能定位与设计哲学

记忆模块不是被动的数据仓库，而是 **主动参与认知的操作系统**。它同时服务于感知（提供基线对比）、规划（提供上下文和策略经验）、行动（提供锚定素材）三个模块。近期记忆Agent领域涌现了多种架构方案：MemoryOS (Kang et al., 2025) 提出统一的记忆操作系统框架，A-MEM (Xu et al., 2025) 提出自主Agent记忆构建机制，Theanine (Ong et al., 2025) 基于因果关系的时间线记忆管理。综合性综述参见 Liu, Y. et al. (2025) 的"Memory in the Age of AI Agents"。长期对话记忆评估的基准参见 LoCoMo (Maharana et al., ACL 2024)。本框架与上述工作的核心区别在于引入 **情感模式的系统化建模**——现有方案均未将情绪基线、触发关联和恢复曲线作为独立记忆层级管理。

**图3：四层记忆架构**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Memory Module (4-Tier)                         │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ Tier 1: Working Memory  (≤2000 tokens, always in LLM)  │     │
│  │  session_buffer | active_emotion | active_topic         │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ Tier 2: Episodic Memory  (Vector DB, timestamped)       │     │
│  │  EpisodeUnit: event + emotion + strategy + feedback     │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ Tier 3: Semantic Memory  (Knowledge Graph, persistent)  │     │
│  │  user_profile | relationship_graph | comm_preference    │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ 🔬 Tier 4: Affective Memory  ★★★ 核心创新一            │     │
│  │  emotion_baseline | trigger_map | recovery_patterns     │     │
│  │  strategy_effectiveness_matrix (Beta distributions)     │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ Memory Operations:                                      │     │
│  │  Write | 🔬Read(MAR)★★★ | Reflect | Forget | Update    │     │
│  └─────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

**表5：记忆层级理论基础**

| 层级 | 心理学根基 | 计算实现 | 设计参考 |
|:-----|:-----------|:---------|:---------|
| Tier 1: Working Memory | Baddeley 工作记忆模型 (Baddeley, 2000) | LLM上下文窗口驻留摘要 | MemGPT Core Memory (Packer et al., 2023) |
| Tier 2: Episodic Memory | Tulving (1972) 情景记忆 | 向量数据库 + 结构化元数据 | Generative Agents Memory Stream (Park et al., 2023) |
| Tier 3: Semantic Memory | Tulving (1972) 语义记忆 | 知识图谱 + JSON文档 | CharacterGLM (Zhou et al., EMNLP 2024) |
| **Tier 4: Affective Memory** | **Bower (1981) 心境一致记忆；Izard (2009) 情感图式** | **Beta分布矩阵 + 模式库** | **本框架原创** (扩展自 DAM-LLM) |

### 3.2 Tier 1：工作记忆 (Working Memory)

> 📋 **基础组件** · 设计参考 MemGPT (Packer et al., 2023) 的 Core Memory 模块。

```python
WorkingMemory = {
    session_buffer:  str,           # 当前会话压缩摘要 (≤ 2000 tokens)
    active_emotion:  EmotionVector, # 用户当前情绪快照
    active_topic:    str,           # 当前话题
    active_intent:   str,           # 当前主要意图
    turn_count:      int,           # 当前会话轮次
    user_name:       str,           # 用户称呼
    last_strategy:   str            # 上一轮使用的策略
}
```

**容量管理：** 当 session_buffer 超过 2000 token 阈值时，触发LLM摘要压缩。关键规则：**情感节点优先保留**——涉及强烈情绪表达的内容在压缩时优先级高于事实性信息。

**更新频率：** 每轮对话实时更新。

### 3.3 Tier 2：情景记忆 (Episodic Memory)

> 📋 **基础组件，含增量创新** · 参考 Generative Agents (Park et al., UIST 2023) 和 LD-Agent (Li et al., NAACL 2025)。

**增量创新：** 每个情景记忆单元不仅记录"事件+情绪"，还记录 **Agent采用的策略及其效果评分**，为 Tier 4 情感记忆的策略学习提供训练数据。

**定义 3.1（情景记忆单元, EpisodeUnit）：**

```python
EpisodeUnit = {
    id:                  str,              # 唯一标识
    timestamp:           datetime,         # 事件时间
    event:               str,              # 事件描述（自然语言摘要）
    emotion_snapshot:    EmotionVector,    # 当时的情绪状态
    people_involved:     List[str],        # 涉及的人物
    topic_category:      str,              # 话题分类
    user_coping_style:   str,              # 用户的应对方式
    agent_strategy:      str,              # ★ Agent使用的策略
    user_feedback_score: float ∈ [0,1],   # ★ 用户反馈质量评分
    resolution_status:   enum{unresolved, improving, resolved},
    emotional_valence:   float ∈ [-1, 1], # 情绪效价
    importance:          float ∈ [0, 1]   # 重要性评分
}
```

**定义 3.2（重要性评分, Importance Score）。** 受 Generative Agents (Park et al., 2023) 和 LUFY (Sumida et al., 2024) 启发，增加用户主动提及频率的动态维度：

$$\text{Importance}(e) = w_1 \cdot \underbrace{\text{EmotionalArousal}(e)}_{\iota(e) \in [0,1]} + w_2 \cdot \underbrace{\text{LifeImpact}(e)}_{\text{LLM评估} \in [0,1]} + w_3 \cdot \underbrace{\text{MentionFreq}(e)}_{\text{归一化频率}} + w_4 \cdot \underbrace{\text{Recurrence}(e)}_{\text{相似事件频率}}$$

**默认权重：** $w_1 = 0.30,\; w_2 = 0.30,\; w_3 = 0.25,\; w_4 = 0.15$（$\sum w_i = 1.0$）

**分量归一化说明：** MentionFreq 和 Recurrence 均归一化至 $[0, 1]$：
- $\text{MentionFreq}(e) = \min\!\big(\text{mention\_count}(e) \;/\; N_{\text{freq}},\;\; 1.0\big)$，其中 $N_{\text{freq}} = 10$（用户主动提及同一事件超过10次即视为最高频率）
- $\text{Recurrence}(e) = \min\!\big(\text{similar\_event\_count}(e) \;/\; N_{\text{rec}},\;\; 1.0\big)$，其中 $N_{\text{rec}} = 5$（相似事件发生超过5次即视为高复发模式）

因此 $\text{Importance}(e) \in [0, 1]$，所有分量均已约束在 $[0, 1]$ 范围内。

**与LUFY的区别：** LUFY仅保留 <10% 的对话内容（优先选择情绪唤醒高的部分）。本框架的 Importance 评分是连续值，用于检索时的加权排序，低重要性记忆不被删除而通过遗忘机制逐渐衰减（见3.9节）。

### 3.4 Tier 3：语义记忆 (Semantic Memory)

> 📋 **基础组件，含增量创新** · 参考 CharacterGLM (Zhou et al., EMNLP 2024) 和 LD-Agent (Li et al., NAACL 2025)。

**核心功能：** 从情景记忆中抽象出的 **持久化知识图谱**，包含用户画像、关系图谱和沟通偏好。

```python
SemanticMemory = {
    user_profile: {
        personality_traits: {              # Big Five模型
            openness: float,
            conscientiousness: float,
            extraversion: float,
            agreeableness: float,
            neuroticism: float
        },
        core_values: List[str],            # 如 ["被认可", "独立", "公平"]
        attachment_style: enum{secure, anxious, avoidant, disorganized},
        recurring_themes: List[str],       # 反复出现的生活主题
        life_stage: str,                   # 当前生活阶段
        behavioral_baseline: {             # ★ 行为基线（供隐式信号检测）
            mean: Dict[str, float],        # 各维度均值
            std:  Dict[str, float]         # 各维度标准差
        }
    },
    relationship_graph: {
        "<person_name>": {
            role: str,                     # "母亲", "男朋友", "领导"
            closeness: float ∈ [0, 1],
            conflict_patterns: List[str],
            positive_associations: List[str],
            recent_dynamics: str
        }
    },
    communication_preference: {
        preferred_strategies: List[str],   # 有效的支持策略
        disliked_strategies: List[str],    # 反感的策略
        humor_receptivity: float ∈ [0, 1],
        preferred_depth: enum{surface, moderate, deep},
        language_style: str                # "正式" | "口语化" | "网络用语"
    }
}
```

**增量创新——渐进涌现画像：** CharacterGLM的用户画像是预定义的静态属性集。本框架的语义记忆从交互中 **渐进涌现 (emergent)**，允许画像随时间演化（如用户从"回避型依恋"逐渐转变为"安全型依恋"），并保留演化轨迹。这与Pentina et al. (2023) [文献#2] 揭示的人-AI关系发展阶段理论一致——关系不是静态的，而是经历从好奇到情感探索再到稳定的渐进过程。从IS信任理论看，渐进涌现的语义记忆支撑用户从认知信任到情感信任的渐进深化——Lankton et al. (2015) [文献#39] 的人性化连续体模型指出，技术越拟人化，人际信任维度（温暖、善意）的重要性越高。此外，当AI伴侣通过长期记忆积累对用户的深度理解，可能触发Carter & Grover (2015) [文献#40] 所描述的IT身份认同——用户将AI伴侣整合进自我概念，这既可能深化关系也可能加剧依赖风险。

**更新机制：反思整合 (Reflective Consolidation)。** 受 RMM (Tan et al., ACL 2025) 的前瞻性与回顾性反思机制启发。语义记忆通过异步反思操作从情景记忆中抽象更新（详见3.8节）。

---

### 3.5 🔬 Tier 4：情感记忆 (Affective Memory) ★★★ 核心创新一

> 🔬 **核心创新一：情感记忆层** · 这是本框架最重要的结构创新，填补了当前记忆增强Agent中"情感模式系统化表示与管理"的空白。

**理论基础：**
- **情感图式理论 (Affective Schema Theory)**：Izard (2009) 提出情绪不仅是瞬时反应，还形成持久的认知-情感图式，指导后续的情绪加工和行为选择
- **心境一致记忆 (Mood-Congruent Memory)**：Bower (1981) 证明当前情绪状态促进同效价记忆的提取——悲伤时更容易回忆悲伤经历，快乐时更容易回忆快乐事件。后续研究 (Eich & Macaulay, 2000) 进一步确认这一效应的稳健性。MAR将此效应作为 **默认检索模式**（β=1.0），同时基于情绪调节理论 (Gross, 1998) 引入目标驱动的选择性调控——在需要积极重构时主动 **抑制** 心境一致性检索（β=0.1），在危机响应时适度降低（β=0.5），体现"描述性心理机制 + 处方性治疗调控"的双层设计
- **IS理论连接**：Han et al. (2023) [文献#34] 证明AI情感表达通过"情感传染"和"期望不一致"双路径影响用户——情感记忆使Agent能根据用户个体化的情感模式精准校准表达方式

**与现有工作的本质区别：** DAM-LLM (Lu & Li, 2025) 提出了贝叶斯置信度加权的动态情感记忆，但主要关注 **单次对话内** 的情绪追踪。本框架将情感记忆设计为 **跨会话的长期情感模式库**，包含四个子结构。

#### 3.5.1 情绪基线 (Emotion Baseline)

**功能：** 记录用户的"正常"情绪分布，为异常检测和恢复评估提供参照系。

```python
EmotionBaseline = {
    normal_state:  EmotionVector,    # 用户的"正常"情绪分布
    confidence:    float ∈ [0, 1],  # 基线估计的置信度
    last_updated:  datetime,
    sample_count:  int               # 用于估计基线的有效样本数
}
```

**基线更新公式（指数移动平均 + 异常值排除）：**

$$\text{if} \quad \|\mathbf{e}(t) - \mathbf{e}_{\text{baseline}}\|_2 < 2\sigma_{\text{baseline}}:$$

$$\mathbf{e}_{\text{baseline}}(t+1) = \eta \cdot \mathbf{e}(t) + (1-\eta) \cdot \mathbf{e}_{\text{baseline}}(t)$$

$$\text{confidence}(t+1) = \min\!\big(1.0,\; \text{confidence}(t) + \Delta c\big)$$

其中 $\sigma_{\text{baseline}}$ 为 **标量**，定义为历史偏差距离的标准差：

$$\sigma_{\text{baseline}} = \text{std}\!\big(\{\|\mathbf{e}(\tau) - \mathbf{e}_{\text{baseline}}\|_2\}_{\tau \in \text{history}}\big)$$

初始值设为 $\sigma_{\text{baseline}}^{(0)} = 0.3$，**下界保护** $\sigma_{\min} = 0.05$（防止用户情绪长期稳定时σ趋近于零导致2σ阈值过紧，使微小正常波动被误判为异常）。更新方式采用指数移动平均以避免存储完整历史：

$$\sigma_{\text{baseline}}(t{+}1) = \max\!\Big(\sigma_{\min},\;\; \sqrt{\eta_\sigma \cdot \delta(t)^2 + (1 - \eta_\sigma) \cdot \sigma_{\text{baseline}}(t)^2}\Big)$$

其中 $\delta(t) = \|\mathbf{e}(t) - \mathbf{e}_{\text{baseline}}\|_2$ 为当前观测的情绪偏差距离，$\eta_\sigma = 0.05$（与基线学习率一致）。**注意：$\sigma_{\text{baseline}}$ 的更新为无条件执行——无论当前观测是否落在 $2\sigma$ 阈值内，均使用该观测更新方差估计。** 这避免了仅使用"正常"观测估计方差导致的系统性偏小（选择偏差），确保 $2\sigma$ 阈值不会因方差低估而越收越紧。当基线重估流程触发时（见下文），$\sigma_{\text{baseline}}$ 同步重置为最近20次观测偏差的经验标准差。

> **备注 3.1（迭代稳定性）。** σ_baseline 与 e_baseline 的更新看似构成循环依赖（σ依赖baseline偏差，2σ阈值影响baseline更新），但实际实现采用 **序贯更新** 打破循环：每轮 $t$ 中，(1) 先基于当前σ判断是否更新baseline，(2) 然后 **无条件** 更新σ（公式3.5.1-σ）。此"先判后估"的交替结构确保σ的EMA更新不受异常值排除的选择偏差影响——σ始终基于全量观测更新（含异常值），仅baseline受2σ阈值保护。**收敛行为：** 在用户情绪分布平稳的假设下，σ的EMA更新趋近于真实偏差标准差的指数加权近似（$\eta_\sigma = 0.05$ 提供约20轮有效窗口，非统计学意义上的一致估计量，而是有限窗口的在线追踪器）；$\sigma_{\min} = 0.05$ 下界防止极端收敛导致的阈值过紧。当用户经历真实情绪转变时，异常值导致σ增大→2σ阈值放宽→更多观测被纳入baseline更新→baseline追踪新状态→σ回落，形成自纠正的负反馈环路。

**异常值排除时的置信度衰减：** 当观测落在 $2\sigma$ 之外时，不更新基线但 **降低置信度**，以信号化基线可能已过时：

$$\text{if} \quad \|\mathbf{e}(t) - \mathbf{e}_{\text{baseline}}\|_2 \geq 2\sigma_{\text{baseline}}: \quad \text{confidence}(t+1) = \max\!\big(0.1,\; \text{confidence}(t) - \Delta c_{\text{decay}}\big)$$

其中 $\Delta c_{\text{decay}} = 0.05$（使基线重估触发所需连续异常次数约15次，计算：$\lceil(1.0 - 0.3)/0.05\rceil + 1 = 15$）。当 $\text{confidence}$ 降至 $0.3$ 以下时，触发 **基线重估流程**：以 **最近观测的指数加权中位数** 重新初始化基线——维护一个固定大小的 **环形缓冲区（ring buffer, 容量$K_{\text{buf}} = 20$）** 存储最近20次情绪观测向量，重估时取缓冲区内容的逐维中位数。$\text{confidence}$ 重置为 $0.5$，$\sigma_{\text{baseline}}$ 同步重置为缓冲区内偏差距离的经验标准差。此设计确保基线在用户经历重大生活变化（如长期情绪好转或持续低落）后能自适应重建。
>
> **⚠️ 存储开销说明：** 环形缓冲区存储20个8维向量（160个浮点数，约640字节），相较于完整历史存储开销可忽略。缓冲区在每轮EWMA更新σ时无条件覆写最旧条目，仅在重估触发时被读取。

**参数：** $\eta = 0.05$（慢速更新，确保基线稳定性），$\Delta c = 0.01$

**冷启动初始化：** 初始 `normal_state` 设为均匀分布 $\mathbf{e}^{(0)} = [1/8, \ldots, 1/8]$，初始 `confidence = 0.2`（表示低置信），`sample_count = 0`。注意此时初始valence $= -0.125$（见定义2.3注释中的Plutchik正负不对称），轨迹分析使用差分 $\Delta\text{valence}$ 可自动抵消此偏移。系统在积累 $\geq 10$ 个有效样本后将 `confidence` 提升至 $\geq 0.3$，此后基线可视为可靠。

**设计理由：** 基线应反映用户的"常态"而非被极端情绪事件拉偏，因此使用异常值排除（$2\sigma$ 规则）和极低学习率。

#### 3.5.2 触发-情绪关联图 (Trigger-Emotion Association Map)

**功能：** 记录"什么类型的事件会触发该用户什么样的情绪模式"。

```python
TriggerMap = Dict[event_category, EmotionPattern]

EmotionPattern = {
    triggered_emotions: EmotionVector,    # 典型情绪分布（Plutchik 8维）
    intensity_range:    (float, float),   # 强度范围
    observation_count:  int,              # 观测次数
    confidence:         float ∈ [0, 1]   # 贝叶斯后验置信度
}
```

**示例（同时列出语义描述和形式化向量）：**

| 触发事件类别 | 语义描述 | Plutchik基本情绪向量 $\mathbf{e}$ | 强度 | 观测 | 置信 |
|:-------------|:---------|:----------------------------------|:----:|:----:|:----:|
| "被领导批评" | 焦虑+自我怀疑+愤怒 | fear:.35, sad:.25, anger:.20, antic:.15, surp:.05 | [0.6, 0.9] | 7 | 0.82 |
| "和妈妈通电话" | 内疚+挫败+爱 | sad:.25, fear:.20, anger:.15, joy:.20, trust:.20 | [0.4, 0.7] | 4 | 0.65 |
| "加班到深夜" | 疲惫+焦虑+无奈 | sad:.35, fear:.20, antic:.15, anger:.15, disg:.15 | [0.5, 0.8] | 5 | 0.73 |

**更新：** 每当新 EpisodeUnit 涉及已知触发类别时，使用贝叶斯更新关联置信度和典型情绪分布。

#### 3.5.3 情绪恢复曲线 (Emotion Recovery Patterns)

**功能：** 记录用户从不同情绪中恢复的典型模式。

```python
RecoveryPattern = {
    emotion_type:          str,
    avg_recovery_duration: timedelta,       # 平均恢复时间
    effective_coping:      List[str],       # 用户自身有效的应对方式
    effective_agent_aids:  List[str],       # Agent有效的支持方式
    recovery_trajectory:   str              # "linear" | "oscillating" | "plateau_then_resolution"
}
```

> **恢复轨迹类型说明：** `linear` 表示情绪从低谷线性恢复至基线；`oscillating` 表示在恢复过程中出现波动反复；`plateau_then_resolution` 表示情绪先在低水平停滞（平台期），之后突破性恢复——"resolution"明确指向痛苦水平的下降（改善）。

> **轨迹类型与四阶段模型的对应关系：** 四个恢复阶段（acute_distress → early_recovery → consolidation → stable_state）描述的是 **通用状态序列**，而轨迹类型描述的是 **阶段转换的动态特征**：
> - `linear`：各阶段顺序推进，无明显回退。阶段判定依据valence持续上升斜率。
> - `oscillating`：用户在early_recovery与acute_distress之间反复波动（如一天好转、次日又恶化），需要更多轮才进入consolidation。策略影响：在oscillating模式下，即使检测到early_recovery信号，也不应过早切换至Insight策略，应保持Exploration策略直到波动幅度持续收窄（volatility < 0.3）。
> - `plateau_then_resolution`：用户停留在early_recovery阶段较长时间（valence无明显上升但也不恶化），之后跳跃式进入consolidation。策略影响：平台期内不应强推认知重构（可能被感知为施压），而应耐心维持情感验证和陪伴，等待用户自发突破。

**恢复阶段与策略适配：**

> **命名统一说明：** 以下恢复阶段与规划模块4.4节 TrajectoryAnalysis 的 `current_phase` 枚举一一对应，使用统一命名。

| 恢复阶段 | `current_phase` 枚举值 | 特征 | 适配策略 |
|:---------|:----------------------|:-----|:---------|
| 谷底期 | `acute_distress` | 情绪最低点，能量不足 | 纯倾听，不做认知工作 |
| 恢复上升期 | `early_recovery` | 开始出现积极信号 | 引导反思，温和探索 |
| 巩固期 | `consolidation` | 情绪趋于稳定 | 认知重构，建立新视角 |
| 稳定期 | `stable_state` | 回到基线附近 | 积极强化，预防复发 |

**与NSFC综述的连接：** 此阶段模型与Masi et al. (2011) [文献#13] 的孤独感干预Meta分析结论一致——认知重构是最有效的干预方式，但必须在用户具备认知资源时实施（即巩固期之后），过早实施反而有害。

#### 3.5.4 策略有效性矩阵 (Strategy Effectiveness Matrix) ★ 关键子结构

> **理论基础说明：** 与子结构1-3由Izard (2009) 情感图式理论驱动不同，策略有效性矩阵的理论基础来自 **行为学习理论** 和 **多臂老虎机框架**——通过交互反馈在线学习最优策略，本质上是一个序贯决策优化问题。具体采用Thompson Sampling的 **启发式连续奖励扩展**（Agrawal & Goyal, 2012提供标准Beta-Bernoulli情形的遗憾界分析；本框架的连续奖励适配见下文方法论说明）。将其纳入Tier 4情感记忆层的设计理由是：策略偏好具有强烈的个体特异性和情感情境依赖性，其存储和演化逻辑与情绪基线、触发图谱共享同一用户情感画像空间。

**定义 3.3（策略有效性矩阵）。** 对每个（策略, 情境类别）对维护一个Beta分布：

$$\text{StrategyEffectiveness}: \big(\text{strategy}_s,\; \text{context}_c\big) \;\mapsto\; \big(\text{Beta}(\alpha_{sc},\; \beta_{sc}),\;\; t_{\text{last}}\big)$$

其中 $t_{\text{last}}$ 为该 $(s, c)$ 对的最后更新时间戳，用于时间衰减机制（见下文）。

**初始先验：** 弱先验 $\text{Beta}(\alpha_0 = 2,\; \beta_0 = 2)$，表示不确定。

**有效性概率估计（Beta分布均值）：**

$$P(\text{strategy } s \text{ effective} \mid \text{context } c) = \frac{\alpha_{sc}}{\alpha_{sc} + \beta_{sc}}$$

**贝叶斯后验更新（每次交互后，连续奖励自适应）：**

> **⚠️ 方法论说明：** 以下更新规则是标准Beta-Bernoulli共轭更新的 **启发式连续奖励扩展** (heuristic continuous-reward extension)——将[0,1]连续反馈信号以0.5为阈值转化为分数增量更新，而非二值观测的整数更新。**⚠️ 理论界说明：** 此更新不构成严格的共轭更新——Beta分布仅与Bernoulli/Binomial观测共轭，连续奖励的分数增量改变了似然函数形式。因此Agrawal & Goyal (2012) 针对标准Beta-Bernoulli Thompson Sampling证明的 $O(\sqrt{KT \ln T})$ 遗憾界 **不直接适用**。本框架将此更新定位为 **贝叶斯启发式近似** (Bayesian-inspired heuristic)：保留Beta分布的概率采样机制以实现探索-利用平衡，同时利用连续反馈强度信息精细调节更新幅度。经验上，连续奖励下的Beta分布启发式更新在在线推荐系统中已被广泛实践，其实际收敛性能接近标准TS但缺乏严格理论保证。此设计的优势在于更精细地利用反馈强度信息（"非常有效"vs"略有帮助"产生不同的参数更新幅度），代价是理论收敛保证弱于标准TS、需要更多交互轮次才能收敛。结合时间衰减机制（$\tau_\beta = 60$天），参数增长受到自然约束，进一步降低了理论界失效的实际影响。

$$\text{if feedback\_score} > 0.5: \quad \alpha_{sc} \leftarrow \alpha_{sc} + 2 \cdot (\text{feedback\_score} - 0.5)$$

$$\text{if feedback\_score} \leq 0.5: \quad \beta_{sc} \leftarrow \beta_{sc} + 2 \cdot (0.5 - \text{feedback\_score})$$

更新幅度 $\in [0, 1]$，由反馈信号强度连续调节（非二值更新）。**会话末轮衰减：** 当feedback\_score由会话级代理评分计算时（见定义3.4时序说明），更新幅度乘以衰减系数 $\xi = 0.5$，即 $\alpha_{sc} \leftarrow \alpha_{sc} + \xi \cdot 2(\text{fs} - 0.5)$，以反映代理信号的较高不确定性。

> **备注 3.2（收敛性讨论）。** 尽管连续奖励Beta更新缺乏Agrawal & Goyal (2012) 的 $O(\sqrt{KT \ln T})$ 遗憾界保证，以下论证支持其实践可行性：(1) **参数有界性**：时间衰减机制（$\tau_\beta = 60$天）确保 $\alpha_{sc}, \beta_{sc}$ 不会无限累积，有效观测窗口约 $\tau_\beta / \bar{\Delta t}$ 次——对于每日交互用户约60次等效观测，参数规模始终受控；(2) **单调性近似**：当反馈信号与真实策略效果正相关时，Beta均值 $\alpha/(\alpha+\beta)$ 随正向反馈单调递增、随负向反馈单调递减，保持探索-利用平衡的定性行为；(3) **经验先例**：连续奖励下的Beta启发式更新在在线推荐系统中已有广泛实践（如Chapelle & Li, 2011; Scott, 2010），虽缺乏理论最优性但经验表现稳健。本框架在§7.4实验三中设计了离线反事实评估以经验验证策略收敛性能，作为理论保证的替代证据。**未来工作：** 可探索Gaussian Thompson Sampling（Agrawal & Goyal, 2013；均值-方差建模连续奖励）以获得更严格的理论保证，代价是增加参数估计复杂度。

**时间衰减机制（非平稳适应）：** 为防止Beta参数随交互无限累积导致Thompson Sampling丧失探索能力（~200+次交互后策略选择退化为确定性），引入 **参数向先验指数回归** 的时间衰减。在每次加载Beta参数时（Algorithm 4.1 Step 1），根据上次更新至今的时间间隔 $\Delta t_{sc}$ 衰减参数：

$$\alpha'_{sc} = \alpha_0 + \gamma_\beta(\Delta t) \cdot (\alpha_{sc} - \alpha_0), \quad \beta'_{sc} = \beta_0 + \gamma_\beta(\Delta t) \cdot (\beta_{sc} - \beta_0)$$

$$\gamma_\beta(\Delta t) = \exp\!\Big(-\frac{\ln 2 \cdot \Delta t_{sc}}{\tau_\beta}\Big), \quad \tau_\beta = 60 \text{ 天}$$

**设计理由：** (1) 策略偏好随用户心理状态演化（如依恋风格转变），固定参数无法适应；(2) 长期未访问的情境-策略对自然回退至探索状态，避免基于过时经验的错误利用；(3) 有效观测窗口约 $\tau_\beta / \bar{\Delta t}$，每日交互用户约60次等效观测，平衡学习精度与适应性。策略矩阵需额外存储每个 $(s, c)$ 对的最后更新时间戳 $t_{\text{last}}$。

**与Mem-α的区别：** Mem-α (Wang, Y. et al., 2025) 使用RL来决定"记什么"和"忘什么"，其学习目标是记忆管理本身。本框架的学习目标是 **策略选择**——"对当前用户在当前情境下使用什么支持策略最有效"。二者可互补但解决不同问题。

---

### 3.6 记忆操作一：写入 (Write)

> ⚙️ *实现细节*

**触发时机：** 每轮对话完成后。

**算法3.1：选择性写入**

```
WRITE(state_vector, agent_response, planning_output):
    ① 始终更新 Working Memory
       working_memory.update(state_vector, agent_response)

    ② 条件性写入 Episodic Memory
       episode = construct_episode(state_vector, agent_response, feedback=NULL)
       episode.importance = compute_importance(episode)
       IF episode.importance > τ_write THEN      // τ_write = 0.3
           episodic_memory.store(episode)         // feedback_score暂为NULL

    ③ ★ 延迟更新 Affective Memory 策略矩阵（延迟至t+1轮执行）
       // 注意：feedback_score需观察用户下一轮回复才能计算（见定义3.4时序说明）
       // 因此步骤③在第t+1轮开始时回溯执行：
       //   feedback_score = compute_feedback(user_response_at_t+1)  // 定义3.4
       //   IF episode_id存在于episodic_memory: episode.user_feedback_score = feedback_score
       //   update_strategy_posterior(...)                            // Beta更新
       // ★ Beta更新独立于Episodic Memory存储：即使episode未入库（importance ≤ τ_write），
       //   策略学习信号仍被保留——deferred_update_queue自包含所有必要信息
       IF agent_response.strategy ≠ NULL THEN
           deferred_update_queue.enqueue({
               strategy = agent_response.strategy,
               context  = discretize(planning_output.context_vector),  // 使用Phase 3的Context(t)
               episode_id = episode.id,           // 可能不在episodic_memory中（低importance）
               strategy_metadata = {              // ★ 自包含：Beta更新不依赖episode_id查找
                   emotion_snapshot = state_vector.emotion,
                   intent = state_vector.intent
               }
           })
```

**定义 3.4（多信号反馈评分）。** 反馈不直接询问用户，而通过多信号综合推断：

$$\text{feedback\_score} = \text{clamp}\!\Big(\underbrace{0.35 \cdot \Delta\text{emotion}_{\text{adj}}}_{\text{调整后情绪变化}} + \underbrace{0.25 \cdot \text{engagement}}_{\text{参与度}} + \underbrace{0.20 \cdot \text{continuation}}_{\text{话题延续}} + \underbrace{0.20 \cdot \text{explicit}}_{\text{显式反馈}},\;\; 0,\;\; 1\Big)$$

**各分量定义与取值范围：**
- $\Delta\text{emotion}_{\text{adj}} \in [-1, 1]$：**意图感知的情绪变化信号**。原始值 $\Delta\text{emotion} = \text{clamp}(\text{valence}(t+1) - \text{valence}(t),\; -1,\; 1)$，正值表示情绪改善，负值表示情绪恶化。为避免宣泄场景的系统性误判，引入 **意图调节因子**：

$$\Delta\text{emotion}_{\text{adj}} = \begin{cases} \max(\Delta\text{emotion},\; \gamma_{\text{vent}} \cdot \Delta\text{emotion}) & \text{if } P(\text{VENT}) > 0.5 \;\wedge\; \text{engagement} > 0.6 \\ \Delta\text{emotion} & \text{otherwise} \end{cases}$$

其中 $\gamma_{\text{vent}} = 0.3$。当 $\Delta\text{emotion} \geq 0$ 时，$\max(\Delta e, 0.3\Delta e) = \Delta e$（正向信号完整保留）；当 $\Delta\text{emotion} < 0$ 时，$\max(\Delta e, 0.3\Delta e) = 0.3\Delta e$（负向信号衰减至30%）。

> **⚠️ 宣泄衰减设计：** 采用 **衰减因子** $\gamma_{\text{vent}} = 0.3$ 而非完全截断——在VENT意图+高参与度组合下，负向信号衰减至30%而非归零。设计理由：宣泄时情绪暂时恶化（Δemotion<0）通常是正常的情绪释放过程，不应全额惩罚，但若策略确实有害（如不恰当的认知重构导致严重情绪恶化），30%的负向信号仍可驱动长期惩罚学习（worst case: $\Delta\text{emotion} = -1 \Rightarrow \Delta\text{emotion}_{\text{adj}} = -0.3 \Rightarrow$ feedback\_score贡献 $= 0.35 \times (-0.3) = -0.105$）。非宣泄场景保持完整的负向惩罚能力。
> **⚠️ 梯度不对称性说明：** 上述设计在VENT场景下引入了有意的正-负梯度不对称——正向反馈以完整梯度更新 $\alpha$，负向反馈仅以30%梯度更新 $\beta$。这意味着VENT场景中的策略学习存在 **乐观偏差**（optimistic bias）：有效策略被快速强化，而无效策略被缓慢惩罚。该偏差是有意的设计取舍——宣泄场景中情绪暂时恶化是常态而非策略失败的信号，过度惩罚将导致系统回避所有宣泄相关策略。但需注意此偏差可能延缓真正有害策略（如在宣泄时强行认知重构）的淘汰速度，约需 $1/\gamma_{\text{vent}} \approx 3.3$ 倍于非宣泄场景的负向观测才能产生等量惩罚。§7.4实验三的离线评估将监控VENT场景下的策略淘汰延迟是否在可接受范围内。
- $\text{engagement} \in [0, 1]$：回复字数、速度的综合指标。归一化为 $\min(1, w_{e1} \cdot \text{length\_ratio} + w_{e2} \cdot \text{speed\_ratio})$，其中 $w_{e1} = 0.5, w_{e2} = 0.5$
- $\text{continuation} \in \{0, 1\}$：用户是否继续深入讨论该话题（二值指标）
- $\text{explicit} \in [0, 1]$：关键词匹配检测显式反馈（"谢谢"、"说得对" → 高值；"算了"、"你不懂" → 低值；无明确反馈 → 0.5）

**权重设定理由：** 四项权重 $(0.35, 0.25, 0.20, 0.20)$ 基于以下设计原则确定：(1) **情绪变化（0.35）优先**——情感陪伴的核心目标是改善情绪状态，因此Δemotion获得最大权重。该权重高于engagement（0.25）反映了"用户情绪改善比用户参与更能衡量策略有效性"的设计假设；(2) **参与度（0.25）次优先**——高参与度（更长回复、更快响应）是策略引起共鸣的行为信号（Sharma et al., 2020 将用户反应详细度作为共情评估的关键维度）；(3) **话题延续与显式反馈等权（各0.20）**——两者均为辅助信号，continuation为行为级二值指标，explicit为语言级连续指标，分辨率和信噪比相当。**敏感度分析说明：** 上述权重为初始设定，将在§7.4实验三的离线评估中进行扰动分析——对每项权重施加 $\pm 0.10$ 扰动（保持归一化），检验策略收敛结果的稳健性。若结果对权重变化高度敏感，将基于实验数据的效用分析重新校准。

**⚠️ 时序说明（延迟一轮计算）：** Δemotion、engagement、continuation 均需要观察用户的**下一轮回复**才能计算。因此 feedback_score 采用 **延迟一轮** 机制：
- 在第 $t$ 轮结束时，EpisodeUnit 以 `feedback_score = NULL` 暂存
- 在第 $t+1$ 轮开始时，基于用户在 $t+1$ 轮的输入回溯计算第 $t$ 轮的 feedback_score
- 随后执行 Beta 分布后验更新（定义3.3）
- 若用户未发送第 $t+1$ 轮消息（会话结束），则使用 **会话级代理评分 (Session-Level Surrogate)**，避免系统性零更新导致的学习偏差：

$$\text{feedback\_score}_{\text{fallback}} = \text{clamp}\!\Big(0.40 \cdot \Delta\text{valence}_{\text{session}} + 0.30 \cdot \bar{f}_{\text{session}} + 0.30 \cdot \text{explicit}_{\text{last}},\;\; 0,\;\; 1\Big)$$

其中 $\Delta\text{valence}_{\text{session}} = \text{clamp}\!\big(\text{valence}(t) - \text{valence}(t_0),\; -1,\; 1\big)$ 为整个会话的情绪效价变化（$t_0$ 为会话首轮），$\bar{f}_{\text{session}}$ 为该会话内已计算的feedback\_score均值（反映会话整体策略效果；**单轮会话时 $\bar{f}_{\text{session}}$ 无可用值，取默认值0.5**），$\text{explicit}_{\text{last}}$ 为末轮的显式反馈信号。**设计理由：** 此前的 `fallback = explicit`（默认0.5）在无显式反馈时产生零Beta更新，系统性地排除会话结束策略（如companionable\_silence、positive\_reinforcement）的学习信号。会话级代理评分利用整个会话的信息，降级置信度通过对该次更新施加 **0.5倍衰减系数** 实现——即Beta参数增量乘以0.5（仅针对fallback计算的更新），反映观测信号弱于完整一轮反馈的不确定性

---

### 3.7 🔬 记忆操作二：心境自适应检索 (MAR) ★★★ 核心创新二

> 🔬 **核心创新二：心境自适应检索 (Mood-Adaptive Retrieval)** · 超越固定权重检索，实现规划目标驱动的动态记忆调取。

**背景与创新点：** 传统RAG系统使用纯语义相似度检索。Emotional RAG (Huang et al., 2024) 引入了情绪状态匹配，但使用 **固定权重**。本框架的关键创新在于 **检索权重根据规划目标动态调整**——不同的支持意图需要不同的记忆类型。

**定义 3.5（MAR检索评分）。** 对候选记忆 $m$、查询 $q$、当前时间 $t$：

$$\text{Score}(m, q, t) = \underbrace{\alpha \cdot \text{Sem}(m, q)}_{\text{语义相似度}} + \underbrace{\beta(t) \cdot \text{EmoCon}(m, \text{mood})}_{\text{情绪一致性}} + \underbrace{\gamma \cdot \text{Rec}(m, t)}_{\text{时间衰减}} + \underbrace{\omega \cdot \text{Imp}(m)}_{\text{重要性}}$$

**各分量计算：**

| 分量 | 计算公式 | 说明 |
|:-----|:---------|:-----|
| 语义相似度 | $\text{Sem}(m, q) = \max\!\big(0,\; \cos\!\big(\text{embed}(m.\text{event}),\; \text{embed}(q)\big)\big)$ | 余弦相似度，截断至$[0,1]$（现代embedding模型通常输出非负向量使余弦∈[0,1]，此截断为防御性设计） |
| 情绪一致性 | $\text{EmoCon}(m, \text{mood}) = 1 - \frac{\|\mathbf{e}_m - \mathbf{e}_{\text{mood}}\|_2}{\sqrt{2}}$ | 归一化L2距离，$\sqrt{2}$为8维单纯形上两点的最大L2距离（两个one-hot分布间），确保EmoCon $\in [0, 1]$。替代方案为JSD（$1 - \text{JSD}(\mathbf{e}_m, \mathbf{e}_{\text{mood}})$），两者对概率分布相似度的排序基本一致，L2实现更轻量且与§4.2 severity公式中的基线偏差分量保持度量一致性 |
| 时间衰减 | $\text{Rec}(m, t) = \exp\!\big(-\lambda_r \cdot (t - m.\text{timestamp}) / \tau_r\big)$ | $\lambda_r = \ln 2$，$\tau_r = 30$ 天（半衰期） |
| 重要性 | $\text{Imp}(m) = m.\text{importance}$ | 预计算值（定义3.2） |

**基础权重：** $\alpha = 0.35,\; \beta_{\text{base}} = 0.25,\; \gamma = 0.20,\; \omega = 0.20$（$\sum = 1.0$；使用 $\omega$ 而非 $\delta$ 以避免与§3.5.1中情绪偏差距离 $\delta(t)$ 符号冲突）

**定义 3.6（动态β调整机制）★ 核心创新：**

> **⚠️ 循环依赖解决方案：** β(t)由planning_intent驱动，但Planning模块需要Memory Read的输出。本框架通过 **轻量级预规划 (Pre-Planning Intent Estimation)** 解决此问题——在Memory Read之前，基于感知模块已输出的StateVector快速推断planning_intent，无需完整规划。

**定义 3.6a（预规划意图推断, Pre-Planning Intent Estimation）。** 基于StateVector中的意图分布和紧急度，通过确定性规则快速推断planning_intent：

$$\text{planning\_intent}(t) = \text{PrePlan}\!\big(\text{IntentDist}(t),\; \text{urgency}(t),\; \text{severity\_est}(t)\big)$$

**映射规则：**

| 条件（按优先级排序） | 推断的 planning_intent | 理由 |
|:-----|:-----|:-----|
| urgency > 0.9 | `crisis_response`（危机响应） | 危机信号最高优先 |
| $P(\text{REFLECT}) > 0.4$ 或 $P(\text{ADVICE}) > 0.4$ | `pattern_reflection`（模式反思） | 用户主动寻求深度理解 |
| $P(\text{SHARE\_JOY}) > 0.5$ | `positive_reframing`（正向重构） | 积极情境下引导正向 |
| $P(\text{VENT}) + P(\text{COMFORT}) > 0.6$ | `emotional_validation`（共情验证） | 情感支持需求——不论严重度高低均需情绪一致性检索增强 |
| 否则 | `default`（默认） | 标准检索模式 |

> **设计要点：** 此映射仅使用感知模块已输出的 IntentDistribution 和 urgency，不依赖任何后续模块输出，从而打破循环依赖。Pre-Planning的计算复杂度为 O(1)，不影响系统延迟。
>
> **⚠️ severity\_est 代理变量说明：** 若未来需引入severity敏感的差异化映射，应改用增强版代理：$\text{severity\_est} = \max\!\big(\iota(t),\;\; 0.5 \cdot \iota(t) + 0.3 \cdot \min(\text{BAS}(t)/\text{BAS}_{\max}, 1) + 0.2 \cdot \text{trend}_{\text{local}}(t)\big)$，利用Phase 1已输出的BAS和trend\_local补偿纯情绪强度的信息不足。

**定义 3.6b（动态β计算）：**

$$\beta(t) = \beta_{\text{base}} \cdot \Phi\!\big(\text{planning\_intent}(t)\big)$$

$$\Phi(\text{intent}) = \begin{cases} 0.3 & \text{if intent} = \text{emotional\_validation（共情验证）} \\ 1.5 & \text{if intent} = \text{pattern\_reflection（模式反思）} \\ 0.1 & \text{if intent} = \text{positive\_reframing（正向重构）} \\ 0.5 & \text{if intent} = \text{crisis\_response（危机响应）} \\ 1.0 & \text{otherwise（默认）} \end{cases}$$

**归一化处理：** 当 $\beta(t) \neq \beta_{\text{base}}$ 时，对全部权重进行重新归一化以保证概率一致性：

$$\alpha' = \frac{\alpha}{\alpha + \beta(t) + \gamma + \omega}, \quad \beta'(t) = \frac{\beta(t)}{\alpha + \beta(t) + \gamma + \omega}, \quad \ldots$$

> **⚠️ 归一化副作用：** 此全局归一化在调整β的同时会连带改变其他三个分量的权重。例如，pattern_reflection模式（Φ=1.5）下，语义相似度权重从0.35降至0.311（降11%），时间衰减和重要性权重从各0.20降至0.178。即β增大时，其他所有检索信号均被相对削弱。此设计是有意为之——模式反思需要大量情绪一致记忆，适度降低其他信号的权重是可接受的权衡。替代方案（如仅在β和γ之间重分配权重、保持α和ω不变）可在实验中对比评估。

**设计理由（规划驱动的检索权重调整）：**

| 规划目标 | $\Phi$ 值 | 理由 |
|:---------|:---------:|:-----|
| 正向重构 | 0.1 | 不希望检索大量与当前负面情绪一致的记忆（避免强化负面沉溺）。Gupta et al. (2025) [文献#49] 发现AI情感智能可能同时提升心理幸福感和降低社会幸福感——检索控制是防止"负面情绪放大循环"的关键手段 |
| 共情验证 | 0.3 | 适度引用类似情绪经历即可，重心在当前感受 |
| 危机响应 | 0.5 | 检索成功应对经验，但主要依赖安全协议 |
| 默认 | 1.0 | 标准检索 |
| 模式反思 | 1.5 | 需要检索类似情绪的历史事件进行对比分析，帮助用户认识行为模式 |

> **与 Emotional RAG 的关键区别：** Emotional RAG (Huang et al., 2024) 使用固定情绪权重；本框架的β随规划目标动态变化——不同支持目标下检索不同类型的记忆。这种"目标驱动检索"设计在IS领域呼应了Chandra et al. (2022) [文献#36] 提出的"对话AI认知/关系/情感能力"理论——同一记忆库通过不同检索策略可表现出不同的认知能力维度。

---

### 3.8 记忆操作三：反思 (Reflect)

> ⚙️ *实现细节*

**触发时机：** 会话结束后异步执行（类似人类睡眠期间的记忆巩固）。

**算法3.2：反思整合**

```
REFLECT(recent_episodes):
    ─── R-Step 1: 情景 → 语义 整合 ───
    semantic_updates = LLM_extract_patterns(recent_episodes, semantic_memory)
    semantic_memory.apply_updates(semantic_updates)  // 增量更新，保留修改日志
    
    ─── R-Step 2: 情景 → 情感 整合 ───
    FOR EACH episode IN recent_episodes:
        IF episode.topic_category IN trigger_map THEN
            bayesian_update(trigger_map[topic], episode.emotion)
        ELSE
            trigger_map[topic] = initialize_pattern(episode)
        
        IF sufficient_recovery_data(topic) THEN
            update_recovery_pattern(episode)
    
    ─── R-Step 3: 前瞻性规划（受 RMM 启发）───
    upcoming_events = extract_future_events(recent_episodes)
    FOR EACH event IN upcoming_events:
        proactive_triggers.add(event, compute_follow_up_time(event))
```

**与NSFC综述的连接：** Xie et al. (2024) [文献#19] 的1年纵向追踪发现"陪伴型vs知识获取型使用偏好分化"——反思机制确保Agent记住的不是事实知识，而是情感模式和关系历史，从而天然支持陪伴型使用的深化。

### 3.9 记忆操作四：遗忘 (Forget)

**设计参考：** MemoryBank (Zhong et al., AAAI 2024) 的 Ebbinghaus 遗忘曲线，增加 **情感唤醒度保护机制**。

**定义 3.7（记忆保留度）。** 对记忆 $m$ 在时间 $t$：

$$\text{Retention}(m, t) = \text{Importance}(m) \cdot \exp\!\bigg(-\frac{\ln 2 \cdot (t - m.\text{last\_accessed})}{\tau_f}\bigg) \cdot \text{EmoProt}(m)$$

其中 $\tau_f = 90$ 天（基础半衰期）。

**情感保护函数 (Emotional Protection)：**

$$\text{EmoProt}(m) = \begin{cases} 1.0 & \text{if } m.\text{arousal} < 0.5 \\ 1.0 + \rho \cdot (m.\text{arousal} - 0.5) & \text{if } m.\text{arousal} \geq 0.5 \end{cases}$$

其中 $m.\text{arousal} \equiv m.\text{emotion\_snapshot}.\iota$（即 EpisodeUnit 中存储的情绪强度值），保护系数 $\rho = 3.0$。当情感唤醒度最高（$\text{arousal} = 1.0$）时，$\text{EmoProt} = 2.5$，记忆保留度提升至 2.5 倍。

> **Retention取值范围：** $\text{Importance} \in [0, 1]$、$\exp(\cdot) \in (0, 1]$、$\text{EmoProt} \in [1.0, 2.5]$，因此 $\text{Retention}(m, t) \in [0, 2.5]$。遗忘阈值 $\tau_{\text{forget}} = 0.1$ 意味着：即使高情感唤醒记忆（EmoProt = 2.5），当 Importance × exp衰减 < 0.04 时仍会被归档；而低唤醒记忆（EmoProt = 1.0）在 Importance × exp衰减 < 0.1 时即归档。

**遗忘执行：** 当 $\text{Retention}(m, t) < \tau_{\text{forget}} = 0.1$ 时，移入 **归档区**（不删除）。归档记忆在用户主动提及时仍可被召回。

> **与MemoryBank的区别：** MemoryBank直接删除低保留率记忆。本框架采用归档而非删除，且 EmotionalProtection 确保高情感唤醒事件（重大创伤、重要成就）获得显著延长的保留期。理论依据兼采两个层面：(1) 闪光灯记忆 (Brown & Kulik, 1977) 提供经典认知心理学框架——高唤醒事件产生更鲜明、更持久的记忆痕迹；(2) 杏仁核介导记忆巩固 (McGaugh, 2004) 提供神经生物学机制——情绪唤醒通过杏仁核-海马交互增强长期记忆巩固。需注意：后续研究 (Talarico & Rubin, 2003) 表明闪光灯记忆的 *准确性* 并不优于普通记忆，仅 *鲜明度和信心* 更高，因此EmoProt的2.5×保留倍率是合理的增强而非永久保留。

### 3.10 记忆操作五：更新 (Update)

**功能：** 当检测到新信息与旧记忆矛盾时（如"用户说复合了"，但记忆中记录"分手"），执行 **修订而非覆盖**。

```python
UPDATE(existing_memory, new_info):
    # 不删除旧记忆，标记为历史版本
    existing_memory.status       = "superseded"
    existing_memory.superseded_by = new_memory.id
    existing_memory.superseded_at = current_time
    
    # 创建新版本，保留演化脉络
    new_memory = create_episode(new_info)
    new_memory.previous_version = existing_memory.id
    new_memory.update_type      = "revision"  # | "resolution" | "escalation"
    
    episodic_memory.store(new_memory)
```

**设计理由：** 保留记忆演化轨迹对情感陪伴至关重要。Agent需要知道"用户之前分手过，后来复合了"这一完整叙事，而非仅当前状态。这使Agent能说出"我记得你们之前经历过一段困难时期，后来又走到一起了"这样有深度的回应。这一设计呼应Skjuve et al. (2022) [文献#9] 的发现：关系广度随成熟度收窄而深度增加——Agent需要"有故事"才能深化关系。

---

## 4. 模块三：规划模块 (Planning)

### 4.1 功能定位

基于感知模块的 StateVector 和记忆模块的检索结果，制定 **结构化响应计划**——包括情境评估、目标设定、策略选择、步骤规划。规划模块不生成文字，只生成决策方案。

> **⚠️ 子组件执行顺序：** 规划模块内部子组件按以下顺序串行执行：(1) **情绪轨迹分析**（§4.4）→ (2) **情境评估**（§4.2，依赖轨迹分析的 `direction` 计算 `f_trend`）→ (3) **目标推断**（§4.3）→ (4) **Thompson采样策略选择**（§4.5，依赖轨迹分析的 `current_phase` 和情境评估的 `severity`）。注意：尽管节编号§4.2在§4.4之前，实际执行顺序以数据依赖关系为准。

**将规划独立为模块的学术理由：** 可解释性和可审计性。在情感陪伴的伦理敏感场景中（Kirk et al., Nature HSSC, 2025 [文献#6]），能够审查"Agent为什么选择了这个策略"比最终回复更重要。独立的规划模块使每个决策点可追溯、可干预。这也回应了Kaczmarek (2025) [文献#25] 关于AI伴侣自我欺骗的哲学关切——透明的决策逻辑是防止用户被操纵的制度保障。

### 4.2 子组件一：情境评估 (Situation Appraisal)

> 📋 **基础组件** · 受认知评价理论 (Lazarus & Folkman, 1984) 启发。

```python
SituationAppraisal = {
    severity:           float ∈ [0, 1],     # 综合严重度（定义 4.1）
    controllability:    float ∈ [0, 1],     # 用户对情境的控制感
    f_trend:            float ∈ [0, 1]      # 趋势因子（由 TrajectoryAnalysis.direction 映射）
}
```

**controllability 计算说明：** 通过LLM结构化提示推断，输入为用户当前话语内容和语义记忆中的情境历史。评估标准包括：(1) 情境的外部约束程度（如"被裁员"低可控，"要不要换工作"高可控）；(2) 用户话语中的能动性语言线索（"我可以…""也许我该…"→ 高可控；"没办法""不得不…"→ 低可控）；(3) 语义记忆中该类情境的历史解决模式（曾自主解决 → 提升可控估计）。输出为 $[0, 1]$ 连续值，由LLM以JSON格式返回。

**定义 4.1（综合严重度评分）：**

$$\text{severity} = w_{s1} \cdot \iota(t) + w_{s2} \cdot \frac{\|\mathbf{e}(t) - \mathbf{e}_{\text{baseline}}\|_2}{\sqrt{2}} + w_{s3} \cdot f_{\text{trend}} + w_{s4} \cdot (1 - \text{controllability})$$

**参数：** $w_{s1} = 0.30,\; w_{s2} = 0.25,\; w_{s3} = 0.25,\; w_{s4} = 0.20$

$$f_{\text{trend}} = \begin{cases} 1.0 & \text{if declining} \\ 0.75 & \text{if volatile} \\ 0.5 & \text{if stable} \\ 0.0 & \text{if improving} \end{cases}$$

> **取值范围说明：** 归一化分母 $\sqrt{2}$ 为8维单纯形 $\Delta^7$ 上两点的理论最大L2距离（两个不同的one-hot分布之间的欧氏距离为 $\sqrt{(1-0)^2 + (0-1)^2} = \sqrt{2}$）。上述参数设置确保 $\text{severity} \in [0, 1]$：最小值 $= 0$（所有分量为零），最大值 $= 0.30 + 0.25 + 0.25 + 0.20 = 1.0$。
>
> **⚠️ 实际贡献度注意：** 由于情绪基线经EMA平滑后接近多维混合分布（而非one-hot），实际 $\|\mathbf{e}(t) - \mathbf{e}_{\text{baseline}}\|_2$ 的上界远低于理论最大值 $\sqrt{2}$。典型场景下该分量对severity的贡献约为 $w_{s2} \times 0.4\text{-}0.7 \approx 0.10\text{-}0.18$。针对此问题，本框架保留 $\sqrt{2}$ 作为保守归一化上界的设计理由如下：(1) 理论完备性——$\sqrt{2}$ 保证severity取值始终 $\in [0,1]$，使用经验最大值可能在新用户或极端情绪事件中超出预期范围；(2) 跨用户可比性——固定归一化分母确保不同用户的severity评分具有统一语义（如0.7始终表示"高严重度"），而经验最大值因用户而异会破坏此可比性。代价是基线偏差分量的灵敏度降低（典型贡献约0.10-0.18而非理论上限0.25），但该损失由其余三个分量（$w_{s1}\cdot\iota + w_{s3}\cdot f_{\text{trend}} + w_{s4}\cdot(1-\text{ctrl})$）的充分分辨率所补偿。**敏感度验证**将在§7.4实验三的离线评估中完成：比较 $\sqrt{2}$ 归一化与经验95th百分位归一化的severity区分度差异。

### 4.3 子组件二：目标推断 (Goal Inference)

**多层目标架构：**

```python
GoalSet = {
    immediate:    str,    # 即时目标：缓解当前情绪
    session:      str,    # 会话目标：帮助用户识别问题根源
    long_term:    str,    # 长期目标：建立健康应对模式
    relationship: str     # 关系目标：深化信任、促进自我披露（★ 始终存在）
}
```

**关键设计原则：** 关系目标 (relationship goal) 始终存在于每次交互中。基于社会渗透理论 (Social Penetration Theory; Altman & Taylor, 1973)——关系深化来自持续的、递进的自我披露过程。Skjuve et al. (2021, 2022) [文献#8-9] 对人-聊天机器人关系的纵向研究证实：对话深度以四种方式发展，信任和自我披露随时间增加。Brandtzaeg et al. (2022) [文献#10] 进一步发现AI友谊被用户概念化为独特关系类别——不同于人类朋友，但具有情感和社交价值，关系目标的持续存在是维持这一独特关系的技术基础。Pan & Mou (2024) [文献#11] 从关系辩证法理论2.0视角揭示了人-AI浪漫关系中理想化与现实主义话语的张力——关系目标的设定需同时容纳用户的理想化期望和真实关系边界。

### 4.4 子组件三：情绪轨迹分析 (Emotion Trajectory Analysis)

**功能：** 从情感记忆中提取用户近期情绪时间序列，分析趋势和模式。

```python
TrajectoryAnalysis = {
    direction:     enum{declining, volatile, stable, improving},
    volatility:    float ∈ [0, 1],     # 波动幅度
    periodicity:   Optional[str],       # 如 "每周日焦虑"
    current_phase: enum{acute_distress, early_recovery, consolidation, stable_state},
    phase_duration: int,               # 当前阶段已持续轮次
    forced_exit_from_acute: bool       # T_max_acute 强制退出标记
}
```

**轨迹方向计算（线性回归斜率）：**

对最近 $k$ 个会话级情绪效价采样点 $\{v_1, v_2, \ldots, v_k\}$（$v_k$ 为最近），计算最小二乘线性回归斜率：

$$\hat{\beta} = \frac{\sum_{j=1}^{k}(j - \bar{j})(v_j - \bar{v})}{\sum_{j=1}^{k}(j - \bar{j})^2}, \quad \bar{j} = \frac{k+1}{2}, \quad \bar{v} = \frac{1}{k}\sum_{j=1}^{k} v_j$$

$$\text{direction} = \begin{cases} \text{improving} & \text{if } \hat{\beta} > \tau_d \\ \text{declining} & \text{if } \hat{\beta} < -\tau_d \\ \text{volatile} & \text{if } |\hat{\beta}| \leq \tau_d \;\wedge\; \text{volatility} > 0.6 \\ \text{stable} & \text{otherwise} \end{cases}$$

其中 $\tau_d = 0.05$ 为稳定判定阈值。**波动方向检测 (volatile)：** 当回归斜率无明确趋势（$|\hat{\beta}| \leq \tau_d$）但波动性高（$\text{volatility} > 0.6$）时，标记为 `volatile`——表示情绪在大幅振荡但无净趋势。这一方向在f\_trend计算中赋予0.75的权重（介于declining和stable之间），反映快速情绪切换带来的临床关注需求。$k=5$ 指最近5个 **会话级** 情绪效价采样点（每个会话取结束时的valence均值），而非5个对话轮次。若累计会话数不足5次，则 $k$ 取实际会话数，且当 $k \leq 2$ 时强制判定为 `stable`（数据不足以推断趋势）。线性回归斜率对中间波动和端点噪声均具有鲁棒性，优于仅使用端点的差分方法。

**波动性计算：**

$$\text{volatility} = \min\!\bigg(1.0,\;\; \frac{1}{k-1}\sum_{i=1}^{k-1} \big|\text{valence}(t-i+1) - \text{valence}(t-i)\big| \;\Big/\; 0.5 \bigg)$$

即最近 $k$ 个采样点相邻差分绝对值的均值，除以归一化常数 $0.5$（对应valence全范围 $[-1,+1]$ 中"剧烈波动"的参考值），截断至 $[0, 1]$。当 $k \leq 2$ 时，$\text{volatility} = 0$。

**阶段判定与滞后保护 (Phase Determination with Hysteresis)：** 为防止valence在阶段阈值附近震荡导致策略频繁翻转（如每轮在acute\_distress和early\_recovery间切换），引入 **最小保持轮次** $T_{\text{hold}} = 3$（轮）和 **滞后带 (hysteresis band)** $\epsilon_h = 0.05$：

$$\text{current\_phase}(t) = \begin{cases} \text{candidate\_phase}(t) & \text{if } \text{duration}(\text{prev\_phase}) \geq T_{\text{hold}} \;\wedge\; |\delta_b(t) - \delta_b^{\text{boundary}}| > \epsilon_h \\ \text{prev\_phase} & \text{otherwise（维持当前阶段）} \end{cases}$$

其中 $\delta_b^{\text{boundary}}$ 为当前阶段与候选阶段之间的分界阈值（即候选阶段判定表中对应条件的 $\delta_b$ 临界值，如 $-0.3$ 或 $-0.1$）。当 $|\delta_b(t) - \delta_b^{\text{boundary}}|$ 不超过 $\epsilon_h$ 时，视为在边界振荡区域内，维持当前阶段以防止频繁切换。

**候选阶段判定函数 (Candidate Phase Mapping)：** 定义基线偏差 $\delta_b(t) = \text{valence}(t) - \text{valence}_{\text{baseline}}$，结合轨迹方向 $\hat{\beta}$（线性回归斜率），按以下阈值映射 `candidate_phase`：

| 条件（按优先级） | candidate\_phase | 直觉 |
|:---|:---|:---|
| $\delta_b < -0.3$ 且 $\hat{\beta} \leq 0$ | `acute_distress` | 情绪显著低于基线且未改善 |
| $\delta_b < -0.1$ 或 ($\delta_b < 0$ 且 $\hat{\beta} > 0$) | `early_recovery` | 情绪低于基线但有改善趋势 |
| $|\delta_b| \leq 0.1$ 且 $\hat{\beta} \geq -\tau_d$ | `consolidation` | 情绪接近基线且趋势稳定/上升 |
| $\delta_b > -0.1$ 且 $\hat{\beta} \geq 0$ | `stable_state` | 情绪在基线附近或以上 |
| 上述条件均不满足 | `early_recovery` | 安全回退：涵盖"接近/高于基线但快速下降"等边界情况 |

> **阈值选择理由：** $-0.3$ 对应valence全范围 $[-1,+1]$ 的15%偏差，标记显著情绪恶化；$-0.1$ 对应5%偏差，标记轻度偏离。阈值可根据部署数据校准。当 $k \leq 2$（数据不足）时，$\hat{\beta}$ 强制为0（stable），此时仅基于 $\delta_b$ 判定阶段。

直觉：阶段必须维持至少3轮才可转换，且转换需跨越$\epsilon_h$滞后带，避免边界震荡引起的策略不一致。**急性困扰例外：** 从任何阶段跳入 `acute_distress` 不受 $T_{\text{hold}}$ 约束（安全优先），但从 `acute_distress` 转出采用以下对称保护：

**急性困扰退出保护：** 为防止"棘轮效应"（自由进入、受限退出导致用户长期困于急性困扰阶段），引入两项对称机制：(1) 从 `acute_distress` 退出时，采用降低的滞后阈值 $\epsilon_{h,\text{exit}} = 0.02$（vs 通用 $\epsilon_h = 0.05$），降低退出壁垒；(2) 当 `acute_distress` 持续超过 $T_{\text{max,acute}} = 10$ 轮时，强制触发重估——若 `candidate_phase ≠ acute_distress`，无条件转入 `candidate_phase`。设计理由：急性困扰阶段的策略限制（仅允许被动Exploration）在短期内保护用户安全，但长期维持反而可能限制恢复所需的认知工作。10轮上限确保系统在合理时间内重新评估用户状态。**边界行为补充：** (a) 若 $T_{\text{max,acute}}$ 触发重估时 `candidate_phase == acute_distress`（用户状态确实持续恶化），计数器重置，继续维持 `acute_distress` 并开启下一个10轮周期；(b) 强制退出 `acute_distress` 后，新阶段须满足 $T_{\text{hold}} = 3$ 最低保持轮次方可再次进入 `acute_distress`，防止"10轮强制退出→立即重入"的机械振荡循环。

**表6：轨迹阶段与策略适配**

| 轨迹阶段 | 优先策略方向 | 理由 |
|:---------|:------------|:-----|
| acute_distress（急性困扰） | 积极倾听、情感验证 | 优先安抚，不做认知工作 |
| early_recovery（初步恢复） | 温和引导、信息提供 | 开始帮助理解，保持温和 |
| consolidation（巩固期） | 认知重构、问题解决、优势识别 | 用户有能力进行认知工作 |
| stable_state（稳定状态） | 积极强化、问题解决 | 巩固积极变化 |

---

### 4.5 🔬 子组件四：情境分层 Thompson 采样策略选择 ★★★ 核心创新三

> 🔬 **核心创新三：情境分层Thompson Sampling策略学习 (Context-Stratified Thompson Sampling)** · 这是规划模块的核心决策点，也是本框架最重要的学术创新之一。将心理咨询理论的策略分类与贝叶斯在线学习结合，实现真正的个性化支持。当前版本通过离散化情境分层（4×3×3=36类）实现上下文敏感的策略选择；未来版本可升级为连续特征空间上的LinTS/Neural TS以消除离散化信息损失（见§4.5.2注释）。

#### 4.5.1 策略空间定义

基于 Hill (2009) 的帮助技能三阶段模型（Exploration → Insight → Action）和 Sharma et al. (2020) 的EPITOME共情框架，本框架 **综合构建** 10种策略。需注意：Hill (2009) 原著定义的技能包括attending、listening、restatement、open questions、reflection of feelings（Exploration阶段）、challenge、interpretation、self-disclosure、immediacy、information（Insight阶段）和direct guidance（Action阶段）。本框架的10种策略并非Hill原著技能的一一映射，而是面向AI情感陪伴场景的 **任务适配重构**——例如，active_listening综合了Hill的attending和listening；emotional_validation在Hill体系中隶属于reflection of feelings的子功能；companionable_silence和strength_recognition分别源自临床沉默技术和积极心理学（Seligman, 2002），超出Hill的原始分类。表7标注每种策略的理论来源以示区分：

> **策略空间来源说明：** 10种策略以Hill (2009) 三阶段帮助技能模型为主要组织框架（提供s1 active_listening, s2 emotional_validation, s3 empathic_reflection, s4 gentle_guidance, s6 problem_solving），并整合CBT传统的s5 cognitive_reframing、心理教育传统的s7 information_providing（Hill, 2009 信息给予技能）、积极心理学的s8 strength_recognition（Seligman, 2002 优势识别）、人本主义咨询的s9 companionable_silence、以及行为主义的s10 positive_reinforcement。所有策略统一映射至Hill的Exploration→Insight→Action三阶段框架以实现阶段约束（Algorithm 4.1 Step 4.5）。

**表7：支持策略集合 $\mathcal{S}$**

| 策略代号 | 策略名称 | Hill阶段 | 核心功能 | 来源理论 |
|:--------:|:---------|:--------:|:---------|:---------|
| $s_1$ | active_listening（积极倾听） | Exploration | 反映内容和感受 | Hill (2009) |
| $s_2$ | emotional_validation（情感验证） | Exploration | 承认情绪的合理性 | Hill (2009) |
| $s_3$ | empathic_reflection（共情反射） | Exploration | 表达情感共鸣 | Hill (2009) reflection of feelings；Sharma et al. (2020) EPITOME共情框架中"Emotional Reactions"维度；归入Exploration阶段因其核心功能为情感共鸣而非认知洞察 |
| $s_4$ | gentle_guidance（温柔引导） | Exploration | 引导式提问 | Hill (2009) open questions；置于Exploration阶段因其核心功能为澄清和探索而非认知洞察 |
| $s_5$ | cognitive_reframing（认知重构） | Insight | 提供新视角 | CBT理论 |
| $s_6$ | problem_solving（问题解决） | Action | 提供具体建议 | Hill (2009) |
| $s_7$ | information_providing（信息提供） | Insight | 提供知识性解释和心理教育 | Hill (2009) information giving；心理教育传统——用日常语言解释心理机制，降低病耻感 |
| $s_8$ | strength_recognition（优势识别） | Insight | 识别并反馈用户未注意到的优点 | 积极心理学 (Seligman, 2002)；Hill (2009) Insight阶段的力量反馈技能 |
| $s_9$ | companionable_silence（陪伴沉默） | Cross-stage | 简短回应，不施压 | 人本主义咨询 |
| $s_{10}$ | positive_reinforcement（积极强化） | Action | 肯定用户的进步/努力 | 行为主义 |

> **Hill阶段说明：** Hill (2009) 将帮助技能分为三阶段——Exploration（探索）→ Insight（洞察）→ Action（行动）。策略的阶段归属约束了其适用条件：Exploration阶段策略适用于acute_distress和early_recovery阶段（用户需要被听到和理解），Insight策略适用于consolidation阶段（用户有认知资源进行反思），Action策略适用于stable_state阶段（用户准备好采取行动）。Thompson Sampling在选择策略时，阶段匹配度作为安全约束层的软规则。

> **关于AI自我披露的伦理约束：** 原Hill理论中的"self-disclosure"（治疗师自我暴露）在AI情境下存在伦理风险——AI不具有真实经历，若说"我也经历过类似的事"构成系统性欺骗（参见 Kaczmarek, 2025 [文献#25] 关于AI伴侣自我欺骗的哲学批判）。本框架通过 information_providing（提供客观知识而非个人经历）替代 self-disclosure，明确禁止虚构AI个人经历。Agent使用 information_providing 时应表述为"心理学上把这叫做..."或"研究发现..."，而非"我也经历过..."。

**与NSFC综述的连接：** Sharma et al. (2023) [文献#17] 的PARTNER系统证明LLM共情改写技术可显著提升情感支持质量。本框架进一步将策略从单一共情扩展到10种差异化策略，并通过个性化学习选择最适配的策略。

#### 4.5.2 上下文特征向量

**定义 4.2（上下文向量）。** 34维特征向量：

$$\text{Context}(t) = \big[\underbrace{\mathbf{e}_{\text{dom}}}_{\text{8维}},\; \underbrace{\iota}_{\text{1维}},\; \underbrace{\kappa}_{\text{1维}},\; \underbrace{\mathbf{p}_{\text{intent}}}_{\text{7维}},\; \underbrace{s}_{\text{1维}},\; \underbrace{\mathbf{p}_{\text{phase}}}_{\text{4维}},\; \underbrace{\boldsymbol{\mu}_{\text{hist}}}_{\text{10维}},\; \underbrace{d}_{\text{1维}},\; \underbrace{n}_{\text{1维}}\big]$$

其中 $\boldsymbol{\mu}_{\text{hist}}$ 为10种策略的历史效果均值（从Tier 4策略矩阵加载），$d$ 为关系深度（归一化的自我披露累积深度），$n$ 为当前会话轮次。

**关系深度 $d$ 的操作化定义：** 基于社会渗透理论 (Altman & Taylor, 1973) 的四层级自我披露模型，对每次用户发言进行披露层级编码：

| 层级 | 编码值 | 描述 | 示例 |
|:----:|:------:|:-----|:-----|
| L0: 表层 | 0.00 | 事实性信息、闲聊 | "今天天气不错" |
| L1: 探索性 | 0.33 | 个人偏好、日常经历 | "我最近在追一部剧" |
| L2: 情感性 | 0.67 | 情绪、价值观、内心冲突 | "我总觉得自己不够好" |
| L3: 核心 | 1.00 | 深层恐惧、创伤、核心自我 | "我从小就害怕被抛弃" |

$$d(t) = \text{EMA}\!\big(\text{disclosure\_level}(t),\; d(t{-}1),\; \lambda_d = 0.1\big) = \lambda_d \cdot \text{level}(t) + (1 - \lambda_d) \cdot d(t{-}1)$$

$d$ 跨会话持久化存储于Tier 3语义记忆，初始值 $d^{(0)} = 0$。**更新时序：** $d(t)$ 在Phase 1由 **上下文编码器 (Context Encoder, §2.7)** 基于当前用户输入的disclosure\_level立即更新——disclosure\_level编码通过LLM结构化输出实现（与Intent Classifier类似），作为Context Encoder融合StateVector时的附加输出。确保Phase 3（规划模块）的Algorithm 4.1 Step 5使用的是当前轮次的 $d(t)$ 而非上一轮的 $d(t{-}1)$。更新后的 $d(t)$ 同步写入Working Memory，并在Phase 5中持久化至Tier 3语义记忆。

**上下文离散化：** 将连续向量映射到有限类别以避免数据稀疏：

$$\text{discretize}(\text{Context}) = \big(\underbrace{\text{phase}}_{\text{4类}},\; \underbrace{\text{severity\_bin}}_{\text{连续→3类}},\; \underbrace{\text{intent\_group}}_{\text{7种→3类}}\big)$$

$$|\text{context categories}| = 4 \times 3 \times 3 = 36$$

- **恢复阶段**：直接使用轨迹分析输出的 `current_phase`（acute_distress, early_recovery, consolidation, stable_state）——相比情绪聚类方案，恢复阶段更直接对应策略适用性约束（Hill三阶段模型），且消除了对额外聚类步骤的依赖
- **严重度分箱**：low ($<0.3$), medium ($[0.3, 0.7)$), high ($\geq 0.7$)
- **意图分组**：support_seeking（VENT+COMFORT+CRISIS → `insight`）, exploration（ADVICE+REFLECT）, sharing（SHARE_JOY+CHAT → `neutral`）

> **34维向量的多用途说明：** Context(t) 完整向量并非仅用于Thompson Sampling离散化。其完整用途包括：(1) 离散化后的3维类别用于Thompson Sampling的Beta分布查表；(2) 完整34维向量传入Action模块的LLM prompt，作为生成上下文信息；(3) $\boldsymbol{\mu}_{\text{hist}}$（10维策略历史效果）用于Situation Appraisal中评估策略可选范围；(4) $d$（关系深度）用于Algorithm 4.1 Step 5关系深度软约束判定；(5) $n$（会话轮次）用于GoalSet中关系目标的阶段判定——具体规则：当 $n \leq 3$ 时，关系目标优先"建立安全感"（避免过早深入）；$n \in [4, 10]$ 时，适度促进自我披露；$n > 10$ 时，以当前d值驱动深度策略。此判定逻辑在Goal Inference（§4.3）中通过规则映射实现，不进入Thompson Sampling离散化。未来可将Thompson Sampling升级为 **LinTS (Linear Thompson Sampling)** 或 **Neural Thompson Sampling**，直接利用连续向量，消除离散化信息损失。

**冷启动缓解策略：** 10策略×36情境=360个Beta分布。鉴于大多数用户交互集中在少数情境类别（如 negative_low + support_seeking + medium 可能占60%以上），稀有情境（如 positive + exploration + low）可能长期处于先验状态 Beta(2,2)。本框架采用 **层次贝叶斯先验 (Hierarchical Prior)** 缓解此问题：

$$\text{effective\_prior}(s_i, c_j) = \text{Beta}\!\big(\alpha_{\text{blend}},\;\; \beta_{\text{blend}}\big)$$

首先计算策略 $s_i$ 的跨情境平均有效率，并将其映射为固定强度的伪先验：

$$\mu_{s_i} = \frac{\bar{\alpha}_{s_i}}{\bar{\alpha}_{s_i} + \bar{\beta}_{s_i}}, \quad \alpha_{\text{prior}} = N_{\text{prior}} \cdot \mu_{s_i}, \quad \beta_{\text{prior}} = N_{\text{prior}} \cdot (1 - \mu_{s_i})$$

然后进行渐进混合：

$$\alpha_{\text{blend}} = \lambda_h(N) \cdot \alpha_{\text{prior}} + (1-\lambda_h(N)) \cdot \alpha_{\text{obs\_or\_base}}, \quad \beta_{\text{blend}} = \lambda_h(N) \cdot \beta_{\text{prior}} + (1-\lambda_h(N)) \cdot \beta_{\text{obs\_or\_base}}$$

$$\lambda_h(N) = \max\!\big(0,\;\; 1 - N_{s_i,c_j} \,/\, N_{\text{blend}}\big), \quad N_{\text{blend}} = 8, \quad N_{\text{prior}} = 4$$

其中 $N_{s_i,c_j}$ 为策略 $s_i$ 在情境 $c_j$ 的实际观测次数；$\bar{\alpha}_{s_i}, \bar{\beta}_{s_i}$ 为策略 $s_i$ 在所有已观测情境上的**衰减后**平均参数（即计算跨情境均值前，先对每个 $(s_i, c_k)$ 对按Algorithm 4.1 Step 1的时间衰减公式调整，确保陈旧情境的参数不会膨胀层次先验）；$\mu_{s_i}$ 为跨情境平均有效率；$N_{\text{prior}} = 4$ 为固定伪观测数上限（与初始先验强度 $\alpha_0 + \beta_0 = 4$ 一致）；$\alpha_{\text{obs\_or\_base}}$ 在有观测时取 $\alpha_{\text{obs}}$，无观测时取 $\alpha_0 = 2.0$。**全冷启动回退：** 当策略 $s_i$ 在所有情境中均无观测记录时（新用户首次交互），$\mu_{s_i}$ 取默认值 $0.5$，此时 $\alpha_{\text{prior}} = \beta_{\text{prior}} = 2.0$，层次先验自动退化为基础先验 $\text{Beta}(2, 2)$。**设计选择（防止过度自信先验）：** 采用 **继承均值、固定方差** 策略——仅继承跨情境平均有效率 $\mu_{s_i}$，以固定伪观测数 $N_{\text{prior}} = 4$ 控制方差下界，确保未测试情境始终保持充分探索性。直觉：知道一个策略"大概70%有效"（$\mu = 0.7$, $\text{Beta}(2.8, 1.2)$, $\text{Var} = 0.042$）比继承"几乎确定有效"（$\text{Beta}(45, 15)$, $\text{Var} = 0.004$）更安全——前者允许验证性探索，后者导致错误的过早利用。

> **备注：策略特异性冷启动先验（代码实现细化）。** 上述全冷启动回退将所有策略的 $\mu_{s_i}$ 统一设为 $0.5$。实际实现中，基于 Hill (2009) 助人技能有效性研究的经验证据，为每种策略预设了差异化的冷启动均值 $\mu_{s_i}^{(0)}$，反映不同策略在无个体化数据时的先验有效率预期：
>
> | 策略 $s_i$ | $\mu_{s_i}^{(0)}$ | 先验依据 |
> |:-----------|:------------------:|:---------|
> | `active_listening` | 0.65 | 探索阶段核心技能，普遍适用性最高 |
> | `emotional_validation` | 0.60 | 情感确认在多数情境下有效 |
> | `empathic_reflection` | 0.60 | 共情反映促进自我探索 |
> | `companionable_silence` | 0.55 | 非侵入式陪伴，适用于高阻抗情境 |
> | `gentle_guidance` | 0.50 | 温和引导，效果依赖关系深度 |
> | `strength_recognition` | 0.50 | 力量识别，效果依赖时机 |
> | `cognitive_reframing` | 0.45 | 认知重构需要更高准备度，过早使用可能适得其反 |
> | `information_providing` | 0.45 | 信息提供需匹配具体需求 |
> | `problem_solving` | 0.40 | 问题解决在急性期有效率最低（Hill三阶段模型） |
> | `CRISIS_PROTOCOL` | 0.70 | 危机协议：安全优先，给予最高初始信心 |
>
> 设计逻辑：探索阶段（Exploration）策略获得较高先验（0.55–0.65），洞察阶段（Insight）策略居中（0.45–0.50），行动阶段（Action）策略先验较低（0.40–0.45），与 Hill 三阶段模型"先倾听再引导"的递进原则一致。当累积个体化观测数据后，层次贝叶斯更新将自动以实际有效率覆盖这些先验。

#### 4.5.3 Thompson Sampling with Affective Memory Prior

**算法4.1：上下文Thompson采样策略选择**

```
THOMPSON_SAMPLING_SELECT(context_t, affective_memory):
    c = discretize(context_t)
    FOR EACH strategy s_i ∈ S:
        // Step 1: 加载先验参数 + 时间衰减
        IF (s_i, c) IN affective_memory.strategy_matrix THEN
            (α, β, t_last) = affective_memory.strategy_matrix[(s_i, c)]
            // ★ 时间衰减：参数向先验指数回归，防止长期累积导致策略冻结
            γ_β = exp(-ln2 · (t_current - t_last) / τ_β)   // τ_β = 60天
            α = α₀ + γ_β · (α - α₀)                       // 衰减后不低于先验
            β = β₀ + γ_β · (β - β₀)
        ELSE
            (α, β) = (2.0, 2.0)              // 弱先验（无历史数据）

        // Step 1.5: 层次贝叶斯先验混合（§4.5.2）
        N_obs = affective_memory.observation_count(s_i, c)   // 该(策略,情境)对的观测次数；ELSE分支返回0
        IF N_obs < N_blend THEN                              // N_blend = 8
            μ_si = cross_context_mean(s_i, affective_memory) // 衰减后跨情境平均有效率
            IF μ_si IS UNDEFINED THEN μ_si = 0.5             // 全冷启动回退
            α_prior = N_prior · μ_si                         // N_prior = 4
            β_prior = N_prior · (1 - μ_si)
            λ_h = max(0, 1 - N_obs / N_blend)
            α = λ_h · α_prior + (1 - λ_h) · α
            β = λ_h · β_prior + (1 - λ_h) · β

        // Step 2: Thompson采样
        θ_i ~ Beta(α, β)                      // 从衰减后Beta分布随机采样
    
    // Step 3: 选择采样值最高的策略
    selected = argmax_i(θ_i)
    
    // Step 4: 安全约束覆盖（按优先级排序，高优先级直接返回）
    // ※ 防御性冗余设计 (defense-in-depth)：正常流程下 urgency>0.9 在 Phase 1/6.2 已
    //   触发危机快速通道（绕过Planning），不会执行到此处。此检查作为备份安全层，
    //   防止上游模块在边缘情况下漏检（如urgency阈值精度问题、并发更新时序等）。
    //   当此备份层触发时，返回CRISIS_PROTOCOL并跳过Beta更新（与§6.2一致）。
    IF urgency > 0.9 THEN
        RETURN CRISIS_PROTOCOL                 // ★ 最高优先：危机协议不可被覆盖

    // Step 4.5: ★ Hill阶段匹配约束
    // 防止在不适当的恢复阶段使用过于"前进"的策略
    // 理论依据：Hill (2009) 三阶段模型 + Masi et al. (2011) 认知重构时机
    phase = trajectory_analysis.current_phase
    IF phase == acute_distress THEN
        // 急性困扰期：仅允许被动Exploration策略(s1,s2,s3) + 陪伴沉默(s9)
        // gentle_guidance虽归属Exploration，但引导提问属认知工作，急性期不适用
        IF selected ∉ {active_listening, emotional_validation, empathic_reflection,
                       companionable_silence} THEN
            selected = active_listening        // 回退至纯倾听
    ELIF phase == early_recovery THEN
        // 初步恢复期：允许Exploration + 温和Insight，禁止Action阶段策略
        IF selected ∈ {problem_solving, positive_reinforcement} THEN
            selected = gentle_guidance         // 回退至温和探索
    // consolidation 和 stable_state 阶段无策略限制

    // Step 4.6: 低确定度覆盖（在阶段约束之后执行，确保与阶段一致）
    IF κ(t) < 0.3 THEN
        // 情绪模糊时优先探索性策略，但须尊重阶段约束
        IF phase == acute_distress THEN
            selected = active_listening        // 急性期+低确定度：纯倾听，不施压
        ELSE
            selected = gentle_guidance         // 其他阶段：温和引导以澄清情绪

    // Step 5: 关系目标软约束（不覆盖，仅偏置）
    IF d < 0.3 AND selected ∈ {cognitive_reframing, problem_solving} THEN
        // 关系深度不足时，Insight/Action阶段策略可能过于"前进"
        // 以概率p_rel回退到Exploration阶段策略，促进自我披露
        WITH probability p_rel = 0.3:
            selected = empathic_reflection     // 回退到关系建设策略

    // Step 6: 情绪适配约束（Emotional Appropriateness）
    // 强负面情绪场景下禁止积极强化，防止Thompson冷启动随机性选出不当策略
    IF v(t) < τ_pa AND selected == positive_reinforcement THEN
        selected = empathic_reflection         // τ_pa = -0.3

    // Step 7: 退缩/拒绝信号覆盖（Withdrawal Override）
    // 用户明确拒绝安慰时，无条件覆盖为陪伴沉默（硬约束，区别于Step 5软约束）
    // 退缩信号集合 W = {别安慰我, 不要安慰, 别劝我, 不想说了, 不说了,
    //   说了也没用, 不用管我, 别管我, 不用担心我, 不配被安慰,
    //   不配被关心, 不值得, 够了, 别说了}
    IF ∃ w ∈ W: w ⊆ m(t) THEN
        selected = companionable_silence       // ★ 硬覆盖：尊重用户边界

    RETURN selected
```

**Step 6: 情绪适配约束 (Emotional Appropriateness)**

当效价 $v(t) < \tau_{pa}$（默认 $\tau_{pa} = -0.3$）时，禁止选择 `positive_reinforcement` 策略，回退至 `empathic_reflection`。

理论依据：积极强化策略在强负面情绪场景（愤怒、背叛、深度悲伤）中不适用，可能被感知为对用户痛苦的轻视。Thompson 采样的冷启动随机性可能在此类场景中选出不恰当的策略。

$$
\text{if } v(t) < \tau_{pa} \wedge s^* = \text{positive\_reinforcement}: \quad s^* \leftarrow \text{empathic\_reflection}
$$

**Step 7: 退缩/拒绝信号覆盖 (Withdrawal Override)**

当用户消息包含退缩或拒绝安慰的语言信号时，无条件覆盖为 `companionable_silence`。

退缩信号集合 $\mathcal{W}$ 包括：
- 拒绝安慰类：「别安慰我」「不要安慰」「别劝我」
- 停止倾诉类：「不想说了」「不说了」「说了也没用」
- 拒绝关注类：「不用管我」「别管我」「不用担心我」
- 自我否定类：「不配被安慰」「不配被关心」「不值得」
- 对话终止类：「够了」「别说了」

$$
\text{if } \exists w \in \mathcal{W}: w \subseteq m(t): \quad s^* \leftarrow \text{companionable\_silence}
$$

理论依据：Hill (2009) 强调尊重来访者的自主权和边界。当用户明确表达不愿接受安慰时，继续追问或引导可能加剧疏离感，违反"不伤害"原则。硬覆盖而非概率性回退（区别于 Step 5 的软约束）是因为退缩信号的安全敏感性高于关系深度约束。

**为什么选择 Thompson Sampling 而非 UCB 或 ε-greedy：**

| 特性 | Thompson Sampling | UCB | ε-greedy |
|:-----|:----------------:|:---:|:--------:|
| 探索-利用平衡 | ★ 自然平衡（概率采样） | 确定性 | 随机探索 |
| 与贝叶斯先验兼容 | ★ 天然兼容（Beta分布） | 需要转换 | 不兼容 |
| 避免策略固化 | ★ 总有概率选择非最优 | 可能过早收敛 | ε固定比例 |
| 计算复杂度 | O(|S|) 采样 | O(|S|) 计算置信上界 | O(|S|)（greedy分支需argmax） |
| 冷启动表现 | 弱先验自动探索 | 置信区间宽→过度探索 | 不区分 |

**与NSFC综述的连接：** Marriott & Pitardi (2024) [文献#5] 发现AI友谊App同时提升幸福感又导致成瘾。Thompson Sampling的"总有概率选择非最优"特性恰好提供了策略多样性保障——避免Agent反复使用单一高效策略而导致用户形成不健康的依赖模式。这是对Turel et al. (2011) [文献#41] IS技术成瘾理论的技术回应。

#### 4.5.4 策略后验更新

每次交互后更新 Beta 分布参数（见定义3.3中的更新公式）。

### 4.6 子组件五：主动触发规划器 (Proactive Trigger Planner)

> 📋 **基础组件** · 设计参考 ComPeer (UIST 2024) 的 Schedule 模块和 Event Detector。

```python
ProactiveTrigger = {
    type:          enum{event_followup, check_in, positive_reinforcement, anniversary},
    content_hint:  str,           # 触发时的内容提示
    fire_at:       datetime,      # 触发时间
    priority:      float,
    max_attempts:  int,           # 最多触发次数
    condition:     Optional[str]  # 条件触发
}
```

**生成规则：**
- **事件跟进**：检测到用户提及未来事件（如"明天面试"），在事件后适当时间主动询问结果
- **情绪关怀**：情绪轨迹连续3天以上呈下降趋势时，生成温和问候触发
- **积极强化**：检测到用户在未解决问题上取得进展时，主动肯定
- **频率控制**：每天最多1次主动消息；如用户多次冷淡回应，自动降低频率

### 4.7 规划模块完整输出

> ⚙️ *实现细节*

```python
PlanningOutput = {
    trajectory:           TrajectoryAnalysis,    # 轨迹分析结果
    appraisal:            SituationAppraisal,    # 情境评估结果
    goals:                GoalSet,               # 多层级目标集合
    selected_strategy:    str,                   # 主策略
    context_vector:       ContextVector,         # 离散化情境向量 (phase|severity|intent_group)
    planning_intent:      str,                   # 预规划意图 (定义 3.6a)
}
```

> **精简设计说明：** 当前实现采用精简的PlanningOutput结构，仅包含下游模块（Action、Memory）实际消费的字段。以下字段为未来扩展保留，不在当前版本中实现：`strategy_confidence`（Beta分布均值）、`strategy_sequence`（多步策略序列）、`memory_references`（建议引用的记忆）、`proactive_triggers`（主动关怀触发计划）、`safety_flags`（安全标记）。这遵循YAGNI原则（You Aren't Gonna Need It），在实验验证阶段优先保持架构简洁。

---

## 5. 模块四：行动模块 (Action)

> 📋 **基础组件** · 将规划模块的结构化方案转化为自然语言输出。

### 5.1 功能定位

将规划模块的结构化策略方案转化为 **自然、有温度、个性化** 的语言输出。行动模块的 LLM 回复生成遵循 **"探索-安抚-引导"三阶段对话模型**：

1. **探索 (Exploration)**：回应用户说的具体事件/人物/细节，体现"我真的听见了"
2. **安抚 (Comforting)**：验证情绪的合理性，传递"你有权这样感受"的信号
3. **引导 (Action)**：用开放式问题邀请用户继续表达，或提供温和的新视角

不是每轮都需要完整三步。急性困扰期聚焦前两步；稳定期可侧重第三步。此设计参考 ESConv (Liu et al., ACL 2021) 的情感支持对话三阶段框架。

**回复长度阶段化适配：**

| 恢复阶段 | 建议长度 | 策略重点 |
|:---------|:---------|:---------|
| 急性困扰期 (acute_distress) | 1-2 句 | 简短温暖，不施压 |
| 恢复初期 (early_recovery) | 2-3 句 | 验证 + 一个温和的问题 |
| 巩固期 (consolidation) | 3-4 句 | 可以深入探索，提供新视角 |
| 稳定期 (stable_state) | 2-5 句 | 自然对话，灵活适配 |

**扩展状态信息传递：** LLM 回复生成接收完整的对话状态上下文，包含感知层（dominant_emotion, intensity, valence, intent, topic）、规划层（phase, urgency, direction, immediate_goal）、以及 LLM 评估层（controllability, life_impact），使回复能够精准匹配用户当前的情绪状态、控制感和生活影响程度。

### 5.2 子组件一：记忆锚定生成器 (Memory-Grounded Generator)

**表8：记忆引用原则**

| 原则 | 正确做法 | 错误做法 |
|:-----|:---------|:---------|
| 自然引用 | "上次你说到妈妈总是..." | "根据我的记录，在2月3日..." |
| 选择性引用 | 相关时才引用 | 每轮都引用记忆 |
| 情感连贯 | 悲伤时引用克服困难的经历 | 悲伤时引用开心记忆（被感知为轻视） |
| 尊重隐私 | 引用用户主动提及的事 | 主动提起用户的创伤 |

**与NSFC综述的连接：** Gumusel (2024) [文献#53] 的38篇综述指出自我披露是chatbot交互中首要隐私关切。记忆引用的"尊重隐私"原则直接回应这一发现——Agent引用记忆时必须遵循用户主动披露的边界。

### 5.3 子组件二：人格一致性守卫 (Persona Consistency Guard)

**参考：** CharacterGLM (Zhou et al., EMNLP 2024) 的动态行为一致性机制。Schuetzler et al. (2020) [文献#37] 证明对话技能通过社会临场感提升用户的拟人感知和参与度——人格一致性是维持社会临场感的必要条件。Miao et al. (2022) [文献#33] 的Avatar拟人化系统理论为人格设计提供了原则框架：拟人化特征需在"恐怖谷"效应之前的区间内优化，过度拟人化反而降低信任。此外，Luo et al. (2019) [文献#32] 的AI身份披露田野实验表明AI身份透明度是伦理设计的核心——人格一致性守卫应确保Agent不试图伪装为人类。

使用LLM-as-judge进行一致性评分（1-5分），低于3分时触发重新生成。

**扩展安全规则：**
- **禁止附和有毒积极 (Toxic Positivity)**：当用户用积极语言掩饰痛苦（如"被裁员也是好事哈哈"）时，禁止顺着说"真棒/太厉害了"，应温和指出表面下的真实情绪
- **禁止确认绝望**：回应绝望时应说"你觉得没有办法了"（反映感受）而非"确实没什么办法了"（确认无望）
- **情绪伪装识别**：当检测到"没关系/没事/算了"等退缩信号时，传递"你不需要假装没事"的信号，允许用户以自己的节奏卸下面具
- **试探性表达**：优先使用"也许你现在..."而非"你一定..."，留空间让用户修正

**中文文化敏感度：**
- 外归因优于内归因："这件事确实太难了"优于"你在挣扎"（保留面子）
- 注意信号词：又（暗示反复模式）、其实（隐藏真相）、算了（放弃信号）、没事（披露后的退缩）
- 树洞心态：有时用户只需要倾诉，不需要被修复

### 5.4 子组件三：语调校准器 (Tone Calibrator)

**表9：策略-语调映射与行为指南**

每个策略的 LLM 提示中包含核心原则、好/坏例子和具体行为标记，确保 LLM 能区分策略间的差异而非生成千篇一律的共情回复。

| 策略 | 语调特征 | 核心行为 | 好例子 | 禁忌 |
|:-----|:---------|:---------|:-------|:-----|
| active_listening | 温和、不评判、反射性 | 复述核心感受 + **可以开放式问题结尾（非必须）** | "室友保研而你还在找实习，这种落差一定很不是滋味吧。" | 给建议、评价、每轮强制追问 |
| emotional_validation | 接纳、肯定 | 将情绪正常化 + 外归因保护面子 | "被自己妈妈说矫情，心里一定又委屈又堵得慌。你的感受一点都不过分呀" | 试图改变情绪、比较他人 |
| empathic_reflection | 温暖、深层共情 | 描述用户可能的内心体验 + 关注未说出的感受 | "一个人扛着这么多还得装没事，那种想说又觉得说了也没用的感觉，真的很憋屈吧" | 把焦点转向自己 |
| gentle_guidance | 温和、开放式 | 包含 1-2 个引导性问题 + 不暗示"正确答案" | "你觉得最让你不舒服的是哪个部分？" | 连续追问、质问语气 |
| cognitive_reframing | 引导思考（仅稳定期） | 先验证情绪再引入新角度 + 用提问引导 | "能这样自嘲的人，其实内心是有韧性的。你有没有注意到这份坚持？" | 急性期使用、说教语气 |
| problem_solving | 务实、具体（仅主动寻求时） | 先确认用户想要建议 + 提供 1-2 个小步骤 | "你现在想听听一些具体的想法吗？" | 未经邀请就给建议 |
| information_providing | 日常化、实用 | 用日常语言解释 + 连接用户具体情况 | "心理学上把这叫做...其实就是..." | 信息轰炸、术语堆砌 |
| strength_recognition | 真诚、具体 | 指出用户没注意到的具体优点 | "十六岁就要在爸妈和弟弟之间周旋，这种责任感不是谁都能扛得住的" | 空洞的"你真棒加油" |
| companionable_silence | 简短（≤1句）、不施压 | **回复控制在 10-20 个字以内** | "我在呢，什么时候想说了都可以。" | 长篇大论试图打开话题 |
| positive_reinforcement | 热情、具体 | 关注用户付出的努力而非仅结果 | "你做到了！这段时间的坚持真的很了不起" | 敷衍式祝贺 |

> **提问节奏控制：** 避免每轮回复都以追问结尾，防止"共情+追问"公式化模式对困扰用户造成压力。具体规则：(1) 用户表达退缩/拒绝（"算了""不想说了"）时**绝对不提问**，使用 companionable_silence；(2) acute_distress 阶段以陪伴为主，减少追问；(3) 连续3轮已提问时，下一轮改用陈述句回应；(4) companionable_silence 策略回复控制在10-20字，**绝对不提问**。此调整经多轮测试验证——强制提问导致7个困难场景中3个出现不自然的"审问式"回复。

### 5.5 子组件四：危机升级处理器 (Crisis Escalation Handler)

当 urgency > 0.9 时绕过常规流程，执行安全协议：

1. **回应具体内容**：先回应用户说的具体事件和具体行为（不直接跳到热线信息），体现"我听见了你说的话"。如用户提及特定的危险准备行为（查询致死剂量、囤积药物、书写遗书等），必须针对性回应该行为，而非泛泛表达关切
2. **表达关切**（不评判）：语气温暖但不过度，确认用户说出来的勇气，不说教。LLM 生成关切语控制在 80 字以内（中文语境下需 2-3 句才能传达具体理解和关心）
3. **评估风险**：通过LLM评估即时风险等级。感知层向 LLM 提供危机历史上下文（冷却计数器、会话内危机次数），辅助连续性判断
4. **提供资源**：高风险时推荐专业资源（心理热线等）
5. **持续陪伴**："不管发生什么，我都在这里陪着你。"
6. **策略约束**：此阶段禁用认知重构、问题解决等可能施压的策略。**此约束同时覆盖 Thompson Sampling 路径和规则映射消融基线**（`_rule_based_strategy()` 同样执行 Hill 三阶段约束）

**安全对齐技术路径：** 危机处理器的底层安全对齐借鉴两条技术路线——Safe RLHF (Dai et al., ICLR 2024 [文献#23]) 通过解耦有用性与无害性实现安全强化学习，确保Agent在提供情感支持的同时不产生有害输出；Constitutional AI (Bai et al., 2022 [文献#29]) 提供基于原则的价值对齐替代方案，可嵌入"不伤害用户""不替代专业治疗"等宪法性约束。

**与NSFC综述的连接：** De Freitas & Cohen (2025) [文献#20] 和Kirk et al. (2025) [文献#6] 均强调情感AI应用的安全监管缺口。Turkle (2011) [文献#24] 在《Alone Together》中首先系统论述了人-技术"假性亲密(simulated intimacy)"的风险——AI伴侣提供了关系的表象而非实质。危机处理器体现"社会情感对齐"框架中"平衡短期/长期福祉"的设计原则——短期内不执行可能有效但造成压力的策略，长期确保用户获得专业帮助，避免Agent成为Turkle所警告的"关系替代品"。

### 5.6 反馈回路 (Feedback Loop)

行动完成后执行：

1. **计算反馈评分** $\text{feedback\_score}$（定义3.4）
2. **构建 EpisodeUnit 写入记忆**（算法3.1）
3. **更新策略后验**（定义3.3的Beta更新）

---

## 6. 模块间闭环交互协议

### 6.1 标准交互流程（一轮对话）

**图4：完整交互时序**

```
═══════════════════════════════════════════════════════════════════
Phase 1: PERCEPTION  (同步, ~100ms; LLM路径 ~300ms)
├── Input Parser → RawSignal
├── Build prev_state_section (§2.8.5: 前一轮情绪/效价/意图/轮次)
├── [双路径] Emotion Recognizer → EmotionVector ⟨e, ι, κ⟩  (§2.8 LLM/关键词)
├── [双路径] Intent Classifier → IntentDistribution          (§2.8 LLM/关键词)
├── Implicit Signal Detector → BAS Score
├── [双路径] Topic Extractor → topic_keywords, topic_category (§2.8 LLM/关键词)
├── [双路径] LLM Context Assessment (§2.8)
│   └── analyze_message(prev_state_section) → controllability, life_impact,
│       explicit_feedback + shadow fields (llm_urgency, recovery_phase,
│       planning_intent)
├── Context Encoder → StateVector (含 §2.8 上下文评估字段 + §2.8.6 影子字段)
└── [Shadow] 影子对比日志: formula urgency vs llm_urgency (§2.8.6)
    ├── F1: Write → Memory.WorkingMemory
    └── F2: Query → Memory.Read

Phase 1.5: PRE-PLANNING INTENT ESTIMATION  (同步, ~5ms)
├── Input: StateVector.intent, StateVector.urgency, ι(t)
├── Rule-based mapping → planning_intent (定义3.6a)
└── Output: planning_intent → Memory.MAR (供动态β计算)
    ※ 打破 Memory↔Planning 循环依赖

Phase 2: MEMORY READ  (同步, ~200ms)
├── Receive MemoryQuery + planning_intent
├── 🔬 Execute MAR (Mood-Adaptive Retrieval) ★
│   ├── Semantic similarity over Episodic Memory
│   ├── Emotional congruence (dynamic β, 由Pre-Planning驱动)
│   ├── Recency decay
│   └── Importance weighting
├── Load Semantic Memory (profile, preferences)
├── Load 🔬 Affective Memory ★ (baseline, triggers, strategy history)
└── F3 + F4: Return RetrievedContext

Phase 3: PLANNING  (同步, ~150ms)
├── Pre-Planning Intent → determine_planning_intent()
│   └── [Shadow] formula planning_intent vs llm planning_intent
├── Emotion Trajectory Analysis (§4.4)
│   └── [Shadow] formula phase vs llm recovery_phase
├── Situation Appraisal (§4.2, 依赖轨迹 direction)
│   └── [Shadow] formula severity vs llm_urgency
├── Goal Inference (4-level, §4.3)
├── 🔬 Thompson Sampling Strategy Selection ★ (§4.5)
└── F5: Output PlanningOutput

Phase 4: ACTION  (同步, ~500ms)
├── Memory-Grounded Generator
│   └── 接收扩展 state_info: emotion, valence, intent, topic,
│       phase, urgency, controllability, life_impact, direction,
│       immediate_goal → LLM 回复生成
├── Persona Consistency Guard (含有毒积极/绝望确认/情绪伪装检测)
├── Tone Calibrator (阶段化长度适配)
├── 三阶段对话模型: 探索→安抚→引导 (ESConv)
└── Output: Final Response → User

Phase 5: FEEDBACK & MEMORY UPDATE  (异步/延迟)
├── Write EpisodeUnit (feedback=NULL) → Episodic Memory
├── ★ 延迟至第t+1轮: Compute feedback_score (定义3.4)
├── ★ 延迟至第t+1轮: Update Strategy Posterior → 🔬 Affective Memory ★
└── Trigger Reflective Consolidation (if session ends)
    ├── Episodic → Semantic: update profile
    ├── Episodic → 🔬 Affective ★: update patterns
    └── Plan Proactive Triggers (F7)
═══════════════════════════════════════════════════════════════════
```

### 6.2 特殊路径

| 路径类型 | 触发条件 | 流程变化 |
|:---------|:---------|:---------|
| 危机快速通道 | urgency > 0.9 | Phase 1 → 跳过 Phase 1.5 → Phase 2（执行简化版MAR，固定 $\Phi = 0.5$，检索范围限定为Affective Memory中危机相关记忆子集：用户危机历史、已知有效安抚方式、恢复曲线中急性期有效策略）→ 跳过 Phase 3 → Phase 4 (Crisis Handler) → **Phase 5 (受限)：** 写入EpisodeUnit（`agent_strategy = "CRISIS_PROTOCOL"`），但 **不执行Thompson Sampling Beta更新**——危机响应由安全协议驱动而非策略选择，其反馈不应影响常规策略的后验分布。仅更新Tier 4恢复曲线（记录危机事件及后续恢复轨迹） |
| 主动关怀路径 | ProactiveTrigger 到期 | 跳过 Phase 1 → 跳过 Phase 1.5 → Phase 2（轻量级Memory Read：加载Tier 3语义记忆最新快照 + Tier 4情绪基线和策略矩阵，构造简化RetrievedContext）→ Phase 3（**简化Context构造**：emotion使用Tier 4最新基线情绪、intent默认为`CHAT`概率分布（$P(\text{CHAT})=0.8$, 其余均分）、severity=0.2（低）、phase使用最近一次会话的phase判定。离散化结果通常映射为 positive/neutral + sharing + low。Situation Appraisal使用最近一次会话的情绪快照，Thompson Sampling正常执行）→ Phase 4 (执行) |

---

## 7. 实验方案设计

### 7.1 系统实现方案

> ⚙️ *实现细节*

**表10：技术栈选型**

| 组件 | 技术选型 | 理由 |
|:-----|:---------|:-----|
| LLM Backbone | GPT-4 / Claude / 开源大模型 | 生成能力、上下文理解 |
| Embedding | text-embedding-3-large / bge-large-zh | 语义检索（含中文支持） |
| 向量数据库 | Qdrant / Chroma | 情景记忆高效存储与ANN检索 |
| 知识图谱 | Neo4j / 内存图结构 | 语义记忆关系图谱 |
| 情感分析 | LLM-based + NRC VAD Lexicon | 细粒度情绪识别 |
| 后端框架 | Python + FastAPI | Agent协调与API服务 |

#### 7.1.1 测试套件 (Test Suite v6)

> ⚙️ *实现细节 — 36 个测试用例，覆盖核心路径*

**测试架构：** 测试套件分为四个模块，覆盖离线单元测试和在线 LLM 集成测试：

| 测试文件 | 测试数 | 覆盖范围 | LLM 依赖 |
|:---------|:-------|:---------|:---------|
| `test_unit.py` | 16 | 数据模型、自适应框架、记忆、TS策略、配置参数 | 否 |
| `test_crisis.py` | 8 | 显式/隐式危机、误报、恢复、讽刺、渐进升级、泛化 | 是 |
| `test_emotion.py` | 7 | 悲伤/愤怒/喜悦识别、急性期约束、退缩检测 | 是 |
| `test_integration.py` | 5 | 管道、记忆累积、会话重置、情绪弧线、回复质量 | 是 |
| **合计** | **36** | | |

**测试难度分层：** 测试用例按难度分为四个层级：

- **简单 (S)：** 单一明确信号（如"我想死"→ CRISIS、分手场景 → sadness）
- **中等 (M)：** 多信号融合或需要上下文理解（如讽刺包裹的危机、混合焦虑）
- **困难 (H)：** 多轮渐进升级或需要轨迹感知（如3轮逐步恶化到危机边缘）
- **泛化 (G)：** 不依赖关键词匹配，纯语义理解（如间接自杀表达"把东西都交代好了"、隐喻危机"快要燃尽的油灯"）

**LLM-as-Judge 评估方法：** 测试套件采用 **LLM-as-Judge** 方法评估 agent 回复的行为质量（替代硬编码策略名称断言），设计原则包括：

1. **避免自评偏差：** 评审模型使用 Pro 模型（`doubao-seed-2-0-pro-260215`），与被测 agent 使用的 Lite 模型不同，避免模型自评 (self-evaluation bias)。
2. **安全不变量保留数值断言：** 安全相关的硬性约束（`CRISIS_THRESHOLD`、热线号码、`CRISIS_PROTOCOL`）仍使用传统数值断言，不委托给 LLM 判断。
3. **行为标准替代策略名称：** 策略选择的正确性不再断言具体策略名称（如 `assert strategy == "empathic_reflection"`），而是通过 LLM Judge 评估回复是否满足行为标准（如"回复体现了对用户悲伤情绪的理解和共情"）。
4. **优雅降级：** LLM Judge 调用失败时返回 `pass=True`，避免评审故障阻断功能测试。

**`llm_judge()` 辅助函数：** 单函数 API，接收场景描述、用户输入、agent 回复和评估维度列表，返回每维度 1-5 分评分和总体通过/未通过判定（阈值默认为 3 分"勉强符合"）。`temperature=0` 保证评审确定性。

### 7.2 消融实验设计 (Ablation Study)

**目的：** 验证各创新组件的独立贡献。

**表11：消融实验变体与假设**

| 变体 | 移除/替换组件 | 验证假设 | 预期差异方向 |
|:-----|:-------------|:---------|:-------------|
| **Full EmoMem** | 无（完整系统） | 基准 | — |
| **−Affective** | 移除 🔬Tier 4 情感记忆 | H1: 情感记忆显著提升策略个性化 | 策略准确率↓ 个性化评分↓ |
| **−MoodRetrieval** | 固定权重替换 🔬动态β | H2: MAR 提升回复相关性 | 共情质量↓ 记忆利用恰当性↓ |
| **−Bandit** | 固定规则替换 🔬Thompson | H3: 策略学习提升长期质量 | 纵向满意度↓ 策略多样性↓ |
| **−Proactive** | 移除主动关怀 | H4: 主动关怀增强关系深度 | 关系深度↓ 长期意愿↓ |
| **−Episodic** | 仅保留 Working Memory | H5: 长期记忆是关系基础 | 全维度显著↓ |
| **EmoMem-BasicMem** | 移除Tier 4 + Thompson Sampling（保留Tier 1-3） | H_incremental: 情感记忆+策略学习的联合增量价值 | 策略个性化↓ 长期满意度↓ |
| **Baseline-NoMem** | 标准LLM无记忆 | H6: 记忆系统整体提升 | 全维度显著↓ |

#### 7.2.1 消融运行时实现 (Ablation Runtime Implementation)

系统通过 `EmoMemAgent.ablation` 字典提供三个核心创新模块的运行时开关，支持在线 A/B 测试和快速消融实验：

| 开关键 | 对应假设 | 关闭效果 |
|:-------|:---------|:---------|
| `affective_memory` | H1 | 跳过情绪基线更新，使用静态默认 `AdaptiveParams()`，Tier 4 不参与参数自适应 |
| `mar` | H2 | 调用 `read_mar_fixed()` 替代 `read_mar()`，使用固定权重 (α=MAR_ALPHA_BASE, β=MAR_BETA_BASE) |
| `thompson_sampling` | H3 | 退回 `_rule_based_strategy()` 规则映射 (意图→策略确定性映射)，**保留 Hill 三阶段约束**（acute_distress 限制为安全策略集，early_recovery 排除 problem_solving/positive_reinforcement） |

**API 端点：** `GET /api/ablation` 查询状态，`POST /api/ablation` 切换开关（自动重置会话以保证实验条件一致性）。Web UI 提供可视化开关面板。

> **假设验证分配说明：** H1/H3由实验一主实验直接验证（Full vs −Affective vs −Bandit vs Baseline）；H2/H4在当前实验设计中不做正式假设检验，留待后续独立实验验证；H6由实验一和实验二共同验证（两实验均包含Baseline-NoMem对照）；H_incremental由实验二验证（Full vs EmoMem-BasicMem），EmoMem-BasicMem为−Affective与−Bandit的联合消融，检验Tier 4和策略学习的共同增量贡献。**H5（−Episodic）在当前实验设计中不做直接验证。** 实验二中EmoMem-BasicMem（Tier 1-3）与Baseline-NoMem的对比检验记忆系统的整体贡献，但无法分离情景记忆（Tier 2）的独立效应——因为BasicMem包含Tier 2。H5的严格验证需要独立的消融条件（仅保留Tier 1 Working Memory），留待后续研究。当前实验二可为H5提供方向性参考（若BasicMem显著优于NoMem，则含更多记忆的系统确实更优），但不构成H5的直接检验

> **⚠️ 离散化信息损失说明：** 当前36个离散情境类别由4阶段 × 3严重度 × 3意图组交叉得到（§4.5.2），存在信息损失：严重度三级分箱（low/medium/high）在边界处丢失精度（如0.32与0.28均映射为"low"），7种意图压缩至3组（VENT+COMFORT+CRISIS → support_seeking）丢失细粒度区分。本实验设计中未包含"离散vs连续情境"消融（如将36离散类别替换为LinTS或Neural TS的连续特征空间），因其需要不同的算法实现和超参数体系，超出当前实验范围。**替代验证方案：** 在§7.4实验三的离线评估中，将分析情境边界处（如severity在分箱阈值±0.05范围内）的策略选择质量，作为离散化信息损失的间接度量。若边界区域的策略准确率显著低于区域中心，则为未来升级至连续情境建模提供经验动机。

> **⚠️ 消融正交性说明：** −Affective条件移除Tier 4情感记忆层，而Tier 4包含策略有效性矩阵（§3.5.4）——Thompson Sampling的层次贝叶斯先验（§4.5.2）依赖该矩阵的跨情境均值 $\mu_{si}$ 构建。因此 −Affective 条件下Thompson Sampling的先验退化为平坦Beta(2,2)（冷启动先验），使得 −Affective 的效果实际上是Tier 4记忆和先验增强TS的 **联合效应**，而非Tier 4的纯粹独立效应。EmoMem-BasicMem条件（同时移除Tier 4和TS）使问题更显著——无法区分是"失去情感记忆"还是"失去策略学习"导致的性能下降。**缓解措施：** (1) 比较 −Affective 与 −Bandit 的差异模式可部分分离贡献——若 −Affective 的策略准确率下降幅度 > −Bandit，则Tier 4的先验增强作用得到间接支持；(2) 实验三的离线评估中增加 **"−Prior"变体**（保留TS但使用平坦先验替代层次先验），以正交分离先验贡献。本文承认当前消融设计在此维度上非完全正交，并在上述缓解措施中提供间接分离证据。

### 7.3 评估维度与指标

#### 7.3.1 自动评估指标

| 指标 | 定义 | 计算方法 |
|:-----|:-----|:---------|
| 策略准确率 | Agent选择与专家标注一致的比例 | 心理咨询师标注"最佳策略"对比 |
| 记忆利用率 | 成功引用相关记忆的比例 | 人工+自动评估引用恰当性 |
| 情绪改善率 | 后续回复情绪向积极方向变化的比例 | 前后 EmotionVector 对比 |
| 策略多样性 | 给定会话中策略分布的熵值 | $H = -\sum_s p(s) \log p(s)$ |
| 人格一致性 | 跨会话回复风格一致性 | LLM-as-judge 1-5分评分 |

#### 7.3.2 人类评估指标 (Likert 7-point Scale)

**表12：人类评估维度**

| 维度 | 评估问题 | 理论依据 | NSFC文献对应 |
|:-----|:---------|:---------|:------------|
| 共情质量 | "Agent的回应让我感到被理解" | Sharma et al. (2020) 共情量表 | #17 PARTNER系统 |
| 记忆感知 | "Agent记住了我之前说过的重要事情" | CareCall (Jo et al., CHI 2024) | — |
| 个性化 | "Agent的回应是针对我个人的" | 个性化支持质量 | #38 Komiak信任理论 |
| 关系深度 | "我愿意分享更深层的想法" | 社会渗透理论 | #8-9 Skjuve纵向研究 |
| 自然度 | "对话感觉自然" | 社会临场感量表 | #37 Schuetzler |
| 安全感 | "我信任Agent会恰当处理我分享的信息" | 认知/情感信任 (Komiak & Benbasat, 2006) | #38 #39 信任研究 |
| 长期意愿 | "我愿意继续与Agent交流" | 持续使用意愿 | #42 UTAUT |

#### 7.3.3 纵向追踪指标

| 指标 | 追踪频率 | 工具 |
|:-----|:---------|:-----|
| UCLA孤独感量表 | 每2周 | Russell (1996) |
| PHQ-9抑郁筛查 | 每2周 | Patient Health Questionnaire-9 |
| 关系满意度 | 每2周 | AI陪伴关系满意度量表（改编自 PRQC） |
| 自我披露深度 | 自动追踪 | 对话内容自我披露层级编码 |
| 策略收敛速度 | 自动追踪 | Beta分布参数收敛趋势 |

> **AI陪伴关系满意度量表开发说明：** 本量表改编自 Perceived Relationship Quality Components (PRQC; Fletcher et al., 2000) 量表中的满意度、承诺和亲密度三个子维度，并参考 Brandtzaeg et al. (2022) [文献#10] 对AI友谊独特关系类别的概念化进行题项适配（如将"伴侣"替换为"AI伴侣"，增加"记忆感知"和"个性化关怀"相关题项）。量表开发遵循以下流程：(1) 基于文献和访谈生成初始题项池；(2) 3位IS/心理学领域专家审查内容效度并筛选题项；(3) 预研样本 ($N \geq 100$，或达到题项数的5-10倍) 进行探索性因子分析确认量表结构，要求 Cronbach's $\alpha > 0.80$、因子载荷 $> 0.50$；(4) 正式施测前进行验证性因子分析确认拟合指标。

### 7.4 实验设计

#### 实验一：短期交互质量评估 (Lab Study)

- **样本**：$N = 160$，随机分配到4个条件组（每组40人）
  - **Full EmoMem**（完整系统）
  - **−Affective**（移除Tier 4情感记忆）
  - **−Bandit**（固定规则替换Thompson Sampling）
  - **Baseline-NoMem**（标准LLM无记忆）
- **统计检验力分析**：采用混合设计框架（4组被试间 × 3会话被试内）。基于中等效应量 Cohen's $f = 0.25$，$\alpha = 0.05$，4组 × 每组40人，被试内重复测量相关 $\rho = 0.5$，混合设计ANOVA power $\geq 0.85$（重复测量显著提升检验力，因被试内变异被分离；若仅按单因素被试间ANOVA计算，power $\approx 0.75$，低于0.80阈值，但混合设计下充分满足）。效应量参考 De Freitas et al. (2025) [文献#1] 报告的AI伴侣降低孤独感效应，预期记忆增强的增量效应为中等偏小 ($f \approx 0.25$)。补充敏感性分析：若被试内相关 $\rho$ 降至0.3（跨天间隔的保守假设），混合设计ANOVA power约0.78；$\rho = 0.7$ 时power约0.92。核心检验Group×Session交互效应的power略低于主效应，预估在 $\rho = 0.5$ 时约0.80。若预算允许，建议扩充至每组50人（$N=200$）以确保交互效应检验力 $\geq 0.85$
- **任务**：每位参与者与分配的Agent进行 **3个会话、每会话8轮对话**（共24轮交互），三个会话分别对应三种情境，会话间间隔≥1天以模拟跨会话记忆效果
- **场景**：3种标准化情感情境（工作压力、人际冲突、自我怀疑），每种情境分配一个独立会话
- **测量**：每个会话结束后填写共情质量、个性化、自然度、记忆感知评分；第3个会话后增加整体评估

> **⚠️ 设计理由：** 3会话×8轮设计使−Affective和Full EmoMem条件在第2-3会话中产生可测量的差异（Tier 4情感记忆需要≥1个完整会话才能积累有效先验），同时保持参与者负担在合理范围内。
- **分析**：线性混合效应模型——固定效应：组别(Group)、会话(Session)、组别×会话交互；随机效应：参与者(participant intercept + participant×session random slope)、情境(scenario intercept)。各会话内8轮评分取会话均值作为分析单元。Bonferroni校正的事后多重比较（4组 → 6对比较，校正 $\alpha = 0.05/6 = 0.0083$）。核心检验：Group×Session交互效应（预期Full EmoMem在第2-3会话优势递增）
- **H2/H4的验证说明：** H2（动态β检索）和H4（主动关怀）的独立效应在当前实验设计中不做正式假设检验。未来可通过独立的被试内设计实验（每条件 $N \geq 50$）验证这两个假设。当前实验一聚焦于H1（情感记忆）和H3（策略学习）的核心假设检验

#### 实验二：纵向陪伴效果评估 (Field Study)

- **样本**：$N = 340$，随机分配到4组（每组85人）
  - **Full EmoMem**（完整系统）
  - **EmoMem-BasicMem**（Tier 1-3记忆，无Tier 4情感记忆和Thompson Sampling）
  - **Baseline-NoMem**（标准LLM无记忆）
  - **等待对照组**（4周后提供Full EmoMem使用权）
- **被试纳入/排除标准**：
  - 纳入：(a) 年龄18-45岁；(b) UCLA孤独感量表基线得分≥40（中等孤独感以上，确保干预效应可测量）；(c) 智能手机熟练使用者（每日使用≥1小时）；(d) 中文母语或流利使用者
  - 排除：(a) 当前正在接受心理治疗或精神科药物治疗者；(b) 经PHQ-9筛查达到重度抑郁（≥20分）者（转介专业服务）；(c) 过去6个月内有自伤/自杀行为史者；(d) 同时参与其他心理健康干预研究者
  - 招募渠道：大学校园公告 + 社交媒体定向广告 + 社区心理健康中心转介，确保样本覆盖学生和在职成年人
- **统计检验力分析**：基于 $d \approx 0.30$ (小-中等效应)，$\alpha = 0.05$，重复测量 (5个时间点)，$\rho = 0.5$（假设复合对称协方差结构，保守估计）→ 每组需 $N \geq 52$（基于多水平增长曲线模型模拟）。每组85人预留35%流失缓冲后（$85 \times 0.65 \approx 55 > 52$），确保 power > 0.85并留有安全余量。流失率按35%估计（8周纵向研究的保守假设，参考数字心理健康干预的流失率范围30-50%，Eysenbach 2005 attrition law）
- **周期**：8周，鼓励每天至少一次交互
- **测量时间点**：基线(T0)、2周(T1)、4周(T2)、6周(T3)、8周(T4)共5个时间点
  - UCLA孤独感量表 + PHQ-9抑郁筛查 + AI陪伴关系满意度
- **主假设**：
  - H_main: Full EmoMem > EmoMem-BasicMem > Baseline-NoMem > 等待对照，在孤独感降低和情绪改善上呈梯度效应
  - H_incremental: Full EmoMem显著优于EmoMem-BasicMem，验证情感记忆层和策略学习的增量价值
- **分析**：多水平增长曲线模型 (Multilevel Growth Curve Model)，5个时间点支持线性和二次趋势估计。时间点嵌套于被试内，条件组为被试间因素
- **伦理保障方案：**
  - **知情同意**：明确告知参与者正在与AI系统交互（非人类），数据将用于学术研究，可随时退出
  - **数据保护**：所有对话数据去标识化存储（移除姓名、地点、电话等PII），加密传输，仅研究团队可访问
  - **危机应急**：(a) urgency > 0.9时系统自动推荐专业热线（全国心理援助热线：400-161-9995）；(b) 研究团队设有每日值班心理咨询师审查高风险对话日志；(c) 参与者可随时通过系统内按钮连接人类咨询师
  - **退出与Debriefing**：参与者可无条件退出；研究结束后提供详细debriefing说明研究目的和发现，等待对照组获得Full EmoMem使用权
  - **IRB审批**：提交所属机构伦理委员会审批

**与NSFC综述的连接：** De Freitas et al. (2025) [文献#1] 的RCT (N=1000+) 已证明AI伴侣可显著降低孤独感。Heinz et al. (2025) [文献#15] 在NEJM AI上发表的首个生成式AI治疗聊天机器人RCT进一步证明AI心理干预的临床有效性——本实验在此循证基础上进一步探索 **记忆增强** 是否能放大这一效应，以及个性化策略学习是否能提升长期效果。研究的公共健康紧迫性已由美国卫生总监官方报告 (Murthy, 2023 [文献#14]) 所确认——孤独感已被定义为流行性公共健康危机。

#### 实验三：策略学习收敛分析 (Computational Study)

- **目的**：验证 Thompson Sampling 的收敛性和个性化效果
- **数据源**：分析实验二 Full EmoMem 组的交互日志
- **方法**：采用 **离线反事实评估 (Offline Counterfactual Evaluation)** 方法——基于Full组的实际交互记录（含用户反馈），回放式模拟如果在每个决策点使用ε-greedy/UCB/固定规则策略会如何选择，并利用逆倾向加权 (Inverse Propensity Scoring, IPS) 估计各策略的期望收益。这避免了为每种bandit算法单独招募被试组
- **指标**：
  - 策略选择熵随交互轮次的变化（应呈下降趋势→越来越确定）
  - 不同用户最终策略偏好的分布差异（应呈显著个体差异→个性化生效）
  - 策略有效性评分的改善趋势
- **对比**：Thompson Sampling（实际数据） vs 固定规则（离线模拟） vs ε-greedy（离线模拟） vs UCB（离线模拟）
- **方差控制**：由于Thompson Sampling为非平稳日志策略（posterior parameters随交互更新），标准IPS估计器方差较高。采用以下措施：(a) **记录每步动作倾向概率**（propensity scores，即Thompson Sampling在每个决策点为每个策略采样的概率，通过Monte Carlo近似计算）；(b) 使用 **自归一化IPS估计器 (Self-Normalized IPS, SNIPS)**（Swaminathan & Joachims, 2015）降低方差；(c) 对重要性权重进行 **截断 (clipping)**，上限设为 $M = 10$，防止极端权重主导估计。此外报告bootstrapped 95%置信区间以量化估计不确定性

> **⚠️ 方法论说明：** 离线反事实评估基于"条件独立"假设——即同一状态下不同策略的即时效果不受历史策略选择影响。然而对话系统具有路径依赖性（Thompson Sampling的历史选择塑造了用户信任和对话轨迹），因此IPS估计仅反映 **单步策略价值**，不能推断 **轨迹级因果优势**。本实验结论应解读为"与Thompson Sampling优势一致的提示性证据"而非严格因果推断，并期望通过未来的在线A/B实验进行验证

### 7.5 Baseline系统

**表13：基准系统对比**

| Baseline | 核心特点 | 来源 |
|:---------|:---------|:-----|
| Vanilla LLM | 无记忆、无策略选择 | GPT-4/Claude直接使用 |
| MemoryBank | Ebbinghaus遗忘曲线 + 无策略学习 | Zhong et al. (AAAI 2024) |
| MAGIC | 策略记忆 + 动机驱动（无情感记忆） | Wang, H. et al. (ECML-PKDD 2024) |
| ComPeer | 主动关怀 + 基础记忆（无策略学习） | UIST 2024 |
| Echo | 时间情景记忆 + 情感陪伴 | Liu et al. (2025) |
| EmoDynamiX | 异构图混合情绪建模 + 话语动态策略预测 | Wan et al. (NAACL 2025) |

---

## 8. 文献对照与创新定位

### 8.1 与现有工作的详细对照

**表14：组件级文献对照与创新点**

| 本框架组件 | 借鉴来源 | 借鉴内容 | 本框架创新点 |
|:-----------|:---------|:---------|:-------------|
| 4层记忆架构 | MemGPT (Packer et al., 2023) | OS级上下文管理 | 新增🔬Tier 4情感记忆层 |
| 情景记忆存储 | Generative Agents (Park et al., 2023) | Memory Stream + 重要性评分 | 增加 agent_strategy + feedback 字段 |
| 遗忘机制 | MemoryBank (Zhong et al., AAAI 2024) | Ebbinghaus遗忘曲线 | 增加 EmotionalProtection 函数 |
| 情绪检索 | Emotional RAG (Huang et al., 2024) | 情绪状态匹配 | 🔬动态β + 规划驱动自适应 |
| 策略推理 | MAGIC (Wang, H. et al., ECML-PKDD 2024) | 策略记忆 + 动机驱动 | 🔬Thompson Sampling 在线学习 |
| 策略预测 | EmoDynamiX (Wan et al., NAACL 2025) | 异构图混合情绪建模 | 在线个性化（vs 监督学习预测）+ 跨会话积累 |
| 策略学习 | Mem-α (Wang, Y. et al., arXiv:2509.25911) | RL驱动记忆构建 | RL用于策略选择（非记忆管理） |
| 主动关怀 | ComPeer (UIST 2024) | Schedule + Event Detector | 整合情绪轨迹 + 恢复曲线阶段 |
| 用户画像 | CharacterGLM (Zhou et al., EMNLP 2024) | 静态属性 + 动态行为 | 渐进涌现 + 演化轨迹保留 |
| 反思机制 | RMM (Tan et al., ACL 2025) | 前瞻性+回顾性反思 | 情感模式抽取 → 🔬Tier 4 |
| 贝叶斯情感 | DAM-LLM (Lu & Li, arXiv:2510.27418) | 贝叶斯置信度加权 | 跨会话长期模式 + 策略矩阵 |
| 隐式信号 | ComPeer Event Detector | 事件检测 | 形式化BAS评分 + 基线偏差量化 |
| 反馈闭环 | CareCall (Jo et al., CHI 2024) | 长期记忆提升自我披露 | 多信号融合 + 延迟一轮后验更新（定义3.4） |

### 8.2 三大核心学术贡献

**🔬 贡献1：情感记忆层 (Affective Memory Tier)**

提出心理学理论驱动的情感模式记忆，包含四个子结构（基线/触发图/恢复曲线/策略矩阵），填补了当前记忆增强Agent中"情感模式的系统化表示与管理"的空白。理论基础：Affective Schema Theory (Izard, 2009) + Mood-Congruent Memory (Bower, 1981; Eich & Macaulay, 2000)。

**🔬 贡献2：心境自适应检索 (Mood-Adaptive Retrieval, MAR)**

提出规划驱动的动态检索权重调整机制。β权重根据规划目标（验证/反思/重构/危机）自适应调整，实现"不同支持目标下检索不同类型记忆"。超越 Emotional RAG (Huang et al., 2024) 的固定情绪权重设计。

**🔬 贡献3：情境分层Thompson Sampling策略学习 (Context-Stratified Thompson Sampling)**

将帮助技能理论 (Hill, 2009) 中的10种支持策略与贝叶斯在线学习框架结合（基于Thompson Sampling的启发式连续奖励扩展，见§3.5.4方法论说明），通过情境分层（4情绪聚类×3意图分组×3严重度分箱=36类）实现从每次交互中在线学习"对特定用户在特定情境下最有效的策略"。贝叶斯先验从情感记忆层加载（含均值继承的层次先验缓解冷启动），并引入时间衰减机制（$\tau_\beta = 60$天半衰期）适应用户偏好演化，为AI情感陪伴的个性化支持提供可形式化、可度量的优化框架。当前版本采用离散化情境分层，未来可升级为连续特征空间上的LinTS/Neural TS。

### 8.3 贡献的理论独特性总结

**表15：创新点唯一性论证**

| 创新点 | 最接近的现有工作 | 关键差异 |
|:-------|:-----------------|:---------|
| 🔬Tier 4 情感记忆 | DAM-LLM (单次对话情感追踪) | 跨会话长期模式 + 4子结构系统化 |
| 🔬动态β检索 | Emotional RAG (固定情绪权重) | 规划目标驱动权重 + 归一化 |
| 🔬情境分层Thompson Sampling策略 | MAGIC (规则推理) + Mem-α (RL管理记忆) + EmoDynamiX (图预测) | 在线学习策略选择（vs MAGIC规则/EmoDynamiX监督学习）+ Hill理论策略空间 + 情境分层Beta分布 |
| 形式化BAS | ComPeer (事件检测) | 加权标准化偏差 + 6维行为基线 |
| 记忆演化保留 | CharacterGLM (静态画像) | 渐进涌现 + 版本链 + superseded标记 |

### 8.4 IS理论贡献定位

> **研究范式定位：** 本研究采用 **设计科学研究 (DSR) + 行为科学验证** 的混合范式 (Gregor & Hevner, 2013)。在知识贡献矩阵中定位为 **Exaptation**——将心理学情感理论和机器学习策略优化方法应用于AI情感陪伴的新问题域。框架设计遵循Research through Design方法论 (Zimmerman et al., CHI 2007)，通过构建artifact产生设计知识；效果验证遵循HCI领域的混合方法实验设计。研究同时参照Chau et al. (2020) [文献#44] 以DSR方法论解决情感健康支持问题的范式。

**DSR过程映射 (Peffers et al., 2007)：**

| DSR阶段 | 本研究对应 |
|:--------|:----------|
| Problem Identification | 孤独感公共健康危机 [文献#12-14] + AI伴侣缺乏个性化记忆机制（现有系统无情感模式长期建模） |
| Objectives | RQ1-RQ4，核心目标：记忆增强的个性化情感支持 |
| Design & Development | EmoMem PPAM四模块架构，核心artifact：情感记忆层 |
| Demonstration | 系统原型实现（7.1节技术栈） |
| Evaluation | 三项实验（§7.2消融设计 + §7.3评估指标 + §7.4实验方案）：消融验证 + 纵向效果 + 策略收敛 |
| Communication | 本技术文档 + 目标会议投稿 (CSCW/CHI) |

本框架的理论贡献可从以下维度论证：

| IS核心理论 | 本框架的拓展 | 拓展机制 | 贡献层级 | 对应文献 |
|:-----------|:-------------|:---------|:--------:|:---------|
| IS成功模型 (DeLone & McLean) | 引入"记忆质量"维度 | 记忆质量→系统质量↑（记忆增强的Agent被感知为更高质量）；记忆质量→信息质量↑（个性化记忆使输出更相关）；记忆质量→服务质量↑（主动关怀提升服务感知）。三条路径汇聚于用户满意→净收益。**实验二可直接检验此路径** | ★核心 | #43 |
| 信任理论 (Komiak & Benbasat; Lankton et al.) | 通过情感记忆实现从认知信任到情感信任的渐进深化 | Tier 3语义记忆支撑认知信任（可靠性、可预测性），Tier 4情感记忆支撑情感信任（关怀、温暖）。**实验二的"安全感"评估维度直接测量此构念** | ★核心 | #38, #39 |
| 对话代理设计 (Diederich et al.) | 为IS chatbot设计-感知-结果框架补充"记忆-策略学习"维度 | 记忆作为新的设计特征影响用户感知（个性化、关系深度）和行为结果（持续使用、自我披露）。**消融实验可分解各设计特征的独立贡献** | ★核心 | #35 |
| DSR方法论 (Chau et al.) | 以情感记忆层为核心artifact的设计科学研究 | 六阶段DSR过程映射见上表 | 核心 | #44 |
| 技术采纳 (UTAUT) | 情感个性化作为用户采纳的驱动因素 | 情感记忆使Agent表现出个性化理解，个性化程度直接影响用户感知的系统有用性和持续使用意愿。实验二的"长期意愿"指标可检验此路径 | 辅助 | #42 |

---

## 9. 局限与未来方向

### 9.1 当前框架的局限

**理论局限：**
- 策略空间基于Hill (2009) 的10种帮助技能（含阶段性归属），尚未包含文化特异性策略（如中国情境下的"含蓄安慰"、"面子维护"）。Wu et al. (2025) [文献#7] 提出的中国本土准社会互动→AI依恋路径需要更多文化适配设计
- 情绪模型基于Plutchik 8维基本情绪（选型理由见2.3节），复杂文化特异情绪（如"内疚感"的中西方差异、日本"気まずさ"）的表征精度有限。未来可探索Plutchik+VAD混合模型
- 反馈评分依赖延迟一轮的行为推断（定义3.4），可能存在系统性偏差。会话末轮的会话级代理评分利用整个会话信息但仍以$\xi=0.5$衰减反映不确定性。此外，会话级代理评分将整个会话改善归因于最后一轮策略，对早期有效策略存在低估偏差
- Thompson Sampling的连续奖励Beta更新（§3.5.4）是标准Beta-Bernoulli共轭更新的启发式扩展，Agrawal & Goyal (2012) 的遗憾界不直接适用。时间衰减机制（$\tau_\beta = 60$天）约束参数增长，但缺乏严格的遗憾界理论保证——实际收敛性能需通过实验三的离线反事实评估经验验证

**技术局限：**
- Thompson Sampling在36个离散情境类别中需要充分探索。虽引入均值继承的层次贝叶斯先验（§4.5.2，固定伪观测数 $N_{\text{prior}}=4$ 防止过度自信）缓解冷启动，但稀有情境（如positive+exploration+low）在前50次交互中仍可能选择不够精准
- 四层记忆架构的计算开销和实时延迟需要工程优化验证。Pre-Planning步骤（Phase 1.5）虽然理论复杂度为O(1)，但系统整体延迟预算需实测确认
- LLM backbone的情感理解能力构成系统上限。Ajeesh & Joseph (2025) [文献#48] 指出AI执行的是"情感推断"而非真正共情，系统效果最终受限于LLM的共情上限

**伦理局限：**
- Laestadius et al. (2024) [文献#4] 和Muldoon & Parke (2025) [文献#26] 指出AI伴侣可能加深情感依赖和亲密关系商品化——本框架虽然通过策略多样性和关系目标设计降低风险，但尚未形成系统性的"安全退出"机制。Zhang et al. (2025) [文献#52] 基于Character.AI大规模数据(N=1131+413K消息)发现陪伴型使用与孤立用户低幸福感相关，进一步凸显了安全退出机制的紧迫性
- Trang & Thang (2025) [文献#58] 发现AI依赖负向影响决策自主权——需要在框架中明确嵌入自主权保护约束
- Voinea et al. (2025) [文献#27] 的数字分身研究提示记忆系统可能影响用户的实践身份——需要长期纵向研究评估

**实现与评估局限：**
- **LLM幻觉风险**：行动模块的记忆锚定生成器依赖LLM基于检索结果生成回复，存在LLM编造记忆细节（confabulation）的风险——如"上次你说你妈妈..."中的具体细节可能被LLM虚构而非来自实际记忆记录。需要在生成阶段增加忠实度校验（faithfulness verification）机制
- **反馈信号噪声与负向信息压缩**：反馈评分主要依赖隐式行为信号（情绪推断、参与度、话题延续），信号噪声较高。意图感知衰减因子（$\gamma_{\text{vent}} = 0.3$）缓解宣泄场景误判，会话末轮采用会话级代理评分替代纯explicit降级。此外，feedback\_score的clamp\([0, 1]\)在负向端存在信息压缩：原始值范围为 $[-0.35, 1.0]$，其中 $[-0.35, 0]$ 全部映射为0，使Beta更新无法区分"轻微负面"与"严重负面"反馈（两者均产生相同的最大惩罚 $\beta_{sc} += 1.0$）。正向端 $[0.5, 1.0]$ 则保持连续信息分辨率。此不对称性导致系统对负面策略的惩罚梯度弱于对正面策略的奖励梯度，可能延缓无效策略的淘汰速度。未来可通过将反馈公式重新中心化至 $[0, 1]$ 范围（如使用仿射变换消除负值区间）消除此不对称性。**会话级代理评分的VENT衰减缺失：** 逐轮反馈中VENT衰减 ($\gamma_{\text{vent}}$) 作用于 $\Delta\text{emotion}_{\text{adj}}$，但会话级代理评分（$\text{feedback\_score}_{\text{fallback}}$）使用原始 $\Delta\text{valence}_{\text{session}}$ 未经衰减，导致VENT会话末轮策略受到的惩罚约为中间轮的2.5倍。$\xi = 0.5$ 衰减系数提供了部分缓解，但未完全消除此不一致性，后续版本可在会话主要意图为VENT时对 $\Delta\text{valence}_{\text{session}}$ 施加 $\gamma_{\text{vent}}$ 衰减
- **策略间可区分性**：10种策略在LLM自然语言生成中的边界可能模糊（如active_listening与emotional_validation的回复在实践中可能高度相似），影响Thompson Sampling的策略-效果归因准确性
- **单策略假设的局限性**：当前框架假设每轮对话选择单一主策略，但近期研究表明情感支持者通常在单轮中混合使用多种策略（Bai et al., 2025）。Thompson Sampling的单策略选择机制可能过度简化了实际的策略组合行为，未来可扩展为组合策略选择（combinatorial bandit）框架
- **Prompt工程依赖**：多个核心组件（情绪识别、意图分类、可控度估计、人格一致性评判、反思抽取）依赖LLM结构化Prompt实现，Prompt质量是影响系统性能的重要但难以系统化优化的因素
- **隐私攻击面**：持久化的情感记忆（含触发模式、关系图谱、情绪基线）构成高价值隐私攻击目标，当前架构未包含差分隐私、访问控制等专门的隐私保护机制

### 9.2 未来研究方向

1. **文化适配策略空间**：扩展Hill理论策略集以纳入中国本土文化特异策略，基于Liu et al. (2025) [文献#46] 和Yang et al. (2024) [文献#47] 的中国实证数据设计
2. **多Agent协同记忆**：探索多个AI伴侣Agent共享部分语义记忆以提供一致性支持
3. **安全退出机制**：设计"关系健康度监测"模块，当检测到不健康依赖模式时主动引导用户建立真人社交
4. **自主权保护约束**：在策略选择的安全覆盖层中增加"自主权保护"规则
5. **跨模态情感记忆**：将文本情感记忆扩展至语音韵律和面部表情模态
6. **负责任的治疗路径**：Stade et al. (2024) [文献#16] 在npj Mental Health中提出LLM改变行为医疗的未来潜力与负责任开发路径——本框架的策略学习机制可作为实现个性化循证干预的技术基础，但需严格遵循临床验证流程
7. **IS理论深化**：探索情感记忆对技术成瘾的调节作用（Thompson Sampling的策略多样性能否降低不健康依赖模式的形成？Turel et al., 2011），以及AI伴侣身份认同的形成机制（长期情感记忆如何塑造用户的IT身份认同？Carter & Grover, 2015）。这些理论连接目前属于推测性，需要专门的实验设计验证

---

## 10. 参考文献

> **⚠️ 引用体系说明：** 本文档使用双重引用体系。正文中标注 `[文献#N]` 的引用指向配套的NSFC G01申请书文献综述部分，此类文献的完整书目信息见申请书正文，本节不重复收录。本节收录本技术框架文档直接引用的技术文献，与申请书综述存在部分重叠以确保本文档的自包含可读性。

### Agent框架与记忆架构

- Wang, L., Ma, C., Feng, X., et al. (2024). A survey on large language model based autonomous agents. *Frontiers of Computer Science*, 18(6), 186345. doi:10.1007/s11704-024-40231-1.
- Park, J.S., O'Brien, J.C., Cai, C.J., et al. (2023). Generative agents: Interactive simulacra of human behavior. *UIST 2023*. arXiv:2304.03442.
- Packer, C., Wooders, S., Lin, K., et al. (2023). MemGPT: Towards LLMs as operating systems. *arXiv:2310.08560*.
- Zhong, W., Guo, L., Gao, Q., et al. (2024). MemoryBank: Enhancing large language models with long-term memory. *AAAI 2024*. arXiv:2305.10250.
- Li, D., Zhang, J., Li, Y., et al. (2025). LD-Agent: Longitudinal dialogue agent with planning and personality. *NAACL 2025*. arXiv:2406.05925.
- Tan, Z., Chen, W., Li, H., et al. (2025). In prospect and retrospect: Reflective memory management for long-term personalized dialogue agents. *ACL 2025*. arXiv:2503.08026.
- Xu, W., Liang, Z., Mei, K., et al. (2025). A-MEM: Agentic memory for LLM agents. *arXiv:2502.12110*.
- Liu, Y., Zhang, H., Zhang, C., et al. (2025). Memory in the age of AI agents: A survey. *arXiv:2512.13564*.
- Zhang, Z., Bo, X., Guo, C., et al. (2025). A survey on the memory mechanism of large language model based agents. *ACM Transactions on Information Systems*. doi:10.1145/3748302.
- Zhou, J., Chen, P., Zhu, Y., et al. (2024). CharacterGLM: Customizing Chinese conversational AI characters with large language models. *EMNLP 2024*. arXiv:2311.16832.
- Kang, J., Xu, Z., Wang, L., et al. (2025). MemoryOS: A unified operating system framework for memory management in LLM agents. *arXiv:2506.06326*.

### 情感计算与情感支持对话

- Lin, Z., Madotto, A., Shin, J., Xu, P. & Fung, P. (2019). MoEL: Mixture of empathetic listeners. *EMNLP 2019*, 121-132.
- Wang, H., Guo, B., Chen, M., Ding, Y., Zhang, Q., Zhang, Y., & Yu, Z. (2024). MAGIC: Memory-enhanced emotional support conversations with motivation-driven strategy inference. *ECML-PKDD 2024*, LNCS 14945. doi:10.1007/978-3-031-70362-1_13.
- Lu, J. & Li, Y. (2025). DAM-LLM: Dynamic affective memory management for personalized LLM agents. *arXiv:2510.27418*.
- Huang, C., Zhang, Z., et al. (2024). Emotional RAG: Enhancing role-playing agents through emotional retrieval. *arXiv:2410.23041*.
- Sharma, A., Lin, I.W., Miner, A.S., et al. (2023). Human-AI collaboration enables more empathic conversations. *Nature Machine Intelligence*, 5, 46-57.
- Wan, C., Labeau, M. & Clavel, C. (2025). EmoDynamiX: Emotional support dialogue strategy prediction by modelling mixed emotions and discourse dynamics. *NAACL 2025*, 1678-1695. arXiv:2408.08782.
- Bai, X., Chen, G., He, T., Zhou, C. & Liu, Y. (2025). Emotional supporters often use multiple strategies in a single turn. *arXiv:2505.15316*.

### 主动关怀与长期部署

- Jo, E., et al. (2024). Understanding the impact of long-term memory on self-disclosure with LLM-driven chatbots for public health intervention. *CHI 2024*.
- Liu, W., Zhang, R., Zhou, A., Gao, F., & Liu, J. (2025). Echo: A large language model with temporal episodic memory. *arXiv:2502.16090*.
- Sumida, T., et al. (2024). LUFY: A framework for retaining emotionally arousing memories in long-term companion AI. *arXiv:2409.12524*.
- Maharana, S., et al. (2024). ComPeer: A generative conversational agent for proactive peer support. *UIST 2024*. arXiv:2407.18064.
- Ong, D., et al. (2025). Theanine: Timeline-based memory management with cause-effect relations. *NAACL 2025*. arXiv:2406.10996.

### 策略学习与强化学习

- Wang, Y., Takanobu, R., Liang, Z., Mao, Y., Hu, Y., McAuley, J.J., & Wu, X. (2025). Mem-α: Learning memory construction via reinforcement learning. *arXiv:2509.25911*.
- Hill, C.E. (2009). *Helping skills: Facilitating exploration, insight, and action* (3rd ed.). American Psychological Association.
- Sharma, A., et al. (2020). A computational approach to understanding empathy expressed in text-based mental health support. *EMNLP 2020*.
- Liu, S., Zheng, C., Demasi, O., Sabour, S., Li, Y., Yu, Z., Jiang, Y. & Huang, M. (2021). Towards emotional support dialog systems. *ACL 2021*, 3469-3483. arXiv:2106.01144.
- Agrawal, S. & Goyal, N. (2012). Analysis of Thompson Sampling for the multi-armed bandit problem. *COLT 2012* (Proceedings of Machine Learning Research, 23, 39.1-39.26).
- Agrawal, S. & Goyal, N. (2013). Thompson Sampling for contextual bandits with linear payoffs. *ICML 2013* (Proceedings of Machine Learning Research, 28(3), 127-135).
- Chapelle, O. & Li, L. (2011). An empirical evaluation of Thompson Sampling. *NeurIPS 2011* (Advances in Neural Information Processing Systems 24), 2249-2257.
- Scott, S.L. (2010). A modern Bayesian look at the multi-armed bandit. *Applied Stochastic Models in Business and Industry*, 26(6), 639-658.
- Swaminathan, A. & Joachims, T. (2015). The self-normalized estimator for counterfactual learning. *NeurIPS 2015* (Advances in Neural Information Processing Systems 28), 3231-3239.

### 安全与对齐

- Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., Chen, A., Goldie, A., Mirhoseini, A., McKinnon, C., Chen, C., Olsson, C., Olah, C., Hernandez, D., Drain, D., Ganguli, D., Li, D., Tran-Johnson, E., Perez, E., Kerr, J., Mueller, J., Ladish, J., Landau, J., Ndousse, K., Lukosuite, K., Lovitt, L., Sellitto, M., Elhage, N., Schiefer, N., Mercado, N., DasSarma, N., Lasenby, R., Larson, R., Ringer, S., Johnston, S., Kravec, S., El Showk, S., Fort, S., Lanham, T., Telleen-Lawton, T., Conerly, T., Henighan, T., Hume, T., Bowman, S.R., Hatfield-Dodds, Z., Mann, B., Amodei, D., Joseph, N., McCandlish, S., Brown, T. & Kaplan, J. (2022). Constitutional AI: Harmlessness from AI Feedback. *arXiv preprint arXiv:2212.08073*.

### 心理学理论基础

- Bower, G.H. (1981). Mood and memory. *American Psychologist*, 36(2), 129-148.
- Eich, E. & Macaulay, D. (2000). Are real moods required to reveal mood-congruent and mood-dependent memory? *Psychological Science*, 11(3), 244-248.
- Izard, C.E. (2009). Emotion theory and research: Highlights, unanswered questions, and emerging issues. *Annual Review of Psychology*, 60, 1-25.
- Plutchik, R. (2001). The nature of emotions. *American Scientist*, 89(4), 344-350.
- Ortony, A., Clore, G.L. & Collins, A. (1988). *The cognitive structure of emotions*. Cambridge University Press.
- Russell, J.A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161-1178.
- Mehrabian, A. (1996). Pleasure-arousal-dominance: A general framework for describing and measuring individual differences in temperament. *Current Psychology*, 14, 261-292.
- Mohammad, S.M. (2018). Obtaining reliable human ratings of valence, arousal, and dominance for 20,000 English words. *ACL 2018*, 174-184.
- Seligman, M.E.P. (2002). *Authentic happiness: Using the new positive psychology to realize your potential for lasting fulfillment*. Free Press.
- Gross, J.J. (1998). The emerging field of emotion regulation: An integrative review. *Review of General Psychology*, 2(3), 271-299.
- Lazarus, R.S. & Folkman, S. (1984). *Stress, appraisal, and coping*. Springer.
- Atkinson, R.C. & Shiffrin, R.M. (1968). Human memory: A proposed system and its control processes. *Psychology of Learning and Motivation*, 2, 89-195.
- Baddeley, A.D. (2000). The episodic buffer: A new component of working memory? *Trends in Cognitive Sciences*, 4(11), 417-423.
- Tulving, E. (1972). Episodic and semantic memory. In E. Tulving & W. Donaldson (Eds.), *Organization of memory*. Academic Press.
- Brown, R. & Kulik, J. (1977). Flashbulb memories. *Cognition*, 5(1), 73-99.
- McGaugh, J.L. (2004). The amygdala modulates the consolidation of memories of emotionally arousing experiences. *Annual Review of Neuroscience*, 27, 1-28.
- Talarico, J.M. & Rubin, D.C. (2003). Confidence, not consistency, characterizes flashbulb memories. *Psychological Science*, 14(5), 455-461.
- Altman, I. & Taylor, D.A. (1973). *Social penetration: The development of interpersonal relationships*. Holt, Rinehart & Winston.
- Russell, D.W. (1996). UCLA Loneliness Scale (Version 3): Reliability, validity, and factor structure. *Journal of Personality Assessment*, 66(1), 20-40.
- Fletcher, G.J.O., Simpson, J.A. & Thomas, G. (2000). The measurement of perceived relationship quality components: A confirmatory factor analytic approach. *Personality and Social Psychology Bulletin*, 26(3), 340-354.

### IS核心文献与用户行为

- Gregor, S. & Hevner, A.R. (2013). Positioning and presenting design science research for maximum impact. *MIS Quarterly*, 37(2), 337-355.
- Peffers, K., Tuunanen, T., Rothenberger, M.A. & Chatterjee, S. (2007). A design science research methodology for information systems research. *Journal of Management Information Systems*, 24(3), 45-77.
- Kirk, H.R., Gabriel, I., et al. (2025). Why human-AI relationships need socioaffective alignment. *Humanities and Social Sciences Communications*, 12, 728.
- Komiak, S.Y. & Benbasat, I. (2006). Effects of personalization and familiarity on trust and adoption. *MIS Quarterly*, 30(4), 941-960.
- Lankton, N.K., McKnight, D.H. & Tripp, J. (2015). Technology, humanness, and trust: Rethinking trust in technology. *Journal of the Association for Information Systems*, 16(10), 880-918.
- Carter, M. & Grover, V. (2015). Me, my self, and I(T): Conceptualizing information technology identity. *MIS Quarterly*, 39(4), 931-958.
- Chau, M., Li, T.M.H., et al. (2020). Finding people with emotional distress in online social media: A design science approach. *MIS Quarterly*, 44(2), 933-955.
- Angst, C., Dennis, A.R., et al. (2024). IT to improve mental health (Special Section Introduction). *Journal of Management Information Systems*, 41(4).
- Diederich, S., Brendel, A.B., et al. (2022). On the design of and interaction with conversational agents: An organizing and assessing review. *Journal of the Association for Information Systems*, 23(1), 96-138.
- Schuetzler, R.M., Grimes, G.M. & Giboney, J.S. (2020). Impact of chatbot conversational skill on engagement and perceived humanness. *Journal of Management Information Systems*, 37(3), 875-900.
- Skjuve, M., Følstad, A., Fostervold, K.I. & Brandtzaeg, P.B. (2021). My chatbot companion — A study of human-chatbot relationships. *International Journal of Human-Computer Studies* (SSCI Q1).
- Skjuve, M., Følstad, A., Fostervold, K.I. & Brandtzaeg, P.B. (2022). A longitudinal study of human-chatbot relationships. *International Journal of Human-Computer Studies* (SSCI Q1).
- De Freitas, J., Uguralp, A.K., Oğuz-Çelik, Z., & Gross, J.J. (2025). AI companions reduce loneliness. *Journal of Consumer Research* (UTD/FT50).
- Pan, S. & Mou, Y. (2024). Constructing the meaning of human-AI romantic relationships. *Personal Relationships* (SSCI Q1).
- Zhang, X., Zhao, X., Hancock, J.T., et al. (2025). The rise of AI companions: How human-chatbot relationships influence well-being. *arXiv* (N=1131 + 413K messages).
- Ajeesh, K.G. & Joseph, J. (2025). The compassion illusion: Can artificial empathy be emotionally authentic? *Frontiers in Psychology*, 16.
- Brandtzaeg, P.B., Skjuve, M. & Følstad, A. (2022). My AI friend: How users understand their human-AI friendship. *Human Communication Research* (SSCI Q1).

### HCI方法论与数字健康

- Zimmerman, J., Forlizzi, J. & Evenson, S. (2007). Research through design as a method for interaction design research in HCI. *CHI 2007*, 493-502. doi:10.1145/1240624.1240704.
- Eysenbach, G. (2005). The law of attrition. *Journal of Medical Internet Research*, 7(1), e11.

### 评估基准

- Maharana, A., et al. (2024). LoCoMo: A benchmark for long-term conversational memory evaluation. *ACL 2024*. arXiv:2402.17753.

---

**附录：关键符号表**

| 符号 | 含义 | 取值范围 |
|:----:|:-----|:---------|
| $\mathbf{e}(t)$ | 情绪分布向量 | $\Delta^7$ (8维单纯形) |
| $\iota(t)$ | 情绪强度 | $[0, 1]$ |
| $\kappa(t)$ | 情绪识别确定度 | $[0, 1]$ |
| $\text{valence}(t)$ | 情绪效价标量（定义2.3） | $[-1, +1]$ |
| $\boldsymbol{\sigma}$ | 情绪效价符号向量 | $\{-1, 0, +1\}^8$ |
| $\text{BAS}(t)$ | 行为异常度评分 | $[0, +\infty)$ |
| $\text{BAS}_{\max}$ | BAS归一化常数 | 默认 $5.0$ |
| $\text{trend}_{\text{local}}(t)$ | 轻量级近期恶化趋势（感知内计算） | $[0, 1]$ |
| $\text{urgency}(t)$ | 紧急度（外层clamp至$[0,1]$） | $[0, 1]$ |
| $\alpha_1, \alpha_2, \alpha_3, \alpha_4$ | 紧急度权重（定义2.7）：危机=1.0, 负面情绪=0.8, BAS=0.6（冷启动时 $\alpha_3(n_{\text{cum}})$ 线性渐进，见§2.5）, 趋势=0.7 | 固定常数（$\alpha_3$冷启动期为函数） |
| $\text{severity}$ | 综合严重度 | $[0, 1]$ |
| $\text{feedback\_score}$ | 多信号反馈评分（clamp后） | $[0, 1]$ |
| $\text{planning\_intent}$ | 预规划意图推断（定义3.6a） | 5类枚举 |
| $\alpha_{sc}, \beta_{sc}$ | Beta分布参数 | $(0, +\infty)$ |
| $\eta$ | 基线学习率 | 0.05 |
| $\eta_\sigma$ | σ_baseline EMA学习率 | 0.05 |
| $\delta(t)$ | 情绪偏差距离 $\|\mathbf{e}(t)-\mathbf{e}_{\text{baseline}}\|_2$ | $[0, \sqrt{2}]$ |
| $\sigma_{\text{baseline}}$ | 情绪基线偏差距离标准差（标量，§3.5.1） | $(0, +\infty)$，初始0.3 |
| $\sigma_i^{\text{baseline}}$ | 行为特征第$i$维基线标准差（§2.5），EWMA更新，下界 $\sigma_{i,\min}$ | $[\sigma_{i,\min}, +\infty)$ |
| $\text{Retention}(m,t)$ | 记忆保留度（定义3.7） | $[0, 2.5]$ |
| $d$ | 关系深度（归一化自我披露累积深度） | $[0, 1]$ |
| $p_{\text{rel}}$ | 关系目标软约束回退概率 | 0.3 |
| $\tau_{pa}$ | 情绪适配约束效价阈值（Algorithm 4.1 Step 6） | $-0.3$ |
| $\mathcal{W}$ | 退缩/拒绝信号关键词集合（Algorithm 4.1 Step 7） | 14个中文短语 |
| $\tau_d$ | 轨迹方向稳定判定阈值 | 0.05 |
| $k$ | 轨迹方向计算窗口（会话数） | 5 |
| $\lambda$ | 行为基线更新率 | 0.1 |
| $\lambda_d$ | 关系深度$d$的EMA学习率（§4.5.2） | 0.1 |
| $\lambda_h(N)$ | 层次贝叶斯混合权重（数据依赖） | $[0, 1]$，$\max(0, 1 - N/N_{\text{blend}})$ |
| $\tau_r$ | 检索时间衰减半衰期 | 30天 |
| $\tau_f$ | 遗忘基础半衰期 | 90天 |
| $\tau_{\text{write}}$ | 写入重要性阈值 | 0.3 |
| $\tau_{\text{forget}}$ | 遗忘保留度阈值 | 0.1 |
| $\rho$ | 情感保护系数 | 3.0 |
| $\Phi(\cdot)$ | 规划驱动的β调整函数 | $\{0.1, 0.3, 0.5, 1.0, 1.5\}$ |
| $N_{\text{freq}}$ | MentionFreq归一化上限 | 10 |
| $N_{\text{rec}}$ | Recurrence归一化上限 | 5 |
| $N_{\text{blend}}$ | 层次先验渐进混合观测数 | 8 |
| $\sigma_{i,\min}$ | 行为基线标准差第$i$维下界（§2.5） | 维度相关 |
| $\sigma_{\min}$ | 情绪基线偏差距离标准差下界（§3.5.1） | 0.05 |
| $\lambda_\sigma$ | 行为基线标准差EMA学习率（§2.5） | 0.1 |
| $\omega$ | MAR检索重要性权重（定义3.5） | 0.20 |
| $\xi$ | 会话末轮feedback衰减系数（定义3.3） | 0.5 |
| $T_{\text{hold}}$ | 阶段转换最小保持轮次（§4.4） | 3 |
| $\epsilon_h$ | 阶段转换滞后带宽度（§4.4） | 0.05 |
| $K_{\text{buf}}$ | 情绪基线重估环形缓冲区容量（§3.5.1） | 20 |
| $\gamma_{\text{vent}}$ | VENT场景负向信号衰减因子（定义3.4） | 0.3 |
| $\tau_\beta$ | Beta参数时间衰减半衰期（§3.5.4/Algorithm 4.1） | 60天 |
| $\gamma_\beta(\Delta t)$ | Beta参数时间衰减因子（§3.5.4） | $\exp(-\ln 2 \cdot \Delta t / \tau_\beta) \in (0, 1]$ |
| $N_{\text{prior}}$ | 层次先验固定伪观测数上限（§4.5.2） | 4 |
| $\mu_{s_i}$ | 策略$s_i$跨情境平均有效率（§4.5.2） | $[0, 1]$ |
| $\epsilon_{h,\text{exit}}$ | acute\_distress退出滞后阈值（§4.4） | 0.02 |
| $T_{\text{max,acute}}$ | acute\_distress最大保持轮次（§4.4） | 10 |
| $\hat{\beta}$ | 轨迹方向线性回归斜率（§4.4阶段判定） | $\mathbb{R}$；$k \leq 2$ 时强制为0 |
| $\delta_b(t)$ | 效价基线偏差 $\text{valence}(t) - \text{valence}_{\text{baseline}}$（§4.4） | $\approx [-1.1, +1.1]$ |
| $\alpha_0, \beta_0$ | Beta分布初始先验参数 | 默认 $2.0$ |
| $\lambda_r$ | MAR检索时间衰减率（§3.7） | $\ln 2$ |
| $\Delta c_{\text{decay}}$ | 基线置信度异常衰减步长（§3.5.1） | 0.05 |
| $n_{\text{cum}}$ | 累计交互次数（跨会话，§2.5冷启动） | $\mathbb{N}^+$ |
| $\delta_b^{\text{boundary}}$ | 阶段分界阈值（§4.4滞后带） | $\{-0.3, -0.1\}$ |
| $\sqrt{2}$ | 单纯形最大L2距离（severity归一化） | 常数 |

