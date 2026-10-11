# 人选祸 · Alea

> 人选的只有祸，福由天定。

一个帮助人看清选择代价的 AI 决策 Agent。它不会告诉你"怎样收益最大"，
而是通过多轮追问，帮你看清自己**想要什么（贪婪）**、**害怕失去什么（恐惧）**，
把每个选项必须付出的代价摊开，最后给出一个明确、可追溯、风险有边界的
**选祸建议**。

核心思想来自路飞《人选天选论》的「人选祸，天选福」：
**能不能如愿不由人决定，但承担哪一个代价，人可以选。**

---

## 目录

- [它实际是怎么工作的](#它实际是怎么工作的)
- [快速开始](#快速开始)
- [验收：五条清单怎么验](#验收五条清单怎么验)
- [架构](#架构)
- [长期记忆系统](#长期记忆系统)
- [RAG 与引用诚实性](#rag-与引用诚实性)
- [API 一览](#api-一览)
- [测试](#测试)
- [项目结构](#项目结构)
- [设计取舍](#设计取舍)
- [V1 明确不做的事](#v1-明确不做的事)

---

## 它实际是怎么工作的

一轮对话里，LangGraph 会跑一条**有状态、可恢复、每次必然结束**的流水线：

```
START → understand → recall → route ─┬→ explore ──┐
                                     ├→ confirm ──┼→ finalize → END
                                     ├→ safety ───┘
                                     └→ knowledge → decide → finalize
```

- **understand**：把对话读成结构化判断（困境 / 选项 / 贪婪 / 恐惧 / 约束 / 矛盾 / 充分度）。
  用户在这一轮纠正过的判断会被真正替换，而不是继续沿用旧版本。
- **recall**：按当前困境从长期记忆里取出**相关的几条**，不是把所有历史塞进上下文。
- **route**：纯规则决定这一轮该"继续问"还是"去确认"还是"给建议"。没有 LLM 调用，
  因此轮次、成本和行为都是可预期的。
- **explore / confirm**：追问，或者把 Agent 自己推测出来的关键判断拿去请你核对。
- **knowledge**：真正的 tool calling —— 模型自己决定检索词，调用 `search_theory` /
  `search_cases` / `recall_user_memory`，最多 3 轮。
- **decide**：输出结构化建议（含"暂缓"和"停止"两种非建议结论）。
  如果关键推断还没被你确认，或建议可能造成不可逆后果，它会**拒绝给结论**并说明还缺什么。
- **finalize**：从这一轮提炼值得长期保存的信息，去重、冲突处理后落库。

关键约束：**不存在无限自由循环**。每轮一定走到 `finalize`，并有软/硬轮次上限
（默认 8 / 16）。到了上限还没问清，Agent 会输出一个明确的「暂缓」结论——而不是硬编一个答案。

---

## 快速开始

### 前置条件

- Python 3.12+、Node.js 20+
- PostgreSQL 16 + pgvector（三种方式任选，见下）
- 一个 OpenAI 兼容协议的模型 API Key

### 1. 配置

```bash
cp .env.example backend/.env
# 编辑 backend/.env，至少填写 LLM_API_KEY、EMBEDDING_API_KEY
```

`LLM_BASE_URL` / `LLM_MODEL` 可以指向任何兼容 OpenAI 协议的端点
（OpenAI、DeepSeek、通义千问、Moonshot、vLLM、Ollama 都可以）。

### 2. 启动数据库

**方式 A：Docker（推荐）**

```bash
docker compose up -d db
```

**方式 B：已经有 PostgreSQL**，建库即可：

```bash
createdb alea && psql alea -c 'CREATE EXTENSION IF NOT EXISTS vector'
```

**方式 C：本机没有 Docker、也没有 PostgreSQL**（本项目自带的开发辅助）：

```bash
cd backend && .venv/Scripts/python -m scripts.local_postgres start
```

它用 `pgserver` 附带的 PostgreSQL 二进制在 `.data/pgdata` 起一个固定端口的实例
（端口 55432，与 Docker 映射一致，所以同一份 `.env` 通用）。停止用 `stop`，清空用 `delete`。

### 3. 后端

```bash
cd backend
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"     # Windows
# source .venv/bin/activate && pip install -e ".[dev]"   # macOS / Linux

.venv/Scripts/python -m alembic upgrade head        # 建表
.venv/Scripts/python -m scripts.ingest_knowledge    # 索引知识库
.venv/Scripts/python -m scripts.serve --reload      # 启动 API（含热重载）
```

> **为什么不用 `uvicorn app.main:app`？**
> LangGraph 的 PostgreSQL checkpointer 基于 psycopg 的**异步**驱动，而它在 Windows 上
> 无法运行在默认的 Proactor 事件循环上——连接永远建立不起来，只会在 30 秒后抛
> `PoolTimeout`。`scripts/serve.py` 会先切换到 `SelectorEventLoop` 再启动 uvicorn。
> Linux / macOS / Docker 里直接用 uvicorn 也没问题。

打开 <http://127.0.0.1:8000/docs> 可以看到全部接口，<http://127.0.0.1:8000/api/health>
会告诉你数据库、模型密钥、知识库片段数是否就绪。

想先用命令行确认真实模型链路是否通：

```bash
.venv/Scripts/python -m scripts.smoke_llm
```

它会跑完一段四轮的困境对话，并逐轮打印 Agent 的判断、检索到的理论来源和最终建议。

### 4. 前端

```bash
cd frontend
npm install
npm run dev
```

打开 <http://localhost:3000>。

### 5. 或者一条命令全起

```bash
docker compose up --build
```

API 容器启动时会自动跑迁移并索引知识库。

---

## 验收：五条清单怎么验

| 验收项 | 怎么验 | 期望看到什么 |
| --- | --- | --- |
| **同一个困境，Agent 动态追问而不是固定问答** | 在页面上描述一个真实困境，连续回答几轮；再把同样的困境换一种问法新开一个会话 | 提问内容随你说的内容变化；追问方向跟随你给出的信息，不是固定的问卷顺序 |
| **用户纠正贪婪或恐惧后，Agent 真正修正状态** | 当右侧"需要你确认"出现推测时，点「不完全是，我来改」并写入你的说法（或直接在对话里说"不是这样，我是怕…"） | 旧版本从右侧消失，新版本标记为「已纠正 / 用户亲述」，后续建议不再引用旧说法 |
| **RAG 确实检索到知识库内容并给出真实来源** | 打开「长期记忆 → 知识库」，在"检索验证"里输入"代价 不可逆"；再回到对话，看右侧「引用」标签 | 两个入口返回同一批片段与相似度；引用卡片显示文件名、章节、来源类型（原文摘录 / 整理笔记 / 案例），可展开原文摘录 |
| **最终给出明确选祸建议，含理由、代价、风险边界** | 聊到信息充分（右侧"信息充分度"过 62%） | 右侧顶部出现建议卡：结论一句话、必须放弃什么、风险等级与应对、底线、各选项代价等级与可逆性、以及重新评估的条件 |
| **关闭会话后新会话能检索、使用并修改长期记忆** | 结束一个会话（浏览器标签任意关掉）；新建会话描述一个相关的新困境；去「长期记忆」页面把它改掉或删掉 | 第二个会话的右侧「记忆」标签出现前一个会话确认过的条目；改动后新会话按新内容检索；轨迹可在「变更记录」里看到 |

一张覆盖这五条的自动化核对见 [docs/acceptance.md](docs/acceptance.md)。

---

## 架构

模块化单体，前后端分离：

```
frontend (Next.js 15 / React 19 / Tailwind 4)
        │  REST + SSE
        ▼
backend (FastAPI)
  ├── api/          HTTP 层：路由、依赖注入、错误映射
  ├── services/    编排：会话事务、知识库摄取、容器装配
  ├── agent/       LangGraph：状态、节点、路由、提示词、工具
  ├── memory/      长期记忆：提取、去重/冲突、检索
  ├── rag/         知识库：加载、切分、检索
  ├── llm/         模型工厂：聊天与向量，OpenAI 协议
  ├── db/          SQLAlchemy 模型、仓储、迁移
  └── core/        配置、日志（含脱敏）、错误
        │
        ▼
PostgreSQL + pgvector  ·  LangGraph Checkpointer 表
```

细节见 [docs/architecture.md](docs/architecture.md)。

---

## 长期记忆系统

不是"把聊天记录存下来"就算记忆。这里分三层：

| 层 | 存在哪 | 作用 |
| --- | --- | --- |
| **短期** | LangGraph checkpointer（按 `thread_id` = 会话 id） | 会话状态、可跨进程恢复，包括结构化分析而不只是消息 |
| **长期语义** | `memories` 表 + pgvector | 目标 / 价值观 / 贪婪 / 恐惧 / 约束 / 偏好 / 事实 |
| **历史决策** | `decision_events` 表 + pgvector | 每个结束会话压缩成一条"当时选了什么、代价是什么" |

每条记忆都带三样元数据，这是整个系统的诚实性基础：

- `provenance`：`user_stated`（用户亲述）/ `user_confirmed`（用户确认过的推测）/ `agent_inferred`（Agent 推测）
- `status`：`candidate`（待确认）/ `active` / `rejected`（用户否认）/ `superseded`（被更新版本取代）
- `evidence`：支撑这条记忆的用户原话

**只有非 `agent_inferred` 的记忆才会被当作事实使用**；推测类的会先作为"确认项"请你核对，
你点了「是我」或改了说法，它才升级为 `user_confirmed` / `user_stated`。

写入时的处理（`memory/consolidator.py`）：

- 余弦相似度 ≥ 0.90 → **合并**（补证据、提置信度、保留更具体的措辞、升级 provenance）
- 相似度在 0.72–0.90 → 交给模型判断是"同一件事 / 更新 / 矛盾 / 无关"，据此合并、取代或并存
- 被取代的旧记忆状态变为 `superseded` 并保留，`superseded_by` 指向新记忆
- 每一次变更都写进 `memory_audit`（谁改的、改前改后、为什么）

读取时**按相关性检索**，不是全量注入：`memory_top_k` 默认 6 条，低于相似度阈值直接不返回。

你可以在「长期记忆」页面查看、修改、软删除、恢复每条记忆，并看到它的完整变更记录。
删除是软删除，审计轨迹保留。详细设计见 [docs/memory.md](docs/memory.md)。

---

## RAG 与引用诚实性

知识库是 `backend/knowledge/` 下的 Markdown，带 front-matter：

```yaml
---
title: 选祸与天选福
source: 人选天选论（路飞）
source_type: curated_note    # verified_excerpt | curated_note | case
verified: false
---
```

- 按 Markdown 标题切分，标题路径作为"引用位置"一并保存，所以每条引用都能指到具体章节。
- 相似度低于 `RAG_MIN_SCORE`（默认 0.15）的片段会被丢弃——宁可不引用，也不拿弱匹配充当依据。
- **`source_type` 会一路传到用户界面**：`verified_excerpt` 才允许说"原文提到"，
  `curated_note` 只能说"资料里把这一点整理为"。

⚠️ **当前仓库里的 `01-core/`、`02-method/` 全部是 `curated_note`**，
即本项目对「人选祸，天选福」思想的整理与工程化表述，**不是原书原文**，标注为 `verified: false`。
这样做是为了让 Agent 永远不可能把整理内容说成原文。要引用原文，
请按 `knowledge/_templates/verified-excerpt-template.md` 自行加入核对过的段落。

---

## API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 数据库 / 模型密钥 / 知识库状态 |
| GET | `/api/config` | 非敏感的运行时配置（模型名、阈值） |
| POST | `/api/sessions` | 新建会话 |
| GET | `/api/sessions` | 会话列表 |
| GET | `/api/sessions/{id}` | 会话详情（含结构化分析、建议、消息） |
| PATCH / DELETE | `/api/sessions/{id}` | 重命名 / 归档 / 软删除 |
| POST | `/api/sessions/{id}/messages` | 跑一轮，返回完整结果 |
| POST | `/api/sessions/{id}/stream` | 跑一轮，SSE 流式返回 |
| POST | `/api/sessions/{id}/confirmations/{cid}` | 确认 / 纠正 / 否认一条推测 |
| GET / POST / PATCH / DELETE | `/api/memories` | 长期记忆管理 |
| POST | `/api/memories/search` | 按语义检索记忆 |
| GET | `/api/memories/{id}/history` | 某条记忆的变更轨迹 |
| POST | `/api/memories/{id}/restore` | 恢复已删除的记忆 |
| GET | `/api/decisions` | 历史决策 |
| PATCH | `/api/decisions/{id}/outcome` | 记录"后来怎么样了" |
| GET | `/api/knowledge/documents` | 知识库文档清单 |
| POST | `/api/knowledge/search` | 直接验证检索结果 |
| POST | `/api/knowledge/reindex` | 增量/强制重建索引 |

SSE 事件类型：`start` · `node` · `token` · `memory` · `decision` · `done` · `error`。

---

## 测试

```bash
cd backend
.venv/Scripts/python -m pytest                        # 全部
.venv/Scripts/python -m pytest -m "not llm and not db" # 只跑不需要外部依赖的
```

标记说明：

- `-m db`：需要真实 PostgreSQL（`scripts.local_postgres start` 或 Docker）
- `-m llm`：需要真实模型端点，会消耗 token

覆盖的关键场景：跨会话记忆检索、记忆纠正、状态恢复、用户隔离、
危险建议拦截、记忆去重与冲突、以及完整的 Agent 流程。

---

## 项目结构

```
alea/
├── backend/
│   ├── app/
│   │   ├── agent/          LangGraph 状态 / 节点 / 路由 / 提示词 / 工具
│   │   ├── api/            路由、依赖、异常映射
│   │   ├── core/           配置、日志脱敏、错误
│   │   ├── db/             ORM 模型、仓储、会话管理
│   │   ├── llm/            聊天与向量模型工厂
│   │   ├── memory/         提取、去重冲突、检索
│   │   ├── rag/            加载、切分、索引
│   │   ├── schemas/        Pydantic 契约
│   │   └── services/       编排层与容器
│   ├── alembic/            数据库迁移
│   ├── knowledge/          公共知识库（Markdown）
│   ├── scripts/            本地数据库、知识库索引、启动入口、端到端冒烟
│   └── tests/
├── frontend/
│   └── src/
│       ├── app/            页面（对话、会话、记忆管理）
│       ├── components/     UI 组件
│       └── lib/            API 客户端与类型
├── docs/                   架构、记忆、工作流、验收
└── docker-compose.yml
```

---

## 设计取舍

**为什么长期记忆不放在 LangGraph Store 里？**
Checkpointer 用了（会话状态持久化与恢复，这正是它擅长的）。但长期记忆需要
来源追溯、去重合并、冲突取代、审计轨迹、软删除恢复和按用户隔离——
这些用 KV + 向量的 namespace 表达会很别扭，还要再维护一套关系表。
所以记忆落在自己的表里，一个真相来源。这是**有意的取舍，不是遗漏**。

**为什么决定阶段的正文由模板渲染，而不是再让模型写一遍？**
结构化结论和给用户看到的文字必须一致。让模型写两遍，早晚会出现"卡片说 A、正文说 B"。
所以 `decide` 只产出结构化对象，正文由 `render_decision()` 从对象渲染。

**为什么路由是纯规则？**
"该继续问还是该给结论"是一个可以用充分度和轮次表达的判断。
交给规则，行为可预测、可测试、不额外烧 token；交给模型，就会开始漂移。

**为什么安全路径有硬编码关键词兜底？**
模型自己会做风险识别，但这条路径失败的代价太高。关键词命中只会**增加**风险标记，
永远不会降低，所以是纯增益。

---

## V1 明确不做的事

24 小时/7 天/30 天行动计划、长期行动追踪与定时复盘、Multi-Agent、
Redis/Celery、知识图谱、社区与付费、完整注册登录、自动爬取内容、模型微调。

数据模型和权限边界已经按多用户设计（每张表都有 `user_id`，每个查询都按用户过滤），
但 V1 只跑本地单用户 Profile，且**不应直接暴露到公网**。

---

## 许可

私人项目，未附许可协议。
