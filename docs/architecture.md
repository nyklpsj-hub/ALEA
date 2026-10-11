# 架构

## 全景

```
┌──────────────────────────────────────────────────────────────┐
│ 浏览器 (Next.js 15 App Router / React 19 / Tailwind 4)         │
│   对话工作台 · 分析面板 · 长期记忆管理 · 知识库检索验证          │
└───────────────┬──────────────────────────────────────────────┘
                │ REST (JSON) + SSE (text/event-stream)
┌───────────────▼──────────────────────────────────────────────┐
│ FastAPI                                                       │
│  api/        路由 · 依赖注入 · 异常 → 统一错误体                │
│  services/   会话事务 · 知识库摄取 · 组合根(AppContainer)       │
│  agent/      LangGraph：状态 · 节点 · 路由 · 提示词 · 工具      │
│  memory/     提取 · 去重冲突 · 检索                            │
│  rag/        加载 · 切分 · 检索                                │
│  llm/        聊天/向量模型工厂（OpenAI 协议）                   │
│  db/         ORM 模型 · 仓储 · 引擎                            │
│  core/       配置 · 日志脱敏 · 错误                             │
└───────────────┬──────────────────────────────────────────────┘
                │ SQLAlchemy 2 (async, asyncpg)
┌───────────────▼──────────────────────────────────────────────┐
│ PostgreSQL 16 + pgvector                                      │
│  业务表：users sessions messages memories memory_audit         │
│          decision_events knowledge_documents knowledge_chunks │
│  LangGraph：checkpoints / checkpoint_writes / checkpoint_blobs│
└──────────────────────────────────────────────────────────────┘
```

## 分层规则

1. **api → services → (agent | memory | rag) → db**。方向不可逆：agent 不知道 HTTP 的存在，
   rag 不知道会话的存在。
2. **提示词集中在 `agent/prompts.py` 与 `memory/prompts.py`**。没有散落在业务代码里的字符串，
   因此改提示词不会顺带改逻辑。
3. **契约集中在 `schemas/`**。同一个 Pydantic 模型既是 LLM 结构化输出的目标，
   也是 LangGraph 状态的字段类型，也是 API 的响应体。三方不可能漂移。
4. **幂等与可重放**。检索、切分、摄取都是幂等的；`ingest --force` 只是绕过校验和检查。

## 依赖注入

`AppContainer`（`services/container.py`）是唯一的组合根，持有 settings、database、
agent runtime、memory service、knowledge service、chat service。

- HTTP 层通过 FastAPI 依赖拿到它（`api/deps.py`）。
- 图节点通过 LangGraph 的 `config["configurable"]["deps"]` 拿到 `AgentDeps`。

没有全局单例查找，也没有循环导入。

## 模型接入

`llm/client.py` 是唯一的模型出口：

- `build_chat_model()` —— `ChatOpenAI`，`base_url` / `model` / `api_key` 全部来自环境变量，
  因此任何兼容 OpenAI 协议的端点都可以直接接入。
- `get_chat_model()` —— 按 (temperature, max_tokens, streaming) 缓存实例。
- `embed_texts()` / `embed_query()` —— 按批次切分，带指数退避重试；
  `EMBEDDING_PROVIDER=local` 时切换到 fastembed 的离线模型。
- `call_structured()` —— 结构化输出 + 重试 + 明确的失败语义。

所有模块都以 `from app.llm import client as llm` 的方式引用，而不是绑定具体函数。
这样替换提供方（生产替换大模型、测试注入假模型）只需要改一处。

## 错误处理

| 层 | 做法 |
| --- | --- |
| 领域层 | 抛 `AppError` 子类，带稳定的 `code` 与 HTTP 状态 |
| API 层 | 统一异常处理器，输出 `{"error": {"code", "message", "details"}}` |
| Agent 层 | 单个节点的失败被降级（记忆检索失败仍可继续对话），不冒泡成 500 |
| 上游 | 超时、瞬时连接错误用 tenacity 重试；最终失败映射为 `llm_unavailable` |

## 日志与隐私

`core/logging.py` 提供 JSON 格式化器，默认对用户文本字段（`content`、`user_text`、
`query`、`preview` 等）做脱敏，只保留长度和前缀。

只有在显式设置 `LOG_USER_CONTENT=true` 时才输出原文——这是本机调试开关，
不该在共享环境打开。

## 平台注意事项：Windows 的事件循环

LangGraph 的 PostgreSQL checkpointer 建立在 psycopg 的**异步**驱动上，而 psycopg 明确
不支持 Windows 默认的 `ProactorEventLoop`（它没有实现 `add_reader`/`add_writer`）。
症状很有迷惑性：数据库明明可连（asyncpg 正常、psql 正常、迁移正常），但
`AsyncConnectionPool.open()` 会一直等到超时，最后抛出一个信息量很低的 `PoolTimeout`。

处理方式：

1. `app/core/compat.py::configure_event_loop_policy()` 在 Windows 上切换到
   `SelectorEventLoop`，必须在**任何事件循环被创建之前**调用。
2. 因此启动入口是 `python -m scripts.serve`（它先设策略再交给 uvicorn），
   Docker / Linux 里则不受影响。
3. `AgentRuntime.start()` 在创建连接池前会检查当前循环类型，记录一条带修复建议的
   error 日志，并把连接失败转成带 hint 的 `ConfigurationError`，避免只看到超时。
4. `tests/conftest.py` 在模块导入时调用同一个函数。

## SSE 事件协议

`POST /api/sessions/{id}/stream` 返回 `text/event-stream`，每条 `data:` 是一个 JSON：

| type | 载荷 | 用途 |
| --- | --- | --- |
| `start` | `session_id`, `user_message_id` | 用户消息已落库 |
| `node` | `node`, `phase` | 图节点推进，界面显示"正在追问/正在检索" |
| `token` | `node`, `text` | 流式文本增量 |
| `memory` | `changes[]` | 本轮写入的长期记忆 |
| `decision` | `decision` | 结论（含 `hold_safety`） |
| `done` | `message`, `analysis`, `decision`, `citations`, `memory_changes` | 收尾，界面一次性对齐 |
| `error` | `message` | 可读的失败原因 |

前端不用 EventSource（它只支持 GET），而是 `fetch` + `ReadableStream` 自行解析帧。

## 为什么不引入更多基础设施

V1 刻意不引入 Redis、Celery、消息队列或独立向量库：

- 检索规模是几百到几千条切片，pgvector 的 HNSW 索引足够，且和业务数据同库同事务。
- 长耗时工作（知识库摄取）只在启动时和手动触发时发生，不需要任务队列。
- 图执行是请求内的，按会话串行，不需要分布式锁。

如果将来需要，`services/` 边界就是切分点：摄取可以搬进 worker，
检索可以换成独立向量服务，而 agent 与 api 不受影响。
