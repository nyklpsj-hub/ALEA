# 长期记忆设计

## 为什么不用"把聊天记录存下来"

聊天记录能回答"当时说了什么"，但回答不了决策真正需要的三个问题：

1. **这个人稳定的偏好与恐惧是什么？**（跨会话不变的东西）
2. **上次类似的选择，他最后选了哪个、付了什么代价？**
3. **哪些是他说过的，哪些是我猜的？**

所以记忆被拆成三部分，各有各的存储和检索路径。

## 三层结构

### 短期记忆 —— LangGraph Checkpointer

- 存的是**图状态**：结构化分析、确认项、已问过的问题、轮次、引用列表。
- 按 `thread_id = session_id` 分隔，每一步之后自动 checkpoint。
- 作用：会话可跨请求、跨进程恢复；而且恢复的不只是消息，是**推理状态本身**。

### 长期语义记忆 —— `memories` 表 + pgvector

八类：`goal` / `value` / `greed` / `fear` / `constraint` / `decision` / `preference` / `fact`。

每条记忆携带三样诚实性元数据：

| 字段 | 取值 | 含义 |
| --- | --- | --- |
| `provenance` | `user_stated` | 用户自己说的 |
| | `user_confirmed` | Agent 推测、用户明确确认过的 |
| | `agent_inferred` | Agent 推测，**未经确认** |
| `status` | `candidate` | 待用户确认（仅 `agent_inferred` 会进入） |
| | `active` | 生效中 |
| | `rejected` | 用户明确否认 |
| | `superseded` | 被更新的版本取代 |
| `evidence` | 文本 | 支撑这条记忆的用户原话（逐字引用） |

**只有 `provenance != agent_inferred` 的记忆会被当作事实使用。**
推测类记忆在注入提示词时会被标注为
「Agent 推测（未经确认，只能作为假设提出）」，并且永远不会成为最终建议的主要理由。

### 历史决策记忆 —— `decision_events` 表 + pgvector

一个会话给出 `recommend` 结论后，会被压缩成一条记录：
处境、选项、最终选择、当时的贪婪/恐惧、代价、风险、小结，以及可回填的 `outcome`
（"后来怎么样了"）。新会话遇到相似处境时，这条记录会被检索出来作为参考。

## 写入流程

```
对话轮次结束 (finalize 节点)
   ↓ 提取：只挑"长期成立 + 与决策相关 + 有原话依据"的信息，每轮最多 5 条
   ↓ 嵌入
   ↓ 去重 / 冲突判定（consolidator.py）
   ↓ 落库 + 写审计
```

### 去重与冲突

| 与已有记忆的余弦相似度 | 处理 |
| --- | --- |
| ≥ 0.90 | **合并**：补充 evidence、取更高的 provenance 与置信度、保留更具体的措辞、重算向量 |
| 0.72 – 0.90 | 交给模型逐条判断：`duplicate` / `update` / `contradict` / `distinct` |
| < 0.72 | 直接新建 |

- `duplicate` → 合并进那条
- `update` / `contradict` → 新建，并把旧记忆置为 `superseded`、`superseded_by` 指向新记忆
- `distinct` → 两条都保留

判断被刻意保守化：不确定是否矛盾时一律 `distinct`——宁可多留一条，也不误删用户的历史。

### 审计

`memory_audit` 是只追加的流水：`created` / `updated` / `confirmed` / `rejected` /
`merged` / `superseded` / `deleted` / `restored`，每条都带 `actor`（`user` 或 `agent`）、
`before` / `after` 快照和原因。界面上「变更记录」按钮展示的就是它。

## 读取流程

```
当前困境 + 用户最新一句话 + 前两条贪婪/恐惧
   ↓ 嵌入
   ↓ memories 表按余弦相似度取 top-k（默认 6），低于 MEMORY_MIN_SCORE 直接丢弃
   ↓ decision_events 表取 top-k/2
   ↓ 命中即记一次使用（last_used_at / use_count）
   ↓ 按 provenance 标注后注入提示词
```

永远不会把全部历史塞进上下文。相关性不足时就是空——空是诚实的，"硬凑一条"不是。

## 确认闭环

推断性、敏感的信息必须经过用户确认才能落库为事实。两条路径都能走通：

1. **界面按钮**：右侧面板的确认卡片 → `POST /api/sessions/{id}/confirmations/{cid}`
   → 写入 `user_confirmed`（确认）或 `user_stated`（改写）记忆，同时把确认状态写回
   checkpointer 与 `sessions.analysis`。
2. **对话文本**：用户直接说"不是这样，我是怕……" → `understand` 节点更新状态并把旧版本
   的 id 填进 `retired_ids`（明确作废），`finalize` 提取时赋予 `user_confirmed` 或
   `user_stated`。

`retired_ids` 是这套设计的关键一环：没有它，Agent 无法区分"模型这轮忘了重复"和
"这条已经作废了"，会把用户已经纠正过的旧判断重新捡回来。

## 用户隔离

- 每张表都有 `user_id`；每个仓储函数都把它作为必需参数。
- 检索 SQL 一律带 `WHERE user_id = :user_id`。
- 没有任何"按 id 取记忆"的路径不校验 user_id（400/404 而不是 403，避免泄露存在性）。
- V1 用固定的本地 profile，但 `api/deps.get_current_user_id()` 是唯一的身份来源，
  将来接入真实认证只需要替换它。

## 与公共知识库的隔离

- 完全不同的表（`memories` vs `knowledge_chunks`）、不同的仓储模块、不同的检索函数。
- 提示词层面也分开：理论知识来自工具返回，用户记忆来自 recall 节点，两者在提示词里
  位于不同小节，标注方式不同。
- 因此不存在"用户私有信息被写进公共知识库"的代码路径——它们连写入函数都不是同一个。
