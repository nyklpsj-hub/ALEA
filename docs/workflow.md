# LangGraph 工作流

## 图

```
                       ┌─────────────┐
  START ──────────────▶│ understand  │  读对话 → 结构化理解
                       └──────┬──────┘
                              ▼
                       ┌─────────────┐
                              │   recall    │  按当前困境检索长期记忆
                       └──────┬──────┘
                              ▼
                       ┌─────────────┐
                              │   route     │  纯规则决定这一轮怎么走
                       └──────┬──────┘
              ┌───────────────┼───────────────┬────────────────┐
              ▼               ▼               ▼                ▼
        ┌──────────┐   ┌───────────┐   ┌──────────┐    ┌───────────┐
        │ explore  │   │  confirm  │   │  safety  │    │ knowledge │
        └────┬─────┘   └─────┬─────┘   └────┬─────┘    └─────┬─────┘
             │               │              │                ▼
             │               │              │          ┌───────────┐
             │               │              │          │  decide   │
             │               │              │          └─────┬─────┘
             └───────────────┴──────────────┴────────────────┘
                              ▼
                       ┌─────────────┐
                              │  finalize   │  落库 + 写记忆
                       └──────┬──────┘
                              ▼
                             END
```

**一轮用户消息 = 一次完整通过。** 没有无限循环，没有"agent 自己决定要不要再来一轮"。

## 状态模型

`AleaState`（`agent/state.py`）是唯一的真相，随每一步 checkpoint：

| 分组 | 字段 |
| --- | --- |
| 标识 | `user_id`, `session_id` |
| 对话 | `messages`（唯一带 reducer：`add_messages`）, `latest_user_text` |
| 分析 | `dilemma`, `greeds`, `fears`, `constraints`, `contradictions`, `open_questions`, `confirmations`, `retired_ids`, `sufficiency`, `risk_flags` |
| 记忆 | `recalled_block`, `recalled_memory_refs`, `memory_changes` |
| 知识 | `citations`, `knowledge_notes`, `tool_calls` |
| 结论 | `decision` |
| 控制 | `phase`, `turn_count`, `route`, `route_reason`, `assistant_reply`, `asked_confirmation_ids`, `asked_directions`, `user_requests_conclusion` |

约定：结构化字段一律是 Pydantic 对象而不是散文，因为前端要渲染它、用户要纠正它、
审计要对比它。只有 `messages` 有 reducer；其余字段由拥有它的节点整体覆写。

## 节点职责

| 节点 | 调用模型？ | 做什么 | 不做什么 |
| --- | --- | --- | --- |
| `understand` | 是（结构化） | 更新困境/动机/约束/矛盾/充分度；解析用户对确认项的确认、纠正、否认；合并风险信号 | 不回复用户 |
| `recall` | 否（仅嵌入） | 按当前困境检索长期记忆与历史决策，渲染为带 provenance 的文本块 | 不写入 |
| `route` | **否** | 决定下一步 | 不做语义判断 |
| `explore` | 是（流式文本） | 复述 + 提出当前最关键的 1–2 个追问 | 不给建议、不列方案 |
| `confirm` | 是（流式文本） | 把 Agent 自己的推测逐条拿去核对，注明依据 | 不诱导、不评价 |
| `knowledge` | 是（工具调用） | 真·tool calling：`search_theory` / `search_cases` / `recall_user_memory`，最多 3 轮，然后归纳要点 | 不编造来源；检索不到就直说 |
| `decide` | 是（结构化） | 产出结构化结论（`recommend` / `defer` / `hold_safety`） | 不写散文（正文由模板渲染） |
| `safety` | 是（流式文本） | 停止决策分析，转为求助引导 | 不下诊断、不给建议 |
| `finalize` | 是（提取） | 提炼长期记忆并落库；重置本轮累积器 | 不在风险状态下写记忆 |

## 路由规则

`route()` 是纯函数，只用状态里的数值和标志：

```
1. 有 risk_flags                      → safety
2. turn_count ≥ HARD_LIMIT            → citations ? decide : knowledge
3. 用户要求结论 且 sufficiency ≥ 0.35 且已有选项 → citations ? decide : knowledge
4. sufficiency < 0.35                 → explore
5. 存在未问过的待确认项                 → confirm
6. sufficiency < SUFFICIENCY_THRESHOLD
     ├─ turn_count ≥ SOFT_LIMIT        → citations ? decide : knowledge
     └─ 否则                           → explore
7. 还没有检索过理论                     → knowledge
8. 否则                               → decide
```

为什么用规则而不是让模型决定：这是"够不够"的判断，用数值和计数就能表达；
交给模型只会让它漂移、变慢、变贵。规则可读、可测、可解释——界面上的
「阶段：追问中」直接来自它。

`SUFFICIENCY_THRESHOLD` 默认 0.62，`SOFT_LIMIT` 8，`HARD_LIMIT` 16，都可通过环境变量调整。

## 边界控制（不靠提示词，靠代码）

`_enforce_decision()` 在模型给出结论之后再检查一遍：

1. `sufficiency < THRESHOLD × 0.75` → 强制降级为 `defer`
2. 引用必须来自**本轮实际检索到的** chunk（`citations` 与 state 求交集）——编造的引用会消失
3. `recommend` 必须有明确的 `chosen_option_id`/`chosen_label`，否则降级为 `defer`
4. 存在 `risk_flags` → 一律 `hold_safety`

`_reconcile()` 在模型的阅读结果进入状态之前检查：

1. 没有原话依据的动机**不能**声称是用户亲述（自动降为 `agent_inferred`、置信度 ≤ 0.6）
2. 用户已确认的条目，即使模型这轮没重复，也会被保留
3. 用户已作废的条目（`retired_ids`）即使被模型重新写出来，也会被丢弃
4. 每轮最多 3 个确认项

安全路径还有一个**关键词兜底**（`screen_risk`）：自杀/自残/伤人/家暴/成瘾等模式命中时，
直接追加风险标记。它只会增加标记，永远不会减少——模型漏掉时它兜住，模型识别到时它不干扰。

## 终止与恢复

- **终止**：每个节点最终都汇入 `finalize → END`，一轮必然结束。
- **收敛**：达到硬上限时，路由会强制进入 `knowledge → decide`，此时 `decide` 通常输出
  `defer`——一个明确的"我还不能给建议，缺的是这些"，而不是硬编一个答案。
- **恢复**：checkpointer 按 `thread_id` 保存状态；若线程丢失，`ChatService._build_input()`
  会从 `sessions.analysis` 与 `messages` 表重建状态。两条路径都有测试覆盖。

## 与"无限自由循环 Agent"的区别

- 没有 `while not done: agent.think()`。每一轮的控制权在用户手上。
- 工具调用被限制在 `knowledge` 节点内部，且有 `agent_max_tool_rounds` 上限。
- 每一个模型调用都有明确的结构化契约，而不是自由文本交给下一环去猜。
- 所有"要不要继续"的判断都在 `route()` 里，用可读的规则表达。
