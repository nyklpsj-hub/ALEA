# 验收清单

五条验收标准，每条都给出**手工步骤**和**自动化对应**。
自动化部分在 `backend/tests/test_acceptance.py` 与 `test_agent_flow.py` 中。

前置：数据库已启动、迁移已执行、知识库已索引、`.env` 已填模型密钥。

```bash
cd backend
.venv/Scripts/python -m scripts.local_postgres start   # 或 docker compose up -d db
.venv/Scripts/python -m alembic upgrade head
.venv/Scripts/python -m scripts.ingest_knowledge
.venv/Scripts/python -m uvicorn app.main:app --reload
# 另一个终端
cd frontend && npm run dev
```

---

## ✅ 1. 同一个困境，Agent 动态追问而不是固定问答

**手工步骤**

1. 新建会话，输入：「我在国企做行政六年了，想辞职做摄影，但很犹豫。」
2. 回答它的追问，例如「我怕的是回不去原来的单位，不是怕没钱。」
3. 再新建一个会话，输入一个结构不同但同类的困境：「我攒了 40 万，是提前还房贷还是读个学位？」
4. 对比两次的追问。

**期望**

- 第二轮追问会**引用你上一句的原话**（"你提到回不去原来的单位……"），
  而不是走一遍固定问卷。
- 两个会话的追问顺序不同：第一个围绕"回退路径"，第二个围绕"钱还是时间"。
- 右上角「阶段」会随进程变化：追问中 → 确认判断 → 检索理论 → 给出建议。

**自动化**

```bash
.venv/Scripts/python -m pytest tests/test_agent_flow.py -k "first_turn_explores" -v
```

---

## ✅ 2. 用户纠正贪婪或恐惧后，Agent 真正修正状态

**手工步骤**

1. 继续上一段对话，直到右侧「需要你确认」出现一条推测。
2. 点「不完全是，我来改」，把内容改成你自己的说法并保存。
   （或者直接在对话框里写「不是这样，我其实是怕……」）
3. 观察右侧「贪婪 / 恐惧」列表与「长期记忆」页面。

**期望**

- 旧版本从列表中消失，不会在下一轮复活。
- 被改写的条目显示为「已纠正」或「用户亲述」。
- 后续建议里出现的理由，是你确认过的说法，而不是被纠正掉的旧版本。
- 「长期记忆」页面里那条记忆的 `provenance` 是 `user_stated` 或 `user_confirmed`，
  点「变更记录」能看到 `actor=user` 的那条修改。

**自动化**

```bash
.venv/Scripts/python -m pytest tests/test_agent_flow.py -k "correction or not_lost or correcting" -v
.venv/Scripts/python -m pytest tests/test_agent_logic.py -k "reconcile or confirmed_motive" -v
```

---

## ✅ 3. RAG 确实检索到了知识库内容，并提供真实来源

**手工步骤**

1. 打开「长期记忆 → 知识库」标签，确认文档列表非空（每份都有切片数）。
2. 在「检索验证」里输入 `代价 不可逆 底线`，回车。
3. 回到对话，聊到信息充分，展开右侧「引用」标签。

**期望**

- 「检索验证」返回带**相似度分数**的片段，分数都 ≥ `RAG_MIN_SCORE`。
- 每张引用卡片显示：文档标题、章节路径、来源类型（原文摘录 / 整理笔记 / 案例）、
  点击可展开原文摘录与文件路径。
- 如果检索不到相关内容，界面显示"知识库没有检索到相关条目"，
  Agent 的结论里也不会出现任何引用——**不会编造**。
- `backend/knowledge/01-core/*.md` 等都是 `curated_note`，界面上会明确写「整理笔记」，
  不会写成"原文"。

**自动化**

```bash
.venv/Scripts/python -m pytest tests/test_api.py -k "knowledge or citations" -v
.venv/Scripts/python -m pytest tests/test_agent_flow.py -k "grounded or citations_are_attached" -v
.venv/Scripts/python -m pytest tests/test_agent_logic.py -k "citations" -v   # 编造引用会被剔除
```

---

## ✅ 4. 最终明确给出一个选祸建议，说明理由、代价与风险边界

**手工步骤**

1. 用一个信息量足够的困境，走过追问与确认阶段。
2. 观察右侧顶部出现的建议卡。
3. 反过来，再用一个信息很少的困境（只说一句就要求结论）。

**期望**

- 信息充分时：明确写出建议承担哪个「祸」、为什么是它而不是别的、
  **必须放弃什么**、风险的等级与应对、**底线**、以及出现什么情况要重新评估。
- 信息不足时：明确输出「暂时不给你结论」，并列出"我还需要知道"的清单——
  不硬凑答案。
- 涉及自伤/暴力/违法/严重健康风险时：不给任何建议，转为求助引导
  （给出 12356、120/110 等真实可用的方向）。

**自动化**

```bash
.venv/Scripts/python -m pytest tests/test_agent_flow.py -k "recommendation or defer or crisis or sufficiency" -v
.venv/Scripts/python -m pytest tests/test_agent_logic.py -k "decision or render" -v
```

---

## ✅ 5. 关闭当前会话后开启新会话，能够正确检索、使用并修改用户确认过的长期记忆

**手工步骤**

1. 在会话 A 中确认一条重要信息（例如"你的硬约束是每月房贷 8000 元不能断"）。
2. 关掉标签页/浏览器，重新打开 <http://localhost:3000>。
3. 新建会话 B，描述一个**相关**的新困境（例如"我又在纠结要不要换工作"）。
4. 观察会话 B 右侧「记忆」标签。
5. 打开「长期记忆」页面，把那条记忆改掉或删掉。
6. 回到会话 B 再发一句话（或新建会话 C），确认检索按新内容进行。

**期望**

- 会话 B 的「记忆」标签里出现会话 A 确认过的条目，标注为「已确认」。
- 会话 B 的追问会**用上**这条记忆（例如先问"房贷这条约束现在有变化吗"）。
- 改动/删除后，新会话检索到的是新内容；已删除的记忆不再被检索。
- 「变更记录」完整保留：`created`（agent）→ `confirmed`（user）→ `edited`（user）→ …

**自动化**

```bash
.venv/Scripts/python -m pytest tests/test_agent_flow.py -k "cross or recalled or survives_a_restart" -v
.venv/Scripts/python -m pytest tests/test_memory.py -v
```

---

## 一次跑完全部验收

```bash
cd backend
ALEA_REQUIRE_DB=1 .venv/Scripts/python -m pytest -v
```

`ALEA_REQUIRE_DB=1` 会让数据库不可达时**报错而不是跳过**——验收时应当这样跑，
避免"因为环境没起来所以通过"。

### 真实模型链路

自动化测试里的模型是脚本化的测试替身，覆盖的是控制流与护栏；**真实模型链路**
需要配好密钥后单独验：

```bash
# backend/.env 里填好 LLM_API_KEY / EMBEDDING_API_KEY，然后：
cd backend
.venv/Scripts/python -m scripts.smoke_llm           # 命令行跑完一段四轮困境
ALEA_REQUIRE_DB=1 .venv/Scripts/python -m pytest -m llm -v
.venv/Scripts/python -m scripts.ingest_knowledge    # 用真实向量模型重建索引
```

`smoke_llm` 会逐轮打印：Agent 判断的贪婪/恐惧/约束、检索到的理论来源
（含文件名、章节、相似度）、写入的长期记忆，以及最终结论的状态。
它跑通即说明真实模型下主链路可用。

### 已经验证过的部分（无需密钥）

| 项目 | 命令 | 结果 |
| --- | --- | --- |
| 全部后端测试 | `pytest -q` | 104 passed, 4 skipped（跳过的是 `-m llm`） |
| 静态检查 | `ruff check app tests scripts` | All checks passed |
| 类型检查 | `mypy app` | Success: no issues found in 51 source files |
| 前端类型 + 构建 | `npm run typecheck && npm run build` | 构建成功 |
| 服务真实启动 | `python -m scripts.serve` | `/api/health` 返回 200，checkpointer 就绪 |
| 缺密钥时的行为 | 未配置 `LLM_API_KEY` 时发起对话 | 503 `configuration_error`，含可执行的修复提示 |

---

## 手工验收记录表

| 项 | 结果 | 备注 |
| --- | --- | --- |
| 1 动态追问 | ☐ | |
| 2 纠正生效 | ☐ | |
| 3 RAG 有真实来源 | ☐ | |
| 4 明确选祸建议 + 代价 + 底线 | ☐ | |
| 5 跨会话记忆检索/使用/修改 | ☐ | |
