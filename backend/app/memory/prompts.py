"""Prompts for long-term memory extraction and consolidation."""

from __future__ import annotations

EXTRACTION_SYSTEM = """\
你是一个长期记忆筛选器。你的任务是从一段心理咨询式对话中，挑出**值得跨会话长期保存**的用户信息。

保存标准（必须同时满足）：
1. 与用户的决策方式、动机或处境直接相关；
2. 在几个月后依然成立，而不是此刻的情绪或一次性细节；
3. 有用户原话作为依据。

类别定义：
- goal：用户明确的重要目标
- value：用户稳定的价值观、优先级排序
- greed：用户想得到的东西（渴望）
- fear：用户害怕失去的东西（恐惧）
- constraint：现实约束（时间、金钱、家庭、健康、法律、技能）
- decision：用户做过的选择及其结果
- preference：用户对沟通或决策方式的偏好
- fact：稳定的客观事实（职业、城市、家庭结构等）

严格遵守：
- **禁止写入任何用户没有明确表达的内容**。如果信息是你推断出来的，provenance 必须是
  agent_inferred，并且 content 要写成可被证伪的陈述。
- 如果某个判断原本是你的推测，而用户在这一轮明确认可了它（例如回答"对""是的""就是这样"），
  provenance 用 user_confirmed，evidence 引用用户认可时的那句话。
- evidence 必须是你从对话中**逐字引用**的用户原话（可截断，不得改写）。
- 不保存第三方隐私（同事、家人姓名等）。
- 不保存用户的临时情绪（"我现在很烦"）。
- 一次提取不超过 5 条。如果没有任何值得保存的信息，返回空列表。
- 敏感或推断性内容宁可少提取：不确定就不要提取。
"""

EXTRACTION_USER = """\
当前困境摘要：{dilemma}

本轮 Agent 记录到的判断（status=confirmed 表示用户已确认过，rejected 表示已被否认）：
{analysis_summary}

最近的对话：
{transcript}

请提取值得长期保存的用户信息。"""


CONFLICT_SYSTEM = """\
你在维护一份关于用户的长期记忆库。现在有一条**新信息**和若干条**已有记忆**，
它们的语义相近，可能重复、可能是更新、也可能互相矛盾。

对每一条已有记忆独立判断，只能选择以下一种：
- "duplicate"：表达的是同一件事，新信息没有提供额外内容；
- "update"：同一件事，但新信息更新、更准确，应当用它替换旧记忆；
- "contradict"：两者互相矛盾（例如"想留在本地"与"决定搬去外地"），新信息代表用户当前状态；
- "distinct"：是相关的两件不同的事，都应当保留。

判断要保守：只要不确定是否矛盾，就选择 "distinct"。
"""

CONFLICT_USER = """\
新信息（{kind}）：{content}
新信息依据：{evidence}

已有记忆：
{existing}

请给出每一条已有记忆的判断。"""


SUMMARIZE_SESSION_SYSTEM = """\
把一次已经结束的决策对话，压缩成一条“历史决策事件”记录，用于将来相似困境的参考。

要求：
- title：不超过 20 字，说明这是什么选择；
- situation：两三句话说明当时的处境；
- greed / fear：用户当时的渴望与恐惧，各不超过 3 条，用短语；
- cost：这个选择必须承担的代价，不超过 3 条；
- risks：主要风险，不超过 3 条；
- summary：一句话总结用户最终如何权衡。
只依据对话内容，不要补充对话中没有的事实。
"""

SUMMARIZE_SESSION_USER = """\
困境：{dilemma}
选项：{options}
最终建议：{decision}

对话记录：
{transcript}
"""
