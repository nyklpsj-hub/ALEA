# 知识库说明

这个目录是「人选祸」Agent 唯一允许引用的理论来源。

## 重要：来源可信度分级

系统在向用户展示引用时，会**原样**带上每条资料的来源类型，用户可以看到：

| `source_type` | 含义 | Agent 允许的说法 |
| --- | --- | --- |
| `verified_excerpt` | 人工核对过的原文摘录 | "原文提到……" |
| `curated_note` | **本项目自己整理**的理论笔记 | "资料里把这一点整理为……" |
| `case` | 案例（来自原文或本项目的示例） | "有一个类似的案例……" |

当前仓库中 `01-core/` 与 `02-method/` 下的文件全部是 `curated_note`，
即**本项目对"人选祸，天选福"思想的整理与工程化表述，不是原书原文**。
它们标注了 `verified: false`。这样做是为了让 Agent 永远不会把整理内容说成原文。

## 加入已核对的原文

如果你手上有《人选天选论》原文，请：

1. 复制 `_templates/verified-excerpt-template.md`；
2. 把它放到**不以 `_` 开头的目录**里，并给一个正常的文件名
   （文件名或路径中任何一层目录以 `_` 开头都不会被索引）；
3. 填写 `source_type: verified_excerpt` 与 `verified: true`；
4. 只粘贴**逐字核对过**的段落，并在 `locator` 里写明出处位置；
5. 运行 `python -m scripts.ingest_knowledge`。

## 文件格式

```markdown
---
title: 文档标题
source: 人选天选论（路飞）
source_type: curated_note
verified: false
tags: [核心概念]
---

# 一级标题

正文……（按 Markdown 标题切分，标题路径会作为引用位置一并保存）
```

重新索引是幂等的：内容没变的文件会被跳过，改动过的会重新切分并向量化。
