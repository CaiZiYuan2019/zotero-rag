# ZoteroRAG 本地文献查询 Skill

## 描述

使用项目内 `.conda` 环境中的 `zoterorag` CLI，对当前已建库的 ZoteroRAG 文献库进行多维度检索（元数据 / 集合分区 / 全文 / 向量），回答问题或按需生成可存档的 Markdown 报告。

## 适用场景

- 需要基于本地已建库文献回答具体科研问题；
- 需要按 Zotero 文件夹（集合）缩小检索范围；
- 需要查找某个主题、功能、结构或方法相关的解析文本或图像证据；
- 需要综合向量检索、全文检索和元数据检索来交叉验证；
- 最终输出需要形成可存档的 Markdown 报告。

## 前置条件

1. 当前工作目录为 ZoteroRAG 项目根目录（如 `E:/ZoteroRAG`）。
2. 项目内已存在 `.conda` 独立环境，且 `zoterorag` 可执行文件位于 `.conda/Scripts/zoterorag.exe`（也可用 `python -m zoterorag.cli`）。
3. 向量库已完成建库（`zoterorag status` 中 `vector_indexes` 状态为 ready）——仅向量检索需要；元数据/全文/集合检索不需要 embedding。
4. 默认文本向量 profile 为 `qwen3vl_cloud_2560_text`（查询前可通过 `zoterorag models list` 确认）。
5. 作者、集合、DOI 等过滤依赖已复制到台账的书目字段。字段缺失时记录缺口。

## 选择查询方式

- **简单查询（默认）**：查某篇文献、列出集合命中或定位一段解析文本时，主代理直接选用下列命令，给出结果与出处。需要报告时再执行步骤 8。
- **精确查询**：用户要求精查、跨文献综合、独立复核或逐次查询可追溯时，主代理界定范围并发起检索，再把命中文档交给精读者。精读完成后，另派专职复核者独立核查重要结论，最后由主代理评估两方证据、完成报告。分派前阅读[精确查询流程与子代理提示词](zoterorag-query/precise-query.md)。

主代理和各子代理只用项目台账、已有书目元数据副本、解析文档及其图像。`document.md` 是解析文本。源 Zotero 库和原始 PDF 仅供用户本人手工阅读、核对或写作。查询跳过既有 `report/`。

## 标准查询流程

### 步骤 0：查看已入库的集合（按需）

列出集合 key（后续 `--collection` / `search-by-collection` 使用）：

```bash
.conda/Scripts/zoterorag.exe list-collections
```

示例（本库）：`超微结构` → `26TP7I8G`，`杂` → `SAURPW6J`。

### 步骤 1：确认环境与库状态

```bash
.conda/Scripts/zoterorag.exe status
.conda/Scripts/zoterorag.exe models list
.conda/Scripts/zoterorag.exe vectors list
```

确认向量库已激活、chunk 数量正常、默认文本模型可用。

### 步骤 2：元数据初筛（可加集合 / 作者 / 年份 / BM25）

无需 embedding。默认 AND + 自定义加权分；可选 BM25（`score` 越大越相关）。

```bash
.conda/Scripts/zoterorag.exe search-metadata "关键词"

# 作者 + 年份 + 文件夹
.conda/Scripts/zoterorag.exe search-metadata "Vater" \
  --author Vater \
  --date-from 1997 --date-to 1997 \
  --collection 26TP7I8G \
  --limit 20

# BM25
.conda/Scripts/zoterorag.exe search-metadata "glutaraldehyde" \
  --collection 26TP7I8G \
  --sort-by bm25

# 短语 / OR；空查询 + 过滤器 = 纯过滤列表
.conda/Scripts/zoterorag.exe search-metadata "\"organ of Corti\"" --collection 26TP7I8G
.conda/Scripts/zoterorag.exe search-metadata "" --author Nakajima --date-from 1971 --date-to 1971
```

查询语法：`"短语"`、`term*` 前缀、`+must` / `-exclude`、`--match-mode any|all`。大小写不敏感。

### 步骤 2b：集合（文件夹）分区检索

```bash
.conda/Scripts/zoterorag.exe search-by-collection 26TP7I8G --query "Vater 1997" --limit 20
.conda/Scripts/zoterorag.exe search-by-collection 26TP7I8G --query "glutaraldehyde" --sort-by bm25
.conda/Scripts/zoterorag.exe search-by-collection 26TP7I8G --include-children --query "bat"
```

默认**不**包含子集合。负例校验：在「超微结构」搜只存在于「杂」的词（如 deepseek）应为空。

### 步骤 3：全文检索（方法学术语优先）

精确术语、固定液名、中文方法词等常不在标题/摘要，应用全文：

```bash
.conda/Scripts/zoterorag.exe search-fulltext "关键词" --chunk-type text
.conda/Scripts/zoterorag.exe search-fulltext "glutaraldehyde" --chunk-type text --sort-by bm25 --limit 10
.conda/Scripts/zoterorag.exe search-fulltext "戊二醛" --chunk-type text --sort-by score
.conda/Scripts/zoterorag.exe search-fulltext "关键词" --chunk-type image
```

FTS 无命中时会 LIKE 回退；该路径无 BM25，会降级为自定义分。极短词（如 `GA`）噪声大，优先用完整词。

### 步骤 4：向量语义检索

```bash
.conda/Scripts/zoterorag.exe search-vector "自然语言问题" \
  --mode text \
  --top-k 20 \
  --embedding-provider qwen3vl \
  --consumer llm_text \
  --image-return none
```

建议从多个角度提问，以覆盖功能、机制、结构、方法等维度。

### 步骤 5：多模态图像检索（可选）

```bash
.conda/Scripts/zoterorag.exe search-vector "自然语言图像描述" \
  --mode multimodal \
  --top-k 15 \
  --embedding-provider qwen3vl \
  --consumer llm_multimodal \
  --image-return file_ref \
  --max-images 10
```

从 `documents show <document_id> --consumer manual` 取 `document.artifact.artifact_dir`。`images[].file_ref` 为相对路径时按此目录解析。建议与文本向量或全文交叉验证，并查看图像确认。

### 步骤 6：定位关键文档并精读解析文本

对元数据、集合、全文和向量命中，直接用返回的 `document_id` 调用 `documents show`。`attachment_key` 标识具体附件，`parent_key`（同 `parent_item_key`）标识父条目。同一父条目可有多个附件，各自保留其文档 ID。仅当 `document_kind` 为 `normalized` 时读取对应解析文件。若为 `metadata_only`，可检查台账中同篇其他附件的解析状态。

```bash
.conda/Scripts/zoterorag.exe documents show <document_id>
```

从 `documents show <document_id> --consumer manual` 取得 `document.artifact.document_md`，在该解析文件中定位关键词，再读完整段落和相邻图注。引用时注明解析文件名、章节或行号。

### 步骤 7：交叉验证

简单查询可按问题选取以下检索面交叉比对；精确查询还需把重要结论交给独立复核者。

- 元数据 / 集合分区：确认文献归属、作者、年份；
- 全文：确认术语出现频率和上下文（方法词尤其重要）；
- 向量：发现语义相关段落或图像；
- 必要时 `documents show` / 读 `document.md`；多模态命中可用 `ReadMediaFile` 核验图像。

**不要依赖自动混合检索**——各 CLI/API 面独立，由本 skill 交叉比对。

### 步骤 8：整理并输出报告

建议包含：查询背景与数据来源；检索策略与关键文献（含 document_id / attachment_key）；分问题回答；表格汇总；无证据时明确说明。

输出路径建议：`E:/ZoteroRAG/report/<YYYYMMDD>/报告标题.md`。

## 常用命令速查

| 用途 | 命令 |
|------|------|
| 列出集合 | `.conda/Scripts/zoterorag.exe list-collections` |
| 检查全局状态 | `.conda/Scripts/zoterorag.exe status` |
| 元数据检索 | `.conda/Scripts/zoterorag.exe search-metadata "关键词"` |
| 元数据 + 过滤 + BM25 | `.conda/Scripts/zoterorag.exe search-metadata "词" --author X --collection KEY --date-from YYYY --date-to YYYY --sort-by bm25` |
| 集合内检索 | `.conda/Scripts/zoterorag.exe search-by-collection KEY --query "词" --sort-by bm25` |
| 全文检索 | `.conda/Scripts/zoterorag.exe search-fulltext "词" --chunk-type text --sort-by bm25` |
| 向量检索（文本） | `.conda/Scripts/zoterorag.exe search-vector "问题" --mode text --top-k 20 --embedding-provider qwen3vl --consumer llm_text --image-return none` |
| 向量检索（多模态） | `.conda/Scripts/zoterorag.exe search-vector "图像描述" --mode multimodal --top-k 15 --embedding-provider qwen3vl --consumer llm_multimodal --image-return file_ref` |
| 查看文档信息 | `.conda/Scripts/zoterorag.exe documents show <document_id>` |
| 文档内关键词定位 | `Grep --path <返回的 document_md 路径> --pattern "关键词" --output_mode content -n` |

## 检索手段选择建议

| 目标 | 优先手段 |
|------|----------|
| 作者 / 年份 / 题名 / DOI / 文件夹 | `search-metadata` 或 `search-by-collection` |
| 固定液、试剂、方法细节（如 glutaraldehyde、戊二醛） | `search-fulltext`（必要时再向量） |
| 语义问题、机制描述 | `search-vector --mode text` |
| 图示 / 超微结构图像 | `search-vector --mode multimodal` |
| 排序对比 | 同一查询分别跑 `--sort-by score` 与 `--sort-by bm25` |

实库抽检（超微结构 `26TP7I8G`）：

- `glutaraldehyde` + `sort-by bm25`：元数据/集合内可命中（如 Nakajima 1971）；全文 BM25 正常且分数 > 0。
- `戊二醛`：元数据常为 0；全文可命中（可能走 LIKE，BM25 分数形态接近加权分）。
- `Glutaric dialdehyde`：文献多用 `glutaraldehyde`，前者可能 0 命中——应改用常用英文拼写。
- `GA`：高度歧义（GABA 等）；`score` 可用，`bm25` 可能因 FTS→LIKE 回退而为 0——改用完整词。

## 简单查询的输出处理建议

- 将检索结果临时保存到 `.tmp/`，避免重复调用 embedding API；
- 关键文档用 `Read` 分段读摘要/方法/结果/讨论；
- 多模态命中用 `ReadMediaFile` 核验；引用图时复制到 `report/<YYYYMMDD>/images/`；
- 结论标注 `document_id` 及所用解析段落、图像或书目字段。

## 注意事项

- **环境路径**：Windows 下 Bash 用 Unix 分隔符与 `.conda/Scripts/zoterorag.exe`。
- **API 调用**：`search-vector` + `qwen3vl` 会打 DashScope，避免短时间大量请求。
- **单模型**：单次向量查询只用一个 embedding profile。
- **无结果**：换同义词、完整拼写、中英文、短语，或从元数据改走全文/向量；集合负例用于确认分区过滤正确。
- **多附件**：同一文献可能有多份 PDF（双语/副本），结果中多条 attachment 属正常。
