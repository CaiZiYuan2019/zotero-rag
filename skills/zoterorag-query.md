# ZoteroRAG 本地文献查询 Skill

## 描述

使用项目内 `.conda` 环境中的 `zoterorag` CLI，对当前已建库的 ZoteroRAG 向量库进行多维度文献检索，并将结果整理为结构化报告。

## 适用场景

- 需要基于本地已建库文献回答具体科研问题；
- 需要查找某个主题、功能、结构或方法相关的原文证据；
- 需要综合向量检索、全文检索和元数据检索来交叉验证；
- 最终输出需要形成可存档的 Markdown 报告。

## 前置条件

1. 当前工作目录为 ZoteroRAG 项目根目录（如 `E:/ZoteroRAG`）。
2. 项目内已存在 `.conda` 独立环境，且 `zoterorag` 可执行文件位于 `.conda/Scripts/zoterorag.exe`。
3. 向量库已完成建库（`zoterorag status` 中 `vector_indexes` 状态为 ready）。
4. 默认文本向量 profile 为 `qwen3vl_cloud_2560_text`（查询前可通过 `zoterorag models list` 确认）。

## 标准查询流程

### 步骤 1：确认环境与库状态

```bash
.conda/Scripts/zoterorag.exe status
.conda/Scripts/zoterorag.exe models list
.conda/Scripts/zoterorag.exe vectors list
```

确认向量库已激活、chunk 数量正常、默认文本模型可用。

### 步骤 2：元数据初筛

用 `search-metadata` 快速定位标题/摘要/作者中包含关键词的文献，无需调用 embedding API。

```bash
.conda/Scripts/zoterorag.exe search-metadata "关键词"
```

### 步骤 3：全文检索

用 `search-fulltext` 在已标准化的 chunk 文本中进行 LIKE 搜索，适合查找精确术语出现位置。

```bash
.conda/Scripts/zoterorag.exe search-fulltext "关键词" --chunk-type text
.conda/Scripts/zoterorag.exe search-fulltext "关键词" --chunk-type image
```

### 步骤 4：向量语义检索

调用 `search-vector` 进行语义检索，使用文本模型和 Qwen embedding provider。

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

当查询涉及论文中的图像时，可使用 `search-vector` 的多模态模式直接对图像内容进行语义检索。

```bash
.conda/Scripts/zoterorag.exe search-vector "自然语言图像描述" \
  --mode multimodal \
  --top-k 15 \
  --embedding-provider qwen3vl \
  --consumer llm_multimodal \
  --image-return file_ref \
  --max-images 10
```

返回结果中的 `images[].file_ref` 为图像在归一化目录下的相对路径（如 `images/img001.jpg`），完整路径为 `data/normalized/<document_id>/images/<file_ref>`。

为确认图像对应的元信息，建议同时运行文本模式向量检索或 `search-fulltext` 对 caption、图注及方法部分进行交叉验证。注意查看图像进行确认。

### 步骤 6：定位关键文档并精读原文

通过检索结果中的 `document_id`，使用 `documents show` 查看文档信息，并直接读取 `data/normalized/<document_id>/document.md`。

```bash
.conda/Scripts/zoterorag.exe documents show <document_id>
```

```bash
# 读取原文（前 400 行）
Read data/normalized/<document_id>/document.md
```

可使用 `Grep` 在文档内定位关键词具体位置，再分段精读。

### 步骤 7：交叉验证

对核心结论应至少通过两种检索方式验证：

- 向量检索发现相关段落或图像；
- 全文检索确认术语出现频率和上下文；
- 必要时通过 `documents show` 读取原文核对；
- 对多模态检索命中的图像，必要时使用 `ReadMediaFile` 读取原始图像进行视觉核验。

### 步骤 8：整理并输出报告

将检索结果整理为 Markdown 报告，建议包含：

1. 查询背景与数据来源；
2. 检索策略与关键文献列表（含 document_id）；
3. 分问题回答，引用具体文献；
4. 表格汇总核心发现；
5. 若某问题在库内无直接证据，应明确说明并列出参考替代文献。

输出路径建议：`E:/ZoteroRAG/report/<YYYYMMDD>/报告标题.md`。

## 常用命令速查

| 用途 | 命令 |
|------|------|
| 检查全局状态 | `.conda/Scripts/zoterorag.exe status` |
| 列出可用模型 | `.conda/Scripts/zoterorag.exe models list` |
| 元数据检索 | `.conda/Scripts/zoterorag.exe search-metadata "关键词"` |
| 全文检索（文本 chunk） | `.conda/Scripts/zoterorag.exe search-fulltext "关键词" --chunk-type text` |
| 向量检索（文本） | `.conda/Scripts/zoterorag.exe search-vector "问题" --mode text --top-k 20 --embedding-provider qwen3vl --consumer llm_text --image-return none` |
| 向量检索（多模态图像） | `.conda/Scripts/zoterorag.exe search-vector "图像描述" --mode multimodal --top-k 15 --embedding-provider qwen3vl --consumer llm_multimodal --image-return file_ref` |
| 查看文档信息 | `.conda/Scripts/zoterorag.exe documents show <document_id>` |
| 文档内关键词定位 | `Grep --path data/normalized/<document_id>/document.md --pattern "关键词" --output_mode content -n` |

## 输出处理建议

- 将检索结果临时保存到 `.tmp/` 目录，避免重复调用 API；
- 对关键文档使用 `Read` 分段读取，优先读取摘要、引言、结果和讨论部分；
- 对多模态检索命中的图像，使用 `ReadMediaFile` 读取原始图像进行视觉核验；
- 如需在报告中引用图像，建议复制到 `report/<YYYYMMDD>/images/` 目录并在 Markdown 中使用相对路径链接，避免直接修改 `data/normalized/` 原始库；
- 使用表格对多个文献的发现进行对比；
- 所有结论必须标注文献来源（文档 ID 或原文引用），避免无依据概括。

## 注意事项

- **环境路径**：Windows 下 Bash 工具使用 Git Bash，命令中应使用 Unix 路径分隔符 `/` 和 `.conda/Scripts/zoterorag.exe`。
- **API 调用**：`search-vector` 在 `--embedding-provider qwen3vl` 时会调用 DashScope embedding API，注意避免在短时间内发起大量请求。
- **向量模型**：单次查询只使用一个 embedding profile，不要跨模型合并结果。
- **无结果情况**：若检索结果不相关，应尝试同义词、更具体的短语、中英文、反向提问（如从功能提问改为从结构提问）。
