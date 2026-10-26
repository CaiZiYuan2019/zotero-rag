# 检索与元数据问题：document_id 语义冲突、书目字段漏收、转换伪表格

> 记录时间：2026-09-18
> 状态：待修复
> 发现来源：2026-09-18 一次多文献检索任务（按 `skills/zoterorag-query.md` 执行，集合「超微结构」`26TP7I8G`，蝙蝠听觉核团超微结构综述，最终产出 5 篇核心文献的汇总报告）
> 与既有文档的关系：本文记录**检索侧 / 元数据侧 / 转换侧**问题；建库删除失效与僵死 job 见 `docs/DATA_HYGIENE_ISSUES_20260909.md`

## 摘要

| # | 问题 | 影响面 | 严重度 |
|---|------|--------|--------|
| 1 | **`document_id` / `chunk_id` 在不同检索面含义不同**：`search-metadata`/`search-by-collection` 的 `document_id` 是**父条目 key**、`chunk_id` 是**附件 key**；其余面 `document_id` 才是附件 key | 全部检索工作流 | **高** |
| 2 | `documents show <父条目key>` **静默返回 `{"document": null}`**，不报错 | 按 skill 步骤 6 操作必然踩中 | 中 |
| 3 | **书目字段清单漏收 `volume` / `issue` / `pages`**，state 库中三字段无任何落点 | 全库；任何引用格式输出 | **高** |
| 4 | `search-by-collection --limit N` **静默截断**，输出无任何截断标志 | 大集合枚举 | 中 |
| 5 | MinerU 把直方图**"补全"为原文不存在的数值表格**（`type: chart`） | 语料可信度 | 中～高 |
| 6 | normalized `document.md` **剥离页眉/页码**，且**图注被误分类为 `page_footnote`** | 书目核验、图版阅读 | 中 |
| 7 | 仓库根目录存在 **0 字节 `state.sqlite`**（真实库在 `data/state/state.sqlite`） | 排查误导 | 低 |

**因果**（详见第 8 节复盘）：问题 **3**（拿不到卷页）叠加问题 **6a**（normalized 产物也查不到页眉，无法自查），
导致本次任务**在最终报告中写入了 3 处错误的卷页信息**——这是本文档中唯一造成实际损失的问题。
问题 **1**、**2** 未造成错误结论，但显著增加了排查成本：曾据 `documents show` 返回 `null` 误判
"这 5 篇没有 normalized 全文"，实际产物一直都在。

---

## 问题 1：`document_id` 语义冲突（父条目 vs 附件）

### 现象

CLI 各检索面返回的 `document_id` 字段**指向两种不同的实体**：

| 检索面 | `document_id` 实际含义 | Nakajima 1971 的取值 |
|---|---|---|
| `search-metadata` | Zotero **父条目** key | `NTXYZEAV` |
| `search-by-collection` | Zotero **父条目** key | `NTXYZEAV` |
| `search-fulltext` | **附件** key | `B6CHNRDC` |
| `search-vector` / `search-mm` | **附件** key | `B6CHNRDC` |
| `documents list` / `documents show` | **附件** key | `B6CHNRDC` |
| `data/normalized/<dir>/` | **附件** key | `B6CHNRDC` |
| `state.sqlite: chunks.document_id` | **附件** key | `B6CHNRDC` |
| `state.sqlite: normalized_artifacts.document_id` | **附件** key | `B6CHNRDC` |

即：**`search-metadata` 与 `search-by-collection` 是仅有的两个"异类"面。**

**并且 `chunk_id` 也冲突**：在 `search-fulltext` / `search-vector` 中 `chunk_id` 是真正的分块标识
（如 `B6CHNRDC:image:00002`），但在元数据检索结果中 `chunk_id` 被用来装**附件 key**（`B6CHNRDC`）。
即**两个字段名 × 两种语义**，且元数据检索的结果 schema 看起来与全文检索**完全同构**
（同样有 `document_id` / `chunk_id` / `title` / `text` / `score` / `metadata` / `images`），
不细看无法察觉取值含义不同。

### 证据（2026-09-18）

```bash
$ zoterorag search-metadata "Nakajima" --limit 3
  top-level keys: ['results']
  document_id='NTXYZEAV'   (父条目 key)   ← 与下一行同名不同义
  chunk_id='B6CHNRDC'      (附件 key)
  metadata.attachment_key='B6CHNRDC'   ← 正确的 key 在这里
  metadata.parent_key='NTXYZEAV'
  ---
  document_id='NTXYZEAV'
  chunk_id='HDQDKJLJ'      (该父条目下的另一个附件)
  metadata.attachment_key='HDQDKJLJ'
  ---

$ zoterorag search-fulltext "calyciferous" --chunk-type text --limit 3
  top-level keys: ['results']
  document_id='B6CHNRDC'   (附件 key)
  chunk_id='B6CHNRDC:text:00004'   ← 真实分块 id
  metadata.attachment_key=None     ← 与元数据检索不同，这里不再重复给
  ---

# 向量检索（chunks 表中的实测格式）
  document_id='B6CHNRDC',  chunk_id='B6CHNRDC:image:00002'
```

注意 `search-metadata --limit 3` 返回的**两行仍是同一个父条目 `NTXYZEAV` 下的两个附件**
（`B6CHNRDC` / `HDQDKJLJ`）——印证 `document_id` 是**条目级**、行是**附件级**。

对照代码 `src/zoterorag/search/metadata.py:59-60`，与上表完全吻合。

数据库侧对照：

```sql
sqlite> SELECT attachment_key, parent_key FROM attachments WHERE attachment_key='B6CHNRDC';
B6CHNRDC | NTXYZEAV
```

`parent_key` 正是 `search-metadata` 返回的那个 `document_id`，而 `B6CHNRDC` 是附件自身 key。

### 影响

1. **按 skill 文档操作会踩空**：`skills/zoterorag-query.md` 步骤 6 与"常用命令速查"表均写 `documents show <document_id>`。若 `<document_id>` 来自步骤 2 的 `search-metadata`（这是 skill 建议的**首选**元数据手段），则该命令必然返回 `null`（见问题 2）。
2. **无法直接定位 normalized 产物**：拿到元数据检索的 `document_id` 后，`data/normalized/<document_id>/` 不存在，须先反查 `attachment_key`。本次任务不得不自建 `document_id ↔ attachment_key` 映射表（`.tmp/norm_map.tsv`）才能继续。
3. **同名字段跨面传递时静默失效**：脚本里把 `search-metadata` 的 `document_id` 存下来，之后传给 `chunks`/向量/`documents` 任一面，都会得到空结果而非报错。
4. **正文里有可用的正确 key，但不在 `document_id` 上**——见「根因」，实际影响因此是"易错"而非"数据拿不到"。

### 根因

**已定位到具体一行**：`src/zoterorag/search/metadata.py:59`。

```python
results = [
    {
        "document_id": row.get("parent_key") or row["attachment_key"],   # ← 语义在这里被改写
        "chunk_id":    row["attachment_key"],                            # ← 与全文检索的 chunk_id 语义冲突
        "title": ...,
        "text": build_metadata_text(row),
        "score": row["score"],
        "metadata": {
            "attachment_key": row["attachment_key"],   # ← 正确的 key 在这里
            "parent_key":     row.get("parent_key"),   # ← 父条目 key 在这里
            ...
```

即：元数据检索**刻意复用**了与全文/向量检索相同的返回 schema（`document_id` / `chunk_id` / `title` /
`text` / `score` / `metadata` / `images`），但其中**两个字段被赋予了不同语义**：

| 字段 | 元数据检索（`metadata.py:59-60`） | 其余所有检索面 |
|---|---|---|
| `document_id` | `parent_key`（父条目 key） | `attachment_key`（附件 key） |
| `chunk_id` | `attachment_key`（附件 key） | 真实分块 id，如 `<key>:image:00002` |

**关键**：正确的附件 key 其实**已经返回了**——在 `metadata.attachment_key`（第 65 行）
以及 `chunk_id`（第 60 行）。所以这不是数据缺失，而是**同一个 key 被放在不同字段名下，
且 `document_id` 这个最显眼的名字承载了最不该承载的那个**。
（本次任务正是靠 `metadata.attachment_key` 才建出映射表的。）

### 建议方案

- **首选**：把 `document_id` 改为 `row["attachment_key"]`，与其余检索面一致；
  父条目 key 已有 `metadata.parent_key` 承载，不会丢信息。`chunk_id` 在元数据检索中
  建议置空或改名为 `attachment_key`，避免与真实 `chunk_id` 混淆。
- **次选**（若须保持向后兼容）：保留现状但在**返回体顶层**显式给出 `document_id_kind`
  （如 `"document_id_kind": "parent_item"`），并让 `documents show` **同时接受两种 key**。
- 无论哪种方案，`documents show` 在未命中时应返回**非零退出码 + 明确错误**，而非 `{"document": null}`（问题 2）。
- `skills/zoterorag-query.md` 步骤 6 应改为 `documents show <metadata.attachment_key>`（在问题修复前）。

---

## 问题 2：`documents show` 未命中时静默返回 null

### 现象

```bash
$ zoterorag documents show NTXYZEAV      # 父条目 key
{ "document": null }

$ zoterorag documents show B6CHNRDC      # 附件 key
{ "document": { "document_id": "B6CHNRDC", "document_kind": "normalized", ... } }
```

两种输入产出**结构相同、仅内容不同**的 JSON，退出码无差异，无 stderr 提示。

### 影响

调用方无法区分三种情况：(a) 该键不存在；(b) 键类型用错；(c) 文档确实没有 normalized 产物。本次任务中该现象被误判为"这 5 篇文献没有 normalized 全文"，直到改用附件 key 才确认产物一直都在。

### 建议方案

- 未命中时返回非零退出码与明确消息（如 `document not found for key 'NTXYZEAV' (looks like a Zotero parent item key; try its attachment key)`）。
- 若实现上可以：识别出传入的是父条目 key 时，直接列出其附件的文档记录。

---

## 问题 3：书目字段清单漏收 `volume` / `issue` / `pages`

### 现象

Zotero 侧**完整存在**这三个字段，但 `state.sqlite.attachments` 表**没有对应列**，
`metadata_json` 也取不到（为**空对象 `{}`**，且后者是刻意设计——见「根因」）。
结果是这三个字段在 state 库中**没有任何可读路径**。

### 证据（2026-09-18）

`data/shadow/zotero.sqlite`（Zotero 影子库）中 5 篇核心文献的字段：

| 文献 | volume | issue | pages |
|---|---|---|---|
| Nakajima 1971 | `50` | `1` | `121-134` |
| Vater 1995 | `358` | `2` | `155-180` |
| Vater, Covey & Casseday 1997 | `289` | `2` | `223-233` |
| Kemmer & Vater 1997 | `287` | `3` | `487-506` |
| Kemmer & Vater 2001 | `203` | `6` | `429-447` |

同 5 篇在 `data/state/state.sqlite` 中：

```
attachments 表列：attachment_key, parent_key, content_type, relative_path, title, abstract,
  date_value, url, classification, source_quality, reasons_json, file_path, file_exists,
  file_size, file_mtime, metadata_json, content_fingerprint, scan_status, first_seen_at,
  last_seen_at, updated_at, creators_json, collections_json, tags_json, item_type,
  publication_title, doi, date_added, date_modified, date_year, date_normalized
  → 无 volume / issue / pages

metadata_json 取值：{}（5 篇全部为空，且这是刻意设计，见「根因」）
```

**读影子库的查询里硬编码了字段清单**：`src/zoterorag/zotero/shadow.py:233-250` 逐个 `LEFT JOIN` 的
书目字段只有 **6 个**——`title` / `abstractNote` / `date` / `url` / `publicationTitle` / `DOI`，
**没有 `volume` / `issue` / `pages`**：

```python
LEFT JOIN itemData title ON ... AND title.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'title')
LEFT JOIN itemData abs   ON ... AND abs.fieldID   = (SELECT fieldID FROM fields WHERE fieldName = 'abstractNote')
LEFT JOIN itemData dt    ON ... AND dt.fieldID    = (SELECT fieldID FROM fields WHERE fieldName = 'date')
LEFT JOIN itemData url   ON ... AND url.fieldID   = (SELECT fieldID FROM fields WHERE fieldName = 'url')
LEFT JOIN itemData pub   ON ... AND pub.fieldID   = (SELECT fieldID FROM fields WHERE fieldName = 'publicationTitle')
LEFT JOIN itemData doi   ON ... AND doi.fieldID   = (SELECT fieldID FROM fields WHERE fieldName = 'DOI')
                                                                          # ← 清单到此为止
```

全仓库范围内，`volume` / `issue` / `pages` 作为 **Zotero 书目字段**从未被读取
（唯一命中的 `pages` 全部是 `selected_pages` / `max_pages_per_extraction` 等 **PDF 抽取页码**，与文献卷页无关）：

```bash
$ grep -rn "fieldName" src/zoterorag --include=*.py
（仅有 title / abstractNote / date / url / publicationTitle / DOI 六个书目字段）
```

### 根因

**不是同步丢弃，而是字段清单漏收。** 两件事叠加：

1. **`metadata_json` 保持 `{}` 是刻意设计，不是缺陷。** 检索增强把书目字段放进**专用列**，
   为的是让"仅书目字段变更"不触发 `attachment_fingerprint()` / `scan_status` 变化。
   `src/zoterorag/db/state.py:533-535` 的注释写得很明确：

   ```python
   # Bibliographic fields added by the retrieval enhancement. They live
   # in dedicated columns (never in metadata_json) so biblio-only
   # changes cannot flip attachment_fingerprint()/scan_status.
   ```

   该约束在 `docs/RETRIEVAL_ENHANCEMENT_FOLLOWUPS.md`「次要项 1」中被记为**已满足**。

2. **源头在更上游：读影子库的 SQL 就只 SELECT 了 6 个书目字段**
   （`shadow.py:233-250`，见上）。`volume` / `issue` / `pages` **从未被读出**，
   因此下游 `state.py:536-550` 的专用列清单里自然也不会有它们——
   那份清单是照着上游 SELECT 的字段建的。

所以链条是：**`shadow.py` 少 SELECT 三个字段 → `state.py` 少三列 → CLI 无字段可输出**。
既没有专用列、又被设计排除在 `metadata_json` 之外，这三个字段在 state 库中**根本没有落点**——
不是"同步时丢了"，而是**从来没有代码去读它**。

### 影响

1. **任何从本库产出的引用格式都不完整**。`search-metadata` / `documents list` / `documents show` 的输出都不含卷页。
2. 本次任务因此**在最终报告中凭印象补全卷页，写错了 3 处**（详见第 8 节）。
3. 下游若要做按卷期分组、去重、引用导出，均无数据支撑。
4. **本库其实"有"部分卷页信息，只是不可用**：DOI 里编码了卷期。见下表——
   但由于 DOI 编码规则**因出版商而异**、且通常**只含起始页**，不能作为通用替代。

   | 文献 | state 库中的 DOI | 可读出的信息 | 是否够用 |
   |---|---|---|---|
   | Nakajima 1971 | `10.1083/jcb.50.1.121` | 卷 50 / 期 1 / **起页** 121（Rockefeller 格式 `vol.issue.startpage`） | 起页对，**止页 134 缺失** |
   | Vater 1995 | `10.1002/cne.903580202` | 卷 358（Wiley CNE 格式，与 PDF 页眉 `358:155-180` 一致） | **页 155-180 全缺** |
   | Kemmer 1997 | `10.1007/s004410050869` | 未解出（Springer 编码，需外部规则） | 不可用 |

   > 上表"可读出的信息"是**按出版商 DOI 编码规则人工推断**的，非本库功能；
   > 编码规则因出版商而异且不稳定，**不能作为通用方案**，仅说明"库内并非全无线索"。

   值得注意的是：**Nakajima 的 DOI 已足以证伪我写错的 `121-142`**（DOI 明确是 `50.1.121`）。
   即错误本可被库内数据拦下，只是 CLI 没有任何路径把它暴露出来。

### 建议方案

需要**两处**改动，缺一不可：

1. **`src/zoterorag/zotero/shadow.py`** —— 在上面的 SELECT 里按同样的 `LEFT JOIN` 模式补三行
   （`volume` / `issue` / `pages`），并让 `_row_to_attachment` 把它们带出来。
2. **`src/zoterorag/db/state.py`** —— 按既有模式补三列：

   ```python
   self._ensure_columns("attachments", {
       "volume": "TEXT",
       "issue":  "TEXT",
       "pages":  "TEXT",
   })
   ```

   走的是与 `publication_title` / `doi` 完全相同的路径，**不会影响 fingerprint**（约束 1 已由现有列证明成立）。

3. 补齐后，让 `search-metadata` / `documents show` / `search-by-collection` 的输出带上这三个字段。
   （`attachments_fts` 是否要一并加列可选——卷页通常不需要全文检索，但 FTS 列漂移时需注意
   迁移，参见 `RETRIEVAL_ENHANCEMENT_FOLLOWUPS.md`「重要项 2」的 FTS 列漂移用例缺口。）

**在补齐之前**，skill 与文档应明确警告：本库不提供卷页，引用格式须回源核验。

### 临时处理（修复前）

> ⚠️ 这是绕过手段，不是修复。产物仅可用于**核验**，不要写进自动化流程。

从 shadow 库直接取：

```sql
-- 在 data/shadow/zotero.sqlite 中执行
SELECT f.fieldName, idv.value
FROM itemData id
JOIN itemDataValues idv ON idv.valueID = id.valueID
JOIN fields f ON f.fieldID = id.fieldID
WHERE id.itemID = ? AND f.fieldName IN ('volume','issue','pages');
```

或从 PDF 解析产物的页眉核验（见问题 6a，这是本次实际使用的、最可靠的手段）。
**两者都不应用来"猜"缺失值**——取不到就标"未核实"。

---

## 问题 4：`search-by-collection --limit N` 静默截断

### 现象

`--limit N` 返回**恰好 N 行**时即已截断，输出中**没有任何截断标志**。

### 证据（2026-09-18）

```bash
$ zoterorag search-by-collection 26TP7I8G --query "" --limit 300  → 300 行（已截断）
$ zoterorag search-by-collection 26TP7I8G --query "" --limit 1000 → 374 行（真实总数）
```

输出顶层键仅 `results`，无 `total` / `has_more` / `truncated` 字段：

```
搜索各面 top-level keys 实测：均为 ['results']
  $ zoterorag search-metadata     ... --limit 3  → ['results']
  $ zoterorag search-fulltext     ... --limit 3  → ['results']
  $ zoterorag search-by-collection ... --limit 1000 → ['results']
```

**即所有检索面共用同一返回外壳，都没有分页 / 截断元信息。**

### 影响

以 `--limit 300` 枚举「超微结构」集合会**静默漏掉 74 行 / 33 篇文献**。本次任务最初即以 300 为界，得到"137 篇文献"，与真实的 170 篇不符——若未复核，会产生**基于不完整语料的结论**。

### 建议方案

- 输出增加 `total` / `has_more`（或 `next_cursor`）。
- 至少在 `len(results) == limit` 时向 stderr 打印警告。

### 临时处理（修复前）

枚举大集合时，**返回值行数必须小于 `--limit`** 才能认为取全；否则逐次提高 limit 直至行数不再增长。

---

## 问题 5：MinerU 把直方图"补全"为原文不存在的数值表格

### 现象

`data/extract_cache/<hash>/extract/*_content_list_v2.json` 中存在 `type: "chart"` 的块，其 `content` 字段是**由视觉模型合成的 markdown 数值表格**。这些表格**在原文中并不存在**——原文只有直方图图像。

### 证据（2026-09-18，Vater, Covey & Casseday 1997 / 附件 `ANTNWASU`）

该文档的 `content_list_v2.json` 含 **3 个 `chart` 块**，例如：

```
{"type":"chart","content":{"image_source":{"path":"images/4173...jpg"},
 "content":"| Layer | Type | Count |\n| ----- | ---- | ----- |\n| IC | si..."}}
{"type":"chart","content":{"image_source":{"path":"images/1d41...jpg"},
 "content":"| somatic area (sq. um) | Number of cells |\n| ... |\n| 100-125 ..."}}
{"type":"chart","content":{"image_source":{"path":"images/5110...jpg"},
 "content":"| Number of perisomatic puncta | GABA | Gly |\n| ... |\n| 0 ..."}}
```

其中**第二、三张对应原文 Fig. 9 的直方图**（原文正文只给出范围值：甘氨酸 puncta 3–15、GABA puncta 0–4，以及平均 8 / 1.2）。**另有一张对应的图在原文中是纯示意图，不含任何数据。**

### 影响

1. 这些**合成数值会进入 `document.md`**，因而被 FTS 与向量索引收录、**可被检索命中**。
2. 因为是"看起来规整的表格"，很容易被下游（人或 LLM）当作**一手定量数据**引用。
3. 合成值与正文叙述可能不一致（本次即观察到 Fig. 5D 的 OCR 值 67% 与正文 "nearly completely" 存在张力）。

### 建议方案

- 在 `normalize` 阶段**识别并标注**来源于 `chart` 类型的块（例如在 `document.md` 中包一层 `<!-- mineru:chart-synthesized -->` 或加"⚠ 该表由图像合成，非原文数据"提示）。
- 或在 `document.md` 中**不渲染** `chart` 块的合成内容，只保留图片引用。
- 至少在 `USAGE.md` / skill 中记录：**凡来自直方图/曲线图的表格数值，一律不可作为一手数据引用。**

### 临时处理（修复前）

引用定量数据时，**只采信正文散文中的数值**，忽略一切 `chart` 来源的表格；必要时回原 PDF 读图。

---

## 问题 6：normalized `document.md` 剥离页眉页码 + 图注误分类

### 6a. 页眉/页码被剥离

`data/normalized/<key>/document.md` **不含任何页眉、页码、期刊卷页信息**。本次任务 5 篇的 `document.md` 中均检索不到期刊页眉（如 `Cell Tissue Res (1997) 289:223–233`）。

**后果**：无法从 normalized 产物核验书目信息；由于问题 3（state 无卷页），**整条链路都拿不到卷页**。

**可用替代**：`data/extract_cache/<hash>/extract/*_content_list.json` 中保留了 `header` 与 `page_number` 块。例如：

```
HEADERS: ['Cell Tissue Res (1997) 289:223–233']
PAGENUM: ['224','225','226','227','228','229','230','231','232','233']
```

建议在 `USAGE.md` 中记录此核验路径，或由 `normalize` 把页眉写回一个独立元数据文件。

### 6b. 图注被误分类为 `page_footnote`

**证据**（Vater 1995 / 附件 `8E95U86L`）：`document.md` 第 217 行只剩裸文本 `Figure 6`，**图注内容整体丢失**。回查 `content_list.json` 可见该图注被标为 `page_footnote`：

```
[page_footnote] Fig. 6. Electron micrographs show different types of boutons in MSO. A,B: Most
                perisomatic endings contain flattened vesicles and form either single or multiple
                contact zones. C: Large R-vesicle bouto...
```

而同一文档中正常的图注（如 Fig. 1–5）未受影响。**后果**：电镜图版失去图注，图版描述不可用（本次任务该篇的 Fig. 6 描述须从原始解析产物恢复）。

### 6c. 页脚残渣混入正文

`ANTNWASU/document.md` 中 `ig.c:&/f` 出现 **6 次**（为 PDF 每页页脚 `J Comp Neurol ...` 之类的解析残渣）。虽不影响语义，但会污染 FTS 与向量。

### 建议方案

- 6b：在 `normalize` 中把 `page_footnote` 里以 `Fig.` / `Figure` 开头的块重新归类为图注，回填到 `document.md` 的图注位置。
- 6c：对明显的页脚模式（短、重复、含不可打印字符组合）做过滤。

---

## 问题 7：仓库根目录存在 0 字节 `state.sqlite`

### 现象

```
$ ls -la state.sqlite
-rw-r--r-- 1 rog 197121 0 Sep 15 15:20 state.sqlite

$ ls -la data/state/state.sqlite     # 真实库
（正常大小）
```

根目录的 `state.sqlite` 是 **0 字节**，真实库位于 `data/state/state.sqlite`。

### 影响

排查问题时若按字面直觉打开 `./state.sqlite`，会得到**"表数量为 0"**的误导性结果（本次任务即先踩此坑，误以为库为空）。另外仓库中还有 `data/tmp_bat_ic_*.json` 等散落的中间产物。

### 建议方案

- 删除该 0 字节文件；若确需存在，写入明确的占位说明或加入 `.gitignore`。
- 顺带清理 `data/tmp_*.json`、`data/diagnostics_tmp/`、`data/tmp_timeout_check/` 等散落产物。

---

## 8. 复盘：这些问题如何导致了一次实际错误

本次检索任务的最终报告（`report/20260918/蝙蝠听觉核团超微结构研究汇总报告.md`）初版中，**5 篇文献有 3 篇的卷页写错**：

| 文献 | 初版（错误） | 实际 |
|---|---|---|
| Vater, Covey & Casseday 1997 | 289(2):203–217 | **289:223–233** |
| Kemmer & Vater 1997 | 287(3):435–450 | **287:487–506** |
| Nakajima 1971 | 50(1):121–**142** | 50(1):121–**134** |

**因果链**：

1. `search-metadata` 返回的记录**不含卷页**（问题 3）；
2. 撰写者未察觉这是**数据缺失**，而当作"未显示"，凭常见卷期**臆测补全**；
3. normalized `document.md` **也查不到页眉**（问题 6a），无法自查；
4. 最终由**子代理回查 PDF 解析产物**才发现，报错后回 `content_list.json` 核验更正。

**值得记下的一点**：这个错误在库内其实**本可被拦下**——Nakajima 那条记录的
`doi = 10.1083/jcb.50.1.121` 已经写明是卷 50 / 期 1 / 起页 121，足以证伪我写的 `121-142`。
但 CLI 没有任何路径把 DOI 里的卷期信息暴露成可用字段（见问题 3 影响 4），
于是"库里有线索"与"用户拿得到"之间断开了。**这类"数据在但不通"的缺口，比纯粹的数据缺失更隐蔽。**

**教训**：本库的元数据**不足以支撑完整引用格式**。修复问题 3 之前，任何从本库产出的报告都必须：(a) 明确标注卷页为"未核实"，或 (b) 回 `data/extract_cache/*/extract/*_content_list.json` 的 `header` 块核验。**绝不可凭常见格式补全。**

**流程教训**：本次错误是**子代理复核**发现的，不是自查发现的。说明在本库当前的数据完整度下，
**"引用格式必须经第二方独立核验"应作为硬性步骤**，而不是可选项。

---

## 相关文件

| 路径 | 说明 |
|---|---|
| `data/state/state.sqlite` | 真实状态库（`attachments` 表缺 volume/issue/pages 列） |
| `data/shadow/zotero.sqlite` | Zotero 影子库（卷页在此可用） |
| `data/normalized/<attachment_key>/document.md` | 检索用全文（无页眉页码） |
| `data/extract_cache/<hash>/extract/*_content_list.json` | MinerU 原始解析（含 header / page_number / chart 块） |
| `src/zoterorag/search/metadata.py:57-86` | 元数据检索的返回构造（**问题 1 根因**：`:59` 改写 `document_id`、`:60` 改写 `chunk_id`） |
| `src/zoterorag/zotero/shadow.py:233-250` | 读影子库的 SELECT，硬编码 6 个书目字段（**问题 3 根因起点**） |
| `src/zoterorag/db/state.py:533-550` | 书目专用列的建表位置（问题 3 根因下游） |
| `skills/zoterorag-query.md` | 受影响：步骤 6 的 `documents show <document_id>` |
| `docs/DATA_HYGIENE_ISSUES_20260909.md` | 相邻问题域（删除失效、僵死 job） |
| `docs/RETRIEVAL_ENHANCEMENT_FOLLOWUPS.md` | 相邻问题域（元数据检索实现审核遗留项；其「次要项 1」即问题 3 涉及的 `metadata_json` 约束） |

---

## 附：核查命令

```bash
# 问题 1：document_id 语义冲突
.conda/Scripts/zoterorag.exe search-metadata "Nakajima" --limit 5      # document_id = 父条目 key
.conda/Scripts/zoterorag.exe search-fulltext "calyciferous" --chunk-type text --limit 5  # document_id = 附件 key
.conda/python.exe -c "import sqlite3;c=sqlite3.connect('data/state/state.sqlite');\
print(c.execute(\"SELECT attachment_key,parent_key FROM attachments WHERE attachment_key='B6CHNRDC'\").fetchall())"

# 问题 2：documents show 静默 null
.conda/Scripts/zoterorag.exe documents show NTXYZEAV    # → {"document": null}
.conda/Scripts/zoterorag.exe documents show B6CHNRDC    # → 正常

# 问题 3：书目字段清单漏收卷页（Zotero 有 vs state 无落点）
.conda/python.exe -c "import sqlite3;c=sqlite3.connect('data/shadow/zotero.sqlite');\
print(c.execute('''SELECT f.fieldName,idv.value FROM itemData id JOIN itemDataValues idv ON idv.valueID=id.valueID
JOIN fields f ON f.fieldID=id.fieldID WHERE id.itemID=149 AND f.fieldName IN ('volume','issue','pages')''').fetchall())"
.conda/python.exe -c "import sqlite3;c=sqlite3.connect('data/state/state.sqlite');\
print([r[1] for r in c.execute('PRAGMA table_info(attachments)')])"      # 无 volume/issue/pages
.conda/python.exe -c "import sqlite3;c=sqlite3.connect('data/state/state.sqlite');\
print(c.execute(\"SELECT metadata_json FROM attachments WHERE attachment_key='B6CHNRDC'\").fetchone())"   # {} 系刻意设计
# 佐证：全仓库读取的 Zotero 书目字段只有 6 个，无 volume/issue/pages
grep -rn "fieldName" src/zoterorag --include=*.py
# → 仅 title / abstractNote / date / url / publicationTitle / DOI

# 问题 4：limit 静默截断（行数 == limit 即已截断）
.conda/Scripts/zoterorag.exe search-by-collection 26TP7I8G --query "" --limit 300  | grep -c document_id
.conda/Scripts/zoterorag.exe search-by-collection 26TP7I8G --query "" --limit 1000 | grep -c document_id

# 问题 5：MinerU 合成图表
#   在 data/extract_cache/<hash>/extract/*_content_list_v2.json 中搜索 type=="chart"

# 问题 6：页眉可核验 / 图注误分类
#   data/extract_cache/<hash>/extract/*_content_list.json 中查看 type=="header" / "page_number" / "page_footnote"
grep -c "ig\.c:&/f" data/normalized/ANTNWASU/document.md    # 页脚残渣
sed -n '217p' data/normalized/8E95U86L/document.md           # 只剩 "Figure 6"

# 问题 7：0 字节 state.sqlite
ls -la state.sqlite data/state/state.sqlite
```
