# 检索增强：已知后续项

记录实现落地后（commit `72d338c`）独立审核指出的重要项与次要项，便于后续排期修复。  
对照方案：[`RETRIEVAL_ENHANCEMENT_PLAN.md`](RETRIEVAL_ENHANCEMENT_PLAN.md)。

## 背景

- 审核结论：有条件通过，已本地 commit。
- 范围：书目元数据、集合分区、查询预处理、BM25；无 hybrid / 无重嵌。

---

## 重要项

### 1. 作者 / 标签过滤走 Python，而非方案中的 `json_each` SQL

**方案预期：** 用 SQLite `json_each(creators_json)` / `json_each(tags_json)` 在数据库侧精确过滤。

**现状：**  
在 [`src/zoterorag/db/state.py`](../src/zoterorag/db/state.py) 中，先用 SQL 加载候选（collection / 日期 / DOI 等），再经 `_passes_python_filters` 做：

- 作者：casefold 子串匹配；
- 标签：casefold 精确匹配。

候选加载受 `_METADATA_SCAN_CAP = 5000` 限制。

**影响：**

| 场景 | 风险 |
|------|------|
| 附件规模远小于 5000（当前库量级） | 基本无感 |
| 空查询 + 仅 `--author` / `--tag` 的纯过滤，且库很大 | 可能只在前 5000 行内筛，静默漏召 |
| `--collection` / 日期 / DOI 等 SQL 侧过滤 | 不受此上限影响 |

**建议修复：** 将 authors/tags 过滤下推为 `json_each`（或等价）SQL；或至少在「仅 author/tag 纯过滤」路径取消/提高扫描上限并分页。

### 2. 测试缺口

| 缺口 | 说明 |
|------|------|
| FTS 列漂移迁移 | 缺少「旧 `attachments_fts` 缺列 → DROP + 重建 + 全量回填」的回归用例 |
| `metadata_search(..., collections=...)` | 几乎只靠 `search_by_collection` 间接覆盖，缺少「过滤只走 `attachment_collections`」的直接断言 |

**建议修复：** 在 `tests/test_state_ledger.py` 或 `tests/test_metadata_search.py` 中补上述用例。

---

## 次要项

### 1. `biblio_fingerprint` 诊断未实现

方案中的「仅内存、不入库、不改 `scan_status`」的书目指纹未做。  
核心约束已满足：书目字段不进 `attachment_fingerprint()`、`metadata_json` 保持 `{}`。

### 2. LIKE 回退不扫作者 / 标签展开文本

FTS 失败或 0 命中时，LIKE 候选仍主要扫 title / abstract 等历史字段，不扫 creators/tags/collections 展开文本。FTS 正常时影响小。

### 3. `_like_candidate_keys(..., filters)` 未使用 `filters`

函数签名接收 `filters`，LIKE 候选阶段未使用，属死参数 / 未完成优化。

### 4. API 测试未覆盖新端点

`tests/test_api_server.py` 未扩展 `/search/collections`、`/search/by-collection`、`/resync-metadata` 等；旧 `/search/metadata` 签名兼容性仍在。

### 5. API / MCP `search-by-collection` 未透传 `match_mode`

CLI 元数据检索支持 `--match-mode`；集合内搜索经 API/MCP 时未必带上 OR 语义。方案未强制，属接口完整度问题。

---

## 非问题说明（避免误解）

- **翻译 / `needs_review` / `report_only` 附件仍会出现在元数据检索结果中**（除非显式 `--classification`）。它们默认不进入 ingest 建库（仅 `included_auto` / `included_manual`）。若产品上要求「扫描入库但元数据检索也不可见」，需另开需求。
- **日期等既有字段变更**会使 `scan_status=changed`（在 content fingerprint 内），与「书目专用字段不刷 fingerprint」不冲突。

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-07-17 | 根据检索增强实现审核结论初稿 |
