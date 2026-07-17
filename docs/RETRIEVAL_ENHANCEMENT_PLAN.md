# ZoteroRAG 检索增强方案（修订版）

> 本版已吸收对照代码的评审结论：修正指纹/回填假设、写死 FTS 迁移与 collection 过滤路径、补齐默认 score 权重与集合递归语义。  
> **范围：实现指引。** 数据库备份位于项目根目录 `backups/`；单元测试位于 `tests/`。

## 1. 现状与问题

基于对 `search/`、`index/`、`db/state.py`、`zotero/*` 的代码复核，现状如下：

| 维度 | 现状 | 关键位置 |
|------|------|----------|
| Zotero shadow | `create_shadow_copy()` 已做**完整只读克隆**（整个 `zotero.sqlite` 复制到 `data/shadow/`），但 `list_attachments()` 只读取了其中一部分表 | `zotero/shadow.py` |
| 元数据已抓取 | `title`、`abstract`、`date_value`、`url`、`classification`、`source_quality` | `zotero/shadow.py`、`db/state.py` |
| 元数据缺失 | **作者、Zotero collections/folders、tags、item type、publication title、DOI、date added/modified** | 未读取 `creators`、`collections`、`collectionItems`、`tags` 等表 |
| 元数据检索 | FTS5 仅覆盖 `title/abstract/relative_path/url/date_value/...`，仅支持 `classification` 过滤；使用自定义加权匹配分排序（title 5/abstract 2/...） | `search/metadata.py`、`db/state.py` |
| 文件夹/集合检索 | **未实现**；`state.sqlite` 无 collection 业务表 | — |
| 查全率短板 | ① 查询按空格切词且所有词必须命中（AND 语义）；② `_build_fts5_query` 给所有词加引号，禁用了 FTS5 前缀；③ 无短语/OR/停用词；④ 元数据预筛选能力极弱 | `db/state.py` `_build_fts5_query` |
| FTS 迁移缺口 | `_migrate_fts5()` 仅 `CREATE IF NOT EXISTS`，且只在 `count==0` 时回填；**已有表缺列时不会重建** | `db/state.py` `_migrate_fts5` |
| 回填机制 | `upsert_attachments()` **始终** `INSERT ... ON CONFLICT DO UPDATE`，`scan_status`（含 `unchanged`）只是标签，**不会跳过写库** | `db/state.py` `upsert_attachments` |
| 不重新建库 | 可行：`_ensure_columns()` 增量加列 + 强化后的 FTS 列检测重建 + 普通 scan 回填；**无需重跑 MinerU 和嵌入**（ingest 按 classification + artifact/embedding 检查点选人，不看 `scan_status`） | `db/state.py`、`pipeline/ingest.py` |

## 2. 目标与约束

1. **元数据检索增强**：支持标题/作者/时间/DOI/期刊/tag 搜索，并支持按时间、标题、相关性排序；新增 BM25 排序作为可选项。
2. **文件夹/集合分区检索**：把 Zotero 的 collections/folders 同步进来，支持按 collection 浏览与过滤。
3. **查全率提升**：解决全大写、停用词、AND-only、前缀被禁用等导致的漏召；通过查询预处理支持短语、前缀、OR 模式。
4. **检索方法保持独立**：混合检索遵循 skill 中“逐方法检索 + 人工交叉验证”的策略；**不实现自动回退/混合机制**，各检索方法失败时按现有方式标准报错。
5. **不重新建库、不重跑 MinerU/嵌入**：仅通过增量 schema 迁移 + shadow 扫描补齐 `state.sqlite` 中的元数据与 FTS5 索引。

### 2.1 已锁定的设计决策（不再留 A/B）

| 决策点 | 锁定选择 |
|--------|----------|
| Collection 存储 | **双写**：`collections_json`（展示 + FTS 展开文本）+ `attachment_collections`（**唯一**精确过滤路径） |
| Collection 过滤 | **禁止**用 `LIKE` 或仅用 JSON 做过滤；一律走 `attachment_collections` |
| 子集合递归 | `search_by_collection(..., include_children=False)` **默认 false**；可选 `True` 时沿 `parent_collection_key` 向下展开 |
| 指纹与回填 | **bibliographic 回填不依赖**扩 `attachment_fingerprint`；见 §4.4 |
| FTS 迁移 | 列集合不一致 → **必须** `DROP` + 重建 + 全量回填；upsert 的 FTS INSERT 列清单必须同步 |
| 默认排序 | 仍为自定义 `metadata_match_score` / `chunk_match_score`；BM25 为显式 `sort_by="bm25"` |
| BM25 score 语义 | API/CLI 对外统一返回 **越大越相关** 的 `score`（内部对 `bm25()` 取负或等价变换） |
| `heading_path` 进 `chunks_fts` | **不在本方案必做范围**（可另开任务） |
| Hybrid / 自动回退 | **不做** |

## 3. 总体策略

- **Phase 1**：补齐 Zotero 元数据与集合信息（作者、collections、tags、item type、publication title、DOI 等）+ 安全回填 + FTS 列迁移。
- **Phase 2**：增强元数据检索（过滤、排序、BM25）与文件夹分区检索；扩展默认 score 权重；API/CLI/MCP。
- **Phase 3**：查询预处理与前缀/短语/OR；fulltext BM25；**不做**自动回退/混合。

## 4. 详细实施步骤

### Phase 1：补齐 Zotero 元数据与集合信息

#### 4.1 扩展 shadow 查询（`src/zoterorag/zotero/shadow.py`）

> `create_shadow_copy()` 已是完整克隆，无需改复制逻辑；只扩展读查询。

- 修改 `ZoteroAttachment` dataclass，新增字段：
  - `authors: list[str]` — 从 `creators` + `itemCreators` + `creatorTypes`，**仅** `creatorType.typeName = 'author'`，按 `orderIndex` 排序；姓名拼接为 `trim(firstName || ' ' || lastName)`，空段去掉；机构名（常见仅 `lastName`）保留。
  - `collections: list[dict[str, str]]` — 从 `collectionItems` + `collections` 取 `collection_key`、`collection_name`、`parent_collection_key`。
  - `tags: list[str]` — 从 `itemTags` + `tags` 取 `name`。
  - `item_type: str | None` — parent item 的 `itemTypes.typeName`（attachment 自身永远是 `attachment`）。
  - `publication_title: str | None`、`doi: str | None` — `itemData`/`itemDataValues` 的 `publicationTitle`、`DOI`。
  - `date_added: str | None`、`date_modified: str | None` — parent（或 attachment）`items.dateAdded` / `dateModified`。
- 修改 `ZoteroShadow.list_attachments()` SQL：
  - 单值字段：LEFT JOIN `itemTypes`、额外 `itemData`（publicationTitle、DOI）、日期字段。
  - **多值字段必须先子查询/CTE 聚合再 LEFT JOIN**，禁止直接 JOIN `itemCreators` / `collectionItems` / `itemTags`（防笛卡尔积）。
  - `GROUP_CONCAT` **分隔符用 `|`**（避免与人名内逗号冲突）；Python 侧按 `|` split 并 strip。
  - 多值子查询一律基于 `COALESCE(p.itemID, a.itemID)`（collections/tags/creators 挂在 parent item 上）。
- 另增 `ZoteroShadow.list_collections()`：扫描 `collections` 表（含 `libraryID`、`parentCollectionID` → key），供 `zotero_collections` 全量同步；**不要**在每条 attachment 上重复展开全集定义。

#### 4.2 扩展状态库 schema（`src/zoterorag/db/state.py`）

- `attachments` 经 `_ensure_columns()` 增量添加：
  - `creators_json TEXT NOT NULL DEFAULT '[]'`
  - `collections_json TEXT NOT NULL DEFAULT '[]'`
  - `tags_json TEXT NOT NULL DEFAULT '[]'`
  - `item_type TEXT`
  - `publication_title TEXT`
  - `doi TEXT`
  - `date_added TEXT`
  - `date_modified TEXT`
  - `date_year INTEGER`（从 `date_value` 解析出的年份，解析失败为 NULL）
  - `date_normalized TEXT`（尽量规范为 `YYYY` / `YYYY-MM` / `YYYY-MM-DD`，解析失败保留原串或 NULL）
- 新建表（一次性 `CREATE TABLE IF NOT EXISTS`）：
  - `zotero_collections`：
    - `collection_key TEXT PRIMARY KEY`
    - `collection_name TEXT NOT NULL`
    - `parent_collection_key TEXT`
    - `library_id INTEGER`
  - `attachment_collections`：
    - `attachment_key TEXT NOT NULL`
    - `collection_key TEXT NOT NULL`
    - `PRIMARY KEY (attachment_key, collection_key)`
    - 索引 `idx_attachment_collections_collection` on `(collection_key)`
- 扩展 `attachments_fts` 列：`creators`、`collections`、`tags`、`publication_title`、`doi`（由 JSON/文本列展开为可搜索纯文本）。
- **FTS 迁移（写死）**：在 `_migrate_fts5()` 中：
  1. 定义期望列集合（有序）。
  2. **列检测方式锁定为解析 `sqlite_master.sql` 字符串**（与现有 `content=` 检测同风格；避免 FTS5 虚表 `PRAGMA table_info` 的隐藏列/rank 边界）。若表不存在、或列集合不一致、或仍是 `content=` 旧式 → `DROP TABLE IF EXISTS attachments_fts`。
  3. `CREATE VIRTUAL TABLE attachments_fts USING fts5(...)`（完整新列集）。
  4. **全量回填**（不仅 `count==0`）：从 `attachments` SELECT，把 `creators_json` / `tags_json` / `collections_json` 展开为空格分隔文本（可用 `json_each` 拼接或 Python 回填辅助）。
  5. 对 `chunks_fts` 本 Phase **不改列**（仅保持现有行为）。
- 索引：`idx_attachments_item_type`；可选 `(date_normalized, title)`；`attachment_collections(collection_key)` 如上。

#### 4.3 更新写入流程（`classifier.py` + `scanner.py`）

- 扩展 `ClassifiedAttachment.to_record()`：写入新 bibliographic 字段（含 JSON 序列化后的 creators/collections/tags）到**独立列**。
- **禁止**把 authors/collections/tags/doi 等新书目字段写入 `metadata_json`。现有 `attachment_fingerprint()` 的 payload 包含 `metadata`；今日 `to_record()` 固定 `"metadata": {}`，保持该约定不变。若把 biblio 塞进 `metadata_json`，会经指纹把整库刷成 `scan_status="changed"`，与 §4.4 目标冲突。
- 在 `scan_shadow_to_ledger()` 中：
  1. `list_attachments()` → classify → `upsert_attachments(records)`（含新列 + 同步更新 **attachments_fts INSERT 列清单**）。
  2. `list_collections()` → `ledger.upsert_collections(...)`。
  3. 根据每条 attachment 的 collections 列表 → `ledger.replace_attachment_collections(attachment_key, keys)`（先删后插或等价替换，保证与 shadow 一致）。
  4. 全量 scan（`limit is None`）时继续 `mark_absent_attachments_deleted`；并对不再出现的 collection / 关联行做清理（`prune_collections` / 按现存 attachment 清理关联）。注意：现有 `mark_absent_attachments_deleted` **不会**自动清 `attachment_collections`，必须由 scanner 显式 prune。

#### 4.4 增量回填与指纹策略（已修正）

**纠正旧版错误假设**：`upsert_attachments` 不会因 `scan_status="unchanged"` 跳过 UPDATE。只要扩展 `to_record()` + upsert 列清单，普通 `scan_shadow_to_ledger()` / `resync-metadata` 就会写入新字段。

**指纹策略（锁定）**：

1. **不要**把 bibliographic 新字段（authors/tags/collections/doi/…）并入现有 `attachment_fingerprint()` 的“内容变更”语义，以免首次 resync 把整库标成 `changed`（虽不自动触发 extract，但污染 review/UI 信号）。
2. 现有 `attachment_fingerprint()` 继续覆盖：parent/content_type/path/title/abstract/date/url/classification/source_quality/reasons/file 事实/`metadata` 等**既有**字段（与今日行为兼容）。`metadata` 继续保持为空对象 `{}`，不承载 biblio。
3. `biblio_fingerprint`：**仅内存诊断**（对 creators/collections/tags/item_type/publication_title/doi/date_added/date_modified 做稳定哈希），**不入库**、**不改 `scan_status`**。仅 biblio 变时：`scan_status` 保持 `unchanged`。
4. CLI：
   - `resync-metadata`：刷新 shadow（若需要）→ `scan_shadow_to_ledger` → 确保 FTS 列迁移已发生。
   - `resync-metadata --rebuild-fts`：强制 DROP/重建 `attachments_fts` 并全量回填（不触发 MinerU/嵌入）。
5. **约束**：本阶段所有路径只改 `state.sqlite` 与 FTS；不修改 ingest 触发条件，不调用 extract/embed。

同步更新的函数列清单（必做）：

- `upsert_attachments`（attachments 列 + **attachments_fts DELETE/INSERT 列**）
- `list_attachments` / `get_attachment` / `search_attachments_metadata` 的 SELECT
- 新增 `upsert_collections` / `replace_attachment_collections` / `list_collections`（ledger 侧）

### Phase 2：增强元数据检索与文件夹分区检索

#### 4.5 增强 `metadata_search`（`search/metadata.py` + `db/state.py`）

扩展参数：

- `date_from` / `date_to` — 优先对 `date_normalized`（及必要时 `date_year`）比较；文档与 docstring **标明**：未规范化的历史脏日期可能漏召。
- `authors: list[str] | None` — 经 `json_each(creators_json)` 做子串/精确策略（实现时固定一种：默认 casefold 子串匹配作者名元素）。
- `collections: list[str] | None` — **仅**通过 `attachment_collections` 过滤。
- `tags: list[str] | None` — `json_each(tags_json)`，禁止 `LIKE` JSON 整串。
- `item_type` / `publication_title` / `doi`
- `attachment_keys: list[str] | None` — 预过滤（供 collection 分区检索复用）。
- `sort_by: str = "score"` — 白名单：`score` | `bm25` | `date` | `date_added` | `title`
- `sort_order: str = "desc"` — 白名单：`asc` | `desc`

**双路径检索（写死，因现状空 query 直接 `return []`）**：

当前 `search_attachments_metadata` 在 terms 为空时短路返回 `[]`，且整体是 FTS-MATCH 优先。Phase 2 必须拆成：

1. **文本路径**：有查询词 → FTS5 `MATCH`（失败则 LIKE 回退）→ 再叠加过滤器（日期/作者/collection/tag/…/`attachment_keys`）→ 排序截断。
2. **纯过滤路径**：无查询词、但存在任一过滤器或 `attachment_keys` → **不走** `attachments_fts MATCH`，直接对 `attachments`（+ `attachment_collections` / `json_each`）做参数化 SQL 过滤 → 排序截断。
3. 无查询词且无任何过滤器 → 仍返回 `[]`（保持“空搜不扫库”）。

`sort_by="bm25"` **仅**在文本路径且 FTS 可用时生效；纯过滤路径若请求 bm25，回退为 `score` 或 `date`（实现时固定为回退 `date_added desc`，并在 API 文档注明）。

排序与评分：

- 默认 `sort_by="score"`：继续用 Python `metadata_match_score`（LIKE 回退同理）。
- **必须扩展权重**（锁定建议值），且打分侧使用**派生纯文本**字段，**禁止**对原始 JSON 串 `str(creators_json)` 做子串匹配（会命中 `["` 等噪声）：

| 字段 | 权重 |
|------|------|
| title | 5.0（保持） |
| doi | 5.0 |
| creators_text（作者名空格拼接） | 4.0 |
| publication_title | 3.0 |
| tags_text | 3.0 |
| abstract | 2.0（保持） |
| collections_text（集合名空格拼接） | 1.5 |
| relative_path | 1.5（保持） |
| url / date_value / keys | 1.0（保持） |

- `sort_by="bm25"`：FTS5 `ORDER BY bm25(attachments_fts)`；对外 `score = -bm25(...)`（或文档化等价映射），保证与其它排序一样“越大越好”。
- `build_metadata_text` / 返回 `metadata` 字典包含：authors、collections、tags、item_type、publication_title、doi、date_added、date_modified。

#### 4.6 文件夹/集合模块（`src/zoterorag/search/collections.py`）

- `list_collections(ledger)` → 扁平列表（含 parent key）。
- `get_collection_tree(ledger)` → 基于 `parent_collection_key` 建树。
- `search_by_collection(ledger, collection_key, query=None, *, include_children=False, **filters)`：
  1. 解析目标 collection key 集合：默认仅自身；`include_children=True` 时 BFS/DFS 收集全部后代 key。
  2. `SELECT attachment_key FROM attachment_collections WHERE collection_key IN (...)`。
  3. 将该 key 集作为 `metadata_search(..., attachment_keys=...)` 预过滤；若 `query` 为空，走 §4.5 **纯过滤路径**列出该分区内附件（仍受 limit 约束）。
- **不做** JSON 过滤备用路径。

#### 4.7 API / CLI / MCP

- API：`/search/metadata` 增参；`GET/POST /search/collections`；`/search/by-collection`（含 `include_children`）。
- CLI：`search-metadata` 增旗标；`list-collections`；`search-by-collection [--include-children]`；`resync-metadata [--rebuild-fts]`。
- MCP：扩展 `zotero_rag_search_metadata`；新增 `zotero_rag_list_collections`、`zotero_rag_search_collection`。

### Phase 3：查全率提升（查询预处理，不做混合）

#### 4.8 `search/query_parser.py`

- `parse_query(query) -> QueryPlan`：
  - `"..."` → phrase
  - `term*` → prefix（**不要**再对该 token 整体加 FTS 引号包死通配）
  - `+term` / `-term` → must / exclude
  - 停用词：默认英文小表 + 可配置列表；中文停用词非必做
  - 停用词删光 → **返回空结果**，不要发非法 `MATCH`
- 生成 FTS5：
  - phrase → `"phrase words"`
  - prefix → `term*`
  - 普通词 → 继续 quote 防特殊字符；默认 AND；`match_mode="any"` → OR
- 不做词干化、不引入新依赖。
- `metadata_search` / `fulltext_search` 入口先 `parse_query`；保留 `casefold` 大小写不敏感行为。

#### 4.9 fulltext BM25 与范围边界

- `fulltext_search` 增加 `sort_by="bm25"`（默认仍 `chunk_match_score`）；对外 score 同样“越大越好”。
- **不**把 `heading_path` 加入 `chunks_fts`（本方案不做）。
- **不**实现 `hybrid_search` / 向量自动回退。

#### 4.10 检索面独立性（重申）

- `/search/metadata`、`/search/fulltext`、`/search/text`、`/search/multimodal` 语义不变。
- 失败按现有方式抛错；上层 skill 负责多方法交叉验证。

## 5. 里程碑

| 里程碑 | 内容 |
|--------|------|
| M1 | Phase 1：shadow 扩展、schema、关联表、FTS 列检测重建、upsert/FTS 列同步、scanner 写 collections、resync-metadata、指纹策略按 §4.4 |
| M2 | Phase 2：过滤/排序/BM25、score 权重扩展、collections 模块（含 `include_children`）、API/CLI/MCP、测试 |
| M3 | Phase 3：query_parser、前缀/短语/OR、停用词、fulltext BM25 |

## 6. 验证方式

- 测试：
  - `tests/test_metadata_search.py`：作者/日期/collection/tag/排序（含 bm25）；score 权重命中作者/doi；**空 query + 过滤器**走纯过滤路径；无词无过滤仍 `[]`。
  - `tests/test_collections.py`：树、过滤、`include_children` true/false；空 query 分区列出。
  - `tests/test_query_parser.py`：短语、前缀、停用词删空、OR。
  - **旧查询语义回归**：现有 `tests/test_metadata_search.py` / fulltext 相关用例在 Phase 3 改 `_build_fts5_query` 后仍通过（默认 AND、普通词仍 quote）。
  - `tests/test_api_server.py`：旧 `/search/metadata` 无新参仍可用。
  - 指纹相关：仅 biblio 变更时 **不应** 把 `scan_status` 刷成整库 `changed`；`metadata_json` 保持 `{}`；ingest plan 无大规模新 extract。
- 真实/测试 shadow：authors（含仅 lastName）、嵌套 collection、tags。
- `resync-metadata --rebuild-fts` 只动 state/FTS，不触达 MinerU/Qwen。

## 7. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 旧库 FTS 缺列仍可启动但搜不到新字段 | `_migrate_fts5` 解析 `sqlite_master.sql` 做列检测 → DROP/重建/全量回填；与 upsert FTS 列清单同步 |
| 误以为 fingerprint 决定是否写库 | 以 upsert 始终 UPDATE 为准；biblio 不并入 content fingerprint |
| 误将 biblio 写入 `metadata_json` → 经 `attachment_fingerprint(metadata=...)` 整库 `changed` | §4.3 禁止；`to_record()` 保持 `metadata: {}`；测试断言 |
| 整库 `scan_status=changed` 噪声 | §4.4 指纹策略；校验 ingest 仍不看 scan_status |
| 空 query + 过滤器时现有代码直接 `[]` | §4.5 纯过滤路径；显式绕开 FTS MATCH |
| 多值 JOIN 笛卡尔积 | 子查询/CTE + `GROUP_CONCAT` 后再 JOIN |
| JSON `LIKE` 误匹配 / 打分吃到 JSON 噪声 | creators/tags 用 `json_each`；打分用 `*_text` 派生字段；collection **只**用关联表 |
| `sort_by` SQL 注入 | 白名单 |
| 日期格式混乱 | `date_year` / `date_normalized`；文档标明残留局限 |
| BM25 原始值越小越好造成 API 困惑 | 对外统一取负/映射为越大越好 |
| shadow 聚合 SQL 变慢 | CTE 聚合；collections 定义用独立 `list_collections` |
| Phase 3 改共用 `_build_fts5_query` 破坏旧语义 | 默认仍 AND + 普通词 quote；加回归测试 |
| 误触发 MinerU/嵌入 | 不改 ingest 选人逻辑；验证 plan 无异常 extract |

## 8. 明确不在范围内

- parent-child / sentence-window chunking
- cross-encoder rerank（保持 `RerankNotSupportedError`）
- 向量侧 hybrid / RRF / 自动回退
- 把 authors/tags 反规范化进每个 chunk 或向量 metadata
- `heading_path` → `chunks_fts`（另开任务）

## 9. 修订说明（相对上一版文档）

| 旧版问题 | 本版处理 |
|----------|----------|
| “不扩 fingerprint → 新字段永不写入” | 已纠正：upsert 始终写库；biblio 不并入 content fingerprint |
| collection “短期 JSON / 长期关联表”双路径 | 写死：JSON 展示 + 关联表过滤 |
| FTS 迁移描述不足 | 写死：解析 `sqlite_master.sql` 列检测 → DROP/重建 + upsert FTS 列同步 |
| 未扩 `metadata_match_score` | Phase 2 锁定权重表 + `*_text` 派生字段 |
| 未定义子集合递归 | 默认 `include_children=False` |
| creators 未限定类型/分隔符 | 仅 `author`；`GROUP_CONCAT` 用 `\|` |
| BM25 对外 score 语义不清 | 统一越大越好 |
| `heading_path` 可选混入 M3 | 移出必做范围 |
| `metadata_json` 冗余备份与指纹冲突 | **禁止** biblio 进 `metadata_json`；保持 `{}` |
| 空 query 分区列出未点明 | §4.5 写死文本路径 + 纯过滤路径 |
| `biblio_fingerprint` 可选过多 | 锁定为仅内存诊断、不入库、不改 `scan_status` |

## 10. 独立审核结论（修订版）

独立只读审核（对照 `upsert_attachments` / `attachment_fingerprint` / `_migrate_fts5` / ingest 选人 / shadow / score / API·CLI·MCP 签名）结论：

- **总评**：可行；审核指出的 2 个重要项与 3 个次要项已回写进上文（§4.2–4.6、§6–7、§9）。
- **与代码一致**：upsert 恒写库、ingest 不看 `scan_status`、FTS 缺列不重建、`_build_fts5_query` 全 AND+引号、现有 metadata 入口均可选加参、无 hybrid/collection 表。
- **可以开工**：上述修正落地后，方案与代码无实质冲突；实现按 M1→M2→M3 推进即可。
