# ZoteroRAG 全项目审查与改进规划（2026-08-05）

> 本文档由 2026-08-05 的全项目只读审查产生（6 个并行只读子代理，覆盖：检索/索引、入库流水线、API/MCP/CLI、配置/数据层、文档、测试工程）。
> 审查全程未修改任何文件、未触碰 `data/` 数据库。原始会话规划文件同名内容。
> 后续讨论与方向决策见 [`FUTURE_PLAN_20260805.md`](FUTURE_PLAN_20260805.md)。

## 数据兼容性分级（本规划核心标注）

- **A 类｜纯代码改动，完全复用现有数据库**：只改 `src/`、`tests/`、文档；读取现有 SQLite/LanceDB 数据的方式不变或仅新增只读路径。零重建风险。
- **B 类｜在线迁移/回填，不重嵌不重新抽取**：要动 `data/` 内数据，但向量与抽取产物原样保留（如 FTS 表重建回填、向量记录元数据 UPDATE、旧版本向量清理）。执行前先跑 `zoterorag backup create`。
- **C 类｜需要重建/重嵌**：chunk 划分或向量空间变化，需重新 normalize + embed（花钱调 API）。本规划**不建议**近期做，仅列出供决策。

---

## 一、正确性 Bug（建议最优先，全部 A 类）

### 检索/接口层（用户可直接感知）

1. **MCP/API 向量检索静默使用 Stub 伪向量** — `embeddings/indexer.py:324` 默认 `allow_stub_provider=True`；MCP `tools.py:212-285` 无 provider 参数，对 qwen 建的索引永远返回噪声结果且无任何告警；API `/search/text` 默认 `embedding_provider="stub"` 同理。修法：MCP/API 改 `allow_stub_provider=False`（走已有 qwen 回退），或在响应中显式标注 provider。
2. **`rerank=true` 在 API 三端点变 500、在 MCP 静默返回空** — `api/app.py:439,472,493` 未捕获 `RerankNotSupportedError`；`mcp/tools.py:91-93` 把错误包装成 `{"results":[]}`，与"零命中"无法区分。修法：统一返回 400/501；MCP schema 移除 rerank 参数。
3. **MCP 组合检索假融合** — `mcp/tools.py:183-233`：三路结果直接拼接截断，元数据结果会先占满 top_k。修法：RRF（按 rank 倒数加权）或按来源配额轮询。
4. **查询解析器两处语义错误** — `search/query_parser.py:94-96`：`+foo*` 的 must 语义在 `match_mode=any` 时被丢弃；`:35,70-78`：`-"短语"` 被拆成丢弃的 `-` + **正向**短语（想排除实际变包含）。
5. **全文检索结果 title 是附件 key 而非标题** — `search/fulltext.py:51`，同 schema 三种 title 语义。修法：join attachments 取 title。
6. **FTS 重建后已删除附件"复活"** — `db/state.py:787-795` 回填 SQL 未排除 `classification='deleted'`。一行 WHERE 修复（A 类代码修复；若曾触发过重建，下次 resync 自愈）。
7. **Zotero 回收站条目未排除** — `zotero/shadow.py:166-258` 查询未过滤 `deletedItems`，拖进回收站的 PDF 仍会入库。修法：SQL 加 `NOT IN (SELECT itemID FROM deletedItems)`（A 类；已入库的回收站条目下次 scan 自动标记 deleted，无需手工清库）。
8. **`zotero_rag_status` 硬编码 profile 名** — `mcp/tools.py:43-44`，改配置后恒报 0。修法：按 `default_for_text/multimodal` 推导。
9. **过期 shadow + 全量扫描 = 批量误标 deleted** — `runtime.py:68-88` + `zotero/scanner.py:76-85` 无护栏。修法：扫描到的数量远低于既有量时拒绝 mark_absent 并告警。

### 入库流水线（花钱/数据安全相关）

10. **未拆分 PDF 续跑超时仅 36 秒 → MinerU 重复扣费** — `pipeline/ingest.py:723-726` 未传 `selected_page_count` → `extractors/cache.py:51-61` 算出 36s → resume 时误判失败并重提整个 PDF。修法：传真实页数或保守默认 500 页档。
11. **分段提交失败时误收历史 extract job** — `pipeline/ingest.py:754-771` 异常分支用了全量 `list_extract_jobs` 而非本次循环的 `segment_job_ids`，resume 会把旧产物混入 merged 文档（内容重复）。
12. **pause/cancel 对执行中 job 无效 + 并发守卫可绕过** — `pipeline/ingest.py:171-240` 执行循环从不回读 job 状态；pause 后新 job 可通过守卫，两个执行体并发写向量库。修法：每篇文档前查状态 + heartbeat/stale 检测。
13. **分类器 `dual.pdf` 子串误伤** — `zotero/classifier.py:13-20`：`residual.pdf`/`individual.pdf` 等误判为翻译件进 needs_review。修法：词边界匹配。
14. **其他中优先级**：`failed_manual_review` 被静默重提（`extractors/manager.py:111-141`）；recovery 的 `normalize`/`extract_zip` 动作无人执行（`recovery.py:62-64`）；远程图片 URL 导致整篇 normalize 失败（`normalize/markdown.py:174-233`）；LanceDB `publish_version` 非原子（`index/lancedb_vector.py:140-149`）；`upsert` 二次迭代 Iterable（`:68-79`）；sanitize consumer 校验不可达（`search/results.py:55`）；API `/providers/status` env 路径不锚定项目根（`api/app.py:207`）。

---

## 二、可新增的检索方式（按数据兼容性排列）

### A 类：复用现有数据库，纯查询侧实现

| # | 功能 | 实现位置 | 工作量 | 说明 |
|---|------|----------|--------|------|
| R1 | **Hybrid 融合检索（RRF）** | 新增 `search/hybrid.py` + CLI/API/MCP 各加一端点 | 中 | 向量 + FTS BM25 各取 top-N 按 rank 融合，无需分数归一化。**性价比最高的检索质量提升**。注意与 `RETRIEVAL_ENHANCEMENT_PLAN.md` §4.10"不做自动混合"的旧决策冲突，需先确认策略转向（skill 文档目前要求人工交叉验证正是因为没有它） |
| R2 | **分页 offset / min_score** | `db/state.py` 加 OFFSET + 三端透传 | 小 | 顺带缓解 `_METADATA_SCAN_CAP=5000` 漏召（FOLLOWUPS 重要项 1） |
| R3 | **FTS snippet/highlight 命中片段** | `db/state.py:2353` 等 SELECT 加 `snippet(chunks_fts,...)` | 小-中 | FTS5 内置、查询时计算，不改表；大幅省 LLM 消费 token |
| R4 | **文档级去重 / MMR** | `embeddings/indexer.py:search_vector_index` 后处理 | 小-中 | 解决同一文献多 chunk 霸榜 |
| R5 | **author/tag 过滤下推 SQL** | `db/state.py:213` `_passes_python_filters` → `json_each` | 中 | FOLLOWUPS 重要项 1，修复大库漏召 |
| R6 | **结构化过滤下推向量检索**（collection/日期/作者过滤向量结果） | 检索侧组合：向量召回后按 `metadata_search` 过滤器过滤 | 小-中 | 不动向量库也能做（后置过滤）；前置下推属 B 类（见 R12） |
| R7 | **查询向量缓存 + 结果缓存** | `indexer.py:345` embed 调用前；`data/embedding_cache/` 已登记可复用 | 小 | 省 DashScope 调用费与延迟 |
| R8 | **Rerank（cross-encoder / qwen3-rerank）** | 接口已全线预留（`rerank` 参数、`RerankNotSupportedError`、`rerank_score` 字段），实现 `RerankProvider` 即可 | 中-大 | 纯查询时精排，复用现有索引；依赖选型是大头 |
| R9 | **查询扩展 / 同义词表 / HyDE** | `search/query_parser.py` 后挂扩展器 | 中 | 解决 USAGE.md:525 已记录的 `GA`/`Glutaric dialdehyde` 漏召案例；同义词表是 A 类，HyDE 需 LLM 调用也是 A 类 |
| R10 | **MCP 补齐只读工具** | `mcp/server.py`：独立 fulltext 工具、ingest 进度、list_chunks | 小 | 保持 MCP 只读定位 |
| R11 | **元数据 FTS 列权重下推**（`bm25(attachments_fts, w...)`） | `db/state.py:_fts_candidate_keys` | 中 | 替代 Python 端 `metadata_match_score`，顺带修候选截断漏召（`state.py:1587-1630`） |

### B 类：在线迁移/回填（先备份，不重嵌不重抽）

| # | 功能 | 影响 | 工作量 |
|---|------|------|--------|
| R12 | **书目元数据回填进向量记录**（year/collections → `VectorRecord.metadata`） | 向量值不变，仅 UPDATE metadata JSON；回填后向量检索可做 SQL 侧预过滤（R6 的彻底版） | 中 |
| R13 | **`heading_path` 加入 `chunks_fts`** | DROP+重建 chunks_fts 表并从现有 chunks 回填（方案 §4.9 已留口）；章节标题命中是科研检索高频诉求 | 中 |
| R14 | **旧版本向量数据清理**（sqlite-local 旧 version 行、LanceDB 旧 version 表，保留最近 1-2 版回滚） | 只删非 active 历史版本，磁盘止血（当前**无界增长**，见问题 19） | 小 |
| R15 | **LanceDB 原生向量列 + ANN 索引** | 从现有 `vector_json` 读出转换写入原生列（向量值不变，不调 API）+ `create_index` | 中-大 |
| R16 | **CJK trigram FTS** | 重建 FTS 表（SQLite 3.34+ trigram tokenizer），改善中文召回 | 中 |

### C 类：需要重建/重嵌（近期不建议，列出供决策）

- **分块策略改进**：超长行硬切分、CJK token 估算加权（当前 `len//4` 对中文低估 2-4 倍，`normalize/markdown.py:646`）、parent-child 双层分块、prev/next 链按真实文档位置排序（`markdown.py:610-613` 当前把全部图片排在文本后）、页码元数据入 chunk（解析 MinerU middle.json 的 page_idx）、表格专属 chunk_type。→ 改变 chunk 划分，需重新 normalize + 重嵌。
- **多模态跨域召回**（图像查询同时召回文本 chunk，`embeddings/indexer.py:357` 当前只搜 image 模态）：若 multimodal profile 未嵌文本 chunk 则需补嵌。
- **prev/next 上下文扩展修复**：依赖分块侧链路重排。

---

## 三、性能与规模隐患（入库/索引侧）

17. **向量检索全程暴力扫 + 逐行 JSON 解析** — `index/local_vector.py:344-403`、`lancedb_vector.py:173-222`：每次查询全表读出、Python 算余弦。**A 类短期修法**：numpy 矩阵化 + 进程内向量缓存；B 类中期：R15 原生 ANN。
18. **每次检索都重开 store 连接** — `embeddings/indexer.py:364-373`：sqlite connect + FileLock + WAL pragma 每查询一次。A 类：进程内按 profile 缓存 store 实例。
19. **增量索引 O(N²) + 旧版本永不清理** — `index/local_vector.py:239`、`lancedb_vector.py:90`：每索引一篇就复制全量 active 记录到新版本，且全项目无任何旧版本删除逻辑，**存储随索引次数无界增长**。A 类修法：批量文档共享一次 publish；存量清理由 R14（B 类）做。机制详解见 `FUTURE_PLAN_20260805.md` 第三节。
20. **嵌入串行 HTTP** — `embeddings/qwen.py:157` 每 chunk 一次请求、ingest 逐文档串行。A 类：chunk 级 4-8 线程 + 文档级 worker 池，wall-clock 3-10×。
21. **MinerU key 整个 poll 期间被持有** — `extractors/manager.py:115-255`：submit 完即应释放，poll 再重新 acquire。A 类。
22. **全文 LIKE 回退无 LIMIT** — `db/state.py:2408-2416`：大库全表物化。A 类一行修复。
23. **集合展开 IN 参数可能超 SQLite 上限（32766）** — `search/collections.py:97`：分批 IN。A 类。

---

## 四、工程、测试与文档

### 测试缺口（A 类，tests/ 与 data/ 完全隔离，无风险）

- `db/state.py`（2696 行）仅 9 个用例：LIKE 回退路径**零覆盖**（match_mode=any 在回退时静默退化为 AND）、prune_*、sort_order、bm25+chunk_type 欠填充、FTS 列漂移重建（FOLLOWUPS 重要项 2）。
- `api/app.py` 约 40 端点仅 12 个用例：ingest 控制/review/backup/extract 等端点无 HTTP 层测试（含 FOLLOWUPS 次要项 4 的新检索端点）。
- CLI 近 600 行分发逻辑多未测；`models activate` 等异常路径直接抛 traceback。
- 建议新增**检索质量 golden dataset**（20-50 条标注查询，recall@k 回归阈值）——为后续 hybrid/rerank 提供量化判据。

### 工程基础设施（A 类）

- **无 CI**（无 `.github/`）：建议 GitHub Actions 最小工作流（ruff + pytest，Windows+Linux 双平台）。
- `pyproject.toml` 无 `[tool.ruff]`/`[tool.pytest.ini_options]`/coverage 配置；`tests/__init__.py:10` 的 sys.path hack 使测试从不验证打包产物。
- 脚手架遗留 `OptionalModuleTestCase`/`call_with_known_kwargs`（`tests/_support.py:19`）会静默丢 kwargs，应改直接 import。

### 文档脱节（A 类，改动小影响大）

- `README.md:5-6` 仍自称"planning repository"——完全过时，应重写。
- `docs/USAGE.md:819` 说 MCP "尚未接入"——MCP 早已实现（`mcp/server.py` + `cli.py:103`）。
- `docs/INSTALLATION_CHECKLIST.md:26` / `ENVIRONMENT_REQUIREMENTS.md:99` 说默认向量库是 sqlite-local、"暂缓安装 lancedb"——实际 lancedb 已是默认后端，按文档安装会跑不动。两份环境文档 ~90% 重复，建议合并。
- `docs/CODEBASE_AUDIT_REPORT.md:39-48` 的"未实现"清单多项已完成，建议追加状态更新一节。

### 配置/数据层小改进（A 类）

- ExtractionManager 的 poll 间隔/超时、分块参数（CHUNK_TOKEN_*）硬编码，应入 `config/settings.py`。
- 配置 default profile 与 DB default 不一致时无告警（`state.py:1017-1070`）。
- `schema_meta.schema_version` 被无条件覆写（`state.py:380-387`），无法检测"库比代码新"。
- 备份：`backup verify` CLI 入口（函数已存在）、备份 retention prune。
- 诊断扩展：FTS 行数 vs attachments 行数等一致性自检（纯 SQL 只读）。

---

## 五、建议实施批次

### 批次 1：正确性修复包（全部 A 类，零数据库风险）

第一节全部高优先级项：stub 静默（1）、rerank 错误（2）、MCP 融合改 RRF 雏形（3）、query_parser（4）、fulltext title（5）、FTS 回填 deleted（6）、回收站排除（7）、status 硬编码（8）、扫描护栏（9）、36s 超时（10）、分段 job 误收（11）、pause 协作（12）、dual.pdf（13）。配套补关键回归测试。预计 3-4 天。

### 批次 2：检索增强包（A 类为主）

R1 hybrid/RRF（需先确认推翻旧决策）、R2 分页、R3 snippet、R4 文档级去重、R5 author/tag 下推、R7 查询缓存、R10 MCP 工具补齐、R11 列权重下推 + FOLLOWUPS 其余次要项（match_mode 透传、LIKE 回退扫 creators/tags、死参数清理）。预计 3-5 天。

### 批次 3：性能与数据治理包（A + B 类，B 类执行前先备份）

17-21 性能项（numpy 化、store 缓存、批量 publish、嵌入并发、key 释放）+ R14 旧版本向量清理 + R12 书目元数据回填（如需 R6 彻底版）。预计 3-5 天。

### 批次 4：工程与文档包（全部 A 类）

CI、ruff/pytest 配置、测试缺口补齐、README/USAGE/环境文档修正、诊断扩展、备份 verify/prune CLI。预计 2-3 天，可与批次 1 并行。

C 类（分块重排、页码入 chunk、跨模态召回）建议等批次 1-3 落地、且确认愿意承担重嵌成本后单独立项。

---

## 执行约束（无论实施哪个批次）

1. 任何触及 `data/` 的操作前必须先 `zoterorag backup create` 并校验 manifest。
2. 测试一律走 `tests/_support.py:workspace_tmpdir()` 隔离工作区，禁止指向真实 `data/`。
3. 关键修改（B/C 类、接口变更、配置变更）执行前展示规划并获用户确认；每轮任务结束或 commit 前启动新只读子代理做代码审核。
