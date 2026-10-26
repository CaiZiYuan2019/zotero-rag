# ZoteroRAG 修改规划：A 类可靠性修复 + H2/H3 + 并行化改造（2026-09-04）

> 本文档是当前唯一执行依据。基于 `docs/CODEBASE_AUDIT_REPORT_20260813.md`（下称"审计报告"），并经 2026-09-04 逐项对照当前代码核实修正——**核实中发现审计报告多处描述需要更正，本文档以修正后结论为准**。
> 范围（用户拍板）：并行化改造 + A 类 9 项 + H2 + H3。B 类其余（M8/M1）与 C 类处理结论见第 5 节。

---

## 0. 决策记录（2026-09-04 用户拍板）

| # | 决策 |
|---|------|
| 1 | extract 任务总寿命上限 `job_max_lifetime_seconds = 21600`（6h，自 submitted_at 起算），超后转 `failed_manual_review` 人工裁决，不自动重提。写入 `config.toml [extract]` |
| 2 | 远端 batch 真失败的最大自动重提次数 `max_resubmit_attempts = 3`。写入 `config.toml [extract]` |
| 3 | 上述两参数的落地与后续任何调整，**必须同步更新 `docs/USAGE.md`** |
| 4 | M10 修复采用**文件锁方案**识别进程死活（见 1.8），与账本原子建 job 互补 |
| 5 | M8 降级为顺手修复（审计报告"唯一数据丢失路径/M8 优先"的定性经核实不成立）；M1 降级为观察项（报告原修复方案会误伤同 PDF 去重设计） |
| 6 | H3 已造成存量数据损坏（QZ4UCADS 乱序入库），含一项**存量清理操作**（见第 3 节） |

---

## 1. A 类修复（Phase 0，9 项）：问题、触发条件、修改方式

公共背景：sqlite-local 后端在写路径上有三层保护（staged 去重 / publish 单事务 / FileLock），LanceDB 后端三层全缺；A 类多为"串行下仅崩溃窗口触发、并行化后变常态竞态"的缺陷，是并行改造的前置必修。

### 1.1 H1 —— LanceDB staged 版本追加无去重

- **问题**：`index/lancedb_vector.py:91-139` 的 `copy_active_records_to_version` 对已存在的 staged 表只做打开复用 + 无条件 `table.add`（137 行），无 `upsert` 的 delete-then-add 幂等。
- **触发**：重嵌文档 D（batch_hash 确定性生成，重试版本名必然相同），进程在 `table.add` 之后、`publish_version` 之前（`embeddings/indexer.py:203-208` 窗口）崩溃；重试时全库 carry-forward 记录被再追加一遍，每重试一次翻一倍 → 计数虚高、同 chunk 重复占 top_k。
- **现状库**：实测无重复行（text 6345 / multimodal 9375，distinct 相同，无孤儿表），属隐患非事实。
- **修改**：`copy_active_records_to_version` 开头（active 解析之后、提前 return 之前）：staged 表已存在且 `version_is_active()` 为假 → `_drop_version_table` 后重建。守卫必须在表名空间比较。
- **验证**：崩溃重入无重复行（新增测试）。

### 1.2 M5 —— publish 两步非原子（含 L8 顺带修复）

- **问题**：`lancedb_vector.py:141-161` 对 meta 表先 `delete`（153）再 `add`（154），无事务。
- **触发**：delete/add 间隙崩溃 → meta 零行 → `active_version()` 回退 `"legacy"`（214-230）→ `search()` 打开已删的 `vectors_legacy` → **静默返回空**。二级后果：重嵌时 copy 复制 0 条 → publish 后全库其他文档向量被永久固化丢失。
- **修改**：delete+add 换成单条 `meta_table.merge_insert("profile_name").when_matched_update_all().when_not_matched_insert_all().execute(...)`（环境已确认 lancedb 0.33.0 支持）。**顺带 L8**：移除/显式化 `"legacy"` 回退，meta 缺失时报错并提示 `vectors verify`。
- **验证**：merge_insert 无 meta 空窗；meta 缺失时显式报错。

### 1.3 M6 —— reembed 无守卫、LanceDB 无跨进程锁

- **问题**：`pipeline/reembed.py:57-119` 完全没有 running 检查；ingest 守卫只查 `kind="ingest"`（`ingest.py:114`），双向互斥缺失。LanceDB 写路径零锁（全文件仅连接缓存锁）。
- **触发**：ingest 与 reembed 同时运行（两个 CLI 进程，或 API 的 `POST /ingest/start` 与 `POST /reembed/from-normalized` 线程池并发）：双方基于同一 active 快照各自 staged、各自 publish，后者 `_drop_version_table(previous)` 物理删除前者的版本表 → 一方文档集静默丢失（账本已记 indexed）。
- **核实补充（比报告严重）**：① ingest 守卫自身有贯穿 plan 阶段的宽 TOCTOU 窗口（job 以 `created` 插入，161 行才置 running）；② sqlite-local 的 FileLock 是逐方法持有，copy→upsert→publish 序列粒度上同样丢更新。
- **修改**：① reembed 加守卫且 kind 互查（与 1.8 的原子方法合并落地）；② LanceDB 加 per-profile `FileLock`，**临界区为 indexer.py:203-208 整个 copy→upsert→publish 序列**（建议抽持锁复合方法或暴露可重入写锁）；sqlite-local 一并改为序列级。
- **验证**：双写者并发发布不丢文档集（双后端）。

### 1.4 M4 —— 嵌入批次卡 running 阻塞 prune

- **问题**：`indexer.py:159-173` 先落 `status="running"` 再嵌入，至 `:213` 置 completed 之间无异常翻转；`prune_vector_index_versions`（`:329-334`）见任何 running 批次即 ValueError 拒绝。全库无翻转孤儿批次的路径。
- **触发**：嵌入中途 Ctrl-C/崩溃/网络异常穿透 → running 行永久孤儿 → 该 profile 的 `vectors prune` 永久拒绝，孤儿版本表无法清理。注意：**文档本身可自愈**（跳过逻辑只对 completed 生效，重试会重嵌入库），危害是 prune 阻塞；且 running 残留使重试必走全流程，成为 H1 的导火索。
- **修改**：`StateLedger` 新增原子方法 `mark_stale_embedding_batches_failed(profile_name, older_than)`（单条 UPDATE）；`prune_vector_index_versions` 前置调用；新配置 `vectors.stale_embedding_batch_hours = 24`。并行化时需把"翻转+检查+删除"纳入同一临界区。
- **验证**：stale running 批次被 prune 自动回收；新鲜 running 仍中止 prune。

### 1.5 M2 —— poll 超时被当硬失败（超时语义重构）

- **问题**：`extractors/manager.py` 两套预算错位：per-job `pages*6+30`（`cache.py:56-67`）首跑不被读取、续跑被 `max(per_job, 21600)` 恒覆盖，实为 6h 一刀切且**每轮续跑重新起算、无任务总寿命概念**；超时抛普通 `MinerUAPIError`，被 `_resume_poll_with_fallback`（405 行）当远端失败 → 重提交覆盖 external_job_id、遗弃在跑远端 batch（双倍扣费）；两处 catch-all 对超时也冷却 key 60s。
- **触发**：远端 batch 超过本地预算（大 PDF + 高峰期）；并行化后批量触发、key 池被集体误冷却抽空。
- **修改**：新增 `PollTimeoutError`；两处超时抛出点（`manager.py:170`、`363`）改抛它；两处 catch-all 特判（只记 `error_code="poll-timeout"`，不落 failed_retryable、不冷却 key、保留 external_job_id）；fallback 前置 `except PollTimeoutError: raise`。超时语义变为"本轮摘牌、下轮续 poll 原 batch"。per-job 预算改为真正生效（下限 30min、上限 6h），新增总寿命上限（决策 1）。
- **验证**：假 provider 永不完成 + 短预算 → job 保持 running、续跑 poll 原 external id、不重提、key 不冷却。

### 1.6 M3 —— empty-result 状态污染（报告"死结"描述已修正）

- **核实修正**：报告所称"manual_review 死结"**不成立**——`recovery.py:72-74` 的 `or local_stage == "poll"` 守卫会把 empty-result 正确归为 poll 续跑。真实问题是状态污染：`mineru.py:377-379` 把"结果列表暂时为空"（排队中）持久化为伪状态 `empty-result` → 不在 `REUSABLE_EXTRACT_STATES`（`manager.py:24`）→ 崩溃后重跑 ingest 时缓存复用失效、**重新提交**远端 batch（丢弃在跑 batch + 双倍扣费）；并行化后 N 个 job 的崩溃窗口残留成倍放大。
- **触发**：poll 循环某轮返回空 extract_result 并落库后，进程在下一轮前被硬杀（Ctrl-C/断电，KeyboardInterrupt 不被 catch-all 捕获）。
- **修改**：`_poll_snapshot` 空结果直接返回 `state="running"`（核心一行）；`recovery.py:72` state 集合可选加 `"empty-result"` 做防御兜底。
- **验证**：空 extract_result → running；recovery 归 poll。

### 1.7 M7 —— ingest resume 只重建计划、永不执行

- **问题**：`ingest.py:274-300` 重建 plan 后置 `planned` 即返回；`cli.py:543-549` 无 `--execute`；全库无消费者把 planned 转为执行。resume 本身无并发守卫，可无差别覆写所有文档的 `ingest_plan` checkpoint（292-298），是守卫的绕过口。
- **触发**：崩溃 → job 残留 running → `ingest start` 被守卫拒绝 → 用户执行 `ingest resume` → 打印成功但什么都没跑，且崩溃痕迹被抹掉。
- **与 M10 的关系**：M10 回答"谁在跑、谁死了"（互斥+死活识别），M7 回答"死了之后怎么接着跑"（接管续跑），是崩溃恢复链的前后两环。
- **修改**：`start_ingest_job` 执行体（160-267）抽为共享函数；`resume_ingest_job` 加 `execute` 参数（execute 时置 running 并复用执行体）；CLI 加 `resume --execute`。守卫放行"恢复自己"（排除当前 job_id 再检查）。
- **验证**：`resume --execute` 完成续跑。

### 1.8 M10 —— ingest 守卫 TOCTOU + 文件锁死活识别（决策 4）

- **问题**：`ingest.py:114-119` 先查后建三步独立事务（`created` → 161 行才 `running`），plan 阶段是秒级宽窗口；`pipeline_jobs` 对 (kind, status) 无唯一约束。反向：job 死在 running 则守卫永久拒绝新 ingest（无租约的软锁）。
- **修改（双层）**：
  - 账基层：`StateLedger` 新增 `create_job_if_no_active(kind, payload, active_kinds=("ingest","reembed"))`，单事务内"查 active → 插 job"。**实现要点**：连接是 `isolation_level=None` 自动提交，方法内必须显式 `BEGIN IMMEDIATE` 才构成跨进程写事务。ingest/reembed 共用。
  - 文件锁层（识别进程死活）：ingest/reembed 共用 `data/state/ingest_runner.lock`（filelock，项目已依赖）。启动流程：非阻塞拿锁 → 拿不到=有活进程 → 拒绝；拿到后查账本 running job → 有则必为死进程残留 → 自动标 `interrupted` + 记事件 → 继续。锁持有全程，finally 释放。**CLI 与 API server 两条入口必须走同一拿锁路径**。
- **验证**：双进程同时 start 仅一方成功；kill 残留 running 后重启自动识别并放行。

### 1.9 M9 —— blocked 状态链路缺失（5 个断点）+ catch-all 无熔断

- **问题**：blocked 链路断在 5 个点。产生点 3 处（写 checkpoint blocked 后静默 return，调用方不知情）：`ingest.py:705-713`（续跑分段）、`:781-789`（新拆分分段）、`:896-922`（单 PDF，`_checkpoint_single_extract_result`，报告漏了这处）；传导点 2 处：`_execute_document_stages`（`:589-643`）不区分 stage 结果、blocked 后继续 normalize/embed 并无条件返回成功；主循环（`:176-234`）无 blocked 桶，终态（`:223`）只有 completed/completed_with_errors 两态。
- **次生灾害**：blocked 后 normalize（`:1012-1033`）捡同附件任一已下载分段残片当全文归一化（静默内容截断），或 `:1018` 抛误导性 ValueError 错标硬失败。catch-all 无熔断：API key 被吊销时 100% 文档失败，终态仍是 completed_with_errors。
- **关键联动**：**M2 修复后 poll 超时不再抛异常，分段未齐将从罕见路径变为常态出口**，blocked 链路是 M2 正确工作的必要配套。
- **修改**：3 个产生点返回 blocked；`_execute_document_stages` 传播并短路 normalize/embed；主循环 blocked 单独计数；终态四态（completed / completed_with_errors / blocked / failed）；catch-all 熔断（连续 N 篇同类异常 → 中止并 `set_job_status(job_id, "failed")`）。
- **验证**：分段未齐文档计入 blocked 且不被归一化；系统性失败触发熔断。
- **遗留跟进（P1，超出本轮范围）**：分段全部 downloaded 后、`_merge_and_checkpoint_segments` 完成前崩溃 → extract checkpoint 停在 "running"，下轮 plan 期 `_all_segment_jobs_done` 判 done → 执行层跳过 extract → normalize 命中 M9 守卫抛 ValueError，增量模式每轮皆然（仅 `mode="full"` 可恢复）。后续应让 `stage_for_extract` 在"分段齐但 checkpoint 非 done 或 merged_full.md 缺失"时返回 pending 以补 merge（与 H2 的 force 重合并同属合并链路加固）。

---

## 2. 并行化改造（Phase 1–3）

### 2.1 Phase 1 嵌入并发

- `embeddings/qwen.py`：`embed()` 内 `ThreadPoolExecutor(max_workers=embed_workers)` + token bucket 限速；**`embed_workers=1` 走原串行路径，零行为变化**。线程池仅存于单次 `embed()` 调用内。
- `config/settings.py`：`EmbeddingProfile` 加 `embed_workers`；`rate_limit` 从死配置变生效；`providers.py:118` 透传。
- 验证：假 client 计数并发在途 >1、输出顺序==输入顺序、workers=1 行为不变。
- **设计定稿（计划审核后补充）**：①并发路径用手动 futures + 首个异常 `shutdown(wait=False, cancel_futures=True)` 快速失败（pool.map 会在故障时烧完全部输入的重试，不可接受），保序由按提交序取 result 保证；②token bucket 按 **HTTP 尝试**计费（acquire 在 `_post_with_retry` 每次 post 前，重试也计入），串行/并发路径同样生效（默认 rps=0 不限速、零行为变化）；③容量 = `max(1, min(rps, embed_workers))`，开局突发不超过 worker 数；④限速作用域 = 单 provider 实例（单文档单 profile 一轮），**非全局**——并发 job 合计速率为各实例之和，38 rps 已按此留余量。
- **✅ 已落地（2026-09-05，commit 13，a96a586）**：`embed()` 内 ThreadPoolExecutor + token bucket 按 HTTP 尝试计费，embed_workers=1 与原串行路径零行为变化；`EmbeddingProfile.embed_workers` 与 `rate_limit` 生效并经 providers.py 透传。

### 2.2 Phase 2 抽取并发（submit 全量 → 并发 harvest）

- `extractors/manager.py` 拆三原语：`submit_only()`（取 key → submit → 落账 → **立即释放**）、`harvest(job_id)`（循环"取 key → 单次 poll → 落账 → 释放 → 无 key sleep"）、`poll_once()` 三处复用。`ensure_extraction()` 保持薄包装。
- `pipeline/ingest.py`：两趟（submit 全量含分段 → `ThreadPoolExecutor(harvest_workers)` 分区收割），完成后沿用现有合并/blocked 逻辑（依赖 1.9）。
- `cli.py:516`：向 `ExtractorKeyPool.from_env_file` 传 `per_key_submit_concurrency`。

**等待/超时/排队语义（M2 + 并行化后的定稿）**：

- key 只在单次 HTTP 调用期间持有；M 个 key 服务 N 个在途 job（N≫M），harvester 分区认领、无 key 状态 sleep。
- 超时=本轮摘牌：`error_code="poll-timeout"`，job 保持 running + external_job_id，不冷却、不重提；下轮续 poll。总寿命 6h（决策 1）超后转 `failed_manual_review`。
- 限速：`submit_min_interval_seconds=1.5`（≤40/min，配额 50）、`poll_max_per_minute=600`（配额 1000 留 40% 余量）、`poll_interval_seconds=30`。

**错误分类与 key 冷却**（替代现状 60s 一刀切）：

| 事件 | job 侧 | key 侧 |
|------|--------|--------|
| 429/配额 | 保持 running，下轮再试 | 冷却数分钟（可配） |
| 网络抖动/5xx | 保持 running，指数退避 | 短冷却或不冷却 |
| 401/403 | 保持 running | 长冷却/禁用；全部 key 被禁 → 本轮中止 + job failed（接 M9 熔断） |
| 本轮看守超时 | poll-timeout，保持 running | 不冷却 |
| 远端 batch 真失败 | failed_retryable，**重提 ≤3 次**（决策 2，计数落 payload） | 不冷却 |

- 验证：记录型假 provider 断言"全部 submit 先于任何 poll"、分段先行提交、key 池账目正确、429/超时分类行为正确。
- **✅ 已落地（2026-09-05，commit 14）**：三原语 `submit_only`/`poll_once`/`harvest` 拆分完成，ensure/resume 为薄包装且对外语义等价（416 测试全绿）；错误分级表五项全部落地（429→冷却 180s+摘牌 rate-limited；401/403→冷却 3600s，全池失效→failed_retryable/"no-keys-available"+raise；408/5xx/网络异常→不冷却+harvest 内指数退避；poll-timeout→摘牌不冷却；远端真失败→failed_retryable+不冷却）。决策2 计数落 payload 且跨轮结转（submit_only 发起时计数、_resume_submit 成功后计数、两入口均受帽）。限速器（submit 最小间隔锁、poll 滑动窗口）为 manager 实例级，构造默认 0/不限（存量测试零行为变化），生产值由 [extract] wiring 注入。key 亲和：`acquire_key(prefer_alias=...)` 优先提交 key（payload["key_alias"] 跟踪最新提交者），取不到退化任意可用 key——**同账号假设**：多把 MINERU_KEY 须属同一 MinerU 账号可互查 batch，否则跨 key poll/download 会失败（经分级→重提帽兜底，有界）。ingest 拆分两趟（submit 全量 → ThreadPoolExecutor(harvest_workers) + `_harvest_one` 路由：DONE 直通/local_stage=="poll"→harvest 确认/其余→resume classify）。
- **附带修复（高危）**：sqlite 共享连接跨线程竞态——pysqlite 未步进的 SELECT 会被另一线程的语句重置导致瞬时读空（实证：无锁 400 读丢 11 次）；已将 `get_extract_job`/`get_extract_job_by_cache_key`/`upsert_extract_job` 的 execute+fetchone 全程纳入 ledger RLock（`set_extract_job_state` 整个读改写体本就在 `with self.conn:` 持锁内）。
- **遗留跟进（P2，超出本轮范围，commit 14 审核发现）**：① state.py 其余读方法（list_extract_jobs/checkpoint 等）的 execute→fetchone 仍非原子，API 线程与 ingest 并发时可能瞬时读空（刷新即恢复，预存在缺陷，本轮未扩散范围）；② 决策2 计数时机不对称：submit_only 发起即计数（submit 失败也耗额度），_resume_submit 成功才计——各自有界，如需严格对称再统一；③ harvest 摘牌最多 overshoot 一个 poll_interval（tail sleep 先于下轮 deadline 检查，有界 30s）；④ pool.map 首个异常传播需等其余分段 harvest 收尾（shutdown wait），兄弟分段成果保留、checkpoint 正确，属两趟设计固有。

### 2.3 Phase 3 批量发布（消灭 O(N²)）

- `embeddings/indexer.py` 拆 `stage_indexed_document()`（纯准备，不开库）与 `publish_staged_documents()`（run hash 命名 → 开库一次 → drop 残留 staged → 一次 copy（exclude 本轮全部文档）→ 逐篇 upsert → 一次 publish → 逐篇落账）。`index_normalized_document()` 保留为单篇包装；`pipeline/reembed.py` 同改。
- 版本命名从 per-doc batch_hash 变 per-run hash；无 schema/数据迁移；per-doc 语义（embedding_batches/checkpoints/跳过逻辑）全部保留；回滚粒度变为单轮（无实际功能损失）。
- 验证：3 篇 staged 一次发布、copy 量==发布前 active 量、中断重跑零重复行（双后端）、reembed 等价；真实库观察 `data/vector_store/` 体积不再 N² 增长。
- **✅ 已落地（2026-09-05，commit 15）**：`stage_indexed_document()`/`publish_staged_documents()`/`StagedDocument` 拆分完成，`index_normalized_document()` 为薄包装（reuse 快路逐字保留，cli/api/mcp 等 4 个既有调用点零改动）；run_hash=sha256(sorted(组内 batch_hashes)) 确定性命名（同批次崩溃重跑同名，残留非 active 版本自愈清扫）；发布=开库一次→清扫全部非 active 残留（delete_version 自带 active 拒删守卫，清扫在 staging 前故无需豁免 run_hash）→一次 `stage_and_publish`（合并 records、exclude 本轮全部文档，等效"逐篇 upsert"且保持 M6 单锁序列）→逐篇落账（batch completed payload 与 checkpoint 均新增 `index_version=run_hash`，carried_forward_chunks 为 run 级 copy 数）。ingest：`batch_publish`（默认 true）下 embed 段只 stage 收集，文档循环结束后（含熔断 break 路径）按 profile 分组逐组发布；**awaiting_publish 记账**杜绝双计数——staged 文档发布成功才进 executed，任一 profile 发布失败整篇记 failed（对齐现状 per-doc 失败语义），部分 profile 失败时成功 profile 的 checkpoint 已落 indexed、下轮只重试失败 profile。reembed 同改（stage 收集+循环后一次 publish）。配置：`[ingest] batch_publish`（IngestConfig + _parse_bool 校验 + cli/app 三处 wiring + config.toml/example + USAGE 同步）。diagnostics `vector_index_provenance` 同时接受 batch_hash（兼容旧数据）与 payload["index_version"]（run_hash 命名）匹配 active_version。测试 430 passed 1 skipped（新增 14 个：3 篇一次发布/中断重跑零重复行/reuse 不经 publish/残留清扫/单篇包装等价/lancedb 编排/reembed 双库内容逐行等价/batch_publish=False 回退/部分 profile 发布失败/配置解析/provenance run_hash 匹配）。
- **与计划字面的两处偏离（commit 15 审核记录，语义更精确）**：① "逐篇 upsert"实现为一次合并 upsert（records 独立、按 record_id 幂等，效果等价且保持单锁序列）；② "全部组发布失败→failed"实现为附带 `not executed` 豁免（有文档经 reuse 快路真正完成时不报 failed，completed_with_errors 更准确）。
- **遗留跟进（P2，commit 15 审核发现）**：① awaiting_publish 归并为 O(文档数×collector 条目) 双重扫描，万级规模才有一次性开销，当前可接受；② 中断重跑零重复行的真实崩溃注入测试仅覆盖 sqlite-local，lancedb 侧由 FakeStore 编排测试+既有原语幂等测试组合覆盖；③ 批量发布后崩溃会使整轮 batch 行留 "running"（现状单篇），prune 的 running 守卫阻塞窗口变宽，由 24h stale 回收兜底（state.py:1346-1379）。

### 2.4 配置参数档（写入 config.toml；标注★的两项调整必须同步 docs/USAGE.md）

| 配置项 | 位置 | 默认值 |
|--------|------|--------|
| `embed_workers` | 各 `[[embedding_profiles]]` | 16 |
| `rate_limit.requests_per_second` | 各 `[[embedding_profiles]]` | 38 |
| `per_key_submit_concurrency` | `[extract]` | 1 |
| `submit_min_interval_seconds` | `[extract]` | 1.5 |
| `harvest_workers` | `[extract]` | 2 |
| `poll_max_per_minute` | `[extract]` | 600 |
| `poll_interval_seconds` | `[extract]` | 30 |
| ★ `job_max_lifetime_seconds` | `[extract]` | 21600 |
| ★ `max_resubmit_attempts` | `[extract]` | 3 |
| `batch_publish` | `[ingest]` | true |
| `stale_embedding_batch_hours` | `[vectors]` | 24 |

CLI 仅新增 `ingest resume --execute`。所有默认值下存量串行行为完全保留，各阶段可独立上线。

---

## 3. H2 / H3 修复与 QZ4UCADS 存量清理

### 3.1 H3 —— 分段回退 checkpoint 乱序（已真实发生）

- **问题**：回退分支（`ingest.py:761-778`）用 `list_extract_jobs` 捞出该附件**全部** extract job（含拆分前历史整篇 job），按 `updated_at DESC`（`state.py:1994`）记录；续跑/合并按列表顺序直接拼接（`merge_extract_segments.py:90-121` 保序不排序）。
- **存量实证**：QZ4UCADS（600 页，3 段）第 3 段 resume 后 updated_at 最新 → checkpoint 逆序 `[401-600, 201-400, 1-200]` → `merged_full.md` 开头与 401-600 段逐字节一致 → 1015 个乱序 chunk 已入库服务检索。
- **修改**（计划审核后修订：原方案"直接用外层 append 的 `segment_job_ids`"会漏掉失败段——append 在 ensure_extraction 返回后才执行）：回退分支改为对每个 planned segment 重算 cache_key 并用 `get_extract_job_by_cache_key` 精确回查（cache_key 推导抽成共享函数 `extraction_cache_context`，与 ensure_extraction 同源防漂移），天然排除历史整篇 job 且覆盖"段 1 即失败"边界；`_checkpoint_segment_blocked` / `_merge_and_checkpoint_segments` 统一按 `selected_pages` 起始页排序防御，ids 与 states 从同一排序列表派生。
- **存量清理（代码修复不自愈，必须手工执行一次）**：删除 `data/extract_cache/merged/QZ4UCADS/` + 该附件的 extract/normalize checkpoint，重跑该篇。分段 artifact 仍在 extract_cache 中，重合并为纯本地操作，**不会重新提交 MinerU**；重嵌只影响该文档。
- **✅ 已执行（2026-09-05，报告：`report/20260905/QZ4UCADS_乱序清理报告.md`）**：实际路径因 M9/H2 已落地而调整——用 `_merge_and_checkpoint_segments(force=True)` 重合并修复 + 仅删 normalize/embed 侧状态 + 增量重跑（按计划字面删 extract checkpoint 反而会误判 done/捡分段残片）。chunks 表不手工删（交由 upsert 原子替换，避免 chunks_fts 残留）。验证全过：页序修复、1015 chunks 重建、双 profile 重嵌完成、vectors verify 0 errors。遗留：PDF 现 606 页（原按 600 页抽取），第 601-606 页未抽取，未来 `mode=full` 会以 401-606 重提 MinerU 段 3。
- **遗留跟进（P1，超出本轮范围，commit 10 审核发现）**：resume 路径只信任 checkpoint 的 `segment_job_ids` 子集，不会按 `compute_page_segments` 重算全集补齐缺失段——若失败发生在后段持久化之前，resume 会把残缺段集合并成 done（静默丢页，需 full rebuild 才能补齐）。治本方案：resume 时按重算全集与持久化 job 求并集补齐。另：旧 checkpoint 的 ids 若已混入整篇 job（`selected_pages=""`），排序后它排最前、合并结果重复且标记 done，无守卫检测——已知实例仅 QZ4UCADS，由存量清理覆盖。

### 3.2 H2 —— 合并 markdown 非原子写 + exists-skip

- **问题**：`merge_extract_segments.py:125` `write_text` 直写无原子保护；`_merge_and_checkpoint_segments`（`ingest.py`）文件存在即跳过重合并，且连 `--mode full` 重建也修不了截断文件（guard 不感知 force，白跑 MinerU 流程）。
- **触发**：>200 页 PDF 合并时进程恰好命中最终文本写入的毫秒级窗口被杀 → 截断文件被永久归一化。
- **修改**：写 `merged_full.md.tmp` + 同目录 `os.replace` 原子改名；`_merge_and_checkpoint_segments` 加 force 支持（full_rebuild 时无条件重合并，本地操作秒级幂等）。
- **验证**：`tests/test_merge_extract_segments.py` 补原子性断言；full 模式可修复截断文件。

---

## 4. 实施顺序与依赖

```
批次1 Phase0-A（向量库写路径）：H1 → M5(+L8) → M6
批次2 Phase0-B（job 生命周期）：M10（原子建job+文件锁）→ M7（resume --execute，依赖 M10 的死活识别）
批次3 Phase0-C（抽取状态机）：M2 → M3 → M9（M9 依赖 M2 的"超时不抛异常"语义）
批次4 Phase0-D（prune 回收）：M4
批次5 H3 修复 + QZ4UCADS 存量清理 → H2
批次6 Phase 1 嵌入并发
批次7 Phase 2 抽取并发（依赖 M2/M3/M9 + 批次2）
批次8 Phase 3 批量发布（依赖 H1/M5/M4）
```

每批次验证：`python -m pytest tests/ -q`（基线 228 passed）+ 该批新增针对性测试；落地后 `zoterorag vectors verify --profile ...`、`zoterorag doctor --verify-vectors`。

---

## 5. 暂不实施项（结论已更新）

| 项 | 结论 | 理由（2026-09-04 核实） |
|----|------|--------------------------|
| M8 空扫描清账 | 降为顺手修复，不单独占批次 | 报告触发路径不成立（shadow 失败会抛异常中止；Zotero 未运行反而安全）；唯一真实触发是配置指向空 Zotero 库；后果为可自动恢复的软删除。一行守卫（空 keys 拒绝执行）可夹在任何批次空隙 |
| M1 身份劫持 | 降为 C 类观察项 | 当前库暴露面为零；"归错文档"不可能（cache_key 相同⟺内容相同）；报告原方案（冲突报错/改唯一约束）会摧毁有意的同 PDF 去重。若做：仅从 ON CONFLICT SET 清单移除 attachment_key 与 payload.input_file 覆盖 |
| C 类 L1–L9 | 暂缓 | L8 已并入批次1（随 M5 改）；L1 留待下次迁移数据目录前处理；其余观察 |

---

## 6. 本地 commit 计划（仅本地，不 push）

每个 commit 独立可测、独立验证（pytest 全绿 + 针对性测试），按批次顺序提交：

| # | commit | 内容 | 主要文件 |
|---|--------|------|----------|
| 1 | `fix(index): dedupe staged LanceDB version rebuild (H1)` | staged 表重入 drop 重建 | `index/lancedb_vector.py` + 测试 |
| 2 | `fix(index): atomic publish via merge_insert, explicit meta-missing error (M5+L8)` | 原子翻牌 + 移除 legacy 静默回退 | `index/lancedb_vector.py` + 测试 |
| 3 | `fix(index): cross-process write lock for vector stores + reembed guard (M6)` | per-profile FileLock（序列级）、reembed/ingest kind 互查 | `index/lancedb_vector.py`、`index/local_vector.py`、`pipeline/reembed.py` + 测试 |
| 4 | `fix(pipeline): atomic job creation and stale-run detection via file lock (M10)` | `create_job_if_no_active`（BEGIN IMMEDIATE）+ `ingest_runner.lock` 死活识别 | `db/state.py`、`pipeline/ingest.py`、`pipeline/reembed.py` + 测试 |
| 5 | `feat(cli): ingest resume --execute (M7)` | 执行体抽共享函数、resume 真执行 | `pipeline/ingest.py`、`cli.py` + 测试 |
| 6 | `fix(extract): distinguish poll timeout from failure (M2)` | PollTimeoutError、不冷却不重提、per-job 预算生效、★`job_max_lifetime_seconds` 入 config + USAGE | `extractors/manager.py`、`extractors/mineru.py`、`config/settings.py`、`config.toml`、`cli.py`、`api/app.py` + 测试 |
| 7 | `fix(extract): map empty poll snapshot to running (M3)` | 消除 empty-result 伪状态 | `extractors/mineru.py`、`extractors/recovery.py` + 测试 |
| 8 | `fix(pipeline): blocked document accounting and systemic-failure breaker (M9)` | blocked 链路 5 断点 + 四态终态 + 熔断 | `pipeline/ingest.py` + 测试 |
| 9 | `fix(vectors): recycle stale running embedding batches before prune (M4)` | stale 翻转 + `stale_embedding_batch_hours` | `embeddings/indexer.py`、`db/state.py`、`config` + 测试 |
| 10 | `fix(pipeline): preserve segment order in fallback checkpoint (H3)` | 按序 segment_job_ids + 合并前排序防御 | `pipeline/ingest.py`、`normalize/merge_extract_segments.py` + 测试 |
| 11 | `fix(normalize): atomic merged markdown write, force remerge (H2)` | 临时文件 + os.replace + force 重合并 | `normalize/merge_extract_segments.py`、`pipeline/ingest.py` + 测试 |
| 12 | （操作，非代码 commit）QZ4UCADS 存量清理 | 删 merged 目录 + checkpoint，重跑该篇；结果记入当日 report/ | 数据操作 |
| 13 | `feat(embed): concurrent embedding with rate limit (Phase 1)` | embed_workers + token bucket；workers=1 零行为变化 | `embeddings/qwen.py`、`config/settings.py`、`providers.py` + 测试 |
| 14 | `feat(extract): submit-all then concurrent harvest (Phase 2)` | 三原语拆分 + harvest 线程池 + 限速/冷却分级 + ★两参数入 config.toml + **docs/USAGE.md 同步** | `extractors/manager.py`、`pipeline/ingest.py`、`cli.py`、`config`、`docs/USAGE.md` + 测试 |
| 15 | `feat(index): batch publish per run (Phase 3)` | stage/publish 拆分 + run hash 版本 + reembed 同改 | `embeddings/indexer.py`、`pipeline/reembed.py` + 测试 |

提交纪律：每 commit 前 `python -m pytest tests/ -q` 全绿；commit 4/6/8 属关键修改（改 job 生命周期与状态机），实施前需按 AGENTS.md 全局规则走只读子代理规划审核 + 用户确认；数据操作（12）前先备份（`backups/` 目录惯例）。
