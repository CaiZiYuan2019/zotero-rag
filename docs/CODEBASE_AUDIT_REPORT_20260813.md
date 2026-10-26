# ZoteroRAG 代码库审查报告（2026-08-13）

> 审查时间：2026-08-13
> 审查范围：`src/zoterorag/` 全部建库管线（extract/normalize/embed/index/state/pipeline）+ 检索消费侧 + 配置（`config/config.toml`、`.env` 装配路径）；参考 `docs/FUTURE_PLAN_20260805.md` 与 `docs/CODEBASE_AUDIT_REPORT.md` 历史结论
> 审查方法：3 个 Explore 子代理并行勘察（架构与建库流程 / 缺陷审查 / 性能与并发），1 个 Plan 子代理设计提速方案；主代理逐项核实关键代码位置与运行环境（lancedb 0.33.0 API、DashScope/MinerU 配额）
> 本轮结论：**仅产出本报告，不改动任何代码、配置与数据目录**（用户决策 2026-08-13）。提速/并发改造设计见第六节，按后续批次实施。

---

## 一、执行摘要

建库管线目前**完全串行**，三处 wall-clock 瓶颈按影响排序：

1. **嵌入（最大瓶颈）**：`qwen.py:157-158` 每 chunk 单独一次阻塞 HTTP 请求。全库 15,735 个 chunk ≈ 15,735 次请求；`config.toml` 中的 `batch_size`/`rate_limit` 是**死配置**（全库无读取点）。
2. **抽取**：MinerU 云端 batch 轮询期间管线整体阻塞（30s 间隔 sleep），API key 从 submit 到 download **全程持有**；分段 PDF 严格串行（上一段全部完成才提交下一段）。`ExtractorKeyPool` 的并发机制（`per_key_submit_concurrency`、in_flight 计数、RLock）已就绪但从未被并发使用。
3. **向量写入 O(N²)**：每索引一篇文档都全库复制一遍向量再翻牌（磁盘膨胀已在 a2e4fc8 修复，残留的是写放大，379 篇规模下每次增量约重写双 profile 约 700MB）。

同时发现 **13 项高危缺陷 + 9 项低危缺陷**。其中 9 项是并发改造的**前置必修项**（并发会直接放大这些缺陷：LanceDB staged 表重复行、非原子版本翻牌、reembed 无并发守卫等）；4 项独立高危缺陷（空扫描清空附件账本、合并文件非原子写等）不阻塞并发改造，建议列为独立批次。

已拍板的设计决策（2026-08-13）：**批量发布**（消灭 O(N²)）、**分阶段批量结构**（不做文档级流水线）、**嵌入放宽 / MinerU 保守**（配额：DashScope 2400 rpm、MinerU submit 50/min + 轮询 1000/min）、**报告先行不改代码**。

---

## 二、架构与建库管线总览

### 2.1 项目定位与入口

Python 单包 RAG 建库工具（Windows）。数据流严格单向：**Zotero 只读** → shadow 快照 → 扫描分类 → MinerU 云端 OCR → markdown 归一化/分块 → DashScope 多模态向量化 → LanceDB 版本化向量库。所有任务状态落在 SQLite StateLedger（`data/state/state.sqlite`）。

- 唯一主接口：`zoterorag` CLI（`src/zoterorag/cli.py`，入口 `main()`），JSON 输出。
- 建库命令序列（`docs/USAGE.md`）：`init-state` → `scan`（shadow 拷贝+分类落账）→ `review`（人工裁决）→ `ingest start --mode incremental --execute`。
- 无定时任务/任务队列，全部靠 CLI 手动触发，幂等可续跑。`serve`（FastAPI）与 `mcp` 已冻结不发展。

### 2.2 管线逐阶段

| 阶段 | 模块 | 关键函数 | 说明 |
|------|------|----------|------|
| 0. 影子拷贝 | `zotero/shadow.py` | `create_shadow_copy()` | sqlite3 backup API 只读备份 Zotero 主库到 `data/shadow/`，绝不写主库 |
| 0. 扫描分类 | `zotero/scanner.py`、`zotero/classifier.py` | `scan_shadow_to_ledger()`、`classify_attachment()` | 按 `attachment_fingerprint()`（`db/state.py:2522`）比对落 scan_status；分类 included_auto/needs_review/report_only/orphan |
| 1. 抽取 | `extractors/manager.py`、`extractors/mineru.py` | `ensure_extraction()` | MinerU 云端 v4 batch API：POST 取上传地址 → PUT 上传 → 轮询 `extract-results/batch/{id}` → 下载 zip 防 zip-slip 解压。**一份 PDF 一个 batch**。>200 页按 `compute_page_segments()`（`extractors/segments.py:15`）切片 |
| 2. 归一化 | `normalize/markdown.py` | `normalize_markdown_document()` | 改写图片引用、Pillow 缩放（≤1600px、≤5MB）、`build_chunks()` 产出文本块+图像块（chunk_id `{doc_id}:text:{n}` / `{doc_id}:image:{n}`） |
| 3. 嵌入 | `embeddings/indexer.py`、`embeddings/qwen.py` | `index_normalized_document()` | 双 profile（text/multimodal）各一轮：batch_hash 跳过检查 → qwen3-vl 嵌入（dim 2560）→ staged 版本 → `publish_version()` 原子翻牌 |
| 4. 检索 | `embeddings/indexer.py:388`、`index/*.py` | `search_vector_index()` | 暴力余弦 top_k；LanceDB（默认）或 sqlite-local 双后端同一接口 |

### 2.3 存储布局

- **StateLedger**（`db/state.py`，WAL）：pipeline_jobs、job_events、checkpoints(subject_id, stage)、extract_jobs（cache_key UNIQUE）、normalized_artifacts、chunks、embedding_batches（batch_hash PK）、vector_indexes（active_version 列）、attachments（书目元数据列）+ FTS5。
- **抽取缓存**：`data/extract_cache/<cache_key>/`（zip + extract/ + manifest.json），cache_key = sha256(pdf 哈希+页段+provider+版本+options+endpoint)。
- **归一化产物**：`data/normalized/<attachment_key>/`（document.md、chunks.jsonl、manifest、images/）。
- **向量库**：`data/vector_store/<profile_dir>/`，LanceDB 每版本一张 `vectors_<hash>.lance` 表 + `vector_meta.lance` 指针表；sqlite-local 回退为单表 `record_id` 带 `{version}:` 前缀。

### 2.4 增量/续跑机制（现状是扎实的）

三层指纹保证幂等：附件层 `content_fingerprint`（new/changed/unchanged）→ 抽取层 cache_key → 嵌入层 batch_hash（profile_hash+doc+chunk_type+inputs）。每篇文档三阶段各打 checkpoint，中断后可续跑；`resume_extraction()` 经 `recovery.py:51` 状态机映射为 skip/normalize/extract_zip/poll/download/submit 动作。任务互斥靠 `start_ingest_job()` 的 running-job 守卫（`ingest.py:114-119`，TOCTOU，见 M10）。

---

## 三、数据规模实测（2026-08-13）

| 指标 | 值 |
|------|-----|
| Zotero storage 内 PDF | 610 个 |
| 账本附件 | 803 条 |
| 已归一化文档 | 379 篇 |
| 抽取缓存 | 362 目录 / 5.7 GB |
| 总 chunks | 15,735 = 6,351 文本 + 9,384 图像 |
| 文本向量（LanceDB） | 6,345 行 / 378 篇 ≈ 17 chunks/篇 |
| 多模态向量 | 9,375 行 / 372 篇 ≈ 25 图/篇 |
| 向量库体积 | 双 profile 合计约 349 MB |
| 嵌入批次记录 | 787 条（≈ 2×379 + 重跑） |
| API key | 1× MinerU、1× DashScope |

全库重建 ≈ 610 次 MinerU 提交 + ~15,735 次嵌入请求 + 每篇一次全库向量复制。

---

## 四、性能瓶颈分析

### 4.1 串行主循环

- `pipeline/ingest.py:176`：`for idx, document in enumerate(plan["documents"])` —— 整库一篇一篇处理，每篇走完 extract→normalize→embed 全阶段才轮到下一篇。
- `pipeline/reembed.py:121` 同样的串行循环。
- `ingest.py:114-119` 硬性守卫禁止并发 ingest（且是 TOCTOU，见 M10）。
- 全仓库唯一的并发设施是 MCP 服务的 `ThreadPoolExecutor(max_workers=4)`（`mcp/server.py:275`，仅工具调用，不跑 ingest）。

### 4.2 嵌入：每 chunk 一次 HTTP（最大 wall-clock 瓶颈）

- `embeddings/qwen.py:157-158`：`embed()` = 列表推导逐条 `_embed_one()`，代码自注「one request per logical input」。单请求 timeout 120s，3 次指数退避重试（含 429）。
- `config/config.toml:25,39` 的 `batch_size = 32`、`rate_limit = {}` 仅存在于 `EmbeddingProfile` 定义（`config/settings.py:69-70`），**全库无任何读取点**——死配置。
- 请求阻塞在 socket I/O 时释放 GIL → 线程池可直接提速（预计 3-10×）。

### 4.3 抽取：阻塞轮询 + key 全程持有

- `extractors/manager.py:115-255`：`ensure_extraction()` 内 submit → poll（30s sleep 循环，预算 21600s）→ download 全部同步完成；key 直到 `finally`（253-255 行）才释放——**等待期间 key 空闲但被占**。
- 分段 PDF：`ingest.py:749-760` 逐段「提交→驱动到下载完成」串行，600 页 PDF = 3 轮串行 submit→poll→download。
- `ExtractorKeyPool`（`key_pool.py:124-146`）：round-robin + `per_key_submit_concurrency`（默认 1）+ in_flight + 冷却，设计上就是为并发 submit 准备的，但 CLI 构造时从未传入该参数（`cli.py:516`）——机制闲置。
- 配额换算：MinerU submit **50 文件/分钟**、轮询 **1000 次/分钟**。当前 1 个 key；约 700 个 job（含分段）按 30s 间隔轮询 = 1400 次/分钟，**并发化后必须限速**（见 6.8）。

### 4.4 向量写入 O(N²)

- `indexer.py:195-211`：每篇文档 `copy_active_records_to_version` → `upsert` → `publish_version`。第 i 篇要复制当时全库 i-1 篇的记录，总复制量 ≈ N²/2。
- LanceDB 实现（`lancedb_vector.py:113-139`）：整表 `to_arrow().to_pylist()` 读进内存再写成新表。
- 磁盘膨胀已在 a2e4fc8 修复（publish 同事务删除直接前任版本），残留的是**写放大**：379 篇规模下每次增量约重写双 profile 约 700MB。
- 批量发布修复后 O(N²)→O(N)，且不改变 per-doc 跳过/记账语义（见 6.6、6.7）。

### 4.5 线程安全现状

- StateLedger：单连接 + RLock 串行（`state.py:299-323`），WAL + busy_timeout 30s——**线程共享安全**。
- sqlite-local 后端：RLock + `filelock.FileLock`（`local_vector.py:122`）跨进程写串行，publish 单事务。
- LanceDB 后端：模块级 `_DB_CACHE`+Lock 只保护连接缓存，**publish/写路径无任何锁**（见 M5/M6）；`warmup_lancedb_connection`（`lancedb_vector.py:425-441`）规避线程池内 `lancedb.connect()` 死锁。

---

## 五、缺陷清单

分级：高 = 数据损坏/静默丢失/死结；低 = 局部正确性/健壮性问题。每项含 位置、失败场景、修复建议。**A 类** = 并发改造前置必修（对应 6.3 Phase 0）；**B 类** = 独立高危（不阻塞并发，建议独立批次）；**C 类** = 低危。

### 5.1 高危缺陷（13 项）

#### H1 [A] LanceDB staged 版本追加无去重 —— 崩溃重试产生重复行
- **位置**：`index/lancedb_vector.py:91-139`（`table.add` 于 137 行），驱动方 `embeddings/indexer.py:195-211`。
- **失败场景**：增量重嵌文档 D，staged 版本 V 已复制全库记录；进程在 `table.add` 与 `publish_version` 之间崩溃。重试时同一 V 再次 staged，全库记录被**再追加一遍** → 计数虚高、检索重复、账本与向量库永久不一致（其他文档被反复重嵌）。sqlite-local 后端因 `ON CONFLICT(record_id) DO UPDATE` 免疫此问题。
- **修复建议**：`copy_active_records_to_version` 开头检查：staged 表已存在且非 active → 先 drop 再重建（保留 `version_is_active` 守卫以兼容同 hash 重发布路径）。

#### H2 [B] 合并 markdown 非原子写 + exists-skip —— 截断文件被永久归一化
- **位置**：`normalize/merge_extract_segments.py:125`（`write_text` 直写，无临时文件+rename）；`pipeline/ingest.py:854`（`if not merged_md.is_file():` 存在即跳过重合并）。
- **失败场景**：写 merged_full.md 中途断电/杀进程 → 截断文件；下次运行见文件存在直接跳过，截断文本被归一化入库，且 extract checkpoint 已落，后续无任何路径触发重合并。
- **修复建议**：写临时文件 + `os.replace` 原子改名；guard 增加尺寸/校验（或重合并幂等覆盖）。

#### H3 [B] 分段回退 checkpoint 记录乱序 job —— 续跑合并出段落乱序
- **位置**：`pipeline/ingest.py:761-778` 回退分支 + `db/state.py:1976-1999`（`list_extract_jobs` 按 `updated_at DESC` 排序）。
- **失败场景**：某段失败触发回退，checkpoint 记录的是该附件**全部** extract job（含拆分前的历史整篇 job），且按更新时间而非分段顺序排列。续跑按此顺序合并 → 整篇文本+分段乱序拼接，静默产生错序文档。
- **修复建议**：回退分支只记录本次已提交的 segment_job_ids（按分段序号排序），过滤历史 job。

#### M1 [B] extract job 身份劫持：`ON CONFLICT(cache_key)` 静默覆盖 attachment_key
- **位置**：`db/state.py:1873-1944`（1895-1896 行 `ON CONFLICT(cache_key) DO UPDATE SET attachment_key = excluded.attachment_key`）；调用方 `extractors/manager.py:115-141`。
- **失败场景**：两个 Zotero 附件指向同一 PDF（cache_key 相同）。A 的 job 失败后，B 的提交复用 A 的 job_id 并 upsert 覆盖 attachment_key → A 的续跑逻辑操作的是身份已变成 B 的记录，结果静默归错文档，无任何异常抛出。
- **修复建议**：冲突时校验 attachment_key/pdf_sha256 一致性，不一致则报错（或按附件分别建 job，cache_key 唯一约束改为 (cache_key, attachment_key)）。

#### M2 [A] poll 超时被当硬失败 —— 重提丢弃仍在运行的远端 batch
- **位置**：`extractors/manager.py:397-414`（`_resume_poll_with_fallback` 捕获**所有** `MinerUAPIError`）；超时抛出点 `manager.py:362-368` 与 `manager.py:168-175`。
- **失败场景**：大 PDF 远端 batch 运行超过本地 poll 预算 → 超时异常被 fallback 当失败处理 → 清 external_job_id 并**整篇重提**，支付双倍抽取费用，丢弃仍在运行的 batch。
- **修复建议**：引入 `PollTimeoutError` 专用异常：超时只更新 error_code=“poll-timeout”，不落 failed_retryable、不冷却 key；续跑自然走 `_resume_poll` 继续轮询原 external_job_id。

#### M3 [A] `empty-result` 落为终态 —— manual_review 死结
- **位置**：`extractors/mineru.py:377-379`（extract_result 为空 → `state="empty-result"`）；`extractors/recovery.py:85`（未知状态一律 manual_review）。
- **失败场景**：batch 还在排队（结果列表暂时为空）时，job 被持久化为 `empty-result`。恢复计划将其归为 manual_review，此后永久停在评审队列，即使远端几分钟后就完成了。
- **修复建议**：poll 快照把空结果映射为 `running`；recovery 对 `empty-result` 额外映射为 poll 动作。

#### M4 [A] 嵌入批次卡 `running` 永远阻塞 prune
- **位置**：`embeddings/indexer.py:159-173`（先落 `status="running"` 再嵌入）；`indexer.py:329-334`（有 running batch 即拒绝 prune）。
- **失败场景**：嵌入中途崩溃（或 Ctrl-C）→ 批次行永远 running → `vectors prune` 从此拒绝执行，孤儿版本无法清理，且无任何代码路径翻转孤儿批次。
- **修复建议**：prune 前把 `updated_at` 超过阈值（新配置 `vectors.stale_embedding_batch_hours`，默认 24）的 running 批次自动置 failed；新鲜 running 批次仍中止 prune。

#### M5 [A] LanceDB publish 两步非原子 —— 崩溃即检索返回空
- **位置**：`index/lancedb_vector.py:141-161`（meta 表先 `delete` 再 `add`）；缺失回退 `lancedb_vector.py:214-230`（`active_version()` 返回 "legacy"）。
- **失败场景**：delete 后 add 前崩溃 → vector_meta 无行 → active_version 回退 "legacy" → 检索返回空，新版本表躺在磁盘上无人认领。
- **修复建议**：改用单条 `merge_insert` 原子翻牌（已实测 lancedb 0.33.0 支持 `Table.merge_insert`）。

#### M6 [A] reembed 无并发守卫、LanceDB 无跨进程锁 —— 并发发布丢失文档集
- **位置**：`pipeline/reembed.py:57-84`（无任何 running 守卫）；对照 `ingest.py:114-119`（仅守卫 kind="ingest"）。
- **失败场景**：ingest 与 reembed 同时运行，双方基于同一 pre-publish 快照各自 staged 并发布，后者覆盖前者 → 一方文档集被静默丢弃。sqlite-local 有 FileLock 保护，LanceDB 完全没有。
- **修复建议**：reembed 增加与 ingest 同款守卫（两者都是向量写者，kind 互查）；LanceDB 增加 per-profile `filelock.FileLock`（仿 `local_vector.py:122`），在 publish/copy 路径持有。

#### M7 [A] `ingest resume` 只重建计划、永不执行
- **位置**：`pipeline/ingest.py:274-300`（重建 plan 后 `set_job_status("planned")` 即返回）；`cli.py:543-549`。
- **失败场景**：整库跑一半崩溃，操作员执行 `ingest resume` 打印成功但什么都没跑，库保持半建成状态。
- **修复建议**：抽取 `start_ingest_job` 的执行体为共享函数，`resume_ingest_job` 增加 `execute` 参数；CLI 增加 `resume --execute`。

#### M8 [B] 空 keys 分支清空整个附件账本
- **位置**：`db/state.py:2423-2447`（`mark_absent_attachments_deleted`：keys 为空时 SQL 无 WHERE 约束 → 全表标记 deleted）；触发路径 `zotero/scanner.py:77-79`（全量扫描）。
- **失败场景**：Zotero 未运行/shadow 备份暂时不可用导致全量扫描得到 0 行 → 803 条附件全部标记 deleted → 后续 ingest 全部归 missing_file 跳过。部分可见扫描有守卫，**完全空扫描没有**——这是全库唯一真正的数据丢失路径。
- **修复建议**：空 keys 时拒绝执行并报错（全量扫描 0 附件本身即异常信号，应中止而非清库）。

#### M9 [A] 阻塞文档计为 executed、catch-all 吞掉系统性失败
- **位置**：`pipeline/ingest.py:176-234`（主循环）；分段未齐时静默 checkpoint blocked 并返回（`ingest.py:705-713`、`781-789`）。
- **失败场景**：① 分段未下载完的文档被计入 executed，job 状态显示 completed，掩盖未完成工作；② API key 中途被吊销时每篇文档都失败，但 catch-all 把每个失败降级为 document_error 事件，job 终态 `completed_with_errors` 而非 failed，无人察觉管线实际什么都没做。
- **修复建议**：`_execute_document_stages` 返回 executed/blocked 状态，blocked 单独计数；终态区分 completed / completed_with_errors / blocked；catch-all 检测系统性失败（如连续 N 篇同类型异常）时中止并标记 job failed。

#### M10 [A] ingest 守卫 TOCTOU 且只匹配 kind="ingest"
- **位置**：`pipeline/ingest.py:114-119`（先查后建，无唯一约束兜底）。
- **失败场景**：两个 `ingest start --execute` 同时通过检查（或 ingest 与 reembed 并发，见 M6），双写向量库。
- **修复建议**：`db/state.py` 增加 `create_job_if_no_active()`，单事务内「查 active → 插 job」，SQLite 写者串行保证原子；ingest/reembed 共用，active_kinds 互查。

### 5.2 低危缺陷（9 项，C 类）

| 编号 | 位置 | 问题 | 修复建议 |
|------|------|------|----------|
| L1 | `indexer.py:521-522` | batch_hash 输入含**绝对图片路径**——迁移库目录即全部重嵌 | hash 输入改用相对路径或仅文件内容指纹 |
| L2 | `normalize/markdown.py:174-181` | Windows 下 MinerU 绝对风格图片引用（`\storage\...`）拼接出不存在路径 → FileNotFoundError | 识别盘符/根路径引用，fallback 到提取目录内查找 |
| L3 | `normalize/markdown.py:232-235` | 单文件路径下缺失一张图片即整篇失败，分段路径却静默跳过（`merge_extract_segments.py:107-108`）——同一文档两分支行为不一致 | 单文件路径改为告警+跳过，与分段路径对齐 |
| L4 | `normalize/markdown.py:610-613` | 图片块全部排在文本块之后，链尾回绕到文档开头 | 图片块按文中出现位置插入 chunk 链 |
| L5 | `normalize/markdown.py:511-647` | token 估算 `len//4` 对英文低估，flush 阈值可被大幅击穿 | 校准估算系数或按字符上限二次切分 |
| L6 | `extractors/mineru.py:416-432` | 下载 zip 先落盘后校验，失败留下半截文件 | 下载到临时名，校验通过再改名 |
| L7 | `pdf_pages.py:19-25`、`ingest.py:729-733` | 页数读取异常静默吞掉并按 500 页回退，30 页文档被拆 3 段或 600 页文档漏拆分 | 失败时告警，并区分「未知」与「超限」两种策略 |
| L8 | `index/lancedb_vector.py:214-230` | meta 缺失时回退 "legacy" 掩盖损坏，检索静默空 | 移除回退或改为显式报错提示 `vectors verify` |
| L9 | `extractors/mineru.py:155-160` | retryability 判定计算后未参与控制流，M2 的重提决策因此盲目 | 让判定结果驱动重提/续跑分支 |

### 5.3 后端差异总评

sqlite-local 后端在**所有**写路径上比 LanceDB 更安全：staged 去重（ON CONFLICT）、publish 单事务、FileLock 跨进程锁。LanceDB 后端缺少全部三层（H1/M5/M6）。批量发布落地后（6.6），LanceDB 的 publish 频率降为每轮一次，暴露面随之缩小，但 H1/M5/M6 仍应先行修复。

---

## 六、提速/并发改造设计（定稿）

### 6.1 决策记录（2026-08-13）

| # | 决策 | 选择 |
|---|------|------|
| 1 | O(N²) 写放大 | **采纳批量发布**（per-run 版本命名，零 schema/数据迁移，见 6.7） |
| 2 | 并发结构 | **分阶段批量**：submit 全量 → 并发 harvest → normalize → 并发 embed；不做文档级流水线 |
| 3 | 并发参数 | **嵌入放宽、MinerU 保守**（按配额留余量，见 6.8） |
| 4 | 本轮范围 | 先出本报告，**不改任何代码**；修复与提速按第七节批次实施 |

### 6.2 总体结构

```
ingest 一轮（batch 模式）：
  Pass 1  submit 全量：对所有待抽取文档（含分段）生成请求 → 限速提交（key 提交后立即释放）
  Pass 2  harvest 并发：harvest_workers 个 worker 各领一批 job，轮询/下载（key 仅围绕单次 HTTP 调用持有）
  Pass 3  normalize：对抽取完成的文档归一化（保持串行或可选并行）
  Pass 4  embed：stage 全部待嵌入文档（嵌入线程池并发）→ 每 profile 一次批量发布
```

### 6.3 Phase 0 可靠性修复（9 项，并发前置必修）

| # | 缺陷 | 修复 | 涉及文件 |
|---|------|------|----------|
| 0.1 | H1 | staged 表重入即 drop 重建（`version_is_active` 守卫保留） | `index/lancedb_vector.py` |
| 0.2 | M5 | `merge_insert` 原子翻牌（lancedb 0.33.0 已确认支持） | `index/lancedb_vector.py` |
| 0.3 | M6 | reembed 守卫（ingest/reembed 互查）+ LanceDB per-profile FileLock | `pipeline/reembed.py`、`index/lancedb_vector.py` |
| 0.4 | M10 | `create_job_if_no_active()` 单事务原子建 job | `db/state.py`、`pipeline/ingest.py` |
| 0.5 | M2 | `PollTimeoutError` 不落 failed、不冷却 key，续跑走 poll | `extractors/manager.py` |
| 0.6 | M3 | empty-result → running；recovery 增加 poll 映射 | `extractors/mineru.py`、`extractors/recovery.py` |
| 0.7 | M4 | prune 前自动回收 stale running 批次 | `embeddings/indexer.py` |
| 0.8 | M7 | 执行体抽共享函数，`resume --execute` 落地 | `pipeline/ingest.py`、`cli.py` |
| 0.9 | M9 | blocked 单独计数，终态四态区分 | `pipeline/ingest.py` |

### 6.4 Phase 1 嵌入并发（最大提速项）

- `embeddings/qwen.py`：`embed()` 内嵌 `ThreadPoolExecutor(max_workers=embed_workers)` + token bucket 限速（`rate_limit.requests_per_second`）；**`embed_workers=1` 时走原串行路径，零行为变化**（存量测试全部保持绿色）。线程池仅存在于单次 `embed()` 调用内，不触碰向量库/账本。
- `config/settings.py`：`EmbeddingProfile` 增加 `embed_workers` 字段，`rate_limit` 从死配置变为生效。
- `providers.py:118` `build_qwen_embedding_provider()` 透传两参数。
- 预估收益 3-10×（I/O 阻塞释放 GIL，`requests` 每次调用独立 Session 线程安全）。

### 6.5 Phase 2 抽取并发（submit 全量 → 并发 harvest）

- `extractors/manager.py` 拆三原语，`ensure_extraction()` 保持为薄包装（存量调用与测试不变）：
  - `submit_only(request)`：取 key → submit → 落账 → **立即释放 key**；
  - `harvest(job_id)`：循环「取 key → 单次 poll → 落账 → 释放 key → sleep」，completed 后「取 key → download → 释放」——sleep 发生在无 key 状态，N 个 harvester 共享 M 个 key；
  - `poll_once(job_id, api_key)` 供 ensure/resume/harvest 三处复用，消除重复轮询循环。
- `pipeline/ingest.py`：批量路径改为两趟（submit 全量含全部分段 → `ThreadPoolExecutor(harvest_workers)` 分区收割），完成后按文档 `_merge_and_checkpoint_segments` / checkpoint blocked（沿用现有逻辑）。
- `cli.py:516`：向 `ExtractorKeyPool.from_env_file` 传入 `per_key_submit_concurrency`（死参数复活）。
- **不并行**：per-job 状态写（账本 RLock 已串行）、向量库写（本阶段嵌入仍逐篇发布，Phase 3 才改）。

### 6.6 Phase 3 批量发布（消灭 O(N²)）

- `embeddings/indexer.py` 拆两函数：
  - `stage_indexed_document(...)`：profile 解析 → chunk 列表 → batch_hash 跳过检查 → 嵌入 → 构建 records，**不开库不写向量**；
  - `publish_staged_documents(...)`：run hash = sha256(profile_hash + 各 doc batch_hash 排序拼接)；开库一次 → drop 残留 staged 表（幂等重跑，配合 0.1）→ **一次** `copy_active_records_to_version(exclude=本轮全部文档)` → 逐篇 upsert → **一次** `publish_version` → 落账（逐篇 completed + checkpoint + register_vector_index）。
- `index_normalized_document()` 保留为单篇包装（stage+publish 各一次），`embed index-normalized` CLI 与存量测试不变。
- `pipeline/reembed.py` 同改（重嵌走同一条 O(N²) 路径）。
- 崩溃安全：中途崩溃 → staged 表未发布，下轮以新 run hash 重发（0.1 保证无重复行）；发布后崩溃 → run hash 确定性重建，`previous != index_version` 守卫防自删。

### 6.7 数据结构变动处理办法（唯一触及数据面的改动）

- **版本表命名**：从 per-doc batch_hash 变为 per-run hash；`vector_indexes.active_version` 的值随之变化。LanceDB 每版本独立成表的结构不变；sqlite-local 的 `record_id` 前缀机制不变。
- **无 schema 迁移、无数据迁移**：现有库的 active 版本继续服务检索，下一次 publish 时由现有机制替换并删除旧表；历史孤儿版本由 `vectors prune` 按原逻辑清理。
- **per-doc 语义全部保留**：`embedding_batches`（batch_hash PK）、`checkpoints(embed:{profile})`、跳过逻辑（`indexer.py:99-147` 读取的是每条向量记录内的 `embedding_batch_hash` 元数据，与版本表名无关——已核实）均不变。
- **回滚粒度**：`vectors prune` 的粒度从「单篇文档的版本」变为「单轮 ingest 的版本」；项目从未提供按单篇回退的命令，无实际功能损失。
- 已实测排除替代路线：lancedb 0.33.0 无 `DBConnection.copy_table`，廉价克隆不可行。

### 6.8 配置参数档（默认值，均可随时调整）

| 配置项 | 位置 | 默认值 | 依据 |
|--------|------|--------|------|
| `embed_workers` | 各 `[[embedding_profiles]]` | **16** | 嵌入放宽；配额 2400 rpm ≈ 40 req/s |
| `rate_limit.requests_per_second` | 各 `[[embedding_profiles]]` | **38** | 留 5% 余量，防 429 风暴 |
| `per_key_submit_concurrency` | `[extract]` | **1** | MinerU 保守 |
| `submit_min_interval_seconds` | `[extract]` | **1.5** | ≤ 40 提交/分钟（配额 50） |
| `harvest_workers` | `[extract]` | **2** | MinerU 保守 |
| `poll_max_per_minute` | `[extract]` | **600** | 约 700 job × 30s 间隔自然速率 1400/min 超配额，限速至 60% 余量 |
| `poll_interval_seconds` | `[extract]` | 30 | 现状保持 |
| `job_max_lifetime_seconds` | `[extract]` | **21600（6h）** | 任务总寿命上限（自 submitted_at 起算），超后转 `failed_manual_review` 人工裁决，不自动重提（用户拍板 2026-09-04） |
| `max_resubmit_attempts` | `[extract]` | **3** | 远端 batch 真失败后的最大自动重提次数，防坏文档反复烧配额（用户拍板 2026-09-04） |
| `batch_publish` | `[ingest]` | **true** | A/B 开关（关 = 现状逐篇发布） |
| `stale_embedding_batch_hours` | `[vectors]` | 24 | 0.7 修复用 |

> 约束：`job_max_lifetime_seconds` 与 `max_resubmit_attempts` 落地及后续任何调整，必须同步更新 `docs/USAGE.md`（用户要求 2026-09-04）。

CLI 仅新增 `ingest resume --execute`（0.8），无其他命令面变化。所有默认值下存量行为（串行）完全保留，各阶段可独立上线。

### 6.9 实施顺序与验证策略

- **顺序**：Phase 0 → 1 → 2 → 3，逐批 commit（Phase 0 每项独立可测，可合并为一批）。
- **每批验证**：`python -m pytest tests/ -q`（现状基线 228 passed）+ 该批新增针对性测试（stub provider 已具备：`extractors/base.py`、`embeddings/base.py`）：
  - 0.1/0.2：崩溃重入无重复行；merge_insert 无 meta 空窗；
  - 0.5：假 provider 永不完成 + 短超时 → job 保持 running、续跑 poll 原 external id、不重提；
  - 0.6：空 extract_result → running；recovery 归 poll；
  - 0.7：stale running 批次被 prune 自动回收；
  - 0.8：`resume --execute` 完成续跑；
  - Phase 1：假 client 计数并发在途请求 >1、输出顺序==输入顺序、embed_workers=1 行为不变；
  - Phase 2：记录型假 provider 断言「全部 submit 先于任何 poll」、分段两段先行提交、两 key 池账目正确；
  - Phase 3：3 篇 staged 后一次发布，copy 量 == 发布前 active 量（非累计）、中断重跑零重复行（双后端）、reembed 等价。
- **真实库验证**：每批落地后 `zoterorag vectors verify --profile ...`、`zoterorag doctor --verify-vectors`；Phase 3 后观察 `data/vector_store/` 体积不再随增量 N² 增长、`vectors prune` 保持安全。

---

## 七、实施批次建议

| 批次 | 内容 | 前置 |
|------|------|------|
| 1 | Phase 0 可靠性修复（0.1-0.9，A 类 9 项） | 无 |
| 2 | Phase 1 嵌入并发 | 批次 1（推荐；线程池本身独立于向量库，技术上无硬依赖） |
| 3 | Phase 2 抽取并发 | 0.4、0.5、0.8（harvester 必须不误杀慢 batch、可续跑） |
| 4 | Phase 3 批量发布（含 reembed） | 0.1、0.2、0.7 |
| 独立批次 | B 类 4 项：**M8 优先**（唯一数据丢失路径）、H2（截断入库）、H3（乱序合并）、M1（身份劫持） | 不阻塞批次 1-4，可与任意批次穿插 |
| 暂缓 | C 类 L1-L9；文档级流水线（全库重建约再省 1 小时，但复杂度/中断处理成本高，增量跑收益小） | 观察批次 1-4 实测效果后再议 |

**风险提示**：批次 3/4 上线前务必完成 0.5/0.1——并发放大下，poll 超时误杀会导致批量重提（扣费），staged 重复行会导致整库检索脏数据。

---

## 八、本轮边界与后续

- 本轮**未改动**任何 `src/`、`config/`、`tests/` 代码与数据目录；本报告为唯一产出。
- 与 2026-06-15 审查报告的关系：其 P0/P1 已全部修复（`docs/CODEBASE_AUDIT_REPORT.md` 复查摘要）；本报告新增缺陷均为其后的增量开发（分段提取、双 profile 嵌入、版本清理）引入。
- 后续：按第七节批次逐批实施，届时另出实施计划；每批独立验证、独立 commit。
