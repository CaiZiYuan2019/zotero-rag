# ZoteroRAG 商业化级本地项目规划

## 1. 目标与边界

ZoteroRAG 是一个本地优先的 Zotero 论文知识库系统，面向长周期、可恢复、可审计的 PDF 转换、向量化和检索工作流。系统默认采用 Python 方案，运行在项目内 `.conda` 环境中；Zotero 只作为只读数据源，MinerU 只是可替换的 PDF 提取后端，向量库默认采用本地嵌入式 LanceDB。

核心目标：

- 支持长时间、昂贵、可断点继续的建库流程，避免重复提交 MinerU 或重复调用 embedding API。
- 支持增量建库、全量重建、单条重建、只重建向量库、保存多个向量库。
- 支持两种检索模式：纯文字检索和多模态检索。
- 支持两类使用者：手动检索用户与外接 LLM/MCP 调用者。
- 保留直接检索接口：标题、作者、发布时间、摘要、DOI、标签、collection、转换全文，不依赖 embedding。
- 对 MinerU、embedding provider、vector profile、查询接口做显式抽象，降低后续替换成本。

当前已确认的本地事实：

- 历史代码位于 `reference/`，包含 MinerU API CLI、MinerU 图片顺序重命名脚本、旧版 Zotero shadow + LanceDB + GUI 单脚本。
- 当前 Zotero 主库路径历史配置为 `E:\ZoteroLib\lib\zotero.sqlite`，storage 为 `E:\ZoteroLib\lib\storage`。
- Zotero 数据库中 PDF 附件约 411 个、HTML 快照约 132 个、Word 约 2 个。
- 双语 PDF 主要以后缀 `_zh-CN_dual.pdf` 出现，但也存在 Scholaread/中英对照类文件。
- Zotero 摘要在 `itemData` 的 `abstractNote` 字段；HTML 快照是附件，通常有父条目。

## 2. 项目结构

推荐目录：

```text
zotero-rag/
  docs/
    PROJECT_PLAN.md
  reference/
    mineru_cli.py
    rename_mineru_images.py
    rag_server.py
  src/zoterorag/
    api/
    cli/
    config/
    db/
    embeddings/
    extractors/
    index/
    mcp/
    normalize/
    pipeline/
    search/
    zotero/
  tests/
  config/
    config.example.toml
  data/              # git ignored
  logs/              # git ignored
  backups/           # git ignored by default
```

运行数据目录：

- `data/state/state.sqlite`：唯一任务账本，记录扫描、任务、产物、索引、进度、备份。
- `data/shadow/zotero.sqlite`：Zotero shadow copy。
- `data/extract_cache/`：MinerU zip、解压目录、API 任务 manifest。
- `data/normalized/`：标准化 Markdown、顺序图片、图片 manifest、chunk manifest。
- `data/vector_store/<profile>/`：不同 embedding profile 的 LanceDB 向量库。
- `data/embedding_cache/<profile>/`：可选 embedding 响应缓存。

## 3. Zotero 同步与 PDF 分类

### 3.1 Shadow 读取

不直接长期读取 Zotero 主库。每次扫描先创建 shadow copy，再从 shadow 中读取：

- 父条目 key、item type、title、authors、date/year、DOI、abstractNote、publicationTitle、url、tags、collections。
- 附件 key、path、contentType、storageHash、storageModTime、文件 size、mtime、sha256。
- 删除条目、缺失文件、无父条目、非 PDF 附件。

增量指纹：

```text
attachment_key
relative_path
file_size
file_mtime
file_sha256
zotero_parent_metadata_hash
extractor_profile_hash
chunker_profile_hash
embedding_profile
```

### 3.2 PDF 分类

所有 PDF 进入候选表，状态为：

- `included_auto`
- `excluded_auto`
- `needs_review`
- `included_manual`
- `excluded_manual`
- `orphan_metadata_only`
- `missing_file`

双语/翻译 PDF 规则：

- 默认不直接删除，也不永久排除，而是进入 review queue。
- 自动匹配原因必须记录：`_zh-CN_dual.pdf`、`dual.pdf`、`双语`、`中英对照`、`Scholaread`、`Immersive` 等。
- `Immersive` 只作为弱信号，不能单独永久排除。
- 支持人工覆盖：
  - `zoterorag include --attachment-key KEY --reason "..."`
  - `zoterorag exclude --attachment-key KEY --reason "..."`
  - `zoterorag review list`
  - `zoterorag review explain --attachment-key KEY`
- 人工覆盖优先级高于关键词规则。

同一条目多 PDF：

- 有普通 PDF + 双语 PDF：默认普通 PDF 建库，双语 PDF 进入 review。
- 只有双语 PDF：允许人工或策略纳入，建库后标记 `source_quality=translated_pdf_only`，检索结果中提示。
- 多个普通 PDF：按规则选主 PDF，其他进入 review。主 PDF 规则按 `is_primary` 配置、文件名相似度、页数、文件大小、加入时间综合判断。

无父条目：

- 只有附件级 title/url/accessDate 的 PDF：默认报告但不参与 MinerU 和向量建库。
- 若能关联到网页快照或网页条目摘要：只将 title/url/abstract/date 进入 metadata/direct-search，不做 PDF 转换。
- 若用户人工 include 无父 PDF：允许建库，但元数据质量标记为 `orphan_manual_include`。

Word/Markdown/HTML：

- Word/Markdown 当前不参与建库，只报告。
- HTML 快照当前不做全文转换；父条目的 title/abstract/date/url 进入 direct-search。

## 4. MinerU 提取层

### 4.1 Provider 抽象

定义 `ExtractorProvider`：

```text
fingerprint(input_file, options) -> extractor_hash
submit(task) -> external_job_id
poll(external_job_id) -> external_state
download(external_job_id) -> artifact
normalize(artifact) -> normalized_document
```

MinerU 是第一个 provider。后续可替换为本地 OCR、GROBID、Marker、Nougat 或其他 PDF parser。

### 4.2 本地缓存

缓存 key：

```text
pdf_sha256 + selected_pages + extractor_name + extractor_version + options_hash
```

命中缓存时：

- 不重新提交 MinerU。
- 不重复下载 zip。
- 可从 zip、解压目录、normalized manifest 任一阶段恢复。

### 4.3 多 API Key 并发与轮询

支持 `mineru_api_keys` 池：

- 每个 key 有别名，不在日志中输出真实 key。
- 默认保守限流：每 key 同时 1 个 submit/upload，poll 可以并发但有 per-key 间隔。
- 全局并发、每 key 并发、poll interval、失败退避、每日上限均可配置。
- 任务分配使用 key pool 轮询，失败 key 进入 cooldown。

MinerU job 持久化字段：

```text
job_id
pdf_sha256
attachment_key
api_key_alias
batch_id
request_payload_hash
submitted_at
last_poll_at
external_state
local_stage
zip_path
extract_dir
error_code
error_message
retry_count
```

断点继续：

- 有 `batch_id` 无 zip：继续 poll/download。
- 有 zip 无 extract：继续解压。
- 有 extract 无 normalized：继续 normalize。
- 有 normalized：跳过 MinerU。
- 失败不无限重试，进入 `failed_retryable` 或 `failed_manual_review`。

## 5. 标准化、文本块与图片块

### 5.1 标准化产物

每篇文档生成：

- `document.md`：标准化 Markdown。
- `images/img001.ext ...`：顺序图片名。
- `images/original_manifest.json`：原始 hash 名、顺序名、sha256、尺寸、文件大小、引用位置。
- `chunks.jsonl`：chunk manifest。
- `document_manifest.json`：PDF、MinerU、标准化、chunker 版本和状态。

### 5.2 图片分辨率处理

保留两套图片：

- 原图：用于人工查看和结果引用，不改分辨率。
- embedding 图：用于 API 调用，按 profile 生成。

默认处理：

- 读取原图宽高、像素数、文件大小、格式。
- embedding 图最长边默认不超过 1600 px，且文件不超过 provider 限制。
- 对阿里云 `qwen3-vl-embedding`，单图不超过 5 MB。
- 对开源 Qwen3-VL-Embedding，本地 profile 使用 `min_pixels/max_pixels` 控制；官方示例默认最大像素约 1280 x 1440 量级。
- 小图、低信息图、重复图不直接丢弃，先标记 `image_quality_flags`，是否跳过由配置决定。

### 5.3 文本块切分

文本块用于纯文字检索和给 LLM 返回文本证据：

- 按 Markdown 标题层级、段落、列表、图注、表格边界切分。
- 默认目标长度 1800-2400 tokens。
- 默认 overlap 500-800 tokens，比旧方案显著扩大，避免论文实验方法、结果解释被切断。
- 不硬切图注、表格行、公式块。
- 每个 text chunk 保存 heading path、前后邻接 chunk id、页码或源位置。

### 5.4 图片块切分

图片块不是简单“每图一个孤立块”，而是图像与上下文绑定：

- 每张图片生成 `image_block`，绑定最近标题、图注、前文窗口、后文窗口。
- 连续图片形成 `image_run`，保留顺序。例如多 panel figure 被 MinerU 拆成连续多张图时，作为同一组处理。
- 由于百炼 `qwen3-vl-embedding` 单请求最多 1 张图片，多图 run 会拆成多个 `image_block_vector`，但共享同一个 `image_run_id`。
- 同一 chunk 命中多个图片向量时，检索层合并为一个结果，避免重复刷屏。

### 5.5 纯文字与多模态索引关系

每篇文档生成两类索引输入：

- `text_chunk`：只含文本，用于 text embedding、BM25、metadata/fulltext。
- `image_block`：含图片 + 图注 + 邻近文本，用于 multimodal embedding。

两者共享 `document_id` 和 citation 元数据，但输出策略不同。

## 6. Embedding 模型与多向量库

### 6.1 官方能力核实

截至本规划编写时：

- 阿里云百炼文档列出 `qwen3-vl-embedding` 支持 2560/2048/1536/1024/768/512/256 维，默认 2560；文本上限 32000 tokens；单请求最多 1 张图片且图片不超过 5 MB；多模态融合向量需使用 DashScope SDK 或 REST API。
- Qwen 官方开源 `Qwen3-VL-Embedding` 示例中，embedding 输入支持 `instruction` 字段；reranker 输入才是显式 `query + documents`。
- 因此本项目不硬编码“query/document 两种 embedding endpoint”。实现时必须做 provider capability 探测：
  - 若 provider 支持 `text_type=query/document` 或等价参数，则 query 使用 query role，文档使用 document role。
  - 若不支持，则 query 使用检索 instruction，文档使用普通 text/image input。
  - 所有 role/instruction 参数进入 `embedding_profile_hash`，避免变更后误复用旧向量。

参考：

- Alibaba Cloud Model Studio Embedding 文档：https://www.alibabacloud.com/help/doc-detail/2842587.html
- Qwen3-VL-Embedding 官方仓库：https://github.com/QwenLM/Qwen3-VL-Embedding

### 6.2 EmbeddingProfile

每个向量模型配置为一个 profile：

```text
profile_name
provider
model
dimension
modality: text | multimodal
query_role_mode
document_role_mode
instruction_template
image_policy
batch_size
rate_limit
vector_store_path
enabled
default_for_text
default_for_multimodal
```

默认：

- `qwen3vl_cloud_2560_text`
- `qwen3vl_cloud_2560_multimodal`

未来可添加：

- `qwen3vl_local_2048`
- `qwen3vl_local_4096`
- 其他 text embedding 模型。

### 6.3 多向量库保存

允许保存多个向量库，但单次查询只使用一个 embedding profile：

- 不跨模型混合 score。
- 不把 2560 维和 2048/4096 维结果合并。
- 查询接口必须允许列出、选择、指定模型。

接口：

- `GET /models/embedding`：列出可用 profile、维度、状态、文档数、chunk 数、是否默认。
- `POST /models/embedding/activate`：设置默认 profile。
- `POST /search/text`：可传 `profile_name`，不传则使用默认 text profile。
- `POST /search/multimodal`：可传 `profile_name`，不传则使用默认 multimodal profile。
- `zoterorag models list`
- `zoterorag search --profile PROFILE ...`

重建：

- `zoterorag reembed --profile NEW --from-normalized`
- 只从 normalized/chunk manifest 重算向量，不重新调用 MinerU。

## 7. 检索模式与结果返回

### 7.1 纯文字检索

纯文字检索面向纯文字 LLM 和不希望处理图片的调用者：

- 输入只能是 text query。
- 召回来源：metadata FTS、全文 BM25、text vector。
- 输出不得包含图片文件、图片 base64、图片 URL 或要求 LLM 查看图片的内容。
- 若命中来源关联图片，只允许返回文本化信息：`has_images=true`、`image_count`、图注文本、图片编号，不返回图像内容。
- citation 包含 Zotero key、title、authors、year、section/page、chunk text、score。

API：

```text
POST /search/text
{
  "query": "...",
  "profile_name": "qwen3vl_cloud_2560_text",
  "filters": {},
  "top_k": 10,
  "consumer": "llm_text"
}
```

### 7.2 多模态检索

多模态检索面向人工查看或多模态 LLM：

- 输入可为 text、image file、image base64、或 text + image。
- 使用 multimodal embedding profile。
- 可返回图片证据，但返回方式按 consumer 区分。

手动检索：

- 返回本地文件路径、缩略图路径、原图路径、图注、上下文文本。
- 不默认 base64，避免响应过大。

LLM/MCP 检索：

- `consumer=llm_text`：强制降级为纯文字输出，不含图像。
- `consumer=llm_multimodal`：返回可控图片载荷。
- 图片返回策略：
  - `image_return=file_ref`：返回本地安全文件引用或 MCP resource id。
  - `image_return=base64`：只对小图或缩略图启用，受 `max_images`、`max_bytes` 限制。
  - `image_return=none`：只返回图注和文本上下文。

API：

```text
POST /search/multimodal
{
  "query_text": "...",
  "query_image": {"type": "file_path|base64", "value": "..."},
  "profile_name": "qwen3vl_cloud_2560_multimodal",
  "consumer": "manual|llm_text|llm_multimodal",
  "image_return": "file_ref|base64|none",
  "top_k": 10,
  "max_images": 5
}
```

### 7.3 直接检索

无需 embedding API：

- `POST /search/metadata`
- `POST /search/fulltext`

支持标题、作者、年份范围、发布时间、摘要、DOI、URL、publication、tag、collection、MinerU 转换全文。

### 7.4 Rerank

暂不实现 rerank。保留扩展点：

- `RerankProvider` 接口。
- 查询请求保留 `rerank=false` 字段，默认永远 false。
- score breakdown 预留 `rerank_score=null`。
- 后续若启用，再接 `qwen3-rerank` 或 `qwen3-vl-rerank`。

## 8. SQLite 写入瓶颈与任务调度

SQLite 单点写入是明确瓶颈，设计上必须避免多 worker 直接抢写：

- `state.sqlite` 开启 WAL。
- 只有一个 state writer 线程/进程负责写 SQLite。
- MinerU worker、embedding worker、index worker 通过内部队列提交状态事件。
- 大字段、日志、API 响应正文、embedding 向量不写入 SQLite 主表，只保存文件路径、hash、摘要字段。
- 高频进度先写内存事件流和日志，按节流策略落库，例如每 N 秒或阶段完成写一次。
- 批量插入使用事务，避免每个 chunk 单独 commit。
- LanceDB 写入也通过单独 index writer 批量写，避免状态库和向量库写入互相阻塞。

关键表：

- `documents`
- `attachments`
- `review_rules`
- `extract_jobs`
- `normalized_artifacts`
- `chunks`
- `embedding_profiles`
- `embedding_batches`
- `vector_indexes`
- `pipeline_jobs`
- `job_events`
- `backups`

## 9. 进度、断点与恢复

建库阶段：

```text
scan_zotero
classify_pdf
extract_submit
extract_poll
extract_download
normalize
chunk
embed_text
embed_multimodal
index
verify
```

进度输出：

- 全局：总文档数、待处理、已完成、失败、review、跳过、ETA。
- 文档级：当前阶段、页数、图片数、chunk 数、embedding batch 数。
- MinerU：key alias、batch id、状态、poll 次数、冷却/失败。
- Embedding：profile、文本 batch、图片 batch、成功数、失败数、估算费用。
- Index：LanceDB 表、写入行数、删除旧行数、校验结果。

断点保存：

- 每个文档阶段完成写 checkpoint。
- 每个 MinerU batch_id 获取后立即落库。
- 每个 embedding batch 成功后记录 batch hash。
- index 写入采用临时表或版本字段，完成后原子切换 active version。

恢复：

- `zoterorag resume`
- `POST /ingest/resume`

恢复时从 state ledger 判断最远安全点，不重复提交已缓存任务。

## 10. 备份与恢复

保留两级备份。

默认快照备份：

- `state.sqlite`
- config 文件
- review/include/exclude 规则
- normalized manifest
- chunk manifest
- LanceDB manifest/profile 信息

全量备份：

- 复制到用户指定文件夹。
- 包含 state、config、review rules、MinerU zip、解压结果、normalized Markdown/images、embedding cache、LanceDB 向量库、日志摘要。
- 不复制 Zotero 原始主库和 Zotero storage，除非用户显式启用 `include_zotero_source=true`。

CLI：

```text
zoterorag backup create --mode snapshot --out D:\Backup\ZoteroRAG
zoterorag backup create --mode full --out D:\Backup\ZoteroRAG
zoterorag backup list
zoterorag backup restore BACKUP_ID
zoterorag verify-index --profile PROFILE
```

备份要求：

- 备份前暂停 writer 或进入只读 checkpoint。
- 写入 backup manifest，包含文件 hash、大小、创建时间、关联 job id。
- 恢复前自动创建当前状态的 pre-restore snapshot。

## 11. 外部接口

FastAPI：

- `GET /health`
- `GET /status`
- `GET /models/embedding`
- `POST /models/embedding/activate`
- `POST /scan`
- `POST /ingest/start`
- `POST /ingest/pause`
- `POST /ingest/resume`
- `POST /ingest/cancel`
- `GET /jobs/{job_id}`
- `GET /review`
- `POST /review/include`
- `POST /review/exclude`
- `GET /documents`
- `GET /documents/{doc_id}`
- `POST /search/text`
- `POST /search/multimodal`
- `POST /search/metadata`
- `POST /search/fulltext`
- `POST /backup/create`
- `GET /backup/list`
- `POST /backup/restore` — **deferred/not implemented**；恢复目前仅通过 CLI `zoterorag backup restore` 执行。

CLI：

- `zoterorag scan`
- `zoterorag ingest --incremental`
- `zoterorag ingest --full`
- `zoterorag ingest --zotero-key KEY`
- `zoterorag resume`
- `zoterorag status`
- `zoterorag review list`
- `zoterorag include --attachment-key KEY`
- `zoterorag exclude --attachment-key KEY`
- `zoterorag models list`
- `zoterorag search`
- `zoterorag search-mm`
- `zoterorag backup create`
- `zoterorag serve`

MCP：

- `zotero_rag_search_text`
- `zotero_rag_search_multimodal`
- `zotero_rag_metadata_search`
- `zotero_rag_get_document`
- `zotero_rag_list_models`
- `zotero_rag_status`

MCP 默认面向 LLM 安全输出；除非调用方声明 `llm_multimodal` 且请求 `file_ref/base64`，否则不返回图片内容。

## 12. 实施阶段

第一阶段：仓库与基础框架

- 初始化 Python 项目、配置系统、日志、SQLite state schema、CLI skeleton、FastAPI skeleton。
- 加入 `.gitignore`，排除 `.conda`、key、token、runtime data、vector store、MinerU 输出。

第二阶段：Zotero scanner 与 review queue

- shadow copy。
- 元数据抽取。
- PDF 分类、双语检测、人工 include/exclude。
- 扫描报告。

第三阶段：MinerU provider

- 多 API key 池。
- submit/poll/download/cache。
- 断点恢复。
- ZIP 解压与历史图片重命名逻辑迁移。

第四阶段：normalize/chunk/image policy

- 标准化 Markdown。
- 图片 manifest、原图/embedding 图。
- text chunk、image block、image run。

第五阶段：embedding 与多 profile LanceDB

- qwen3vl cloud provider。
- text profile 与 multimodal profile。
- 多向量库保存、列出、选择、重建。

第六阶段：检索接口

- metadata/fulltext/text vector/multimodal vector。
- 手动检索与 LLM 检索不同返回格式。
- MCP tools。

第七阶段：备份、恢复、验证

- snapshot/full backup。
- restore。
- verify-index。
- 长任务进度与恢复测试。

## 13. 测试与验收

必须覆盖：

- 双语 PDF 自动 review 与人工 include/exclude。
- `Immersive` 弱匹配不导致不可逆排除。
- only-dual PDF 可人工建库且检索结果带提示。
- 无父 PDF 默认不转换。
- MinerU 多 key 保守并发、poll 恢复、失败 cooldown。
- 二次增量不重复 MinerU、不重复 embedding。
- 修改 embedding profile 后只重建向量库。
- 多 vector profile 可列出、可指定、单次查询只用一个 profile。
- 纯文字检索结果绝不包含图片文件/base64。
- 多模态手动检索返回文件引用，多模态 LLM 检索按参数返回 file_ref/base64/none。
- SQLite 单 writer 下长任务状态稳定落库。
- 中断后 resume 从 checkpoint 继续。
- snapshot/full backup 可创建；full backup 可复制到指定目录并校验 manifest。

