# ZoteroRAG 使用手册

## 目录

1. [环境准备](#1-环境准备)
2. [配置说明](#2-配置说明)
3. [命令行接口 (CLI)](#3-命令行接口-cli)
4. [建库流程](#4-建库流程)
5. [检索接口](#5-检索接口)
6. [HTTP API](#6-http-api)
7. [MCP 工具](#7-mcp-工具)
8. [备份与恢复](#8-备份与恢复)
9. [监控与诊断](#9-监控与诊断)
10. [典型工作流](#10-典型工作流)
11. [架构说明](#11-架构说明)

---

## 1. 环境准备

### 1.1 系统要求

- Python >= 3.11
- Zotero 7.x（本地安装，作为只读数据源）
- Conda 环境（推荐，项目默认使用 `.conda`）

### 1.2 安装

```bash
cd ZoteroRAG

# 核心依赖（状态管理、配置、CLI）
pip install -e .

# 完整安装（API服务、外部provider、PDF处理）
pip install -e ".[api,providers,pdf]"

# 开发环境（测试、lint）
pip install -e ".[api,providers,pdf,dev]"

# LanceDB 可选后端
pip install lancedb
```

### 1.3 初始化

```bash
# 创建运行时目录并初始化 state.sqlite
zoterorag init-state
```

初始化后 `data/` 目录结构：
```
data/
  state/state.sqlite          # 唯一任务账本
  shadow/zotero.sqlite        # Zotero 影子副本
  extract_cache/              # MinerU 下载缓存
  normalized/                 # 标准化 Markdown / 图片 / chunk
  vector_store/<profile>/     # 向量库（每 profile 一个）
  embedding_cache/<profile>/  # embedding 响应缓存（可选）
```

---

## 2. 配置说明

### 2.1 主配置文件 (`config/config.example.toml`)

```toml
[paths]
zotero_db = "E:/ZoteroLib/lib/zotero.sqlite"   # Zotero 主库（只读）
zotero_storage = "E:/ZoteroLib/lib/storage"     # Zotero 附件存储（只读）
data_dir = "data"                                # 运行时数据根目录

[server]
host = "127.0.0.1"                               # API 监听地址
port = 8765                                       # API 端口
require_api_token = true                          # 是否要求 token 鉴权

[[embedding_profiles]]
name = "qwen3vl_cloud_2560_text"
provider = "dashscope"
model = "qwen3-vl-embedding"
dimension = 2560
modality = "text"
enabled = true
default_for_text = true
default_for_multimodal = false

[[embedding_profiles]]
name = "qwen3vl_cloud_2560_multimodal"
provider = "dashscope"
model = "qwen3-vl-embedding"
dimension = 2560
modality = "multimodal"
enabled = true
default_for_text = false
default_for_multimodal = true
```

### 2.2 API 密钥文件 (`.env`)

```bash
# MinerU PDF 提取（支持多 key 轮询）
MINERU_KEY=your_mineru_api_key
# MINERU_KEY_2=second_key    # 可选：更多 key

# 百炼 Qwen Embedding
BAILIAN_KEY=your_bailian_api_key

# 可选：自定义 MinerU 端点
# MINERU_APPLY_UPLOAD_URL=https://mineru.example.com/api/v4/file-urls/batch
# MINERU_BATCH_RESULT_URL=https://mineru.example.com/api/v4/extract-results/batch/{batch_id}

# 可选：自定义 Embedding 端点
# DASHSCOPE_MULTIMODAL_EMBEDDING_URL=https://custom.endpoint/embed

# 注意：BAILIAN_URL 不会被 embedding 模块使用（它通常指向 chat API）
```

### 2.3 API 鉴权 Token

```bash
# 设置环境变量即可启用 token 鉴权
export ZOTERORAG_API_TOKEN="your-secret-token"

# 不设置时：仅允许 loopback (127.0.0.1, ::1, localhost) 访问
# require_api_token=false 时：非 loopback 需要 token
```

---

## 3. 命令行接口 (CLI)

### 3.1 全局参数

```bash
zoterorag --config path/to/config.toml <command>
```

所有命令输出 JSON，方便脚本解析。

### 3.2 命令索引

| 命令 | 用途 | API调用 |
|------|------|---------|
| `init-state` | 初始化运行时目录和 state.sqlite | 无 |
| `status` | 全局状态摘要 | 无 |
| `serve` | 启动 FastAPI 控制服务器 | 无 |
| `doctor` | 非侵入式诊断 | 无 |
| `progress` | 详细建库进度 | 无 |
| `shadow-copy` | 创建 Zotero 影子副本 | 无 |
| `scan` | 影子复制 + 扫描分类 | 无 |
| `providers status` | 检查外部 API 配置 | 无 |
| `models list/activate` | 管理 embedding 配置文件 | 无 |
| `vectors list/verify` | 查看/校验向量索引 | 无 |
| `jobs list/show` | 查看流水线作业 | 无 |
| `ingest start/pause/resume/cancel` | 建库流水线控制 | `--execute` 时 |
| `review list/include/exclude/explain` | 人工审核管理 | 无 |
| `attachments` | 列出附件扫描结果 | 无 |
| `documents list/show` | 查看文档记录 | 无 |
| `search-metadata` | 元数据搜索 | 无 |
| `search-fulltext` | 全文搜索 | 无 |
| `search-vector` | 向量搜索 | 指定 `qwen3vl` 时 |
| `backup create/list/verify/restore` | 备份管理 | 无 |
| `extract dry-run/jobs/recovery-plan` | 提取作业管理 | `dry-run`(真实) |
| `normalize markdown/list/chunks` | 标准化管理 | 无 |
| `embed index-normalized/batches` | 嵌入管理 | 指定 `qwen3vl` 时 |
| `reembed` | 向量重建 | `--execute` 时 |
| `inspect-shadow` | 直接查看影子库 | 无 |

---

## 4. 建库流程

### 4.1 流程概览

```
         ┌─────────────── 本地只读 ───────────────┐
         │                                          │
    shadow-copy ──→ scan ──→ review ──→ ingest     │
         │            │         │           │       │
    Zotero只读   自动分类   人工审核    plan/execute │
    SQLite备份   翻译检测   include/     │           │
                           exclude      ↓           │
         └───────────────────────────────┘           │
                                                    │
         ┌──────────── 外部API调用 ────────────┐     │
         │                                      │     │
         extract ──→ normalize ──→ embed ──→ index  │
         MinerU       Markdown     Qwen    LanceDB   │
         API v4       标准化      DashScope 或SQLite │
                      chunk切分               │     │
                     图片衍生物               ↓     │
                                             向量库  │
         └──────────────────────────────────────┘
```

### 4.2 首次建库

#### 步骤 1：扫描 Zotero

```bash
# 一键完成：复制影子库 + 扫描所有附件 + 自动分类
zoterorag scan

# 分步执行：
zoterorag shadow-copy              # 只做影子复制
zoterorag scan --no-refresh-shadow # 使用已有影子库扫描

# 测试用：限制扫描数量
zoterorag scan --limit 10
```

**分类结果说明：**

| 分类 | 含义 | 后续行为 |
|------|------|---------|
| `included_auto` | 普通 PDF，可以建库 | 默认进入 ingest |
| `needs_review` | 双语/翻译 PDF，需人工确认 | 需人工审核 |
| `report_only` | Word/HTML/非PDF附件 | 不参与建库，仅报告 |
| `orphan_metadata_only` | 无父条目 PDF | 默认不转换 |
| `missing_file` | 文件不存在 | 标记缺失 |
| `included_manual` | 人工纳入 | 进入 ingest |
| `excluded_manual` | 人工排除 | 不参与建库 |

**翻译 PDF 自动检测规则：**
- 强信号（文件名匹配）：`_zh-CN_dual.pdf`, `dual.pdf`, `_dual`, `双语`, `中英对照`, `Scholaread`
- 弱信号：`Immersive`, `沉浸式`（不单独触发排除，仅标记）
- 同一条目有普通 PDF + 双语 PDF：默认普通 PDF 建库，双语 PDF 进入 review
- 只有双语 PDF：允许人工纳入，建库后标记 `source_quality=translated_pdf_only`

#### 步骤 2：人工审核

```bash
# 查看待审核列表
zoterorag review list

# 解释某条附件的分类理由（含策略说明和建议操作）
zoterorag review explain --attachment-key <KEY>

# 手动纳入/排除
zoterorag review include --attachment-key <KEY> --reason "这是唯一版本，原文不可得"
zoterorag review exclude --attachment-key <KEY> --reason "确认是低质量机翻PDF"

# 人工审核优先级高于关键词规则
```

#### 步骤 3：制定建库计划

```bash
# 增量模式（默认）：只处理新增/变更的文档
zoterorag ingest start --mode incremental

# 全量重建：所有文档重新走完整流水线
zoterorag ingest start --mode full

# 只看某篇文档
zoterorag ingest start --zotero-key <ATTACHMENT_KEY_OR_PARENT_KEY>

# 只建纯文本（跳过多模态 embedding）
zoterorag ingest start --text-only
```

计划输出每个文档的状态矩阵：
```json
{
  "document_id": "...",
  "title": "...",
  "stages": [
    {"stage": "extract",      "status": "pending"},
    {"stage": "normalize",    "status": "blocked"},
    {"stage": "embed:text",   "status": "blocked"},
    {"stage": "embed:multimodal", "status": "blocked"}
  ],
  "next_stage": "extract:pending"
}
```

阶段状态：
- `pending` — 需要执行
- `done` — 已完成，可跳过
- `blocked` — 依赖前置阶段完成
- `skipped` — 已跳过（如无多模态配置时跳过多模态 embedding）

#### 步骤 4：执行建库

```bash
# 完整流水线：MinerU 提取 → 标准化 → Qwen 向量化
zoterorag ingest start --mode incremental --execute

# 只做文本 embedding（不调多模态）
zoterorag ingest start --text-only --execute

# 单篇文档
zoterorag ingest start --zotero-key <KEY> --execute
```

**执行流程：** extract → normalize → embed(text) → embed(multimodal)

**容错机制：** 单篇文档失败不影响其他文档；最终状态 `completed`（全部成功）或 `completed_with_errors`（部分失败）。

### 4.3 增量更新

```bash
# 重新扫描 Zotero（检测新增/修改/删除的附件）
zoterorag scan

# 增量建库（只处理 scan_status=new/changed 的文档）
zoterorag ingest start --mode incremental --execute
```

**增量指纹**（任一变化触发重新处理）：
```
attachment_key + relative_path + file_size + file_mtime
+ file_sha256 + parent_metadata_hash + extractor_profile_hash
+ chunker_profile_hash + embedding_profile
```

### 4.4 断点续跑

建库过程可随时中断（Ctrl+C），每个文档每个阶段完成后写入 checkpoint。

```bash
# 暂停/恢复/取消
zoterorag ingest pause --job-id <JOB_ID>
zoterorag ingest resume --job-id <JOB_ID>
zoterorag ingest cancel --job-id <JOB_ID>

# 查看所有作业
zoterorag jobs list
zoterorag jobs show --job-id <JOB_ID>
```

**各阶段恢复策略：**

| 中断点 | 恢复行为 |
|--------|---------|
| MinerU 已提交，未轮询 | 继续 poll |
| MinerU 已完成，未下载 | 继续 download ZIP |
| ZIP 已下载，未解压 | 继续解压 |
| 已解压，未标准化 | 继续 normalize |
| 已标准化，未向量化 | 继续 embed |
| 全部完成 | 跳过（不重复 API 调用） |

**Embedding 去重：** 通过 batch hash（profile hash + document ID + chunk hashes）检测已完成的 embedding batch，命中时返回 `reused_existing=true`，完全不调用 API。

### 4.5 仅重建向量库（不调 MinerU）

切换 embedding 模型或向量维度时，只需重建向量：

```bash
# 先看计划
zoterorag reembed --from-normalized --profile new_model --plan-only

# 全量重建文本向量（真实 Qwen API）
zoterorag reembed --from-normalized --profile new_model --execute

# 限制范围
zoterorag reembed --from-normalized --profile new_model --document-id <ID> --execute

# 强制重建（忽略已有 checkpoint）
zoterorag reembed --from-normalized --profile new_model --execute --force

# 测试用（存根 provider，不调 API）
zoterorag reembed --from-normalized --profile new_model --execute --allow-stub-provider
```

**Profile hash 变更检测：** 当 embedding 模型的配置（name/provider/model/dimension/modality/instruction_template/image_policy）发生变化时，系统自动检测并标记需要重建的文档。

### 4.6 手动分步执行

```bash
# 1. 提取单篇 PDF（调用真实 MinerU API）
zoterorag extract dry-run --pdf /path/to/paper.pdf

# 2. 标准化 MinerU 输出（纯本地，无网络）
zoterorag normalize markdown \
  --markdown mineru_output/full.md \
  --document-id DOC001

# 3. 嵌入单篇（测试用存根）
zoterorag embed index-normalized \
  --profile qwen3vl_cloud_2560_text \
  --document-id DOC001

# 3. 嵌入单篇（真实 Qwen API）
zoterorag embed index-normalized \
  --profile qwen3vl_cloud_2560_text \
  --document-id DOC001 \
  --embedding-provider qwen3vl

# 4. 查看嵌入批次
zoterorag embed batches --profile qwen3vl_cloud_2560_text
```

### 4.7 管理 embedding 模型

```bash
# 列出所有配置的模型
zoterorag models list

# 激活文本搜索的默认模型
zoterorag models activate --profile qwen3vl_local_4096 --mode text

# 激活多模态搜索的默认模型
zoterorag models activate --profile qwen3vl_cloud_2560_multimodal --mode multimodal
```

**原则：** 单次查询只使用一个 embedding profile，不跨模型混合 score，不合并不同维度的向量。

---

## 5. 检索接口

### 5.1 元数据搜索（无需 embedding）

```bash
# 按标题/作者/摘要/DOI/期刊/标签搜索
zoterorag search-metadata "transformer attention"

# 限定分类
zoterorag search-metadata "CRISPR" --classification included_auto
```

搜索字段：title, authors, date, year, DOI, abstractNote, publicationTitle, url, tags, collections

### 5.2 全文搜索（无需 embedding）

```bash
# SQLite LIKE 搜索标准化 chunk 文本
zoterorag search-fulltext "pollen tube guidance"

# 限定 chunk 类型
zoterorag search-fulltext "LUREs" --chunk-type text
zoterorag search-fulltext "figure 3" --chunk-type image
```

### 5.3 向量搜索

**文本搜索（纯文字检索）：**

```bash
# 存根 provider（无 API 调用，用于测试）
zoterorag search-vector "defensin LURE pollen" --mode text --embedding-provider stub

# 真实 Qwen API（调用 DashScope）
zoterorag search-vector "defensin LURE pollen" \
  --mode text \
  --embedding-provider qwen3vl \
  --top-k 10

# 指定 profile
zoterorag search-vector "query" \
  --mode text \
  --profile qwen3vl_cloud_2560_text \
  --embedding-provider qwen3vl

# LLM 消费者模式（不返回图片）
zoterorag search-vector "query" \
  --mode text \
  --consumer llm_text \
  --image-return none
```

**多模态搜索（文本+图片）：**

```bash
# 纯文本查询 + 多模态 profile
zoterorag search-vector "cell signaling pathway" \
  --mode multimodal \
  --embedding-provider qwen3vl

# 文本 + 图片文件查询
zoterorag search-vector "similar figures to this" \
  --mode multimodal \
  --query-image-file /path/to/query_image.png \
  --consumer manual \
  --image-return file_ref

# 文本 + base64 图片查询
zoterorag search-vector "similar to this diagram" \
  --mode multimodal \
  --query-image-base64 $(base64 query.png) \
  --consumer llm_multimodal \
  --image-return base64 \
  --max-images 3 \
  --max-image-bytes 524288
```

### 5.4 Consumer 与 Image Return 规则

| consumer | 含义 | 图片行为 |
|----------|------|---------|
| `manual` | 人工查看 | 返回本地文件路径 |
| `llm_text` | 纯文本 LLM | **强制无图片**（最安全） |
| `llm_multimodal` | 多模态 LLM | 按 `image_return` 参数返回 |

| image_return | 含义 | 限制 |
|-------------|------|------|
| `file_ref` | 返回本地安全文件路径或 MCP resource id | 不返回 base64 |
| `base64` | 返回 base64 编码图片 | 受 `max_images`、`max_bytes` 限制 |
| `none` | 只返回图注和文本上下文 | 无图片数据 |

**输出安全保证：**
- 纯文字检索（`mode=text`）**永不**返回图片文件/base64/图片 URL
- 图片命中仅以文本形式报告：`has_images=true`, `image_count`, 图注文本
- `consumer=llm_text` 强制降级为纯文字输出

---

## 6. HTTP API

### 6.1 启动服务

```bash
# 默认 127.0.0.1:8765
zoterorag serve

# 自定义地址和端口
zoterorag serve --host 0.0.0.0 --port 9999

# 仅做预检（不启动服务）
zoterorag serve --check
```

### 6.2 端点列表

#### 系统

| 方法 | 路径 | 鉴权 | 说明 |
|------|------|------|------|
| GET | `/health` | 无 | 健康检查 |
| GET | `/status` | 有 | 运行时状态 |
| GET | `/diagnostics` | 有 | 非侵入式诊断 |
| GET | `/providers/status` | 有 | 外部 API 配置状态 |

#### 模型管理

| 方法 | 路径 | 鉴权 | 说明 |
|------|------|------|------|
| GET | `/models/embedding` | 有 | 列出 embedding 模型 |
| POST | `/models/embedding/activate` | 有 | 设置默认模型 |

#### 向量索引

| 方法 | 路径 | 鉴权 | 说明 |
|------|------|------|------|
| GET | `/vectors` | 有 | 列出向量索引 |
| GET | `/vectors/{profile_name}/verify` | 有 | 校验索引完整性 |

#### 建库控制

| 方法 | 路径 | 鉴权 | 说明 |
|------|------|------|------|
| POST | `/scan` | 有 | 影子复制 + 扫描分类 |
| POST | `/ingest/start` | 有 | 创建/执行建库作业 |
| POST | `/ingest/pause` | 有 | 暂停作业 |
| POST | `/ingest/resume` | 有 | 恢复作业 |
| POST | `/ingest/cancel` | 有 | 取消作业 |

#### 审核

| 方法 | 路径 | 鉴权 | 说明 |
|------|------|------|------|
| POST | `/review/include` | 有 | 手动纳入 |
| POST | `/review/exclude` | 有 | 手动排除 |
| GET  | `/review` | 有 | 查看审核规则和候选项 |
| GET  | `/review/explain/{attachment_key}` | 有 | 分类解释 |

#### 文档和附件

| 方法 | 路径 | 鉴权 | 说明 |
|------|------|------|------|
| GET | `/attachments` | 有 | 列出附件 |
| GET | `/documents` | 有 | 列出文档 |
| GET | `/documents/{document_id}` | 有 | 文档详情（含 chunk） |

#### 搜索

| 方法 | 路径 | 鉴权 | 说明 |
|------|------|------|------|
| POST | `/search/metadata` | 有 | 元数据搜索 |
| POST | `/search/fulltext` | 有 | 全文搜索 |
| POST | `/search/text` | 有 | 文本向量搜索 |
| POST | `/search/multimodal` | 有 | 多模态向量搜索 |

#### 备份

| 方法 | 路径 | 鉴权 | 说明 |
|------|------|------|------|
| POST | `/backup/create` | 有 | 创建备份 |
| GET  | `/backup/list` | 有 | 列出备份 |
| POST | `/backup/restore-plan` | 有 | 备份恢复计划 |

#### 作业和进度

| 方法 | 路径 | 鉴权 | 说明 |
|------|------|------|------|
| GET | `/jobs` | 有 | 列出作业 |
| GET | `/jobs/{job_id}` | 有 | 作业详情 |
| GET | `/progress` | 有 | 详细进度 |
| GET | `/extract/jobs` | 有 | 提取作业列表 |
| GET | `/extract/recovery-plan` | 有 | 提取恢复计划 |
| GET | `/normalize/artifacts` | 有 | 标准化产物列表 |
| GET | `/normalize/chunks/{document_id}` | 有 | 文档 chunk 列表 |
| GET | `/embed/batches` | 有 | 嵌入批次列表 |

#### 分步操作

| 方法 | 路径 | 鉴权 | 说明 |
|------|------|------|------|
| POST | `/embed/index-normalized` | 有 | 索引单个标准化文档 |
| POST | `/reembed/plan` | 有 | 重建计划 |
| POST | `/reembed/from-normalized` | 有 | 执行向量重建 |

### 6.3 API 调用示例

```bash
# 健康检查
curl http://127.0.0.1:8765/health

# 状态查询
curl -H "Authorization: Bearer $ZOTERORAG_API_TOKEN" \
  http://127.0.0.1:8765/status

# 文本搜索
curl -X POST http://127.0.0.1:8765/search/text \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "pollen tube attraction mechanism",
    "top_k": 10,
    "consumer": "llm_text"
  }'

# 多模态搜索
curl -X POST http://127.0.0.1:8765/search/multimodal \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "similar cell structure",
    "query_image": {"type": "file_path", "value": "/path/to/image.png"},
    "consumer": "llm_multimodal",
    "image_return": "file_ref",
    "max_images": 3
  }'

# 启动建库
curl -X POST http://127.0.0.1:8765/ingest/start \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "incremental",
    "include_multimodal": true,
    "execute": true
  }'
```

---

## 7. MCP 工具

### 7.1 工具列表

| 工具名 | 用途 | 图片安全 |
|--------|------|---------|
| `zotero_rag_status` | 运行时状态和进度 | N/A |
| `zotero_rag_list_models` | 列出 embedding 模型 | N/A |
| `zotero_rag_metadata_search` | 元数据搜索 | 强制无图片 |
| `zotero_rag_search_text` | 多源文本搜索（元数据+全文+向量） | 强制无图片 |
| `zotero_rag_search_multimodal` | 多模态向量搜索 | 默认无图片 |
| `zotero_rag_get_document` | 获取文档详情 | 默认无图片 |

### 7.2 安全规则

- **`zotero_rag_search_text`**：始终使用 `consumer=llm_text`, `image_return=none`，**决不**返回图片内容
- **`zotero_rag_search_multimodal`**：默认 `consumer=llm_text`（无图片）；仅当显式声明 `consumer=llm_multimodal` 且请求 `file_ref` 或 `base64` 时才返回图片
- **`zotero_rag_get_document`**：默认 `consumer=llm_text`，移除图片 chunk
- **错误处理**：Rerank 错误作为警告返回而非崩溃；向量索引不可用时返回空结果 + 警告

### 7.3 多源搜索 (`zotero_rag_search_text`)

此工具同时查询三个来源并标记出处：
- **metadata** — Zotero 条目元数据（标题/作者/摘要）
- **fulltext** — 标准化全文 chunk（SQLite LIKE）
- **text_vector** — 文本向量搜索（embedding API）

每个结果标注 `retrieval_source` 字段。

---

## 8. 备份与恢复

### 8.1 创建备份

```bash
# 快照备份（配置、state.sqlite、规则、manifest信息）
zoterorag backup create --mode snapshot --out backups/

# 全量备份（快照 + MinerU缓存 + 标准化产物 + embedding缓存 + 向量库）
zoterorag backup create --mode full --out D:/Backup/ZoteroRAG/
```

### 8.2 管理与恢复

```bash
# 列出备份
zoterorag backup list

# 校验备份完整性
zoterorag backup verify --backup <BACKUP_ID>

# 查看恢复计划（不执行）
zoterorag backup restore --backup <BACKUP_ID>

# 执行恢复（自动创建 pre-restore 快照）
zoterorag backup restore --backup <BACKUP_ID> --confirm
```

**恢复保障：** 恢复前自动创建当前状态的 pre-restore snapshot，可回滚。

---

## 9. 监控与诊断

### 9.1 状态查看

```bash
# 全局摘要
zoterorag status

# 详细进度（含每个阶段的计数）
zoterorag progress

# 不带 ingest plan 的进度
zoterorag progress --no-ingest-plan
```

进度报告包含：
- **全局**：总文档数、待处理、已完成、失败、review、跳过
- **文档级**：当前阶段、页数、图片数、chunk 数
- **MinerU**：key alias、batch id、状态、poll 次数
- **Embedding**：profile、文本/图片 batch、成功/失败
- **Index**：后端类型、写入行数、校验结果

### 9.2 诊断

```bash
# 非侵入式诊断
zoterorag doctor

# 含向量索引校验
zoterorag doctor --verify-vectors

# 含向量存储自检（在临时目录测试）
zoterorag doctor --self-test-vector-store
```

诊断项：
- state.sqlite 可读写
- Zotero 源路径存在
- 影子库可读
- 向量索引注册状态
- 外部 API 连通性
- Provider 配置完整性

### 9.3 Provider 检查

```bash
zoterorag providers status
# 输出 MinerU key 数量、冷却状态
# 输出 Qwen embedding 是否已配置
```

---

## 10. 典型工作流

### 10.1 首次建库（260篇PDF）

```bash
# 1. 初始化
zoterorag init-state

# 2. 检查配置
zoterorag providers status
zoterorag doctor

# 3. 扫描 Zotero（纯本地，约1分钟）
zoterorag scan

# 4. 快速浏览审核项
zoterorag review list

# 5. 处理关键审核项
zoterorag review explain --attachment-key <KEY>
zoterorag review include --attachment-key <KEY> --reason "唯一版本"
zoterorag review exclude --attachment-key <KEY> --reason "低质量机翻"

# 6. 预览计划
zoterorag ingest start --mode incremental
# 输出：260篇 extract pending

# 7. 执行建库（调用外部API，可能需要数小时）
zoterorag ingest start --mode incremental --execute

# 8. 监控进度（另一个终端）
zoterorag progress
zoterorag status

# 9. 中断后恢复
zoterorag ingest resume --job-id <JOB_ID>

# 10. 校验
zoterorag vectors verify --profile qwen3vl_cloud_2560_text
zoterorag vectors verify --profile qwen3vl_cloud_2560_multimodal

# 11. 备份
zoterorag backup create --mode full --out backups/
```

### 10.2 日常增量更新

```bash
zoterorag scan                         # 检测新增/变更
zoterorag ingest start --mode incremental --execute
```

### 10.3 切换 Embedding 模型

```bash
# 1. 在 config.toml 中添加新 profile
# 2. 初始化（注册新 profile）
zoterorag init-state

# 3. 预览重建计划
zoterorag reembed --from-normalized --profile new_model --plan-only

# 4. 执行重建（只重算向量，不重新调 MinerU）
zoterorag reembed --from-normalized --profile new_model --execute

# 5. 切换默认模型
zoterorag models activate --profile new_model --mode text

# 6. 验证
zoterorag vectors verify --profile new_model
```

### 10.4 测试与调试

```bash
# 使用存根 provider，不调用外部 API
zoterorag embed index-normalized --profile qwen3vl_cloud_2560_text --document-id DOC001
zoterorag search-vector "test" --mode text --embedding-provider stub
zoterorag reembed --from-normalized --profile qwen3vl_cloud_2560_text --execute --allow-stub-provider

# 限制范围
zoterorag scan --limit 5
zoterorag ingest start --zotero-key <SINGLE_KEY> --execute
```

---

## 11. 架构说明

### 11.1 核心设计原则

1. **Zotero 只读**：所有操作通过影子副本（`data/shadow/zotero.sqlite`）进行，绝不写入 Zotero 主库
2. **断点可恢复**：每个阶段完成后写入 checkpoint，中断后从最远安全点继续
3. **API 去重**：通过 hash 检测已缓存的 MinerU 批处理和 Embedding batch，避免重复调用
4. **单写入器**：SQLite 通过 WAL 模式和单 writer 线程保证一致性
5. **Provider 抽象**：MinerU、Embedding、VectorDB 均为可替换实现

### 11.2 数据流

```
Zotero (只读源)
  │
  ▼
shadow DB ──→ scan/classify ──→ attachments (state DB)
                                    │
                                    ▼
                              extract_jobs ──→ MinerU API ──→ extract_cache/
                                    │
                                    ▼
                              normalized_artifacts ──→ normalize ──→ normalized/
                                    │
                                    ▼
                              chunks ──→ Qwen API ──→ vector_store/
                                    │
                                    ▼
                              vector_indexes ──→ search APIs
```

### 11.3 关键抽象

| 层 | 接口 | 实现 |
|----|------|------|
| Extractor | `ExtractorProvider` (Protocol) | `MinerUProvider`, `StubExtractorProvider` |
| Embedding | `EmbeddingProvider` (Protocol) | `Qwen3VLEmbeddingProvider`, `StubEmbeddingProvider` |
| Vector Store | `open_vector_store()` 工厂 | `LocalVectorStore` (SQLite), `LanceDBVectorStore` |
| Search | `search_vector_index()` | 统一接口，按 profile 路由 |
| Rerank | `RerankProvider` (预留) | 暂不实现 |

### 11.4 向量库后端

| 后端 | 文件 | 适用场景 |
|------|------|---------|
| `sqlite-local` | `data/vector_store/<profile>/vectors.sqlite` | 默认，零依赖，适合中小型库 |
| `lancedb` | `data/vector_store/<profile>/` (目录) | 生产环境，需 `pip install lancedb` |

通过 `open_vector_store(path, backend="sqlite-local"|"lancedb")` 工厂函数统一创建。
