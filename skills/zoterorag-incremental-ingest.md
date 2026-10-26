# ZoteroRAG 增量建库与进度监视 Skill

## 描述

使用项目内 `.conda` 环境中的 `zoterorag` CLI 完成**增量建库**（`scan → dry-run 计划 → execute`），并在长时间执行期间用配套监视器脚本跟踪进度、在终态自动汇报。监视器脚本位于同名文件夹：`skills/zoterorag-incremental-ingest/monitor_ingest.py`。

## 适用场景

- Zotero 中新增或修改了文献，需要把变化增量入库；
- 需要先看计划、确认待处理量，再决定是否真正执行；
- 建库耗时较长（MinerU 提取 + 向量化），需要在**另一个上下文**里持续跟踪进度而不阻塞主流程；
- 中断/崩溃后需要判断当前状态并续跑。

## 前置条件

1. 工作目录为 ZoteroRAG 项目根目录（如 `E:/ZoteroRAG`）。
2. `.conda` 环境可用（`.conda/python.exe`）。
3. `.env` 中已配置 `MINERU_KEY`（可多把，逗号分隔）与 `BAILIAN_KEY`。
4. 执行前确认没有其他 ingest/reembed 在跑（runner 锁 `data/state/ingest_runner.lock` 不应存在）。

## 标准流程

### 步骤 0：环境与 key 检查

```bash
.conda/python.exe -m zoterorag providers status
```

确认 MinerU key 数量、冷却状态（`cooldown_remaining_seconds` 应为 0）以及 `BAILIAN_KEY` 已配置。

### 步骤 1：扫描（**必须先做**）

```bash
.conda/python.exe -m zoterorag scan
```

`scan` = 复制 Zotero 影子库 + 扫描全部附件 + 自动分类，只写本地 `state.sqlite`，不调用任何外部 API。

> **关键**：增量计划只反映**上一次 scan 的快照**。不先 scan，计划看不到这期间 Zotero 的任何新增/修改——此时「计划为空」只代表「相对上次扫描已是最新」，不代表与 Zotero 现状一致。

### 步骤 2：dry-run 出计划（不执行）

```bash
.conda/python.exe -m zoterorag ingest start --mode incremental \
  > .temp/ingest_plan_$(date +%Y%m%d).json
```

解析 `plan.summary` 判断待处理量：

```bash
.conda/python.exe -c "
import json
d = json.load(open('.temp/ingest_plan_20260909.json', encoding='utf-8'))
print(d['job']['job_id'], d['job']['status'])
print(json.dumps(d['plan']['summary'], indent=2, ensure_ascii=False))
"
```

判读要点：

- `next_stages: {"complete": N}` ⇒ 没有待处理工作；
- `next_stages: {"extract:pending": M}` ⇒ 有 M 篇需要走完整流水线；
- 阶段语义：`pending` 需执行 / `done` 可跳过 / `blocked` 依赖前置 / `skipped` 已跳过。

**dry-run 会创建一条 `planned` 状态的 job 并持久化**。空计划 job 会累积，可用 `ingest cancel <job_id>` 清理。

### 步骤 3：执行（放后台）

```bash
.conda/python.exe -m zoterorag ingest start --mode incremental --execute \
  > .temp/ingest_execute_$(date +%Y%m%d).json \
  2> .temp/ingest_execute_$(date +%Y%m%d).err
```

执行期间：

- runner 锁被持有，新的 ingest/reembed 会被拒绝；
- stdout（JSON）与 stderr（逐篇进度日志）分别重定向到 `.temp/`；
- 完成后从结果 JSON 读取 `executed` / `skipped` / `blocked` / `failed` 四个数组。

### 步骤 4：挂进度监视器

先从计划或 `jobs list` 拿到 `job_id`，再启动监视器（后台运行）：

```bash
.conda/python.exe skills/zoterorag-incremental-ingest/monitor_ingest.py \
  <JOB_ID> <KEY1> <KEY2> ...
```

**用法**

| 参数 | 说明 |
|---|---|
| `<job_id>` | 必填，`ingest start --execute` 创建/返回的 job id |
| `[attachment_key ...]` | 可选，只跟踪这些附件的 extract 状态；省略则只跟踪 job 状态 |

**环境变量**

| 变量 | 默认 | 说明 |
|---|---|---|
| `ZOTERORAG_STATE_DB` | `file:data/state/state.sqlite?mode=ro` | 状态库（只读打开） |
| `ZOTERORAG_MONITOR_POLL_SECONDS` | `180` | 轮询间隔（秒） |

**输出格式**

```text
job=running extract_done=0/8 states={T5D8HG7T:none, D63TMUF9:none, ...}
job=running extract_done=2/8 states={T5D8HG7T:none, 29H2VU2B:downloaded, ...}
[terminal] ingest job completed
```

只在状态签名变化时输出一行，job 进入终态后打印 `[terminal]` 并退出。

**终态集合**：`completed` / `completed_with_errors` / `failed` / `cancelled` / `interrupted` / `blocked`。

### 步骤 5：终态校验

```bash
# 1) 结果 JSON 的四个数组
.conda/python.exe -c "
import json; d = json.load(open('.temp/ingest_execute_20260909.json', encoding='utf-8'))
print('executed', len(d['executed']), 'skipped', len(d['skipped']),
      'blocked', len(d['blocked']), 'failed', len(d['failed']))
"

# 2) 重跑一次 dry-run，应为 0 pending（闭环验证）
.conda/python.exe -m zoterorag ingest start --mode incremental > .temp/verify.json

# 3) 向量索引增量
.conda/python.exe -m zoterorag vectors list
```

`executed` 各篇的 chunk 数可从 stderr 日志的 `normalize: <KEY>...` 行读取。

## 监视器已知盲区

**按 `attachment_key` 统计，重复附件会漏计。**
若同一份 PDF 在 Zotero 中被导入两次（字节完全相同 ⇒ `pdf_sha256` 相同 ⇒ `cache_key` 相同），MinerU 提取会被缓存去重：两个文档**共享同一批 segment job**，而 job 行只挂在其中一个 `attachment_key` 下。此时监视器会把另一个附件一直显示为 `none`，得出错误的 `7/8`。

**判据**：若终态时某篇仍为 `none`，但 `ingest_execute_*.err` 里该 key 有完整的 `normalize:` / `embed:` 行，则大概率是重复附件，而非失败。可查 `checkpoints` 表的 `extract` 阶段 `segment_job_ids` 确认两个文档是否指向同一批 job。

## 常见坑

| 坑 | 说明 |
|---|---|
| 不 scan 就出计划 | 计划只是上次 scan 的快照；必须先 `scan` |
| `planned` job 累积 | 每次 dry-run 都建一条 job，空计划需手动 cancel |
| 重复附件共享 extract job | 见上文「已知盲区」 |
| 已删除文献的产物不清理 | Zotero 删除后 `scan` 只标记 `deleted`，chunks/normalized/向量仍在且仍可检索，详见 `docs/DATA_HYGIENE_ISSUES_20260909.md` |
| 僵死 `running` job 不回收 | 中断残留的 `running` extract job 绕过 `job_max_lifetime_seconds`，同样见上述文档 |
| 监视器脚本语法 | 脚本内避免 f-string 嵌套同类引号（Python 3.11 会报 `f-string: unmatched '('`），统一用 `%` 格式化 |
| 中断后 | 直接重新 `ingest start --execute` 即可，checkpoint 会跳过已完成文档；锁空闲时残留的 `created`/`running` job 会被自动标为 `interrupted` |

## 相关文档

- `docs/USAGE.md` — 完整 CLI/API 手册（第 4 章建库流程、第 10.2 节日常增量更新）
