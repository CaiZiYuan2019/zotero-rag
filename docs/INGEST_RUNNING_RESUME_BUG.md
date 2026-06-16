# Ingest 断点续传：running extract job 未恢复导致 normalize 失败

> 记录时间：2026-06-16
> 修复时间：2026-06-16
> 状态：已修复

## 现象

在 `ingest start --mode full --execute` 运行过程中，部分文献在 extract 阶段报：

```text
extract: running (cache_hit=True)
-> FAILED: no completed extract job for <attachment_key>; run extraction first
```

受影响实例：

| attachment_key | external_job_id | state | local_stage | submitted_at | last_poll_at |
|---|---|---|---|---|---|
| `3AVY5BJD` | `d19b9023-7089-494f-8f46-7fe3aa3a53d5` | `running` | `poll` | 2026-06-16T04:13:58Z | 2026-06-16T04:14:32Z |

当前数据库中共有 **3 条** `state='running' AND local_stage='poll'` 的 extract job。

## 根因

### 1. `ensure_extraction()` 对缓存的 running job 直接返回

`src/zoterorag/extractors/manager.py:91-113`：

```python
existing = self.ledger.get_extract_job_by_cache_key(cache_key)
if existing is not None and existing["state"] in REUSABLE_EXTRACT_STATES:
    return ExtractionResult(job=existing, cache_hit=True)
```

`REUSABLE_EXTRACT_STATES = {"submitted", "running", "completed", "downloaded"}`。  
因此，当某个 PDF 已经有一个 `running` 的缓存 job 时，`ensure_extraction()` 不会重新 submit，也不会继续 poll，而是直接把该 job 返回给调用方。

### 2. `_execute_extract_stage()` 未恢复 running job 就标为 done

`src/zoterorag/pipeline/ingest.py:542-606`：

- 调用 `ensure_extraction()` 拿到 `state=running` 的缓存 job；
- 仅对 `state == "completed" AND local_stage != "downloaded"` 调用 `resume_extraction()`；
- 对 `submitted`/`running` 状态没有任何恢复逻辑；
- 直接 checkpoint 为 `"extract": "done"`。

### 3. normalize 阶段要求已完成/downloaded

`src/zoterorag/pipeline/ingest.py:621-626`：

```python
extract_jobs = [
    j for j in ledger.list_extract_jobs(limit=None)
    if j.get("attachment_key") == attachment_key and j["state"] in DONE_EXTRACT_STATES
]
if not extract_jobs:
    raise ValueError(f"no completed extract job for {attachment_key}; run extraction first")
```

`DONE_EXTRACT_STATES = {"downloaded", "normalized"}`。  
由于 extract 实际还是 `running`，normalize 找不到已完成 job，抛出错误。

## 触发条件

- 第一次运行：`ensure_extraction()` 走 `submit → poll 循环 → download`，正常情况会到 `downloaded`；
- 如果在 poll 循环完成前进程被中断（Ctrl+C、崩溃、超时异常未被正确捕获等），数据库里会留下 `state=running, local_stage=poll`；
- 再次运行 `ingest start`：`ensure_extraction()` 命中缓存，返回 `running`，pipeline 误标 done，normalize 失败。

所以这是 **断点续传/恢复场景** 的 bug，单次不间断运行不会触发。

## 预期修复方案

### 方案 A：在 extract 阶段主动 resume running job（推荐）

在 `_execute_extract_stage()` 中，调用 `ensure_extraction()` 后增加：

```python
# 现有逻辑
if result.job["state"] == "completed" and result.job.get("local_stage") != "downloaded":
    result = extract_manager.resume_extraction(result.job["job_id"])

# 新增：缓存 job 还在 running/submitted 时，继续 poll 到完成/失败/超时
if result.job["state"] in {"submitted", "running"}:
    result = extract_manager.resume_extraction(result.job["job_id"])
```

`resume_extraction()` 内部已通过 `classify_extract_job()` 识别为 `action="poll"`，会调用 `_resume_poll()` 继续轮询，直到：

- 任务完成 → 继续 download → 返回 `downloaded`；
- 任务失败 → 标记 `failed_retryable` 并抛异常；
- 超时 → 抛 `MinerUAPIError`。

### 方案 B：不阻塞当前文档，但不要把 extract 标成 done

如果担心 resume poll 阻塞整个 pipeline，可以改为：

```python
if result.job["state"] not in DONE_EXTRACT_STATES:
    ledger.checkpoint(attachment_key, "extract", "blocked", {
        "reason": f"extract job still {result.job['state']}",
        "extract_job_id": result.job["job_id"],
    })
    return
```

这样该文献会被跳过，下次 `ingest start` 再试。

### 建议最终方案

采用 **A + 兜底 B**：

1. `submitted`/`running` 时主动 `resume_extraction()`；
2. resume 后若仍不在 `DONE_EXTRACT_STATES`，则 checkpoint 为 `blocked` 而不是 `done`。

## 临时绕过方法（修复前）

如果不想现在改代码，可以手动清除卡住的 extract job 记录，让下次运行时重新 submit：

```bash
python - <<'PY'
import sqlite3
conn = sqlite3.connect('data/state/state.sqlite')
cur = conn.cursor()
# 删除指定 attachment_key 的 running job，或全部 running job
# cur.execute("DELETE FROM extract_jobs WHERE attachment_key = '3AVY5BJD';")
cur.execute("DELETE FROM extract_jobs WHERE state = 'running' AND local_stage = 'poll';")
conn.commit()
conn.close()
PY
```

> 注意：这会丢弃已提交的 MinerU 远端任务信息。如果 MinerU 服务端其实还在处理，会重新提交并扣费。

## 修复后的单篇重试流程

修复并重新运行后，对失败的单篇文献执行：

```bash
zoterorag ingest start --mode full --execute --zotero-key <attachment_key>
```

这会走完整流程：`MinerU extract → normalize → text embed → multimodal embed`。

当前数据库中需要重试的 attachment key 可通过以下命令获取：

```bash
zoterorag extract jobs --state running
# 或
zoterorag extract recovery-plan --state running
```

## 相关文件

- `src/zoterorag/pipeline/ingest.py` — `_execute_extract_stage()`、`DONE_EXTRACT_STATES`
- `src/zoterorag/extractors/manager.py` — `ensure_extraction()`、`resume_extraction()`、`REUSABLE_EXTRACT_STATES`
- `src/zoterorag/extractors/recovery.py` — `classify_extract_job()`
- `data/state/state.sqlite` — `extract_jobs` 表记录当前状态


## 修复记录

### 代码改动

1. **`src/zoterorag/pipeline/ingest.py`** — `_execute_extract_stage()`
   - 在已有的 `completed` 恢复下载逻辑之后，增加对 `submitted`/`running` 缓存任务的 `resume_extraction()` 调用。
   - 增加安全兜底：若 resume 后仍不在 `DONE_EXTRACT_STATES`，将 extract stage checkpoint 为 `blocked` 并直接返回，避免误进 normalize。

2. **`src/zoterorag/extractors/manager.py`** — `_resume_poll()`
   - 由单次 poll 改为循环轮询，直到 `completed`/`failed`/超时。
   - 每次轮询更新 ledger 中的 `state`、`local_stage`、`last_poll_at`。
   - 完成则进入 `_resume_download()`；失败则标记 `failed_retryable` 并抛出 `MinerUAPIError`。

3. **`tests/test_ingest_pipeline.py`**
   - 新增 `TwoPhaseStubExtractorProvider`：第一次 poll 返回 `running`，第二次返回 `completed`。
   - 新增 `test_running_extract_job_is_resumed_and_completes`：模拟中断后遗留的 running job，验证完整 resume → download → normalize → embed 流程成功。

### 验证结果

- `python -m pytest tests/test_ingest_pipeline.py tests/test_extract_recovery.py -q` ⇒ **12 passed**
- `python -m pytest tests/ -q` ⇒ **238 passed, 1 warning**

### 对现有数据的影响

- 修改只影响 extract stage 的执行逻辑。
- 已 `downloaded`/`normalized` 的文献在 plan 阶段即被标记为 `complete`，不会被重新处理。
- 已存在的 normalized artifacts、chunks、embedding batches、vector indexes 不会被修改或删除。
- 对 `running` job 的恢复使用原 `external_job_id`，不会重复提交 MinerU 任务。
