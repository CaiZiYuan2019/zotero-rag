# Ingest 中断后状态 stuck 修复计划

## 问题
`zoterorag ingest start --execute` 被 Ctrl+C 中断后，`pipeline_jobs` 中对应 ingest job 的 `status` 仍保持 `running`，导致下次启动报 "an ingest job is already running"。

根因：`start_ingest_job()` 在 `src/zoterorag/pipeline/ingest.py:160` 设置 `running`，但文档处理循环只捕获 `Exception`，未捕获 `KeyboardInterrupt`（`BaseException`）。因此 Ctrl+C 穿透函数，第 221-232 行的最终状态设置代码未执行。

## 修改范围
仅修改 `src/zoterorag/pipeline/ingest.py` 中 `start_ingest_job()` 的执行路径。

## 具体改动
1. 在 `ledger.set_job_status(job_id, "running")` 之后，把文档循环和最终状态结算整体包入 `try/except KeyboardInterrupt/except Exception/finally`。
2. `KeyboardInterrupt`：
   - 记录 `complete/interrupted` job event；
   - 设置 job status 为 `interrupted`；
   - 重新抛出异常，让 CLI 仍以非零退出码结束。
3. `Exception`（非预期整体失败）：
   - 记录 `complete/failed` job event 并带 traceback；
   - 设置 job status 为 `failed`；
   - 重新抛出。
4. 正常完成路径保持不变，仍设置 `completed` 或 `completed_with_errors`。
5. 保留原有 per-document `except` 行为，单篇失败不应触发整体失败路径。

## 测试补充
- 在 `tests/test_ingest_pipeline.py` 新增测试：模拟 `_execute_document_stages` 抛出 `KeyboardInterrupt`，断言 ingest job 最终状态为 `interrupted` 且添加了事件。
- 运行 `pytest tests/test_ingest_pipeline.py` 与 `pytest tests/` 验证无回归。

## 文档
- 更新 `docs/USAGE.md` 第 4.4 节“断点续跑”，说明 Ctrl+C 后 ingest job 会进入 `interrupted` 状态，可用 `ingest resume` 或重新 `ingest start` 继续。

## 提交
- 修改前：确认当前工作区 clean 并已提交。
- 修改后：代码 + 测试 + 文档一起本地提交。
