# ZoteroRAG 未来规划讨论纪要（2026-08-05）

> 本文档记录 2026-08-05 关于项目未来方向的讨论结论，配套审查报告见 [`PROJECT_AUDIT_20260805.md`](PROJECT_AUDIT_20260805.md)。
> 状态：讨论纪要 + 方向建议，未定稿实施。标注「待决策」的条目需用户拍板后再立项。

## 〇、用户实际工作流（一切规划的前提）

- 检索行为全部发生在 agent 会话内：LLM 按 `skills/zoterorag-query.md` 驱动 **CLI** 做检索（元数据 → 全文 → 向量 → 多模态交叉验证），用途集中在「汇总信息」与「精确查找图片/文段」；命中后再开 subagent 精读。
- **MCP 与 HTTP API 目前没有接入方**，也无明确未来需求。
- 管理操作（scan/ingest/进度/review）目前只能用 CLI 长 JSON 输出，体验差。

---

## 一、MCP/API 去留与两个扩展方向

### 结论：MCP/API 功能完善**冻结**，CLI 为唯一主接口

- 不再投入 MCP 工具补齐（审查 R10）、API pydantic 化等完善性工作；现有端点维持可用即可。
- 审查中的 stub 静默问题（审查#1）对 CLI 无影响（skill 流程显式传 `--embedding-provider qwen3vl`），可在 `skills/zoterorag-query.md` 加一句"忘传 provider 会静默得到 stub 结果"的警告作为零成本兜底，代码侧修复降级为低优先级。
- 「MCP 组合检索假融合」「rerank 错误映射」等 MCP/API 专属 bug 同步降级，除非未来出现真实接入方。

### 扩展方向 A：文献发现（找引用文献/库外文献入库）

**核心约束：永远不写 Zotero 本地数据库。** 本项目数据流是 Zotero → shadow 只读拷贝 → ingest，任何外部写入本地 `zotero.sqlite` 都有损坏主库风险，且与"不伤数据库"原则冲突。

可行的三条集成路径（按推荐度排序）：

1. **Zotero Web API 写入（推荐，前提是用户开了 Zotero 同步）**：项目把发现到的文献（DOI/标题/年份）通过官方 Web API 添加到 zotero.org 账户 → Zotero 客户端同步到本地库 → 下次 `zoterorag scan` 自动入库。全程走官方通道，零本地写风险。需要用户提供 Zotero API key + user ID（`.env` 配置）。
2. **发现暂存区 + 人工导入（零风险兜底）**：项目内维护 discovery 队列（新表或 JSON），记录候选文献与来源（哪篇种子文献的引用/被引）；导出为 RIS/BibTeX/DOI 清单，用户在 Zotero 里用「按标识符添加」（magic wand）或文件导入。多一步人工操作，但完全不依赖同步设置。
3. ~~直接写本地 Zotero 库~~：明确排除。

发现能力本身（与入库解耦，可先做）：

- **引用追踪**：从种子文献出发查 references / citations。建议用 OpenAlex / Semantic Scholar / Crossref API（免费、稳定、无反爬），用户已有的谷歌学术接口可作为补充通道（注意 Google Scholar 无官方 API，长期稳定性差）。
- **库内去重**：每个候选先过 `metadata_search`（DOI 精确 + 标题模糊）标注"已在库中/库外"，避免重复推荐。
- 产出形态：`zoterorag discover --from <attachment_key>` CLI 命令 + 结果落库，agent 流程可直接消费（与现有 skill 工作流一致）。

### 扩展方向 B：论文写作引用核查

用户痛点：agent+subagent 逐篇深读能跑通但效率低。效率低的根源是**所有引用都走了最贵的 LLM 精读路径**。建议分层，让 LLM 只处理真正需要判断的少数条目：

- **第 1 层｜参考文献解析（确定性，无 LLM）**：从稿件 PDF/docx 提取参考文献列表。可选 GROBID/anystyle 等专用解析器，或一次性 LLM 抽取后落盘复用。
- **第 2 层｜库内匹配与元数据校验（确定性，无 LLM）**：每条引用过 `metadata_search`：DOI 精确匹配 → 标题模糊匹配。输出三档：`found`（附附件 key）、`not_in_library`、`mismatch`（年份/作者顺序/期刊与库内记录不一致——这是"引用准确性"里最高频的错误类型）。此层一次批量跑完全部引用。
- **第 3 层｜论断支持度核查（LLM，仅对标记条目）**：定位正文引用句 → 用现有检索拿被引文献相关 chunk → LLM 判 支持/曲解/无关。默认只对用户指定章节或第 2 层标记为可疑的条目跑，把 LLM 开销压缩到原来的零头。
- 产出：`zoterorag cite-check <manuscript>` CLI 命令产出结构化报告（每条引用的核查状态 + 证据链接），agent 流程只接力第 3 层。

两个方向的共性：都是「确定性批量预处理 + LLM 只碰短名单」，与现有 agent+skill 工作流无缝衔接，且全程只读现有库。

**待决策**：方向 A 是否具备 Zotero 同步条件（决定走 Web API 还是暂存区）；方向 B 的稿件格式（PDF 还是 Word）与是否需要第 3 层。

---

## 二、抽取层（MinerU / PaddleOCR / 提速）

### 现状确认：provider 分离设计成立

`extractors/base.py` 定义了 provider 接口，`manager.py` 统一管 cache_key 去重、job 生命周期、key pool。`reference/paddleocr_core.py` 是百度 AI Studio 的 PaddleOCR-VL 云端 jobs API 客户端（submit → poll → 下载 jsonl），**与 MinerU 客户端形态完全同构**，接入不存在架构障碍。

### PaddleOCR 接入的的实际工作项

1. 新 provider 类 `extractors/paddleocr.py`：submit/poll/download 逻辑可直接移植 `reference/paddleocr_core.py`（去掉硬编码 token，改走 `providers.py` 的 `.env` 体系）。
2. **输出适配层（主要工作量）**：PaddleOCR 返回 per-page jsonl，结构与 MinerU zip 产物不同，需转换为项目统一的 artifact 契约（merged markdown + images 目录 + 可选 content_list）。归一化层（`normalize/`）不变。
3. provider 选择机制：config 增加 `extract.provider = "mineru" | "paddleocr"`（或按附件类型路由），`ExtractConfig` 相应扩展。
4. 注意 `reference/paddleocr_core.py` 中 token 是硬编码占位，且该服务是第三方云端——敏感文献外传与额度问题需用户确认可接受（与 MinerU 云端的性质相同）。

**待决策**：PaddleOCR 的定位——替代 MinerU、作为失败回退、还是仅用于 MinerU 效果差的特定文献类型（如扫描件/复杂表格）。

### 建库提速与可靠性（独立于换 OCR，全部 A 类）

按审查结果，收益从大到小：

1. **嵌入并发**（审查#20）：qwen 每 chunk 一次串行 HTTP 是当前最大 wall-clock 瓶颈，chunk 级 4-8 线程可 3-10×。
2. **抽取并发**：MinerU key 在 submit 后即释放、poll 再重新 acquire（审查#21）；分段 PDF 改「全部 submit → 逐个 harvest」（审查 F2）；文档级 worker 池（审查 F1，2-4 worker）。
3. **可靠性修复先行**：36 秒续跑超时（审查#10，直接省钱）、分段失败误收历史 job（审查#11）、pause/cancel 协作与 stale 检测（审查#12）——**提速前必须先修#12，否则并发会放大并发写向量库的风险**。
4. 重试策略：`failed_retryable` 的退避重试参数（次数/间隔）入 config；`failed_manual_review` 不再被自动重提。

---

## 三、索引 O(N²) 与版本膨胀：机制详解（对应审查#19）

### 设计初衷

`embeddings/indexer.py:195-208` 的注释写明了动机：**发布必须原子**——检索只能看到完整版本，中断的 embed 不能暴露写了一半的向量集。为此每索引一篇文档都走「staged version → publish 切换」：

```
index_normalized_document(一篇文档):
  1. copy_active_records_to_version(index_version=batch_hash, exclude=[本文档])
     # 把当前 active 版本里【全库】记录复制到以本篇 batch_hash 命名的新版本
  2. upsert(本文档的 chunks, index_version=batch_hash)
  3. publish_version(batch_hash)   # 翻牌：active 指针指向新版本
```

### 为什么这是 O(N²)

逐篇索引 N 篇文档时，第 i 篇要复制当时全库的 (i-1) 篇记录。总复制量 = 0+1+2+…+(N-1) ≈ N²/2 倍单篇体量。库里有 M 个 chunk 后，每新增一篇文献都要把 M 行向量（每行是 2560 维的 JSON 数组，几十 KB）完整重写一遍——建库越到后期越慢，且总写入量随库规模平方增长。

LanceDB 后端更重：`copy_active_records_to_version`（`index/lancedb_vector.py:113`）先把整个 active 表 `to_arrow().to_pylist()` **全量读进内存**，再写成一张新表。

### 为什么磁盘无界增长

版本以**每篇文档的 batch_hash** 命名，意味着每篇文档产生一个版本：

- sqlite-local：所有版本堆在同一张 `vectors` 表（`record_id` 带 `{version}:` 前缀区分），全项目 grep 确认**没有任何 `DELETE FROM vectors`**——旧版本行永久残留。
- LanceDB：每个版本是一整张 `vectors_{batch_hash}` 表，`publish_version` 只翻 `vector_meta` 指针，旧表从不 `drop_table`（`lancedb_vector.py:53` 的 "LanceDB handles cleanup automatically" 注释指的是连接资源，不是表数据）。

即：**每做过一次增量索引/重嵌，磁盘上就永久多一份全库向量的完整副本**。例如 active 向量集 100MB、累计 50 次单篇索引 → 约 5GB 死数据。可用资源管理器直接看 `data/vector_store/` 体积验证。

### 修复方向（分层）

1. **A 类（纯代码，立刻可做）：批量 publish。** ingest 循环内积累多篇文档，一轮 ingest 只做一次 copy-forward + 一次 publish，把 O(N²) 降回 O(N)。不改变版本化语义，现有库直接受益。
2. **B 类（动 data/，先备份）：旧版本清理。** publish 成功后删除非 active 版本（保留最近 1-2 版用于回滚）：sqlite-local 是 `DELETE FROM vectors WHERE index_version NOT IN (...)` + VACUUM；LanceDB 是 `drop_table` 旧版本表。同时提供一次性命令清掉历史存量。
3. **中期（可选）：取消 copy-forward 机制本身。** sqlite-local 的 `vectors` 表本来就有 `index_version` 列和 `(profile_name, modality, index_version)` 索引（`local_vector.py:184-189`），检索完全可以按 active_version 过滤——「完整快照」只是诸多原子性方案中的一种。改为就地 upsert + meta 翻牌可彻底消灭复制，代价是翻牌瞬间的可见性语义需要重新设计（如双缓冲 meta 或短事务），属于机制重构，建议在 1+2 见效后再评估。

### 与其他审查项的联动

- 批量 publish（方向 1）落地后，「LanceDB `publish_version` 非原子」（审查#14 子项）的影响面也随之缩小。
- 检索侧暴力扫描（审查#17）与版本膨胀叠加放大：旧版本越多，`data/vector_store` 越大，但检索只读 active 版本，两者互不加重——**检索慢是独立问题**，靠 numpy 化 + store 缓存解决。

---

## 四、建库管理 GUI

### 结论：推荐「FastAPI 静态 Web 面板」，CLI 保持完整可用

三个形态对比：

| 形态 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| **Web 面板（挂现有 FastAPI）** | 复用全部现有端点与鉴权（loopback + token）；浏览器访问，跨平台；单页 HTML/JS 无需前端工具链；与 CLI 天然共存 | 需要先补「异步 ingest」端点 | **推荐** |
| TUI（textual/rich） | 终端原生、CLI 用户习惯延续 | 仍是文本界面，进度可视化有限；Windows 终端兼容性坑 | 备选 |
| 桌面 GUI（PySide/Tkinter） | 独立窗口 | 打包/依赖最重，与现有架构最不契合 | 不推荐 |

「分离工作」的正确理解：**前端独立、后端共享**——面板只是 FastAPI 下挂的一个静态页面（如 `/ui`），所有操作仍走现有 REST 端点，CLI 与 GUI 操作的是同一份 state，互不冲突。

### MVP 功能集（只覆盖建库管理，不做检索界面）

1. **仪表盘**：附件/chunk/向量索引计数、磁盘占用（含第三节的版本膨胀体积，正好作为 R14 清理的可视化入口）。
2. **任务控制**：ingest 计划预览（plan）→ 启动 → 实时进度条（轮询 `/progress`）→ pause/resume/cancel 按钮。
3. **评审队列**：`needs_review` 列表 + include/exclude 一键操作（替代现在 `review.py` + CLI 的 JSON 交互）。
4. **扫描与日志**：scan 触发 + 结果摘要、job event 流展示。
5. **诊断页**：现有 `diagnostics.py` 结果的可视化。

### 前置依赖（都是审查已列出的 A 类项）

- **异步 ingest 端点**（审查 F11）：当前 `/ingest/start?execute=true` 同步阻塞，HTTP 挂全程；GUI 必须改成「提交即返回 job_id + 后台线程执行 + 状态轮询」。这是 GUI 的地基，也顺带修掉审查#10 子项。
- 审查#12（pause/cancel 协作 + stale 检测）：否则 GUI 上的暂停按钮是摆设。

### 工作量估计

后端调整（异步 ingest + progress 完善）1-2 天；单页面板 MVP 2-4 天；合计约一周，可分两期（先任务控制 + 仪表盘，再评审/诊断）。

**待决策**：确认走 Web 面板路线；是否接受 MVP 不含检索界面（检索继续走 agent+CLI）。

---

## 五、方向优先级建议（汇总）

| 顺序 | 事项 | 理由 |
|------|------|------|
| 1 | 可靠性修复三件套（审查#10/#11/#12）+ 批量 publish（三.1） | 省钱 + 为一切并发/GUI 打底，全部 A 类 |
| 2 | GUI Web 面板 MVP（含异步 ingest 前置） | 用户明确的管理痛点 |
| 3 | 建库提速（嵌入并发 + 抽取并发 + key 释放） | 依赖 #12 修复，wall-clock 收益最大 |
| 4 | 引用核查 CLI（方向 B 第 1-2 层） | 确定性实现、不依赖外部条件，agent 流程立即受益 |
| 5 | 文献发现（方向 A） | 依赖 Zotero 同步条件决策，且需要外部 API 接入工作 |
| 6 | PaddleOCR provider | 依赖定位决策（替代/回退/特定类型） |
| 7 | 旧版本向量清理 R14（B 类，先备份） | 磁盘治理，可在任意时点插入 |
| 冻结 | MCP/API 完善、rerank、HyDE、分块重构（C 类） | 无接入方 / 无需求 / 需重嵌 |

---

## 六、第二轮决策记录（2026-08-05 当日拍板）

1. **建库首要任务（立即执行）**：修复 ① LanceDB/向量库体积无界膨胀（审查#19/R14）与 ② MinerU 超时重复扣费（审查#10）。**保留现有串行逻辑，不做并发等大改动。**
2. **向量库修复方式**：最小修改原则，候选机制为「双槽乒乓」或「publish 成功后立即清理旧版本」，不大动代码结构与 staged→publish 原子语义。已派只读子代理核查全路径（重点是 reused_existing_embedding 跳过逻辑是否依赖旧版本存在）后确认最终方案，再请用户批准实施。
3. **MinerU 修复方式（用户已指定）**：页数未知时用保守默认值（不再默认 1 页）；`_resume_poll` 的 deadline 取合理值（不低于 manager 级默认）；页数已知时正确传递 `selected_page_count`。
4. **引用核查（方向 B）定位调整**：LLM 使用成本**不设限**，效果优先——每个引用处开一个 subagent 复核也可接受（还覆盖引用与上下文文意的配合判断）。认可 LLM 是当前最优方案；可叠加的增强：解析 docx 中 Zotero 插件域数据精确定位引用、增加「文献存真性核查」（被引文献是否真实存在）。定位为**本项目的补充/插件**，后续单独立项设计，不进当前修复批次。
5. **GUI**：暂缓，后续再议（原第四节方案保留备查）。
6. 优先级表相应调整：第 1 位 = 向量库膨胀修复 + MinerU 超时修复（本次）；嵌入/抽取并发提速顺延至其后；其余排序不变。
