# 待安装环境清单

本文件列出 ZoteroRAG 后续开发和集成测试需要安装的环境与 Python 包。当前只记录安装规划，不自动安装任何依赖，也不触发 MinerU 或 qwen API。

目标 Python 环境：

```powershell
conda activate E:\ZoteroRAG\.conda
python -m pip install --upgrade pip
```

## 1. 当前基础环境

用途：运行 CLI、SQLite 状态库、Zotero shadow copy、标准化、chunk、SQLite 本地向量后端和大多数 `unittest`。

安装命令：

```powershell
python -m pip install -e .
```

说明：

- 当前 `pyproject.toml` 的基础依赖为空，核心控制面尽量使用 Python 标准库。
- SQLite 来自 Python 标准库，不需要单独安装。
- 当前默认向量实现是 `sqlite-local`，不需要外部向量数据库服务。

## 2. MinerU 与 qwen API Provider

用途：真实 MinerU PDF 转换、qwen3-vl-embedding 文本/多模态嵌入请求。

安装命令：

```powershell
python -m pip install -e .[providers]
```

等价展开：

```powershell
python -m pip install "requests>=2.32"
```

涉及模块：

- `src/zoterorag/extractors/mineru.py`
- `src/zoterorag/embeddings/qwen.py`
- `reference/mineru_cli.py`

注意：

- `.env` 中的 `MINERU_URL`、`MINERU_KEY`、`BAILIAN_URL`、`BAILIAN_KEY` 只用于本地测试，不能提交。
- 常规测试必须使用 fake client 或 dry run，不应大规模调用真实 MinerU/qwen。
- MinerU 真实转换 timeout 按 `pages * 6 + 30` 秒预留。

## 3. FastAPI 控制服务

用途：运行 HTTP 控制和检索接口、API route 测试、后续 MCP/LLM 外接服务。

安装命令：

```powershell
python -m pip install -e .[api]
```

等价展开：

```powershell
python -m pip install "fastapi>=0.115" "uvicorn>=0.30" "httpx>=0.27"
```

涉及模块：

- `src/zoterorag/api/app.py`
- `src/zoterorag/api/server.py`
- `tests/test_api_security.py`

说明：

- `fastapi` 是 API 应用本体。
- `uvicorn` 是本地服务启动器。
- `httpx` 被 FastAPI/Starlette 的 `TestClient` 使用。

## 4. PDF 页数与图片处理

用途：真实 PDF 页数读取、MinerU 任务 timeout 估算、embedding 图片降采样、图片格式与尺寸规范化。

安装命令：

```powershell
python -m pip install -e .[pdf]
```

等价展开：

```powershell
python -m pip install "PyMuPDF>=1.24" "Pillow>=10.4"
```

涉及功能：

- 读取 PDF 页数，避免人工传错页数。
- 为 qwen3-vl-embedding 生成受控尺寸的图片副本。
- 检查图片宽高、像素数、文件大小和格式。

注意：

- 原图应保留用于人工查看。
- embedding 图应按 profile 生成，默认最长边和文件大小受 provider 限制约束。

## 5. LanceDB 向量后端

用途：实现规划中的本地嵌入式向量库后端，替代或并存于当前 `sqlite-local` 后端。

安装命令：

```powershell
python -m pip install -e .[lancedb]
```

等价展开：

```powershell
python -m pip install "lancedb>=0.18"
```

建议暂缓条件：

- 在 LanceDB adapter 和迁移/验证测试写好之前，不需要为了当前测试安装。
- 如果后续发现 LanceDB 需要显式 `pyarrow` 版本钉住，再把 `pyarrow>=16` 加入 `pyproject.toml` 的 `lancedb` extra。

## 6. 开发工具

用途：可选的测试/格式检查工具。当前测试套件仍可直接用标准库 `unittest` 运行。

安装命令：

```powershell
python -m pip install -e .[dev]
```

等价展开：

```powershell
python -m pip install "pytest>=8" "ruff>=0.6"
```

说明：

- `pytest` 当前不是必需项。
- `ruff` 用于后续统一 lint/format，但现在还没有强制 CI。

## 7. 下一阶段推荐一次性安装

用于“小规模真实 MinerU 转换 + 少量 qwen embedding 验证 + API 服务验证”的推荐组合：

```powershell
python -m pip install -e ".[api,providers,pdf]"
```

暂不包含：

- `lancedb`：等 LanceDB adapter 实现后再装。
- `dev`：只在需要 `pytest`/`ruff` 工作流时安装。

## 8. 旧 reference 脚本依赖

仅在直接运行旧脚本时需要，不是新项目主路径依赖：

```powershell
python -m pip install customtkinter requests lancedb langchain-text-splitters PyMuPDF
```

涉及文件：

- `reference/rag_server.py`
- `reference/mineru_cli.py`

旧脚本只作为参考，不建议继续扩展其依赖结构。
