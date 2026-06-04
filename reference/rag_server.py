import os
import sys
import json
import time
import shutil
import sqlite3
import subprocess
import threading
import uuid
import traceback
from datetime import datetime
import customtkinter as ctk
import requests
import lancedb
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ================= 绝对路径全局配置 =================
ZOTERO_DB_PATH = r"E:\ZoteroLib\lib\zotero.sqlite"
ZOTERO_STORAGE_DIR = r"E:\ZoteroLib\lib\storage"

RAG_ROOT = r"E:\ZoteroRAG"
SHADOW_DB_PATH = os.path.join(RAG_ROOT, r"lib\shadow_db\zotero_shadow.sqlite")
MINERU_OUT_DIR = os.path.join(RAG_ROOT, r"lib\mineru_output")
VECTOR_DB_PATH = os.path.join(RAG_ROOT, r"lib\vector_store")
SYNC_STATE_PATH = os.path.join(RAG_ROOT, r"script\sync_state.json")

# 本地 Embedding API (server.py)
EMBEDDING_API_URL = "http://127.0.0.1:7997/v1/embeddings"
MINERU_ENV_PATH = r"E:\MinerU"
# ====================================================

class TextRedirector:
    """将标准输出/错误重定向至 GUI Textbox"""
    def __init__(self, widget):
        self.widget = widget

    def write(self, str_data):
        self.widget.insert(ctk.END, str_data)
        self.widget.see(ctk.END)

    def flush(self):
        pass


class ZoteroIngestManager:
    """后台建库核心调度类"""
    def __init__(self, log_func):
        self.log = log_func
        self._stop_event = threading.Event()
        self._init_directories()

    def _init_directories(self):
        """初始化物理目录结构"""
        for path in [
            os.path.dirname(SHADOW_DB_PATH),
            MINERU_OUT_DIR,
            VECTOR_DB_PATH,
            os.path.dirname(SYNC_STATE_PATH)
        ]:
            os.makedirs(path, exist_ok=True)

    def _get_last_sync_time(self) -> str:
        if os.path.exists(SYNC_STATE_PATH):
            try:
                with open(SYNC_STATE_PATH, 'r', encoding='utf-8') as f:
                    return json.load(f).get("last_sync_time", "1970-01-01 00:00:00")
            except Exception:
                pass
        return "1970-01-01 00:00:00"

    def _set_last_sync_time(self, current_time: str):
        with open(SYNC_STATE_PATH, 'w', encoding='utf-8') as f:
            json.dump({"last_sync_time": current_time}, f)

    def _create_shadow_copy(self):
        self.log("[系统] 正在创建 Zotero 数据库影子拷贝...\n")
        shutil.copy2(ZOTERO_DB_PATH, SHADOW_DB_PATH)
        self.log("[系统] 影子拷贝创建完毕，已实现物理读写隔离。\n")

    def _extract_zotero_items(self, last_sync_time: str) -> list:
        """从影子数据库提取增量 PDF 任务"""
        conn = sqlite3.connect(SHADOW_DB_PATH)
        cursor = conn.cursor()
        
        query = """
        SELECT 
            p.key AS parentKey,
            p.dateModified AS dateModified,
            a_item.key AS attachmentKey,
            ia.path AS relativePath,
            idv.value AS title
        FROM itemAttachments ia
        JOIN items a_item ON ia.itemID = a_item.itemID
        JOIN items p ON ia.parentItemID = p.itemID
        LEFT JOIN itemData id ON p.itemID = id.itemID AND id.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'title')
        LEFT JOIN itemDataValues idv ON id.valueID = idv.valueID
        WHERE ia.contentType = 'application/pdf'
          AND p.dateModified > ?
        ORDER BY p.dateModified ASC
        """
        cursor.execute(query, (last_sync_time,))
        results = cursor.fetchall()
        conn.close()

        tasks = []
        for row in results:
            parent_key, date_mod, att_key, rel_path, title = row
            if not rel_path or not rel_path.startswith("storage:"):
                continue
            
            filename = rel_path.replace("storage:", "")
            abs_pdf_path = os.path.join(ZOTERO_STORAGE_DIR, att_key, filename)
            
            if os.path.exists(abs_pdf_path):
                tasks.append({
                    "parent_key": parent_key,
                    "date_modified": date_mod,
                    "attachment_key": att_key,
                    "pdf_path": abs_pdf_path,
                    "title": title or "Untitled Document"
                })
        return tasks

    def _run_mineru(self, pdf_path: str) -> str:
        """调用外部 MinerU 环境执行 PDF 转换"""
        self.log(f"[MinerU] 正在解析: {os.path.basename(pdf_path)}\n")
        
        cmd_str = f'call conda activate "{MINERU_ENV_PATH}" && mineru -p "{pdf_path}" -o "{MINERU_OUT_DIR}" -m auto --source local'
        
        try:
            result = subprocess.run(
                cmd_str, 
                shell=True,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True, 
                encoding='utf-8',
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            out_folder = os.path.join(MINERU_OUT_DIR, pdf_name)
            
            # 动态遍历寻找 MinerU 产生的 .md 文件
            if os.path.exists(out_folder):
                for root, dirs, files in os.walk(out_folder):
                    for file in files:
                        if file.endswith(".md"):
                            self.log(f"[MinerU] 成功定位 Markdown: {file}\n")
                            return os.path.join(root, file)
                            
            self.log(f"[MinerU 异常] 转换似乎完成，但在 {out_folder} 下未找到 .md 文件\n")
            self.log(f"[MinerU 日志] {result.stderr}\n{result.stdout}\n")
            return None
                
        except Exception as e:
            self.log(f"[MinerU 异常] 子进程调用失败: {str(e)}\n")
            return None

    def _get_embeddings_batch(self, texts: list) -> list:
        """
        【优化 1】调用本地 FastAPI 获取批次向量。
        直接向 server 传递 List[str]，触发底层 PyTorch GPU Batch 加速。
        """
        try:
            # 批量计算耗时较长，增加 timeout
            resp = requests.post(EMBEDDING_API_URL, json={"input": texts}, timeout=120)
            if resp.status_code == 200:
                data = resp.json()["data"]
                # 根据 index 重新排序，严格保证向量和文本段顺序一致
                sorted_data = sorted(data, key=lambda x: x["index"])
                return [item["embedding"] for item in sorted_data]
            else:
                self.log(f"[API 异常] Embedding HTTP {resp.status_code}: {resp.text}\n")
                return None
        except Exception as e:
            self.log(f"[API 异常] 批量 Embedding 请求失败: {str(e)}\n")
            return None

    def stop(self):
        self._stop_event.set()

    def run_pipeline(self, full_rebuild=False, test_mode=False):
        self._stop_event.clear()
        
        try:
            last_sync = "1970-01-01 00:00:00" if full_rebuild else self._get_last_sync_time()
            mode_tag = "测试建库 (仅2篇)" if test_mode else ("强制全量建库" if full_rebuild else "增量更新")
            self.log(f"=== 启动流水线 [{mode_tag}] | 时间戳基线: {last_sync} ===\n")
            
            self._create_shadow_copy()
            tasks = self._extract_zotero_items(last_sync)
            
            if not tasks:
                self.log("[系统] 无增量文献需要处理。\n")
                return
                
            if test_mode:
                tasks = tasks[:2]
                self.log(f"[系统] 测试模式已开启，仅提取前 {len(tasks)} 篇 PDF。\n")
            else:
                self.log(f"[系统] 扫描到 {len(tasks)} 篇待处理/更新的 PDF。\n")
            
            db = lancedb.connect(VECTOR_DB_PATH)
            table_name = "zotero_knowledge_base"
            
            # 【优化 2】扩展片段长度与重叠区
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,    # 增大单个 Chunk 容量，容纳更多学术上下文
                chunk_overlap=250,  # 增大前后重叠，避免将关键实验步骤拦腰截断
                separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""]
            )

            max_date_processed = last_sync

            for idx, task in enumerate(tasks):
                if self._stop_event.is_set():
                    self.log("\n[系统] 接收到中断信号，流水线安全终止。\n")
                    break
                    
                self.log(f"\n---> 处理进度 [{idx+1}/{len(tasks)}]: {task['title'][:30]}...\n")
                
                md_path = self._run_mineru(task['pdf_path'])
                if not md_path:
                    continue
                    
                with open(md_path, 'r', encoding='utf-8') as f:
                    md_text = f.read()
                    
                chunks = text_splitter.split_text(md_text)
                self.log(f"[Chunking] 划分为 {len(chunks)} 个文本片段。\n")
                
                records = []
                # 批次大小：根据显存调整，0.6B 模型设为 16-32 通常能吃满算力
                batch_size = 16 
                
                for i in range(0, len(chunks), batch_size):
                    if self._stop_event.is_set():
                        break
                        
                    batch_chunks = chunks[i:i+batch_size]
                    self.log(f"  - 正在向量化批次 {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} ...\n")
                    
                    vectors = self._get_embeddings_batch(batch_chunks)
                    
                    if vectors and len(vectors) == len(batch_chunks):
                        for chunk, vector in zip(batch_chunks, vectors):
                            records.append({
                                "id": str(uuid.uuid4()),
                                "zotero_key": task['parent_key'],
                                "title": task['title'],
                                "chunk_text": chunk,
                                "vector": vector,
                                "mineru_md_path": md_path
                            })
                    else:
                        self.log(f"[API 异常] 批次 {i//batch_size + 1} 向量化失败，该批次被跳过。\n")

                if self._stop_event.is_set():
                    break

                if records:
                    if table_name in db.table_names():
                        table = db.open_table(table_name)
                        table.delete(f"zotero_key = '{task['parent_key']}'")
                        table.add(records)
                    else:
                        db.create_table(table_name, data=records)
                    self.log(f"[数据库] 成功插入 {len(records)} 条向量记录。\n")
                
                if not test_mode and task['date_modified'] > max_date_processed:
                    max_date_processed = task['date_modified']
                    self._set_last_sync_time(max_date_processed)

            self.log(f"\n=== 流水线执行完毕 | 当前时间戳基线: {max_date_processed} ===\n")

        except Exception as e:
            self.log(f"\n[严重异常] 流水线崩溃:\n{traceback.format_exc()}\n")


class App(ctk.CTk):
    """GUI 控制面板"""
    def __init__(self):
        super().__init__()
        self.title("Agentic RAG - Zotero 后台数据管线")
        self.geometry("750x500")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        self.frame_top = ctk.CTkFrame(self)
        self.frame_top.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        self.btn_incr = ctk.CTkButton(self.frame_top, text="增量更新 (推荐)", command=lambda: self.start_thread(full=False, test_mode=False))
        self.btn_incr.pack(side="left", padx=5, pady=10)
        
        self.btn_test = ctk.CTkButton(self.frame_top, text="测试建库 (仅2篇)", fg_color="teal", hover_color="darkcyan", command=lambda: self.start_thread(full=False, test_mode=True))
        self.btn_test.pack(side="left", padx=5, pady=10)
        
        self.btn_full = ctk.CTkButton(self.frame_top, text="强制全量建库", fg_color="darkred", hover_color="red", command=lambda: self.start_thread(full=True, test_mode=False))
        self.btn_full.pack(side="left", padx=5, pady=10)
        
        self.btn_stop = ctk.CTkButton(self.frame_top, text="中断执行", state="disabled", fg_color="gray", command=self.stop_thread)
        self.btn_stop.pack(side="right", padx=5, pady=10)
        
        self.textbox = ctk.CTkTextbox(self, font=("Consolas", 12))
        self.textbox.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        
        sys.stdout = TextRedirector(self.textbox)
        sys.stderr = sys.stdout
        
        self.manager = ZoteroIngestManager(self.append_log)
        self.worker_thread = None

    def append_log(self, text):
        self.textbox.insert(ctk.END, text)
        self.textbox.see(ctk.END)

    def start_thread(self, full=False, test_mode=False):
        self.btn_incr.configure(state="disabled")
        self.btn_test.configure(state="disabled")
        self.btn_full.configure(state="disabled")
        self.btn_stop.configure(state="normal", fg_color="darkorange")
        
        self.worker_thread = threading.Thread(target=self._run_wrapper, args=(full, test_mode), daemon=True)
        self.worker_thread.start()

    def _run_wrapper(self, full, test_mode):
        self.manager.run_pipeline(full_rebuild=full, test_mode=test_mode)
        self.btn_incr.configure(state="normal")
        self.btn_test.configure(state="normal")
        self.btn_full.configure(state="normal")
        self.btn_stop.configure(state="disabled", fg_color="gray")

    def stop_thread(self):
        self.append_log("\n[系统] 正在发送中断信号，等待当前子任务释放...\n")
        self.manager.stop()

if __name__ == "__main__":
    ctk.set_appearance_mode("Dark")
    app = App()
    app.mainloop()