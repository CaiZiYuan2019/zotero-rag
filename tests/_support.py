from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import contextmanager
import importlib
import inspect
from pathlib import Path
import shutil
from typing import Any
import unittest
import uuid


ROOT = Path(__file__).resolve().parents[1]
WORK_TMP = Path.home() / ".codex" / "memories" / "zoterorag-test-work"


class OptionalModuleTestCase(unittest.TestCase):
    def import_first_available(self, module_names: Sequence[str]):
        for module_name in module_names:
            try:
                return importlib.import_module(module_name)
            except ModuleNotFoundError:
                continue
        self.skipTest(f"none of the candidate modules exist: {', '.join(module_names)}")

    def get_first_attr(self, obj: Any, names: Iterable[str]) -> Any:
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        available = ", ".join(sorted(name for name in dir(obj) if not name.startswith("_")))
        self.fail(f"none of the candidate attributes exist: {tuple(names)}; available: {available}")

def call_with_known_kwargs(func: Any, /, **kwargs: Any) -> Any:
    signature = inspect.signature(func)
    accepted: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name in kwargs:
            accepted[name] = kwargs[name]
    return func(**accepted)


@contextmanager
def workspace_tmpdir(prefix: str):
    WORK_TMP.mkdir(parents=True, exist_ok=True)
    path = WORK_TMP / f"{prefix}{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
