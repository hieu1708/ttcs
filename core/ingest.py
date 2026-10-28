"""Nạp câu hỏi / bài tập từ nhiều file PDF/TXT/MD (dùng chung cho giao diện người dùng và quản trị)."""
import hashlib
import re
from pathlib import Path

from . import documents, extraction


def fingerprint(files, *extra):
    h = hashlib.sha256()
    for name, content in files:
        h.update(name.encode("utf-8") + b"\0" + content + b"\0")
    for x in extra:
        h.update(str(x).encode("utf-8") + b"\0")
    return h.hexdigest()


def load_items(files, use_llm=False, provider=None, with_solutions=True):
    """files: [(tên_file, bytes)]. Trả về (items, errors); mỗi item có thêm khoá "source" = tên file.
    use_llm: dùng LLM để trích xuất khi file không theo mẫu [EXERCISE]; nếu LLM không tìm được gì thì quay lại heuristic."""
    items, errors, seen = [], [], set()
    for name, content in files:
        try:
            text = documents.extract_text(name, content)
        except documents.DocumentError as exc:
            errors.append(f"{name}: {exc}")
            continue
        found = []
        if use_llm and provider and not re.search(r"(?im)^\[EXERCISE\]", text):
            prefix = re.sub(r"\W+", "_", Path(name).stem.lower())[:16] or "doc"
            found, errs = extraction.llm_extract(text, provider, with_solutions=with_solutions, prefix=prefix)
            errors += [f"{name}: {e}" for e in errs]
        if not found:
            try:
                found = documents.parse_text_items(text, name, with_solutions=with_solutions)
            except documents.DocumentError as exc:
                errors.append(f"{name}: {exc}")
                continue
        for it in found:
            base, n = it["id"], 2
            while it["id"] in seen:  # id trùng giữa các file
                it["id"] = f"{base}_{n}"
                n += 1
            seen.add(it["id"])
            it["source"] = name
            items.append(it)
    return items, errors
