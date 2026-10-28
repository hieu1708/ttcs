"""Trích xuất bài tập từ văn bản tự do bằng LLM (dùng khi file không theo mẫu và heuristic không đủ)."""
import json
import re

from . import llm
from .documents import _finalize

EXTRACT_PROMPT = """You are a careful data-extraction assistant.
The text below was extracted from a document (PDF/TXT) that contains programming exercises.
Extract EVERY programming exercise that appears in the text and return them as a JSON array.

Each array element must be an object with:
  "title": short title of the exercise,
  "prompt": the complete problem statement, copied faithfully (keep input/output format and examples),
  "solutions": {{ "python": "...", "cpp": "...", "c": "...", "java": "..." }} - ONLY solutions that are literally present in the text (omit languages that are not present; use {{}} if none),
  "tests": [ {{"input": "...", "output": "..."}} ] - ONLY sample input/output pairs literally given in the text (use [] if none).

Rules: never invent solutions or tests; do not summarize the statement; ignore text that is not an exercise.
Return ONLY the JSON array, with no commentary and no markdown fence.
{solution_rule}
TEXT:
\"\"\"
{text}
\"\"\"
"""
_NO_SOLUTION_RULE = "Do not include solutions: always use \"solutions\": {}.\n"


def chunk_text(text, max_chars=9000):
    """Cắt văn bản thành các đoạn ≤ max_chars, ưu tiên cắt ở dòng trống."""
    chunks, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        if cur and len(cur) + len(para) + 2 > max_chars:
            chunks.append(cur)
            cur = ""
        while len(para) > max_chars:
            chunks.append(para[:max_chars])
            para = para[max_chars:]
        cur = (cur + "\n\n" + para) if cur else para
    if cur.strip():
        chunks.append(cur)
    return chunks


def parse_json_list(text):
    """Lấy mảng JSON từ câu trả lời của LLM (chịu được ```json fence và văn bản thừa)."""
    text = re.sub(r"```(?:json)?", "", text or "")
    start = text.find("[")
    if start < 0:
        return []
    for end in range(len(text), start, -1):
        if text[end - 1] != "]":
            continue
        try:
            data = json.loads(text[start:end])
        except json.JSONDecodeError:
            continue
        return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []
    # mảng bị cắt giữa chừng: nhặt từng object đầy đủ
    out, depth, begin = [], 0, None
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            if depth == 0:
                begin = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and begin is not None:
                try:
                    out.append(json.loads(text[begin:i + 1]))
                except json.JSONDecodeError:
                    pass
    return out


def llm_extract(text, provider, with_solutions=True, prefix="llm", max_chunks=15, progress_cb=None):
    """Trả về (danh sách bài đã chuẩn hóa, danh sách lỗi)."""
    chunks = chunk_text(text)
    errors = []
    if len(chunks) > max_chunks:
        errors.append(f"Document is long: only the first {max_chunks} of {len(chunks)} parts were processed.")
        chunks = chunks[:max_chunks]
    raw_items = []
    for i, chunk in enumerate(chunks, 1):
        if progress_cb:
            progress_cb(i - 1, len(chunks))
        prompt = EXTRACT_PROMPT.format(text=chunk, solution_rule="" if with_solutions else _NO_SOLUTION_RULE)
        reply, err = llm.generate(provider, prompt, temperature=0.0, timeout=300, max_tokens=8192)
        if err:
            errors.append(f"Part {i}: {err}")
            continue
        found = parse_json_list(reply)
        if not found:
            errors.append(f"Part {i}: no exercise recognised.")
        raw_items.extend(found)
    if progress_cb:
        progress_cb(len(chunks), len(chunks))
    for raw in raw_items:
        if not with_solutions:
            raw["solutions"] = {}
        raw.pop("id", None)
    return _finalize(raw_items, prefix), errors
