"""Sinh test case tự động cho đề bài chỉ có câu hỏi (không có test).

Quy trình: mô hình tham chiếu viết lời giải Python + danh sách input -> chạy lời giải để lấy output kỳ vọng.
Nếu đề có sẵn ví dụ Input/Output trong tài liệu, lời giải tham chiếu BẮT BUỘC phải khớp các ví dụ đó,
nếu không bài bị loại (tránh làm "đáp án sai" trở thành chuẩn chấm).
Test tự sinh chỉ đáng tin tương đối: kết quả pass@k phụ thuộc vào mô hình tham chiếu.
"""
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import llm, sandbox
from .extraction import parse_json_list  # noqa: F401  (dùng lại tiện ích JSON)

TEST_PROMPT = """You are preparing an automatic grader for a programming problem.

Problem:
{question}

Do two things:
1. Write a CORRECT reference solution in Python 3 that reads all input from stdin and prints the result to stdout in exactly the required format.
2. Write {count} diverse test inputs as a JSON array of strings. Each string is the exact stdin text (use \\n for newlines and end with a trailing newline). Include edge cases (minimum/maximum values, empty-ish cases, duplicates) and keep each input small enough to run instantly.
{examples}
Output format, exactly two code blocks and nothing else:
```python
<reference solution>
```
```json
["input 1", "input 2"]
```
"""
MAX_INPUT_CHARS = 5000


def _example_hint(tests):
    if not tests:
        return ""
    shown = json.dumps([t["input"] for t in tests[:3]], ensure_ascii=False)
    return f"The problem statement already contains sample inputs; include them too: {shown}\n"


def _parse_reply(text):
    text = llm.strip_thinking(text)
    blocks = re.findall(r"```([\w+#-]*)[ \t]*\r?\n(.*?)```", text, re.DOTALL)
    code = next((b for tag, b in blocks if tag.lower() in ("python", "py", "python3")), "")
    raw_json = next((b for tag, b in blocks if tag.lower() == "json"), "")
    inputs = []
    try:
        data = json.loads(raw_json) if raw_json else []
        inputs = [x if isinstance(x, str) else str(x) for x in data] if isinstance(data, list) else []
    except json.JSONDecodeError:
        pass
    return code.strip(), inputs


def generate_tests(item, provider, count=8, timeout=10):
    """Trả về (tests, reference_code, lỗi). tests = [{"input","output"}] (gồm cả ví dụ gốc của đề nếu có)."""
    examples = item.get("tests") or []
    prompt = TEST_PROMPT.format(question=item["prompt"], count=count, examples=_example_hint(examples))
    last = ""
    for _ in range(2):  # thử lại một lần nếu định dạng sai
        reply, err = llm.generate(provider, prompt, temperature=0.2, timeout=300, max_tokens=4096)
        if err:
            return [], "", err
        code, inputs = _parse_reply(reply)
        if not code or not inputs:
            last = "Reference model did not return a solution and a JSON list of inputs."
            continue
        seen, uniq = set(), []
        for x in inputs:
            x = x.replace("\r\n", "\n")
            if len(x) <= MAX_INPUT_CHARS and x.strip() and x not in seen:
                seen.add(x)
                uniq.append(x if x.endswith("\n") else x + "\n")
        if examples:  # lời giải tham chiếu phải khớp các ví dụ có sẵn của đề
            check = sandbox.run_tests(code, "python", examples, timeout=timeout)
            if not check["passed"]:
                last = f"Reference solution contradicts the examples in the document ({check['status']}: {check['message']})."
                continue
            uniq = [x for x in uniq if x not in {e["input"] for e in examples}]
        outs, run_err = sandbox.run_outputs(code, "python", uniq, timeout=timeout)
        if run_err:
            last = f"Reference solution failed: {run_err}"
            continue
        tests = list(examples) + [{"input": i, "output": o} for i, o in zip(uniq, outs)]
        if tests:
            return tests, code, ""
        last = "No usable test inputs."
    return [], "", last


def add_tests(items, provider, count=8, workers=3, progress_cb=None):
    """Bổ sung test cho các bài chưa có test. Trả về (bài đã có test, {id: lý do loại}).
    Bài vốn đã có test được giữ nguyên."""
    ready, failed = [], {}
    todo = [it for it in items if not it["tests"]]
    done = 0
    lock = threading.Lock()
    results = {}
    if todo:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futs = {pool.submit(generate_tests, it, provider, count): it for it in todo}
            for fut in as_completed(futs):
                it = futs[fut]
                results[it["id"]] = fut.result()
                with lock:
                    done += 1
                if progress_cb:
                    progress_cb(done, len(todo), it["title"])
    for it in items:
        if it["tests"]:
            ready.append(it)
            continue
        tests, code, err = results.get(it["id"], ([], "", "skipped"))
        if tests:
            new = dict(it)
            new["tests"] = tests
            new["auto_tests"] = True
            new["solutions"] = dict(it["solutions"]) or {"python": code}
            ready.append(new)
        else:
            failed[it["id"]] = err
    return ready, failed
