"""Đánh giá pass@k: sinh n lời giải/bài, chạy test, tính ước lượng không chệch (Chen et al., 2021)."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from . import llm, rag, sandbox, storage
from .prompts import build_prompt


def pass_at_k(n, c, k):
    """pass@k = 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def evaluate(problems, provider, language, n=5, ks=(1, 5), temperature=0.8, rag_mode="none",
             top_k=3, template=None, test_timeout=5, gen_timeout=300, exclude_self=True,
             progress_cb=None, max_consecutive_errors=10, workers=1):
    """workers > 1: gửi nhiều request song song (LLM local cần bật OLLAMA_NUM_PARALLEL tương ứng).
    progress_cb(done, total, message) luôn được gọi từ luồng chính."""
    template = template or storage.load_settings()["prompts"]["eval"]
    ks = [k for k in ks if k <= n]
    started = time.time()
    workers = max(1, int(workers))

    # 1) Truy xuất RAG và dựng prompt cho từng bài (tuần tự, rẻ)
    prompts, hits_by, rag_err = [], [], []
    for p in problems:
        hits, err = [], ""
        if rag_mode in ("rag", "graph"):
            try:
                excl = [p["id"]] if exclude_self else []
                hits = rag.get_kb().retrieve(p["prompt"], k=top_k, mode=rag_mode, exclude_ids=excl)
            except Exception as e:  # noqa: BLE001
                err = str(e)
        hits_by.append(hits)
        rag_err.append(err)
        prompts.append(build_prompt(template, p["prompt"], language, hits=hits))

    # 2) Mỗi (bài, lần sinh) là một tác vụ độc lập
    stop = threading.Event()

    def run_one(pi, si):
        if stop.is_set():
            return pi, si, None
        text, err = llm.generate(provider, prompts[pi], temperature=temperature, timeout=gen_timeout)
        if err:
            return pi, si, {"code": "", "status": "api_error", "message": err}
        code = llm.extract_code(text, language)
        res = sandbox.run_tests(code, language, problems[pi]["tests"], timeout=test_timeout)
        return pi, si, {"code": code, "status": res["status"], "message": res["message"]}

    total = len(problems) * n
    results = [[None] * n for _ in problems]
    done, consec, aborted = 0, 0, ""
    if progress_cb:
        progress_cb(0, total, f"0/{total}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, pi, si) for pi in range(len(problems)) for si in range(n)]
        for fut in as_completed(futures):
            pi, si, sample = fut.result()
            if sample is None:
                continue  # bị huỷ do dừng sớm
            results[pi][si] = sample
            done += 1
            if sample["status"] == "api_error":
                consec += 1
                if consec >= max_consecutive_errors and not aborted:
                    aborted = sample["message"]
                    stop.set()
            else:
                consec = 0
            if progress_cb:
                progress_cb(done, total, f"{done}/{total} · {problems[pi]['title']}")

    # 3) Tổng hợp. Khi dừng sớm chỉ giữ các bài đã sinh đủ n lời giải.
    rows = []
    for pi, p in enumerate(problems):
        samples = results[pi]
        if any(s is None for s in samples):
            continue
        c = sum(1 for s in samples if s["status"] == "passed")
        row = {"id": p["id"], "title": p["title"], "n": n, "c": c,
               "context_ids": [h["exercise"]["id"] for h in hits_by[pi]], "rag_error": rag_err[pi],
               "samples": samples}
        for k in ks:
            row[f"pass@{k}"] = pass_at_k(n, c, k)
        rows.append(row)

    summary = {f"pass@{k}": float(np.mean([r[f"pass@{k}"] for r in rows])) if rows else 0.0 for k in ks}
    return {
        "provider": provider.get("name"), "model": provider.get("model"),
        "language": language, "n": n, "ks": ks, "temperature": temperature,
        "rag_mode": rag_mode, "top_k": top_k, "exclude_self": exclude_self, "workers": workers,
        "num_problems": len(rows), "planned_problems": len(problems), "aborted": aborted,
        "seconds": round(time.time() - started, 1),
        "api_errors": sum(1 for r in rows for s in r["samples"] if s["status"] == "api_error"),
        "summary": summary, "rows": rows,
    }
