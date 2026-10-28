"""Ước lượng mức độ ảnh hưởng của tri thức truy xuất (RAG / GraphRAG) lên code do LLM sinh ra.

Đây là phép đo THỰC NGHIỆM trên mã nguồn (không phải nhân quả tuyệt đối). Ý tưởng:
- Chuẩn hóa code thành chuỗi token rồi lấy tập "shingle" (cụm n token liên tiếp, n = 3).
- overlap_pct  : % shingle của code sinh ra xuất hiện trong lời giải tham chiếu được truy xuất.
- net_pct      : như trên nhưng LOẠI các shingle mà LLM thuần (không RAG, chạy cùng câu hỏi) cũng tự sinh ra.
                 Đây là phần code mà tri thức truy xuất thực sự "đóng góp thêm" => chỉ số ảnh hưởng chính.
- change_pct   : % khác biệt của code so với code của LLM thuần (1 - Jaccard).
- per_ref      : phần đóng góp (%) của từng bài tham chiếu (shingle chung được chia đều cho các bài chứa nó).
- vector_pct / graph_pct : phần ảnh hưởng đến từ bài tìm theo vector và bài mở rộng qua đồ thị (chỉ ý nghĩa với GraphRAG).
"""
import re

from .prompts import pick_solution

_TOKEN = re.compile(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|==|!=|<=|>=|\+\+|--|&&|\|\||<<|>>|->|::|\S")
_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*|#(?!include|define)[^\n]*", re.DOTALL)
_STRING = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'')
SHINGLE = 3


def tokens(code):
    code = _COMMENT.sub(" ", code or "")
    code = _STRING.sub('"S"', code)
    return _TOKEN.findall(code)


def shingles(code, n=SHINGLE):
    toks = tokens(code)
    if len(toks) < n:
        return set()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def similarity(a, b):
    """Jaccard giữa hai đoạn code (0..1)."""
    sa, sb = shingles(a), shingles(b)
    if not sa and not sb:
        return 1.0 if tokens(a) == tokens(b) else 0.0
    return len(sa & sb) / len(sa | sb)


def analyze(code, hits, language, baseline_code=None):
    """Trả về dict các chỉ số (đều là % 0..100) hoặc None nếu không có tri thức truy xuất."""
    if not hits:
        return None
    gen = shingles(code)
    refs = []
    for h in hits:
        lang, ref_code = pick_solution(h["exercise"], language)
        refs.append({"id": h["exercise"]["id"], "title": h["exercise"]["title"], "via": h["via"],
                     "score": h["score"], "lang": lang, "set": shingles(ref_code)})
    base = shingles(baseline_code) if baseline_code is not None else None

    out = {"has_baseline": base is not None, "shingles": len(gen)}
    if not gen:
        out.update(overlap_pct=0.0, net_pct=None if base is None else 0.0, change_pct=None, per_ref=[],
                   vector_pct=0.0, graph_pct=0.0)
        return out

    union_refs = set().union(*[r["set"] for r in refs]) if refs else set()
    influenced = gen & union_refs
    if base is not None:
        influenced = influenced - base  # chỉ tính phần mà LLM thuần không tự sinh ra
    out["overlap_pct"] = 100.0 * len(gen & union_refs) / len(gen)
    out["net_pct"] = 100.0 * len(influenced) / len(gen) if base is not None else None
    out["change_pct"] = 100.0 * (1.0 - similarity(code, baseline_code)) if baseline_code is not None else None

    # đóng góp từng bài: mỗi shingle chung chia đều cho các bài chứa nó
    credit = [0.0] * len(refs)
    for sh in influenced:
        owners = [i for i, r in enumerate(refs) if sh in r["set"]]
        for i in owners:
            credit[i] += 1.0 / len(owners)
    total_pct = 100.0 * len(influenced) / len(gen)
    credit_sum = sum(credit)
    per_ref, vec, grp = [], 0.0, 0.0
    for r, c in zip(refs, credit):
        pct = total_pct * c / credit_sum if credit_sum else 0.0
        per_ref.append({"id": r["id"], "title": r["title"], "via": r["via"], "score": r["score"],
                        "lang": r["lang"], "pct": pct})
        if r["via"] == "graph":
            grp += pct
        else:
            vec += pct
    out.update(per_ref=per_ref, vector_pct=vec, graph_pct=grp)
    return out


def headline(info):
    """Chỉ số chính để vẽ biểu đồ: ảnh hưởng ròng nếu có baseline, ngược lại độ trùng với bài tham chiếu."""
    if not info:
        return 0.0
    return info["net_pct"] if info.get("net_pct") is not None else info["overlap_pct"]
