"""RAG (vector) và GraphRAG (đồ thị bài tập) không cần thư viện nặng."""
import hashlib
import math
import re
import threading

import numpy as np

from . import config, llm, storage

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text):
    words = [w.lower() for w in _TOKEN.findall(text)]
    return words + [a + "_" + b for a, b in zip(words, words[1:])]  # unigram + bigram


def _doc_text(ex):
    return "\n".join([ex["title"], ex["prompt"], " ".join(ex["tags"])] + list(ex["solutions"].values()))


class TfidfIndex:
    """TF-IDF thuần numpy: chạy offline, hỗ trợ tiếng Việt và code."""

    def __init__(self, docs):
        toks = [tokenize(d) for d in docs]
        df = {}
        for t in toks:
            for w in set(t):
                df[w] = df.get(w, 0) + 1
        self.vocab = {w: i for i, w in enumerate(df)}
        n = len(docs)
        self.idf = np.zeros(len(self.vocab))
        for w, i in self.vocab.items():
            self.idf[i] = math.log((1 + n) / (1 + df[w])) + 1
        self.matrix = np.vstack([self._vec(t) for t in toks])

    def _vec(self, tokens):
        v = np.zeros(len(self.vocab))
        for w in tokens:
            i = self.vocab.get(w)
            if i is not None:
                v[i] += 1
        v = np.log1p(v) * self.idf
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def query(self, text):
        return self.matrix @ self._vec(tokenize(text))


def _hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class KnowledgeBase:
    def __init__(self, exercises, embed_cfg, rag_cfg):
        self.ex = exercises
        self.ids = [e["id"] for e in exercises]
        self.id2idx = {i: n for n, i in enumerate(self.ids)}
        self.embed_cfg = embed_cfg
        self.rag_cfg = rag_cfg
        self.backend = embed_cfg.get("backend", "tfidf")
        self.matrix = None
        self.tfidf = None
        self.graph = {}

    # ------------ Xây dựng chỉ mục ------------
    def build(self):
        if not self.ex:
            return
        docs = [_doc_text(e) for e in self.ex]
        if self.backend == "tfidf":
            self.tfidf = TfidfIndex(docs)
            self.matrix = self.tfidf.matrix
        else:
            m = np.array(self._embed_docs(docs), dtype=float)
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            self.matrix = m / np.where(norms == 0, 1, norms)
        self._build_graph()

    def _embed_docs(self, docs):
        cache = storage._read(config.EMBED_CACHE_FILE, {})
        key = f"{self.backend}|{self.embed_cfg.get('model')}|{self.embed_cfg.get('base_url')}"
        old = cache.get(key, {})
        bucket = {}
        todo = []
        for e, d in zip(self.ex, docs):
            h = _hash(d)
            if old.get(e["id"], {}).get("hash") == h:
                bucket[e["id"]] = old[e["id"]]
            else:
                todo.append((e["id"], d, h))
        for i in range(0, len(todo), 16):
            chunk = todo[i:i + 16]
            vecs = llm.embed([c[1] for c in chunk], self.embed_cfg)
            for (eid, _, h), v in zip(chunk, vecs):
                bucket[eid] = {"hash": h, "vec": v}
        cache[key] = bucket
        storage._write(config.EMBED_CACHE_FILE, cache)
        return [bucket[e["id"]]["vec"] for e in self.ex]

    def _build_graph(self):
        """GraphRAG nhẹ: nút = bài tập; cạnh = chung tag hoặc độ tương đồng vector cao."""
        thr = float(self.rag_cfg.get("graph_sim_threshold", 0.35))
        sim = self.matrix @ self.matrix.T
        self.graph = {i: {} for i in self.ids}
        n = len(self.ids)
        for a in range(n):
            for b in range(a + 1, n):
                shared = len(set(self.ex[a]["tags"]) & set(self.ex[b]["tags"]))
                w = 0.0
                if shared:
                    w = min(1.0, 0.3 + 0.2 * shared)
                if sim[a, b] >= thr:
                    w = max(w, float(sim[a, b]))
                if w > 0:
                    self.graph[self.ids[a]][self.ids[b]] = w
                    self.graph[self.ids[b]][self.ids[a]] = w

    def stats(self):
        edges = sum(len(v) for v in self.graph.values()) // 2
        return {"nodes": len(self.ids), "edges": edges, "backend": self.backend}

    # ------------ Truy xuất ------------
    def _scores(self, query):
        if self.backend == "tfidf":
            return self.tfidf.query(query)
        v = np.array(llm.embed([query], self.embed_cfg)[0], dtype=float)
        n = np.linalg.norm(v)
        return self.matrix @ (v / n if n else v)

    def retrieve(self, query, k=3, mode="rag", exclude_ids=()):
        """mode='rag': top-k theo vector. mode='graph': thêm các bài hàng xóm trong đồ thị."""
        if not self.ex or self.matrix is None:
            return []
        scores = self._scores(query)
        excl = set(exclude_ids)
        order = [i for i in np.argsort(-scores) if self.ids[i] not in excl and scores[i] > 0]
        seeds = order[:k]
        hits = [{"exercise": self.ex[i], "score": float(scores[i]), "via": "vector"} for i in seeds]
        if mode == "graph" and seeds:
            seen = {self.ids[i] for i in seeds}
            cand = {}
            for i in seeds:
                for nid, w in self.graph.get(self.ids[i], {}).items():
                    if nid in seen or nid in excl:
                        continue
                    cand[nid] = max(cand.get(nid, 0.0), float(scores[i]) * w)
            extra = sorted(cand.items(), key=lambda x: -x[1])[: max(1, (k + 1) // 2)]
            for nid, sc in extra:
                hits.append({"exercise": self.ex[self.id2idx[nid]], "score": sc, "via": "graph"})
        return hits


# ---------------- Cache toàn cục (dùng chung giữa các phiên) ----------------
_KB = None
_LOCK = threading.Lock()


def get_kb(force=False):
    global _KB
    with _LOCK:
        if _KB is None or force:
            s = storage.load_settings()
            kb = KnowledgeBase(storage.load_exercises(), s["embedding"], s["rag"])
            kb.build()
            _KB = kb
        return _KB


def invalidate_kb():
    global _KB
    _KB = None
