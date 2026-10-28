import copy
import hashlib
import json
import os
import tempfile
import time

from . import config
from .prompts import DEFAULT_PROMPTS

DEFAULT_SETTINGS = {
    "providers": [
        {"name": "Qwen2.5", "type": "ollama", "model": "qwen2.5-coder:14b",
         "base_url": "http://localhost:11434", "api_key": ""},
    ],
    "embedding": {"backend": "tfidf", "model": "nomic-embed-text",
                  "base_url": "http://localhost:11434", "api_key": ""},
    "rag": {"top_k": 3, "graph_sim_threshold": 0.8},
    "prompts": copy.deepcopy(DEFAULT_PROMPTS),
}


def _read(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ---------------- Settings ----------------
def load_settings():
    saved = _read(config.SETTINGS_FILE, {})
    s = copy.deepcopy(DEFAULT_SETTINGS)
    for key, val in saved.items():
        if isinstance(val, dict) and isinstance(s.get(key), dict):
            s[key].update(val)
        else:
            s[key] = val
    return s


def save_settings(s):
    _write(config.SETTINGS_FILE, s)


# ---------------- Exercises ----------------
def normalize_lang(name):
    n = str(name).strip().lower()
    if n in ("python", "py", "python3"):
        return "python"
    if n in ("cpp", "c++", "cxx", "cc"):
        return "cpp"
    if n == "c":
        return "c"
    if n in ("java", "java17"):
        return "java"
    return None


def normalize_exercise(raw):
    """Chuẩn hóa 1 bài tập. Hỗ trợ cả định dạng cũ {prompt, reference_code}."""
    if not isinstance(raw, dict) or not str(raw.get("prompt", "")).strip():
        raise ValueError("missing prompt")
    prompt = str(raw["prompt"]).strip()

    solutions = {}
    if isinstance(raw.get("solutions"), dict):
        for k, v in raw["solutions"].items():
            lang = normalize_lang(k)
            if lang and str(v).strip():
                solutions[lang] = str(v).strip("\n")
    if not solutions and str(raw.get("reference_code", "")).strip():
        solutions["python"] = str(raw["reference_code"]).strip("\n")

    tests = []
    for tc in raw.get("tests") or []:
        if isinstance(tc, dict) and "input" in tc and "output" in tc:
            tests.append({"input": str(tc["input"]), "output": str(tc["output"])})

    tags = raw.get("tags") or []
    if isinstance(tags, str):
        tags = [x.strip() for x in tags.split(",") if x.strip()]

    ex_id = str(raw.get("id") or "ex_" + hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8])
    title = str(raw.get("title") or prompt.replace("\n", " ")[:60])
    return {"id": ex_id, "title": title, "prompt": prompt,
            "tags": [str(x) for x in tags], "solutions": solutions, "tests": tests}


def load_exercises():
    out = []
    for raw in _read(config.EXERCISES_FILE, []):
        try:
            out.append(normalize_exercise(raw))
        except ValueError:
            continue
    return out


def save_exercises(items):
    _write(config.EXERCISES_FILE, items)


def upsert_exercises(new_items):
    items = load_exercises()
    index = {e["id"]: i for i, e in enumerate(items)}
    added = updated = 0
    for ex in new_items:
        if ex["id"] in index:
            items[index[ex["id"]]] = ex
            updated += 1
        else:
            index[ex["id"]] = len(items)
            items.append(ex)
            added += 1
    save_exercises(items)
    return added, updated


def delete_exercises(ids):
    ids = set(ids)
    save_exercises([e for e in load_exercises() if e["id"] not in ids])


# ---------------- Evaluation history ----------------
def load_history():
    return _read(config.HISTORY_FILE, [])


def append_history(result):
    hist = load_history()
    result = dict(result)
    result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    hist.append(result)
    _write(config.HISTORY_FILE, hist)


def clear_history():
    _write(config.HISTORY_FILE, [])
