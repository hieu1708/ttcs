"""Kết nối LLM: Ollama (cục bộ), OpenAI-compatible và Anthropic (Internet)."""
import os
import re
import threading
import time

import requests

PROVIDER_TYPES = ["ollama", "openai", "anthropic"]


# ---------------- Xử lý văn bản ----------------
def strip_thinking(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


_LANG_TAGS = {
    "python": {"python", "py", "python3"},
    "cpp": {"cpp", "c++", "cc", "cxx"},
    "c": {"c"},
    "java": {"java"},
}


def extract_code(text, language="python"):
    """Lấy code từ khối ```...```; ưu tiên khối đúng ngôn ngữ, nếu nhiều khối lấy khối dài nhất."""
    text = strip_thinking(text)
    blocks = re.findall(r"```([\w+#-]*)[ \t]*\r?\n(.*?)```", text, re.DOTALL)
    if blocks:
        preferred = [b for tag, b in blocks if tag.lower() in _LANG_TAGS.get(language, set())]
        pool = preferred or [b for _, b in blocks]
        return max(pool, key=len).strip()
    m = re.search(r"```([\w+#-]*)[ \t]*\r?\n(.*)", text, re.DOTALL)  # khối code bị cắt, chưa đóng
    if m:
        return m.group(2).strip()
    return text.strip()


# ---------------- Gọi API ----------------
class ApiError(RuntimeError):
    def __init__(self, status, body, retry_after=None):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status, self.body, self.retry_after = status, body, retry_after


def _post(url, payload, headers, timeout):
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    if r.status_code >= 400:
        wait = None
        try:
            wait = float(r.headers.get("Retry-After", ""))
        except ValueError:
            m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', r.text)  # Gemini
            if m:
                wait = float(m.group(1))
        raise ApiError(r.status_code, r.text, wait)
    return r.json()


# Giới hạn tốc độ theo từng mô hình (rpm = số request tối đa mỗi phút, 0 = không giới hạn)
_RL_LOCK = threading.Lock()
_NEXT_SLOT = {}


def _throttle(provider):
    rpm = float(provider.get("rpm") or 0)
    if rpm <= 0:
        return
    key = provider.get("name") or provider.get("model")
    with _RL_LOCK:
        now = time.time()
        slot = max(_NEXT_SLOT.get(key, 0.0), now)
        _NEXT_SLOT[key] = slot + 60.0 / rpm
    if slot > now:
        time.sleep(slot - now)


def _generate_once(provider, prompt, temperature, timeout, max_tokens):
    ptype = provider.get("type", "ollama")
    model = provider.get("model", "")
    base = (provider.get("base_url") or "").rstrip("/")
    key = provider.get("api_key") or ""
    if ptype == "ollama":
        base = base or "http://localhost:11434"
        payload = {
            "model": model, "prompt": prompt, "stream": False, "keep_alive": "30m",
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        think = str(provider.get("think") or "auto").lower()
        if think in ("off", "on"):  # tắt/bật chế độ suy nghĩ của model (Qwen3...)
            payload["think"] = think == "on"
        data = _post(f"{base}/api/generate", payload, {}, timeout)
        return data.get("response", "")
    if ptype == "openai":
        base = base or "https://api.openai.com/v1"
        key = key or os.environ.get("OPENAI_API_KEY", "")
        data = _post(f"{base}/chat/completions", {
            "model": model, "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }, {"Authorization": f"Bearer {key}"}, timeout)
        return data["choices"][0]["message"].get("content") or ""
    if ptype == "anthropic":
        base = base or "https://api.anthropic.com"
        key = key or os.environ.get("ANTHROPIC_API_KEY", "")
        data = _post(f"{base}/v1/messages", {
            "model": model, "max_tokens": max_tokens, "temperature": min(temperature, 1.0),
            "messages": [{"role": "user", "content": prompt}],
        }, {"x-api-key": key, "anthropic-version": "2023-06-01"}, timeout)
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    raise ValueError(f"Unknown provider type: {ptype}")


def generate(provider, prompt, temperature=0.2, timeout=180, max_tokens=4096, retries=4):
    """Trả về (text, error). Tự thử lại (chờ tăng dần) khi gặp 429/5xx/timeout/mất kết nối."""
    last_err = ""
    for attempt in range(retries + 1):
        _throttle(provider)
        wait, retryable = None, False
        try:
            text = _generate_once(provider, prompt, temperature, timeout, max_tokens)
            return strip_thinking(text), None
        except ApiError as e:
            last_err = str(e)
            retryable = e.status in (408, 429, 500, 502, 503, 504)
            wait = e.retry_after
            if e.status == 429 and re.search(r"PerDay|per day|daily", e.body, re.I):
                retryable = False  # hết hạn mức theo NGÀY: thử lại cũng vô ích
                last_err += "  [Daily quota exhausted - wait for reset or enable billing]"
        except requests.exceptions.Timeout:
            last_err, retryable = f"Timeout after {timeout}s", True
        except requests.exceptions.ConnectionError:
            last_err = f"Cannot connect to {provider.get('base_url') or provider.get('type')} (server down / no internet?)"
            retryable = True
        except Exception as e:  # noqa: BLE001
            return "", str(e)
        if not retryable or attempt == retries:
            break
        time.sleep(min(wait if wait else 5 * (2 ** attempt), 90))
    return "", last_err


def embed(texts, cfg):
    """Embedding cho backend 'ollama' hoặc 'openai' (OpenAI-compatible)."""
    backend = cfg.get("backend")
    model = cfg.get("model", "")
    base = (cfg.get("base_url") or "").rstrip("/")
    if backend == "ollama":
        base = base or "http://localhost:11434"
        data = _post(f"{base}/api/embed", {"model": model, "input": texts}, {}, 120)
        return data["embeddings"]
    if backend == "openai":
        base = base or "https://api.openai.com/v1"
        key = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        data = _post(f"{base}/embeddings", {"model": model, "input": texts},
                     {"Authorization": f"Bearer {key}"}, 120)
        return [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]
    raise ValueError(f"Backend '{backend}' does not support embeddings")
