"""Chạy code do LLM sinh ra với test case (stdin/stdout) cho Python, C++, C và Java.

LƯU Ý BẢO MẬT: đây chỉ là cách ly mức tiến trình (thư mục tạm + timeout + giới hạn tài nguyên trên Linux/macOS).
Khi triển khai công khai trên Internet, hãy chạy toàn bộ ứng dụng trong Docker/VM không có quyền mạng.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from functools import partial


def check_toolchain():
    return {
        "python": True,
        "cpp": bool(shutil.which("g++") or shutil.which("clang++")),
        "c": bool(shutil.which("gcc") or shutil.which("clang")),
        "java": bool(shutil.which("javac") and shutil.which("java")),
    }


def _limits(cpu_seconds, limit_memory=True):
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 << 20, 10 << 20))
        if limit_memory:
            try:
                resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
            except (ValueError, OSError):
                pass
    except Exception:  # noqa: BLE001
        pass


def _run(cmd, stdin_text, timeout, cwd, limit=True, java=False):
    kwargs = dict(input=stdin_text, capture_output=True, timeout=timeout, cwd=cwd,
                  encoding="utf-8", errors="replace")
    if os.name == "posix" and (limit or java):
        # Java: chỉ giới hạn CPU/file (không giới hạn RLIMIT_AS), CPU rộng hơn vì JIT/GC chạy đa luồng
        kwargs["preexec_fn"] = partial(_limits, int(timeout) * (4 if java else 1) + 2, not java)
    try:
        p = subprocess.run(cmd, **kwargs)
        return p.returncode, p.stdout, p.stderr, False
    except subprocess.TimeoutExpired:
        return None, "", "", True
    except OSError as e:
        return -1, "", str(e), False


def _lines(s):
    lines = [l.rstrip() for l in s.replace("\r\n", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def _result(passed, status, message="", ok=0, total=0):
    return {"passed": passed, "status": status, "message": message,
            "tests_passed": ok, "tests_total": total}


def _java_main_class(code):
    """Tìm tên class chứa main() để đặt tên file đúng (mặc định Main)."""
    m = re.search(r"public\s+(?:final\s+|abstract\s+)*class\s+(\w+)", code)
    if m:
        return m.group(1)
    main_at = re.search(r"static\s+void\s+main\s*\(", code)
    if main_at:
        before = re.findall(r"\bclass\s+(\w+)", code[:main_at.start()])
        if before:
            return before[-1]
    return "Main"


def _write(path, code):
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)


def _prepare(code, language, d):
    """Biên dịch (nếu cần). Trả về (cmd, run_limits, lỗi) với lỗi = (status, message) hoặc None."""
    exe = os.path.join(d, "main.exe" if os.name == "nt" else "main")
    if language == "python":
        path = os.path.join(d, "main.py")
        _write(path, code)
        return [sys.executable, "-I", path], True, None
    if language in ("cpp", "c"):
        if language == "cpp":
            cc, flags, fname, label = shutil.which("g++") or shutil.which("clang++"), ["-O2", "-std=c++17"], "main.cpp", "g++/clang++"
        else:
            cc, flags, fname, label = shutil.which("gcc") or shutil.which("clang"), ["-O2", "-std=c17"], "main.c", "gcc/clang"
        if not cc:
            return None, True, ("compile_error", f"{label} not found")
        src = os.path.join(d, fname)
        _write(src, code)
        cmd = [cc, *flags, "-o", exe, src] + ([] if language == "cpp" else ["-lm"])
        rc, _, err, to = _run(cmd, None, 60, d, limit=False)
        if to or rc != 0:
            return None, True, ("compile_error", ("Compile timeout" if to else err)[:500])
        return [exe], True, None
    if language == "java":
        javac, java = shutil.which("javac"), shutil.which("java")
        if not javac or not java:
            return None, True, ("compile_error", "javac/java not found")
        cls = _java_main_class(code)
        src = os.path.join(d, cls + ".java")
        _write(src, code)
        rc, _, err, to = _run([javac, "-encoding", "UTF-8", src], None, 60, d, limit=False)
        if to or rc != 0:
            return None, False, ("compile_error", ("Compile timeout" if to else err)[:500])
        # JVM cần nhiều bộ nhớ ảo nên không đặt RLIMIT_AS; giới hạn bằng cờ JVM
        cmd = [java, "-Xmx512m", "-Xss64m", "-XX:+UseSerialGC", "-XX:TieredStopAtLevel=1", "-Xshare:auto", "-cp", d, cls]
        return cmd, False, None
    return None, True, ("unsupported", f"Unsupported language: {language}")


def run_tests(code, language, tests, timeout=5):
    """status: passed | wrong_answer | runtime_error | timeout | compile_error | no_code | no_tests | unsupported"""
    total = len(tests)
    if not code.strip():
        return _result(False, "no_code", "Empty code", 0, total)
    if not tests:
        return _result(False, "no_tests", "No test cases", 0, 0)
    with tempfile.TemporaryDirectory() as d:
        cmd, limit, error = _prepare(code, language, d)
        if error:
            return _result(False, error[0], error[1], 0, total)
        for i, tc in enumerate(tests, 1):
            rc, out, err, to = _run(cmd, tc["input"], timeout, d, limit=limit, java=language == "java")
            if to:
                return _result(False, "timeout", f"Test {i}: timeout > {timeout}s", i - 1, total)
            if rc != 0:
                return _result(False, "runtime_error", f"Test {i}: {err.strip()[-400:]}", i - 1, total)
            if _lines(out) != _lines(tc["output"]):
                msg = f"Test {i}: expected {tc['output'].strip()[:80]!r}, got {out.strip()[:80]!r}"
                return _result(False, "wrong_answer", msg, i - 1, total)
        return _result(True, "passed", "", total, total)


def run_outputs(code, language, inputs, timeout=10):
    """Chạy một chương trình trên nhiều input, trả về (danh sách output, lỗi). Dùng để sinh đáp án test tự động."""
    if not code.strip():
        return [], "Empty code"
    if not inputs:
        return [], "No inputs"
    with tempfile.TemporaryDirectory() as d:
        cmd, limit, error = _prepare(code, language, d)
        if error:
            return [], f"{error[0]}: {error[1]}"
        outs = []
        for i, text in enumerate(inputs, 1):
            rc, out, err, to = _run(cmd, text, timeout, d, limit=limit, java=language == "java")
            if to:
                return [], f"Input {i}: timeout > {timeout}s"
            if rc != 0:
                return [], f"Input {i}: runtime error: {err.strip()[-300:]}"
            outs.append(out)
        return outs, ""
