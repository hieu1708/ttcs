import json
import re
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from .storage import normalize_exercise

SUPPORTED_EXTENSIONS = {"pdf", "txt", "md"}
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_TEXT_CHARS = 2_000_000

EXERCISE_TEMPLATE = """[EXERCISE]
ID: sum_two_doc
TITLE: Tổng hai số
TAGS: basics, math
PROMPT:
Đọc hai số nguyên a và b, in ra a + b.
SOLUTION_PYTHON:
a, b = map(int, input().split())
print(a + b)
SOLUTION_CPP:
#include <iostream>
using namespace std;
int main() { long long a, b; cin >> a >> b; cout << a + b; }
SOLUTION_C:
#include <stdio.h>
int main(void) { long long a, b; scanf("%lld %lld", &a, &b); printf("%lld", a + b); return 0; }
SOLUTION_JAVA:
import java.util.Scanner;
public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print(sc.nextLong() + sc.nextLong());
    }
}
TESTS:
[
  {"input": "3 5\\n", "output": "8\\n"},
  {"input": "-2 7\\n", "output": "5\\n"}
]
"""

_FIELD_RE = re.compile(
    r"(?m)^(ID|TITLE|TAGS|PROMPT|SOLUTION_PYTHON|SOLUTION_CPP|SOLUTION_C|SOLUTION_JAVA|TESTS):[\t ]*(.*)$"
)
_SOLUTION_FIELDS = {
    "SOLUTION_PYTHON": "python",
    "SOLUTION_CPP": "cpp",
    "SOLUTION_C": "c",
    "SOLUTION_JAVA": "java",
}


class DocumentError(ValueError):
    pass


def _decode_text(content):
    for encoding in ("utf-8-sig", "utf-16", "cp1258"):
        try:
            return content.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise DocumentError("Text file must use UTF-8, UTF-16 or Windows-1258 encoding.")


def extract_text(filename, content):
    if not content:
        raise DocumentError("The uploaded file is empty.")
    if len(content) > MAX_FILE_BYTES:
        raise DocumentError("The uploaded file exceeds the 20 MB limit.")
    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DocumentError("Supported formats: PDF, TXT and Markdown.")
    if suffix == "pdf":
        try:
            reader = PdfReader(BytesIO(content))
            if reader.is_encrypted:
                raise DocumentError("Password-protected PDF files are not supported.")
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except DocumentError:
            raise
        except Exception as exc:
            raise DocumentError(f"Cannot read PDF: {exc}") from exc
    else:
        text = _decode_text(content)
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise DocumentError("No text found. Scanned PDFs require OCR before uploading.")
    if len(text) > MAX_TEXT_CHARS:
        raise DocumentError("The extracted text exceeds the 2,000,000 character limit.")
    return text


def parse_exercises(text):
    blocks = re.split(r"(?im)^\[EXERCISE\][\t ]*\n?", text)
    if len(blocks) == 1:
        raise DocumentError("Each exercise must start with [EXERCISE].")
    items = []
    for number, block in enumerate(blocks[1:], 1):
        matches = list(_FIELD_RE.finditer(block))
        fields = {}
        for i, match in enumerate(matches):
            name = match.group(1)
            if name in fields:
                raise DocumentError(f"Exercise {number}: duplicate field {name}.")
            end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
            value = match.group(2) + block[match.end():end]
            fields[name] = value.strip("\n").strip()
        if not fields.get("PROMPT"):
            raise DocumentError(f"Exercise {number}: PROMPT is required.")
        raw = {
            "id": fields.get("ID", ""),
            "title": fields.get("TITLE", ""),
            "prompt": fields["PROMPT"],
            "tags": fields.get("TAGS", ""),
            "solutions": {lang: fields.get(field, "") for field, lang in _SOLUTION_FIELDS.items()},
        }
        if fields.get("TESTS"):
            try:
                raw["tests"] = json.loads(fields["TESTS"])
            except json.JSONDecodeError as exc:
                raise DocumentError(f"Exercise {number}: invalid TESTS JSON: {exc.msg}.") from exc
            if not isinstance(raw["tests"], list):
                raise DocumentError(f"Exercise {number}: TESTS must be a JSON list.")
        try:
            items.append(normalize_exercise(raw))
        except ValueError as exc:
            raise DocumentError(f"Exercise {number}: {exc}.") from exc
    ids = [item["id"] for item in items]
    if len(set(ids)) != len(ids):
        raise DocumentError("Exercise IDs must be unique within one document.")
    return items


# ---------------------------------------------------------------------------
# Đọc tài liệu tự do (không theo mẫu [EXERCISE]): tách câu hỏi, lời giải và ví dụ Input/Output
# ---------------------------------------------------------------------------
_KEYWORD_MARK = re.compile(
    r"(?im)^[ \t]*(?:#{1,4}[ \t]*)?(?:b[àa]i\s*t[ậa]p|b[àa]i|c[âa]u\s*h[ỏo]i|c[âa]u|problem|question|exercise|task|q)"
    r"[ \t]*\.?[ \t]*(\d+)[ \t]*[\.\):：\-–—]?[ \t]*(.*)$"
)
_NUMBER_MARK = re.compile(r"(?m)^[ \t]*(\d{1,3})[\.\)][ \t]+(\S.*)$")
_SOLUTION_MARK = re.compile(
    r"(?im)^[ \t]*(?:#{1,4}[ \t]*)?(?:l[ờo]i\s*gi[ảa]i|[đd][áa]p\s*[áa]n|c[áa]ch\s*gi[ảa]i|h[ướu][ớo]ng\s*d[ẫa]n\s*gi[ảa]i|b[àa]i\s*gi[ảa]i|"
    r"solution|answer|reference\s*solution)[ \t]*(?:\(([^)\n]*)\))?[ \t]*[:：]?[ \t]*$"
)
_EXAMPLE_RE = re.compile(
    r"(?is)(?:sample\s*)?(?:input|đầu\s*vào|input\s*mẫu)[ \t]*\d*[ \t]*[:：]?[ \t]*\n(.*?)\n[ \t]*"
    r"(?:sample\s*)?(?:output|đầu\s*ra|output\s*mẫu)[ \t]*\d*[ \t]*[:：]?[ \t]*\n(.*?)"
    r"(?=\n[ \t]*\n|\n[ \t]*(?:sample\s*)?(?:input|đầu\s*vào|ví\s*dụ|example|giải\s*thích|explanation|note|ghi\s*chú)\b|\Z)"
)
_FENCE_RE = re.compile(r"```([\w+#-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
_MIN_QUESTION_CHARS = 12


def guess_language(code, hint=""):
    """Đoán ngôn ngữ lập trình của một đoạn code (python / cpp / c / java) hoặc None."""
    from .storage import normalize_lang
    lang = normalize_lang(hint) if hint else None
    if lang:
        return lang
    if re.search(r"\bpublic\s+class\b|System\.out|static\s+void\s+main", code):
        return "java"
    if re.search(r"#include\s*<(?:iostream|bits/stdc\+\+\.h|vector|string|algorithm)>|std::|\bcout\b|\bcin\b|using\s+namespace", code):
        return "cpp"
    if re.search(r"#include\s*<std(?:io|lib)\.h>|\bprintf\s*\(|\bscanf\s*\(", code):
        return "c"
    if re.search(r"(?m)^\s*(?:def |import |from \w+ import |print\(|\w+\s*=\s*(?:int|input|list|map)\()", code):
        return "python"
    return None


def _extract_solutions(text, label_hint=""):
    """Từ phần lời giải trả về {ngôn ngữ: code}. Ưu tiên khối ```...```; nếu không có thì coi cả đoạn là code."""
    solutions = {}
    fences = _FENCE_RE.findall(text)
    if fences:
        for tag, body in fences:
            lang = guess_language(body, tag)
            if lang and lang not in solutions:
                solutions[lang] = body.strip("\n")
        return solutions
    body = text.strip("\n")
    if body.strip():
        lang = guess_language(body, label_hint)
        if lang:
            solutions[lang] = body
    return solutions


def _extract_examples(text):
    tests = []
    for m in _EXAMPLE_RE.finditer(text):
        inp, out = m.group(1).strip("\n"), m.group(2).strip("\n")
        if inp.strip() and out.strip():
            tests.append({"input": inp + "\n", "output": out + "\n"})
    return tests


def _split_blocks(text):
    """Tách tài liệu thành các khối (tiêu đề, nội dung). Trả về [] nếu không thấy dấu hiệu nhiều câu hỏi."""
    matches = list(_KEYWORD_MARK.finditer(text))
    if len(matches) < 2:
        nums = list(_NUMBER_MARK.finditer(text))
        # danh sách đánh số hợp lệ: bắt đầu từ 1 và tăng dần
        seq = [m for m in nums]
        ok = len(seq) >= 2 and int(seq[0].group(1)) == 1 and all(
            int(b.group(1)) > int(a.group(1)) for a, b in zip(seq, seq[1:]))
        matches = seq if ok else []
    if not matches:
        return []
    blocks = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        head = (m.group(2) or "").strip()
        body = (head + "\n" + text[m.end():end]).strip()
        blocks.append((f"{m.group(1)}", head, body))
    return blocks


def extract_questions(text, with_solutions=True):
    """Trích danh sách bài từ văn bản: [{id?, title, prompt, solutions, tests}] (chưa chuẩn hóa).
    Không thấy dấu hiệu nhiều câu hỏi -> coi cả tài liệu là một câu hỏi."""
    blocks = _split_blocks(text)
    if not blocks:
        blocks = [("", "", text.strip())]
    items = []
    for number, head, body in blocks:
        statement, sol_text, hint = body, "", ""
        sm = _SOLUTION_MARK.search(body)
        if sm:
            statement, sol_text, hint = body[:sm.start()].strip(), body[sm.end():].strip(), sm.group(1) or ""
        if len(statement) < _MIN_QUESTION_CHARS:
            continue
        title = head or statement.splitlines()[0]
        item = {"title": title[:80], "prompt": statement,
                "solutions": _extract_solutions(sol_text, hint) if (with_solutions and sol_text) else {},
                "tests": _extract_examples(statement)}
        items.append(item)
    return items


def _finalize(raw_items, prefix):
    """Chuẩn hóa + gán id duy nhất theo thứ tự trong tài liệu."""
    out, seen = [], set()
    for n, raw in enumerate(raw_items, 1):
        try:
            ex = normalize_exercise(raw)
        except ValueError:
            continue
        if not raw.get("id"):
            ex["id"] = f"{prefix}_{n}_{ex['id'][-6:]}"
        if ex["id"] in seen:
            continue
        seen.add(ex["id"])
        out.append(ex)
    return out


def parse_text_items(text, filename="doc", with_solutions=True):
    """Như parse_document_items nhưng nhận văn bản đã trích xuất."""
    if re.search(r"(?im)^\[EXERCISE\]", text):
        return parse_exercises(text)
    raw = extract_questions(text, with_solutions=with_solutions)
    prefix = re.sub(r"\W+", "_", Path(filename).stem.lower())[:16] or "doc"
    items = _finalize(raw, prefix)
    if not items:
        raise DocumentError("No question found in the document.")
    return items


def parse_document_items(filename, content, with_solutions=True):
    """Đọc PDF/TXT/MD thành danh sách bài (đã chuẩn hóa).
    - Có khối [EXERCISE] -> dùng định dạng mẫu (đầy đủ lời giải + test).
    - Ngược lại -> tự tách theo 'Bài 1', 'Câu 2', 'Problem 3', '1.', '2.'... kèm lời giải / ví dụ nếu nhận ra."""
    return parse_text_items(extract_text(filename, content), filename, with_solutions)


def parse_exercise_document(filename, content):
    return parse_document_items(filename, content, with_solutions=True)
