import re

LANG_INFO = {
    "python": {"name": "Python", "fence": "python"},
    "cpp": {"name": "C++", "fence": "cpp"},
    "c": {"name": "C", "fence": "c", "note": "Write standard C17 with an int main(void) or int main() entry point."},
    "java": {"name": "Java", "fence": "java", "note": "Use no package declaration and put the entry point in public class Main."},
}
ANSWER_LANGS = {"vi": "Vietnamese", "en": "English"}

# Khi bài tham chiếu không có lời giải đúng ngôn ngữ đang hỏi thì dùng ngôn ngữ gần nhất
LANG_FALLBACK = {
    "python": ["cpp", "c", "java"],
    "cpp": ["c", "java", "python"],
    "c": ["cpp", "java", "python"],
    "java": ["cpp", "c", "python"],
}


def pick_solution(exercise, language):
    """Chọn lời giải tham chiếu cho `language`: (ngôn ngữ, code) hoặc (None, "")."""
    sols = exercise.get("solutions", {})
    for lang in [language] + LANG_FALLBACK.get(language, []):
        if sols.get(lang):
            return lang, sols[lang]
    for lang, code in sols.items():
        if code:
            return lang, code
    return None, ""

DEFAULT_PROMPTS = {
    "user": """You are a patient and precise programming tutor.
Task: solve the user's exercise in {language}.

{context}

Instructions:
1. Prefer the approach, naming style and structure of the reference solutions when they are relevant; do not copy blindly if the problem is different.
2. Write all explanations in {answer_language}.
3. Give a short explanation of the idea, then ONE complete, runnable {language} solution inside a single ```{fence} code block.
4. If the references do not cover the problem, solve it from first principles and say so briefly.

User question:
{question}
""",
    "eval": """You are an expert competitive programmer.
Write a complete {language} program that solves the problem below.
The program must read ALL input from standard input and print the result to standard output, exactly in the required format, with no extra text or prompts.

{context}

Output ONLY one ```{fence} code block. No explanation.

Problem:
{question}
""",
}


def render(template, **kw):
    """Thay {key} trong một lượt (giá trị thay vào không bị quét lại)."""
    return re.sub(r"\{(\w+)\}", lambda m: str(kw.get(m.group(1), m.group(0))), template)


def format_context(hits, language):
    """Biến danh sách kết quả truy xuất thành đoạn ngữ cảnh đưa vào prompt."""
    if not hits:
        return ""
    blocks = []
    for i, h in enumerate(hits, 1):
        ex = h["exercise"]
        lang, code = pick_solution(ex, language)
        block = f"### Reference {i}: {ex['title']}\nProblem: {ex['prompt']}\n"
        if lang:
            info = LANG_INFO.get(lang, {"name": lang, "fence": lang})
            block += f"Reference solution ({info['name']}):\n```{info['fence']}\n{code}\n```\n"
        blocks.append(block)
    return (
        "Reference exercises retrieved from the knowledge base "
        "(follow their approach, naming and I/O conventions closely when they fit the problem):\n\n"
        + "\n".join(blocks)
    )


def build_prompt(template, question, language, answer_language="en", hits=None):
    info = LANG_INFO[language]
    prompt = render(
        template,
        question=question,
        language=info["name"],
        fence=info["fence"],
        answer_language=ANSWER_LANGS.get(answer_language, "English"),
        context=format_context(hits or [], language),
    )
    if info.get("note"):
        prompt += "\n\n" + info["note"]
    return prompt
