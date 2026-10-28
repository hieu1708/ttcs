import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

EXERCISES_FILE = DATA_DIR / "exercises.json"
SETTINGS_FILE = DATA_DIR / "settings.json"
EMBED_CACHE_FILE = DATA_DIR / "embeddings_cache.json"
HISTORY_FILE = DATA_DIR / "eval_history.json"


ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin")

# Yêu cầu nghiệm thu
MIN_EVAL_PROBLEMS = 150     # bộ kiểm tra tối thiểu 50 câu
PASS_K = 5                  # pass@5
PASS_TARGET = 0.80          # >= 80%
