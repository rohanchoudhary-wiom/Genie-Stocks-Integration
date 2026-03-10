"""
SINGLE FILE CAPTURING ALL PYTHON CODE TO FEED INTO CONTEXT FOR LLMS
"""

import os

APP_DIR = "./"
OUTPUT_FILE = "combined_app.py"

EXCLUDE_DIRS = {"venv", ".venv", "__pycache__"}

def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1", errors="ignore") as f:
            return f.read()

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for root, dirs, files in os.walk(APP_DIR):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        for file in sorted(files):
            if file.endswith(".py"):
                path = os.path.join(root, file)
                out.write(f"\n\n# ===== FILE: {path} =====\n\n")
                out.write(read_file(path))

print("Done ->", OUTPUT_FILE)