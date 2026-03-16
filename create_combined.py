"""
SINGLE FILE CAPTURING ALL PYTHON CODE TO FEED INTO CONTEXT FOR LLMS
Supports:
- .py files
- .ipynb (code cells only)
"""

import os
import json

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

def read_ipynb_code(path):
    code = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                code.append("".join(cell.get("source", [])))

    except Exception:
        pass

    return "\n".join(code)

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for root, dirs, files in os.walk(APP_DIR):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        for file in sorted(files):
            path = os.path.join(root, file)

            if file.endswith(".py"):
                out.write(f"\n\n# ===== FILE: {path} =====\n\n")
                out.write(read_file(path))

            elif file.endswith(".ipynb"):
                out.write(f"\n\n# ===== NOTEBOOK: {path} =====\n\n")
                out.write(read_ipynb_code(path))

print("Done ->", OUTPUT_FILE)