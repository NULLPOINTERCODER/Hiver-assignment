"""Setup directory hierarchy and initial packages."""
import os

dirs = [
    "data/raw", "data/processed", "data/sample",
    "configs",
    "notebooks",
    "src",
    "src/data", "src/intent", "src/retrieval", "src/generation", "src/agent", "src/evaluation", "src/utils",
    "models/intent_classifier", "models/reranker",
    "vectorstore",
    "evaluation",
    "results",
    "reports", "reports/figures",
    "app",
    "api",
    "scripts",
    "tests"
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    if d.startswith("src") or d in ["app", "api", "tests"]:
        init_path = os.path.join(d, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w", encoding="utf-8") as f:
                f.write('"""Package initialization."""\n')

print("All directories and __init__.py files created successfully.")
