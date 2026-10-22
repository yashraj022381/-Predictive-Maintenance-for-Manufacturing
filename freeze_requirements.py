"""Write requirements files pinned to the versions installed right now."""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

RUNTIME = ["numpy", "pandas", "scikit-learn", "joblib", "pyyaml", "python-dotenv",
           "matplotlib", "shap", "sentence-transformers", "langchain-core",
           "langchain-groq", "langgraph", "fastapi", "uvicorn", "streamlit", "requests"]
XGB = ["xgboost"]
DEV = ["pytest", "httpx"]
EXTRAS = {"uvicorn": "uvicorn[standard]"}

def pins(names):
    out = []
    for n in names:
        try:
            out.append(f"{EXTRAS.get(n, n)} == {version(n)}")
        except PackageNotFoundError:
            print("NOT INSTALLED:", n)
    return out

def write(path: str, lines: list[str]) -> None:
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

write("requirements.txt", pins(RUNTIME))
write("requirements-xgb.txt", pins(XGB))
write("requirements-dev.txt", ["-r requirements.txt", "-r requirements-xgb-text"] + pins(DEV))
write("requirements-torch.txt", [f"torch=={version('torch').split('+')[0]}"])

for f in ("requirements.txt", "requirements-xgb.txt"):
    print(f"--- {f} ---")
    print(Path(f).read_text(encoding="utf-8"))
