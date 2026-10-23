
"""Lightweight manual search (TF-IDF keyword matching). Same .search() as ManualIndex."""
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.loader import load_config
from src.genai.rag import chunk_manual


class ManualIndexLite:
    def __init__(self, config_path: str = "configs/config.yaml"):
        cfg = load_config(config_path)
        manual_dir = Path(cfg["paths"].get("manuals_dir", "data/manuals"))
        self.chunks = [c for p in sorted(manual_dir.glob("*.md")) for c in chunk_manual(p)]
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform([c["text"] for c in self.chunks])

    def search(self, query: str, k: int = 3) -> list[dict]:
        scores = (self.matrix @ self.vec.transform([query]).T).toarray().ravel()
        return [{**self.chunks[i], "score": float(scores[i])}
                for i in scores.argsort()[::-1][:k]]


if __name__ == "__main__":
    index = ManualIndexLite()
    print(f"Indexed {len(index.chunks)} sections: {[c['id'] for c in index.chunks]}\n")
    for q in ["Bypass ratio higher", "Coolant bleed lower", "HPC outlet temperature higher",
              "how urgent is it when fewer than 30 cycles are left"]:
        print(f"Q: {q}")
        for h in index.search(q, k=3):
            print(f"   {h['id']:8s} score {h['score']:.2f}")
