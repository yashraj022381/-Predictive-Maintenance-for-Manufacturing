"""Phase 6a: a tiny RAG engine. Splits manuals into sections, turns each into
numbers (embeddings), and finds the sections closest in meaning to a question."""
import json
import re
import numpy as np

from pathlib import Path
#from sentence_transformers import SentenceTransformer
from src.data.loader import load_config

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ID_PATTERN = re.compile(r"^([A-Z]{2, 4}-\d{2})")
DASHES = re.compile(r"[\u2010-\u2015\u2212]")


def chunk_manual(path: Path) -> list[dict]:
    """Each '## ' heading starts one chunk (one procedure)."""
    #text = path.read_text(encoding="utf-8")
    text = DASHES.sub("-", path.read_text(encoding="utf-8"))
    doc_title = text.splitlines()[0].lstrip("# ").strip()
    chunks = []
    for block in re.split(r"\n(?=## )", text)[1:]:
        heading, _, body = block.partition("\n")
        heading = heading.lstrip("# ").strip()
        #heading = DASHES.sub("-", heading.lstrip("# ").strip())
        #m = ID_PATTERN.match(heading)
        chunks.append({
            "id": heading.split(":")[0].strip(), #m.group(1) if m else heading,
            "title": heading,
            "source": path.name,
            "text": f"{doc_title} | {heading}\n{body.strip()}",
        })
    return chunks


class ManualIndex:
    def __init__(self, config_path: str = "configs/config.yaml"):
        cfg = load_config(config_path)
        self.manual_dir = Path(cfg["paths"].get("manuals_dir", "data/manuals"))
        self.art = Path(cfg["paths"]["artifacts_dir"])
        self._encoder = None
        if (self.art / "manual_chunks.json").exists():
            self.chunks = json.loads((self.art / "manual_chunks.json").read_text())
            self.emb = np.load(self.art / "manual_embeddings.npy")
        else:
            self.build()

    #@property
    #def encoder(self):
    #    if self._encoder is None:
    #        self._encoder = SentenceTransformer(EMBED_MODEL)
    #    return self._encoder
    
    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer   # only loaded when needed
            self._encoder = SentenceTransformer(EMBED_MODEL)
        return self._encoder

    def build(self) -> None:
        self.chunks = [c for p in sorted(self.manual_dir.glob("*.md"))
                       for c in chunk_manual(p)]
        self.emb = self.encoder.encode([c["text"] for c in self.chunks],
                                        normalize_embeddings=True)
        self.art.mkdir(parents=True, exist_ok=True)
        np.save(self.art / "manual_embeddings.npy", self.emb)
        (self.art / "manual_chunks.json").write_text(json.dumps(self.chunks, indent=2))

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = self.encoder.encode([query], normalize_embeddings=True)[0]
        scores = self.emb @ q
        return [{**self.chunks[i], "score": float(scores[i])}
                for i in np.argsort(-scores)[:k]]


if __name__ == "__main__":
    index = ManualIndex()
    index.build()
    print(f"Indexed {len(index.chunks)} sections: {[c['id'] for c in index.chunks]}\n")
    tests = [
        "Bypass ratio higher",
        "Coplant bleed lower",
        "HPC outlet temperature higher",
        "how urgent is it when fewer than 30 cycles are left"
    ]

    for q in tests:
        hits = index.search(q, k=3)
        print(f"Q: {q}")
        for h in hits:
            print(f"   {h['id']:8s} score {h['score']:.2f} ({h['title']})")
        
    
