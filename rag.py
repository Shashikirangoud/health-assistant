from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_chunks(folder="knowledge", max_chars=800):
    chunks = []
    for f in sorted(Path(folder).glob("*.txt")):
        lines = f.read_text(encoding="utf-8").splitlines()
        if not lines:
            continue
        source = lines[0].replace("SOURCE:", "").strip()
        paragraphs = [p.strip() for p in "\n".join(lines[1:]).split("\n\n") if p.strip()]
        current = ""
        for p in paragraphs:
            if len(current) + len(p) > max_chars and current:
                chunks.append({"text": current, "source": source, "file": f.name})
                current = ""
            current += p + "\n\n"
        if current.strip():
            chunks.append({"text": current.strip(), "source": source, "file": f.name})
    return chunks


class Retriever:
    def __init__(self, folder="knowledge"):
        self.chunks = load_chunks(folder)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = (
            self.vectorizer.fit_transform([c["text"] for c in self.chunks])
            if self.chunks else None
        )

    def search(self, query, k=3, min_score=0.1):
        if self.matrix is None:
            return []
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        top = scores.argsort()[::-1][:k]
        return [self.chunks[i] for i in top if scores[i] >= min_score]