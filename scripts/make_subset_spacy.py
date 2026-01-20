import random
from pathlib import Path
import spacy
from spacy.tokens import DocBin

IN_PATH = Path("data/train.spacy")
OUT_DIR = Path("curve/subsets")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    nlp = spacy.blank("en")
    db = DocBin().from_disk(IN_PATH)
    docs = list(db.get_docs(nlp.vocab))
    total = len(docs)

    budgets = [50, 100, 200, 400, 800, 1600, 3200]  # adjust if you want
    seeds = [0, 1, 2]

    for seed in seeds:
        rng = random.Random(seed)
        idxs = list(range(total))
        rng.shuffle(idxs)

        for b in budgets:
            b = min(b, total)
            chosen = [docs[i] for i in idxs[:b]]
            out_path = OUT_DIR / f"train_{b}_seed{seed}.spacy"
            DocBin(docs=chosen).to_disk(out_path)
            print("Wrote", out_path)

    print("Done.")

if __name__ == "__main__":
    main()
