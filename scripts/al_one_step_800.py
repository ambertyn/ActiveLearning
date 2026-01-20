import random
import math
from pathlib import Path

import spacy
from spacy.tokens import DocBin

TRAIN_PATH = Path("data/train.spacy")
DEV_PATH = Path("data/dev.spacy")
TEST_PATH = Path("data/test.spacy")

# Use your already-trained seed model from the 50-example run
SEED_MODEL_DIR = Path("curve/runs/run_50_seed0/model-best")  # change to model-last if needed

OUT_DIR = Path("al")
OUT_DIR.mkdir(exist_ok=True, parents=True)

SEED = 0
START_L = 50
TARGET_BUDGET = 800           # total labeled after selection
POOL_SAMPLE = 2000            # score only this many pool docs to fit 30 min CPU
BEAM_WIDTH = 8
BEAM_DENSITY = 0.0001
TOPK = 5                      # use top 5 beam scores


def load_docs(path: Path):
    nlp_blank = spacy.blank("en")
    db = DocBin().from_disk(path)
    return list(db.get_docs(nlp_blank.vocab))


def beam_uncertainty(nlp, texts):
    """
    Returns list of uncertainty scores for texts.
    Uses beam parses from spaCy NER: if top candidates have similar scores,
    the model is uncertain.
    """
    ner = nlp.get_pipe("ner")
    docs = [nlp.make_doc(t) for t in texts]
    beams = ner.beam_parse(docs, beam_width=BEAM_WIDTH, beam_density=BEAM_DENSITY)

    scores = []
    for doc, beam in zip(docs, beams):
        # get candidate parses (score, ents)
        parses = list(ner.moves.get_beam_parses(beam))  # list of (score, ents)
        parses = parses[:TOPK]

        top = [score for score, _ in parses]
        if not top:
            scores.append(0.0)
            continue

        # normalize to probabilities
        m = max(top)
        exps = [math.exp(x - m) for x in top]
        Z = sum(exps)
        probs = [e / Z for e in exps]

        # entropy
        ent = -sum(p * math.log(p + 1e-12) for p in probs)
        scores.append(ent)

    return scores



def write_docbin(docs, path: Path):
    DocBin(docs=docs).to_disk(path)


def main():
    # 1) load training docs
    all_train = load_docs(TRAIN_PATH)

    rng = random.Random(SEED)
    idxs = list(range(len(all_train)))
    rng.shuffle(idxs)

    L_idx = idxs[:START_L]
    U_idx = idxs[START_L:]

    L = [all_train[i] for i in L_idx]
    U = [all_train[i] for i in U_idx]

    need = TARGET_BUDGET - START_L
    print(f"Initial labeled: {len(L)} | Pool: {len(U)} | Need to select: {need}")

    # 2) load seed model
    if not SEED_MODEL_DIR.exists():
        raise SystemExit(f"Seed model not found: {SEED_MODEL_DIR} (use model-last if needed)")
    nlp = spacy.load(SEED_MODEL_DIR)

    # 3) score a subset of pool for speed
    pool_subset = U[:min(POOL_SAMPLE, len(U))]
    texts = [d.text for d in pool_subset]

    print(f"Scoring {len(texts)} pool docs with beam uncertainty...")
    u_scores = beam_uncertainty(nlp, texts)

    scored = list(zip(u_scores, pool_subset))
    scored.sort(key=lambda x: x[0], reverse=True)

    picked = [d for _, d in scored[:min(need, len(scored))]]

    # If we didn't score enough pool docs to fill the batch, fill remainder randomly
    if len(picked) < need:
        remaining = need - len(picked)
        print(f"Filling remaining {remaining} selections randomly (POOL_SAMPLE too small).")
        picked.extend(U[len(pool_subset):len(pool_subset)+remaining])

    L_al = L + picked
    print(f"AL labeled set size: {len(L_al)}")

    out_train = OUT_DIR / "train_al_800.spacy"
    write_docbin(L_al, out_train)
    print(f"Wrote: {out_train}")

    print("\nNEXT: train on this file with spaCy CLI:")
    print(f"python -m spacy train configs/config.cfg --output al/run_al_800 --paths.train {out_train} --paths.dev data/dev.spacy")
    print("Then evaluate:")
    print("python -m spacy evaluate al/run_al_800/model-best data/test.spacy -o al/metrics_al_800.json")


if __name__ == "__main__":
    main()
