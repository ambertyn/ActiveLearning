import json
from pathlib import Path

import spacy
from spacy.tokens import DocBin, Doc
from spacy.vocab import Vocab
from spacy.util import filter_spans

IN_PATH = Path("bc5cdr_frozen.json")
OUT_DIR = Path("data")

def iob_to_char_spans(tokens, tag_ids, tag_names):
    """
    Rebuild text as " ".join(tokens) and convert BIO tags into char spans.
    Returns (text, spans) where spans are (start, end, label).
    """
    # text reconstruction + token char offsets
    starts, ends = [], []
    pos = 0
    text_parts = []
    for i, tok in enumerate(tokens):
        if i > 0:
            text_parts.append(" ")
            pos += 1
        starts.append(pos)
        text_parts.append(tok)
        pos += len(tok)
        ends.append(pos)
    text = "".join(text_parts)

    spans = []
    i = 0
    while i < len(tokens):
        tag = tag_names[tag_ids[i]]
        if tag.startswith("B-"):
            label = tag[2:]  # "Disease" or "Chemical" based on your label_names
            j = i + 1
            while j < len(tokens) and tag_names[tag_ids[j]] == f"I-{label}":
                j += 1
            spans.append((starts[i], ends[j-1], label))
            i = j
        else:
            i += 1

    return text, spans

def split_to_docbin(split_examples, tag_names):
    vocab = Vocab()
    db = DocBin(store_user_data=False)
    skipped_empty = 0

    for ex in split_examples:
        tokens = ex.get("tokens", [])
        tags = ex.get("ner_tags", [])
        if not tokens or not tags or len(tokens) != len(tags):
            skipped_empty += 1
            continue

        text, spans = iob_to_char_spans(tokens, tags, tag_names)

        # create Doc with exact tokens/spaces
        spaces = [True] * len(tokens)
        spaces[-1] = False
        doc = Doc(vocab, words=tokens, spaces=spaces)

        ents = []
        for start, end, label in spans:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is not None:
                ents.append(span)

        doc.ents = filter_spans(ents)
        db.add(doc)

    return db, skipped_empty

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with IN_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    tag_names = data["label_names"]  # e.g. ['O','B-Disease','I-Disease','B-Chemical','I-Chemical']

    for split_name, out_name in [("train", "train.spacy"), ("validation", "dev.spacy"), ("test", "test.spacy")]:
        examples = data["splits"][split_name]
        db, skipped = split_to_docbin(examples, tag_names)
        db.to_disk(OUT_DIR / out_name)
        print(f"{split_name}: wrote {len(examples) - skipped} docs to {OUT_DIR/out_name} (skipped {skipped})")

    print("Done. Files in data/: train.spacy dev.spacy test.spacy")
    print("Entity labels learned will be:", ["Disease", "Chemical"])

if __name__ == "__main__":
    main()
