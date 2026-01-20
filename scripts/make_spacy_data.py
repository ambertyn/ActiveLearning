from datasets import load_dataset
import spacy
from spacy.tokens import DocBin, Doc
from spacy.vocab import Vocab

DATASET_NAME = "ghadeermobasher/BC5CDR-Chemical-Disease"

def iob_to_spans(tokens, tags, tag_names):
    """
    Convert token-level IOB tags to spans in character offsets.
    Returns: (text, spans_as_offsets) where spans are (start_char, end_char, label)
    """
    # Reconstruct text with single spaces between tokens (simple and consistent)
    # Also track token start/end char offsets in this reconstructed text
    starts = []
    ends = []
    parts = []
    pos = 0
    for tok in tokens:
        if parts:
            parts.append(" ")
            pos += 1
        starts.append(pos)
        parts.append(tok)
        pos += len(tok)
        ends.append(pos)
    text = "".join(parts)

    spans = []
    i = 0
    while i < len(tokens):
        tag = tag_names[tags[i]]
        if tag.startswith("B-"):
            label = tag[2:]
            j = i + 1
            while j < len(tokens) and tag_names[tags[j]] == f"I-{label}":
                j += 1
            span_start = starts[i]
            span_end = ends[j - 1]
            spans.append((span_start, span_end, label))
            i = j
        else:
            i += 1

    return text, spans

def build_docbin(split_ds, tag_names):
    vocab = Vocab()
    db = DocBin(store_user_data=False)

    for ex in split_ds:
        tokens = ex["tokens"]
        tags = ex["ner_tags"]

        text, spans = iob_to_spans(tokens, tags, tag_names)

        # Create Doc with given tokens + spaces so it matches reconstructed text
        words = tokens
        spaces = [True] * len(words)
        spaces[-1] = False
        doc = Doc(vocab, words=words, spaces=spaces)

        ents = []
        for start, end, label in spans:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is not None:
                ents.append(span)

        doc.ents = ents
        db.add(doc)

    return db

def main():
    ds = load_dataset(DATASET_NAME, trust_remote_code=True)
    tag_names = ds["train"].features["ner_tags"].feature.names

    train_db = build_docbin(ds["train"], tag_names)
    dev_db = build_docbin(ds["validation"], tag_names)
    test_db = build_docbin(ds["test"], tag_names)

    train_db.to_disk("train.spacy")
    dev_db.to_disk("dev.spacy")
    test_db.to_disk("test.spacy")

    print("Saved: train.spacy, dev.spacy, test.spacy")
    print("Entity labels used:", ["Chemical", "Disease"], "(these come from tag names)")

if __name__ == "__main__":
    main()
