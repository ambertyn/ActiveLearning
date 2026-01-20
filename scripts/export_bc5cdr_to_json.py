from datasets import load_dataset
import json

DATASET_NAME = "ghadeermobasher/BC5CDR-Chemical-Disease"

ds = load_dataset(DATASET_NAME)  # loads from cache now

label_names = ds["train"].features["ner_tags"].feature.names

out = {
    "label_names": label_names,
    "splits": {}
}

for split in ["train", "validation", "test"]:
    items = []
    for ex in ds[split]:
        items.append({
            "id": ex["id"],
            "tokens": ex["tokens"],
            "ner_tags": ex["ner_tags"],  # ints
        })
    out["splits"][split] = items

with open("bc5cdr_frozen.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)

print("Saved bc5cdr_frozen.json")
print("Labels:", label_names)
print("Sizes:", {s: len(out["splits"][s]) for s in out["splits"]})
