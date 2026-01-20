from datasets import load_dataset

ds = load_dataset("ghadeermobasher/BC5CDR-Chemical-Disease", trust_remote_code=True)

print(ds)
print("train:", len(ds["train"]))
print("validation:", len(ds["validation"]))
print("test:", len(ds["test"]))

print("Label names:", ds["train"].features["ner_tags"].feature.names)
print("First example:", ds["train"][0])
print(ds["train"].features["ner_tags"].feature.names)
