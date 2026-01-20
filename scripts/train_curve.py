import os
import random
import numpy as np
import pandas as pd

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    set_seed,
)
import evaluate


DATASET_NAME = "ghadeermobasher/BC5CDR-Chemical-Disease"
MODEL_NAME = "bert-base-cased"

# budgets: add/remove as needed
BUDGETS = [50, 100, 200, 400, 800, 1600, "full"]
SEEDS = [0, 1, 2]

OUT_CSV = "results_curve.csv"


def load_bc5cdr():
    return load_dataset(DATASET_NAME, trust_remote_code=True)


def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=False):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
    )

    labels = []
    for i, word_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)  # ignore special tokens
            elif word_id != prev_word_id:
                label_ids.append(word_labels[word_id])
            else:
                # same word split into multiple sub-tokens
                if label_all_tokens:
                    label_ids.append(word_labels[word_id])
                else:
                    label_ids.append(-100)
            prev_word_id = word_id
        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized


def make_subset(train_ds, budget, seed):
    if budget == "full":
        return train_ds
    # shuffle deterministically and take first N
    return train_ds.shuffle(seed=seed).select(range(min(int(budget), len(train_ds))))


def main():
    ds = load_bc5cdr()
    train_full = ds["train"]
    val_ds = ds["validation"]
    test_ds = ds["test"]

    label_names = train_full.features["ner_tags"].feature.names
    num_labels = len(label_names)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    metric = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        preds = np.argmax(predictions, axis=-1)

        true_labels = []
        true_preds = []
        for pred_seq, lab_seq in zip(preds, labels):
            seq_labels = []
            seq_preds = []
            for pred_id, lab_id in zip(pred_seq, lab_seq):
                if lab_id == -100:
                    continue
                seq_labels.append(label_names[lab_id])
                seq_preds.append(label_names[pred_id])
            true_labels.append(seq_labels)
            true_preds.append(seq_preds)

        results = metric.compute(predictions=true_preds, references=true_labels)
        # seqeval returns dict with overall_precision/recall/f1/accuracy
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # preprocess validation/test once (same for all runs)
    val_tok = val_ds.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
    test_tok = test_ds.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)

    results_rows = []

    for budget in BUDGETS:
        for seed in SEEDS:
            set_seed(seed)

            train_ds = make_subset(train_full, budget, seed)
            train_tok = train_ds.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)

            model = AutoModelForTokenClassification.from_pretrained(
                MODEL_NAME,
                num_labels=num_labels,
            )

            args = TrainingArguments(
                output_dir=f"runs/{MODEL_NAME}_bc5cdr_{budget}_seed{seed}",
                learning_rate=2e-5,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                num_train_epochs=3,
                weight_decay=0.01,
                eval_strategy="epoch",
                save_strategy="no",
                logging_strategy="epoch",

                seed=seed,
                report_to="none",
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_tok,
                eval_dataset=val_tok,
                tokenizer=tokenizer,
                data_collator=DataCollatorForTokenClassification(tokenizer),
                compute_metrics=compute_metrics,
            )

            trainer.train()

            test_metrics = trainer.evaluate(test_tok)
            f1 = float(test_metrics.get("eval_f1", 0.0))

            results_rows.append({
                "budget": str(budget),
                "seed": seed,
                "test_f1": f1,
            })

            print(f"[DONE] budget={budget} seed={seed} test_f1={f1:.4f}")

            pd.DataFrame(results_rows).to_csv(OUT_CSV, index=False)

    print("\nSaved:", OUT_CSV)


if __name__ == "__main__":
    main()
