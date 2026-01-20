import pandas as pd

THRESHOLD_TYPE = "95pct"  # or "within_1pt"

df = pd.read_csv("results_curve.csv")
pivot = df.groupby("budget")["test_f1"].agg(["mean", "std"]).reset_index()
print(pivot.sort_values("budget", key=lambda s: s.map(lambda x: 10**9 if x=="full" else int(x))))

full_f1 = float(pivot[pivot["budget"]=="full"]["mean"].iloc[0])

if THRESHOLD_TYPE == "95pct":
    target = 0.95 * full_f1
    print("\nFull F1:", full_f1)
    print("Target (95% of full):", target)
    candidates = pivot[pivot["mean"] >= target].copy()
else:
    target = full_f1 - 0.01
    candidates = pivot[pivot["mean"] >= target].copy()

# exclude "full" itself when finding minimum
candidates = candidates[candidates["budget"] != "full"]
candidates["budget_int"] = candidates["budget"].astype(int)
best = candidates.sort_values("budget_int").head(1)

if len(best) == 0:
    print("\nNo subset reached the target.")
else:
    b = int(best["budget_int"].iloc[0])
    m = float(best["mean"].iloc[0])
    s = float(best["std"].iloc[0])
    print(f"\nLeast needed training size: {b} examples (mean F1={m:.4f} ± {s:.4f})")
