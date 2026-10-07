"""Train the Normal-vs-Attack classifier and save everything the app needs.

Usage:  python train.py            (downloads NSL-KDD automatically if missing)
"""
import argparse
import json
import time
from pathlib import Path

import joblib
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import (CATEGORICAL, FEATURES, NUMERIC, build_preprocessor,
                        ensure_data, load_dataset)

ROOT = Path(__file__).parent


def evaluate(pipe, X, y):
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": p,
        "recall": r,
        "f1": f1,
        "roc_auc": roc_auc_score(y, proba),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),  # [[TN, FP],[FN, TP]]
    }


def build_feature_spec(X):
    """Metadata so the web form can build proper widgets + sensible defaults."""
    spec = {}
    for c in FEATURES:
        s = X[c]
        if c in CATEGORICAL:
            spec[c] = {"kind": "cat", "options": sorted(s.unique().tolist()),
                       "default": s.mode().iloc[0]}
            continue
        uniq = set(s.dropna().unique().tolist())
        kind = "binary" if uniq <= {0, 1} else ("rate" if s.max() <= 1.0 else "count")
        spec[c] = {"kind": kind, "min": float(s.min()), "max": float(s.max()),
                   "default": float(s.median())}
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--trees", type=int, default=100)
    args = ap.parse_args()
    data_dir = Path(args.data_dir)

    ensure_data(data_dir)
    X_train, y_train, _ = load_dataset(data_dir / "KDDTrain+.txt")
    X_test, y_test, test_labels = load_dataset(data_dir / "KDDTest+.txt")
    print(f"Train: {X_train.shape}  attacks={y_train.mean():.1%} | Test: {X_test.shape}")

    def make(clf):
        return Pipeline([("prep", build_preprocessor()), ("clf", clf)])

    candidates = {
        "RandomForest": make(RandomForestClassifier(
            n_estimators=args.trees, min_samples_leaf=2, n_jobs=-1, random_state=42)),
        "ExtraTrees": make(ExtraTreesClassifier(
            n_estimators=args.trees, min_samples_leaf=2, n_jobs=-1, random_state=42)),
    }

    # 1) model selection on a validation split carved out of the TRAIN file
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
    comparison = {}
    for name, pipe in candidates.items():
        t0 = time.time()
        pipe.fit(X_fit, y_fit)
        m = evaluate(pipe, X_val, y_val)
        m["train_seconds"] = round(time.time() - t0, 1)
        comparison[name] = m
        print(f"[val] {name}: acc={m['accuracy']:.4f} f1={m['f1']:.4f}")
    best_name = max(comparison, key=lambda n: comparison[n]["f1"])

    # 2) refit the winner on the full training file, report on official test set
    best = candidates[best_name]
    best.fit(X_train, y_train)
    test_metrics = evaluate(best, X_test, y_test)
    print(f"[test] {best_name}: acc={test_metrics['accuracy']:.4f} "
          f"recall={test_metrics['recall']:.4f} f1={test_metrics['f1']:.4f}")

    # 3) save artifacts
    (ROOT / "model").mkdir(exist_ok=True)
    joblib.dump(best, ROOT / "model" / "nids_model.joblib", compress=3)

    names = CATEGORICAL + NUMERIC  # column order produced by the preprocessor
    imp = best.named_steps["clf"].feature_importances_
    ranked = sorted(zip(names, imp), key=lambda t: -t[1])
    meta = {
        "best_model": best_name,
        "validation_comparison": comparison,
        "test_metrics": test_metrics,
        "feature_importance": [{"feature": f, "importance": float(v)} for f, v in ranked[:20]],
        "top_features": [f for f, _ in ranked[:12]],
        "feature_spec": build_feature_spec(X_train),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    (ROOT / "model" / "metadata.json").write_text(json.dumps(meta, indent=2))

    # 4) demo CSV for the upload tab (with header + ground-truth label)
    (ROOT / "sample_data").mkdir(exist_ok=True)
    sample = X_test.assign(label=test_labels).sample(300, random_state=1)
    sample.to_csv(ROOT / "sample_data" / "sample_traffic.csv", index=False)
    print("Saved model/nids_model.joblib, model/metadata.json, sample_data/sample_traffic.csv")


if __name__ == "__main__":
    main()
