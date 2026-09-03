"""TF-IDF + logistic regression baseline — spec requirement 4: "the same one that beat
every v1 activation result, 66.39% single-fit / 76.17% +/- 2.07 five-fold CV." Run on
the exact same frozen splits as the activation methods, on raw prompt TEXT (not
activations at all) — this is the floor every activation-based number in this project
must clear before it means anything.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def fit_and_score_single(
    train_texts: list[str], train_labels: np.ndarray, test_texts: list[str], test_labels: np.ndarray,
    seed: int = 42,
) -> dict:
    """Single fit on train, scored once on test — mirrors the activation methods'
    train/test protocol exactly (same split, same "fit once, score once" discipline)."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(x_train, train_labels)
    predictions = clf.predict(x_test)

    return {
        "accuracy": float((predictions == np.asarray(test_labels)).mean()),
        "predictions": predictions,
        "n_train": len(train_texts),
        "n_test": len(test_texts),
    }


def cross_val_score_5fold(all_texts: list[str], all_labels: np.ndarray, seed: int = 42) -> dict:
    """5-fold stratified CV over the full combined set — the second number the spec
    names (76.17% +/- 2.07 in v1). Reported alongside, not instead of, the single-fit
    number, matching how v1's own table reported both."""
    labels = np.asarray(all_labels)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_accuracies = []

    for train_idx, test_idx in skf.split(all_texts, labels):
        train_texts = [all_texts[i] for i in train_idx]
        test_texts = [all_texts[i] for i in test_idx]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        x_train = vectorizer.fit_transform(train_texts)
        x_test = vectorizer.transform(test_texts)

        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(x_train, labels[train_idx])
        predictions = clf.predict(x_test)
        fold_accuracies.append((predictions == labels[test_idx]).mean())

    fold_accuracies = np.array(fold_accuracies)
    return {
        "mean_accuracy": float(fold_accuracies.mean()),
        "std_accuracy": float(fold_accuracies.std(ddof=1)),
        "fold_accuracies": fold_accuracies.tolist(),
        "n": len(all_texts),
    }
