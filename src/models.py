"""Small, reproducible classifiers for the UCI HAR feature vectors."""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def evaluate_models(X_train, y_train, X_test, y_test, labels, class_names):
    """Fit a simple baseline and random forest; return test-set outputs."""
    models = {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Random forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        results[name] = {
            "model": model,
            "predictions": predicted,
            "accuracy": accuracy_score(y_test, predicted),
            "macro_f1": f1_score(y_test, predicted, labels=labels, average="macro", zero_division=0),
            "report": classification_report(
                y_test, predicted, labels=labels, target_names=class_names,
                output_dict=True, zero_division=0,
            ),
            "confusion": confusion_matrix(y_test, predicted, labels=labels),
        }
    return results
