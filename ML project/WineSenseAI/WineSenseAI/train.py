import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize

FEATURES = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
            "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]

# ---------- 1. Dataset collection ----------
df = pd.read_csv("winequality-red.csv", sep=";")

# ---------- 2. Data preprocessing ----------
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
after = len(df)
missing = int(df.isnull().sum().sum())

def bin_quality(q):
    if q <= 4: return "Poor"
    if q <= 6: return "Average"
    return "Good"

df["quality_class"] = df["quality"].apply(bin_quality)

X = df[FEATURES].values
y = df["quality_class"].values
classes_sorted = sorted(set(y))  # alphabetical: Average, Good, Poor

# representative "typical" quality score per class, computed from real data (not invented)
rep_score = df.groupby("quality_class")["quality"].mean().round().astype(int).to_dict()

# ---------- 3/4. Feature selection + train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 5. Model training + 6. Hyperparameter tuning (GridSearchCV) ----------
dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid={"max_depth":[3,4,5,6,8,10], "min_samples_split":[2,5,10], "criterion":["gini","entropy"]},
    cv=5, scoring="f1_weighted", n_jobs=-1
)
dt_grid.fit(X_train, y_train)
dt_best = dt_grid.best_estimator_

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={"n_estimators":[60,100], "max_depth":[6,8,10], "min_samples_split":[2,5]},
    cv=5, scoring="f1_weighted", n_jobs=-1
)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_

# ---------- 7. Cross-validation ----------
dt_cv = cross_val_score(dt_best, X_train, y_train, cv=5, scoring="accuracy")
rf_cv = cross_val_score(rf_best, X_train, y_train, cv=5, scoring="accuracy")

def evaluate(model, name):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=model.classes_.tolist())

    # macro-average one-vs-rest ROC/AUC
    y_test_bin = label_binarize(y_test, classes=model.classes_.tolist())
    fpr_grid = np.linspace(0, 1, 21)
    tprs = []
    aucs = []
    for i in range(len(model.classes_)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], proba[:, i])
        aucs.append(auc(fpr, tpr))
        tprs.append(np.interp(fpr_grid, fpr, tpr))
    macro_tpr = np.mean(tprs, axis=0)
    macro_auc = float(np.mean(aucs))

    return {
        "name": name,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "auc": round(macro_auc, 4),
        "confusion_matrix": cm.tolist(),
        "confusion_labels": model.classes_.tolist(),
        "roc_fpr": fpr_grid.tolist(),
        "roc_tpr": macro_tpr.tolist(),
    }

dt_metrics = evaluate(dt_best, "Decision Tree")
rf_metrics = evaluate(rf_best, "Random Forest")
dt_metrics["cv_accuracy_mean"] = round(float(dt_cv.mean()), 4)
dt_metrics["cv_accuracy_std"] = round(float(dt_cv.std()), 4)
rf_metrics["cv_accuracy_mean"] = round(float(rf_cv.mean()), 4)
rf_metrics["cv_accuracy_std"] = round(float(rf_cv.std()), 4)
dt_metrics["best_params"] = dt_grid.best_params_
rf_metrics["best_params"] = rf_grid.best_params_

feature_importance = sorted(
    zip(FEATURES, rf_best.feature_importances_.tolist()),
    key=lambda t: -t[1]
)

# ---------- Export trees to JSON for real client-side inference ----------
def export_tree(tree, classes):
    t = tree.tree_
    n = t.node_count
    feature = t.feature.tolist()
    threshold = t.threshold.tolist()
    left = t.children_left.tolist()
    right = t.children_right.tolist()
    value = t.value.reshape(n, -1)
    value = (value / value.sum(axis=1, keepdims=True))
    value = np.round(value, 4).tolist()
    threshold = [round(x, 5) for x in threshold]
    return {"feature": feature, "threshold": threshold, "left": left, "right": right, "value": value}

rf_trees = [export_tree(est, rf_best.classes_) for est in rf_best.estimators_]
dt_tree = export_tree(dt_best, dt_best.classes_)

model_bundle = {
    "features": FEATURES,
    "classes": rf_best.classes_.tolist(),
    "representative_score": {k: int(v) for k, v in rep_score.items()},
    "decision_tree": dt_tree,
    "decision_tree_classes": dt_best.classes_.tolist(),
    "random_forest": rf_trees,
    "dataset_info": {
        "source": "UCI Machine Learning Repository — Wine Quality (red wine) Dataset, P. Cortez et al.",
        "samples_raw": before,
        "samples_after_dedup": after,
        "duplicates_removed": before - after,
        "missing_values_found": missing,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "class_distribution": {k: int(v) for k, v in pd.Series(y).value_counts().to_dict().items()}
    }
}

metrics_bundle = {
    "decision_tree": dt_metrics,
    "random_forest": rf_metrics,
    "feature_importance": feature_importance,
    "better_model": "Random Forest" if rf_metrics["f1_score"] >= dt_metrics["f1_score"] else "Decision Tree"
}

with open("model_bundle.json", "w") as f:
    json.dump(model_bundle, f)

with open("metrics.json", "w") as f:
    json.dump(metrics_bundle, f, indent=2)

# ---------- Save real sklearn models for the Flask backend ----------
joblib.dump({
    "decision_tree": dt_best,
    "random_forest": rf_best,
    "features": FEATURES,
    "representative_score": {k: int(v) for k, v in rep_score.items()},
}, "wine_model.joblib")

print(json.dumps(metrics_bundle, indent=2))
print("RF trees:", len(rf_trees), "nodes per tree (avg):", np.mean([len(t["feature"]) for t in rf_trees]))
print("Saved wine_model.joblib for the Flask backend.")
