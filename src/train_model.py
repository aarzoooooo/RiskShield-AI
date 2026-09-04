import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "data/model_data.csv"
MODEL_FILE = "data/risk_model.pkl"

TARGET = "Is Account Takeover"


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)

print("\nTarget distribution:")
print(df[TARGET].value_counts())


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

y = df[TARGET].astype(int)

X = df.drop(columns=[TARGET]).copy()


# ============================================================
# 3. REMOVE RAW TIMESTAMP / ID COLUMNS
# ============================================================

# Raw timestamps are not useful in their original string form.
# We already have engineered time features such as:
# login_hour, login_day, is_weekend, etc.

columns_to_drop = []

for col in X.columns:

    col_lower = col.lower()

    # Remove raw timestamp/date columns
    if (
        "timestamp" in col_lower
        or "datetime" in col_lower
        or col_lower == "date"
    ):
        columns_to_drop.append(col)

    # Remove identifiers
    elif col_lower in [
        "user id",
        "user_id",
        "account id",
        "account_id",
        "session id",
        "session_id",
        "index"
    ]:
        columns_to_drop.append(col)


if columns_to_drop:

    print("\nDropping columns:")
    print(columns_to_drop)

    X = X.drop(columns=columns_to_drop)


# ============================================================
# 4. CONVERT BOOLEAN COLUMNS
# ============================================================

for col in X.columns:

    if X[col].dtype == "bool":

        X[col] = X[col].astype(int)


# ============================================================
# 5. HANDLE CATEGORICAL / STRING COLUMNS
# ============================================================

print("\nProcessing feature columns...")

category_mappings = {}

for col in X.columns:

    if X[col].dtype == "object":

        try:
            converted = pd.to_datetime(
                X[col],
                errors="coerce",
                format="mixed"
            )

            valid_ratio = converted.notna().mean()

        except Exception:
            valid_ratio = 0

        if valid_ratio > 0.80:

            X[col] = (
                converted.astype("int64") // 10**9
            )

        else:

            # Save the exact category mapping used during training
            X[col] = X[col].astype("category")

            categories = X[col].cat.categories.tolist()

            category_mappings[col] = {
                value: code
                for code, value in enumerate(categories)
            }

            X[col] = X[col].cat.codes


# ============================================================
# 6. HANDLE MISSING / INFINITE VALUES
# ============================================================

X = X.replace([np.inf, -np.inf], np.nan)

X = X.fillna(0)


# Make absolutely sure everything is numeric
for col in X.columns:

    if not pd.api.types.is_numeric_dtype(X[col]):

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        ).fillna(0)


print("\nFinal feature columns:")
print(X.columns.tolist())

print("\nNumber of features:", X.shape[1])


# ============================================================
# 7. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("SPLITTING DATA")
print("=" * 60)

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

print("\nTraining target:")
print(y_train.value_counts())

print("\nValidation target:")
print(y_val.value_counts())

print("\nTest target:")
print(y_test.value_counts())


# ============================================================
# 8. TRAIN RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=2,
    max_features="sqrt",

    # Important for our highly imbalanced dataset
    class_weight="balanced_subsample",

    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training complete!")


# ============================================================
# 9. VALIDATION PROBABILITIES
# ============================================================

val_prob = model.predict_proba(X_val)[:, 1]


print("\n" + "=" * 60)
print("VALIDATION METRICS")
print("=" * 60)

print(
    "ROC-AUC:",
    round(roc_auc_score(y_val, val_prob), 4)
)

print(
    "PR-AUC:",
    round(average_precision_score(y_val, val_prob), 4)
)


# ============================================================
# 10. FIND BEST THRESHOLD
# ============================================================

print("\n" + "=" * 60)
print("THRESHOLD SEARCH")
print("=" * 60)

precision, recall, thresholds = precision_recall_curve(
    y_val,
    val_prob
)

# Calculate F1 manually
f1_scores = (
    2 * precision * recall /
    (precision + recall + 1e-10)
)

best_index = np.argmax(f1_scores)

# precision_recall_curve returns one extra precision/recall value
if best_index < len(thresholds):

    best_threshold = thresholds[best_index]

else:

    best_threshold = 0.5


best_f1 = f1_scores[best_index]


print(
    "Best threshold:",
    round(float(best_threshold), 4)
)

print(
    "Best validation F1:",
    round(float(best_f1), 4)
)


# ============================================================
# 11. TEST SET
# ============================================================

test_prob = model.predict_proba(X_test)[:, 1]

test_pred = (
    test_prob >= best_threshold
).astype(int)


# ============================================================
# 12. FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL RESULTS — HELD-OUT TEST SET")
print("=" * 60)

test_precision = precision_score(
    y_test,
    test_pred,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_pred,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_pred,
    zero_division=0
)

print("Precision:", round(test_precision, 4))
print("Recall:", round(test_recall, 4))
print("F1 Score:", round(test_f1, 4))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_pred,
        target_names=["Normal", "Account Takeover"],
        zero_division=0
    )
)
# ============================================================
# 13. CONFUSION MATRIX + FALSE POSITIVE COST
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_pred
).ravel()

print("True Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)


# False Positive Rate
false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

print(
    "\nFalse Positive Rate:",
    round(false_positive_rate, 4)
)

print(
    "False Positive Rate (%):",
    round(false_positive_rate * 100, 2),
    "%"
)

print(
    "False Positives:",
    fp
)

print(
    "False Positives / Legitimate Logins:",
    f"{fp} / {fp + tn}"
)

# ============================================================
# 14. FINAL ROC-AUC / PR-AUC
# ============================================================

print("\nROC-AUC:")

print(
    round(
        roc_auc_score(
            y_test,
            test_prob
        ),
        4
    )
)


print("\nPR-AUC:")

print(
    round(
        average_precision_score(
            y_test,
            test_prob
        ),
        4
    )
)


# ============================================================
# 15. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("TOP FEATURES")
print("=" * 60)

importance = pd.DataFrame({

    "feature": X.columns,

    "importance":
        model.feature_importances_

})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(
    importance.head(15).to_string(
        index=False
    )
)


# ============================================================
# 16. SAVE MODEL
# ============================================================

model_package = {

    "model": model,

    "threshold":
        float(best_threshold),

    "features":
        X.columns.tolist(),

    "category_mappings":
        category_mappings,

    "metrics": {

        "precision":
            float(test_precision),

        "recall":
            float(test_recall),

        "f1":
            float(test_f1),

        "roc_auc":
            float(roc_auc_score(y_test, test_prob)),

        "pr_auc":
            float(average_precision_score(y_test, test_prob)),

        "false_positive_rate":
            float(false_positive_rate),

        "false_positives":
            int(fp),

        "false_negatives":
            int(fn),

        "true_positives":
            int(tp),

        "true_negatives":
            int(tn),

        "test_samples":
            int(len(y_test))

    }

}


joblib.dump(
    model_package,
    MODEL_FILE
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(
    f"Saved to: {MODEL_FILE}"
)

print("\nDONE!")
