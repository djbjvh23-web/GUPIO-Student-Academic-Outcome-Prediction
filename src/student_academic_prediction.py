# ============================================================
# GUPIO - STUDENT ACADEMIC OUTCOME PREDICTION
# Option 1: Student Academic Outcome Prediction
# ============================================================

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

RANDOM_STATE = 42

DATA_PATH = "data/dataset.csv"
OUTPUT_DIR = "outputs"
PLOTS_DIR = "outputs/plots"
RESULTS_DIR = "outputs/results"
MODELS_DIR = "models"


# Create folders if they do not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("GUPIO - STUDENT ACADEMIC OUTCOME PREDICTION")
print("=" * 70)

print("\n[1] Loading dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"\nDataset not found at: {DATA_PATH}\n"
        "Make sure your CSV is inside the data folder."
    )

# Gupio/UCI dataset uses semicolon-separated values
df = pd.read_csv(DATA_PATH, sep=";")

print(f"Dataset loaded successfully.")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# 3. BASIC DATASET INSPECTION
# ============================================================

print("\n" + "=" * 70)
print("[2] DATASET INSPECTION")
print("=" * 70)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())


# ============================================================
# 4. CHECK TARGET
# ============================================================

TARGET_COLUMN = "Target"

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"\nTarget column '{TARGET_COLUMN}' was not found.\n"
        f"Available columns are:\n{df.columns.tolist()}"
    )

print("\nTarget classes:")
print(df[TARGET_COLUMN].value_counts())

print("\nTarget percentages:")
print(
    df[TARGET_COLUMN]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 5. TARGET DISTRIBUTION PLOT
# ============================================================

print("\n[3] Creating target distribution plot...")

plt.figure(figsize=(8, 5))

df[TARGET_COLUMN].value_counts().plot(kind="bar")

plt.title("Student Academic Outcome Distribution")
plt.xlabel("Academic Outcome")
plt.ylabel("Number of Students")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(PLOTS_DIR, "target_distribution.png"),
    dpi=300
)

plt.close()

print("Saved: outputs/plots/target_distribution.png")


# ============================================================
# 6. REMOVE LEAKAGE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("[4] LEAKAGE PREVENTION")
print("=" * 70)

# These features contain semester-performance information.
# They must NOT be used for the primary prediction model.

LEAKAGE_FEATURES = [
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
]

existing_leakage_features = [
    column
    for column in LEAKAGE_FEATURES
    if column in df.columns
]

print("\nLeakage features found:")

if existing_leakage_features:
    for column in existing_leakage_features:
        print(" -", column)
else:
    print("No exact leakage feature names were found.")

# Remove leakage features
df_model = df.drop(
    columns=existing_leakage_features,
    errors="ignore"
)

print(
    f"\nNumber of columns after leakage removal: "
    f"{df_model.shape[1]}"
)


# ============================================================
# 7. SEPARATE FEATURES AND TARGET
# ============================================================

print("\n" + "=" * 70)
print("[5] PREPARING FEATURES AND TARGET")
print("=" * 70)

X = df_model.drop(columns=[TARGET_COLUMN])
y = df_model[TARGET_COLUMN]

print(f"Number of input features: {X.shape[1]}")
print(f"Number of target classes: {y.nunique()}")

print("\nTarget classes:")
print(sorted(y.unique()))


# ============================================================
# 8. IDENTIFY NUMERIC AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumeric features:")
print(len(numeric_features))

print("\nCategorical features:")
print(len(categorical_features))


# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("[6] TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")


# ============================================================
# 10. PREPROCESSING
# ============================================================

print("\n[7] Creating leakage-safe preprocessing pipeline...")

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
    ]
)


# ============================================================
# 11. MODEL 1 - LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("[8] MODEL 1 - LOGISTIC REGRESSION")
print("=" * 70)

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE
            )
        ),
    ]
)

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)


# ============================================================
# 12. MODEL 2 - RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("[9] MODEL 2 - RANDOM FOREST")
print("=" * 70)

random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced"
            )
        ),
    ]
)

random_forest_model.fit(X_train, y_train)

random_forest_predictions = random_forest_model.predict(X_test)


# ============================================================
# 13. EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_name, y_true, predictions):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0
    )

    print("\n" + "-" * 60)
    print(model_name)
    print("-" * 60)

    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Macro Precision   : {precision:.4f}")
    print(f"Macro Recall      : {recall:.4f}")
    print(f"Macro F1          : {f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            predictions,
            zero_division=0
        )
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1
    }


# ============================================================
# 14. EVALUATE BOTH MODELS
# ============================================================

print("\n" + "=" * 70)
print("[10] MODEL EVALUATION")
print("=" * 70)

logistic_results = evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_predictions
)

random_forest_results = evaluate_model(
    "Random Forest",
    y_test,
    random_forest_predictions
)


# ============================================================
# 15. MODEL COMPARISON
# ============================================================

results = pd.DataFrame(
    [
        logistic_results,
        random_forest_results
    ]
)

print("\n" + "=" * 70)
print("[11] MODEL COMPARISON")
print("=" * 70)

print(results.to_string(index=False))

results.to_csv(
    os.path.join(
        RESULTS_DIR,
        "model_comparison.csv"
    ),
    index=False
)

print(
    "\nSaved: outputs/results/model_comparison.csv"
)


# ============================================================
# 16. CONFUSION MATRICES
# ============================================================

print("\n[12] Creating confusion matrices...")

fig, ax = plt.subplots(figsize=(7, 6))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    logistic_predictions,
    ax=ax,
    cmap="Blues"
)

plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "confusion_matrix_logistic_regression.png"
    ),
    dpi=300
)

plt.close()


fig, ax = plt.subplots(figsize=(7, 6))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    random_forest_predictions,
    ax=ax,
    cmap="Greens"
)

plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "confusion_matrix_random_forest.png"
    ),
    dpi=300
)

plt.close()

print("Confusion matrices saved.")


# ============================================================
# 17. SELECT FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("[13] FINAL MODEL SELECTION")
print("=" * 70)

best_index = results["Macro F1"].idxmax()

best_model_name = results.loc[
    best_index,
    "Model"
]

if best_model_name == "Logistic Regression":
    final_model = logistic_model
    final_predictions = logistic_predictions
else:
    final_model = random_forest_model
    final_predictions = random_forest_predictions

print(f"Selected final model: {best_model_name}")

print(
    f"Final Macro F1: "
    f"{results.loc[best_index, 'Macro F1']:.4f}"
)


# ============================================================
# 18. SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODELS_DIR,
    "final_model.joblib"
)

joblib.dump(
    final_model,
    final_model_path
)

print(
    f"\nFinal model saved to: {final_model_path}"
)


# ============================================================
# 19. SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("[14] SAMPLE PREDICTIONS")
print("=" * 70)

sample_count = min(10, len(X_test))

sample_X = X_test.iloc[:sample_count]

sample_actual = y_test.iloc[:sample_count]

sample_predicted = final_model.predict(
    sample_X
)

sample_predictions = pd.DataFrame(
    {
        "Actual": sample_actual.values,
        "Predicted": sample_predicted
    }
)

print(sample_predictions.to_string(index=False))

sample_predictions.to_csv(
    os.path.join(
        RESULTS_DIR,
        "sample_predictions.csv"
    ),
    index=False
)


# ============================================================
# 20. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("[15] FEATURE INTERPRETATION")
print("=" * 70)

try:

    classifier = final_model.named_steps["classifier"]

    fitted_preprocessor = final_model.named_steps[
        "preprocessor"
    ]

    feature_names = fitted_preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):

        importance_values = classifier.feature_importances_

    elif hasattr(classifier, "coef_"):

        importance_values = np.mean(
            np.abs(classifier.coef_),
            axis=0
        )

    else:

        importance_values = None

    if importance_values is not None:

        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importance_values
            }
        )

        importance_df = importance_df.sort_values(
            "Importance",
            ascending=False
        )

        print("\nTop 15 predictive features:")

        print(
            importance_df.head(15).to_string(
                index=False
            )
        )

        importance_df.head(15).to_csv(
            os.path.join(
                RESULTS_DIR,
                "top_features.csv"
            ),
            index=False
        )

except Exception as error:

    print(
        "Feature importance could not be calculated:"
    )

    print(error)


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PROJECT COMPLETED")
print("=" * 70)

print(f"""
Dataset:
    {DATA_PATH}

Training samples:
    {len(X_train)}

Testing samples:
    {len(X_test)}

Logistic Regression Macro F1:
    {logistic_results["Macro F1"]:.4f}

Random Forest Macro F1:
    {random_forest_results["Macro F1"]:.4f}

Selected Model:
    {best_model_name}

Important output folders:
    outputs/plots/
    outputs/results/
    models/

Final model:
    models/final_model.joblib
""")

print("=" * 70)
print("DONE")
print("=" * 70)