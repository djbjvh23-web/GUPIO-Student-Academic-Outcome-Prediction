# ============================================================
# GUPIO AI/ML ENGINEER ASSIGNMENT
# Student Academic Outcome Prediction
#
# Target:
#   Dropout / Enrolled / Graduate
#
# Models:
#   1. Logistic Regression
#   2. Random Forest
#
# Important:
#   The primary model must use only information available
#   at student enrollment time.
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

DATA_PATH = "data/data.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("GUPIO - STUDENT ACADEMIC OUTCOME PREDICTION")
print("=" * 70)


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\n[1] Loading dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"\nDataset not found at: {DATA_PATH}\n"
        "Please place the Gupio-supplied data.csv inside the data folder."
    )

# Gupio says the supplied CSV is semicolon-delimited.
df = pd.read_csv(DATA_PATH, sep=";")

print("Dataset loaded successfully.")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# 4. BASIC DATA INSPECTION
# ============================================================

print("\n" + "=" * 70)
print("[2] DATASET INSPECTION")
print("=" * 70)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
for column in df.columns:
    print("-", column)

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

if missing_values.sum() == 0:
    print("No missing values found.")

print("\nDuplicate records:")
duplicate_count = df.duplicated().sum()
print(duplicate_count)


# ============================================================
# 5. TARGET COLUMN
# ============================================================

TARGET_COLUMN = "Target"

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"\nTarget column '{TARGET_COLUMN}' was not found.\n"
        f"Available columns are:\n{list(df.columns)}"
    )

print("\nTarget distribution:")
target_counts = df[TARGET_COLUMN].value_counts()

print(target_counts)

print("\nTarget percentages:")
print(
    (df[TARGET_COLUMN].value_counts(normalize=True) * 100)
    .round(2)
)


# ============================================================
# 6. TARGET DISTRIBUTION VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 5))

target_counts.plot(kind="bar")

plt.title("Student Academic Outcome Distribution")
plt.xlabel("Academic Outcome")
plt.ylabel("Number of Students")
plt.xticks(rotation=0)
plt.tight_layout()

target_plot_path = os.path.join(
    OUTPUT_DIR,
    "target_distribution.png"
)

plt.savefig(target_plot_path, dpi=300)
plt.show()

print(f"\nTarget distribution saved to: {target_plot_path}")


# ============================================================
# 7. IDENTIFY DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("[3] FEATURE TYPE ANALYSIS")
print("=" * 70)

X_all = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

numeric_columns = X_all.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_columns = X_all.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical columns:")
for column in numeric_columns:
    print("-", column)

print("\nCategorical columns:")
for column in categorical_columns:
    print("-", column)


# ============================================================
# 8. IDENTIFY PROHIBITED SEMESTER-PERFORMANCE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("[4] DATA LEAKAGE CHECK")
print("=" * 70)

# Expected names from the Gupio assignment.
# The matching below is made robust to minor whitespace differences.

PROHIBITED_FEATURES = [
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
    "Curricular units 2nd sem (without evaluations)"
]


def normalize_column_name(name):
    """
    Normalize column names only for matching purposes.
    The original dataset itself is NOT modified.
    """
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "")
    )


normalized_actual_columns = {
    normalize_column_name(column): column
    for column in df.columns
}

leakage_columns_found = []

for prohibited_column in PROHIBITED_FEATURES:

    normalized_prohibited = normalize_column_name(
        prohibited_column
    )

    if normalized_prohibited in normalized_actual_columns:
        actual_name = normalized_actual_columns[
            normalized_prohibited
        ]

        leakage_columns_found.append(actual_name)


print("\nProhibited semester-performance columns found:")

if leakage_columns_found:
    for column in leakage_columns_found:
        print("LEAKAGE COLUMN:", column)
else:
    print("WARNING: No prohibited columns were matched.")
    print(
        "Check the actual column names before continuing."
    )


# ============================================================
# 9. REMOVE LEAKAGE FEATURES
# ============================================================

X = df.drop(columns=[TARGET_COLUMN])

X = X.drop(
    columns=leakage_columns_found,
    errors="ignore"
)

print("\nOriginal feature count:", X_all.shape[1])
print("Features after leakage removal:", X.shape[1])

print("\nRemoved leakage features:")
for column in leakage_columns_found:
    print("-", column)


# ============================================================
# 10. BASIC EDA OF NUMERICAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("[5] BASIC EDA")
print("=" * 70)

print("\nNumerical feature summary:")
print(X.describe().T)


# ============================================================
# 11. SIMPLE FEATURE-VS-TARGET ANALYSIS
# ============================================================

# Select a few numeric columns for visualization.
# This is only for EDA; the model uses all permitted features.

available_numeric = [
    column
    for column in X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns
]

# Avoid creating too many charts.
eda_columns = available_numeric[:6]

print("\nFeatures selected for basic EDA:")
for column in eda_columns:
    print("-", column)


for column in eda_columns:

    plt.figure(figsize=(8, 5))

    for target_class in sorted(y.unique()):

        subset = X.loc[
            y == target_class,
            column
        ]

        plt.hist(
            subset.dropna(),
            alpha=0.5,
            label=str(target_class),
            bins=20
        )

    plt.title(f"{column} by Academic Outcome")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()

    safe_name = (
        str(column)
        .replace("/", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
    )

    path = os.path.join(
        OUTPUT_DIR,
        f"eda_{safe_name}.png"
    )

    plt.savefig(path, dpi=300)
    plt.show()


# ============================================================
# 12. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("[6] TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records :", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# ============================================================
# 13. PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("[7] PREPROCESSING")
print("=" * 70)

numeric_features = X_train.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("Numeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))


# Numeric preprocessing
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# Categorical preprocessing
# This handles categorical columns if the supplied dataset contains any.
# handle_unknown='ignore' prevents errors for unseen test categories.
categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            __import__(
                "sklearn.preprocessing",
                fromlist=["OneHotEncoder"]
            ).OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ],
    remainder="drop"
)


# ============================================================
# 14. MODEL 1 — LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("[8] MODEL 1 — LOGISTIC REGRESSION")
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
        )
    ]
)

print("Training Logistic Regression...")

logistic_model.fit(
    X_train,
    y_train
)

logistic_predictions = logistic_model.predict(
    X_test
)

print("Logistic Regression training complete.")


# ============================================================
# 15. MODEL 2 — RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("[9] MODEL 2 — RANDOM FOREST")
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
                n_jobs=-1
            )
        )
    ]
)

print("Training Random Forest...")

random_forest_model.fit(
    X_train,
    y_train
)

random_forest_predictions = random_forest_model.predict(
    X_test
)

print("Random Forest training complete.")


# ============================================================
# 16. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    y_true,
    predictions
):

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

    print("\n" + "-" * 70)
    print(model_name)
    print("-" * 70)

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
# 17. EVALUATE BOTH MODELS
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
# 18. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    [
        logistic_results,
        random_forest_results
    ]
)

print("\n" + "=" * 70)
print("[11] MODEL COMPARISON")
print("=" * 70)

print(results_df)

results_path = os.path.join(
    OUTPUT_DIR,
    "model_comparison.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print(
    f"\nModel comparison saved to: {results_path}"
)


# ============================================================
# 19. CONFUSION MATRIX — LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("[12] CONFUSION MATRIX")
print("=" * 70)

classes = sorted(y.unique())

cm_logistic = confusion_matrix(
    y_test,
    logistic_predictions,
    labels=classes
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_logistic,
    display_labels=classes
)

fig, ax = plt.subplots(figsize=(7, 6))

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()

logistic_cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix_logistic_regression.png"
)

plt.savefig(
    logistic_cm_path,
    dpi=300
)

plt.show()


# ============================================================
# 20. CONFUSION MATRIX — RANDOM FOREST
# ============================================================

cm_rf = confusion_matrix(
    y_test,
    random_forest_predictions,
    labels=classes
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf,
    display_labels=classes
)

fig, ax = plt.subplots(figsize=(7, 6))

disp.plot(
    ax=ax,
    cmap="Greens",
    values_format="d"
)

plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()

rf_cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix_random_forest.png"
)

plt.savefig(
    rf_cm_path,
    dpi=300
)

plt.show()


# ============================================================
# 21. CLASS IMBALANCE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("[13] CLASS IMBALANCE ANALYSIS")
print("=" * 70)

class_distribution = (
    y.value_counts()
    .rename("Count")
    .to_frame()
)

class_distribution["Percentage"] = (
    class_distribution["Count"]
    / len(y)
    * 100
)

print(class_distribution.round(2))


largest_class = class_distribution["Count"].idxmax()
smallest_class = class_distribution["Count"].idxmin()

print(
    f"\nLargest class : {largest_class}"
)

print(
    f"Smallest class: {smallest_class}"
)


# ============================================================
# 22. FINAL MODEL SELECTION
# ============================================================

print("\n" + "=" * 70)
print("[14] FINAL MODEL SELECTION")
print("=" * 70)

# Primary selection criterion:
# Macro F1 because the assignment explicitly requires
# attention to all three classes and class imbalance.

best_model_name = results_df.loc[
    results_df["Macro F1"].idxmax(),
    "Model"
]

if best_model_name == "Logistic Regression":

    final_model = logistic_model
    final_predictions = logistic_predictions

else:

    final_model = random_forest_model
    final_predictions = random_forest_predictions


print(
    f"Selected final model: {best_model_name}"
)

print(
    "\nSelection criterion: highest Macro F1."
)

print(
    "The final decision should also be discussed using "
    "per-class performance and the confusion matrix."
)


# ============================================================
# 23. FINAL MODEL METRICS
# ============================================================

final_accuracy = accuracy_score(
    y_test,
    final_predictions
)

final_precision = precision_score(
    y_test,
    final_predictions,
    average="macro",
    zero_division=0
)

final_recall = recall_score(
    y_test,
    final_predictions,
    average="macro",
    zero_division=0
)

final_f1 = f1_score(
    y_test,
    final_predictions,
    average="macro",
    zero_division=0
)

print("\nFinal Model Results:")
print(f"Model           : {best_model_name}")
print(f"Accuracy        : {final_accuracy:.4f}")
print(f"Macro Precision : {final_precision:.4f}")
print(f"Macro Recall    : {final_recall:.4f}")
print(f"Macro F1        : {final_f1:.4f}")


# ============================================================
# 24. SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("[15] SAMPLE PREDICTIONS")
print("=" * 70)

# Select 5 actual test records.
sample_students = X_test.head(5)

sample_predictions = final_model.predict(
    sample_students
)

sample_output = sample_students.copy()

sample_output["Actual Outcome"] = (
    y_test.loc[sample_students.index].values
)

sample_output["Predicted Outcome"] = (
    sample_predictions
)

print(
    sample_output[
        [
            "Actual Outcome",
            "Predicted Outcome"
        ]
    ]
)

sample_path = os.path.join(
    OUTPUT_DIR,
    "sample_predictions.csv"
)

sample_output.to_csv(
    sample_path,
    index=False
)

print(
    f"\nSample predictions saved to: {sample_path}"
)


# ============================================================
# 25. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("[16] FEATURE INTERPRETATION")
print("=" * 70)

if best_model_name == "Random Forest":

    # Get the fitted preprocessing component.
    fitted_preprocessor = (
        final_model.named_steps["preprocessor"]
    )

    classifier = (
        final_model.named_steps["classifier"]
    )

    # Get feature names after preprocessing.
    try:
        feature_names = (
            fitted_preprocessor
            .get_feature_names_out()
        )

        importances = classifier.feature_importances_

        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importances
            }
        ).sort_values(
            by="Importance",
            ascending=False
        )

        print("\nTop 15 predictive features:")

        print(
            importance_df.head(15).to_string(
                index=False
            )
        )

        importance_path = os.path.join(
            OUTPUT_DIR,
            "feature_importance.csv"
        )

        importance_df.to_csv(
            importance_path,
            index=False
        )

        # Plot top 15 features.
        top_features = importance_df.head(15)

        plt.figure(figsize=(10, 7))

        plt.barh(
            top_features["Feature"][::-1],
            top_features["Importance"][::-1]
        )

        plt.title(
            "Top Predictive Features - Random Forest"
        )

        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")

        plt.tight_layout()

        feature_plot_path = os.path.join(
            OUTPUT_DIR,
            "feature_importance.png"
        )

        plt.savefig(
            feature_plot_path,
            dpi=300
        )

        plt.show()

        print(
            f"\nFeature importance saved to: "
            f"{importance_path}"
        )

    except Exception as error:

        print(
            "\nCould not extract feature importance."
        )

        print("Reason:", error)

else:

    # Logistic Regression coefficient interpretation.
    fitted_preprocessor = (
        final_model.named_steps["preprocessor"]
    )

    classifier = (
        final_model.named_steps["classifier"]
    )

    try:

        feature_names = (
            fitted_preprocessor
            .get_feature_names_out()
        )

        coefficients = classifier.coef_

        # Average absolute coefficient across classes.
        mean_absolute_coefficient = np.mean(
            np.abs(coefficients),
            axis=0
        )

        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Mean Absolute Coefficient":
                    mean_absolute_coefficient
            }
        ).sort_values(
            by="Mean Absolute Coefficient",
            ascending=False
        )

        print(
            "\nTop 15 predictive features:"
        )

        print(
            importance_df.head(15).to_string(
                index=False
            )
        )

        importance_path = os.path.join(
            OUTPUT_DIR,
            "feature_importance.csv"
        )

        importance_df.to_csv(
            importance_path,
            index=False
        )

        print(
            f"\nFeature interpretation saved to: "
            f"{importance_path}"
        )

    except Exception as error:

        print(
            "\nCould not extract Logistic Regression "
            "feature coefficients."
        )

        print("Reason:", error)


# ============================================================
# 26. SAVE FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("[17] SAVING FINAL MODEL")
print("=" * 70)

model_path = os.path.join(
    OUTPUT_DIR,
    "final_student_outcome_model.joblib"
)

joblib.dump(
    final_model,
    model_path
)

print(
    f"Final model saved to: {model_path}"
)


# ============================================================
# 27. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PROJECT COMPLETE")
print("=" * 70)

print(f"""
Final model:
{best_model_name}

Final metrics:
Accuracy        : {final_accuracy:.4f}
Macro Precision : {final_precision:.4f}
Macro Recall    : {final_recall:.4f}
Macro F1        : {final_f1:.4f}

Outputs generated in:
{OUTPUT_DIR}/

Important:
- Semester-performance leakage features were excluded.
- Train/test split was performed before model fitting.
- Preprocessing is inside the ML pipeline.
- Final model was selected using Macro F1.
- Actual model results are reported.
""")

print("=" * 70)