"""
Скрипт навчання та збереження моделі для прогнозу виживання на Титаніку.
"""

from pathlib import Path
import joblib
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

NUMERICAL_FEATURES = ["age", "fare", "sibsp", "parch"]
CATEGORICAL_FEATURES = ["pclass", "sex", "embarked"]
FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "survived"


def load_data() -> pd.DataFrame:
    raw_df = sns.load_dataset("titanic")
    return raw_df[[TARGET_COLUMN] + FEATURE_COLUMNS].copy()


def build_preprocessor() -> ColumnTransformer:
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", num_transformer, NUMERICAL_FEATURES),
        ("cat", cat_transformer, CATEGORICAL_FEATURES),
    ])


def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    results = {}

    # 1. Baseline
    print("1. Навчання Baseline (DummyClassifier)...")
    dummy = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DummyClassifier(strategy="most_frequent")),
    ])
    dummy.fit(X_train, y_train)
    pred_dummy = dummy.predict(X_test)
    proba_dummy = dummy.predict_proba(X_test)[:, 1]
    results["DummyClassifier"] = {
        "pipeline": dummy,
        "accuracy": accuracy_score(y_test, pred_dummy),
        "f1": f1_score(y_test, pred_dummy, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba_dummy),
        "params": "most_frequent",
    }

    # 2. Decision Tree
    print("2. Підбір параметрів для DecisionTreeClassifier...")
    dt = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42)),
    ])
    dt_grid = GridSearchCV(
        dt,
        {"classifier__max_depth": [3, 4, 5, 6, 8], "classifier__min_samples_split": [2, 5, 10]},
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    dt_grid.fit(X_train, y_train)
    best_dt = dt_grid.best_estimator_
    pred_dt = best_dt.predict(X_test)
    proba_dt = best_dt.predict_proba(X_test)[:, 1]
    results["DecisionTreeClassifier"] = {
        "pipeline": best_dt,
        "accuracy": accuracy_score(y_test, pred_dt),
        "f1": f1_score(y_test, pred_dt),
        "roc_auc": roc_auc_score(y_test, proba_dt),
        "params": dt_grid.best_params_,
    }

    # 3. MLP Classifier
    print("3. Підбір параметрів для MLPClassifier...")
    mlp = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", MLPClassifier(random_state=42, max_iter=800)),
    ])
    mlp_grid = GridSearchCV(
        mlp,
        {
            "classifier__hidden_layer_sizes": [(64,), (32, 16)],
            "classifier__alpha": [0.01, 0.1, 0.5],
            "classifier__learning_rate_init": [0.005, 0.01],
        },
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    mlp_grid.fit(X_train, y_train)
    best_mlp = mlp_grid.best_estimator_
    pred_mlp = best_mlp.predict(X_test)
    proba_mlp = best_mlp.predict_proba(X_test)[:, 1]
    results["MLPClassifier"] = {
        "pipeline": best_mlp,
        "accuracy": accuracy_score(y_test, pred_mlp),
        "f1": f1_score(y_test, pred_mlp),
        "roc_auc": roc_auc_score(y_test, proba_mlp),
        "params": mlp_grid.best_params_,
    }

    return results


def main():
    df = load_data()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor()
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor)

    print("\nРезультати моделей на тестовій вибірці:")
    for name, m in results.items():
        print(f"{name:24} | Acc: {m['accuracy']:.4f} | F1: {m['f1']:.4f} | ROC-AUC: {m['roc_auc']:.4f}")

    # Обираємо найкращу модель за комбінацією F1 та ROC-AUC
    best_name = max(["DecisionTreeClassifier", "MLPClassifier"], key=lambda k: (results[k]["f1"], results[k]["roc_auc"]))
    print(f"\nОбрана найкраща модель: {best_name}")

    output_path = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(results[best_name]["pipeline"], output_path)
    print(f"Модель збережено у {output_path}")


if __name__ == "__main__":
    main()
