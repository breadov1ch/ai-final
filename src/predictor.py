"""
Модуль завантаження моделі, валідації вхідних даних та розрахунку прогнозу.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import pandas as pd

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"

MAX_TRAINING_AGE = 80.0
MAX_TRAINING_FARE = 512.33
MAX_TRAINING_SIBSP = 8
MAX_TRAINING_PARCH = 6

_CACHED_MODEL = None


def load_model(model_path: Optional[Path] = None):
    """Завантажує модель один раз і кешує її."""
    global _CACHED_MODEL
    if _CACHED_MODEL is not None and model_path is None:
        return _CACHED_MODEL

    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Файл моделі не знайдено: {path}. Запустіть python src/train.py")

    model = joblib.load(path)
    if model_path is None:
        _CACHED_MODEL = model
    return model


def validate_input(data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Перевіряє типи та діапазони значень."""
    if not isinstance(data, dict):
        raise TypeError("Дані мають бути передані як словник.")

    required_fields = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    for field in required_fields:
        if field not in data or data[field] is None:
            raise ValueError(f"Відсутнє або порожнє обов'язкове поле: '{field}'.")

    try:
        pclass = int(data["pclass"])
        age = float(data["age"])
        sibsp = int(data["sibsp"])
        parch = int(data["parch"])
        fare = float(data["fare"])
    except (ValueError, TypeError) as err:
        raise ValueError(f"Помилка формату числових даних: {err}")

    if pclass not in [1, 2, 3]:
        raise ValueError(f"Клас каюти (pclass) має бути 1, 2 або 3. Отримано: {pclass}")
    if age < 0:
        raise ValueError(f"Вік не може бути від'ємним. Отримано: {age}")
    if sibsp < 0:
        raise ValueError(f"Кількість родичів (sibsp) не може бути від'ємною. Отримано: {sibsp}")
    if parch < 0:
        raise ValueError(f"Кількість родичів (parch) не може бути від'ємною. Отримано: {parch}")
    if fare < 0:
        raise ValueError(f"Ціна квитка (fare) не може бути від'ємною. Отримано: {fare}")

    sex = str(data["sex"]).strip().lower()
    if sex not in ["male", "female"]:
        raise ValueError("Стать (sex) повинна бути 'male' або 'female'.")

    embarked = str(data["embarked"]).strip().upper()
    if embarked not in ["S", "C", "Q"]:
        raise ValueError("Порт посадки (embarked) повинен бути 'S', 'C' або 'Q'.")

    warnings: List[str] = []
    if age > MAX_TRAINING_AGE:
        warnings.append(f"Вік ({age:.0f} р.) перевищує максимальний у навчальному датасеті (80 років).")
    if fare > MAX_TRAINING_FARE:
        warnings.append(f"Ціна квитка (£{fare:.2f}) більша за максимум у датасеті (£512.33).")
    if sibsp > MAX_TRAINING_SIBSP:
        warnings.append(f"Кількість братів/сестер ({sibsp}) більша за типовий максимум датасету ({MAX_TRAINING_SIBSP}).")
    if parch > MAX_TRAINING_PARCH:
        warnings.append(f"Кількість батьків/дітей ({parch}) більша за типовий максимум датасету ({MAX_TRAINING_PARCH}).")

    return {
        "pclass": pclass,
        "sex": sex,
        "age": age,
        "sibsp": sibsp,
        "parch": parch,
        "fare": fare,
        "embarked": embarked,
    }, warnings


def _explain_factors(passenger: Dict[str, Any], proba: float) -> Tuple[str, List[Dict[str, Any]]]:
    """Формує просте пояснення впливу ознак на прогноз."""
    factors = []

    if passenger["sex"] == "female":
        factors.append({
            "feature": "Жіноча стать",
            "impact": "positive",
            "description": "Правило 'жінки та діти першими' суттєво підвищувало шанси."
        })
    else:
        factors.append({
            "feature": "Чоловіча стать",
            "impact": "negative",
            "description": "Чоловіків саджали в шлюпки в останню чергу."
        })

    if passenger["pclass"] == 1:
        factors.append({
            "feature": "1-й клас",
            "impact": "positive",
            "description": "Каюти знаходилися на верхніх палубах із прямим доступом до шлюпок."
        })
    elif passenger["pclass"] == 3:
        factors.append({
            "feature": "3-й клас",
            "impact": "negative",
            "description": "Нижні палуби судна та складна дорога до виходів зменшували шанси."
        })

    if passenger["age"] < 14:
        factors.append({
            "feature": f"Вік дитини ({passenger['age']:.0f} р.)",
            "impact": "positive",
            "description": "Діти мали пріоритет при евакуації."
        })
    elif passenger["age"] > 60:
        factors.append({
            "feature": f"Похилий вік ({passenger['age']:.0f} р.)",
            "impact": "negative",
            "description": "Літнім людям було важче дістатися шлюпок у паніці."
        })

    family_size = passenger["sibsp"] + passenger["parch"] + 1
    if 2 <= family_size <= 4:
        factors.append({
            "feature": f"Невелика сім'я ({family_size} ос.)",
            "impact": "positive",
            "description": "Сім'ї з 2-4 людей підтримували одне одного при посадці."
        })
    elif family_size > 4:
        factors.append({
            "feature": f"Велика сім'я ({family_size} ос.)",
            "impact": "negative",
            "description": "Великій родині було важко зібратися разом і знайти вільні місця в одній шлюпці."
        })

    positives = [f["feature"] for f in factors if f["impact"] == "positive"]
    negatives = [f["feature"] for f in factors if f["impact"] == "negative"]

    if proba >= 0.5:
        summary = f"Ймовірність порятунку: {proba * 100:.1f}%. Фактори на користь: {', '.join(positives) if positives else 'сприятливі параметри'}."
    else:
        summary = f"Ймовірність порятунку: {proba * 100:.1f}%. Головні фактори ризику: {', '.join(negatives) if negatives else 'загальна статистика'}."

    return summary, factors


def predict_survival(passenger_data: Dict[str, Any], model_path: Optional[Path] = None) -> Dict[str, Any]:
    """Головна функція інференсу."""
    model = load_model(model_path)
    cleaned_data, warnings = validate_input(passenger_data)
    df_input = pd.DataFrame([cleaned_data])

    raw_prediction = int(model.predict(df_input)[0])
    survival_proba = float(model.predict_proba(df_input)[0][1])
    confidence = float(max(survival_proba, 1.0 - survival_proba) * 100.0)

    verdict = "Вижив би" if raw_prediction == 1 else "Не вижив би"
    explanation, factors = _explain_factors(cleaned_data, survival_proba)

    return {
        "prediction": raw_prediction,
        "survival_probability": round(survival_proba * 100.0, 1),
        "confidence": round(confidence, 1),
        "verdict": verdict,
        "warnings": warnings,
        "explanation": explanation,
        "factors": factors,
        "passenger": cleaned_data,
    }
