"""
Автоматичні модульні та інтеграційні тести для перевірки ML-моделі та предиктора.
"""

from pathlib import Path
import pytest
from src.predictor import DEFAULT_MODEL_PATH, load_model, predict_survival, validate_input


def test_model_file_exists():
    """Перевіряє наявність файлу збереженої моделі на диску."""
    assert DEFAULT_MODEL_PATH.exists(), f"Файл {DEFAULT_MODEL_PATH} не знайдено! Спочатку запустіть train.py."


def test_model_pipeline_structure():
    """Перевіряє, що завантажений об'єкт є валідним пайплайном з препроцесором та класифікатором."""
    model = load_model()
    assert hasattr(model, "predict"), "Модель повинна мати метод predict."
    assert hasattr(model, "predict_proba"), "Модель повинна мати метод predict_proba."
    assert "preprocessor" in model.named_steps, "Пайплайн повинен містити крок 'preprocessor'."
    assert "classifier" in model.named_steps, "Пайплайн повинен містити крок 'classifier'."


def test_predict_survival_valid_output_schema():
    """Перевіряє коректність структури та типів значень у відповіді предиктора."""
    passenger = {
        "pclass": 2,
        "sex": "female",
        "age": 28.0,
        "sibsp": 1,
        "parch": 0,
        "fare": 26.0,
        "embarked": "S",
    }
    result = predict_survival(passenger)

    assert isinstance(result, dict)
    for key in [
        "prediction",
        "survival_probability",
        "confidence",
        "verdict",
        "warnings",
        "explanation",
        "factors",
        "passenger",
    ]:
        assert key in result, f"Ключ '{key}' відсутній у відповіді предиктора."

    assert result["prediction"] in [0, 1]
    assert 0.0 <= result["survival_probability"] <= 100.0
    assert 50.0 <= result["confidence"] <= 100.0
    assert isinstance(result["warnings"], list)
    assert len(result["warnings"]) == 0
    assert isinstance(result["explanation"], str) and len(result["explanation"]) > 0
    assert isinstance(result["factors"], list) and len(result["factors"]) > 0


def test_predict_rose_vs_jack_historical_logic():
    """Перевіряє історичну достовірність: у 1-му класі для жінки шанс значно вищий, ніж для чоловіка 3-го класу."""
    rose = {
        "pclass": 1,
        "sex": "female",
        "age": 17.0,
        "sibsp": 0,
        "parch": 1,
        "fare": 120.0,
        "embarked": "S",
    }
    jack = {
        "pclass": 3,
        "sex": "male",
        "age": 20.0,
        "sibsp": 0,
        "parch": 0,
        "fare": 7.25,
        "embarked": "S",
    }

    res_rose = predict_survival(rose)
    res_jack = predict_survival(jack)

    assert res_rose["survival_probability"] > res_jack["survival_probability"]
    assert res_rose["prediction"] == 1
    assert res_jack["prediction"] == 0


def test_boundary_warnings():
    """Перевіряє генерацію попереджень при виході параметрів за межі навчального датасету."""
    passenger_extreme = {
        "pclass": 1,
        "sex": "male",
        "age": 95.0,  # > 80
        "sibsp": 10,  # > 8
        "parch": 8,   # > 6
        "fare": 750.0,  # > 512.33
        "embarked": "C",
    }
    result = predict_survival(passenger_extreme)
    assert len(result["warnings"]) >= 4
    warning_text = " ".join(result["warnings"])
    assert "95.0" in warning_text or "80" in warning_text
    assert "750" in warning_text or "512" in warning_text


@pytest.mark.parametrize(
    "invalid_data, expected_err",
    [
        ({"pclass": 1, "sex": "male", "age": -5, "sibsp": 0, "parch": 0, "fare": 10, "embarked": "S"}, "від'ємним"),
        ({"pclass": 1, "sex": "male", "age": 25, "sibsp": -1, "parch": 0, "fare": 10, "embarked": "S"}, "від'ємною"),
        ({"pclass": 1, "sex": "male", "age": 25, "sibsp": 0, "parch": -1, "fare": 10, "embarked": "S"}, "від'ємною"),
        ({"pclass": 1, "sex": "male", "age": 25, "sibsp": 0, "parch": 0, "fare": -20, "embarked": "S"}, "від'ємною"),
        ({"pclass": 4, "sex": "male", "age": 25, "sibsp": 0, "parch": 0, "fare": 10, "embarked": "S"}, "pclass"),
        ({"pclass": 1, "sex": "robot", "age": 25, "sibsp": 0, "parch": 0, "fare": 10, "embarked": "S"}, "sex"),
        ({"pclass": 1, "sex": "male", "age": 25, "sibsp": 0, "parch": 0, "fare": 10, "embarked": "XYZ"}, "embarked"),
    ],
)
def test_input_validation_errors(invalid_data, expected_err):
    """Перевіряє, що некоректні дані викликають ValueError з інформативним повідомленням."""
    with pytest.raises(ValueError) as excinfo:
        validate_input(invalid_data)
    assert expected_err in str(excinfo.value)


def test_missing_fields_validation():
    """Перевіряє реакцію валідатора на відсутність обов'язкових полів."""
    incomplete_data = {"pclass": 1, "sex": "female"}
    with pytest.raises(ValueError) as excinfo:
        validate_input(incomplete_data)
    assert "Відсутнє або порожнє обов'язкове поле" in str(excinfo.value)
