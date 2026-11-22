"""
Веб-додаток: «Чи вижив би я на Титаніку?»
"""

from pathlib import Path
import streamlit as st
from src.predictor import (
    DEFAULT_MODEL_PATH,
    predict_survival,
)

# Базові налаштування сторінки
st.set_page_config(
    page_title="Чи вижив би я на Титаніку?",
    page_icon="🚢",
    layout="wide",
)

# Перевірка наявності навченої моделі
if not DEFAULT_MODEL_PATH.exists():
    st.error("Файл моделі models/model.joblib не знайдено. Спочатку запустіть: python src/train.py")
    st.stop()

# Пресети для швидкої перевірки
PRESETS = {
    "Власний вибір": None,
    "Роуз (1 клас, 17 років)": {
        "pclass": 1,
        "sex": "female",
        "age": 17,
        "fare": 120.0,
        "sibsp": 0,
        "parch": 1,
        "embarked": "S",
    },
    "Джек (3 клас, 20 років)": {
        "pclass": 3,
        "sex": "male",
        "age": 20,
        "fare": 7.25,
        "sibsp": 0,
        "parch": 0,
        "embarked": "S",
    },
    "Капітан Сміт (1 клас, 62 роки)": {
        "pclass": 1,
        "sex": "male",
        "age": 62,
        "fare": 80.0,
        "sibsp": 0,
        "parch": 0,
        "embarked": "S",
    },
    "Дитина з 3 класу (4 роки)": {
        "pclass": 3,
        "sex": "female",
        "age": 4,
        "fare": 16.5,
        "sibsp": 1,
        "parch": 1,
        "embarked": "S",
    },
}

# Початкові значення у session_state
if "pclass" not in st.session_state:
    st.session_state.pclass = 2
    st.session_state.sex = "female"
    st.session_state.age = 28
    st.session_state.fare = 30.0
    st.session_state.sibsp = 0
    st.session_state.parch = 0
    st.session_state.embarked = "S"


def apply_preset():
    name = st.session_state.preset_choice
    data = PRESETS.get(name)
    if data:
        st.session_state.pclass = data["pclass"]
        st.session_state.sex = data["sex"]
        st.session_state.age = data["age"]
        st.session_state.fare = float(data["fare"])
        st.session_state.sibsp = data["sibsp"]
        st.session_state.parch = data["parch"]
        st.session_state.embarked = data["embarked"]


# Бічна панель з параметрами
st.sidebar.header("Параметри пасажира")

st.sidebar.selectbox(
    "Швидкий пресет:",
    options=list(PRESETS.keys()),
    key="preset_choice",
    on_change=apply_preset,
)

pclass = st.sidebar.radio(
    "Клас каюти:",
    options=[1, 2, 3],
    format_func=lambda x: f"{x}-й клас ({'1-й клас' if x == 1 else '2-й клас' if x == 2 else '3-й клас'})",
    key="pclass",
)

sex = st.sidebar.radio(
    "Стать:",
    options=["female", "male"],
    format_func=lambda x: "Жіноча" if x == "female" else "Чоловіча",
    key="sex",
)

age = st.sidebar.slider(
    "Вік (років):",
    min_value=0,
    max_value=100,
    step=1,
    key="age",
)

fare = st.sidebar.number_input(
    "Ціна квитка (£):",
    min_value=0.0,
    max_value=600.0,
    step=5.0,
    key="fare",
)

col_s, col_p = st.sidebar.columns(2)
with col_s:
    sibsp = st.number_input("Брати / сестри / шлюб:", min_value=0, max_value=10, step=1, key="sibsp")
with col_p:
    parch = st.number_input("Батьки / діти:", min_value=0, max_value=10, step=1, key="parch")

embarked = st.sidebar.selectbox(
    "Порт посадки:",
    options=["S", "C", "Q"],
    format_func=lambda x: {"S": "Саутгемптон (S)", "C": "Шербур (C)", "Q": "Квінстаун (Q)"}[x],
    key="embarked",
)

# Розрахунок прогнозу
passenger = {
    "pclass": pclass,
    "sex": sex,
    "age": age,
    "fare": fare,
    "sibsp": sibsp,
    "parch": parch,
    "embarked": embarked,
}

res = predict_survival(passenger)

# Головний екран
st.title("🚢 Чи вижив би я на Титаніку?")
st.write("Змініть параметри пасажира у бічній панелі, і результат перерахується автоматично.")

col_main, col_factors = st.columns([1, 1], gap="large")

with col_main:
    # Вердикт
    if res["prediction"] == 1:
        st.success("### 🎉 Ви б вижили!")
    else:
        st.error("### 🌊 На жаль, шанси вижити малі...")

    # Основні показники
    m1, m2, m3 = st.columns(3)
    m1.metric("Шанс вижити", f"{res['survival_probability']}%")
    m2.metric("Впевненість моделі", f"{res['confidence']}%")
    m3.metric("Розмір родини", f"{sibsp + parch + 1} ос.")

    st.write("**Рівень ймовірності:**")
    st.progress(float(res["survival_probability"]) / 100.0)

    # Попередження про незвичні значення
    if res["warnings"]:
        for w in res["warnings"]:
            st.warning(w)

with col_factors:
    st.subheader("Чому модель прийняла таке рішення?")
    st.info(res["explanation"])

    st.write("**Основні фактори:**")
    for f in res["factors"]:
        badge = "🟢" if f["impact"] == "positive" else "🔴" if f["impact"] == "negative" else "⚪"
        st.write(f"{badge} **{f['feature']}** — {f['description']}")
