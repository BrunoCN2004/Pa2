# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="Predicción de Fraude Bancario",
    page_icon="💳",
    layout="wide"
)

# =========================================================
# CARGA DE MODELOS
# =========================================================

@st.cache_resource
def load_models():
    logistic_model = joblib.load("model/logistic_regression_model.pkl")
    random_forest_model = joblib.load("model/random_forest_model.pkl")
    return logistic_model, random_forest_model

logistic_model, random_forest_model = load_models()

# =========================================================
# TITULO
# =========================================================

st.title("💳 Sistema de Predicción de Fraude Financiero")
st.markdown("""
Esta aplicación utiliza dos modelos de Machine Learning para detectar posibles fraudes financieros:
- Regresión Logística
- Random Forest
""")

# =========================================================
# FORMULARIO
# =========================================================

st.header("📋 Ingrese los datos de la transacción")

col1, col2 = st.columns(2)

with col1:
    monto = st.number_input("Monto de la transacción", min_value=0.0, value=100.0)
    edad = st.number_input("Edad del cliente", min_value=18, max_value=100, value=30)
    saldo = st.number_input("Saldo de la cuenta", min_value=0.0, value=5000.0)
    num_transacciones = st.number_input("Número de transacciones del día", min_value=0, value=3)

with col2:
    tipo_transaccion = st.selectbox(
        "Tipo de transacción",
        ["Transferencia", "Retiro", "Pago", "Compra"]
    )

    dispositivo = st.selectbox(
        "Dispositivo utilizado",
        ["Móvil", "Laptop", "PC", "Tablet"]
    )

    ubicacion = st.selectbox(
        "Ubicación",
        ["Local", "Internacional"]
    )

# =========================================================
# PREPROCESAMIENTO
# =========================================================

def preprocess_input():

    data = {
        "monto": monto,
        "edad": edad,
        "saldo": saldo,
        "num_transacciones": num_transacciones,
        "tipo_transaccion": tipo_transaccion,
        "dispositivo": dispositivo,
        "ubicacion": ubicacion
    }

    df = pd.DataFrame([data])

    # One Hot Encoding
    df = pd.get_dummies(df)

    # =====================================================
    # COLUMNAS EXACTAS DEL ENTRENAMIENTO
    # =====================================================

    expected_columns = logistic_model.feature_names_in_

    # Agregar columnas faltantes
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    # Eliminar columnas extras
    df = df[expected_columns]

    return df

# =========================================================
# PREDICCIÓN
# =========================================================

if st.button("🔍 Predecir Fraude"):

    input_data = preprocess_input()

    # =====================================================
    # REGRESIÓN LOGÍSTICA
    # =====================================================

    logistic_prediction = logistic_model.predict(input_data)[0]
    logistic_probability = logistic_model.predict_proba(input_data)[0][1]

    # =====================================================
    # RANDOM FOREST
    # =====================================================

    rf_prediction = random_forest_model.predict(input_data)[0]
    rf_probability = random_forest_model.predict_proba(input_data)[0][1]

    # =====================================================
    # RESULTADOS
    # =====================================================

    st.header("📊 Resultados")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Regresión Logística")

        if logistic_prediction == 1:
            st.error("⚠️ Fraude Confirmado")
        else:
            st.success("✅ No Fraude")

        st.metric(
            label="Probabilidad de Fraude",
            value=f"{logistic_probability:.2%}"
        )

    with col2:
        st.subheader("🌲 Random Forest")

        if rf_prediction == 1:
            st.error("⚠️ Fraude Confirmado")
        else:
            st.success("✅ No Fraude")

        st.metric(
            label="Probabilidad de Fraude",
            value=f"{rf_probability:.2%}"
        )

    # =====================================================
    # ALERTA GENERAL
    # =====================================================

    if logistic_prediction == 1 or rf_prediction == 1:
        st.warning("""
        🚨 ALERTA:
        Al menos uno de los modelos detectó una posible transacción fraudulenta.
        Se recomienda revisar la operación.
        """)
    else:
        st.success("""
        ✅ Ambos modelos indican que la transacción no presenta señales de fraude.
        """)
