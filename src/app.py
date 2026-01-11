import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap

# =========================
# IMPORT MODULE
# =========================
from SHAP import explain_prediction
from llm import analyze_shap_with_llm
from report import export_prediction_docx

# =========================
# LOAD MODEL & DATA
# =========================
model = joblib.load("models/best_random_forest.pkl")
df = pd.read_csv("dataset/data_new.csv")

# Mapping cho encoding
manufacturer_map = dict(zip(
    df["Manufacturer"].unique(),
    df["Manufacturer_le"].unique()
))

model_freq_map = df["Model"].value_counts().to_dict()

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="Car Sales Prediction",
    layout="centered"
)

st.title("🚗 Hệ thống dự báo số lượng khách hàng mua xe")
st.write("Nhập thông tin xe để dự đoán **Total (số lượng người mua)**")

# =========================
# INPUT FORM
# =========================
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        price = st.number_input("💰 Price", min_value=0.0, value=500000.0)
        power = st.number_input("🔋 Power", min_value=0.0, value=100.0)
        manufacturer = st.selectbox(
            "🏭 Manufacturer",
            options=manufacturer_map.keys()
        )

    with col2:
        transmission = st.number_input("⚙️ Transmission", min_value=0.0, value=1.0)
        engine_cc = st.number_input("🛠 Engine CC", min_value=0.0, value=1500.0)
        model_name = st.selectbox(
            "🚘 Model",
            options=df["Model"].unique()
        )

    fuel = st.selectbox(
        "⛽ Fuel Type",
        options=["automatic", "diesel", "petrol"]
    )

    submit = st.form_submit_button("🔮 Predict")

# =========================
# PREDICTION + SHAP + LLM
# =========================
if submit:
    # ---------- ENCODING ----------
    manufacturer_le = manufacturer_map[manufacturer]
    model_freq = model_freq_map.get(model_name, 1)

    fuel_automatic = 1 if fuel == "automatic" else 0
    fuel_diesel = 1 if fuel == "diesel" else 0
    fuel_petrol = 1 if fuel == "petrol" else 0

    input_data = pd.DataFrame([{
        "Price": price,
        "Transmission": transmission,
        "Power": power,
        "Engine CC": engine_cc,
        "Manufacturer_le": manufacturer_le,
        "Model_freq": model_freq,
        "Fuel_automatic": fuel_automatic,
        "Fuel_diesel": fuel_diesel,
        "Fuel_petrol": fuel_petrol
    }])

    # ---------- PREDICT ----------
    prediction = model.predict(input_data)[0]
    st.success(f"📈 Dự đoán Total: **{int(prediction):,}**")

    # =========================
    # SHAP EXPLANATION
    # =========================
    st.subheader("🔍 Giải thích dự đoán (SHAP)")

    shap_df, shap_values = explain_prediction(model, input_data)

    st.markdown("### 📊 Bảng đóng góp các yếu tố")
    st.dataframe(shap_df)

    fig, ax = plt.subplots()
    shap.plots.bar(shap_values[0], show=False)
    st.pyplot(fig)

    # =========================
    # LLM ANALYSIS
    # =========================
    st.subheader("🤖 AI phân tích và đề xuất giải pháp")

    with st.spinner("AI đang phân tích kết quả ..."):
        llm_result = analyze_shap_with_llm(
            prediction=prediction,
            shap_df=shap_df,
            top_k=5
        )

    st.write(llm_result)

    # =========================
    # EXPORT WORD REPORT (KHÔNG RESET)
    # =========================
    st.subheader("📄 Xuất báo cáo")

    docx_buffer = export_prediction_docx(
        input_data=input_data.iloc[0].to_dict(),
        prediction=prediction,
        shap_df=shap_df,
        llm_analysis=llm_result
    )

    st.download_button(
        label="⬇️ Tải báo cáo Word (.docx)",
        data=docx_buffer,
        file_name="car_sales_report.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
