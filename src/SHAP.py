import shap
import pandas as pd

def explain_prediction(model, input_data):
 
    # TreeExplainer cho RandomForest
    explainer = shap.TreeExplainer(model)

    # SHAP values (CÁCH MỚI - KHÔNG LỖI)
    shap_values = explainer(input_data)

    # Tạo bảng SHAP dạng số
    shap_df = pd.DataFrame({
        "feature": input_data.columns,
        "value": input_data.iloc[0].values,
        "shap_value": shap_values[0].values
    }).sort_values(
        by="shap_value",
        key=abs,
        ascending=False
    )

    return shap_df, shap_values
