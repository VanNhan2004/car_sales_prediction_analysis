# llm.py
# ---------------------------------------
# Phân tích SHAP bằng LLM (LLaMA 3.2 - Ollama)
# ---------------------------------------

import requests
import json
import pandas as pd

# Ollama local API
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"


def analyze_shap_with_llm(
    prediction: float,
    shap_df: pd.DataFrame,
    top_k: int = 5
) -> str:
    # =========================
    # 1. Lấy top yếu tố ảnh hưởng nhất
    # =========================
    shap_df = shap_df.copy()
    shap_df = shap_df.reindex(
        shap_df["shap_value"].abs().sort_values(ascending=False).index
    ).head(top_k)

    # =========================
    # 2. Chuyển SHAP thành text
    # =========================
    shap_text = ""
    for _, row in shap_df.iterrows():
        effect = "increase" if row["shap_value"] > 0 else "decrease"
        shap_text += (
            f"- {row['feature']} = {row['value']} "
            f"→ {effect} Total by {abs(row['shap_value']):,.0f}\n"
        )

    # =========================
    # 3. Prompt cho LLM
    # =========================
    prompt = f"""
Bạn là AI trả lời và giải đáp các phân tích kinh doanh trong lĩnh vực ô tô dựa trên dữ liệu, cần đưa ra nhận xét để doanh nghiệp nên nhaaph xe về yếu tố gì và không về yếu tố gì.

MÔ HÌNH DỰ ĐOÁN:
- Số lượng người mua (Total buyers) dự đoán: {prediction:,.0f}

KẾT QUẢ PHÂN TÍCH:
{shap_text}

YÊU CẦU:
Hãy viết MỘT ĐOẠN PHÂN TÍCH DẠNG BÁO CÁO, bằng tiếng Việt, theo cấu trúc sau:

- Từ kết quả trên nhận thấy những yếu tố nào đang LÀM TĂNG số lượng người mua.
- Những yếu tố nào đang LÀM GIẢM số lượng người mua.
- Từ đó đề xuất CẦN LÀM GÌ (tăng/giảm/ưu tiên yếu tố nào) để nâng cao số lượng người mua và doanh thu.

QUY ĐỊNH:
- Viết gọn gàng rành mạch
- Trả lời chuẩn chỉnh và thực tế
- Không dùng thuật ngữ kỹ thuật
- Văn phong trang trọng, phù hợp báo cáo hoặc khóa luận
"""


    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    # =========================
    # 4. Gọi Ollama API
    # =========================
    response = requests.post(
        OLLAMA_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=120
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"LLM request failed: {response.status_code} - {response.text}"
        )

    # =========================
    # 5. Trả kết quả
    # =========================
    return response.json().get("response", "").strip()
