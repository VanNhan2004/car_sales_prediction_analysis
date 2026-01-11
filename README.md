# 🚗 Car Sales Prediction & Business Decision Support System

Hệ thống dự báo số lượng khách hàng mua xe và hỗ trợ ra quyết định kinh
doanh dựa trên Machine Learning, SHAP và LLM (Ollama -- LLaMA 3.2).

## 📌 Mục tiêu dự án

-   Dự đoán số lượng khách hàng mua xe (Total buyers)
-   Giải thích yếu tố làm tăng / giảm nhu cầu mua
-   Đề xuất quyết định kinh doanh thực tế cho chủ đại lý
-   Xuất báo cáo phục vụ quản lý và khóa luận

## 🧠 Công nghệ sử dụng

-   Python 3.10
-   Streamlit
-   Random Forest
-   SHAP
-   Ollama (LLaMA 3.2)
-   python-docx

## 📁 Cấu trúc thư mục

PREDICT_CAR/ - src/app.py - src/llm.py - src/SHAP.py - src/report.py -
models/best_random_forest.pkl - dataset/data_new.csv

## 🚀 Cách chạy

pip install -r requirements.txt\
ollama pull llama3.2\
streamlit run src/app.py

## 👤 Tác giả

Nguyễn Văn Nhân -- Đại học Nguyễn Tất Thành
