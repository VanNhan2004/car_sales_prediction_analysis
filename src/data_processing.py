import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Load data
df = pd.read_csv("dataset/CarBuyers.csv")

# 2. Drop cột không dùng
df = df.drop(columns=['Male', 'Female', 'Unknown'])

# 3. Xử lý Total (có dấu phẩy)
df["Total"] = df["Total"].astype(str)
df["Total"] = df["Total"].str.replace(",", "", regex=False)
df["Total"] = df["Total"].astype(int)

# 4. Label Encoding - Manufacturer
le_manufacturer = LabelEncoder()
df["Manufacturer_le"] = le_manufacturer.fit_transform(df["Manufacturer"])

# 5. Frequency Encoding - Model
model_freq = df["Model"].value_counts()
df["Model_freq"] = df["Model"].map(model_freq)

# 6. One-hot Encoding - Fuel
df = pd.get_dummies(df, columns=["Fuel"], prefix="Fuel")

# 7. Chuyển Fuel bool → int
fuel_cols = [c for c in df.columns if c.startswith("Fuel_")]
df[fuel_cols] = df[fuel_cols].astype(int)

# 8. Lưu file SAU KHI xử lý xong
df.to_csv("dataset/data_new.csv", index=False)

# 9. Kiểm tra
print(df.dtypes)

