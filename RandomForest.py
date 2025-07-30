import pandas as pd
import json
import cassandra.util
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV 

# Kết nối Apache Cassandra
cloud_config = {
    'secure_connect_bundle': '/home/tee/doanbigdata/secure-connect-doanbigdata.zip'
}
with open("/home/tee/Downloads/doanbigdata-token.json") as f:
    secrets = json.load(f)

CLIENT_ID = secrets["clientId"]
CLIENT_SECRET = secrets["secret"]

auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()
session.set_keyspace('final')

# Lấy dữ liệu từ Apache Cassandra
rows = session.execute("SELECT * FROM air_quality")
data = pd.DataFrame(rows, columns=["date", "time", "ah", "c6h6_gt", "co_gt", "nmhc_gt", "no2_gt", 
                                   "nox_gt", "pt08_s1_co", "pt08_s2_nmhc", "pt08_s3_nox", "pt08_s4_no2", "pt08_s5_o3", "rh", "t"])

# Chuyển đổi cột date về dạng datetime
data["date"] = data["date"].apply(lambda x: x.date() if isinstance(x, cassandra.util.Date) else x)
data["date"] = pd.to_datetime(data["date"])
# print(data[["date", "co_gt", "no2_gt"]].head())
# print(f"Số bản ghi từ Cassandra: {data.shape[0]}")

# Hàm MAP: Trích xuất dữ liệu cần thiết
def map_function_(data):
    mapped_data = []
    for _, row in data.iterrows():
        key = (row["date"], row["time"])
        value = {k: v for k, v in row.items() if k not in ["date", "time"]}  # Loại bỏ date, time
        mapped_data.append((key, value))
    return mapped_data

# Hàm REDUCE: Tính AQI cho từng giá trị 
def reduce_function(mapped_data):
    # Hàm tính AQI cho từng chất ô nhiễm
    def calculate_aqi(concentration, breakpoints):
        for low, high, AQI_low, AQI_high in breakpoints:
            if low <= concentration <= high:
                return ((AQI_high - AQI_low) / (high - low)) * (concentration - low) + AQI_low
        return None  # Giá trị ngoài phạm vi

    # Bảng ngưỡng AQI cho NO2
    aqi_no2_breakpoints = [
        (0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
        (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 2049, 301, 400)
    ]

    # Bảng ngưỡng AQI cho CO
    aqi_co_breakpoints = [
        (0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300), (30.5, 40.4, 301, 400)
    ]

    # Tính AQI cho từng bản ghi
    aqi_values = []
    for (date, time), value in mapped_data:  # Lấy key (date, time) và value
        # Kiểm tra t
        if "no2_gt" not in value or "co_gt" not in value:
            continue
        
        aqi_no2 = calculate_aqi(value["no2_gt"], aqi_no2_breakpoints)
        aqi_co = calculate_aqi(value["co_gt"], aqi_co_breakpoints)

        # Chọn AQI lớn nhất giữa các chất ô nhiễm
        valid_aqi = [aqi for aqi in [aqi_no2, aqi_co] if aqi is not None]
        aqi_max = max(valid_aqi) if valid_aqi else 0  # Nếu rỗng, gán AQI = 0

        aqi_values.append({
            "date": date, "time": time, "AQI_CO": aqi_co, "AQI_NO2": aqi_no2, "AQI": aqi_max
        })

    # Chuyển danh sách thành DataFrame
    aqi_df = pd.DataFrame(aqi_values)

    # Nếu rỗng, trả về bảng trống
    if aqi_df.empty:
        return pd.DataFrame(columns=["date", "time", "AQI_CO", "AQI_NO2", "AQI"])

    # Phân loại ô nhiễm (pollution = 1 nếu AQI >= 100)
    aqi_df["pollution"] = aqi_df["AQI"].apply(lambda x: 1 if x >= 100 else 0)
    return aqi_df

# Áp dụng map và reduce
mapped_data = map_function_(data)
daily_aqi_df = reduce_function(mapped_data)

# Kết hợp dữ liệu ô nhiễm + AQI
merged_data = pd.merge(data, daily_aqi_df, on=["date", "time"], how="left")
print(f"Số bản ghi sau merge: {merged_data.shape[0]}")
print(merged_data["date"].value_counts().head(20))  # Xem có ngày nào bị lặp nhiều lần không
print(merged_data.describe())
print(f"Số bản ghi sau MAP: {len(mapped_data)}")
print(f"Số bản ghi sau REDUCE: {daily_aqi_df.shape[0]}")

# Xử lý dữ liệu cho mô hình
X = merged_data[["co_gt", "no2_gt", "nox_gt", "ah", "c6h6_gt", "nmhc_gt", "pt08_s1_co", "pt08_s2_nmhc", "pt08_s3_nox", "pt08_s4_no2", "pt08_s5_o3", "rh", "t"]]
y = merged_data["pollution"]  # 0 = Không ô nhiễm, 1 = Ô nhiễm

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf_model = RandomForestClassifier(criterion="gini", max_features=4, n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Độ chính xác trên tập test: {accuracy:.2%}")
print("Báo cáo phân loại:")
print(report)

test_results = X_test.copy()
test_results["Actual"] = y_test
test_results["Predicted"] = y_pred
test_results.to_csv("test_results.csv", index=False)

# Lưu 20 
test_samples = test_results.head(20).to_dict(orient="records")
with open("test_samples.json", "w") as f:
    json.dump(test_samples, f)

# Tính các chỉ số đánh giá
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, average="weighted"),
    "recall": recall_score(y_test, y_pred, average="weighted"),
    "f1_score": f1_score(y_test, y_pred, average="weighted")
}

# Lưu vào tệp JSON
with open("rf_metrics.json", "w") as f:
    json.dump(metrics, f)

# Test thử dự đoán với dữ liệu mới
def predict_pollution(co, no2, nox, ah, c6h6, nmhc, pt08_s1_co, pt08_s2_nmhc, pt08_s3_nox, pt08_s4_no2, pt08_s5_o3, rh, t):
    input_data = pd.DataFrame([[co, no2, nox, ah, c6h6, nmhc, pt08_s1_co, pt08_s2_nmhc, pt08_s3_nox, pt08_s4_no2, pt08_s5_o3, rh, t]], columns=["co_gt", "no2_gt", "nox_gt", "ah", "c6h6_gt", "nmhc_gt", "pt08_s1_co", "pt08_s2_nmhc", "pt08_s3_nox", "pt08_s4_no2", "pt08_s5_o3", "rh", "t"])
    #input_data = pd.DataFrame([[co, no2]], columns=["co_gt", "no2_gt"])
    prediction = rf_model.predict(input_data)[0]
    if prediction == 0:
        return 0  # Không ô nhiễm
    else:
        return 1  # Ô nhiễm

# Dự đoán thử
predicted_pollution = predict_pollution(3.9,60.8,53.1,0.99,4.6,198.9,1015.8,525.7,967.4,702.5,512.1,26.3,37.0)
print(f"Dự đoán: {predicted_pollution}")

# Lưu mô hình tốt nhất
joblib.dump(rf_model, "/home/tee/doanbigdata/rf_model.pkl")