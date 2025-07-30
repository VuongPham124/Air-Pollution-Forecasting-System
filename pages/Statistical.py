import streamlit as st
import plotly.express as px
import pandas as pd
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import json
import cassandra.util

cloud_config= {
  'secure_connect_bundle': 'secure-connect-doanbigdata.zip'
}

astra_token = st.secrets["ASTRA_DB_TOKEN"]
astra_token_dict = json.loads(astra_token)

CLIENT_ID = astra_token_dict["clientId"]
CLIENT_SECRET = astra_token_dict["secret"]

auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()
session.set_keyspace('final')

rows = session.execute("SELECT * FROM air_quality")
data = pd.DataFrame(rows, columns=["date", "time", "ah", "c6h6_gt", "co_gt", "nmhc_gt", "no2_gt", 
                                   "nox_gt", "pt08_s1_co", "pt08_s2_nmhc", "pt08_s3_nox", "pt08_s4_no2", "pt08_s5_o3", "rh", "t"])

data["date"] = data["date"].apply(lambda x: x.date() if isinstance(x, cassandra.util.Date) else x)
data["date"] = pd.to_datetime(data["date"])

def map_function_(data):
    mapped_data = []
    for _, row in data.iterrows():
        try:
            date_value = pd.to_datetime(row["date"], errors="coerce")
            time_value = row["time"]
            key = (date_value, time_value)
            value = row.to_dict()
            mapped_data.append((key, value))
        except Exception as e:
            print(f"Lỗi khi xử lý hàng dữ liệu: {e}")
    return mapped_data
def reduce_function_find_date(mapped_data, month, year, actual_column):
    reduced_data = []
    for key, value in mapped_data:
        date_value = key[0]
        if date_value.month == month and date_value.year == year:
            reduced_data.append({"date": date_value, actual_column: value[actual_column]})
    if not reduced_data:
        return None
    reduced_df = pd.DataFrame(reduced_data)
    daily_avg_df = reduced_df.groupby("date")[actual_column].mean().reset_index()
    return daily_avg_df
st.title("Biểu đồ chỉ số ô nhiễm theo tháng")

pollutant_mapping = {
    "CO(GT)": "co_gt",
    "NOx(GT)": "nox_gt",
    "NO2(GT)": "no2_gt",
    "NMHC(GT)": "nmhc_gt",
    "C6H6(GT)": "c6h6_gt",
    "T": "t",
    "RH": "rh",
    "AH": "ah"
}

col1, col2, col3 = st.columns(3)
with col1:
    month = st.selectbox("Chọn tháng", list(range(1, 13)), index=2)
with col2:
    year = st.selectbox("Chọn năm", sorted(data["date"].dt.year.unique()), index=0)
with col3:
    selected_pollutant = st.selectbox("Chọn chỉ số ô nhiễm", list(pollutant_mapping.keys()))

if st.button("Hiển thị biểu đồ"):
    actual_column = pollutant_mapping[selected_pollutant]

    mapped_data = map_function_(data)

    daily_avg = reduce_function_find_date(mapped_data, month, year, actual_column)

    if daily_avg is None or daily_avg.empty:
        st.warning("Không có dữ liệu cho tháng này.")
    else:
        fig = px.line(
            daily_avg, 
            x="date", 
            y=actual_column, 
            title=f"{selected_pollutant} trung bình theo ngày - Tháng {month}/{year}",
            labels={"date": "Ngày", actual_column: selected_pollutant},
            line_shape="spline"
        )
        st.plotly_chart(fig)
