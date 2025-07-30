import streamlit as st
from streamlit_option_menu import option_menu
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import pandas as pd
import json
import os
import datetime
from collections import defaultdict
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

# Truy vấn dữ liệu từ Apache Cassandra
query = "SELECT * FROM air_quality LIMIT 100"
rows = session.execute(query)
data = pd.DataFrame(rows)

data["date"] = data["date"].apply(lambda x: str(x) if isinstance(x, cassandra.util.Date) else x)
data["date"] = pd.to_datetime(data["date"])
data["date"] = data["date"].dt.strftime("%Y-%m-%d")
data["time"] = data["time"].astype(str).str.split(".").str[0]  # Định dạng thời gian

def map_function(data):
    mapped_data = []
    for _, row in data.iterrows():
        key = (row["date"], row["time"])
        value = {
            "co_gt": row["co_gt"],
            "no2_gt": row["no2_gt"],
            "nox_gt": row["nox_gt"],
            "ah": row["ah"],
            "c6h6_gt": row["c6h6_gt"],
            "nmhc_gt": row["nmhc_gt"],
            "pt08_s1_co": row["pt08_s1_co"],
            "pt08_s2_nmhc": row["pt08_s2_nmhc"],
            "pt08_s3_nox": row["pt08_s3_nox"],
            "pt08_s4_no2": row["pt08_s4_no2"],
            "pt08_s5_o3": row["pt08_s5_o3"],
            "rh": row["rh"],
            "t": row["t"]
        }
        mapped_data.append((key, value))
    return mapped_data

def reduce_function(mapped_data):
    reduced_data = defaultdict(lambda: defaultdict(list))
    
    for (key, value) in mapped_data:
        date, time = key
        for pollutant, val in value.items():
            reduced_data[(date, time)][pollutant].append(val)
    
    final_data = {}
    for key, values in reduced_data.items():
        avg_values = {pollutant: sum(vals) / len(vals) for pollutant, vals in values.items()}
        final_data[key] = avg_values
        # final_data[key] = values
    
    return final_data

st.title("Hệ thống giám sát ô nhiễm không khí")
st.write("Dữ liệu được lấy từ bảng air_quality trong Cassandra")

selected = option_menu(
    menu_title=None,
    options=["Home", "Data Preview"],
    default_index=0,
    orientation="horizontal",
)

if selected == "Home":
    st.write("Chào mừng đến với hệ thống giám sát chất lượng không khí!")

# if selected == "Data Preview":
#     st.header("Data Preview")
#     st.dataframe(data)

if selected == "Data Preview":
    st.header("Data Preview")
    mapped = map_function(data)
    reduced = reduce_function(mapped)
    reduced_df = pd.DataFrame.from_dict(reduced, orient='index')
    reduced_df.reset_index(inplace=True)
    reduced_df.rename(columns={"level_0": "Date", "level_1": "Time"}, inplace=True)
    reduced_df["Time"] = reduced_df["Time"].astype(str).str.split(".").str[0] 
    st.dataframe(reduced_df.style.format({"co_gt": "{:.1f}", "no2_gt": "{:.1f}", "nox_gt": "{:.1f}", "ah": "{:.1f}", "c6h6_gt": "{:.1f}", "nmhc_gt": "{:.1f}", "pt08_s1_co": "{:.1f}", "pt08_s2_nmhc": "{:.1f}", "pt08_s3_nox": "{:.1f}", "pt08_s4_no2": "{:.1f}", "pt08_s5_o3": "{:.1f}", "rh": "{:.1f}", "t": "{:.1f}"}))