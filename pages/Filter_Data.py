import streamlit as st
from streamlit_option_menu import option_menu
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import pandas as pd
import json
import datetime
import cassandra.util
from collections import defaultdict
import plotly.express as px

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

with st.sidebar:
    selected = option_menu(
        menu_title="Filter Data",
        options=["Date", "Chỉ số ô nhiễm"],
        default_index=0,
        orientation="vertical",
    )

def map_function_(data):
    mapped_data = []
    for _, row in data.iterrows():
        date_value = pd.to_datetime(row["date"], errors="coerce")  # Đảm bảo 'date' là datetime
        time_value = row["time"]
        key = (date_value, time_value)
        value = row.to_dict()
        mapped_data.append((key, value))
    return mapped_data


def reduce_function_find_date(mapped_data, start_date, end_date):
    reduced_data = defaultdict(list)
    
    # Chuyển đổi start_date và end_date sang Timestamp nếu cần
    start_date = pd.to_datetime(start_date, errors="coerce")
    end_date = pd.to_datetime(end_date, errors="coerce")
    
    for key, value in mapped_data:
        date_value = key[0]  # key[0] chính là cột date
        if isinstance(date_value, pd.Timestamp) and start_date <= date_value <= end_date:
            reduced_data[key].append(value)
    
    return reduced_data

# def reduce_function_find_cogt(mapped_data, co_range=None, co_value=None):
#     reduced_data = defaultdict(list)
    
#     for key, value in mapped_data:
#         if co_range:
#             if co_range[0] <= value["co"] <= co_range[1]:
#                 reduced_data[key].append(value)
#         elif co_value is not None:
#             if value["co"] == co_value:
#                 reduced_data[key].append(value)
    
#     return reduced_data

def reduce_function_find_pollutant(mapped_data, pollutant, value_range=None, value_fixed=None):
    reduced_data = defaultdict(list)
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

    if pollutant not in pollutant_mapping:
        print(f"Không tìm thấy chỉ số {pollutant} trong danh sách!")
        return reduced_data
    
    pollutant_key = pollutant_mapping[pollutant]
    EPSILON = 1e-5  # Sai số nhỏ để tránh lỗi dấu phẩy động

    for key, value in mapped_data:
        if pollutant_key not in value:
            continue

        try:
            pollutant_value = float(value[pollutant_key])
            pollutant_value = round(pollutant_value, 2)  # Làm tròn để tránh lỗi dấu phẩy động
        except ValueError:
            print(f"Không thể chuyển đổi {value[pollutant_key]} thành số!")
            continue

        if value_range is not None and isinstance(value_range, (tuple, list)) and len(value_range) == 2:
            if value_range[0] <= pollutant_value <= value_range[1]:
                reduced_data[key].append(value)

        elif value_fixed is not None:
            value_fixed = round(float(value_fixed), 2)
            if abs(pollutant_value - value_fixed) < EPSILON:
                reduced_data[key].append(value)

    return reduced_data

mapped_data = map_function_(data)

if selected == "Date":
    st.header("Tìm kiếm dữ liệu theo ngày")
    st.write("Dữ liệu được lấy từ bảng air_quality trong Cassandra")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Ngày bắt đầu", datetime.date(2004, 3, 10), min_value=datetime.date(2004, 3, 10), max_value=datetime.date(2005, 4, 4))
    with col2:
        end_date = st.date_input("Ngày kết thúc", datetime.date(2004, 3, 10), min_value=datetime.date(2004, 3, 10), max_value=datetime.date(2005, 4, 4))
        
    reduced_data = reduce_function_find_date(mapped_data, start_date, end_date)

    filtered_data = pd.DataFrame([value for values in reduced_data.values() for value in values])

    filtered_data["date"] = filtered_data["date"].dt.strftime("%Y-%m-%d")  # Y-M-D
    filtered_data["time"] = pd.to_datetime(filtered_data["time"], format="%H:%M:%S.%f").dt.strftime("%H:%M:%S")  # H:M:S

    st.write(f"Dữ liệu từ {start_date} đến {end_date}:")
    st.dataframe(filtered_data)
    if st.button("Reset", type="primary"):
        pass

if selected == "Chỉ số ô nhiễm":
    pollutants = ["CO(GT)", "NOx(GT)", "NO2(GT)", "T", "RH", "AH"]
    st.title("🔍 Tìm kiếm dữ liệu ô nhiễm không khí")
    st.write("Dữ liệu được lấy từ bảng air_quality trong Cassandra")
    col1, col2 = st.columns(2)
    with col1:
        selected_pollutant = st.selectbox("Chọn chỉ số ô nhiễm", pollutants)
    with col2:
        selected_filter = st.radio("Chọn kiểu lọc", ["Khoảng giá trị", "Giá trị cố định"])

    # Lấy khoảng giá trị cho chỉ số được chọn
    pollutant_ranges = {
        "CO(GT)": (0.1, 11.9),
        "NOx(GT)": (2.0, 1479.0),
        "NO2(GT)": (2.0, 340.0),
        "NMHC(GT)": (7.0, 1189.0),
        "C6H6(GT)": (0.1, 63.7),
        "T": (0.1, 44.6),
        "RH": (9.2, 88.7),
        "AH": (0.2, 2.2)
    }
    min_val, max_val = pollutant_ranges[selected_pollutant]

    if selected_filter == "Khoảng giá trị":
        st.subheader(f"Lọc dữ liệu theo khoảng giá trị {selected_pollutant.upper()}")
        value_range = st.slider(
            f"Chọn khoảng giá trị {selected_pollutant}",
            min_val, max_val, (min_val, max_val), 0.1
        )
        
        if st.button("Lọc dữ liệu"):
            filtered_data = reduce_function_find_pollutant(mapped_data, selected_pollutant, value_range=value_range)
            df_filtered = pd.DataFrame([val for values in filtered_data.values() for val in values])
            if df_filtered.empty:
                st.warning(f"Không có dữ liệu phù hợp với điều kiện đã chọn!")
            else:
                df_filtered["date"] = df_filtered["date"].dt.strftime("%Y-%m-%d")
                df_filtered["time"] = pd.to_datetime(df_filtered["time"], format="%H:%M:%S.%f").dt.strftime("%H:%M:%S") 
                st.dataframe(df_filtered)

            # Xuất dữ liệu thành file CSV
            csv = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Tải xuống CSV",
                data=csv,
                file_name="filtered_data.csv",
                    mime="text/csv",
                )
        if st.button("Reset", type="primary"):
            pass

    elif selected_filter == "Giá trị cố định":
        st.subheader(f"Lọc dữ liệu theo giá trị cố định {selected_pollutant.upper()}")
        value_fixed = st.number_input(f"Nhập giá trị {selected_pollutant.upper()}", min_value=min_val, max_value=max_val, step=0.1)
        
        try:
            # value_fixed = float(value_fixed)
            value_fixed = round(value_fixed, 1)
            if st.button("Lọc dữ liệu"):
                filtered_data = reduce_function_find_pollutant(mapped_data, selected_pollutant, value_fixed=value_fixed)
                df_filtered = pd.DataFrame([val for values in filtered_data.values() for val in values])
                if df_filtered.empty:
                    st.warning(f"Không có dữ liệu phù hợp với giá trị {value_fixed} của {selected_pollutant.upper()}!")
                else:
                    df_filtered["date"] = df_filtered["date"].dt.strftime("%Y-%m-%d") 
                    df_filtered["time"] = pd.to_datetime(df_filtered["time"], format="%H:%M:%S.%f").dt.strftime("%H:%M:%S")
                    st.dataframe(df_filtered)

                    csv = df_filtered.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Tải xuống CSV",
                        data=csv,
                        file_name="filtered_data.csv",
                            mime="text/csv",
                        )
            if st.button("Reset", type="primary"):
                pass
        except ValueError:
            st.error("Vui lòng nhập một số hợp lệ!")