# Air Pollution Forecasting System

A machine learning system for forecasting Air Quality Index (AQI) based on historical pollutant data, with distributed storage on Apache Cassandra (Astra DB) and an interactive Streamlit dashboard.

---

## Overview

This project collects and processes air quality data (CO, NO₂, NOx, C6H6, and sensor readings), stores it in Apache Cassandra, trains a Random Forest model to forecast AQI, and visualizes results through a multi-page Streamlit web application.

---

## Architecture

```
Raw CSV Data (Air_quality_final.csv / Air_quality_official.csv)
        ↓
Data Cleaning & Preprocessing (cleaning_data.ipynb)
        ↓
Load to Apache Cassandra / Astra DB (load_data_to_cassandra.ipynb)
        ↓
MapReduce Aggregation (Home.py)
        ↓
Random Forest Model Training (RandomForest.py)
        ↓
Streamlit Dashboard (Home.py + pages/)
```

---

## Features

- **Data Storage:** Air quality data stored in Apache Cassandra (Astra DB cloud) for scalable, distributed access
- **MapReduce:** Custom map/reduce functions to aggregate pollutant readings by date and time
- **Forecasting:** Random Forest model trained on historical CO and NO₂ concentrations to predict AQI
- **Dashboard:** Multi-page Streamlit app with data preview, pollution trend visualization, and AQI forecast

---

## Pollutants Tracked

| Feature | Description |
|---|---|
| CO_GT | Carbon Monoxide concentration |
| NO2_GT | Nitrogen Dioxide concentration |
| NOx_GT | Nitrogen Oxides concentration |
| C6H6_GT | Benzene concentration |
| NMHC_GT | Non-Methane Hydrocarbons |
| PT08 sensors | Metal oxide sensor readings |
| T / RH / AH | Temperature, Relative & Absolute Humidity |

---

## Project Structure

```
├── Home.py                        # Main Streamlit app (data preview + MapReduce)
├── RandomForest.py                # Model training & prediction
├── cleaning_data.ipynb            # Data cleaning & preprocessing
├── load_data_to_cassandra.ipynb   # Load data into Cassandra
├── Air_quality_final.csv          # Cleaned dataset
├── Air_quality_official.csv       # Raw dataset
└── pages/                         # Additional Streamlit pages
```

---

## Setup

### 1. Install dependencies

```bash
pip install streamlit cassandra-driver pandas scikit-learn streamlit-option-menu
```

### 2. Configure Astra DB credentials

In your Streamlit secrets file (`.streamlit/secrets.toml`):

```toml
ASTRA_DB_TOKEN = '{"clientId": "your_client_id", "secret": "your_secret"}'
```

Place `secure-connect-doanbigdata.zip` in the project root.

### 3. Load data to Cassandra

Run `load_data_to_cassandra.ipynb` to populate the `air_quality` table.

### 4. Run the app

```bash
streamlit run Home.py
```

---

## Tech Stack

- **Language:** Python
- **Storage:** Apache Cassandra (Astra DB cloud)
- **Modeling:** Scikit-learn (Random Forest)
- **Processing:** Pandas, custom MapReduce
- **Dashboard:** Streamlit
