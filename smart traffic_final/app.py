import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
from streamlit.components.v1 import html

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Advanced Traffic Dashboard", layout="wide")


# ---------------------------------------------------------
# LOAD CUSTOM CSS (for Streamlit widgets & layout)
# ---------------------------------------------------------
def load_css(file_name: str):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"⚠️ CSS file '{file_name}' not found. Using default Streamlit theme.")


load_css("styles.css")


# ---------------------------------------------------------
# LOAD DATA (supports drag & drop)
# ---------------------------------------------------------
@st.cache_data
def load_default():
    df = pd.read_csv("simulated_traffic_data.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


uploaded_file = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
else:
    df = load_default()


# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.title("Filters")

# Date range filter
min_date = df["Timestamp"].min().date()
max_date = df["Timestamp"].max().date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date])

if isinstance(date_range, list) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

df = df[
    (df["Timestamp"].dt.date >= start_date)
    & (df["Timestamp"].dt.date <= end_date)
]

# State filter
states = sorted(df["State"].unique())
selected_state = st.sidebar.selectbox("State", states)

# City filter
cities = sorted(df[df["State"] == selected_state]["City"].unique())
selected_city = st.sidebar.selectbox("City", ["All Cities"] + cities)

df_filtered = df[df["State"] == selected_state]
if selected_city != "All Cities":
    df_filtered = df_filtered[df_filtered["City"] == selected_city]


# ---------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------
st.title("Advanced Traffic Dashboard")
st.caption("Real-time analytics, maps & ML predictions on simulated traffic data.")

# Tabs — last tab is the raw HTML view
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Dashboard",
        "🗺 Maps",
        "🤖 ML Predictions",
        "📁 Upload Data",
        "🌐 HTML View",
    ]
)


# ---------------------------------------------------------
# TAB 1 — DASHBOARD
# ---------------------------------------------------------
with tab1:
    st.subheader(f"Traffic Overview – {selected_state}")

    # KPI metrics
    avg_traffic = df_filtered["VehicleCount"].mean()
    max_traffic = df_filtered["VehicleCount"].max()
    try:
        peak_hour = (
            df_filtered.groupby("HourOfDay")["VehicleCount"]
            .mean()
            .idxmax()
        )
    except ValueError:
        peak_hour = "-"

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Traffic", f"{avg_traffic:.1f}")
    col2.metric("Peak Hour", f"{peak_hour}:00" if peak_hour != "-" else "-")
    col3.metric("Maximum Traffic", int(max_traffic))

    # Line chart
    st.markdown("### 📈 Traffic Over Time")
    if not df_filtered.empty:
        fig1 = px.line(
            df_filtered,
            x="Timestamp",
            y="VehicleCount",
            color="City",
            title=f"Traffic Over Time – {selected_state}",
        )
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("No data available for the selected filters.")

    # Heatmap
    st.markdown("### 🔥 Peak Hour Heatmap")
    if not df_filtered.empty:
        heatmap_df = (
            df_filtered.groupby(["DayOfWeek", "HourOfDay"])["VehicleCount"]
            .mean()
            .reset_index()
        )
        fig_heat = px.density_heatmap(
            heatmap_df,
            x="HourOfDay",
            y="DayOfWeek",
            z="VehicleCount",
            color_continuous_scale="Inferno",
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Not enough data to build heatmap.")

    # Compare 2 cities
    st.markdown("### 🏙 Compare Two Cities")
    if len(cities) >= 2:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            compare_city1 = st.selectbox("City 1", cities, key="city1")
        with col_c2:
            compare_city2 = st.selectbox(
                "City 2", cities, index=1, key="city2"
            )

        comp_df = df[
            (df["City"].isin([compare_city1, compare_city2]))
            & (df["State"] == selected_state)
        ]

        if not comp_df.empty:
            fig_compare = px.line(
                comp_df,
                x="Timestamp",
                y="VehicleCount",
                color="City",
                title=f"Comparison: {compare_city1} vs {compare_city2}",
            )
            st.plotly_chart(fig_compare, use_container_width=True)
        else:
            st.info("No data to compare these cities.")
    else:
        st.info("Need at least two cities in this state to compare.")


# ---------------------------------------------------------
# TAB 2 — MAPS
# ---------------------------------------------------------
with tab2:
    st.subheader("Geographical Traffic Maps")

    city_coords = {
        "Hartford": (41.7658, -72.6734),
        "New Haven": (41.3083, -72.9279),
        "Stamford": (41.0534, -73.5387),
        "Boston": (42.3601, -71.0589),
        "Worcester": (42.2626, -71.8023),
        "Springfield": (42.1015, -72.5898),
        "Newark": (40.7357, -74.1724),
        "Jersey City": (40.7178, -74.0431),
        "Paterson": (40.9168, -74.1718),
        "New York City": (40.7128, -74.0060),
        "Buffalo": (42.8864, -78.8784),
        "Rochester": (43.1566, -77.6088),
    }

    df_map = df[df["City"].isin(city_coords.keys())].copy()
    df_map["Lat"] = df_map["City"].map(lambda x: city_coords[x][0])
    df_map["Lon"] = df_map["City"].map(lambda x: city_coords[x][1])

    st.markdown("#### 🗽 NYC Zoom-Level Traffic Map")
    if not df_map.empty:
        fig_nyc = px.scatter_mapbox(
            df_map,
            lat="Lat",
            lon="Lon",
            size="VehicleCount",
            color="VehicleCount",
            hover_name="City",
            zoom=6,
            color_continuous_scale="Turbo",
            height=550,
        )
        fig_nyc.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig_nyc, use_container_width=True)
    else:
        st.info("No mappable data available.")

    st.markdown("#### 🎞 Animated Traffic Map (by HourOfDay)")
    if not df_map.empty:
        fig_anim = px.scatter_mapbox(
            df_map,
            lat="Lat",
            lon="Lon",
            size="VehicleCount",
            color="VehicleCount",
            animation_frame="HourOfDay",
            hover_name="City",
            zoom=4,
            color_continuous_scale="Inferno",
            height=600,
        )
        fig_anim.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig_anim, use_container_width=True)
    else:
        st.info("No data for animated map.")


# ---------------------------------------------------------
# TAB 3 — ML PREDICTIONS
# ---------------------------------------------------------
with tab3:
    st.subheader("ML Predictions")

    # ---------- RandomForest ----------
    st.markdown("### 🌲 RandomForest Regression")

    ml_df = df[["HourOfDay", "DayOfWeek", "VehicleCount"]].copy()
    ml_df["DayOfWeek"] = ml_df["DayOfWeek"].astype("category").cat.codes

    X = ml_df[["HourOfDay", "DayOfWeek"]]
    y = ml_df["VehicleCount"]

    rf = RandomForestRegressor()
    rf.fit(X, y)

    col_rf1, col_rf2 = st.columns(2)
    with col_rf1:
        pred_hour = st.slider("Hour of Day", 0, 23, 18)
    with col_rf2:
        day_options = sorted(df["DayOfWeek"].unique())
        pred_day = st.selectbox("Day of Week", day_options)

    pred_day_num = pd.Categorical(
        [pred_day], categories=day_options
    ).codes[0]

    rf_result = rf.predict([[pred_hour, pred_day_num]])[0]
    st.success(f"RandomForest Prediction: **{int(rf_result)} vehicles**")

    st.markdown("---")

    # ---------- LSTM ----------
    st.markdown("### 📈 PyTorch LSTM Next-Hour Forecast")

    seq_df = df_filtered.sort_values("Timestamp")
    series = seq_df["VehicleCount"].values.astype(float)

    if len(series) > 30:
        window = 24
        X_lstm, y_lstm = [], []

        for i in range(len(series) - window):
            X_lstm.append(series[i : i + window])
            y_lstm.append(series[i + window])

        X_lstm = np.array(X_lstm)
        y_lstm = np.array(y_lstm)

        X_lstm = X_lstm.reshape(X_lstm.shape[0], window, 1)
        X_tensor = torch.tensor(X_lstm, dtype=torch.float32)
        y_tensor = torch.tensor(y_lstm, dtype=torch.float32)

        class LSTMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(1, 32, batch_first=True)
                self.fc = nn.Linear(32, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])
                return out

        model = LSTMModel()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for _ in range(5):  # light training for demo
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = criterion(output.squeeze(), y_tensor)
            loss.backward()
            optimizer.step()

        last_seq = torch.tensor(
            series[-window:].reshape(1, window, 1), dtype=torch.float32
        )
        lstm_pred = model(last_seq).item()

        st.info(
            f"🔮 LSTM Forecast (Next Hour): **{int(lstm_pred)} vehicles** "
            f"based on last {window} time steps."
        )
    else:
        st.warning(
            "Not enough data in the filtered selection to train the LSTM model."
        )


# ---------------------------------------------------------
# TAB 4 — UPLOAD
# ---------------------------------------------------------
with tab4:
    st.subheader("Upload Custom Traffic Data")
    st.write(
        "Upload a CSV file to override the dataset for all tabs. "
        "Required columns: **State, City, Timestamp, HourOfDay, DayOfWeek, VehicleCount**."
    )

    if uploaded_file:
        st.success("Custom dataset loaded successfully and applied to all views!")
    else:
        st.info("Using default `simulated_traffic_data.csv` dataset.")


# ---------------------------------------------------------
# TAB 5 — RAW HTML DASHBOARD VIEW
# ---------------------------------------------------------
with tab5:
    st.subheader("Embedded HTML Dashboard Layout")

    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        html(html_content, height=900, scrolling=True)
    except FileNotFoundError:
        st.error(
            "index.html not found. Please make sure it is in the same folder as app.py."
        )
