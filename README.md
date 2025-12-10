# Advanced Traffic Dashboard 🚦

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.26-green.svg)](https://streamlit.io/)

An **interactive traffic analytics dashboard** built with **Streamlit**, integrating **data visualization**, **geospatial mapping**, and **machine learning** in one platform.

[![Open Dashboard](https://img.shields.io/badge/Live_Dashboard-Streamlit-brightgreen?style=for-the-badge)](https://mvpfinal-vnj5ddjlifva5g27nuwj6d.streamlit.app/)


---

## Table of Contents

- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
- [File Structure](#file-structure)  
- [Dataset Requirements](#dataset-requirements)  
- [Dependencies](#dependencies)  
- [Screenshots](#screenshots)  
- [Conclusion](#conclusion)

---

## Features

- **Dynamic Dashboard:** KPIs like average traffic, peak hours, maximum traffic.  
- **Interactive Charts:** Line charts & heatmaps for traffic trends.  
- **Geospatial Maps:** City-level scatter maps and animated traffic maps.  
- **ML Predictions:**  
  - **RandomForest Regression** – Predict traffic by hour/day.  
  - **PyTorch LSTM** – Forecast next-hour traffic.  
- **Custom Dataset Upload:** Override default dataset with your own CSV.  
- **HTML Dashboard View:** Embed pre-built HTML dashboards.  
- **Custom Styling:** Smooth UI with CSS enhancements.  

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/advanced-traffic-dashboard.git
cd advanced-traffic-dashboard
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage


Run the Streamlit app:

```bash
streamlit run app.py
```

1. Use the **sidebar filters** for Date Range, State, and City.  
2. Navigate between tabs:  
   - **Dashboard** – Traffic KPIs & comparisons.  
   - **Maps** – City & animated traffic maps.  
   - **ML Predictions** – RandomForest & LSTM forecasts.  
   - **Upload Data** – Load your CSV dataset.  
   - **HTML View** – Embedded HTML dashboards.  

---

## File Structure

```
advanced-traffic-dashboard/
│
├── app.py               # Main Streamlit app
├── styles.css           # Custom CSS for UI
├── simulated_traffic_data.csv  # Sample dataset
├── index.html           # Optional HTML dashboard view
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## Dataset Requirements

CSV must include:

- `State` – State name  
- `City` – City name  
- `Timestamp` – Datetime (YYYY-MM-DD HH:MM:SS)  
- `HourOfDay` – Hour (0–23)  
- `DayOfWeek` – Day (0=Monday, 6=Sunday)  
- `VehicleCount` – Number of vehicles  

---

## Dependencies

- `streamlit`  
- `pandas`  
- `numpy`  
- `plotly`  
- `scikit-learn`  
- `torch`  

Install all:

```bash
pip install -r requirements.txt
```

---

## Screenshots

<img width="1923" height="763" alt="ml1" src="https://github.com/user-attachments/assets/e665e980-d91d-4f1f-ac6d-22879059d039" />


<img width="1900" height="726" alt="ml2" src="https://github.com/user-attachments/assets/97eb5265-4ea1-4040-a64a-5177fe515136" />


<img width="1922" height="750" alt="ml3" src="https://github.com/user-attachments/assets/5fbc9ef8-3ba3-497b-af69-95cc4b732994" />

## Conclusion

Smart Traffic Volume Prediction using Machine Learning provides a comprehensive solution for analyzing, visualizing, and forecasting urban traffic patterns. This interactive dashboard combines data-driven insights, geospatial mapping, and predictive models to support smarter city planning and traffic management. Future enhancements could include real-time data integration, multi-city support, and additional machine learning models for improved accuracy.


