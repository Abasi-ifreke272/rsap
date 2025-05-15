import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import joblib
import lzma
import gdown
import os
import requests
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")


# -------------------------
# 📦 Load Data Function
# -------------------------
def load_data(url):
    try:
        df = pd.read_csv(url, low_memory=False)
        return df
    except Exception as e:
        st.error(f"Error loading data from {url}: {e}")
        return None

# -------------------------
# 🧠 Prediction Tab
# -------------------------
def accident_severity_prediction_tab(df_merged):
    st.markdown("### 🧠 Accident Severity Prediction")
    st.markdown("This section displays the model evaluation for accident severity prediction using Random Forest and Neural Network models.")

    try:
        if df_merged is not None:
            ml_df = df_merged[[
                'vehicle_type',
                'age_of_driver',
                'road_surface_conditions',
                'junction_detail',
                'light_conditions',
                'weather_conditions',
                'speed_limit',
                'accident_severity'
            ]].copy()

            ml_df['accident_severity'] = pd.to_numeric(ml_df['accident_severity'], errors='coerce').astype('Int64')
            ml_df['junction_detail'] = ml_df['junction_detail'].replace([99, -1], 'Unknown/Missing')
            ml_df_cleaned = ml_df.dropna().copy()

            ml_df_encoded = pd.get_dummies(ml_df_cleaned, columns=[
                'vehicle_type',
                'road_surface_conditions',
                'junction_detail',
                'light_conditions',
                'weather_conditions'
            ])

            X = ml_df_encoded.drop('accident_severity', axis=1)
            y = ml_df_encoded['accident_severity'].astype(int)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            # updated model in new_folder
            try:
                rf_model = joblib.load('compressed_rf_model_v9.0.joblib.gz')
                nn_model = joblib.load('compressed_nn_model.joblib.gz')
                print(f"Random Forest model type: {type(rf_model)}")
                print(f"NN model type: {type(nn_model)}")
            except Exception as model_error:
                st.error(f"⚠️ Could not load model: {model_error}")
                print("Error loading models")

            
            rf_y_pred = rf_model.predict(X_test)
            nn_y_pred = nn_model.predict(X_test)

            # Evaluation
            for model_name, y_pred in [("Random Forest", rf_y_pred), ("Neural Network", nn_y_pred)]:
                st.subheader(f"Model Evaluation - {model_name}")
                acc = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {acc:.2f}")
                st.text("Classification Report:")
                st.text(classification_report(y_test, y_pred))

                cm = confusion_matrix(y_test, y_pred)
                cm_labels = ['Fatal', 'Serious', 'Slight']
                fig_cm = ff.create_annotated_heatmap(cm, x=cm_labels, y=cm_labels, colorscale='Blues')
                fig_cm.update_layout(xaxis_title='Predicted', yaxis_title='Actual', xaxis=dict(side='bottom'), yaxis=dict(autorange='reversed'))
                st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.warning("Merged data is not available.")
    except Exception as e:
        st.error(f"Error in prediction tab: {e}")

# -------------------------
# 📥 Load Datasets
# -------------------------
DATA_URL_COLLISIONS = "datasets/dft-road-casualty-statistics-collision-2023.csv"
DATA_URL_CASUALTIES = "datasets/dft-road-casualty-statistics-casualty-2023.csv"
DATA_URL_VEHICLES = "datasets/dft-road-casualty-statistics-vehicle-2023.csv"

df_collision = load_data(DATA_URL_COLLISIONS)
df_casualties = load_data(DATA_URL_CASUALTIES)
df_vehicles = load_data(DATA_URL_VEHICLES)

df_merged = None
if df_collision is not None and df_casualties is not None and df_vehicles is not None:
    try:
        df_merged = pd.merge(df_collision, df_casualties, on='accident_index', how='inner')
        df_merged = pd.merge(df_merged, df_vehicles, on='accident_index', how='inner')
    except Exception as e:
        st.error(f"Error merging datasets: {e}")

# -------------------------
# 🧭 App UI
# -------------------------
st.title("UK Road Accident Analysis")
st.subheader("Based on 2023 Data")
st.markdown("Use the filters on the left sidebar to explore the data.")

# -------------------------
# 🎛️ Sidebar Filters
# -------------------------
st.sidebar.header("🔎 Filters")
selected_junctions = []
selected_regions = []
selected_severities = []
selected_months = []

if df_collision is not None:
    try:
        junction_type_labels = {
            0: "Not at junction", 1: "Roundabout", 2: "Mini-roundabout", 3: "T or staggered junction",
            5: "Slip road", 6: "Crossroads", 7: "More than 4 arms", 8: "Private drive", 9: "Other junction"
        }
        available_junctions = sorted([j for j in df_collision['junction_detail'].dropna().unique() if j not in [99, -1]])
        selected_junctions = st.sidebar.multiselect(
            "Select Junction Types", options=available_junctions,
            default=[1, 2, 3], format_func=lambda x: junction_type_labels.get(x, str(x))
        )

        available_regions = sorted(df_collision['local_authority_ons_district'].dropna().unique())
        selected_regions = st.sidebar.multiselect("Select Regions", available_regions, default=available_regions[:5])

        severity_map = {1: "Fatal", 2: "Serious", 3: "Slight"}
        available_severities = df_collision['accident_severity'].dropna().unique()
        selected_severities = st.sidebar.multiselect("Select Severity", available_severities, default=available_severities, format_func=lambda x: severity_map.get(x, str(x)))

        df_merged['date'] = pd.to_datetime(df_merged['date'], errors='coerce')
        df_merged['month'] = df_merged['date'].dt.month
        months_map = {i: name for i, name in enumerate(["January", "February", "March", "April", "May", "June",
                                                        "July", "August", "September", "October", "November", "December"], start=1)}
        available_months = sorted(df_merged['month'].dropna().unique())
        selected_months = st.sidebar.multiselect("Select Month(s)", available_months, default=available_months, format_func=lambda x: months_map.get(x, f"Month {x}"))
    except Exception as e:
        st.sidebar.error(f"Error in filters: {e}")

# -------------------------
# 🔍 Filter Data
# -------------------------
df_filtered = None
if df_merged is not None:
    try:
        df_filtered = df_merged[
            df_merged['junction_detail'].isin(selected_junctions) &
            df_merged['local_authority_ons_district'].isin(selected_regions) &
            df_merged['accident_severity'].isin(selected_severities) &
            df_merged['month'].isin(selected_months)
        ].copy()
        df_filtered = df_filtered.dropna(subset=['latitude', 'longitude'])
        df_filtered['rounded_location'] = df_filtered['latitude'].round(4).astype(str) + ", " + df_filtered['longitude'].round(4).astype(str)
    except Exception as e:
        st.error(f"Error filtering data: {e}")

# -------------------------
# 📊 Aggregation
# -------------------------
intersection_accident_counts = pd.DataFrame()
if df_filtered is not None and not df_filtered.empty:
    try:
        intersection_accident_counts = (
            df_filtered.groupby('rounded_location')
            .size()
            .sort_values(ascending=False)
            .reset_index(name='accident_frequency')
        )
        intersection_accident_counts[['latitude', 'longitude']] = intersection_accident_counts['rounded_location'].str.split(', ', expand=True).astype(float)
        max_freq = intersection_accident_counts['accident_frequency'].max()
        intersection_accident_counts['marker_size'] = intersection_accident_counts['accident_frequency'] / max_freq
    except Exception as e:
        st.error(f"Error aggregating intersection data: {e}")

top_n = st.sidebar.slider("Top N Intersections", 1, 100, 10)

# -------------------------
# 🧾 Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "About", "📍 Map", "📊 Data & Stats", "Insights", "Download", "🧠 Accident Severity Prediction"
])

with tab1:
    st.markdown("### ℹ️ About")
    st.markdown("This dashboard analyzes UK road accident data for 2023.")

with tab2:
    st.markdown("### 🗺️ Map of Top Intersections")
    if not intersection_accident_counts.empty:
        map_data = intersection_accident_counts[['latitude', 'longitude', 'accident_frequency']].head(top_n)
        map_data['marker_size'] = map_data['accident_frequency'] / intersection_accident_counts['accident_frequency'].max() * 10
        st.map(map_data, size='marker_size')
    else:
        st.warning("No data to display on the map.")

with tab3:
    st.markdown("### 🚦 Top Intersections with Accidents")
    if not intersection_accident_counts.empty:
        display_df = intersection_accident_counts[['rounded_location', 'latitude', 'longitude', 'accident_frequency']].head(top_n)
        st.dataframe(display_df)
        fig_bar = px.bar(display_df, x='rounded_location', y='accident_frequency', color='accident_frequency', color_continuous_scale='Reds')
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)

        severity_dist = df_filtered['accident_severity'].map(severity_map).value_counts().reset_index()
        severity_dist.columns = ['Severity', 'Count']
        fig_pie = px.pie(severity_dist, names='Severity', values='Count')
        st.plotly_chart(fig_pie, use_container_width=True)

with tab4:
    st.markdown("### 📌 Insights")
    if not intersection_accident_counts.empty:
        most = intersection_accident_counts.iloc[0]
        st.markdown(f"- Most dangerous location: **{most['rounded_location']}** with **{most['accident_frequency']}** accidents.")
    st.markdown(f"- Filtered data contains **{len(df_filtered)}** accident records." if df_filtered is not None else "No data.")

with tab5:
    st.markdown("### 📤 Download")
    if not intersection_accident_counts.empty:
        csv = intersection_accident_counts.head(top_n).to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="top_intersections.csv", mime="text/csv")

with tab6:
    accident_severity_prediction_tab(df_merged)

# -------------------------
# ✅ End of Script
# -------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and UK Road Safety Data")
