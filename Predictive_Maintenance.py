import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# ======================
# LOAD MODEL + DATA
# ======================
model = joblib.load("xgboost_failure_type.pkl")
le = joblib.load("label_encoder.pkl")

df = pd.read_excel("ai4i2020_cleaned.xlsx")

df.columns = df.columns.str.replace('[', '', regex=False)
df.columns = df.columns.str.replace(']', '', regex=False)
df.columns = df.columns.str.replace(' ', '_')

st.set_page_config(layout="wide")
st.title("Predictive Maintenance System")

FIG_SIZE = (3.2, 2)
type_map = {"L": 0, "M": 1, "H": 2}

# ==========
# NAVIGATION
# ==========
page = st.sidebar.radio(
    "Navigation",
    [
        "Exploratory Data Analysis",
        "Model Performance",
        "Prediction",
        "Dashboard"
    ]
)

# =========================================================
# 1. EDA PAGE
# =========================================================
if page == "Exploratory Data Analysis":

    st.subheader("Failure Analytics Dashboard")

    col1, col2 = st.columns(2)

    # ---------------- PARETO
    with col1:

        failure_counts = {
            "TWF": df["TWF"].sum(),
            "HDF": df["HDF"].sum(),
            "PWF": df["PWF"].sum(),
            "OSF": df["OSF"].sum(),
            "RNF": df["RNF"].sum()
        }

        pareto_df = pd.DataFrame(list(failure_counts.items()),
                                 columns=["Failure_Type", "Count"]).sort_values("Count", ascending=False)

        pareto_df["Cumulative_%"] = pareto_df["Count"].cumsum() / pareto_df["Count"].sum() * 100

        fig, ax1 = plt.subplots(figsize=(5,3))
        ax1.bar(pareto_df["Failure_Type"], pareto_df["Count"], color="steelblue")
        ax2 = ax1.twinx()
        ax2.plot(pareto_df["Failure_Type"], pareto_df["Cumulative_%"], color="red", marker="o")

        st.pyplot(fig)

    # ---------------- FEATURE IMPORTANCE
    with col2:

        feature_names = [
            "Type",
            "Air_temperature",
            "Process_temperature",
            "RPM",
            "Torque",
            "Tool_wear"
        ]

        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance")

        fig, ax = plt.subplots(figsize=(5,3))
        ax.barh(fi_df["Feature"], fi_df["Importance"], color="green")
        
        ax.set_title("Feature Importance (XGBoost Model)")
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Features")

        st.pyplot(fig)

    st.markdown("---")

    # ---------------- SCATTER MATRIX
    st.markdown("### Scatter Matrix (Failure Pattern)")

    df_fail = df[df["Machine_failure"] == 1]
    df_ok = df[df["Machine_failure"] == 0]

    df_sample = pd.concat([
        df_fail.sample(min(len(df_fail), 500), random_state=42),
        df_ok.sample(min(len(df_ok), 1000), random_state=42)
    ])

    features = [
        "Air_temperature_K",
        "Process_temperature_K",
        "Rotational_speed_rpm",
        "Torque_Nm",
        "Tool_wear_min"
    ]

    def get_failure_type(row):
        if row["TWF"] == 1: return "TWF"
        if row["HDF"] == 1: return "HDF"
        if row["PWF"] == 1: return "PWF"
        if row["OSF"] == 1: return "OSF"
        if row["RNF"] == 1: return "RNF"
        return "No Failure"

    df_sample["Failure_Type"] = df_sample.apply(get_failure_type, axis=1)

    sns_plot = sns.pairplot(
        df_sample,
        vars=features,
        hue="Failure_Type",
        diag_kind="hist"
    )

    st.pyplot(sns_plot)

# =========================================================
# 2. Model Performance
# =========================================================

elif page == "Model Performance":

    st.title("Model Performance (Test Set Evaluation)")

    # ======================
    # IMPORT METRICS
    # ======================
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
        roc_curve,
        auc
    )
    from sklearn.preprocessing import label_binarize

    # ======================
    # LOAD TEST SET
    # ======================
    X_test = joblib.load("X_test.pkl")
    y_test = joblib.load("y_test.pkl")

    X_test = X_test.astype(float)

    # ======================
    # PREDICTIONS
    # ======================
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # ======================
    # METRICS
    # ======================
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    st.subheader("Overall Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("Precision", f"{precision:.4f}")
    col3.metric("Recall", f"{recall:.4f}")
    col4.metric("F1 Score", f"{f1:.4f}")

    st.markdown("---")

    # ======================
    # CONFUSION MATRIX
    # ======================
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    st.pyplot(fig)

    st.markdown("---")

    # ======================
    # ROC CURVE (MULTICLASS)
    # ======================
    st.subheader("ROC Curve (Multiclass)")

    y_test_bin = label_binarize(y_test, classes=np.arange(len(le.classes_)))

    fig, ax = plt.subplots(figsize=(7, 5))

    for i in range(len(le.classes_)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label=f"{le.classes_[i]} (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Multiclass")

    ax.legend()

    st.pyplot(fig)

    st.markdown("---")


# =========================================================
# 3. PREDICTION PAGE
# =========================================================
elif page == "Prediction":

    machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
    df_filtered = df[df["Type"] == machine_type]

    st.subheader("Pediction with XGBOOST")

    col1, col2, col3, col4, col5 = st.columns(5)

    # ---------------- Air Temp
    with col1:
        st.markdown("Air Temp (K)")
        fig, ax = plt.subplots()
        ax.hist(df_filtered["Air_temperature_K"], bins=25)
        air_temp = st.slider("K", 290.0, 310.0, 300.0)
        ax.axvline(air_temp, color="red")
        st.pyplot(fig)

    # ---------------- Process Temp
    with col2:
        st.markdown("Process Temp (K)")
        fig, ax = plt.subplots()
        ax.hist(df_filtered["Process_temperature_K"], bins=25)
        process_temp = st.slider("K", 300.0, 320.0, 310.0)
        ax.axvline(process_temp, color="red")
        st.pyplot(fig)

    # ---------------- RPM
    with col3:
        st.markdown("RPM")
        fig, ax = plt.subplots()
        ax.hist(df_filtered["Rotational_speed_rpm"], bins=25)
        rpm = st.slider("rpm", 1000, 2500, 1500)
        ax.axvline(rpm, color="red")
        st.pyplot(fig)

    # ---------------- Torque
    with col4:
        st.markdown("Torque")
        fig, ax = plt.subplots()
        ax.hist(df_filtered["Torque_Nm"], bins=25)
        torque = st.slider("Nm", 10.0, 80.0, 40.0)
        ax.axvline(torque, color="red")
        st.pyplot(fig)

    # ---------------- Tool Wear
    with col5:
        st.markdown("Tool Wear")
        fig, ax = plt.subplots()
        ax.hist(df_filtered["Tool_wear_min"], bins=25)
        tool_wear = st.slider("min", 0, 250, 100)
        ax.axvline(tool_wear, color="red")
        st.pyplot(fig)

    # ======================
    # PREDICTION
    # ======================
    input_data = np.array([[type_map[machine_type],
                            air_temp,
                            process_temp,
                            rpm,
                            torque,
                            tool_wear]])

    probs = model.predict_proba(input_data)[0]
    pred = np.argmax(probs)

    failure_type = le.inverse_transform([pred])[0]
    confidence = float(np.max(probs))

    st.subheader("Result")

    colA, colB = st.columns(2)

    with colA:
        if failure_type == "No_Failure":
            st.success("✅ HEALTHY")
        else:
            st.error(f"⚠ FAILURE: {failure_type}")

    with colB:
        st.metric("Confidence", f"{confidence:.2f}")

# =========================================================
# 4. MAINTENANCE DASHBOARD
# =========================================================
elif page == "Dashboard":

    st.title("Dashboard")

    df_base = df.copy()
    df_base["Date_Time"] = pd.to_datetime(df_base["Date_Time"])
    df_base = df_base.sort_values("Date_Time")

    col1, col2 = st.columns([3, 2])

    with col2:

        st.markdown("### Filters")

        machine_filter = st.multiselect(
            "Select Machine(s)",
            df_base["Machine_ID"].unique(),
            default=df_base["Machine_ID"].unique()
        )

        df_filtered = df_base[df_base["Machine_ID"].isin(machine_filter)]

        min_date = df_filtered["Date_Time"].min()
        max_date = df_filtered["Date_Time"].max()

        date_range = st.date_input(
            "Select Time Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if len(date_range) == 2:
            start_date, end_date = date_range
            df_filtered = df_filtered[
                (df_filtered["Date_Time"] >= pd.to_datetime(start_date)) &
                (df_filtered["Date_Time"] <= pd.to_datetime(end_date))
            ]

    # ======================
    # KPI
    # ======================
    with col1:

        k1, k2, k3 = st.columns(3)

        with k1:
            st.metric("Failure Rate", f"{df_filtered['Machine_failure'].mean()*100:.2f}%")
        with k2:
            st.metric("Total Failures", int(df_filtered["Machine_failure"].sum()))
        with k3:
            st.metric("Machines", df_filtered["Machine_ID"].nunique())

    st.markdown("---")

    # ======================
    # FAILURE ANALYSIS
    # ======================
    st.subheader("Failure Analysis")

    colA, colB = st.columns(2)

    with colA:

        machine_failure = df_filtered.groupby("Machine_ID")["Machine_failure"].sum().reset_index()

        fig_pie = px.pie(
            machine_failure,
            names="Machine_ID",
            values="Machine_failure",
            hole=0.4
        )

        st.plotly_chart(fig_pie, use_container_width=True)

    with colB:

        failure_types = ["TWF", "HDF", "PWF", "OSF", "RNF"]

        failure_long = df_filtered.melt(
            id_vars=["Machine_ID"],
            value_vars=failure_types,
            var_name="Failure_Type",
            value_name="Count"
        )

        failure_long = failure_long[failure_long["Count"] > 0]

        fig_bar = px.bar(
            failure_long,
            x="Failure_Type",
            y="Count",
            color="Machine_ID",
            barmode="stack"
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ======================
    # TREND ANALYSIS
    # ======================
    st.subheader("Interactive Trend Analysis")

    metric = st.selectbox("Select Parameter", [
        "Air_temperature_K",
        "Process_temperature_K",
        "Rotational_speed_rpm",
        "Torque_Nm",
        "Tool_wear_min",
        "Machine_failure"
    ])

    trend_df = df_filtered.groupby(
        ["Date_Time", "Machine_ID"]
    )[metric].mean().reset_index()

    fig = px.line(
        trend_df,
        x="Date_Time",
        y=metric,
        color="Machine_ID",
        markers=True
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ======================
    # FAILURE TREND
    # ======================
    st.subheader("Failure Trend Over Time")

    # ----------------------
    # Failure Type Selector
    # ----------------------
    failure_option = st.selectbox(
        "Select Failure Type",
        ["Machine_failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
    )

    # ----------------------
    # Aggregate Trend
    # ----------------------
    failure_trend = df_filtered.groupby(
        ["Date_Time", "Machine_ID"]
    )[failure_option].sum().reset_index()

    # ----------------------
    # Dynamic Title
    # ----------------------
    fig2 = px.line(
        failure_trend,
        x="Date_Time",
        y=failure_option,
        color="Machine_ID",
        markers=True,
        title=f"{failure_option} Trend Over Time"
    )

    fig2.update_layout(
        yaxis_title="Failure Count",
        xaxis_title="Date Time"
    )

    st.plotly_chart(fig2, use_container_width=True)


    
