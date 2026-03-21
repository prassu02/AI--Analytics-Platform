
# Required Libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import optuna
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score

# ML Models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

# Unsupervised / Semi
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.semi_supervised import LabelPropagation, SelfTrainingClassifier

from prophet import Prophet
from stable_baselines3 import PPO
import gymnasium as gym

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# UI
st.set_page_config(layout="wide")
st.title("🚀 AI Analytics Platform")

# ======================================================
# 1 FILE UPLOAD
# ======================================================

file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])

if file:

    # LOAD
    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

    st.success("Dataset Loaded")
    st.dataframe(df.head())
    st.dataframe(df.tail())

    st.write("Shape:", df.shape)
    st.write("Total Values:", df.size)
    st.write("Statistics:", df.describe())

    # ======================================================
    # 2 DATA CLEANING
    # ======================================================

    df = df.drop_duplicates()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

    # ======================================================
    # 3 FEATURE ENGINEERING
    # ======================================================

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        df[f"{col}_square"] = df[col]**2
        df[f"{col}_log"] = np.log1p(np.abs(df[col]) + 1)

    # ======================================================
    # 4 DASHBOARD BUILDER
    # ======================================================

    st.subheader("📊 Dashboard Builder")

    chart = st.selectbox("Chart Type",
        ["Histogram","Scatter","Box","Line","Bar","Pie"])

    x = st.selectbox("X Column", df.columns)

    if chart == "Histogram":
        fig = px.histogram(df, x=x)

    elif chart == "Scatter":
        y_col = st.selectbox("Y Column", df.columns)
        fig = px.scatter(df, x=x, y=y_col)

    elif chart == "Line":
        fig = px.line(df, y=x)

    elif chart == "Box":
        fig = px.box(df, y=x)

    elif chart == "Bar":
        y_col = st.selectbox("Y Column", df.columns)
        fig = px.bar(df, x=x, y=y_col)

    elif chart == "Pie":
        pie_data = df[x].value_counts().reset_index()
        pie_data.columns = [x, "count"]
        fig = px.pie(pie_data, names=x, values="count")

    st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # 5 AUTOML
    # ======================================================

    st.subheader("🤖 AutoML")

    target = st.selectbox("Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    X = pd.get_dummies(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    # Detect task
    if y.dtype == "object" or len(np.unique(y)) < 20:
        task = "classification"
    else:
        task = "regression"

    if task == "classification":

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        models = {
            "LogisticRegression": LogisticRegression(),
            "RandomForest": RandomForestClassifier(),
            "SVM": SVC(),
            "KNN": KNeighborsClassifier(),
            "XGBoost": XGBClassifier(),
            "LightGBM": LGBMClassifier(),
            "CatBoost": CatBoostClassifier(verbose=0)
        }

        metric = "Accuracy"

    else:

        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(),
            "SVR": SVR(),
            "KNN": KNeighborsRegressor(),
            "XGBoost": XGBRegressor(),
            "LightGBM": LGBMRegressor(),
            "CatBoost": CatBoostRegressor(verbose=0)
        }

        metric = "R2"

    scores = {}

    for name,model in models.items():
        model.fit(X_train,y_train)
        pred = model.predict(X_test)

        score = accuracy_score(y_test,pred) if metric=="Accuracy" else r2_score(y_test,pred)
        scores[name] = score

    result = pd.DataFrame(scores.items(),columns=["Model","Score"]).sort_values("Score",ascending=False)
    st.dataframe(result)

    best_model = models[result.iloc[0]["Model"]]

    # ======================================================
    # 6 HYPERPARAMETER TUNING
    # ======================================================

    st.subheader("⚙ Hyperparameter Optimization")

    def objective(trial):

        n = trial.suggest_int("n_estimators",50,200)
        depth = trial.suggest_int("max_depth",3,10)

        if task == "classification":
            model = RandomForestClassifier(n_estimators=n,max_depth=depth)
            model.fit(X_train,y_train)
            return accuracy_score(y_test,model.predict(X_test))

        else:
            model = RandomForestRegressor(n_estimators=n,max_depth=depth)
            model.fit(X_train,y_train)
            return r2_score(y_test,model.predict(X_test))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective,n_trials=10)

    st.write("Best Parameters:",study.best_params)

    # ======================================================
    # 7 EXPLAINABLE AI + FEATURE IMPORTANCE DASHBOARD
    # ======================================================

    st.subheader("🧠 Explainable AI Dashboard")

# Train best model again (safe)
best_model.fit(X_train, y_train)

# ---------- FEATURE IMPORTANCE (GLOBAL) ----------
st.markdown("### 📊 Feature Importance (Global)")

if hasattr(best_model, "feature_importances_"):

    importances = best_model.feature_importances_

    feat_df = pd.DataFrame({
        "Feature": pd.DataFrame(X).columns,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    fig_imp = px.bar(
        feat_df.head(15),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Features"
    )

    st.plotly_chart(fig_imp, use_container_width=True)

else:
    st.info("Feature importance not available for this model")

# ---------- SHAP EXPLAINER ----------
st.markdown("### 🔍 SHAP Global Explanation")

try:
    explainer = shap.Explainer(best_model, X_train)
    shap_values = explainer(X_test)

    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(plt.gcf())

except Exception as e:
    st.warning(f"SHAP failed: {e}")

# ---------- SHAP DEPENDENCE ----------
st.markdown("### 🔗 SHAP Feature Interaction")

feature_names = pd.DataFrame(X).columns.tolist()

selected_feature = st.selectbox(
    "Select feature for SHAP dependence",
    feature_names
)

try:
    shap.dependence_plot(
        selected_feature,
        shap_values.values,
        X_test,
        show=False
    )
    st.pyplot(plt.gcf())

except:
    st.info("Dependence plot not supported for this model")

# ---------- LOCAL EXPLANATION ----------
st.markdown("### 🎯 Individual Prediction Explanation")

row_id = st.slider("Select Row", 0, len(X_test)-1, 0)

try:
    shap.plots.waterfall(shap_values[row_id])
    st.pyplot(plt.gcf())

except:
    st.info("Waterfall plot not supported")

# ---------- MODEL PERFORMANCE VISUAL ----------
st.markdown("### 📈 Model Performance")

pred = best_model.predict(X_test)

if metric == "Accuracy":

    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_test, pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)

else:

    fig = px.scatter(
        x=y_test,
        y=pred,
        labels={"x":"Actual","y":"Predicted"},
        title="Actual vs Predicted"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # 8 CLUSTERING
    # ======================================================

    st.subheader("📍 Clustering")

    k = st.slider("Clusters",2,10,3)
    clusters = KMeans(n_clusters=k).fit_predict(X)

    comp = PCA(n_components=2).fit_transform(X)

    cluster_df = pd.DataFrame({
        "PC1": comp[:,0],
        "PC2": comp[:,1],
        "Cluster": clusters
    })

    st.plotly_chart(px.scatter(cluster_df,x="PC1",y="PC2",color="Cluster"))

    # ======================================================
    # 9 ANOMALY DETECTION
    # ======================================================

    st.subheader("🚨 Anomaly Detection")

    df["anomaly"] = IsolationForest().fit_predict(X)
    st.write(df["anomaly"].value_counts())

    # ======================================================
    # 10 SEMI-SUPERVISED
    # ======================================================

    st.subheader("🔁 Semi-Supervised Learning")

    if task == "classification":

        y_semi = y_train.copy()
        mask = np.random.rand(len(y_semi)) < 0.3
        y_semi[mask] = -1

        semi = LabelPropagation()
        semi.fit(X_train,y_semi)

        pred = semi.predict(X_test)
        st.write("LabelPropagation Accuracy:",accuracy_score(y_test,pred))

        # Self Training
        st.subheader("🔁 Self-Training Classifier")

        base_model = RandomForestClassifier()
        self_model = SelfTrainingClassifier(base_model)

        self_model.fit(X_train,y_semi)
        pred2 = self_model.predict(X_test)

        st.write("SelfTraining Accuracy:",accuracy_score(y_test,pred2))

    else:
        st.info("Semi-supervised works only for classification")

    # ======================================================
    # 11 TIME SERIES
    # ======================================================

    if "date" in df.columns:

        st.subheader("📈 Forecast")

        df["date"] = pd.to_datetime(df["date"])
        ts = df[["date",target]].rename(columns={"date":"ds",target:"y"})

        model = Prophet()
        model.fit(ts)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        st.pyplot(model.plot(forecast))

    # ======================================================
    # 12 RL
    # ======================================================

    st.subheader("🤖 Reinforcement Learning")

    env = gym.make("CartPole-v1")
    rl_model = PPO("MlpPolicy",env)
    rl_model.learn(total_timesteps=2000)

    st.success("RL Training Done")

    # ======================================================
    # 13 PDF REPORT
    # ======================================================

    st.subheader("📦 Generate Report")

    if st.button("Create PDF"):

        c = canvas.Canvas("report.pdf",pagesize=letter)

        best_score = result.iloc[0]["Score"]

        c.drawString(100,750,"AI Report")
        c.drawString(100,720,f"Rows: {df.shape[0]}")
        c.drawString(100,700,f"Columns: {df.shape[1]}")
        c.drawString(100,680,f"Score: {best_score}")

        c.save()

        with open("report.pdf","rb") as f:
            st.download_button("Download",f,"report.pdf")

    else:
        st.info("Upload dataset to start")
