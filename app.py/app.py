# Required- Python-Libraries

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import optuna
import matplotlib.pyplot as plt

# Model- training/ Future- Engineering

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score

# ML- Algorithms

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

# CLUSTER- Techniques

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.semi_supervised import LabelPropagation

# Self-Training Classifier

from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier

from prophet import Prophet
from stable_baselines3 import PPO
import gymnasium as gym

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

st.set_page_config(layout="wide")
st.title("🚀 AI Analytics Platform")

# ======================================================
# 1 FILE UPLOAD
# ======================================================

file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])

if file:

    # ---------------- LOAD DATA ----------------
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.success("Dataset Loaded")

    st.dataframe(df.head())
    st.dataframe(df.tail())

    st.write("Shape:", df.shape)
    st.write("Total Values:", df.size)
    st.write("Statistical Info:")
    st.write(df.describe())

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

    chart = st.selectbox(
        "Chart Type",
        ["Histogram","Scatter","Box","Line","Bar","Pie"]
    )

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

    if y.dtype == "object":

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

        if metric == "Accuracy":
            score = accuracy_score(y_test,pred)
        else:
            score = r2_score(y_test,pred)

        scores[name] = score

    result = pd.DataFrame(
        scores.items(),
        columns=["Model","Score"]
    ).sort_values("Score",ascending=False)

    st.dataframe(result)

    best_model_name = result.iloc[0]["Model"]
    best_model = models[best_model_name]


    # ======================================================
    # 6 HYPERPARAMETER TUNING
    # ======================================================
    st.subheader("⚙ Hyperparameter Optimization")

    if metric=="Accuracy":

        def objective(trial):

            n=trial.suggest_int("n_estimators",50,200)
            depth=trial.suggest_int("max_depth",3,10)

            model=RandomForestClassifier(
                n_estimators=n,max_depth=depth)

            model.fit(X_train,y_train)

            pred=model.predict(X_test)

            return accuracy_score(y_test,pred)

    else:

        def objective(trial):

            n=trial.suggest_int("n_estimators",50,200)
            depth=trial.suggest_int("max_depth",3,10)

            model=RandomForestRegressor(
                n_estimators=n,max_depth=depth)

            model.fit(X_train,y_train)

            pred=model.predict(X_test)

            return r2_score(y_test,pred)

    study=optuna.create_study(direction="maximize")
    study.optimize(objective,n_trials=10)

    st.write("Best Parameters:",study.best_params)

    # ======================================================
    # 7 EXPLAINABLE AI
    # ======================================================

    st.subheader("🧠 Explainable AI")

    best_model.fit(X_train,y_train)
    explainer = shap.Explainer(best_model,X_train)
    shap_values = explainer(X_test)

    shap.summary_plot(shap_values,X_test,show=False)
    st.pyplot(plt.gcf())

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
    # 10 SEMI SUPERVISED
    # ======================================================

st.subheader("🔁 Self-Training Classifier")

if task == "classification":

    # Create unlabeled data
    y_semi = y_train.copy()
    mask = np.random.rand(len(y_semi)) < 0.3
    y_semi[mask] = -1

    base_model = RandomForestClassifier()

    self_model = SelfTrainingClassifier(base_model)

    self_model.fit(X_train, y_semi)

    pred = self_model.predict(X_test)

    st.success("Self-Training Accuracy:")
    st.write(accuracy_score(y_test, pred))

else:
    st.info("Only works for classification datasets")

    # ======================================================
    # 11 TIME SERIES
    # ======================================================

    if "date" in df.columns:

        st.subheader("📈 Forecast")

        df["date"] = pd.to_datetime(df["date"])

        ts = df[["date",target]]
        ts.columns = ["ds","y"]

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
    model = PPO("MlpPolicy",env)
    model.learn(total_timesteps=2000)

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
