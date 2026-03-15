
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import optuna
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.semi_supervised import LabelPropagation

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from prophet import Prophet

from stable_baselines3 import PPO
import gymnasium as gym

from openai import OpenAI

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


st.set_page_config(layout="wide")
st.title("🚀 Enterprise AI Analytics Platform")

# =====================================================
# DATA UPLOAD
# =====================================================

file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])

if file:

    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.success("Dataset Loaded")
    st.dataframe(df.head())

# =====================================================
# DATA CLEANING
# =====================================================

    df = df.drop_duplicates()

    for col in df.columns:

        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])

        else:
            df[col] = df[col].fillna(df[col].mean())

# =====================================================
# FEATURE ENGINEERING
# =====================================================

    numeric = df.select_dtypes(include=np.number).columns

    for col in numeric:

        df[f"{col}_square"] = df[col] ** 2
        df[f"{col}_log"] = np.log1p(np.abs(df[col]) + 1)

# =====================================================
# DASHBOARD BUILDER
# =====================================================

    st.subheader("📊 Dashboard Builder")

    chart = st.selectbox("Chart Type",
        ["Histogram","Scatter","Box","Line"])

    x = st.selectbox("X Column", df.columns)

    if chart == "Histogram":
        fig = px.histogram(df,x=x)

    elif chart == "Scatter":
        y = st.selectbox("Y Column", df.columns)
        fig = px.scatter(df,x=x,y=y)

    elif chart == "Line":
        fig = px.line(df,y=x)

    else:
        fig = px.box(df,y=x)

    st.plotly_chart(fig,use_container_width=True)

# =====================================================
# AUTOML
# =====================================================

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

    result = pd.DataFrame(scores.items(),
                          columns=["Model","Score"]
                          ).sort_values("Score",ascending=False)

    st.dataframe(result)

    best_model_name = result.iloc[0]["Model"]
    best_model = models[best_model_name]

    st.success(f"Best Model: {best_model_name}")

# =====================================================
# HYPERPARAMETER TUNING
# =====================================================

st.subheader("⚙ Hyperparameter Optimization")

if metric == "Accuracy":   # Classification

    def objective(trial):

        n = trial.suggest_int("n_estimators",50,200)
        depth = trial.suggest_int("max_depth",3,10)

        model = RandomForestClassifier(
            n_estimators=n,
            max_depth=depth
        )

        model.fit(X_train,y_train)

        pred = model.predict(X_test)

        return accuracy_score(y_test,pred)

else:   # Regression

    def objective(trial):

        n = trial.suggest_int("n_estimators",50,200)
        depth = trial.suggest_int("max_depth",3,10)

        model = RandomForestRegressor(
            n_estimators=n,
            max_depth=depth
        )

        model.fit(X_train,y_train)

        pred = model.predict(X_test)

        return r2_score(y_test,pred)


study = optuna.create_study(direction="maximize")

study.optimize(objective,n_trials=10)

st.write("Best Parameters:", study.best_params)

# =====================================================
# EXPLAINABLE AI
# =====================================================

    st.subheader("🧠 Explainable AI")

    explainer = shap.Explainer(best_model,X_train)
    shap_values = explainer(X_test)

    shap.summary_plot(shap_values,X_test,show=False)

    fig = plt.gcf()
    st.pyplot(fig)

# =====================================================
# CLUSTERING
# =====================================================

    st.subheader("📍 Clustering")

    k = st.slider("Clusters",2,10,3)

    kmeans = KMeans(n_clusters=k)
    clusters = kmeans.fit_predict(X)

    pca = PCA(n_components=2)
    comp = pca.fit_transform(X)

    cluster_df = pd.DataFrame({

        "PC1": comp[:,0],
        "PC2": comp[:,1],
        "Cluster": clusters

    })

    fig = px.scatter(cluster_df,x="PC1",y="PC2",color="Cluster")

    st.plotly_chart(fig)

# =====================================================
# ANOMALY DETECTION
# =====================================================

    st.subheader("🚨 Anomaly Detection")

    iso = IsolationForest()

    df["anomaly"] = iso.fit_predict(X)

    st.write(df["anomaly"].value_counts())

# =====================================================
# SEMI SUPERVISED
# =====================================================

    st.subheader("Semi-Supervised Learning")

    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    semi = LabelPropagation()

    semi.fit(X_train,y_train)

    pred = semi.predict(X_test)

    st.write("Accuracy:",accuracy_score(y_test,pred))

# =====================================================
# TIME SERIES
# =====================================================

    if "date" in df.columns:

        st.subheader("📈 Forecasting")

        df["date"] = pd.to_datetime(df["date"])

        ts = df[["date",target]]
        ts.columns = ["ds","y"]

        model = Prophet()

        model.fit(ts)

        future = model.make_future_dataframe(periods=30)

        forecast = model.predict(future)

        fig = model.plot(forecast)

        st.pyplot(fig)

# =====================================================
# RL DEMO
# =====================================================

    st.subheader("🤖 Reinforcement Learning")

    env = gym.make("CartPole-v1")

    rl_model = PPO("MlpPolicy",env)

    rl_model.learn(total_timesteps=2000)

    st.success("RL Training Complete")

# =====================================================
# DATASET CHAT
# =====================================================

    st.subheader("💬 Dataset Chat")

    api_key = st.text_input("OpenAI API Key",type="password")

    question = st.text_input("Ask about dataset")

    if api_key and question:

        client = OpenAI(api_key=api_key)

        prompt = f"""
        Dataset columns: {list(df.columns)}
        Question: {question}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}]
        )

        st.write(response.choices[0].message.content)

# =====================================================
# PDF REPORT
# =====================================================

    st.subheader("📦 Generate Business Report")

    if st.button("Create PDF Report"):

        report="AI_Report.pdf"

        c = canvas.Canvas(report,pagesize=letter)

        best_score = result.iloc[0]["Score"]

        c.drawString(100,750,"AI Data Analysis Report")
        c.drawString(100,720,f"Rows: {df.shape[0]}")
        c.drawString(100,700,f"Columns: {df.shape[1]}")
        c.drawString(100,680,f"Best Model Score: {best_score}")

        c.save()

        with open(report,"rb") as f:

            st.download_button(
                "Download Report",
                f,
                file_name="AI_Report.pdf"
            )

else:

    st.info("Upload dataset to start analysis")
