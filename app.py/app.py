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

# Optional torch
try:
    import torch
except:
    torch = None

# UI
st.set_page_config(layout="wide")
st.title("🚀 AI Analytics Platform")

# Upload
file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])

if file:

    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

    st.success("Dataset Loaded")
    st.dataframe(df.head())
    st.dataframe(df.tail())

    st.write("Shape:", df.shape)
    st.write("Statistics:", df.describe())

    # ----------------------------
    # CLEANING
    # ----------------------------
    df = df.drop_duplicates()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna("Unknown")

    # ----------------------------
    # FEATURE ENGINEERING
    # ----------------------------
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        df[f"{col}_square"] = df[col]**2
        df[f"{col}_log"] = np.log1p(np.abs(df[col]) + 1)

    # ----------------------------
    # DASHBOARD
    # ----------------------------
    st.subheader("📊 Dashboard")

    chart = st.selectbox("Chart", ["Histogram","Scatter","Line","Box","Bar","Pie"])
    x = st.selectbox("X", df.columns)

    if chart == "Histogram":
        fig = px.histogram(df, x=x)
    elif chart == "Scatter":
        y_col = st.selectbox("Y", df.columns)
        fig = px.scatter(df, x=x, y=y_col)
    elif chart == "Line":
        fig = px.line(df, y=x)
    elif chart == "Box":
        fig = px.box(df, y=x)
    elif chart == "Bar":
        y_col = st.selectbox("Y", df.columns)
        fig = px.bar(df, x=x, y=y_col)
    else:
        pie_data = df[x].value_counts().reset_index()
        pie_data.columns = [x,"count"]
        fig = px.pie(pie_data, names=x, values="count")

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # AUTOML
    # ----------------------------
    st.subheader("🤖 AutoML")

    target = st.selectbox("Target", df.columns)

    if len(np.unique(df[target])) < 2:
        st.error("Target must have at least 2 classes")
        st.stop()

    X = df.drop(columns=[target])
    y = df[target]

    X = pd.get_dummies(X)
    X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce')
    X = X.replace([np.inf,-np.inf], np.nan)
    X = X.fillna(X.mean())
    X = X.fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Detect task
    if y.dtype == "object" or len(np.unique(y)) < 20:
        task = "classification"
    else:
        task = "regression"

    from collections import Counter
    class_counts = Counter(y)

    if task=="classification" and all(v>=2 for v in class_counts.values()):
        X_train,X_test,y_train,y_test = train_test_split(
            X,y,test_size=0.2,random_state=42,stratify=y)
    else:
        X_train,X_test,y_train,y_test = train_test_split(
            X,y,test_size=0.2,random_state=42)

    if task=="classification":
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        models = {
            "LR":LogisticRegression(max_iter=1000),
            "RF":RandomForestClassifier(),
            "SVM":SVC(),
            "KNN":KNeighborsClassifier(),
            "XGB":XGBClassifier(),
            "LGBM":LGBMClassifier(),
            "CAT":CatBoostClassifier(verbose=0)
        }
        metric="Accuracy"
    else:
        models = {
            "LR":LinearRegression(),
            "RF":RandomForestRegressor(),
            "SVR":SVR(),
            "KNN":KNeighborsRegressor(),
            "XGB":XGBRegressor(),
            "LGBM":LGBMRegressor(),
            "CAT":CatBoostRegressor(verbose=0)
        }
        metric="R2"

    scores={}
    for name,m in models.items():
        m.fit(X_train,y_train)
        p=m.predict(X_test)
        s=accuracy_score(y_test,p) if metric=="Accuracy" else r2_score(y_test,p)
        scores[name]=s

    result=pd.DataFrame(scores.items(),columns=["Model","Score"]).sort_values("Score",ascending=False)
    st.dataframe(result)

    best=models[result.iloc[0]["Model"]]

    # ----------------------------
    # SHAP
    # ----------------------------
    st.subheader("🧠 Explainable AI")

    try:
        X_train_df=pd.DataFrame(X_train)
        X_test_df=pd.DataFrame(X_test)

        explainer=shap.Explainer(best,X_train_df)
        shap_values=explainer(X_test_df)

        shap.summary_plot(shap_values,X_test_df,show=False)
        st.pyplot(plt.gcf()); plt.clf()
    except Exception as e:
        st.warning(e)

    # ----------------------------
    # CLUSTERING
    # ----------------------------
    st.subheader("📍 Clustering")
    k=st.slider("Clusters",2,10,3)
    clusters=KMeans(n_clusters=k).fit_predict(X)
    comp=PCA(n_components=2).fit_transform(X)

    st.plotly_chart(px.scatter(x=comp[:,0],y=comp[:,1],color=clusters))

    # ----------------------------
    # ANOMALY
    # ----------------------------
    st.subheader("🚨 Anomaly")
    df["anomaly"]=IsolationForest().fit_predict(X)
    st.write(df["anomaly"].value_counts())

    # ----------------------------
    # SEMI
    # ----------------------------
    if task=="classification":
        st.subheader("🔁 Semi-Supervised")
        y_semi=y_train.copy()
        y_semi[np.random.rand(len(y_semi))<0.3]=-1

        semi=LabelPropagation().fit(X_train,y_semi)
        st.write("LP:",accuracy_score(y_test,semi.predict(X_test)))

    # ----------------------------
    # TIME SERIES
    # ----------------------------
    if "date" in df.columns:
        st.subheader("📈 Forecast")
        df["date"]=pd.to_datetime(df["date"])
        ts=df[["date",target]].rename(columns={"date":"ds",target:"y"})
        m=Prophet().fit(ts)
        future=m.make_future_dataframe(30)
        st.pyplot(m.plot(m.predict(future)))

    # ----------------------------
    # RL
    # ----------------------------
    st.subheader("🤖 RL")
    env=gym.make("CartPole-v1")
    PPO("MlpPolicy",env).learn(2000)
    st.success("RL Done")

    # ----------------------------
    # PDF
    # ----------------------------
    if st.button("Create PDF"):
        c=canvas.Canvas("report.pdf",pagesize=letter)
        c.drawString(100,750,"AI Report")
        c.save()
        with open("report.pdf","rb") as f:
            st.download_button("Download",f,"report.pdf")

else:
    st.info("Upload dataset to start")
