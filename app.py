# ===============================
# Smart Parking Occupancy Prediction System
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Parking Space Availability", layout="wide")

# -------------------------------
# USER DATABASE (SESSION BASED)
# -------------------------------
if "users" not in st.session_state:
    st.session_state.users = {
        "admin": {"password": "1234", "role": "Admin"},
        "user1": {"password": "user123", "role": "User"}
    }

# -------------------------------
# SESSION STATE
# -------------------------------
for key in ["logged_in", "current_user", "role", "page"]:
    if key not in st.session_state:
        st.session_state[key] = False if key == "logged_in" else ""

# -------------------------------
# BACKGROUND IMAGE FUNCTION
# -------------------------------
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:url("data:image/jpg;base64,{encoded}") center/cover no-repeat fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# -------------------------------
# AUTH STYLES
# -------------------------------
def auth_style():
    st.markdown("""
    <style>
    .title {
        text-align:center;
        font-size:46px;
        font-weight:800;
        color:#00FFD5;
        margin-top:30px;
        text-shadow:0 0 12px #00FFD5;
    }

    /* 🔥 UPDATED LOGIN TITLE STYLE */
    .subtitle {
        text-align:center;
        color:white;
        font-size:30px;      /* Increased size */
        font-weight:800;     /* Bold */
        letter-spacing:3px;
        margin-bottom:25px;
        text-transform:uppercase;
    }

    .box {
        width:420px;
        margin:auto;
        margin-top:50px;
        padding:2.5rem;
        background:rgba(255,255,255,0.15);
        backdrop-filter:blur(14px);
        border-radius:20px;
        box-shadow:0 20px 40px rgba(0,0,0,0.7);
    }

    .stButton button {
        width:100%;
        background:linear-gradient(90deg,#00FFD5,#00B3FF);
        color:black;
        border-radius:14px;
        height:45px;
        font-weight:bold;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# LOGIN PAGE
# -------------------------------
def login():
    set_background("city-square.jpg")
    auth_style()

    st.markdown('<div class="title">PARKING SPACE AVILABILITY</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">LOGIN</div>', unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        users = st.session_state.users
        if username in users and users[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.current_user = username
            st.session_state.role = users[username]["role"]
            st.success("Login Successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

    if st.button("Register"):
        st.session_state.page = "register"
        st.rerun()

    if st.button("Forgot Password"):
        st.session_state.page = "forgot"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# REGISTER PAGE
# -------------------------------
def register():
    set_background("city-square.jpg")
    auth_style()

    st.markdown('<div class="box">', unsafe_allow_html=True)
    st.subheader("📝 Register")

    username = st.text_input("New Username")
    password = st.text_input("New Password", type="password")
    role = st.selectbox("Role", ["User", "Admin"])

    if st.button("Create Account"):
        if username in st.session_state.users:
            st.error("User already exists")
        else:
            st.session_state.users[username] = {
                "password": password,
                "role": role
            }
            st.success("Account created successfully")
            st.session_state.page = ""
            st.rerun()

    if st.button("Back to Login"):
        st.session_state.page = ""
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# FORGOT PASSWORD PAGE
# -------------------------------
def forgot_password():
    set_background("city-square.jpg")
    auth_style()

    st.markdown('<div class="box">', unsafe_allow_html=True)
    st.subheader("🔑 Forgot Password")

    username = st.text_input("Username")

    if st.button("Reset Password"):
        if username in st.session_state.users:
            st.session_state.users[username]["password"] = "1234"
            st.success("Password reset to default: 1234")
        else:
            st.error("User not found")

    if st.button("Back to Login"):
        st.session_state.page = ""
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# AUTH ROUTER
# -------------------------------
if not st.session_state.logged_in:
    if st.session_state.page == "register":
        register()
    elif st.session_state.page == "forgot":
        forgot_password()
    else:
        login()
    st.stop()

# -------------------------------
# AFTER LOGIN BACKGROUND
# -------------------------------
set_background("istockphoto-881782390-170667a.jpg")

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("🏠 Navigation")
st.sidebar.write(f"👤 {st.session_state.current_user}")
st.sidebar.write(f"🔑 Role: {st.session_state.role}")

pages = ["Prediction"]
if st.session_state.role == "Admin":
    pages.insert(0, "Dashboard")

page = st.sidebar.radio("Go to", pages)

if st.sidebar.button("🚪 Logout"):
    for key in ["logged_in", "current_user", "role"]:
        st.session_state[key] = False if key == "logged_in" else ""
    st.rerun()

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("parking.csv")

df = load_data()

# -------------------------------
# DASHBOARD (ADMIN ONLY)
# -------------------------------
if page == "Dashboard":
    st.title("📊 Parking Data Dashboard")
    st.dataframe(df.head())
    st.write(df.describe())
    st.write(df.isnull().sum())

    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# -------------------------------
# MODEL TRAINING
# -------------------------------
def train_model():
    df_model = df.copy()
    le = LabelEncoder()

    for col in df_model.select_dtypes(include='object'):
        df_model[col] = le.fit_transform(df_model[col])

    X = df_model.iloc[:, :-1]
    y = df_model.iloc[:, -1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, scaler, mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred), X.columns

model, scaler, mae, r2, feature_names = train_model()

# -------------------------------
# PREDICTION PAGE
# -------------------------------
if page == "Prediction":
    st.title("🚗 Parking Occupancy Prediction")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    user_input = []
    for feature in feature_names:
        user_input.append(st.number_input(f"Enter {feature}", value=0.0))

    if st.button("Predict"):
        prediction = model.predict(
            scaler.transform(np.array(user_input).reshape(1, -1))
        )
        st.success(f"Predicted Parking Occupancy: **{prediction[0]:.2f}**")