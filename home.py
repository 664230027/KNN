import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Iris KNN Classifier", layout="wide")

# ============================================================
# HEADER
# ============================================================
st.markdown("""
    <h1 style='text-align:center; color:#4B4B4B;'>🌸 Iris Flower Classifier 🌸</h1>
    <h4 style='text-align:center; color:#7D7D7D;'>K-Nearest Neighbors (KNN) by Fluke</h4>
""", unsafe_allow_html=True)

st.write("")

# ============================================================
# TOP IMAGES
# ============================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<h4 style='text-align:center;'>Setosa</h4>", unsafe_allow_html=True)
    st.image("./img/iris3.jpg")

with col2:
    st.markdown("<h4 style='text-align:center;'>Versicolor</h4>", unsafe_allow_html=True)
    st.image("./img/iris1.jpg")

with col3:
    st.markdown("<h4 style='text-align:center;'>Virginica</h4>", unsafe_allow_html=True)
    st.image("./img/iris2.jpg")

st.markdown("---")

# ============================================================
# LOAD DATA
# ============================================================
st.markdown("""
<div style="background-color:#FFC2C2;padding:18px;border-radius:10px;border:2px solid #B34D4D;">
<h4 style="text-align:center; color:#4B4B4B;">📊 ข้อมูลชุดข้อมูลดอกไม้ (Iris Dataset)</h4>
</div>
""", unsafe_allow_html=True)

df = pd.read_csv("./data/iris.csv")
st.write(df.head())

st.markdown("---")

# ============================================================
# PREDICTION PANEL
# ============================================================
st.markdown("""
<div style="background-color:#C2F0FF;padding:18px;border-radius:10px;border:2px solid #4DA6B3;">
<h4 style="text-align:center; color:#4B4B4B;">🔮 ทำนายดอกไม้ด้วย KNN</h4>
</div>
""", unsafe_allow_html=True)

colA, colB = st.columns(2)

with colA:
    st.subheader("กรอกข้อมูลใบไม้:")
    sp_len = st.number_input("Sepal Length", min_value=1.0, max_value=8.0, value=5.1)
    sp_wd  = st.number_input("Sepal Width",  min_value=1.0, max_value=5.0, value=3.5)

with colB:
    st.subheader("กรอกข้อมูลดอก:")
    pt_len = st.slider("Petal Length", 0.1, 7.0, 1.4)
    pt_wd  = st.slider("Petal Width", 0.1, 3.0, 0.2)

st.write("")

# ============================================================
# TRAIN MODEL
# ============================================================
X = df.drop("variety", axis=1)
y = df["variety"]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# ============================================================
# PREDICT BUTTON
# ============================================================
if st.button("✨ ทำนายผลดอกไม้"):
    inp = np.array([[pt_len, pt_wd, sp_len, sp_wd]])
    result = model.predict(inp)[0]

    st.success(f"🌸 ผลการทำนาย: **{result}** 🌸")

    if result == "Setosa":
        st.image("./img/iris3.jpg")
    elif result == "Versicolor":
        st.image("./img/iris1.jpg")
    else:
        st.image("./img/iris2.jpg")
else:
    st.info("กรอกข้อมูลให้ครบ จากนั้นกดปุ่มเพื่อทำนาย")

st.markdown("---")
st.markdown("<h5 style='text-align:center; color:#999;'>Developed by Fluke</h5>", unsafe_allow_html=True)
