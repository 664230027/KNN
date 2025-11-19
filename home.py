from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Fluke Iris Classifier", layout="wide")

# ---------------------- HEADER -------------------------
st.markdown("<h1 style='text-align:center; color:#2E4053;'>🌸 Iris Flower Classification by Fluke 🌸</h1>", unsafe_allow_html=True)
st.image("./img/fluke.jpg", width=350)

st.markdown("---")

# ---------------------- FLOWER IMAGES -------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<h3 style='text-align:center;'>Setosa</h3>", unsafe_allow_html=True)
    st.image("./img/iris3.jpg")

with col2:
    st.markdown("<h3 style='text-align:center;'>Versicolor</h3>", unsafe_allow_html=True)
    st.image("./img/iris1.jpg")

with col3:
    st.markdown("<h3 style='text-align:center;'>Virginica</h3>", unsafe_allow_html=True)
    st.image("./img/iris2.jpg")

st.markdown("---")

# ---------------------- DATASET -------------------------
html_7 = """
<div style="background-color:#EC7063;padding:15px;border-radius:12px; border:2px solid black;">
<center><h4 style="color:white;">📊 สถิติข้อมูลดอกไม้ (Iris Dataset)</h4></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)

dt = pd.read_csv("./data/iris.csv")
st.write(dt.head(10))

# ค่า sum
dt1 = dt['petallength'].sum()
dt2 = dt['petalwidth'].sum()
dt3 = dt['sepallength'].sum()
dt4 = dt['sepalwidth'].sum()

dx = pd.DataFrame(
    [dt1, dt2, dt3, dt4],
    index=["Petal Length", "Petal Width", "Sepal Length", "Sepal Width"],
    columns=["Sum"]
)

if st.button("แสดงแผนภูมิ Bar Chart"):
    st.bar_chart(dx)
else:
    st.info("กดปุ่มเพื่อดูการจินตทัศน์ข้อมูล")

st.markdown("---")

# ---------------------- PREDICTION -------------------------
html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:12px; border:2px solid black;">
<center><h4 style="color:white;">🔮 ทำนายข้อมูลดอกไม้ (Predict Flower)</h4></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)

colA, colB = st.columns(2)

with colA:
    pt_len = st.slider("เลือกค่า Petal Length", 0.1, 7.0, 1.4)
    pt_wd  = st.slider("เลือกค่า Petal Width", 0.1, 3.0, 0.2)

with colB:
    sp_len = st.number_input("เลือกค่า Sepal Length", min_value=1.0, max_value=10.0, value=5.1)
    sp_wd  = st.number_input("เลือกค่า Sepal Width", min_value=1.0, max_value=5.0, value=3.5)

st.markdown("")

# ---------------------- MODEL -------------------------
if st.button("ทำนายผล"):
    X = dt.drop('variety', axis=1)
    y = dt['variety']

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)

    x_input = np.array([[pt_len, pt_wd, sp_len, sp_wd]])
    out = model.predict(x_input)

    st.success(f"🌸 ผลการทำนายคือ: **{out[0]}** 🌸")

    # แสดงรูปภาพตามผลลัพธ์
    if out[0] == 'Setosa':
        st.image("./img/iris3.jpg")
    elif out[0] == 'Versicolor':
        st.image("./img/iris1.jpg")
    else:
        st.image("./img/iris2.jpg")

else:
    st.info("กดปุ่มเพื่อทำนายผล")

st.markdown("---")
st.markdown("<h5 style='text-align:center; color:gray;'>Developed by Fluke 💻</h5>", unsafe_allow_html=True)
