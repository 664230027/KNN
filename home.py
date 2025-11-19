from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np

st.title("🌸 KNN Iris Classifier")

# -------------------------------
# โหลดข้อมูล
# -------------------------------
st.subheader("ข้อมูลชุด Iris")
dt = pd.read_csv("./data/iris.csv")
st.dataframe(dt.head())

# -------------------------------
# แสดงสถิติโดยรวม (Sum)
# -------------------------------
st.subheader("สถิติค่าโดยรวมของข้อมูล")

dt_sum = dt.drop("variety", axis=1).sum()
st.bar_chart(dt_sum)

# -------------------------------
# ส่วนทำนายผล
# -------------------------------
st.subheader("🔍 ทำนายผลด้วย KNN")

col1, col2 = st.columns(2)

with col1:
    pt_len = st.slider("Petal Length", 0.1, 7.0, 1.4)
    pt_wd  = st.slider("Petal Width", 0.1, 3.0, 0.2)

with col2:
    sp_len = st.number_input("Sepal Length", 0.1, 10.0, 5.1)
    sp_wd  = st.number_input("Sepal Width", 0.1, 10.0, 3.5)

# -------------------------------
# สร้างและเทรนโมเดล KNN
# -------------------------------
X = dt.drop("variety", axis=1)
y = dt["variety"]

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# -------------------------------
# ทำนายผล
# -------------------------------
if st.button("ทำนายผล"):
    x_input = np.array([[pt_len, pt_wd, sp_len, sp_wd]])
    result = knn.predict(x_input)[0]

    st.success(f"ผลลัพธ์ที่ทำนายได้: **{result}**")

    if result == "Setosa":
        st.image("./img/iris1.jpg", width=200)
    elif result == "Versicolor":
        st.image("./img/iris2.jpg", width=200)
    else:
        st.image("./img/iris3.jpg", width=200)
