import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("model.pkl", "rb"))

st.title("Prediksi Nilai Matematika Siswa")

gender = st.selectbox("Gender", ["female", "male"])
race = st.selectbox("Race", ["group A","group B","group C","group D","group E"])
parent = st.selectbox("Pendidikan Orang Tua", [
    "high school","some college","associate's degree",
    "bachelor's degree","master's degree"
])
lunch = st.selectbox("Lunch", ["standard","free/reduced"])
prep = st.selectbox("Test Preparation", ["none","completed"])
reading = st.number_input("Reading Score", 0, 100)
writing = st.number_input("Writing Score", 0, 100)

if st.button("Prediksi"):
    data = pd.DataFrame({
        'gender':[gender],
        'race_ethnicity':[race],
        'parental_level_of_education':[parent],
        'lunch':[lunch],
        'test_preparation_course':[prep],
        'reading_score':[reading],
        'writing_score':[writing]
    })

    pred = model.predict(data)
    st.success(f"Prediksi Math Score: {round(pred[0],2)}")
