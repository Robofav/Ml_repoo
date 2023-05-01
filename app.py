import streamlit as st
import pickle

with open("model.obj", 'rb') as f:
    model = pickle.load(f)
    
with open("scaler.obj", 'rb') as f:
    scaler = pickle.load(f)
    
st.title("Customer Segmentation Prediction Model")

fields = {
    "mean": st.number_input("Mean"),
    "categ_0": st.number_input("Category 0"),
    "categ_1": st.number_input("Category 1"),
    "categ_2": st.number_input("Category 2"),
    "categ_3": st.number_input("Category 3"),
    "categ_4": st.number_input("Category 4"),
}

if st.button("Predict"):
    data = list(fields.values())
    data = scaler.transform([data])
    st.write(model.predict(data)[0])  
