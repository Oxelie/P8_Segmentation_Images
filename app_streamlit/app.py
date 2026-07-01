import streamlit as st
import requests
import base64

list_img = requests.get("http://127.0.0.1:4444/list_img").json()


st.title("Welcome to the Home page!")

selected = st.selectbox("Select a image", options=list_img, format_func=lambda x: x["path"])

response = requests.post("http://127.0.0.1:4444/select_img", json={"image_index": selected["index"]})

image_orig = base64.b64decode(response.json()["image"])
mask = base64.b64decode(response.json()["mask"])

col1, col2 = st.columns(2)
with col1:
    st.image(image_orig)
with col2:
    st.image(mask)
    
    
col1, col2, col3 = st.columns([2,1,2])
    
with col2:
    st.markdown("""
                    <style>
                    div.stButton > button {
                        background-color: red;
                        color: white;
                        border: none;
                    }
                    div.stButton > button:hover {
                        background-color: darkred;
                        color: white;
                    }
                    </style>
                """, unsafe_allow_html=True)
    
    gros_bouton = st.button("§ PREDICT §")


col1, col2, col3 = st.columns([1,3,1])
with col2:
    if gros_bouton:
        response = requests.post("http://127.0.0.1:4444/predict", json={"image_index": selected["index"]})
        mask_pred = base64.b64decode(response.json()["img_mask_pred"])
        st.image(mask_pred)