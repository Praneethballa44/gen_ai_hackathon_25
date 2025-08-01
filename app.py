# app.py
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="StudyMate", layout="centered")
st.title("📘 StudyMate: Ask Questions from PDF")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with open("temp.pdf", "rb") as f:
        response = requests.post(f"{API_URL}/upload", files={"file": f})

    if response.status_code == 200:
        st.success("✅ PDF uploaded successfully!")
    else:
        st.error(f"Upload failed: {response.text}")

# Ask Question
st.markdown("---")
question = st.text_input("Ask a question:")
if st.button("Submit"):
    if not question:
        st.warning("Please enter a question.")
    else:
        try:
            res = requests.post(f"{API_URL}/ask_question", json={"question": question})
            if res.status_code == 200:
                st.markdown("**🧠 Answer:** " + res.json()["answer"])
            else:
                st.error("Error fetching answer.")
        except requests.exceptions.ConnectionError:
            st.error("Backend is not running.")
