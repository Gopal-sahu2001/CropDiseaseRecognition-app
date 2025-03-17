import gdown
import streamlit as st

url= "https://drive.google.com/file/d/1TcEQ092-zqJ9YiPMdtiICZVDdACvtGlF/view?usp=sharing"
output = "model/model_diseases.h5"
gdown.download(url, output, quiet=False)


st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
