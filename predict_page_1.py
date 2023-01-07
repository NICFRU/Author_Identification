import streamlit as st
import pandas as pd
from age_classifier import classifierNLP

def show_predict_page_1():
    st.subheader(
    """
    Das ist die topic prediction.
    """
    )

    st.write(
    """
    Gib deinen Text hier ein:
    """
    )

    txt = st.text_area("Your text")

    if txt !="":
        pred = classifierNLP(txt)
        st.write(pred)
    