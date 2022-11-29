import streamlit as st
import pandas as pd


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
        st.write(txt)