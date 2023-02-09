import streamlit as st
from time import sleep


from select_page import select_page
from predict_page import show_predict_page

st.set_page_config(
    page_title = "IntSem-Frontend",
    layout="wide"
    )

def draw_all(key,plot=False):
    st.write("""
    # Integrationsseminar Frontend

    ## Folgende Pages können gefunden werden
       NLP Modelle

    """)

with st.sidebar:
    draw_all("sidebar")

def main():
    st.title("Integrationsseminar Frontend")
    st.write("---")
    menu = ["--select--",  "NLP Modelle"]
    page = st.sidebar.selectbox("Choose your page:", menu)

    if page =="--select--":
        select_page()
    
    elif page == "NLP Modelle":
        show_predict_page()



if __name__ == "__main__":
    main()