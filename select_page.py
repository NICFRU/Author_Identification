import streamlit as st

def select_page():
    st.markdown(
        """ 
        In diesem Frontend sind die Modelle unseres Projekts zu finden. 
        Man kann Modelle zu diesen Themen finden: 

        - gender prediction
        - star sign prediction
        - age prediction
            - Regression (Sklearn)
            - Klassifikation (Huggingface sowie Sklearn)
        - Hate/ Sentiment Analyse
        """)
    st.write("Alle Modelle wurden auf Englisch trainiert und verstehen somit auch nur englischen Input!")
    st.markdown("""
        ---
        """
        )
    
    st.subheader("Meme of the project:")
    st.image("brace-yourself-nlp.jpg")
    st.caption("https://makeameme.org/meme/brace-yourself-nlp")