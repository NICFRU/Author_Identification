import streamlit as st
import pandas as pd
from classification import text_splitter,pred_gender,pred_stern, age_classifier, multi_line_hatespeechNLP, multi_line_sentimentNLP,single_line_hatespeechNLP,single_line_sentimentNLP
from streamlit_toggle import st_toggle_switch
import streamlit as st


def show_predict_page():

    st.subheader(
    """
    NLP Modelle
    """
    )

    st.write(
    """
    Gib hier deinen Text ein:
    """
    )
    txt = st.text_area("Your text:",key= "NLP")

    colum1, colum2,colum3= st.columns(3)

    with colum1:
        age = st.checkbox('Alters Vorhersage')
        if age == True:
            task = st.radio(
                            "Welche Methode soll zur Altervorhersage gewählt werden:",
                            ('Regression', 'Klassifizierung')
                            )
            
            if task == 'Klassifizierung':
                pipe = st.radio(
                            "Welche Pipeline soll für die Klassifizierung genutzt werden:",
                            ('Huggingface', 'Sklearn')
                            )
        

    with colum2:
        gender = st.checkbox('Gender Vorhersage')
        sternzeichen = st.checkbox('Sternzeichen Vorhersage')


    with colum3:
        multiline = None
        hatespeechbox = st.checkbox('Hate Speech Erkennung')
        sentimentbox = st.checkbox('Sentiment Analyse')

        if hatespeechbox == True or sentimentbox == True:
                multiline = st.radio(
                            "Soll der Text als gesamtes oder Satzweise analysiert werden:",
                            ('Multiline', 'Singleline')
                                    )





    st.write("---")
    
    if txt !="":
        if age == True:
            if task == 'Regression':
                st.subheader(
                """
                Age Klassifzierung mit der Sklearn Pipeline und Regression:
                """
                )

                with st.spinner('Age Klassfizierung wird durchgeführt...'):
                    age = age_classifier(txt,task)

                st.write(f"Der Autor dieses Textes ist {age} Jahre alt.")


            elif task == 'Klassifizierung':


                if  pipe == 'Sklearn':
                    st.subheader(
                    """
                    Age Klassifzierung mit der Sklearn Pipeline und Klassifikation:
                    """
                    )

                    with st.spinner('Age Klassfizierung wird durchgeführt...'):
                        age = age_classifier(txt,pipe)

                    st.write(f"Der Autor dieses Textes ist {age} Jahre alt.")

                elif pipe == 'Huggingface':
                    st.subheader(
                    """
                    Age Klassifzierung mit der Sklearn Pipeline und Klassifikation:
                    """
                    )

                    with st.spinner('Age Klassfizierung wird durchgeführt...'):
                        age = age_classifier(txt,pipe)

                    st.write(f"Der Autor dieses Textes ist {age} Jahre alt.")


                else:
                    pass


        if gender == True:
            st.subheader(
            """
            Gender Klassifzierung mit der Sklearn Pipeline und Klassifikation:
            """
            )
            with st.spinner('Gender Klassfizierung wird durchgeführt...'):
                gen = pred_gender(txt)
            st.write(f"Der Autor dieses Textes hat das Geschlecht {gen}.")


        if sternzeichen == True:
            st.subheader(
            """
            Sternzeicheen Klassifzierung mit der Sklearn Pipeline und Klassifikation:
            """
            )
            with st.spinner('Sternzeichen Klassfizierung wird durchgeführt...'):
                sign = pred_stern(txt)
            st.write(f"Der Autor dieses Textes hat das Sternzeichen {sign}.")

        if multiline == 'Singleline':
            if hatespeechbox == True:
                st.subheader(
                """
                Hate Speech Erkennung mit dem Hate-speech-CNERG/dehatebert-mono-german Modell:
                """
                )

                with st.spinner('Hate Speech Erkennung wird durchgeführt...'):
                    prob, pred, fig = single_line_hatespeechNLP(txt)

                st.write(f"Mit einer Wahrscheinlichkeit von {prob}% sagt das Modell {pred} vorraus.")
                st.plotly_chart(fig)

            if sentimentbox == True:
                st.subheader(
                """
                Sentiment Analyse mit dem cardiffnlp/twitter-xlm-roberta-base-sentiment Modell:
                """
                )

                with st.spinner('Sentiment Analyse wird durchgeführt...'):
                        prob, pred,fig = single_line_sentimentNLP(txt)


                st.write(f"Mit einer Wahrscheinlichkeit von {prob}% sagt das Modell vorraus, dass dieser Text {pred} ist.")
                st.plotly_chart(fig)

        elif multiline == 'Multiline':
            sentlist = text_splitter(txt)

            if hatespeechbox == True:
                st.subheader(
                """
                Multi Text Hate Speech Erkennung mit dem Hate-speech-CNERG/dehatebert-mono-german Modell:
                """
                )
                with st.spinner('Hate Speech Erkennung wird durchgeführt...'):
                    hfig = multi_line_hatespeechNLP(sentlist)
                
                st.plotly_chart(hfig)

            if sentimentbox == True:
                st.subheader(
                """
                Multi Text Sentiment Analyse mit dem cardiffnlp/twitter-xlm-roberta-base-sentiment Modell:
                """
                )
                with st.spinner('Sentiment Analyse wird durchgeführt...'):
                    sfig = multi_line_sentimentNLP(sentlist)
                
                st.plotly_chart(sfig)
        else:
            pass
            