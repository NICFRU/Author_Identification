# app/Streamlit/NLP_Projekt

FROM python:3.7

WORKDIR /app

RUN git clone https://github.com/Coreprog/Front-IntSem.git .

RUN pip3 install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["app.py"] 