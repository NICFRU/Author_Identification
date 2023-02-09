# Frontend fürs NLP-Projekt welches via Streamlit betrieben wird

Ordner mit Dockerfile downloaden oder einfach nur das Dockerfile downloaden. **Bitte sicherstellen das Docker auf dem Endgerät installiert ist.** 

Mit dem folgenden Befehl wird dann das Docker Image erstellt:
``` Docker
docker build -t frontendnlp .
```

Wenn das Image erstellt ist, wird das Image mit folgendem Befehl ausgeführt:
``` Docker
docker run -p 8501:8501 frontendnlp
```

Nachdem das Image nun läuft, kann im Browser folgender localhost aufgerufen werden:
```
http://localhost:8501
```


## Zu beachten:

Beim erstmaligen Aufruf des Frontend und erstmaligen Ausführung der Modelle, müssen diese erst vom Huggigface Hub runtergeladen werden. Dies könnte je nach Internet Verbindung seine Zeit brauchen. Sobald diese jedoch einmal runter geladen sind, können diese ohne Probleme genutzt werden.
