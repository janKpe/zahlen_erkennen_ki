# zahlen_erkennen_ki

Diese Ki kann Zahlen aus dem MNIST-Datensatz erkennen und klassifizieren, die Modelarchitektur und der Code können auf andere Bilder-Datensätze angewendet werden wenn die anzahl der Neuronen angepasst wird.

### Installation
1. Python und Pip 3 Installieren
2. `pip install -r ./requirements.txt` ausführen
3. Wenn eine Nvidia Grafikkarte vorhanden ist wird empfohlen entsprechende Treiber installieren da dies die Performance deutlich verbessert

Die Datei `train.py` trainiert das Netz und die Datei `test.py` öffnet zufällige Bilder aus dem MNIST-Datensatzes und klassifiziert diese
