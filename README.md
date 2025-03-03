# RL-basierte Sentiment-Analyse für RSS-Feeds (3 Klassen)

Dieses Projekt demonstriert, wie man mittels Reinforcement Learning (REINFORCE) ein einfaches Sentiment-Analyse-Modell erstellt, das zwischen drei Klassen unterscheidet:
- **NEGATIV**
- **NEUTRAL**
- **POSITIV**

Zusätzlich kannst du über Kommandozeilenparameter das Lernen beeinflussen, z. B. Lernrate, Discount‑Faktor, Update‑Frequenz und Dropout‑Rate. Das Policy-Netzwerk verwendet einen Dropout‑Layer, um Überanpassung zu vermeiden. Die Trainingsresultate werden gespeichert und mittels eines Dashboards interaktiv visualisiert.

## Inhalt

- **train.py**: Führt das Training des RL-basierten Sentiment-Analyse-Modells durch. Es werden Dummy-Daten verwendet, und nach dem Training erfolgt eine Auswertung von RSS-Feeds.
- **visualizer.py**: Visualisiert den Trainingsverlauf (Episode Rewards, Training Loss, Evaluation Returns).
- **config.json**: Konfigurationsdatei für RSS-Feeds, den betrachteten Zeitraum, Episodenanzahl etc.
- **dashboard.py**: Ein Streamlit-Dashboard, das die Trainingsresultate und weitere Auswertungen anzeigt.
- **requirements.txt**: Liste der benötigten Pakete.

## Einrichtung

1. Repository klonen.
2. (Optional) Virtuelle Umgebung erstellen:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Auf Windows: venv\Scripts\activate
