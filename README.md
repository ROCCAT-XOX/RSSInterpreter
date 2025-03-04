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


```
rl-sentiment-analysis/
├── config/                    # Your existing config folder
├── data/                      # Your existing data folder
├── utils/                     # Your existing utils folder
├── api/                       # New API folder
│   └── app.py                 # Flask API service
├── frontend/                  # New React frontend
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── RSSFeedAnalyzer.jsx
│   │   │   └── TrainingMonitor.jsx
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   └── tailwind.config.js
├── train.py                   # Your existing training script
├── dashboard.py               # Your existing dashboard
├── requirements.txt           # Updated requirements file
└── README.md                  # New documentation
```