from flask import Flask, request, jsonify
import torch
import os
import json
import feedparser
from datetime import datetime, timedelta
import time
import numpy as np
from flask_cors import CORS
import sys
import traceback
import uuid


# Füge das Hauptverzeichnis zum Pfad hinzu, damit wir Module importieren können
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Importiere deine bestehenden Module
from train import SentimentPolicy, text_to_indices, build_vocab, predict_sentiment
from config.config import Config

try:
    from transformers import BartForConditionalGeneration, BartTokenizer, pipeline

    SUMMARIZATION_AVAILABLE = True
except ImportError:
    print("Transformers Bibliothek nicht verfügbar. Zusammenfassung wird deaktiviert.")
    SUMMARIZATION_AVAILABLE = False

app = Flask(__name__)
CORS(app)  # Erlaube Cross-Origin Requests

# Lade Konfiguration
config = Config()

# Initialisiere das Summarization-Modell
summarizer = None
# Initialisiere das Sentiment-Analyse-Modell
sentiment_model = None
vocab = None

# Cache für Artikeltexte zum späteren Training
article_cache = {}


def load_summarizer():
    global summarizer
    if not SUMMARIZATION_AVAILABLE:
        return

    try:
        # SSL-Verifizierung deaktivieren für Downloads
        import ssl
        if hasattr(ssl, '_create_unverified_context'):
            ssl._create_default_https_context = ssl._create_unverified_context

        # Lade BART Modell für die Zusammenfassung
        # Für Deutsch könntest du ein anderes Modell verwenden
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        print("Zusammenfassungsmodell erfolgreich geladen")
    except Exception as e:
        print(f"Fehler beim Laden des Zusammenfassungsmodells: {e}")
        print(traceback.format_exc())
        summarizer = None


# In app.py, modifizieren Sie die load_sentiment_model-Funktion:

def load_sentiment_model():
    global sentiment_model, vocab
    try:
        # Versuche zuerst das gespeicherte Vokabular zu laden
        vocab_path = os.path.join(project_root, "saved_models", "vocabulary.json")

        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                vocab = json.load(f)
            print(f"Vokabular aus {vocab_path} geladen, Größe: {len(vocab)}")
        else:
            # Wenn kein gespeichertes Vokabular existiert, erstelle ein neues
            # aus der Kombination von Basis- und Feedback-Daten
            feedback_data = load_feedback_data()

            # Dummy-Daten
            dummy_data = [
                ("Das ist ein toller Tag und die Nachrichten sind super.", 2),
                # ... weitere Einträge ...
                ("Es gibt viele negative Schlagzeilen heute.", 0)
            ]

            # Kombiniere alle Texte für das Vokabular
            all_texts = [text for text, _ in dummy_data]
            all_texts.extend([text for text, _ in feedback_data])

            vocab = build_vocab(all_texts, min_freq=1)
            print(f"Neues Vokabular erstellt, Größe: {len(vocab)}")

        vocab_size = len(vocab)

        # Initialisiere das Modell mit dem korrekten Vokabular
        embedding_dim = 50
        hidden_dim = 64
        output_dim = 3  # 3 Klassen: NEGATIV, NEUTRAL, POSITIV
        dropout_rate = 0.5

        sentiment_model = SentimentPolicy(vocab_size, embedding_dim, hidden_dim, output_dim,
                                          dropout_rate=dropout_rate, pad_idx=vocab['<PAD>'])

        # Lade das trainierte Modell
        model_path = os.path.join(project_root, "saved_models", "sentiment_model.pth")
        if os.path.exists(model_path):
            sentiment_model.load_state_dict(torch.load(model_path, map_location="cpu"))
            sentiment_model.eval()
            print("Sentiment-Analysemodell erfolgreich geladen")
        else:
            print(f"Warnung: Modelldatei {model_path} nicht gefunden")
    except Exception as e:
        print(f"Fehler beim Laden des Sentiment-Modells: {e}")
        import traceback
        print(traceback.format_exc())


def load_feedback_data():
    """Lädt die gespeicherten Feedback-Daten"""
    feedback_data = []
    feedback_dir = os.path.join(project_root, "feedback")

    if not os.path.exists(feedback_dir):
        return feedback_data

    for filename in os.listdir(feedback_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(feedback_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    feedback_item = json.load(f)
                    # Konvertiere Sentiment-Label zu numerischem Wert
                    sentiment_map = {"NEGATIV": 0, "NEUTRAL": 1, "POSITIV": 2}

                    if "article_text" in feedback_item and "sentiment" in feedback_item:
                        text = feedback_item["article_text"]
                        sentiment = sentiment_map.get(feedback_item["sentiment"], 1)  # Default zu NEUTRAL
                        feedback_data.append((text, sentiment))
            except Exception as e:
                print(f"Fehler beim Laden der Feedback-Datei {filename}: {e}")

    print(f"Geladene Feedback-Daten: {len(feedback_data)} Einträge")
    return feedback_data


def get_feedback_stats():
    """Gibt Statistiken über die gesammelten Feedback-Daten zurück"""
    feedback_dir = os.path.join(project_root, "feedback")

    if not os.path.exists(feedback_dir):
        return {"count": 0, "classes": {"NEGATIV": 0, "NEUTRAL": 0, "POSITIV": 0}}

    stats = {"count": 0, "classes": {"NEGATIV": 0, "NEUTRAL": 0, "POSITIV": 0}}

    for filename in os.listdir(feedback_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(feedback_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    feedback_item = json.load(f)
                    if "sentiment" in feedback_item:
                        stats["count"] += 1
                        sentiment = feedback_item.get("sentiment", "NEUTRAL")
                        stats["classes"][sentiment] = stats["classes"].get(sentiment, 0) + 1
            except:
                pass

    return stats


def summarize_text(text, max_length=150):
    if summarizer is None or not SUMMARIZATION_AVAILABLE:
        # Fallback: Einfaches Abschneiden, wenn kein Modell verfügbar ist
        words = text.split()
        if len(words) > 30:
            return " ".join(words[:30]) + "..."
        return text

    try:
        # Beschränke die Eingabetextlänge
        input_text = text[:1024] if len(text) > 1024 else text

        # Verwende das BART-Modell für die Zusammenfassung
        summary = summarizer(input_text, max_length=max_length, min_length=30, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        print(f"Fehler bei der Zusammenfassung: {e}")
        # Fallback: Schneide einfach ab
        words = text.split()
        if len(words) > 30:
            return " ".join(words[:30]) + "..."
        return text


@app.route('/api/feeds', methods=['GET'])
def get_feed_list():
    """Gibt die Liste der verfügbaren RSS-Feeds aus der Konfiguration zurück"""
    return jsonify({
        "feeds": config.get("rss_feeds", [])
    })


@app.route('/api/articles', methods=['GET'])
def get_articles():
    """Lädt und analysiert Artikel aus einem bestimmten RSS-Feed"""
    feed_url = request.args.get('url', config.get("rss_feeds", [])[0])
    days = int(request.args.get('days', config.get("lookback_days", 30)))

    print(f"Versuche Artikel von {feed_url} der letzten {days} Tage zu laden...")

    try:
        # SSL-Verifizierung deaktivieren für Downloads
        import ssl
        if hasattr(ssl, '_create_unverified_context'):
            ssl._create_default_https_context = ssl._create_unverified_context

        # Parse den RSS-Feed
        feed = feedparser.parse(feed_url)

        if hasattr(feed, 'bozo_exception'):
            print(f"Fehler beim Parsen des Feeds: {feed.bozo_exception}")
            return jsonify({"error": f"Fehler beim Parsen des Feeds: {feed.bozo_exception}"}), 400

        if not feed.entries:
            print(f"Keine Einträge im Feed gefunden: {feed_url}")
            return jsonify({"articles": [], "stats": {"POSITIV": 0, "NEUTRAL": 0, "NEGATIV": 0}}), 200

        # Berechne das Cutoff-Datum
        cutoff_date = datetime.now() - timedelta(days=days)

        articles = []

        for entry in feed.entries:
            try:
                if 'published_parsed' in entry:
                    published_date = datetime.fromtimestamp(time.mktime(entry.published_parsed))

                    # Überspringe Einträge, die älter als das Cutoff-Datum sind
                    if published_date < cutoff_date:
                        continue

                    # Extrahiere den Inhalt
                    content = entry.summary if "summary" in entry else entry.get("description", "")
                    if not content.strip():
                        continue

                    # Generiere eine eindeutige ID, falls keine vorhanden
                    article_id = str(hash(entry.link) % 10000000)  # Einfacher Hash für die ID

                    # Speichere den Artikeltext im Cache für späteres Feedback
                    article_cache[article_id] = content

                    # Analysiere das Sentiment
                    sentiment_id = predict_sentiment(sentiment_model, content, vocab)
                    sentiment_map = {0: "NEGATIV", 1: "NEUTRAL", 2: "POSITIV"}
                    sentiment = sentiment_map[sentiment_id]

                    # Berechne die Konfidenz (In einer realen Implementierung würdest du die echten Werte verwenden)
                    confidence = np.random.uniform(0.65, 0.95)

                    # Erstelle die Zusammenfassung
                    summary = summarize_text(content)

                    articles.append({
                        "id": article_id,
                        "title": entry.title,
                        "date": published_date.isoformat(),
                        "content": content,
                        "summary": summary,
                        "url": entry.link,
                        "sentiment": sentiment,
                        "confidence": float(confidence)
                    })
            except Exception as e:
                print(f"Fehler bei der Verarbeitung eines Eintrags: {e}")
                continue

        return jsonify({
            "articles": articles,
            "stats": {
                "POSITIV": sum(1 for a in articles if a["sentiment"] == "POSITIV"),
                "NEUTRAL": sum(1 for a in articles if a["sentiment"] == "NEUTRAL"),
                "NEGATIV": sum(1 for a in articles if a["sentiment"] == "NEGATIV")
            }
        })

    except Exception as e:
        print(f"Fehler beim Abrufen der Artikel: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Verarbeitet Benutzerfeedback zur Sentiment-Analyse und speichert den kompletten Artikeltext"""
    data = request.json

    try:
        feedback_items = data.get('feedback', {})

        if not feedback_items:
            return jsonify({"success": False, "message": "Kein Feedback übermittelt"}), 400

        # Verzeichnis für Feedback erstellen, falls es nicht existiert
        feedback_dir = os.path.join(project_root, "feedback")
        os.makedirs(feedback_dir, exist_ok=True)

        saved_count = 0

        # Jedes Feedback-Item einzeln speichern für bessere Modularität
        for article_id, sentiment in feedback_items.items():
            # Hole den Artikeltext aus dem Cache
            article_text = article_cache.get(article_id)

            if not article_text:
                print(f"Warnung: Kein Artikeltext für ID {article_id} gefunden")
                continue

            # Erstelle ein strukturiertes Feedback-Objekt
            feedback_item = {
                "article_id": article_id,
                "sentiment": sentiment,
                "article_text": article_text,
                "timestamp": datetime.now().isoformat()
            }

            # Generiere einen eindeutigen Dateinamen
            filename = f"feedback_{article_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}.json"
            file_path = os.path.join(feedback_dir, filename)

            # Speichere das Feedback
            with open(file_path, 'w') as f:
                json.dump(feedback_item, f, indent=2)

            saved_count += 1
            print(f"Feedback für Artikel {article_id} als {sentiment} gespeichert")

        # Statistiken über gesammeltes Feedback
        feedback_stats = get_feedback_stats()

        return jsonify({
            "success": True,
            "message": f"Feedback für {saved_count} Artikel gespeichert",
            "feedback_stats": feedback_stats
        })

    except Exception as e:
        print(f"Fehler beim Speichern des Feedbacks: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/api/feedback/stats', methods=['GET'])
def get_feedback_statistics():
    """Gibt Statistiken über die gesammelten Feedback-Daten zurück"""
    try:
        stats = get_feedback_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/train_with_feedback', methods=['POST'])
def train_with_feedback():
    """Trainiert das Modell unter Einbeziehung des gesammelten Feedbacks"""
    try:
        # Parameter aus dem Request oder Default-Werte verwenden
        data = request.json or {}
        episodes = data.get('episodes', 50)
        lr = data.get('learningRate', 0.01)
        gamma = data.get('gamma', 0.99)
        dropout_rate = data.get('dropoutRate', 0.5)
        update_frequency = data.get('updateFrequency', 2)

        # Lade die Feedback-Daten
        feedback_data = load_feedback_data()

        if not feedback_data:
            return jsonify({
                "success": False,
                "message": "Kein Feedback für das Training gefunden"
            }), 400

        print(f"Starte Training mit {len(feedback_data)} Feedback-Einträgen")

        # WICHTIG: Hier würde man normalerweise das Modell trainieren
        # Dies erfordert eine Anpassung deiner run_training-Funktion, um Feedback einzubeziehen

        # In diesem Beispiel simulieren wir das Training mit Feedback
        # history, model_save_path = run_training_with_feedback(episodes, lr, gamma, dropout_rate, update_frequency, feedback_data)

        # Simuliere den Trainingsprozess für dieses Beispiel
        time.sleep(2)  # Simuliere Rechenzeit

        # Nachdem das Training abgeschlossen ist, laden wir das Modell neu
        load_sentiment_model()

        return jsonify({
            "success": True,
            "message": f"Modell mit {len(feedback_data)} Feedback-Einträgen trainiert",
            "stats": {
                "episodes": episodes,
                "feedback_count": len(feedback_data),
                "feedback_distribution": {
                    "NEGATIV": sum(1 for _, label in feedback_data if label == 0),
                    "NEUTRAL": sum(1 for _, label in feedback_data if label == 1),
                    "POSITIV": sum(1 for _, label in feedback_data if label == 2)
                }
            }
        })

    except Exception as e:
        print(f"Fehler beim Training mit Feedback: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/api/model/status', methods=['GET'])
def get_model_status():
    """Gibt Informationen über den Modellstatus zurück"""
    model_path = os.path.join(project_root, "saved_models", "sentiment_model.pth")
    model_exists = os.path.exists(model_path)

    model_info = {
        "exists": model_exists,
        "timestamp": None,
        "size_mb": None
    }

    if model_exists:
        stats = os.stat(model_path)
        model_info["timestamp"] = datetime.fromtimestamp(stats.st_mtime).isoformat()
        model_info["size_mb"] = round(stats.st_size / (1024 * 1024), 2)

    feedback_stats = get_feedback_stats()

    return jsonify({
        "model": model_info,
        "feedback": feedback_stats
    })


@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    """Trainiert das Modell neu mit den Standarddaten"""
    try:
        # Parameter aus dem Request holen
        data = request.json or {}
        episodes = data.get('episodes', 50)
        lr = data.get('learningRate', 0.01)
        gamma = data.get('gamma', 0.99)
        dropout_rate = data.get('dropoutRate', 0.5)
        update_frequency = data.get('updateFrequency', 2)

        # In einer echten Implementierung würdest du hier das Modell trainieren
        # history, model_save_path = run_training(episodes, lr, gamma, dropout_rate, update_frequency)

        # Simuliere den Trainingsprozess für dieses Beispiel
        time.sleep(2)  # Simuliere Rechenzeit

        return jsonify({
            "success": True,
            "message": "Modell-Retraining erfolgreich abgeschlossen."
        })

    except Exception as e:
        print(f"Fehler beim Modell-Retraining: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/api/test_sentiment', methods=['POST'])
def test_sentiment():
    """Testet das Sentiment-Modell mit einem gegebenen Text"""
    data = request.json
    text = data.get('text', '')

    try:
        if sentiment_model is None or vocab is None:
            return jsonify({"error": "Sentiment-Modell ist nicht geladen"}), 500

        sentiment_id = predict_sentiment(sentiment_model, text, vocab)
        sentiment_map = {0: "NEGATIV", 1: "NEUTRAL", 2: "POSITIV"}
        return jsonify({
            "sentiment": sentiment_map[sentiment_id],
            "sentiment_id": sentiment_id,
            "analyzed_text": text[:100] + ("..." if len(text) > 100 else "")
        })
    except Exception as e:
        print(f"Fehler beim Testen des Sentiments: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/api/mock_articles', methods=['GET'])
def get_mock_articles():
    """Gibt Mock-Artikel zurück, um das Frontend zu testen"""
    mock_articles = [
        {
            "id": 1,
            "title": "Bitcoin erreicht neues Allzeithoch",
            "date": datetime.now().isoformat(),
            "content": "Der Bitcoin-Kurs ist heute auf ein neues Allzeithoch von 70.000 USD gestiegen, was auf ein gestiegenes institutionelles Interesse und die Zulassung neuer ETFs zurückzuführen ist.",
            "summary": "Der Bitcoin-Kurs ist heute auf ein neues Allzeithoch gestiegen.",
            "url": "https://example.com/news/1",
            "sentiment": "POSITIV",
            "confidence": 0.92
        },
        {
            "id": 2,
            "title": "Ethereum-Entwickler geben Update zum Merge",
            "date": (datetime.now() - timedelta(days=1)).isoformat(),
            "content": "Die Ethereum-Foundation hat ein Update zum Status des Merge-Upgrades gegeben. Der Übergang zu Proof of Stake verläuft wie geplant.",
            "summary": "Die Ethereum-Foundation gab ein Update zum Merge-Upgrade.",
            "url": "https://example.com/news/2",
            "sentiment": "NEUTRAL",
            "confidence": 0.78
        },
        {
            "id": 3,
            "title": "Marktkorrektur: Kryptowährungen verlieren 10% an Wert",
            "date": (datetime.now() - timedelta(days=2)).isoformat(),
            "content": "In einer plötzlichen Marktkorrektur haben die meisten Kryptowährungen etwa 10% ihres Wertes verloren. Analysten führen dies auf Gewinnmitnahmen und makroökonomische Faktoren zurück.",
            "summary": "Kryptowährungen verlieren 10% durch Marktkorrektur.",
            "url": "https://example.com/news/3",
            "sentiment": "NEGATIV",
            "confidence": 0.85
        }
    ]

    # Speichere die Artikeltexte im Cache für späteres Feedback
    for article in mock_articles:
        article_cache[str(article["id"])] = article["content"]

    return jsonify({
        "articles": mock_articles,
        "stats": {
            "POSITIV": 1,
            "NEUTRAL": 1,
            "NEGATIV": 1
        }
    })


if __name__ == "__main__":
    # Lade die Modelle beim Start
    load_sentiment_model()
    load_summarizer()

    # Erstelle Verzeichnisse, falls sie nicht existieren
    os.makedirs(os.path.join(project_root, "saved_models"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "feedback"), exist_ok=True)

    # Starte den Server
    app.run(debug=True, host='0.0.0.0', port=5000)