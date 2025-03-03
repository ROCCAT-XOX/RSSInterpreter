import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import os
import sys
import traceback
import torch


# Füge den Pfad zum config-Ordner hinzu
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config"))

try:
    from config import config
    from train import run_training, predict_sentiment, build_vocab, text_to_indices, SentimentPolicy
except ImportError as e:
    st.error(f"Fehler beim Importieren der Module: {e}")
    st.error(traceback.format_exc())
    config = None
    run_training = None
    predict_sentiment = None


def main():
    st.title("🤖 Sentiment Analysis Dashboard")
    st.sidebar.header("Training Konfiguration")

    # Parameter-Eingaben in der Seitenleiste
    episodes = st.sidebar.number_input(
        "Anzahl der Trainingsepisoden", min_value=1, max_value=1000,
        value=config.get("episodes", 50), step=1)
    lr = st.sidebar.slider("Lernrate", 0.0001, 0.1, value=0.01, format="%.4f")
    gamma = st.sidebar.slider("Discount Faktor", 0.9, 0.999, value=0.99, format="%.3f")
    dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 1.0, value=0.5, step=0.05)
    update_freq = st.sidebar.number_input(
        "Update Frequency (Samples pro Update)", min_value=1, value=2, step=1)

    # Lookback-Dauer (wie weit in die Vergangenheit geschaut wird)
    lookback_days = st.sidebar.number_input(
        "Lookback (Tage)", min_value=1, max_value=365,
        value=config.get("lookback_days", 30), step=1)

    # Ergebnisse werden im in der Config definierten Ordner gespeichert
    results_dir = config.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, config.get("visualization_save_path", "training_results.png"))

    st.sidebar.subheader("Aktionen")
    train_button = st.sidebar.button("Training starten")
    predict_button = st.sidebar.button("Vorhersage durchführen")

    # Falls Trainingsresultate vorhanden sind, zeige sie an
    if os.path.exists(results_path):
        st.image(results_path, caption="Trainingsergebnisse", use_container_width=True)
    else:
        st.info("Keine Trainingsergebnisse gefunden. Bitte führe zuerst das Training durch.")

    if train_button:
        st.info("Training wird gestartet. Dies kann einige Zeit in Anspruch nehmen...")
        try:
            # Starte das Training mit den angegebenen Parametern
            history, model_save_path = run_training(episodes, lr, gamma, dropout_rate, update_freq)
            st.success("Training abgeschlossen!")
            if os.path.exists(results_path):
                st.image(results_path, caption="Trainingsergebnisse", use_container_width=True)
            st.write(f"Modell wurde unter {model_save_path} gespeichert.")
        except Exception as e:
            st.error(f"Fehler beim Training: {e}")
            st.error(traceback.format_exc())

    if predict_button:
        st.subheader("Vorhersage")
        sample_text = st.text_area("Beispieltext eingeben", "Heute ist ein durchschnittlicher Tag, nichts Besonderes.")
        try:
            # Baue das Vokabular aus dem Dummy-Datensatz (als Beispiel)
            dummy_data = [
                ("Das ist ein toller Tag und die Nachrichten sind super.", 2),
                ("Heute läuft alles großartig und die Stimmung ist positiv.", 2),
                ("Die Wirtschaft ist im Aufschwung und alles ist bestens.", 2),
                ("Ich freue mich auf die guten Nachrichten.", 2),
                ("Heute ist ein durchschnittlicher Tag.", 1),
                ("Es ist ein normaler Tag, nichts Besonderes.", 1),
                ("Das ist ein schrecklicher Tag, alles läuft schief.", 0),
                ("Die Nachrichten sind deprimierend und traurig.", 0),
                ("Ich bin enttäuscht von den aktuellen Entwicklungen.", 0),
                ("Es gibt viele negative Schlagzeilen heute.", 0)
            ]
            train_texts = [text for text, _ in dummy_data]
            vocab = build_vocab(train_texts, min_freq=1)
            vocab_size = len(vocab)
            # Erstelle eine neue Instanz des Modells mit den gleichen Parametern
            embedding_dim = 50
            hidden_dim = 64
            output_dim = 3
            model = SentimentPolicy(vocab_size, embedding_dim, hidden_dim, output_dim,
                                    dropout_rate=dropout_rate, pad_idx=vocab['<PAD>'])
            model_path = os.path.join("saved_models", "sentiment_model.pth")
            if not os.path.exists(model_path):
                st.error("Kein trainiertes Modell gefunden. Bitte führe zuerst das Training durch.")
                return
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            pred = predict_sentiment(model, sample_text, vocab)
            sentiment_map = {0: "NEGATIV", 1: "NEUTRAL", 2: "POSITIV"}
            st.write(f"Vorhersage: {sentiment_map[pred]}")
        except Exception as e:
            st.error(f"Fehler bei der Vorhersage: {e}")
            st.error(traceback.format_exc())

    st.subheader("Auswertung")
    st.write("Weitere Metriken und Evaluationsergebnisse können hier angezeigt werden.")


if __name__ == "__main__":
    main()
