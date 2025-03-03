import os
import json
import random
import time
import requests
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from utils.visualizer import visualize_training


# --- Textvorverarbeitung ---

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())


def build_vocab(texts, min_freq=1):
    freq = {}
    for text in texts:
        tokens = tokenize(text)
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1
    vocab = {'<PAD>': 0, '<UNK>': 1}
    index = 2
    for token, count in freq.items():
        if count >= min_freq:
            vocab[token] = index
            index += 1
    return vocab


def text_to_indices(text, vocab):
    tokens = tokenize(text)
    return [vocab.get(token, vocab['<UNK>']) for token in tokens]


# --- Dataset und DataLoader ---

class SentimentDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data  # List of (Text, Label)
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        indices = text_to_indices(text, self.vocab)
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in texts])
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return padded_texts, lengths, labels


# --- Policy-Netzwerk mit Dropout (3 Klassen) ---

class SentimentPolicy(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout_rate=0.5, pad_idx=0):
        super(SentimentPolicy, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, lengths):
        embedded = self.embedding(text)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        hidden = self.dropout(hidden[-1])
        logits = self.fc(hidden)
        return logits


# --- RL-Agent (REINFORCE) ---

class SentimentAgent:
    def __init__(self, policy, optimizer, gamma=0.99):
        self.policy = policy
        self.optimizer = optimizer
        self.gamma = gamma
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, text_tensor, length):
        logits = self.policy(text_tensor, length)
        probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.saved_log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(float(reward))

    def update(self):
        R = 0
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        loss = -torch.stack([lp * R for lp, R in zip(self.saved_log_probs, returns)]).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.saved_log_probs = []
        self.rewards = []
        return loss.item()


# --- Vorhersage (deterministisch) ---

def predict_sentiment(policy, text, vocab):
    policy.eval()
    indices = text_to_indices(text, vocab)
    tensor_indices = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    length = torch.tensor([len(indices)])
    with torch.no_grad():
        outputs = policy(tensor_indices, length)
        prediction = torch.argmax(outputs, dim=1).item()
    return prediction


# --- Kommandozeilenargumente ---

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="RL Sentiment Analysis Training")
    parser.add_argument('--episodes', type=int, default=50, help='Anzahl der Trainingsepisoden')
    parser.add_argument('--lr', type=float, default=0.01, help='Lernrate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount-Faktor')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout-Rate im Netzwerk')
    parser.add_argument('--update_frequency', type=int, default=2, help='Anzahl Samples pro Update')
    parser.add_argument('--save_path', type=str, default='saved_models', help='Pfad zum Speichern des Modells')
    return parser.parse_args()


# --- Hauptfunktion ---

def main():
    args = parse_args()
    # Achte hier auf den korrekten Pfad zur config-Datei
    with open("config/config.json", "r") as f:
        config_data = json.load(f)
    feed_urls = config_data.get("rss_feeds", [])
    days = config_data.get("days", 30)
    episodes = args.episodes
    update_frequency = args.update_frequency

    # Dummy-Daten: 0 = NEGATIV, 1 = NEUTRAL, 2 = POSITIV
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

    # Vokabular aufbauen
    train_texts = [text for text, _ in dummy_data]
    vocab = build_vocab(train_texts, min_freq=1)
    vocab_size = len(vocab)
    print("Vocabulary size:", vocab_size)

    dataset = SentimentDataset(dummy_data, vocab)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # Netzwerkparameter
    embedding_dim = 50
    hidden_dim = 64
    output_dim = 3
    policy = SentimentPolicy(vocab_size, embedding_dim, hidden_dim, output_dim,
                             dropout_rate=args.dropout_rate, pad_idx=vocab['<PAD>'])
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    agent = SentimentAgent(policy, optimizer, gamma=args.gamma)

    history = {"episode_rewards": [], "losses": [], "eval_returns": []}

    for episode in range(episodes):
        episode_reward = 0
        sample_counter = 0
        random.shuffle(dummy_data)
        for text, label in dummy_data:
            indices = text_to_indices(text, vocab)
            text_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
            length = torch.tensor([len(indices)])
            action = agent.select_action(text_tensor, length)
            reward = 1.0 if action == label else -1.0
            agent.store_reward(reward)
            episode_reward += reward
            sample_counter += 1
            if sample_counter % update_frequency == 0:
                loss_val = agent.update()
                history["losses"].append(loss_val)
        if agent.rewards:
            loss_val = agent.update()
            history["losses"].append(loss_val)
        history["episode_rewards"].append(episode_reward)
        print(f"Episode {episode + 1}/{episodes}: Reward = {episode_reward}")
        if (episode + 1) % 10 == 0:
            eval_reward = 0
            for text, label in dummy_data:
                pred = predict_sentiment(policy, text, vocab)
                eval_reward += (1.0 if pred == label else -1.0)
            avg_eval = eval_reward / len(dummy_data)
            history["eval_returns"].append(avg_eval)
            print(f"Evaluation nach Episode {episode + 1}: Durchschnittlicher Reward = {avg_eval:.2f}")

    # RSS-Feed-Auswertung
    if feed_urls:
        print(f"\nRSS Feed Ergebnisse (Artikel der letzten {days} Tage):")
        sentiment_map = {0: "NEGATIV", 1: "NEUTRAL", 2: "POSITIV"}
        sentiment_counts = {"NEGATIV": 0, "NEUTRAL": 0, "POSITIV": 0}
        headers = {'User-Agent': 'Mozilla/5.0'}
        all_entries = []
        for rss_url in feed_urls:
            response = requests.get(rss_url, headers=headers)
            # Nutze den Parser aus data/rss_parser.py – hier direkt:
            import feedparser
            feed = feedparser.parse(response.content)
            all_entries.extend(feed.entries)
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_entries = []
        for entry in all_entries:
            if 'published_parsed' in entry:
                published_date = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                if published_date >= cutoff_date:
                    entry.published_date = published_date
                    recent_entries.append(entry)
        if not recent_entries:
            print(f"Keine Artikel der letzten {days} Tage gefunden.")
        else:
            for entry in recent_entries:
                title = entry.title
                text = entry.summary if "summary" in entry else entry.get("description", "")
                if not text.strip():
                    continue
                pred = predict_sentiment(policy, text, vocab)
                sentiment_label = sentiment_map[pred]
                sentiment_counts[sentiment_label] += 1
                print("Artikel:", title)
                print("Veröffentlicht:", entry.published_date.strftime("%Y-%m-%d"))
                print("Vorhersage:", sentiment_label)
                print("-" * 80)
            print("Aggregierte Ergebnisse:")
            for label, count in sentiment_counts.items():
                print(f"{label}: {count} Artikel")

    # Ergebnisse speichern und visualisieren
    save_path = config_data.get("visualization_save_path", "training_results.png")
    # Speichere die Visualisierung im in config.json definierten Ordner (z. B. results_dir)
    results_dir = config_data.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, save_path)
    visualize_training(history, save_path=save_path)
    model_dir = args.save_path
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, "sentiment_model.pth")
    torch.save(policy.state_dict(), model_save_path)
    print("Training abgeschlossen.")
    print("Modell und Ergebnisse wurden gespeichert.")


def run_training(episodes, lr, gamma, dropout_rate, update_frequency):
    """
    Führt das Training des RL-basierten Sentiment-Analyse-Modells durch.

    Parameter:
        episodes (int): Anzahl der Trainingsepisoden.
        lr (float): Lernrate.
        gamma (float): Discount-Faktor.
        dropout_rate (float): Dropout-Rate im Netzwerk.
        update_frequency (int): Anzahl der Samples, nach denen ein Update durchgeführt wird.

    Returns:
        tuple: (history, model_save_path)
    """
    # Lade Konfiguration aus config.json
    with open("config/config.json", "r") as f:
        config_data = json.load(f)
    feed_urls = config_data.get("rss_feeds", [])
    days = config_data.get("days", 30)

    # Dummy-Daten: 0 = NEGATIV, 1 = NEUTRAL, 2 = POSITIV
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

    # Vokabular aufbauen
    train_texts = [text for text, _ in dummy_data]
    vocab = build_vocab(train_texts, min_freq=1)
    vocab_size = len(vocab)
    print("Vocabulary size:", vocab_size)

    dataset = SentimentDataset(dummy_data, vocab)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # Netzwerk-Parameter
    embedding_dim = 50
    hidden_dim = 64
    output_dim = 3  # 3 Klassen: NEGATIV, NEUTRAL, POSITIV
    policy = SentimentPolicy(vocab_size, embedding_dim, hidden_dim, output_dim,
                             dropout_rate=dropout_rate, pad_idx=vocab['<PAD>'])
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    agent = SentimentAgent(policy, optimizer, gamma=gamma)

    history = {"episode_rewards": [], "losses": [], "eval_returns": []}

    for episode in range(episodes):
        episode_reward = 0
        sample_counter = 0
        random.shuffle(dummy_data)
        for text, label in dummy_data:
            indices = text_to_indices(text, vocab)
            text_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
            length = torch.tensor([len(indices)])
            action = agent.select_action(text_tensor, length)
            reward = 1.0 if action == label else -1.0
            agent.store_reward(reward)
            episode_reward += reward
            sample_counter += 1
            if sample_counter % update_frequency == 0:
                loss_val = agent.update()
                history["losses"].append(loss_val)
        if agent.rewards:
            loss_val = agent.update()
            history["losses"].append(loss_val)
        history["episode_rewards"].append(episode_reward)
        print(f"Episode {episode + 1}/{episodes}: Reward = {episode_reward}")
        if (episode + 1) % 10 == 0:
            eval_reward = 0
            for text, label in dummy_data:
                pred = predict_sentiment(policy, text, vocab)
                eval_reward += (1.0 if pred == label else -1.0)
            avg_eval = eval_reward / len(dummy_data)
            history["eval_returns"].append(avg_eval)
            print(f"Evaluation nach Episode {episode + 1}: Durchschnittlicher Reward = {avg_eval:.2f}")

    # Ergebnisse speichern und visualisieren
    results_dir = config_data.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    vis_filename = config_data.get("visualization_save_path", "training_results.png")
    save_path = os.path.join(results_dir, vis_filename)
    visualize_training(history, save_path=save_path)
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, "sentiment_model.pth")
    torch.save(policy.state_dict(), model_save_path)
    print("Training abgeschlossen. Modell und Ergebnisse wurden gespeichert.")

    return history, model_save_path


if __name__ == "__main__":
    main()
