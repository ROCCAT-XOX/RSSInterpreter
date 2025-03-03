# utils/visualizer.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def visualize_training(history, save_path=None):
    plt.figure(figsize=(16, 12))

    # Episode Rewards
    plt.subplot(2, 2, 1)
    plt.plot(history['episode_rewards'], 'b-', label='Episode Reward')
    plt.title('Episode Rewards', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.grid(True, alpha=0.3)
    if len(history['episode_rewards']) > 5:
        window = min(10, len(history['episode_rewards']) // 3)
        rolling_mean = pd.Series(history['episode_rewards']).rolling(window=window).mean()
        plt.plot(rolling_mean, 'r-', linewidth=2, label=f'{window}-Episode Moving Avg')
    plt.legend()

    # Training Losses
    plt.subplot(2, 2, 2)
    plt.plot(history['losses'], 'm-', label='Training Loss')
    plt.title('Training Loss', fontsize=14)
    plt.xlabel('Update Iterationen', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Evaluation Returns
    plt.subplot(2, 2, 3)
    if history['eval_returns']:
        eval_indices = np.linspace(1, len(history['episode_rewards']), len(history['eval_returns']))
        plt.plot(eval_indices, history['eval_returns'], 'g-o', label='Evaluation Reward')
        plt.title('Evaluation Returns', fontsize=14)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Durchschnittlicher Reward', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'Keine Evaluierungsdaten verfügbar',
                 horizontalalignment='center', verticalalignment='center', fontsize=14)

    # Histogramm der Evaluation Returns
    plt.subplot(2, 2, 4)
    if history['eval_returns']:
        plt.hist(history['eval_returns'], bins=10, color='c', alpha=0.7)
        plt.title('Verteilung der Evaluation Returns', fontsize=14)
        plt.xlabel('Durchschnittlicher Reward', fontsize=12)
        plt.ylabel('Häufigkeit', fontsize=12)
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Keine Evaluierungsdaten verfügbar',
                 horizontalalignment='center', verticalalignment='center', fontsize=14)

    plt.suptitle('Trainingsergebnisse für RL-basierte Sentiment-Analyse', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trainingsergebnisse gespeichert unter: {save_path}")
    else:
        plt.show()
    plt.close()
