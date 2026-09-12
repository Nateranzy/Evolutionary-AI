import gymnasium as gym
import numpy as np

# Cria o ambiente: um táxi que precisa pegar e deixar passageiros
env = gym.make("Taxi-v4")

# Tabela Q: uma linha por estado possível, uma coluna por ação possível
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Hiperparâmetros
alpha = 0.1          # taxa de aprendizado
gamma = 0.99         # importância de recompensas futuras
epsilon = 1.0        # taxa de exploração (começa explorando bastante)
epsilon_min = 0.01
epsilon_decay = 0.995
n_episodes = 10000

recompensas_por_episodio = []

for episode in range(n_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Exploração vs. explotação
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Atualização da Q-table (é aqui que o agente "aprende")
        q_table[state, action] += alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )
        state = next_state

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    recompensas_por_episodio.append(total_reward)

    # Mostra o progresso a cada 1000 episódios
    if (episode + 1) % 1000 == 0:
        media_recente = np.mean(recompensas_por_episodio[-1000:])
        print(f"Episódio {episode + 1} | Recompensa média (últimos 1000): {media_recente:.2f} | Epsilon: {epsilon:.3f}")

env.close()
print("\nTreino concluído! A recompensa média deve subir com o tempo — isso é a evolução acontecendo.")

# ===== Visualização do agente treinado =====
env_visual = gym.make("Taxi-v4", render_mode="human")
n_episodios_visuais = 5

for ep in range(n_episodios_visuais):
    state, _ = env_visual.reset()
    done = False
    total_reward = 0

    print(f"\n--- Episódio de demonstração {ep + 1} ---")

    while not done:
        action = np.argmax(q_table[state])  # sem exploração, só usa o que aprendeu
        state, reward, terminated, truncated, _ = env_visual.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Recompensa do episódio: {total_reward}")

env_visual.close()