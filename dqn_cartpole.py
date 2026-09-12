import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# ===== Rede neural que estima o valor de cada ação =====
class DQN(nn.Module):
    def __init__(self, n_estados, n_acoes):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(n_estados, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_acoes)
        )

    def forward(self, x):
        return self.rede(x)


# ===== Replay buffer: memória de experiências =====
class ReplayBuffer:
    def __init__(self, capacidade=10000):
        self.buffer = deque(maxlen=capacidade)

    def adicionar(self, estado, acao, recompensa, proximo_estado, done):
        self.buffer.append((estado, acao, recompensa, proximo_estado, done))

    def amostrar(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        estados, acoes, recompensas, proximos_estados, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(estados)),
            torch.LongTensor(acoes),
            torch.FloatTensor(recompensas),
            torch.FloatTensor(np.array(proximos_estados)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


# ===== Configuração do ambiente =====
env = gym.make("CartPole-v1")
n_estados = env.observation_space.shape[0]
n_acoes = env.action_space.n

rede_principal = DQN(n_estados, n_acoes)
rede_alvo = DQN(n_estados, n_acoes)
rede_alvo.load_state_dict(rede_principal.state_dict())  # começa igual à principal

otimizador = optim.Adam(rede_principal.parameters(), lr=1e-3)
buffer = ReplayBuffer()

# Hiperparâmetros
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
atualizar_rede_alvo_a_cada = 10   # episódios
n_episodios = 1000

recompensas_por_episodio = []

for episodio in range(n_episodios):
    estado, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Exploração vs. explotação
        if np.random.rand() < epsilon:
            acao = env.action_space.sample()
        else:
            with torch.no_grad():
                q_valores = rede_principal(torch.FloatTensor(estado))
                acao = torch.argmax(q_valores).item()

        proximo_estado, recompensa, terminated, truncated, _ = env.step(acao)
        done = terminated or truncated
        total_reward += recompensa

        buffer.adicionar(estado, acao, recompensa, proximo_estado, done)
        estado = proximo_estado

        # Treina a rede com uma amostra aleatória do buffer
        if len(buffer) >= batch_size:
            estados_b, acoes_b, recompensas_b, proximos_b, dones_b = buffer.amostrar(batch_size)

            q_atual = rede_principal(estados_b).gather(1, acoes_b.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                q_proximo = rede_alvo(proximos_b).max(1)[0]
                q_alvo = recompensas_b + gamma * q_proximo * (1 - dones_b)

            perda = nn.functional.mse_loss(q_atual, q_alvo)

            otimizador.zero_grad()
            perda.backward()
            otimizador.step()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    recompensas_por_episodio.append(total_reward)

    # Atualiza a rede-alvo periodicamente
    if episodio % atualizar_rede_alvo_a_cada == 0:
        rede_alvo.load_state_dict(rede_principal.state_dict())

    if (episodio + 1) % 20 == 0:
        media_recente = np.mean(recompensas_por_episodio[-20:])
        print(f"Episódio {episodio + 1} | Recompensa média (últimos 20): {media_recente:.2f} | Epsilon: {epsilon:.3f}")

env.close()
# ===== Salvar o modelo treinado =====
torch.save(rede_principal.state_dict(), "dqn_cartpole.pth")
print("Modelo salvo em dqn_cartpole.pth")
print("\nTreino concluído! No CartPole-v1, uma recompensa média perto de 500 significa que o agente aprendeu a equilibrar a haste quase perfeitamente.")

# ===== Visualização do agente treinado =====
env_visual = gym.make("CartPole-v1", render_mode="human")
n_episodios_visuais = 5

for ep in range(n_episodios_visuais):
    estado, _ = env_visual.reset()
    done = False
    total_reward = 0

    print(f"\n--- Episódio de demonstração {ep + 1} ---")

    while not done:
        with torch.no_grad():
            q_valores = rede_principal(torch.FloatTensor(estado))
            acao = torch.argmax(q_valores).item()  # sem exploração, só usa o que aprendeu

        estado, reward, terminated, truncated, _ = env_visual.step(acao)
        done = terminated or truncated
        total_reward += reward

    print(f"Recompensa do episódio: {total_reward}")

env_visual.close()  

import gymnasium as gym
import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, n_estados, n_acoes):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(n_estados, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_acoes)
        )

    def forward(self, x):
        return self.rede(x)


env = gym.make("CartPole-v1", render_mode="human")
n_estados = env.observation_space.shape[0]
n_acoes = env.action_space.n

rede = DQN(n_estados, n_acoes)
rede.load_state_dict(torch.load("dqn_cartpole.pth"))
rede.eval()  # modo de avaliação, não de treino

for ep in range(5):
    estado, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        with torch.no_grad():
            acao = torch.argmax(rede(torch.FloatTensor(estado))).item()
        estado, reward, terminated, truncated, _ = env.step(acao)
        done = terminated or truncated
        total_reward += reward

    print(f"Recompensa: {total_reward}")

env.close()
