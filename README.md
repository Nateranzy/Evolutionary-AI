# IA Evolutiva — Reinforcement Learning em Python

Projeto de portfólio explorando **reinforcement learning**: uma IA que aprende e melhora seu comportamento através de tentativa, erro e recompensa — sem ser programada com regras fixas.

## O que tem aqui

### 1. Q-Learning tabular (`main.py`)
Um táxi (ambiente `Taxi-v4` do Gymnasium) aprende a pegar e deixar passageiros no lugar certo usando uma tabela de valores Q, atualizada a cada ação.

- **Resultado**: recompensa média foi de -168.70 (episódio 1000, quase todo aleatório) até se estabilizar em ~7.3 (episódio 10000), próximo do máximo teórico do ambiente.

### 2. Deep Q-Network — DQN (`dqn_cartpole.py`)
Evolução do primeiro projeto: em vez de uma tabela, uma **rede neural em PyTorch** aprende a estimar o valor de cada ação, permitindo lidar com estados contínuos. O agente equilibra uma haste em pé no ambiente `CartPole-v1`.

- **Replay buffer**: memória de experiências passadas usada para treinar a rede de forma mais estável.
- **Rede-alvo (target network)**: cópia da rede principal, atualizada periodicamente, usada para estabilizar o alvo de treino.
- **Bug corrigido**: o código inicial tratava `truncated` (limite de passos atingido — na prática, um sucesso) da mesma forma que `terminated` (queda real), o que ensinava a rede a evitar sobreviver até o fim. Corrigido separando os dois no cálculo do valor-alvo.
- **Resultado final**: após a correção e mais episódios de treino, o agente atingiu a pontuação máxima do ambiente (**500/500**) de forma consistente.

### 3. Demonstração do modelo treinado (`demo_cartpole.py`)
Carrega os pesos salvos (`dqn_cartpole.pth`) e mostra o agente jogando, sem precisar retreinar.

## Tecnologias

- Python 3.12
- [Gymnasium](https://gymnasium.farama.org/) — ambientes de reinforcement learning
- [PyTorch](https://pytorch.org/) — rede neural do DQN
- NumPy

## Como rodar

```bash
python -m venv venv
venv\Scripts\Activate.ps1      # Windows
pip install gymnasium numpy torch pygame

python main.py              # Q-Learning tabular (Taxi)
python dqn_cartpole.py      # Treina o DQN (CartPole) e salva o modelo
python demo_cartpole.py     # Demonstra o modelo já treinado
```

## Aprendizados

Este projeto foi uma introdução prática a reinforcement learning, cobrindo desde os fundamentos (Q-Learning tabular) até uma implementação de rede neural (DQN), incluindo diagnóstico e correção de um bug real de instabilidade de treino.
