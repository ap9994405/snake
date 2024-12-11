import torch
import torch.optim as optim
import random
from model import LinearQNet
# gamma 未來性 0~1
# optimizer 優化的step
# epsilon 突變率
# epsilon_decay 逐步降低突變率
class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=128, gamma=0.9, epsilon=0.95, epsilon_decay=0.99):
        self.model = LinearQNet(state_size, hidden_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.last_action = None

    def select_action(self, state):
        if random.random() < self.epsilon:
            available_actions = [0, 1, 2, 3]
            if self.last_action is not None:
                # 移除与当前方向相反的动作
                opposite_action = (self.last_action + 2) % 4
                available_actions.remove(opposite_action)
            return random.choice(available_actions)  # 隨機選擇動作
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            return torch.argmax(self.model(state_tensor)).item()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        target = reward + self.gamma * torch.max(self.model(next_state)) * (1 - done)   # 用於計算Q值
        prediction = self.model(state)[action]
        loss = (target - prediction).pow(2).mean()  # Loss Function
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

