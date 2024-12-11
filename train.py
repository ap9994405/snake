import time
from game import SnakeGameAI
from agent import DQNAgent
import torch

def train():
    total_score = 0
    Max_score = 0
    game = SnakeGameAI()
    agent = DQNAgent(state_size=10, action_size=4)
    episodes = 500

    start_time = time.time()
    
    for episode in range(episodes):
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, 0.001)
        state = game.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, score = game.step(action)
            agent.train_step(state, action, reward, next_state, done)
            state = next_state
        if (score > Max_score):
            Max_score = score
        total_score += score
        print(f"Episode {episode+1}, Score: {score}")  

    print(f"total_score: {total_score}")   
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")

    return agent  # 返回訓練完成的代理

if __name__ == "__main__":
    trained_agent = train()
    # 保存模型
    torch.save(trained_agent.model.state_dict(), "snake_ai_model.pth")
    print("模型已保存為 snake_ai_model.pth")
