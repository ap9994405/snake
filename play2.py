import torch
from game import SnakeGameAI
from agent import DQNAgent

def play(model_path="snake_ai_model.pth", episodes=100):
    # 初始化游戏和加载模型
    game = SnakeGameAI()
    agent = DQNAgent(state_size=10, action_size=4)
    
    # 加载训练好的模型权重
    agent.model.load_state_dict(torch.load(model_path))
    agent.model.eval()  # 设置为推理模式

    total_score = 0
    highest_score = 0

    # 多次运行游戏
    for episode in range(episodes):
        state = game.reset()
        done = False
        score = 0

        while not done:
            # 根据当前状态选择动作
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
                action = torch.argmax(agent.model(state_tensor)).item()

            # 执行动作
            next_state, _, done, score = game.step(action)
            state = next_state

        total_score += score
        if score > highest_score:
            highest_score = score
        print(f"Episode {episode + 1}/{episodes}, Score: {score}")

    # 输出平均分
    average_score = total_score / episodes
    print(f"Completed {episodes} episodes. Average Score: {average_score:.2f}, Highest Score: {highest_score}")

if __name__ == "__main__":
    play()