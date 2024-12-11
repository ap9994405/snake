import torch
import pygame
from game import SnakeGameAI
from agent import DQNAgent

def play():
   # 初始化游戏和加载模型
    game = SnakeGameAI()
    agent = DQNAgent(state_size=10, action_size=4)
    
    # 加载训练好的模型权重
    model_path="snake_ai_model.pth"
    agent.model.load_state_dict(torch.load(model_path))
    agent.model.eval()  # 设置为推理模式

    clock = pygame.time.Clock()
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

        # 可視化遊戲
        draw_game(game)
        clock.tick(20)  # 控制遊戲速度

    print(f"遊戲結束，得分: {score}")

def draw_game(game):
    # 初始化 Pygame 窗口
    pygame.init()
    screen = pygame.display.set_mode((game.width, game.height))
    pygame.display.set_caption("AI 貪吃蛇")
    screen.fill((0, 0, 0))

    # 繪製蛇
    for segment in game.snake[1:]:
        pygame.draw.rect(screen, (0, 255, 0), (segment[0], segment[1], game.block_size, game.block_size))

    head = game.snake[0]
    direction = game.direction
    if direction == (-game.block_size, 0):  # 向左
        points = [
            (head[0] + game.block_size, head[1]),
            (head[0], head[1] + game.block_size // 2),
            (head[0] + game.block_size, head[1] + game.block_size),
        ]
    elif direction == (game.block_size, 0):  # 向右
        points = [
            (head[0], head[1]),
            (head[0] + game.block_size, head[1] + game.block_size // 2),
            (head[0], head[1] + game.block_size),
        ]
    elif direction == (0, -game.block_size):  # 向上
        points = [
            (head[0], head[1] + game.block_size),
            (head[0] + game.block_size // 2, head[1]),
            (head[0] + game.block_size, head[1] + game.block_size),
        ]
    elif direction == (0, game.block_size):  # 向下
        points = [
            (head[0], head[1]),
            (head[0] + game.block_size // 2, head[1] + game.block_size),
            (head[0] + game.block_size, head[1]),
        ]
    pygame.draw.polygon(screen, (255, 255, 0), points)  # 黃色箭頭表示蛇頭

    # 繪製食物
    pygame.draw.rect(screen, (255, 0, 0), (game.food[0], game.food[1], game.block_size, game.block_size))

    # 刷新屏幕
    pygame.display.flip()

if __name__ == "__main__":
    play()
