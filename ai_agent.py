import numpy as np
from game_logic import SnakeGameLogic

class SnakeGameAI:
    def __init__(self, width=400, height=400, block_size=20):
        self.game = SnakeGameLogic(width, height, block_size)
        self.steps_limit = 500
        self.steps_without_food = 0
        self.prev_direction = None
        self.danger = 0

    def reset(self):
        return self.game.reset()
    
    def step(self, action):
        """根據動作執行一步遊戲"""
        state, score, done = self.game.step(action)
        reward = self.calculate_reward(state, done)
        return state, reward, done, score

    def calculate_reward(self, state, done):
        """計算當前動作的獎勵"""
        if done:
            return -100  # 撞牆或撞到自己
        head = state['snake'][0]
        food = state['food']
        distance = ((head[0] - food[0]) ** 2 + (head[1] - food[1]) ** 2) ** 0.5
        reward = -0.1  # 每一步的基本懲罰
        if distance == 0:
            reward += 50  # 吃到食物的獎勵
            self.steps_without_food = 0  # 重置步數計數器
        else :
            self.steps_without_food += 1    # 步數計數器


        return reward

    def calculate_safe_space(self, head):
        directions = [
            (-self.block_size, 0),(0, -self.block_size), 
            (self.block_size, 0), (0, self.block_size)
        ]
        safe_space = 0
        for dx, dy in directions:
            next_pos = (head[0] + dx, head[1] + dy)
            if not self._is_collision(next_pos):
                safe_space += 1
        return safe_space

    def get_state(self):
        """返回 AI 可以使用的遊戲狀態"""
        state = self.game.get_game_state()
        head = state['snake'][0]
        food = state['food']

        dx_food = 1 if food[0] > head[0] else (-1 if food[0] < head[0] else 0)
        dy_food = 1 if food[1] > head[1] else (-1 if food[1] < head[1] else 0)

        # 是否有障礙物
        dangers = [
            self.game._is_collision((head[0] - self.game.block_size, head[1])),
            self.game._is_collision((head[0] + self.game.block_size, head[1])),
            self.game._is_collision((head[0], head[1] - self.game.block_size)),
            self.game._is_collision((head[0], head[1] + self.game.block_size)),
        ]

        return np.array(dangers + [dx_food, dy_food], dtype=int)