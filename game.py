import numpy as np
import pygame
import random
import math

class SnakeGameAI:
    def __init__(self, width=400, height=400, block_size=20):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.block_size = block_size
        self.steps_limit = 500
        self.steps_without_food = 0
        self.prev_direction = None
        self.danger = 0
        self.reset()

    def distance(self):
        return ((self.snake[0][0] - self.food[0]) ** 2 + (self.snake[0][1] - self.food[1]) ** 2) ** 0.5

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

    def reset(self):
        self.snake = [(100, 100), (80, 100), (60, 100)]
        self.food = self._place_food()
        self.direction = (self.block_size, 0)  # 初始向右
        self.prev_direction = self.direction
        self.score = 0
        self.steps_without_food = 0
        return self._get_state()

    def step(self, action):
        # 轉換動作為方向
        directions = [(-self.block_size, 0), (self.block_size, 0), (0, -self.block_size), (0, self.block_size)]
        self.direction = directions[action]
        
        # 移動蛇
        head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
        self.snake.insert(0, head)

        # 計算移動前的距離
        distance_before = self.distance()

        # 增加步數計數器
        self.steps_without_food += 1

        # 判斷是否吃到食物
        if head == self.food:
            self.score += 1
            self.food = self._place_food()
            reward = 50
            self.steps_without_food = 0  # 重置步數計數器

        else:
            self.snake.pop()
            reward = 0
            self.danger = 0
            distance_after = self.distance()
            if distance_after < distance_before:
                reward += 1  # 靠近食物的獎勵
            else:
                reward += -0.2  # 遠離食物額外懲罰

            if (self.food[0] < head[0] and self.direction == (-self.block_size, 0)) or \
            (self.food[0] > head[0] and self.direction == (self.block_size, 0)) or \
            (self.food[1] < head[1] and self.direction == (0, -self.block_size)) or \
            (self.food[1] > head[1] and self.direction == (0, self.block_size)):
                reward += 0.2
            else:
                reward += -0.2
            if self.calculate_safe_space(head) < 2:
                self.danger = 1
                reward += -2
        # 檢查是否超過 50 步未吃到食物
        if self.steps_without_food >= self.steps_limit:
            return self._get_state(), -100, True, self.score  # 大懲罰並結束遊戲

        # 撞牆或撞到自己
        if self._is_collision(head):
            return self._get_state(), -100, True, self.score

        return self._get_state(), reward, False, self.score

    def _place_food(self):
        while True:
            x = random.randint(0, (self.width - self.block_size) // self.block_size) * self.block_size
            y = random.randint(0, (self.height - self.block_size) // self.block_size) * self.block_size
            if (x, y) not in self.snake:  # 確保食物不與蛇重疊
                return (x, y)

    def _is_collision(self, point):
        return (point in self.snake[1:] or  # 撞到自己
                point[0] < 0 or point[1] < 0 or
                point[0] >= self.width or point[1] >= self.height)

    def _get_state(self):
        head = self.snake[0]


        dx_food = 0 if self.food[0] == head[0] else (1 if self.food[0] > head[0] else -1)   # x 轴相对位置
        dy_food = 0 if self.food[1] == head[1] else (1 if self.food[1] > head[1] else -1)   # y 轴相对位置
        # dx_food = (self.food[0] - head[0]) // self.block_size  # x 轴相对位置
        # dy_food = (self.food[1] - head[1]) // self.block_size  # y 轴相对位置

        dx_direction = self.direction[0] // self.block_size
        dy_direction = self.direction[1] // self.block_size
        state = [
            # 危險方向
            self._is_collision((head[0] - self.block_size, head[1])),
            self._is_collision((head[0] + self.block_size, head[1])),
            self._is_collision((head[0], head[1] - self.block_size)),
            self._is_collision((head[0], head[1] + self.block_size)),
            # 食物方向
            dx_food,
            dy_food,
            # 當前方向
            dx_direction,
            dy_direction,
            # 尺寸
            self.score > 100,
            # 死路
            self.danger
        ]
        return np.array(state, dtype=int)
