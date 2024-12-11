import random

class SnakeGameLogic:
    def __init__(self, width=400, height=400, block_size=20):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.reset()

    def reset(self):
        self.snake = [(100, 100), (80, 100), (60, 100)]
        self.food = self._place_food()
        self.direction = (self.block_size, 0)  # 初始向右
        self.score = 0
        return self.get_game_state()
    
    def step(self, action):
        # 轉換動作為方向
        directions = [(-self.block_size, 0), (self.block_size, 0), (0, -self.block_size), (0, self.block_size)]
        self.direction = directions[action]
        
        # 移動蛇
        head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
        self.snake.insert(0, head)

        # 判斷是否吃到食物
        if head == self.food:
            self.score += 1
            self.food = self._place_food()
        else:
            self.snake.pop()    # 沒有吃到食物就移除尾部

        # 判斷遊戲是否結束
        done = self._is_collision(head)
        return self.get_game_state(), self.score, done

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
    
    def get_game_state(self):
        """返回當前遊戲狀態"""
        return {
            'snake': self.snake,
            'food': self.food,
            'direction': self.direction,
            'score': self.score
        }