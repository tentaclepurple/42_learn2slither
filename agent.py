import os
import random
import torch
import numpy as np
from collections import deque
from environment import Direction, Point


# Constants for the agent
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Linear_QNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_path='model.pth'):
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(
                    self.model(next_state[idx])
                    )

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


class Agent:
    def __init__(self, load_model=None, dont_learn=False):
        self.n_games = 0
        self.epsilon = 80  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(19, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.dont_learn = dont_learn

        if load_model and os.path.exists(load_model):
            try:
                print(f"Loading model from {load_model}")
                self.model.load_state_dict(torch.load(load_model))
                print("Model loaded successfully!")
                if dont_learn:
                    self.epsilon = -1000  # no exploration
                else:
                    self.epsilon = 20
            except Exception as e:
                print(f"Error loading model: {e}")

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        food_good1_visible_left = game.is_food_visible(Direction.LEFT,
                                                       "good1")
        food_good1_visible_right = game.is_food_visible(Direction.RIGHT,
                                                        "good1")
        food_good1_visible_up = game.is_food_visible(Direction.UP,
                                                     "good1")
        food_good1_visible_down = game.is_food_visible(Direction.DOWN,
                                                       "good1")

        food_good2_visible_left = game.is_food_visible(Direction.LEFT,
                                                       "good2")
        food_good2_visible_right = game.is_food_visible(Direction.RIGHT,
                                                        "good2")
        food_good2_visible_up = game.is_food_visible(Direction.UP,
                                                     "good2")
        food_good2_visible_down = game.is_food_visible(Direction.DOWN,
                                                       "good2")

        food_bad_visible_left = game.is_food_visible(Direction.LEFT,
                                                     "bad")
        food_bad_visible_right = game.is_food_visible(Direction.RIGHT,
                                                      "bad")
        food_bad_visible_up = game.is_food_visible(Direction.UP,
                                                   "bad")
        food_bad_visible_down = game.is_food_visible(Direction.DOWN,
                                                     "bad")

        state = [
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            food_good1_visible_left,
            food_good1_visible_right,
            food_good1_visible_up,
            food_good1_visible_down,

            food_good2_visible_left,
            food_good2_visible_right,
            food_good2_visible_up,
            food_good2_visible_down,

            food_bad_visible_left,
            food_bad_visible_right,
            food_bad_visible_up,
            food_bad_visible_down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if self.dont_learn:
            return

        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        if not self.dont_learn:
            self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        if self.dont_learn:
            use_random = random.randint(0, 1000) < 1  # 0.1% aleatory
        else:
            # normal behaviour exploration-exploitation
            self.epsilon = max(80 - self.n_games, 0)
            use_random = random.randint(0, 200) < self.epsilon

        final_move = [0, 0, 0]
        if use_random:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
