import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np


# Basic definitions
pygame.init()
font = pygame.font.Font('arial.ttf', 20)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)  # Good food
RED = (255, 0, 0)    # Bad food
BLUE1 = (0, 0, 255)  # Snake border
BLUE2 = (0, 100, 255)  # Snake fill
BLACK = (0, 0, 0)    # Background

# Constants
BLOCK_SIZE = 20
SPEED = 40


class SnakeGameAI:
    def __init__(self, size=20, is_visual=True):
        self.size = size
        self.w = size * BLOCK_SIZE
        self.h = size * BLOCK_SIZE
        self.is_visual = is_visual

        if is_visual:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake AI')
        else:
            self.display = None

        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        center_x = (self.size // 2) * BLOCK_SIZE
        center_y = (self.size // 2) * BLOCK_SIZE
        self.head = Point(center_x, center_y)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        # Create foods
        self.food_good1 = None
        self.food_good2 = None
        self.food_bad = None
        self._place_food_good1()
        self._place_food_good2()
        self._place_food_bad()
        self.frame_iteration = 0

    def _place_food_good1(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food_good1 = Point(x, y)

        if (self.food_good1 in self.snake or
            (self.food_bad and self.food_good1 == self.food_bad) or
                (self.food_good2 and self.food_good1 == self.food_good2)):
            self._place_food_good1()

    def _place_food_good2(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food_good2 = Point(x, y)
        if (self.food_good2 in self.snake or
            (self.food_bad and self.food_good2 == self.food_bad) or
                (self.food_good1 and self.food_good2 == self.food_good1)):
            self._place_food_good2()

    def _place_food_bad(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food_bad = Point(x, y)
        if (self.food_bad in self.snake or
            (self.food_good1 and self.food_bad == self.food_good1) or
                (self.food_good2 and self.food_bad == self.food_good2)):
            self._place_food_bad()

    def play_step(self, action):
        self.frame_iteration += 1

        # events
        if self.is_visual:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        else:
            pygame.event.pump()

        old_head = self.head
        old_distance_good1 = abs(old_head.x - self.food_good1.x) + \
            abs(old_head.y - self.food_good1.y)
        old_distance_good2 = abs(old_head.x - self.food_good2.x) + \
            abs(old_head.y - self.food_good2.y)
        old_distance_bad = abs(old_head.x - self.food_bad.x) + \
            abs(old_head.y - self.food_bad.y)

        old_food_visible_good1 = self.is_food_visible(food_type="good1")
        old_food_visible_good2 = self.is_food_visible(food_type="good2")
        old_food_visible_bad = self.is_food_visible(food_type="bad")

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False

        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food_good1:
            self.score += 1
            reward = 10
            self._place_food_good1()
        elif self.head == self.food_good2:
            self.score += 1
            reward = 10
            self._place_food_good2()
        elif self.head == self.food_bad:
            if len(self.snake) <= 2:
                game_over = True
                reward = -10
                return reward, game_over, self.score
            else:
                self.snake.pop()
                reward = -5
                self._place_food_bad()
        else:
            new_distance_good1 = abs(self.head.x - self.food_good1.x) + \
                abs(self.head.y - self.food_good1.y)
            new_distance_good2 = abs(self.head.x - self.food_good2.x) + \
                abs(self.head.y - self.food_good2.y)
            new_distance_bad = abs(self.head.x - self.food_bad.x) + \
                abs(self.head.y - self.food_bad.y)

            new_food_visible_good1 = self.is_food_visible(food_type="good1")
            new_food_visible_good2 = self.is_food_visible(food_type="good2")
            new_food_visible_bad = self.is_food_visible(food_type="bad")

            # reward for getting closer to the food
            if old_food_visible_good1 or new_food_visible_good1:
                if new_distance_good1 < old_distance_good1:
                    reward += 5
                elif new_distance_good1 > old_distance_good1:
                    reward -= 1

            if old_food_visible_good2 or new_food_visible_good2:
                if new_distance_good2 < old_distance_good2:
                    reward += 5
                elif new_distance_good2 > old_distance_good2:
                    reward -= 1

            if old_food_visible_bad or new_food_visible_bad:
                if new_distance_bad > old_distance_bad:
                    reward += 2
                elif new_distance_bad < old_distance_bad:
                    reward -= 1

            self.snake.pop()

        if self.is_visual:
            self._update_ui()

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - \
                BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        if not self.is_visual:
            return
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x,
                                                              pt.y,
                                                              BLOCK_SIZE,
                                                              BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4,
                                                              pt.y+4,
                                                              12,
                                                              12))

        pygame.draw.rect(self.display, GREEN, pygame.Rect(self.food_good1.x,
                                                          self.food_good1.y,
                                                          BLOCK_SIZE,
                                                          BLOCK_SIZE))
        pygame.draw.rect(self.display, GREEN, pygame.Rect(self.food_good2.x,
                                                          self.food_good2.y,
                                                          BLOCK_SIZE,
                                                          BLOCK_SIZE))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food_bad.x,
                                                        self.food_bad.y,
                                                        BLOCK_SIZE,
                                                        BLOCK_SIZE))
        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN,
                      Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def is_food_visible(self, direction=None, food_type="good1"):
        head = self.head

        if food_type == "good1":
            food = self.food_good1
        elif food_type == "good2":
            food = self.food_good2
        else:  # "bad"
            food = self.food_bad

        if direction is None:
            return (self.is_food_visible(Direction.LEFT, food_type) or
                    self.is_food_visible(Direction.RIGHT, food_type) or
                    self.is_food_visible(Direction.UP, food_type) or
                    self.is_food_visible(Direction.DOWN, food_type))

        if direction == Direction.RIGHT:
            if head.y == food.y and head.x < food.x:
                for body in self.snake[1:]:
                    if body.y == head.y and head.x < body.x < food.x:
                        return False
                return True
        elif direction == Direction.LEFT:
            if head.y == food.y and head.x > food.x:
                for body in self.snake[1:]:
                    if body.y == head.y and food.x < body.x < head.x:
                        return False
                return True
        elif direction == Direction.UP:
            if head.x == food.x and head.y > food.y:
                for body in self.snake[1:]:
                    if body.x == head.x and food.y < body.y < head.y:
                        return False
                return True
        elif direction == Direction.DOWN:
            if head.x == food.x and head.y < food.y:
                for body in self.snake[1:]:
                    if body.x == head.x and head.y < body.y < food.y:
                        return False
                return True

        return False
