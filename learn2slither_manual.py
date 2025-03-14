import pygame
import random
import argparse
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.Font('arial.ttf', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
BLOCK_SIZE = 20


class SnakeGameHuman:

    def __init__(self, size=20, speed=10):
        self.size = size
        self.w = size * BLOCK_SIZE
        self.h = size * BLOCK_SIZE
        self.speed = speed

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake - Modo Humano')
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

        self.food_good1 = None
        self.food_good2 = None
        self.food_bad = None
        self._place_food_good1()
        self._place_food_good2()
        self._place_food_bad()

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

    def play_step(self):
        # 1. Controles del usuario
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and \
                        self.direction != Direction.RIGHT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT and \
                        self.direction != Direction.LEFT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP and \
                        self.direction != Direction.DOWN:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN and\
                        self.direction != Direction.UP:
                    self.direction = Direction.DOWN

        self._move(self.direction)
        self.snake.insert(0, self.head)

        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score

        if self.head == self.food_good1:
            self.score += 1
            self._place_food_good1()
        elif self.head == self.food_good2:
            self.score += 1
            self._place_food_good2()
        elif self.head == self.food_bad:
            if len(self.snake) <= 2:
                game_over = True
                return game_over, self.score
            else:
                self.snake.pop()
                self.snake.pop()
                self._place_food_bad()
        else:
            self.snake.pop()
        self._update_ui()
        self.clock.tick(self.speed)

        return game_over, self.score

    def _is_collision(self):
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < \
            0 or self.head.y > \
                self.h - BLOCK_SIZE or self.head.y < 0:
            return True
        if self.head in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
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

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)


def play_human_mode():
    parser = argparse.ArgumentParser(description='Snake - Human mode')
    parser.add_argument('-size', type=int, default=20, help='Board size')
    parser.add_argument('-speed', type=int, default=10, help='Speed')
    args = parser.parse_args()

    game = SnakeGameHuman(size=args.size, speed=args.speed)
    while True:
        game_over, score = game.play_step()

        if game_over:
            break
    print(f'Final Score: {score}')
    waiting = True
    font = pygame.font.Font('arial.ttf', 35)
    text = font.render(f"Game over! Score: {score}", True, WHITE)
    text_rect = text.get_rect(center=(game.w/2, game.h/2))
    game.display.blit(text, text_rect)
    pygame.display.flip()

    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                waiting = False
                break

    pygame.quit()


if __name__ == '__main__':
    play_human_mode()
