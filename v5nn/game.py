import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):
        self.frame_iteration += 1
        # 1. procesar eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # Obtener estado actual antes de moverse
        old_head = self.head
        old_food_visible = self.is_food_visible()
        old_distance = abs(old_head.x - self.food.x) + abs(old_head.y - self.food.y)
        
        # 2. moverse
        self._move(action)
        self.snake.insert(0, self.head)
        
        # 3. comprobar fin del juego
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. colocar nueva comida o simplemente moverse
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            # Obtener nuevo estado después de moverse
            new_food_visible = self.is_food_visible()
            new_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
            
            # Si la comida es visible
            if old_food_visible or new_food_visible:
                # Recompensa por acercarse o castigo por alejarse
                if new_distance < old_distance:
                    reward = 5  # Recompensa por acercarse
                elif new_distance > old_distance:
                    reward = -1  # Castigo por alejarse
            
            self.snake.pop()
        
        # 5. actualizar UI y reloj
        self._update_ui()
        self.clock.tick(SPEED)
        
        # 6. devolver resultado
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False


    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

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

    def is_food_visible(self, direction=None):
        """
        Comprueba si la comida es visible desde la cabeza de la serpiente en línea recta
        en la dirección especificada. Si no se especifica dirección, comprueba todas.
        """
        head = self.head
        food = self.food
        
        if direction is None:
            return (self.is_food_visible(Direction.LEFT) or
                    self.is_food_visible(Direction.RIGHT) or
                    self.is_food_visible(Direction.UP) or
                    self.is_food_visible(Direction.DOWN))
        
        # Comprueba si la comida está en línea recta en la dirección indicada
        if direction == Direction.RIGHT:
            if head.y == food.y and head.x < food.x:
                # Comprueba si hay alguna parte del cuerpo bloqueando la visión
                for body_part in self.snake[1:]:
                    if body_part.y == head.y and head.x < body_part.x < food.x:
                        return False
                return True
        
        elif direction == Direction.LEFT:
            if head.y == food.y and head.x > food.x:
                for body_part in self.snake[1:]:
                    if body_part.y == head.y and food.x < body_part.x < head.x:
                        return False
                return True
        
        elif direction == Direction.UP:
            if head.x == food.x and head.y > food.y:
                for body_part in self.snake[1:]:
                    if body_part.x == head.x and food.y < body_part.y < head.y:
                        return False
                return True
        
        elif direction == Direction.DOWN:
            if head.x == food.x and head.y < food.y:
                for body_part in self.snake[1:]:
                    if body_part.x == head.x and head.y < body_part.y < food.y:
                        return False
                return True
        
        return False