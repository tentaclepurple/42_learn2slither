#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import pygame
import numpy as np

class Environment:
    def __init__(self, visual_mode=True, board_size=10):
        self.board_size = board_size
        self.visual_mode = visual_mode
        self.cell_size = 40
        
        # Initialize board and game elements
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.snake = []  # List of (x, y) coordinates, head is at index 0
        self.direction = None  # 0: up, 1: right, 2: down, 3: left
        self.green_apples = []
        self.red_apple = None
        
        # Initialize pygame if visual mode is enabled
        if visual_mode:
            pygame.init()
            self.window_size = (board_size * self.cell_size, board_size * self.cell_size)
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Snake Game")
            self.font = pygame.font.SysFont('Arial', 20)
        
        # Reset to initialize the game
        self.reset()
    
    def reset(self):
        """Reset the environment for a new game."""
        # Clear the board
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        
        # Initialize snake with length 3 at random position
        start_x = random.randint(2, self.board_size - 3)
        start_y = random.randint(2, self.board_size - 3)
        
        # Random initial direction
        self.direction = random.randint(0, 3)
        
        # Create snake based on initial direction
        self.snake = []
        if self.direction == 0:  # Up
            for i in range(3):
                self.snake.append((start_x, start_y + i))
        elif self.direction == 1:  # Right
            for i in range(3):
                self.snake.append((start_x - i, start_y))
        elif self.direction == 2:  # Down
            for i in range(3):
                self.snake.append((start_x, start_y - i))
        elif self.direction == 3:  # Left
            for i in range(3):
                self.snake.append((start_x + i, start_y))
        
        # Mark snake cells on the board
        for x, y in self.snake:
            self.board[y, x] = 1
        
        # Place 2 green apples
        self.green_apples = []
        for _ in range(2):
            self._place_green_apple()
        
        # Place 1 red apple
        self._place_red_apple()
        
        # Render the initial state
        if self.visual_mode:
            self.render()
        
        return self.get_board_state()
    
    def _place_green_apple(self):
        """Place a green apple in a random empty cell."""
        empty_cells = []
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.board[y, x] == 0:
                    empty_cells.append((x, y))
        
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[y, x] = 2  # 2 represents green apple
            self.green_apples.append((x, y))
    
    def _place_red_apple(self):
        """Place a red apple in a random empty cell."""
        empty_cells = []
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.board[y, x] == 0:
                    empty_cells.append((x, y))
        
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[y, x] = 3  # 3 represents red apple
            self.red_apple = (x, y)
    
    def step(self, action):
        """
        Execute action and return reward, new snake length, and game_over flag.
        Action: 0 (up), 1 (right), 2 (down), 3 (left)
        """
        # Update direction based on action
        self.direction = action
        
        # Calculate new head position
        head_x, head_y = self.snake[0]
        if self.direction == 0:  # Up
            new_head = (head_x, head_y - 1)
        elif self.direction == 1:  # Right
            new_head = (head_x + 1, head_y)
        elif self.direction == 2:  # Down
            new_head = (head_x, head_y + 1)
        elif self.direction == 3:  # Left
            new_head = (head_x - 1, head_y)
        
        # Check for wall collision
        new_x, new_y = new_head
        if new_x < 0 or new_x >= self.board_size or new_y < 0 or new_y >= self.board_size:
            return -50, len(self.snake), True  # High negative reward for hitting wall
        
        # Check for self collision
        if new_head in self.snake:
            return -50, len(self.snake), True  # High negative reward for hitting self
        
        # Check for apple collision
        ate_green_apple = new_head in self.green_apples
        ate_red_apple = new_head == self.red_apple
        
        # Update snake position
        self.snake.insert(0, new_head)
        
        # Process apple consumption
        if ate_green_apple:
            # Remove the eaten green apple
            self.green_apples.remove(new_head)
            self.board[new_y, new_x] = 1  # Mark as snake
            # Place a new green apple
            self._place_green_apple()
            reward = 25  # Higher positive reward for eating green apple
        elif ate_red_apple:
            # Remove the tail (in addition to the one below)
            if self.snake:
                tail = self.snake.pop()
                if tail in self.snake:  # If there are duplicate coordinates, don't clear the board
                    pass
                else:
                    self.board[tail[1], tail[0]] = 0  # Clear the tail
            
            # Place a new red apple
            self.red_apple = None
            self.board[new_y, new_x] = 1  # Mark as snake
            self._place_red_apple()
            
            reward = -15  # Negative reward for eating red apple
        else:
            # Remove the tail
            tail = self.snake.pop()
            if tail in self.snake:  # If there are duplicate coordinates, don't clear the board
                pass
            else:
                self.board[tail[1], tail[0]] = 0  # Clear the tail
            
            self.board[new_y, new_x] = 1  # Mark the new head

            # Calcular distancia al apple más cercano (reward por acercarse)
            min_distance_to_green = float('inf')
            for apple_x, apple_y in self.green_apples:
                distance = abs(new_x - apple_x) + abs(new_y - apple_y)  # Manhattan distance
                min_distance_to_green = min(min_distance_to_green, distance)
            
            # Recompensa por estar cerca de una manzana verde
            if min_distance_to_green < self.board_size:
                reward = 1.0 / (min_distance_to_green + 1)  # Más cerca = mayor recompensa
            else:
                reward = -0.1  # Small negative reward for just moving
        
        # Extra reward for long snake
        if len(self.snake) > 10:
            reward += 5  # Bonus for achieving target length
        
        # Check if snake length is 0
        if not self.snake:
            return -100, 0, True  # Mayor penalización por perder completamente
        
        return reward, len(self.snake), False
    
    def get_board_state(self):
        """Return the current state of the board for the interpreter."""
        return {
            'board': self.board.copy(),
            'snake': self.snake.copy(),
            'direction': self.direction,
            'green_apples': self.green_apples.copy(),
            'red_apple': self.red_apple
        }
    
    def render(self):
        """Render the current state of the game using pygame."""
        if not self.visual_mode:
            return
        
        # Clear the screen
        self.screen.fill((0, 0, 0))
        
        # Draw grid
        for x in range(self.board_size):
            for y in range(self.board_size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                  self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)
        
        # Draw snake
        for i, (x, y) in enumerate(self.snake):
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                              self.cell_size, self.cell_size)
            if i == 0:  # Head
                pygame.draw.rect(self.screen, (0, 100, 255), rect)  # Light blue for head
            else:  # Body
                pygame.draw.rect(self.screen, (0, 0, 255), rect)  # Blue for body
        
        # Draw green apples
        for x, y in self.green_apples:
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                              self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (0, 255, 0), rect)  # Green
        
        # Draw red apple
        if self.red_apple:
            x, y = self.red_apple
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                              self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (255, 0, 0), rect)  # Red
        
        # Update display
        pygame.display.flip()
        
        # Process events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)