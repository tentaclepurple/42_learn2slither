import argparse
import os
import pygame
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from IPython import display

# Import original modules
from game import SnakeGameAI, Direction, Point, SPEED
from model import Linear_QNet, QTrainer
from agent import Agent, train as original_train
from helper import plot

def parse_args():
    """
    Parse command line arguments for the Snake AI training.
    """
    parser = argparse.ArgumentParser(description='Train an AI to play Snake with customized parameters')
    parser.add_argument('-save', type=str, help='Path to save the model')
    parser.add_argument('-load', type=str, help='Path to load the model')
    parser.add_argument('-speed', type=float, default=SPEED, 
                       help='Game speed (0 for maximum speed, 0.1+ for slower speeds)')
    parser.add_argument('-sessions', type=int, default=0, 
                       help='Number of game sessions (0 for infinite)')
    parser.add_argument('-visual', choices=['on', 'off'], default='on', 
                       help='Enable or disable visualization')
    parser.add_argument('-dontlearn', action='store_true', 
                       help='Disable learning (only play with current model)')
    parser.add_argument('-step', action='store_true', 
                       help='Step by step execution (press any key to continue)')
    
    return parser.parse_args()

# Extend SnakeGameAI class for visual and step mode
class ExtendedSnakeGameAI(SnakeGameAI):
    def __init__(self, w=640, h=480, visual='on'):
        self.visual_enabled = visual == 'on'
        super().__init__(w, h)
        
        # Override display if visual is off
        if not self.visual_enabled:
            self.display = pygame.Surface((w, h))
    
    def _update_ui(self):
        """
        Update the game UI but only display if visual is enabled
        """
        super()._update_ui()
        if not self.visual_enabled:
            # Skip display update
            pass
    
    def play_step(self, action, step_mode=False):
        """
        Extended play_step that supports step-by-step execution
        """
        self.frame_iteration += 1
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # Step mode - wait for keypress
        if step_mode and self.visual_enabled:
            print("Press any key to continue to next step...")
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                    elif event.type == pygame.KEYDOWN:
                        waiting = False
                        break
                pygame.display.flip()
                self.clock.tick(5)
        
        # Get current state before moving
        old_head = self.head
        old_food_visible = self.is_food_visible()
        old_distance = abs(old_head.x - self.food.x) + abs(old_head.y - self.food.y)
        
        # Move
        self._move(action)
        self.snake.insert(0, self.head)
        
        # Check for game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
            
        # Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            # New state after moving
            new_food_visible = self.is_food_visible()
            new_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
            
            # If food is visible
            if old_food_visible or new_food_visible:
                # Reward for getting closer, penalty for moving away
                if new_distance < old_distance:
                    reward = 5  # Increased reward for moving toward food in line of sight
                elif new_distance > old_distance:
                    reward = -1  # Penalty for moving away from visible food
            
            self.snake.pop()
        
        # Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)
        
        return reward, game_over, self.score
    
    def is_food_visible(self, direction=None):
        """
        Check if food is visible from snake's head in a straight line
        in the specified direction. If no direction is specified, check all.
        """
        head = self.head
        food = self.food
        
        if direction is None:
            return (self.is_food_visible(Direction.LEFT) or
                    self.is_food_visible(Direction.RIGHT) or
                    self.is_food_visible(Direction.UP) or
                    self.is_food_visible(Direction.DOWN))
        
        # Check if food is in straight line in the given direction
        if direction == Direction.RIGHT:
            if head.y == food.y and head.x < food.x:
                # Check if any body part blocks vision
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

# Extended Agent class with loading/saving functionality
class ExtendedAgent(Agent):
    def __init__(self, load_path=None, dont_learn=False):
        super().__init__()
        self.dont_learn = dont_learn
        
        # Load model if path provided
        if load_path:
            self.load_model(load_path)
            # Reduce randomness for a loaded model
            self.epsilon = 20
    
    def load_model(self, path):
        """
        Load model from specified path
        """
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            print(f"Model loaded from {path}")
            return True
        print(f"No model found at {path}")
        return False
    
    def save_model(self, path):
        """
        Save model to specified path
        """
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def get_state(self, game):
        """
        Get state with visibility information
        """
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Check if food is visible in each direction
        food_visible_left = game.is_food_visible(Direction.LEFT)
        food_visible_right = game.is_food_visible(Direction.RIGHT)
        food_visible_up = game.is_food_visible(Direction.UP)
        food_visible_down = game.is_food_visible(Direction.DOWN)

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food visible in line of sight (replaces absolute location)
            food_visible_left,
            food_visible_right,
            food_visible_up,
            food_visible_down
            ]

        return np.array(state, dtype=int)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        """Only train if learning is enabled"""
        if not self.dont_learn:
            super().train_short_memory(state, action, reward, next_state, done)
    
    def train_long_memory(self):
        """Only train if learning is enabled"""
        if not self.dont_learn:
            super().train_long_memory()

def train_cli():
    """
    Train function for the command line interface with all options
    """
    args = parse_args()
    
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = ExtendedAgent(load_path=args.load, dont_learn=args.dontlearn)
    game = ExtendedSnakeGameAI(visual=args.visual)
    
    session_count = 0
    
    print("Starting Snake AI with the following settings:")
    print(f"  - Save model to: {args.save if args.save else 'Disabled'}")
    print(f"  - Load model from: {args.load if args.load else 'None (starting fresh)'}")
    print(f"  - Game speed: {args.speed}")
    print(f"  - Session limit: {args.sessions if args.sessions > 0 else 'Infinite'}")
    print(f"  - Visualization: {args.visual}")
    print(f"  - Learning: {'Disabled' if args.dontlearn else 'Enabled'}")
    print(f"  - Step-by-step mode: {'Enabled' if args.step else 'Disabled'}")
    print("Press Ctrl+C to stop the training at any time.")
    
    try:
        while True:
            # Check if we've reached the session limit
            if args.sessions > 0 and session_count >= args.sessions:
                print(f"Completed {args.sessions} sessions as requested.")
                break
            
            # Get old state
            state_old = agent.get_state(game)
            
            # Get move
            final_move = agent.get_action(state_old)
            
            # Perform move and get new state
            reward, done, score = game.play_step(final_move, step_mode=args.step)
            state_new = agent.get_state(game)
            
            # Train
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)
            
            if done:
                # Reset game and train long memory
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()
                
                # Save if we beat the record
                if score > record:
                    record = score
                    if args.save:
                        agent.save_model(args.save)
                        print(f"New record! Model saved to {args.save}")
                
                print(f'Game {agent.n_games}, Score {score}, Record: {record}')
                
                # Update plots
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                
                if args.visual == 'on':
                    try:
                        plot(plot_scores, plot_mean_scores)
                    except:
                        # If plotting fails, continue without it
                        pass
                
                session_count += 1
            
            # Control game speed
            if args.speed > 0:
                game.clock.tick(args.speed)
            else:
                # Run as fast as possible
                game.clock.tick()
                
    except KeyboardInterrupt:
        print("\nTraining stopped by user.")
        if args.save:
            agent.save_model(args.save)
            print(f"Final model saved to {args.save}")
    
    print(f"Training completed. Final record: {record}")

if __name__ == "__main__":
    train_cli()