#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
import curses
import traceback
from environment import Environment
from agent import Agent
from interpreter import Interpreter

def parse_arguments():
    parser = argparse.ArgumentParser(description='Learn2Slither - Snake AI with Q-Learning')
    parser.add_argument('-sessions', type=int, default=10, help='Number of training sessions')
    parser.add_argument('-save', type=str, default=None, help='Path to save the model')
    parser.add_argument('-load', type=str, default=None, help='Path to load a model')
    parser.add_argument('-visual', choices=['on', 'off'], default='on', help='Enable/disable visual mode')
    parser.add_argument('-dontlearn', action='store_true', help='Disable learning (for evaluation)')
    parser.add_argument('-step-by-step', action='store_true', help='Enable step-by-step mode')
    parser.add_argument('-speed', type=float, default=0.1, help='Speed of visualization (lower is faster, 0 for max speed)')
    return parser.parse_args()

def run_with_curses(stdscr, args):
    # Hide cursor
    curses.curs_set(0)
    
    # Initialize colors
    if curses.has_colors():
        curses.start_color()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)  # Default
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Green apple
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)    # Red apple
        curses.init_pair(4, curses.COLOR_BLUE, curses.COLOR_BLACK)   # Snake
        curses.init_pair(5, curses.COLOR_YELLOW, curses.COLOR_BLACK) # Info
    
    # Get screen dimensions
    max_y, max_x = stdscr.getmaxyx()
    
    # Create models directory if it doesn't exist
    if args.save and not os.path.exists(os.path.dirname(args.save)):
        os.makedirs(os.path.dirname(args.save))
    
    # Initialize components
    environment = Environment(visual_mode=(args.visual == 'on'))
    interpreter = Interpreter()
    agent = Agent(learning_disabled=args.dontlearn)
    
    # Load model if specified
    if args.load:
        stdscr.addstr(0, 0, f"Load trained model from {args.load}")
        stdscr.refresh()
        agent.load_model(args.load)
    
    max_length = 0
    max_duration = 0
    
    # Training loop
    for session in range(args.sessions):
        # Reset environment for new session
        environment.reset()
        length = 3  # Initial snake length
        duration = 0
        
        # Game loop
        game_over = False
        while not game_over:
            # Clear screen
            stdscr.clear()
            
            # Display session info
            stdscr.addstr(0, 0, f"Session: {session+1}/{args.sessions} | Length: {length} | Duration: {duration}", 
                         curses.color_pair(5))
            
            # Get current state (snake's vision)
            board_state = environment.get_board_state()
            state = interpreter.get_state_representation(board_state)
            
            # Display state in terminal
            if args.visual == 'on':
                for i, line in enumerate(state):
                    stdscr.addstr(2 + i, 2, line)
            
            # Agent selects an action
            action = agent.choose_action(state)
            
            # Display action in terminal
            if args.visual == 'on':
                action_names = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
                stdscr.addstr(2 + len(state) + 1, 2, f"Action: {action_names[action]}", curses.color_pair(5))
            
            # Execute action and get reward
            reward, new_length, game_over = environment.step(action)
            length = new_length
            duration += 1
            
            # Get new state
            new_board_state = environment.get_board_state()
            new_state = interpreter.get_state_representation(new_board_state)
            
            # Agent learns from experience
            if not args.dontlearn:
                agent.learn(state, action, reward, new_state, game_over)
            
            # Display reward
            if args.visual == 'on':
                stdscr.addstr(2 + len(state) + 2, 2, f"Reward: {reward}", curses.color_pair(5))
                
                # Display model info
                stdscr.addstr(2 + len(state) + 3, 2, f"Exploration rate: {agent.exploration_rate:.2f}", 
                             curses.color_pair(5))
                
                # Draw board ASCII representation
                for y in range(environment.board_size):
                    for x in range(environment.board_size):
                        cell_x = max_x - environment.board_size * 2 - 2 + x * 2
                        cell_y = 2 + y
                        
                        # Skip if outside terminal bounds
                        if cell_x >= max_x or cell_y >= max_y:
                            continue
                            
                        value = environment.board[y, x]
                        if (x, y) == environment.snake[0]:  # Snake head
                            stdscr.addstr(cell_y, cell_x, "H", curses.color_pair(4))
                        elif value == 1:  # Snake body
                            stdscr.addstr(cell_y, cell_x, "S", curses.color_pair(4))
                        elif value == 2:  # Green apple
                            stdscr.addstr(cell_y, cell_x, "G", curses.color_pair(2))
                        elif value == 3:  # Red apple
                            stdscr.addstr(cell_y, cell_x, "R", curses.color_pair(3))
                        else:  # Empty
                            stdscr.addstr(cell_y, cell_x, ".", curses.color_pair(1))
                
                # Update pygame display if enabled
                environment.render()
                
                # Refresh screen
                stdscr.refresh()
                
                if args.step_by_step:
                    stdscr.addstr(max_y - 1, 0, "Pulsa ESPACIO para continuar (q para salir)...", curses.color_pair(5))
                    stdscr.refresh()
                    key = stdscr.getch()
                    if key == ord('q'):
                        return max_length, max_duration
                else:
                    # Pausa configurable para visualización
                    if args.speed > 0:
                        time.sleep(args.speed)
                    
                    # Comprobación no bloqueante de teclas
                    stdscr.nodelay(True)
                    key = stdscr.getch()
                    if key == ord('q'):
                        return max_length, max_duration
                    stdscr.nodelay(False)
            
            # Check for user interrupt
            stdscr.nodelay(True)
            key = stdscr.getch()
            if key == ord('q'):
                return max_length, max_duration
            stdscr.nodelay(False)
        
        # Update stats
        max_length = max(max_length, length)
        max_duration = max(max_duration, duration)
        
        # Brevemente mostrar mensaje de game over sin pausar
        stdscr.clear()
        stdscr.addstr(0, 0, f"Session {session+1}/{args.sessions} - Game Over! Length: {length}, Duration: {duration}", curses.color_pair(5))
        stdscr.addstr(1, 0, "Iniciando nueva sesión... (q para salir)", curses.color_pair(5))
        stdscr.refresh()
        
        # Pequeña pausa no bloqueante
        stdscr.nodelay(True)
        time.sleep(0.5)  # Solo medio segundo para ver el resultado
        key = stdscr.getch()
        if key == ord('q'):
            return max_length, max_duration
        stdscr.nodelay(False)
    
    # Display final stats
    stdscr.clear()
    stdscr.addstr(0, 0, f"Game over, max length = {max_length}, max duration = {max_duration}", 
                 curses.color_pair(5))
    
    # Save model if specified
    if args.save:
        stdscr.addstr(1, 0, f"Save learning state in {args.save}", curses.color_pair(5))
        agent.save_model(args.save)
    
    # Solo mostramos el mensaje 3 segundos y salimos sin esperar tecla
    stdscr.addstr(3, 0, "Terminando programa en 3 segundos...", curses.color_pair(5))
    stdscr.refresh()
    
    # Cuenta regresiva
    for i in range(3, 0, -1):
        time.sleep(1)
        stdscr.addstr(3, 0, f"Terminando programa en {i-1} segundos...", curses.color_pair(5))
        stdscr.refresh()
    
    return max_length, max_duration

def main():
    args = parse_arguments()
    
    if args.visual == 'on':
        try:
            # Run with curses interface
            max_length, max_duration = curses.wrapper(run_with_curses, args)
            print(f"Game over, max length = {max_length}, max duration = {max_duration}")
            if args.save:
                print(f"Model saved to {args.save}")
        except Exception as e:
            print(f"Error occurred: {e}")
            traceback.print_exc()
    else:
        # Run without visual feedback (training only)
        # Initialize components
        environment = Environment(visual_mode=False)
        interpreter = Interpreter()
        agent = Agent(learning_disabled=args.dontlearn)
        
        # Load model if specified
        if args.load:
            print(f"Load trained model from {args.load}")
            agent.load_model(args.load)
        
        max_length = 0
        max_duration = 0
        
        # Training loop
        for session in range(args.sessions):
            environment.reset()
            length = 3
            duration = 0
            game_over = False
            
            while not game_over:
                board_state = environment.get_board_state()
                state = interpreter.get_state_representation(board_state)
                action = agent.choose_action(state)
                reward, new_length, game_over = environment.step(action)
                length = new_length
                duration += 1
                new_board_state = environment.get_board_state()
                new_state = interpreter.get_state_representation(new_board_state)
                
                if not args.dontlearn:
                    agent.learn(state, action, reward, new_state, game_over)
            
            max_length = max(max_length, length)
            max_duration = max(max_duration, duration)
            
            print(f"Session {session+1}/{args.sessions} - Length: {length}, Duration: {duration}")
        
        print(f"Game over, max length = {max_length}, max duration = {max_duration}")
        
        if args.save:
            print(f"Save learning state in {args.save}")
            agent.save_model(args.save)

if __name__ == "__main__":
    main()