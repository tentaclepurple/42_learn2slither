#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Interpreter:
    def __init__(self):
        # Define symbols for state representation
        self.WALL = 'W'
        self.HEAD = 'H'
        self.SNAKE = 'S'
        self.GREEN_APPLE = 'G'
        self.RED_APPLE = 'R'
        self.EMPTY = '0'
    
    def get_state_representation(self, board_state):
        """
        Convert the board state to a snake vision representation.
        Return a list of strings representing what the snake sees in the 4 directions.
        """
        board = board_state['board']
        snake = board_state['snake']
        direction = board_state['direction']
        
        # Get snake head coordinates
        head_x, head_y = snake[0]
        
        # Define the 4 vision directions (relative to the snake's orientation)
        # Each is a tuple of (dx, dy) for vision line
        vision_directions = [
            (0, -1),  # Up
            (1, 0),   # Right
            (0, 1),   # Down
            (-1, 0)   # Left
        ]
        
        # Generate vision strings for each direction
        vision = []
        board_size = board.shape[0]
        
        for dx, dy in vision_directions:
            vision_line = ""
            x, y = head_x, head_y
            
            # Look in this direction until we hit a wall
            while True:
                x += dx
                y += dy
                
                # Check if we hit a wall
                if x < 0 or x >= board_size or y < 0 or y >= board_size:
                    vision_line += self.WALL
                    break
                
                # Get what's in this cell
                cell_value = board[y, x]
                
                if cell_value == 0:  # Empty
                    vision_line += self.EMPTY
                elif cell_value == 1:  # Snake body
                    vision_line += self.SNAKE
                elif cell_value == 2:  # Green apple
                    vision_line += self.GREEN_APPLE
                elif cell_value == 3:  # Red apple
                    vision_line += self.RED_APPLE
            
            vision.append(vision_line)
        
        # Format the vision strings into the required display format
        up_vision = vision[0]
        right_vision = vision[1]
        down_vision = vision[2]
        left_vision = vision[3]
        
        # Create the middle part (horizontal vision)
        middle = left_vision + self.HEAD + right_vision
        
        # Construct the final representation
        state_representation = []
        
        # Add lines for up vision
        for char in up_vision:
            state_representation.append(' ' * (len(left_vision)) + char)
        
        # Add the middle line
        state_representation.append(middle)
        
        # Add lines for down vision
        for char in down_vision:
            state_representation.append(' ' * (len(left_vision)) + char)
        
        return state_representation
    
    def get_compact_state(self, state_representation):
        """
        Convert the visual state representation to a compact string format
        that can be used as a key in the Q-table.
        """
        # Join all the lines into a single string
        return '|'.join(state_representation)