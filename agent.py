#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import pickle
import numpy as np
from interpreter import Interpreter

class Agent:
    def __init__(self, learning_disabled=False):
        self.interpreter = Interpreter()
        self.learning_disabled = learning_disabled
        
        # Q-learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.9
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.1
        
        # Initialize Q-table as a dictionary
        self.q_table = {}
        
        # Number of possible actions
        self.num_actions = 4  # 0: up, 1: right, 2: down, 3: left
    
    def get_q_value(self, state, action):
        """Get Q-value for a state-action pair, or initialize if not present."""
        compact_state = self.interpreter.get_compact_state(state)
        
        if compact_state not in self.q_table:
            self.q_table[compact_state] = np.zeros(self.num_actions)
        
        return self.q_table[compact_state][action]
    
    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy."""
        compact_state = self.interpreter.get_compact_state(state)
        
        # Initialize Q-values for this state if not present
        if compact_state not in self.q_table:
            self.q_table[compact_state] = np.zeros(self.num_actions)
        
        # Exploration: choose a random action
        if not self.learning_disabled and random.random() < self.exploration_rate:
            return random.randint(0, self.num_actions - 1)
        
        # Exploitation: choose the best action (with random tie-breaking)
        q_values = self.q_table[compact_state]
        best_actions = np.where(q_values == np.max(q_values))[0]
        return random.choice(best_actions)
    
    def learn(self, state, action, reward, next_state, done):
        """Update Q-value using the Q-learning update rule."""
        if self.learning_disabled:
            return
        
        # Get compact state representations
        compact_state = self.interpreter.get_compact_state(state)
        compact_next_state = self.interpreter.get_compact_state(next_state)
        
        # Initialize Q-values for next state if not present
        if compact_next_state not in self.q_table:
            self.q_table[compact_next_state] = np.zeros(self.num_actions)
        
        # Get current Q-value
        current_q = self.q_table[compact_state][action]
        
        # Calculate maximum Q-value for next state
        max_next_q = np.max(self.q_table[compact_next_state]) if not done else 0
        
        # Calculate target Q-value
        target_q = reward + self.discount_factor * max_next_q
        
        # Update Q-value
        self.q_table[compact_state][action] = current_q + self.learning_rate * (target_q - current_q)
        
        # Decay exploration rate
        self.exploration_rate = max(self.min_exploration_rate, 
                                   self.exploration_rate * self.exploration_decay)
    
    def save_model(self, filepath):
        """Save the Q-table to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'exploration_rate': self.exploration_rate
            }, f)
    
    def load_model(self, filepath):
        """Load the Q-table from a file."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.exploration_rate = data['exploration_rate']
        except Exception as e:
            print(f"Error loading model: {e}")
            