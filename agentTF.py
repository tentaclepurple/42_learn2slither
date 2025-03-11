#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import os

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self, learning_disabled=False):
        # Parámetros de aprendizaje
        self.learning_disabled = learning_disabled
        self.learning_rate = 0.001
        self.discount_factor = 0.99
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01
        self.replay_memory_size = 10000
        self.batch_size = 64
        self.update_target_frequency = 10
        
        # Tamaño del estado y número de acciones
        self.input_size = 32  # Codificación del estado
        self.hidden_size = 128
        self.num_actions = 4  # up, right, down, left
        
        # Memoria de experiencia
        self.memory = deque(maxlen=self.replay_memory_size)
        
        # Crear redes neuronales
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.input_size, self.hidden_size, self.num_actions).to(self.device)
        self.target_net = DQN(self.input_size, self.hidden_size, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizador
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Contador de pasos para actualizar la red objetivo
        self.steps_done = 0
        
    def state_to_tensor(self, state):
        """Convierte el estado (representación de visión) en un tensor para la red neuronal"""
        # Codificación one-hot de cada celda en cada dirección
        encoding = []
        
        # Mapa de caracteres a índices
        char_to_idx = {
            'W': 0,  # Wall
            'H': 1,  # Head
            'S': 2,  # Snake body
            'G': 3,  # Green apple
            'R': 4,  # Red apple
            '0': 5   # Empty
        }
        
        # Para cada línea en la representación de estado
        for line in state:
            for char in line:
                if char in char_to_idx:
                    # One-hot encoding (6 posibles valores)
                    one_hot = [0] * 6
                    one_hot[char_to_idx[char]] = 1
                    encoding.extend(one_hot)
                elif char == ' ':
                    # Ignorar espacios
                    continue
                else:
                    # Si hay un carácter desconocido, usar vector de ceros
                    encoding.extend([0] * 6)
        
        # Asegurar que el tensor tenga el tamaño correcto
        # Rellenar o truncar si es necesario
        if len(encoding) < self.input_size:
            encoding.extend([0] * (self.input_size - len(encoding)))
        else:
            encoding = encoding[:self.input_size]
            
        return torch.FloatTensor(encoding).unsqueeze(0).to(self.device)
        
    def choose_action(self, state):
        """Selecciona una acción usando la política epsilon-greedy"""
        if self.learning_disabled:
            self.exploration_rate = 0.0
            
        # Exploración: elegir acción aleatoria
        if random.random() < self.exploration_rate:
            return random.randint(0, self.num_actions - 1)
        
        # Explotación: elegir la mejor acción según la red
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state)
            q_values = self.policy_net(state_tensor)
            return q_values.max(1)[1].item()
    
    def remember(self, state, action, reward, next_state, done):
        """Almacena la experiencia en la memoria de repetición"""
        self.memory.append((state, action, reward, next_state, done))
    
    def learn(self, state, action, reward, next_state, done):
        """Realiza el aprendizaje con DQN"""
        if self.learning_disabled:
            return
            
        # Guardar experiencia en memoria
        self.remember(state, action, reward, next_state, done)
        
        # Realizar entrenamiento solo si hay suficientes muestras
        if len(self.memory) < self.batch_size:
            # Actualizar tasa de exploración
            self.exploration_rate = max(self.min_exploration_rate, 
                                     self.exploration_rate * self.exploration_decay)
            return
            
        # Obtener minibatch aleatorio
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for exp in minibatch:
            states.append(self.state_to_tensor(exp[0]))
            actions.append(exp[1])
            rewards.append(exp[2])
            next_states.append(self.state_to_tensor(exp[3]))
            dones.append(exp[4])
        
        # Convertir a tensores
        states = torch.cat(states)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states = torch.cat(next_states)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        
        # Calcular valores Q actuales
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Calcular valores Q objetivo
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            next_q_values[dones] = 0.0  # Si el episodio termina, no hay recompensa futura
            target_q_values = rewards + self.discount_factor * next_q_values
        
        # Calcular pérdida y optimizar
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        # Limitar gradientes para evitar explosión
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Actualizar la red objetivo periódicamente
        self.steps_done += 1
        if self.steps_done % self.update_target_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Actualizar tasa de exploración
        self.exploration_rate = max(self.min_exploration_rate, 
                                 self.exploration_rate * self.exploration_decay)
    
    def save_model(self, filepath):
        """Guarda el modelo y parámetros de entrenamiento"""
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        save_data = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'exploration_rate': self.exploration_rate,
            'steps_done': self.steps_done
        }
        torch.save(save_data, filepath)
        
    def load_model(self, filepath):
        """Carga el modelo y parámetros de entrenamiento"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.exploration_rate = checkpoint['exploration_rate']
            self.steps_done = checkpoint['steps_done']
            
            # Asegurar que la red de política esté en modo entrenamiento
            self.policy_net.train()
            # Asegurar que la red objetivo esté en modo evaluación
            self.target_net.eval()
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")