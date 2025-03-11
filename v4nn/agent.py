# agent_optimized.py
import torch
import random
import numpy as np
from collections import deque
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class SnakeQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name):
        model_folder_path = os.path.dirname(file_name)
        if model_folder_path and not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        torch.save(self.state_dict(), file_name)


class OptimizedAgent:
    def __init__(self, verbose=False):
        self.n_games = 0
        self.epsilon = 80  # Exploración inicial alta
        self.min_epsilon = 0  # Mínimo de exploración
        self.gamma = 0.9  # Factor de descuento
        self.memory = deque(maxlen=MAX_MEMORY)
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]  # Acciones absolutas disponibles
        self.action_size = 3  # [recto, derecha, izquierda] - acciones relativas
        self.verbose = verbose
        self.snake_length = 3
        self.max_snake_length = 3
        self.last_action = None  # Última acción tomada
        
        # Mapeo de caracteres para procesamiento eficiente
        self.char_to_idx = {
            'W': 0,   # Pared
            'S': 0,   # Segmento de cuerpo (igual que pared = peligro)
            'G': 1,   # Manzana verde
            'R': 2,   # Manzana roja
            '0': 3,   # Espacio vacío
            'H': 4    # Cabeza (no debería aparecer en la visión)
        }
        
        # Crear modelo más eficiente (11 entradas, 256 neuronas ocultas, 3 salidas)
        self.model = SnakeQNet(11, 256, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
    
    def get_state(self, state_str):
        """
        Convierte la visión de la serpiente en un estado compacto similar al original
        """
        directions = state_str.split(',')
        
        # Cada dirección representa: [UP, DOWN, LEFT, RIGHT]
        # Extraer información importante de cada dirección
        
        # 1. Detectar peligro (pared o cuerpo) en cada dirección
        danger_straight = '0' not in directions[0][:1] or 'W' in directions[0][:1] or 'S' in directions[0][:1]
        danger_down = '0' not in directions[1][:1] or 'W' in directions[1][:1] or 'S' in directions[1][:1]
        danger_left = '0' not in directions[2][:1] or 'W' in directions[2][:1] or 'S' in directions[2][:1] 
        danger_right = '0' not in directions[3][:1] or 'W' in directions[3][:1] or 'S' in directions[3][:1]
        
        # 2. Determinar dirección actual basada en la última acción
        dir_up = self.last_action == "UP" if self.last_action else False
        dir_down = self.last_action == "DOWN" if self.last_action else True  # Dirección inicial
        dir_left = self.last_action == "LEFT" if self.last_action else False
        dir_right = self.last_action == "RIGHT" if self.last_action else False
        
        # 3. Detectar comida en la visión
        food_up = 'G' in directions[0]
        food_down = 'G' in directions[1]
        food_left = 'G' in directions[2]
        food_right = 'G' in directions[3]
        
        # Adaptamos el formato para imitar el enfoque original
        # Peligros relativos a la dirección actual
        danger_front = (dir_up and danger_straight) or (dir_down and danger_down) or \
                       (dir_left and danger_left) or (dir_right and danger_right)
                       
        danger_right = (dir_up and danger_right) or (dir_down and danger_left) or \
                      (dir_left and danger_straight) or (dir_right and danger_down)
                      
        danger_left = (dir_up and danger_left) or (dir_down and danger_right) or \
                     (dir_left and danger_down) or (dir_right and danger_straight)
        
        # Comida relativa a la posición actual
        food_front = (dir_up and food_up) or (dir_down and food_down) or \
                    (dir_left and food_left) or (dir_right and food_right)
                    
        food_right = (dir_up and food_right) or (dir_down and food_left) or \
                    (dir_left and food_up) or (dir_right and food_down)
                    
        food_left = (dir_up and food_left) or (dir_down and food_right) or \
                   (dir_left and food_down) or (dir_right and food_up)
        
        # Estado final compacto (11 valores binarios)
        state = [
            # Peligro en 3 direcciones relativas
            danger_front,
            danger_right,
            danger_left,
            
            # Dirección actual
            dir_left,
            dir_right,
            dir_up,
            dir_down,
            
            # Comida en la visión (relativa)
            food_left,
            food_right, 
            food_up,
            food_down
        ]
        
        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done):
        """Almacena experiencia en memoria"""
        # Convertir acción al formato [recto, derecha, izquierda]
        action_idx = self.get_relative_action_idx(action)
        action_one_hot = [0] * self.action_size
        action_one_hot[action_idx] = 1
        
        self.memory.append((state, action_one_hot, reward, next_state, done))
    
    def get_relative_action_idx(self, action):
        """Convierte acción absoluta a índice relativo [recto, derecha, izquierda]"""
        if not self.last_action:
            # Primera acción, cualquier dirección es válida
            return 0  # Por defecto "recto"
        
        # Mapeo direccional
        directions = ["UP", "RIGHT", "DOWN", "LEFT"]
        current_idx = directions.index(self.last_action) if self.last_action in directions else 0
        action_idx = directions.index(action) if action in directions else 0
        
        # Calcular diferencia relativa
        diff = (action_idx - current_idx) % 4
        
        # Convertir a [recto, derecha, izquierda]
        if diff == 0:  # Misma dirección = recto
            return 0
        elif diff == 1:  # 90° derecha
            return 1
        else:  # diff == 3 (270° derecha = 90° izquierda) o diff == 2 (180° vuelta)
            return 2  # Tomamos izquierda para vuelta de 180° también
    
    def get_absolute_action(self, relative_action):
        """Convierte acción relativa [recto, derecha, izquierda] a absoluta"""
        if not self.last_action:
            # Primera acción o sin historial, usar DOWN como default
            self.last_action = "DOWN"
        
        # Mapeo direccional
        directions = ["UP", "RIGHT", "DOWN", "LEFT"]
        current_idx = directions.index(self.last_action)
        
        # Calcular nueva dirección basada en acción relativa
        if relative_action == 0:  # Recto
            new_idx = current_idx
        elif relative_action == 1:  # Derecha
            new_idx = (current_idx + 1) % 4
        else:  # Izquierda
            new_idx = (current_idx - 1) % 4
        
        return directions[new_idx]
    
    def train_long_memory(self):
        """Entrena con experiencias almacenadas en memoria"""
        if len(self.memory) < BATCH_SIZE:
            mini_batch = self.memory
        else:
            mini_batch = random.sample(self.memory, BATCH_SIZE)
        
        states, actions, rewards, next_states, dones = zip(*mini_batch)
        self.train_step(states, actions, rewards, next_states, dones)
    
    def train_step(self, state, action, reward, next_state, done):
        """Implementa un paso de entrenamiento Q-learning"""
        # Convertir a tensores
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        
        # Manejar dimensiones para lotes o ejemplos individuales
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)
        
        # Predicción de valores Q actuales
        pred = self.model(state)
        
        # Crear target clonando la predicción
        target = pred.clone()
        
        # Actualizar target usando ecuación de Bellman
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            target[idx][torch.argmax(action[idx])] = Q_new
        
        # Optimización
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
    
    def train_short_memory(self, state, action, reward, next_state, done):
        """Entrena con una sola transición"""
        self.train_step(state, action, reward, next_state, done)
    
    def choose_action(self, state, training=True):
        """Selecciona acción usando política epsilon-greedy"""
        # Actualizar epsilon basado en juegos jugados
        self.epsilon = max(self.min_epsilon, 80 - self.n_games)
        
        # Exploración vs explotación
        final_move_idx = 0  # Por defecto, seguir recto
        
        if training and random.randint(0, 200) < self.epsilon:
            # Exploración: acción aleatoria relativa
            final_move_idx = random.randint(0, 2)  # [recto, derecha, izquierda]
        else:
            # Explotación: usar modelo
            # Primero convertir el estado a formato numérico
            state_array = self.get_state(state)
            state_tensor = torch.tensor(state_array, dtype=torch.float).unsqueeze(0)
            prediction = self.model(state_tensor)
            final_move_idx = torch.argmax(prediction).item()
        
        # Convertir índice relativo a acción absoluta
        action = self.get_absolute_action(final_move_idx)
        
        # Actualizar última acción
        self.last_action = action
        
        return action
    
    def update_snake_length(self, new_length):
        """Actualiza longitud de la serpiente"""
        self.snake_length = new_length
        if new_length > self.max_snake_length:
            self.max_snake_length = new_length
    
    def save_model(self, filepath):
        """Guarda el modelo en disco"""
        if not filepath.endswith(".pth"):
            filepath += ".pth"
        
        try:
            self.model.save(filepath)
            
            # Guardar metadatos
            metadata_path = filepath.replace(".pth", ".meta")
            with open(metadata_path, "w") as f:
                import json
                json.dump({
                    "n_games": self.n_games,
                    "epsilon": self.epsilon,
                    "max_snake_length": self.max_snake_length
                }, f)
            
            if self.verbose:
                print(f"Modelo guardado en {filepath}")
        except Exception as e:
            print(f"Error al guardar: {e}")
    
    def load_model(self, filepath):
        """Carga modelo desde disco"""
        if not filepath.endswith(".pth"):
            filepath += ".pth"
        
        try:
            state_dict = torch.load(filepath)
            self.model.load_state_dict(state_dict)
            
            # Cargar metadatos
            metadata_path = filepath.replace(".pth", ".meta")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    import json
                    data = json.load(f)
                    self.n_games = data.get("n_games", 0)
                    self.epsilon = data.get("epsilon", 80 - self.n_games)
                    self.max_snake_length = data.get("max_snake_length", 3)
            
            if self.verbose:
                print(f"Modelo cargado desde {filepath}")
        except Exception as e:
            print(f"Error al cargar modelo: {e}")
    
    def learn(self, state, action, reward, next_state):
        """Interfaz unificada para aprendizaje"""
        done = reward <= -10  # Penalización grande indica fin
        
        # Procesar estados
        state_array = self.get_state(state)
        next_state_array = self.get_state(next_state)
        
        # Convertir acción a formato one-hot
        action_idx = self.get_relative_action_idx(action)
        action_one_hot = [0] * self.action_size
        action_one_hot[action_idx] = 1
        
        # Entrenamiento de memoria corta
        self.train_short_memory(state_array, action_one_hot, reward, next_state_array, done)
        
        # Guardar experiencia
        self.remember(state_array, action, reward, next_state_array, done)
    
    def decay_exploration(self):
        """Actualiza contadores y exploración"""
        self.n_games += 1
        # No necesitamos decaimiento manual ya que epsilon se calcula en cada acción
    
    def reset_session(self):
        """Reinicia para nueva sesión"""
        self.snake_length = 3
        self.last_action = None