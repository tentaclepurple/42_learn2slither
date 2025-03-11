# agent_pytorch.py
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class SnakeModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        """
        Guarda el modelo en el archivo especificado.
        No añade directorios adicionales.
        """
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # Añadir dimensión de lote si tenemos un solo ejemplo
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # Predicción de valores Q con estado actual
        pred = self.model(state)

        # Objetivo para el entrenamiento
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # Solo actualizamos el valor Q de la acción tomada
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # Entrenamiento
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


class SnakeAgent:
    def __init__(self, state_size=40, hidden_size=256, verbose=False):
        self.n_games = 0
        self.epsilon = 1.0  # Exploración inicial
        self.min_epsilon = 0.01  # Exploración mínima
        self.epsilon_decay = 0.995  # Tasa de decaimiento
        self.gamma = 0.9  # Factor de descuento
        self.memory = deque(maxlen=MAX_MEMORY)
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.action_size = len(self.actions)
        self.verbose = verbose
        self.dontlearn_enabled = True
        self.snake_length = 3
        self.max_snake_length = 3
        self.state_size = state_size
        
        # Crear modelo y entrenador
        self.model = SnakeModel(state_size, hidden_size, self.action_size)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        # Mapeo de caracteres a valores numéricos para procesamiento
        self.char_to_value = {
            'W': -1.0,   # Pared
            'H': 0.5,   # Cabeza de serpiente
            'S': -0.9,   # Segmento de cuerpo
            'G': 1.0,   # Manzana verde
            'R': -0.8,  # Manzana roja
            '0': 0.1    # Espacio vacío
        }
        
        # Cache para estados
        self.state_cache = {}
    
    def _preprocess_state(self, state_str):
        """
        Convierte el estado en formato de cadena a un vector numérico
        """
        if state_str in self.state_cache:
            return self.state_cache[state_str]
            
        # Dividir estado por direcciones
        directions = state_str.split(',')
        state_vector = []
        
        # Procesar cada dirección (hasta 10 celdas)
        for direction in directions:
            for char in direction[:10]:  # Limitar a 10 celdas por dirección
                state_vector.append(self.char_to_value.get(char, 0.0))
        
        # Asegurar longitud fija del vector
        if len(state_vector) < self.state_size:
            state_vector.extend([0.0] * (self.state_size - len(state_vector)))
        elif len(state_vector) > self.state_size:
            state_vector = state_vector[:self.state_size]
        
        # Guardar en caché y devolver
        result = np.array(state_vector)
        self.state_cache[state_str] = result
        return result
    
    def dontlearn(self):
        """Desactiva el aprendizaje y utiliza solo explotación"""
        self.dontlearn_enabled = False
        self.epsilon = 0  # Sin exploración
    
    def remember(self, state, action, reward, next_state, done):
        """Almacena experiencia en la memoria"""
        action_idx = self.actions.index(action)
        action_one_hot = [0] * self.action_size
        action_one_hot[action_idx] = 1
        
        # Guardar experiencia en la memoria
        self.memory.append((state, action_one_hot, reward, next_state, done))
    
    def train_long_memory(self):
        """Entrena con un lote de experiencias de la memoria"""
        if len(self.memory) < BATCH_SIZE:
            mini_batch = self.memory
        else:
            mini_batch = random.sample(self.memory, BATCH_SIZE)
        
        states, actions, rewards, next_states, dones = zip(*mini_batch)
        
        # Preprocesar todos los estados juntos
        processed_states = [self._preprocess_state(s) for s in states]
        processed_next_states = [self._preprocess_state(s) for s in next_states]
        
        self.trainer.train_step(processed_states, actions, rewards, processed_next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        """Entrena con una sola experiencia"""
        action_idx = self.actions.index(action)
        action_one_hot = [0] * self.action_size
        action_one_hot[action_idx] = 1
        
        # Preprocesar estados
        state_processed = self._preprocess_state(state)
        next_state_processed = self._preprocess_state(next_state)
        
        self.trainer.train_step(state_processed, [action_one_hot], reward, next_state_processed, done)
    
    def learn(self, state, action, reward, next_state):
        """
        Interfaz unificada para aprendizaje
        """
        if not self.dontlearn_enabled:
            return
        
        done = reward <= -20  # Asumimos que penalizaciones grandes indican fin
        
        # Entrenamiento de memoria corta
        self.train_short_memory(state, action, reward, next_state, done)
        
        # Guardar experiencia
        self.remember(state, action, reward, next_state, done)
    
    def choose_action(self, state, training=True):
        """
        Selecciona acción usando política epsilon-greedy
        """
        # Exploración vs explotación
        if training and random.random() < self.epsilon:
            # Exploración: acción aleatoria
            return random.choice(self.actions)
        
        # Explotación: usar modelo para predecir mejor acción
        state_processed = self._preprocess_state(state)
        state_tensor = torch.tensor(state_processed, dtype=torch.float).unsqueeze(0)
        
        with torch.no_grad():
            prediction = self.model(state_tensor)
        
        action_idx = torch.argmax(prediction).item()
        return self.actions[action_idx]
    
    def update_snake_length(self, new_length):
        """Actualiza la longitud de la serpiente"""
        self.snake_length = new_length
        if new_length > self.max_snake_length:
            self.max_snake_length = new_length
    
    def decay_exploration(self):
        """Reduce la tasa de exploración después de cada episodio"""
        self.n_games += 1
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        if self.verbose and self.n_games % 10 == 0:
            print(f"Juego {self.n_games}, Epsilon: {self.epsilon:.4f}")
    
    def save_model(self, filepath):
        """Guarda el modelo con método alternativo"""
        try:
            # Convertir a formato numpy antes de guardar
            import numpy as np
            state_dict = self.model.state_dict()
            numpy_dict = {k: v.cpu().numpy() for k, v in state_dict.items()}
            np.save(filepath, numpy_dict)
            
            # Guardar metadatos en formato JSON
            with open(f"{filepath}_meta.json", "w") as f:
                import json
                json.dump({
                    "n_games": self.n_games,
                    "epsilon": self.epsilon,
                    "max_snake_length": self.max_snake_length
                }, f)
            
            print(f"Modelo guardado en formato numpy: {filepath}.npy")
        except Exception as e:
            print(f"Error al guardar: {e}")
    
    def load_model(self, filepath):
        """Carga modelo desde disco con manejo mejorado de extensiones"""
        try:
            # Determinar si trabajamos con formato npy
            if filepath.endswith(".npy"):
                # Cargar desde formato numpy
                import numpy as np
                numpy_dict = np.load(filepath, allow_pickle=True).item()
                
                # Convertir de numpy a tensores
                state_dict = {k: torch.tensor(v) for k, v in numpy_dict.items()}
                self.model.load_state_dict(state_dict)
                
                # Intentar cargar metadatos
                metadata_path = filepath.replace(".npy", "_meta.json")
            else:
                # Asegurar extensión correcta para PyTorch
                if not filepath.endswith(".pth"):
                    filepath += ".pth"
                    
                # Cargar modelo PyTorch estándar
                state_dict = torch.load(filepath)
                self.model.load_state_dict(state_dict)
                
                # Ruta para metadatos
                metadata_path = filepath.replace(".pth", ".meta")
            
            # Cargar metadatos si existen
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    import json
                    data = json.load(f)
                    self.n_games = data.get("n_games", 0)
                    self.epsilon = data.get("epsilon", 0.1)
                    self.max_snake_length = data.get("max_snake_length", 3)
            
            if self.verbose:
                print(f"Modelo cargado desde {filepath}")
                
        except Exception as e:
            print(f"Error al cargar modelo: {e}")
    
    def reset_session(self):
        """Reinicia para una nueva sesión de entrenamiento"""
        self.snake_length = 3
        self.state_cache = {}  # Limpiar caché de estados