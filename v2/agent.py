# agent.py
import random
import json
import os


class QLearningAgent:
    def __init__(
        self,
        actions=["UP", "DOWN", "LEFT", "RIGHT"],
        learning_rate=0.1,
        discount_factor=0.95,
        min_exploration=0.01,
        exploration_rate=0.3,
        exploration_decay=0.995,
        verbose=False,
    ):
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.min_exploration = min_exploration
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = {}
        self.verbose = verbose
        self.dontlearn_enabled = True
        self.snake_length = 3
        self.current_position = (0, 0)
        self.position_history = []
        self.max_snake_length = 3
        self.duration = 0
        
        # Mapeo de objetos observados a sus valores de recompensa
        self.rewards = {
            "G": 10,    # Manzana verde (positivo)
            "R": -5,    # Manzana roja (negativo)
            "W": -20,   # Pared (negativo grande)
            "S": -20,   # Cuerpo de la serpiente (negativo grande)
            "0": -0.1   # Espacio vacío (pequeño negativo para fomentar movimiento)
        }

    def dontlearn(self):
        """
        Desactiva el aprendizaje y fuerza al agente a jugar
        de manera determinista basado en la tabla Q.
        """
        self.dontlearn_enabled = False
        self.exploration_rate = 0.0  # Sin exploración, solo explotación

    def learn(self, state, action, reward, next_state):
        """
        Actualiza la tabla Q basada en la recompensa recibida y el siguiente estado.
        Implementa la ecuación de Bellman para Q-learning.
        
        Parameters:
        state (str): Estado actual
        action (str): Acción tomada
        reward (float): Recompensa recibida
        next_state (str): Estado siguiente
        """
        if not self.dontlearn_enabled:
            return
            
        # Asegurar que el estado actual existe en la tabla Q
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}
            
        # Asegurar que el siguiente estado existe en la tabla Q
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in self.actions}
            
        # Actualizar el valor Q usando la ecuación de Bellman
        max_next_q = max(self.q_table[next_state].values())
        current_q = self.q_table[state][action]
        
        # Q(s,a) = Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
        
        # Actualizar la duración de la sesión
        self.duration += 1

    def choose_action(self, state, training=True):
        """
        Selecciona una acción basada en el estado actual.
        Implementa un balance entre exploración y explotación.
        
        Parameters:
        state (str): Estado actual, formato de visión en 4 direcciones
        training (bool): Indica si estamos en modo entrenamiento
        
        Returns:
        str: Acción seleccionada (UP, DOWN, LEFT, RIGHT)
        """
        # Si el estado no está en la tabla Q, lo inicializamos
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.actions}
            
        # Estrategia epsilon-greedy para balance exploración/explotación
        if training and random.random() < self.exploration_rate:
            # Exploración: acción aleatoria
            return random.choice(self.actions)
        else:
            # Explotación: mejor acción según la tabla Q
            q_values = self.q_table[state]
            
            # Obtener acciones con el valor Q máximo (puede haber empates)
            max_q = max(q_values.values())
            best_actions = [action for action, q_value in q_values.items() 
                           if q_value == max_q]
            
            # Si hay empate, seleccionar una aleatoriamente
            return random.choice(best_actions)
    
    def update_snake_length(self, new_length):
        """
        Actualiza la longitud de la serpiente y registra el máximo alcanzado.
        
        Parameters:
        new_length (int): Nueva longitud de la serpiente
        """
        self.snake_length = new_length
        if new_length > self.max_snake_length:
            self.max_snake_length = new_length
    
    def calculate_reward(self, state, action, next_state, cell_content):
        """
        Calcula la recompensa basada en el resultado de la acción.
        
        Parameters:
        state (str): Estado antes de la acción
        action (str): Acción tomada
        next_state (str): Estado después de la acción
        cell_content (str): Contenido de la celda donde se movió ('G', 'R', 'W', 'S', '0')
        
        Returns:
        float: Recompensa calculada
        """
        # Recompensa base según el contenido de la celda
        reward = self.rewards.get(cell_content, 0)
        
        # Recompensas adicionales basadas en objetivos del proyecto
        if cell_content == 'G':
            # Bonificación por longitud alcanzada
            if self.snake_length >= 10:
                reward += 20  # Bonificación extra por alcanzar el objetivo de longitud 10
            
        elif cell_content in ['W', 'S']:
            # Penalización por fin de juego (choque)
            reward -= 10 * self.snake_length  # Mayor penalización si la serpiente era larga
        
        return reward

    def decay_exploration(self):
        """
        Disminuye progresivamente la tasa de exploración para favorecer la explotación
        a medida que el agente aprende.
        """
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay
        )

    def save_model(self, filepath):
        """
        Guarda el modelo (tabla Q y otros parámetros) en un archivo JSON.
        
        Parameters:
        filepath (str): Ruta del archivo donde guardar el modelo
        """
        # Si no termina en .json, añadir la extensión
        if not filepath.endswith(".json"):
            filepath += ".json"

        # Crear directorio "models" si no existe
        models_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "models"
        )
        os.makedirs(models_dir, exist_ok=True)

        # Ruta completa del archivo
        save_path = os.path.join(models_dir, os.path.basename(filepath))

        # Guardar el modelo como JSON
        with open(save_path, "w") as f:
            json.dump(
                {
                    "q_table": self.q_table,
                    "learning_rate": self.learning_rate,
                    "discount_factor": self.discount_factor,
                    "exploration_rate": self.exploration_rate,
                    "exploration_decay": self.exploration_decay,
                    "max_snake_length": self.max_snake_length,
                    "duration": self.duration
                },
                f
            )
        
        if self.verbose:
            print(f"Modelo guardado en {save_path}")

    def load_model(self, filepath):
        """
        Carga un modelo previamente guardado.
        
        Parameters:
        filepath (str): Ruta del archivo que contiene el modelo
        """
        with open(filepath, "r") as f:
            data = json.load(f)
            self.q_table = data.get("q_table", {})
            self.learning_rate = float(data.get("learning_rate", 0.1))
            self.discount_factor = float(data.get("discount_factor", 0.95))
            self.exploration_rate = float(data.get("exploration_rate", 0.3))
            self.exploration_decay = float(data.get("exploration_decay", 0.995))
            self.max_snake_length = int(data.get("max_snake_length", 3))
            self.duration = int(data.get("duration", 0))
        
        if self.verbose:
            print(f"Modelo cargado desde {filepath}")
            print(f"Estadísticas del modelo: Longitud máxima = {self.max_snake_length}, Duración = {self.duration}")

    def reset_session(self):
        """
        Reinicia los valores para una nueva sesión de entrenamiento,
        manteniendo la tabla Q y otros parámetros de aprendizaje.
        """
        self.snake_length = 3
        self.position_history = []
        self.current_position = (0, 0)
        self.duration = 0