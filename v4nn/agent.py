import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import math

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        
        # Ampliamos el número de estados para incluir información de exploración
        self.model = Linear_QNet(15, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        # Mapa para seguimiento de celdas visitadas
        self.visited_cells = set()
        
        # Información para rastrear la última comida vista
        self.food_seen = False
        self.last_food_dir = [0, 0, 0, 0]  # [left, right, up, down]
        
        # Contador para movimientos sin progreso
        self.moves_without_progress = 0
        self.previous_distance = 0
        
        # Para rastrear la última recompensa
        self.last_score = 0

    def reset(self, game):
        """Reinicia el estado del agente entre partidas"""
        self.visited_cells = set()
        self.food_seen = False
        self.last_food_dir = [0, 0, 0, 0]
        self.moves_without_progress = 0
        self.previous_distance = self._calculate_distance(game.head, game.food)
        self.last_score = 0

    def _calculate_distance(self, point1, point2):
        """Calcula la distancia Manhattan entre dos puntos"""
        return abs(point1.x - point2.x) + abs(point1.y - point2.y)

    def _is_in_vision_range(self, head, food, max_distance=100):
        """Determina si la comida está en el rango de visión (limitado por distancia)"""
        # Calcula la distancia Manhattan
        distance = self._calculate_distance(head, food)
        return distance <= max_distance

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # La celda actual está visitada
        self.visited_cells.add((head.x, head.y))
        
        # Verifica si los puntos adyacentes ya han sido visitados
        left_visited = (point_l.x, point_l.y) in self.visited_cells
        right_visited = (point_r.x, point_r.y) in self.visited_cells
        up_visited = (point_u.x, point_u.y) in self.visited_cells
        down_visited = (point_d.x, point_d.y) in self.visited_cells

        # Detecta si la comida está en línea recta (visión ortogonal)
        # No verifica bloqueos, solo si está en la misma fila o columna
        food_left = False
        food_right = False
        food_up = False
        food_down = False
        
        if self._is_in_vision_range(head, game.food):
            # Comida a la izquierda en línea recta
            if game.food.y == head.y and game.food.x < head.x:
                food_left = True
                self.food_seen = True
                self.last_food_dir = [1, 0, 0, 0]
                
            # Comida a la derecha en línea recta
            elif game.food.y == head.y and game.food.x > head.x:
                food_right = True
                self.food_seen = True
                self.last_food_dir = [0, 1, 0, 0]
                
            # Comida arriba en línea recta
            elif game.food.x == head.x and game.food.y < head.y:
                food_up = True
                self.food_seen = True
                self.last_food_dir = [0, 0, 1, 0]
                
            # Comida abajo en línea recta
            elif game.food.x == head.x and game.food.y > head.y:
                food_down = True
                self.food_seen = True
                self.last_food_dir = [0, 0, 0, 1]
        
        # Si la comida fue vista anteriormente, pero ya no está en línea recta
        elif self.food_seen:
            food_left, food_right, food_up, food_down = self.last_food_dir

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
            
            # Food location (visible solo en línea recta)
            food_left,
            food_right,
            food_up,
            food_down,
            
            # Células visitadas (para estimular exploración)
            left_visited,
            right_visited,
            up_visited,
            down_visited
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Aumentamos la exploración al inicio
        self.epsilon = 100 - self.n_games if self.n_games < 80 else 20
        
        final_move = [0, 0, 0]
        
        # Exploración aleatoria
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def calculate_dynamic_reward(self, game, done, score):
        """Calcula recompensas dinámicas para fomentar exploración"""
        reward = 0
        head = game.snake[0]
        
        # Penalización por muerte
        if done:
            return -10
            
        # Recompensa por comer
        if score > self.last_score:
            self.last_score = score
            reward += 10
            # Resetear el contador de movimientos sin progreso
            self.moves_without_progress = 0
            return reward
            
        # Calcular distancia actual a la comida
        current_distance = self._calculate_distance(head, game.food)
        
        # Recompensa por acercarse a la comida (si es visible)
        if self.food_seen and current_distance < self.previous_distance:
            reward += 0.5
            self.moves_without_progress = 0
        else:
            # Penalización por no acercarse
            self.moves_without_progress += 1
            
        # Penalización por estar dando vueltas sin progreso
        if self.moves_without_progress > 50:
            reward -= 0.2
        
        # Recompensa por explorar nuevas celdas
        if (head.x, head.y) not in self.visited_cells or len(self.visited_cells) < 10:
            reward += 0.1
            
        # Actualizar distancia previa
        self.previous_distance = current_distance
            
        return reward


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    # Inicializar estado del agente
    agent.reset(game)
    
    while True:
        # Obtener estado actual
        state_old = agent.get_state(game)

        # Obtener acción
        final_move = agent.get_action(state_old)

        # Realizar movimiento y obtener nuevo estado
        default_reward, done, score = game.play_step(final_move)
        
        # Calcular recompensa dinámica
        dynamic_reward = agent.calculate_dynamic_reward(game, done, score)
        
        state_new = agent.get_state(game)

        # Entrenar memoria a corto plazo
        agent.train_short_memory(state_old, final_move, dynamic_reward, state_new, done)

        # Recordar para entrenar memoria a largo plazo
        agent.remember(state_old, final_move, dynamic_reward, state_new, done)

        if done:
            # Reiniciar juego
            game.reset()
            agent.n_games += 1
            
            # Entrenar memoria a largo plazo
            agent.train_long_memory()
            
            # Reiniciar estado del agente
            agent.reset(game)

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()