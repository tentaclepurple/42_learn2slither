import pygame
import random
import numpy as np
import argparse
import torch
import os
from collections import deque
from enum import Enum
from collections import namedtuple
import matplotlib.pyplot as plt
from IPython import display

# Definiciones básicas
pygame.init()
font = pygame.font.Font('arial.ttf', 20)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Colores
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)  # Comida buena
RED = (255, 0, 0)    # Comida mala
BLUE1 = (0, 0, 255)  # Serpiente borde
BLUE2 = (0, 100, 255)  # Serpiente relleno
BLACK = (0, 0, 0)    # Fondo

# Constantes
BLOCK_SIZE = 20
SPEED = 40
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

# Clase del juego Snake
class SnakeGameAI:
    def __init__(self, size=20, is_visual=True):
        # Calcular dimensiones basadas en tamaño
        self.size = size
        self.w = size * BLOCK_SIZE
        self.h = size * BLOCK_SIZE
        self.is_visual = is_visual
        
        # Inicializar display solo si es visual
        if is_visual:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake AI')
        else:
            # Modo sin visualización
            self.display = None
        
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Estado inicial del juego
        self.direction = Direction.RIGHT
        # Centrar la serpiente en el tablero
        center_x = (self.size // 2) * BLOCK_SIZE
        center_y = (self.size // 2) * BLOCK_SIZE
        self.head = Point(center_x, center_y)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        # Crear las comidas (2 buenas, 1 mala)
        self.food_good1 = None
        self.food_good2 = None
        self.food_bad = None
        self._place_food_good1()
        self._place_food_good2()
        self._place_food_bad()
        self.frame_iteration = 0

    def _place_food_good1(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food_good1 = Point(x, y)
        # Evitar colocar sobre la serpiente o otras comidas
        if (self.food_good1 in self.snake or 
            (self.food_bad and self.food_good1 == self.food_bad) or
            (self.food_good2 and self.food_good1 == self.food_good2)):
            self._place_food_good1()
            
    def _place_food_good2(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food_good2 = Point(x, y)
        # Evitar colocar sobre la serpiente o otras comidas
        if (self.food_good2 in self.snake or 
            (self.food_bad and self.food_good2 == self.food_bad) or
            (self.food_good1 and self.food_good2 == self.food_good1)):
            self._place_food_good2()

    def _place_food_bad(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food_bad = Point(x, y)
        # Evitar colocar sobre la serpiente o comidas buenas
        if (self.food_bad in self.snake or 
            (self.food_good1 and self.food_bad == self.food_good1) or
            (self.food_good2 and self.food_bad == self.food_good2)):
            self._place_food_bad()

    def play_step(self, action):
        self.frame_iteration += 1
        
        # Eventos
        if self.is_visual:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        else:
            pygame.event.pump()  # Mantener pygame vivo
            
        # Obtener posición antes de moverse
        old_head = self.head
        old_distance_good1 = abs(old_head.x - self.food_good1.x) + abs(old_head.y - self.food_good1.y)
        old_distance_good2 = abs(old_head.x - self.food_good2.x) + abs(old_head.y - self.food_good2.y)
        old_distance_bad = abs(old_head.x - self.food_bad.x) + abs(old_head.y - self.food_bad.y)
        
        old_food_visible_good1 = self.is_food_visible(food_type="good1")
        old_food_visible_good2 = self.is_food_visible(food_type="good2")
        old_food_visible_bad = self.is_food_visible(food_type="bad")
        
        # Moverse
        self._move(action)
        self.snake.insert(0, self.head)
        
        # Comprobar fin del juego
        reward = 0
        game_over = False
        
        # Colisión con pared o consigo misma
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Comida buena o mala
        if self.head == self.food_good1:
            # Comió comida buena 1
            self.score += 1
            reward = 10
            self._place_food_good1()
        elif self.head == self.food_good2:
            # Comió comida buena 2
            self.score += 1
            reward = 10
            self._place_food_good2()
        elif self.head == self.food_bad:
            # Comió comida mala
            if len(self.snake) <= 2:  # Si sólo tiene 3 segmentos (mínimo)
                game_over = True
                reward = -10
                return reward, game_over, self.score
            else:
                # Reducir tamaño
                self.snake.pop()  # Ya quitamos uno más abajo, aquí quitamos otro
                reward = -5
                self._place_food_bad()
        else:
            # Sin comer, calcular recompensa basada en proximidad
            new_distance_good1 = abs(self.head.x - self.food_good1.x) + abs(self.head.y - self.food_good1.y)
            new_distance_good2 = abs(self.head.x - self.food_good2.x) + abs(self.head.y - self.food_good2.y)
            new_distance_bad = abs(self.head.x - self.food_bad.x) + abs(self.head.y - self.food_bad.y)
            
            new_food_visible_good1 = self.is_food_visible(food_type="good1")
            new_food_visible_good2 = self.is_food_visible(food_type="good2")
            new_food_visible_bad = self.is_food_visible(food_type="bad")
            
            # Recompensa por acercarse a comida buena 1
            if old_food_visible_good1 or new_food_visible_good1:
                if new_distance_good1 < old_distance_good1:
                    reward += 5  # Recompensa por acercarse a comida buena
                elif new_distance_good1 > old_distance_good1:
                    reward -= 1  # Castigo por alejarse de comida buena
            
            # Recompensa por acercarse a comida buena 2
            if old_food_visible_good2 or new_food_visible_good2:
                if new_distance_good2 < old_distance_good2:
                    reward += 5  # Recompensa por acercarse a comida buena
                elif new_distance_good2 > old_distance_good2:
                    reward -= 1  # Castigo por alejarse de comida buena
            
            # Recompensa por alejarse de comida mala
            if old_food_visible_bad or new_food_visible_bad:
                if new_distance_bad > old_distance_bad:
                    reward += 2  # Recompensa por alejarse de comida mala
                elif new_distance_bad < old_distance_bad:
                    reward -= 1  # Castigo por acercarse a comida mala
            
            self.snake.pop()
        
        # Actualizar UI
        if self.is_visual:
            self._update_ui()
        
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Límites del mapa
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Colisión con el cuerpo
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        if not self.is_visual:
            return
            
        self.display.fill(BLACK)
        
        # Dibujar serpiente
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        # Dibujar comidas buenas (verdes)
        pygame.draw.rect(self.display, GREEN, pygame.Rect(self.food_good1.x, self.food_good1.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, GREEN, pygame.Rect(self.food_good2.x, self.food_good2.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Dibujar comida mala (roja)
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food_bad.x, self.food_bad.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Dibujar puntuación
        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [recto, derecha, izquierda]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # sin cambio
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # giro derecha
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # giro izquierda

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
        
    def is_food_visible(self, direction=None, food_type="good1"):
        """
        Comprueba si la comida es visible desde la cabeza en línea recta
        food_type: "good1", "good2" o "bad" para indicar qué comida buscar
        """
        head = self.head
        
        if food_type == "good1":
            food = self.food_good1
        elif food_type == "good2":
            food = self.food_good2
        else:  # "bad"
            food = self.food_bad
        
        if direction is None:
            return (self.is_food_visible(Direction.LEFT, food_type) or
                    self.is_food_visible(Direction.RIGHT, food_type) or
                    self.is_food_visible(Direction.UP, food_type) or
                    self.is_food_visible(Direction.DOWN, food_type))
        
        # Comprobar visibilidad según dirección
        if direction == Direction.RIGHT:
            if head.y == food.y and head.x < food.x:
                for body in self.snake[1:]:
                    if body.y == head.y and head.x < body.x < food.x:
                        return False
                return True
        
        elif direction == Direction.LEFT:
            if head.y == food.y and head.x > food.x:
                for body in self.snake[1:]:
                    if body.y == head.y and food.x < body.x < head.x:
                        return False
                return True
        
        elif direction == Direction.UP:
            if head.x == food.x and head.y > food.y:
                for body in self.snake[1:]:
                    if body.x == head.x and food.y < body.y < head.y:
                        return False
                return True
        
        elif direction == Direction.DOWN:
            if head.x == food.x and head.y < food.y:
                for body in self.snake[1:]:
                    if body.x == head.x and head.y < body.y < food.y:
                        return False
                return True
        
        return False

# Modelo neural
class Linear_QNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_path='model.pth'):
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

# Entrenador
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # Valores Q predichos con estado actual
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

# Agente
class Agent:
    def __init__(self, load_model=None, dont_learn=False):
        self.n_games = 0
        self.epsilon = 80  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(19, 256, 3)  # Ahora 19 entradas (2 comidas buenas + 1 mala)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.dont_learn = dont_learn
        
        # Cargar modelo si existe
        if load_model and os.path.exists(load_model):
            try:
                print(f"Loading model from {load_model}")
                self.model.load_state_dict(torch.load(load_model))
                print(f"Model loaded successfully!")
                # Si dontlearn está activo, usar casi siempre el modelo
                if dont_learn:
                    self.epsilon = -1000  # Garantiza que casi nunca use aleatorio
                else:
                    self.epsilon = 20  # Algo de exploración para continuar entrenando
            except Exception as e:
                print(f"Error loading model: {e}")

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

        # Comprobar visibilidad de comidas
        food_good1_visible_left = game.is_food_visible(Direction.LEFT, "good1")
        food_good1_visible_right = game.is_food_visible(Direction.RIGHT, "good1")
        food_good1_visible_up = game.is_food_visible(Direction.UP, "good1")
        food_good1_visible_down = game.is_food_visible(Direction.DOWN, "good1")
        
        food_good2_visible_left = game.is_food_visible(Direction.LEFT, "good2")
        food_good2_visible_right = game.is_food_visible(Direction.RIGHT, "good2")
        food_good2_visible_up = game.is_food_visible(Direction.UP, "good2")
        food_good2_visible_down = game.is_food_visible(Direction.DOWN, "good2")
        
        food_bad_visible_left = game.is_food_visible(Direction.LEFT, "bad")
        food_bad_visible_right = game.is_food_visible(Direction.RIGHT, "bad")
        food_bad_visible_up = game.is_food_visible(Direction.UP, "bad")
        food_bad_visible_down = game.is_food_visible(Direction.DOWN, "bad")

        state = [
            # Peligro recto
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Peligro derecha
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Peligro izquierda
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Dirección movimiento
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Comida BUENA 1 visible en línea recta
            food_good1_visible_left,
            food_good1_visible_right,
            food_good1_visible_up,
            food_good1_visible_down,
            
            # Comida BUENA 2 visible en línea recta
            food_good2_visible_left,
            food_good2_visible_right,
            food_good2_visible_up,
            food_good2_visible_down,
            
            # Comida MALA visible en línea recta
            food_bad_visible_left,
            food_bad_visible_right,
            food_bad_visible_up,
            food_bad_visible_down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if self.dont_learn:
            return
            
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        if not self.dont_learn:
            self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Si dont_learn está activo, casi nunca usamos acciones aleatorias
        if self.dont_learn:
            # Usar 99.9% el modelo
            use_random = random.randint(0, 1000) < 1  # 0.1% aleatorio
        else:
            # Comportamiento normal de exploración/explotación
            self.epsilon = max(80 - self.n_games, 0)
            use_random = random.randint(0, 200) < self.epsilon
        
        final_move = [0,0,0]
        if use_random:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

def train():
    # Parsear argumentos
    parser = argparse.ArgumentParser(description='Train Snake AI')
    parser.add_argument('-save', type=str, help='Path to save model')
    parser.add_argument('-load', type=str, help='Path to load model')
    parser.add_argument('-speed', type=float, default=SPEED, help='Game speed (0 for max)')
    parser.add_argument('-sessions', type=int, default=0, help='Number of sessions (0 for infinite)')
    parser.add_argument('-visual', choices=['on', 'off'], default='on', help='Enable visualization')
    parser.add_argument('-dontlearn', action='store_true', help='Disable learning')
    parser.add_argument('-step', action='store_true', help='Step by step mode')
    parser.add_argument('-size', type=int, default=20, help='Size of the board (number of cells)')
    args = parser.parse_args()
    
    # Inicializar
    is_visual = args.visual == 'on'
    agent = Agent(load_model=args.load, dont_learn=args.dontlearn)
    game = SnakeGameAI(size=args.size, is_visual=is_visual)
    
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    sessions = 0
    
    # Mostrar configuración
    print(f"Starting with configuration:")
    print(f" - Save model: {args.save or 'None'}")
    print(f" - Load model: {args.load or 'None'}")
    print(f" - Speed: {args.speed}")
    print(f" - Visual: {args.visual}")
    print(f" - Board size: {args.size} x {args.size}")
    print(f" - Learning: {'Disabled' if args.dontlearn else 'Enabled'}")
    print(f" - Step mode: {'Enabled' if args.step else 'Disabled'}")
    
    try:
        while True:
            # Comprobar límite de sesiones
            if args.sessions > 0 and sessions >= args.sessions:
                break
                
            # Obtener estado actual
            state_old = agent.get_state(game)

            # Obtener acción
            final_move = agent.get_action(state_old)

            # Paso a paso
            if args.step and is_visual:
                print("Press any key to continue...")
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            quit()
                        elif event.type == pygame.KEYDOWN:
                            waiting = False
                    
            # Ejecutar acción
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            # Entrenar memoria corta
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

            # Control de velocidad
            if args.speed > 0 and is_visual:
                game.clock.tick(args.speed)
            
            if done:
                # Game over - reiniciar y entrenar
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                # Guardar récord
                if score > record:
                    record = score
                    if args.save:
                        print(f"New record! Saving model to {args.save}")
                        agent.model.save(args.save)

                # Actualizar estadísticas
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                sessions += 1
                
                # Mostrar información
                if args.speed == 0 and not is_visual:
                    if agent.n_games % 20 == 0:  # Mostrar cada 20 juegos en modo rápido
                        print(f'Game {agent.n_games}, Score {score}, Record: {record}, Avg: {mean_score:.1f}')
                else:
                    print(f'Game {agent.n_games}, Score {score}, Record: {record}')
                
                # Gráficos
                if is_visual:
                    try:
                        plot(plot_scores, plot_mean_scores)
                    except:
                        pass
                        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if args.save:
            print(f"Saving final model to {args.save}")
            agent.model.save(args.save)
    
    print(f"Training completed: {agent.n_games} games, Record: {record}")

if __name__ == '__main__':
    train()