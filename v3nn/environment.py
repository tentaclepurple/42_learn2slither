# environment.py
import random
import pygame
import time


class SnakeEnvironment:
    """
    Clase que representa el entorno del juego Snake.
    Gestiona el tablero, la serpiente, las manzanas y las reglas del juego.
    """
    
    def __init__(self, board_size=10, display_enabled=True):
        """
        Inicializa el entorno del juego.
        
        Parameters:
        board_size (int): Tamaño del tablero cuadrado (por defecto 10x10)
        display_enabled (bool): Indica si se debe mostrar la interfaz gráfica
        """
        self.board_size = board_size
        self.display_enabled = display_enabled
        self.board = [[None for _ in range(board_size)] for _ in range(board_size)]
        
        # Colores para la representación gráfica
        self.colors = {
            'EMPTY': (200, 200, 200),
            'SNAKE': (0, 0, 255),
            'GREEN_APPLE': (0, 255, 0),
            'RED_APPLE': (255, 0, 0),
            'WALL': (100, 100, 100),
            'HEAD': (50, 50, 255)
        }
        
        # Inicialización de Pygame si el display está habilitado
        if self.display_enabled:
            pygame.init()
            self.cell_size = 40
            self.display = pygame.display.set_mode(
                (board_size * self.cell_size, board_size * self.cell_size)
            )
            pygame.display.set_caption("Snake Game")
            self.font = pygame.font.SysFont(None, 24)
        
        # Variables de estado del juego
        self.reset()
    
    def reset(self):
        """
        Reinicia el entorno para una nueva sesión de juego.
        Coloca la serpiente y las manzanas en posiciones aleatorias.
        
        Returns:
        str: Estado inicial que ve la serpiente
        """
        # Limpiar el tablero
        for i in range(self.board_size):
            for j in range(self.board_size):
                self.board[i][j] = '0'  # Espacio vacío
        
        # Crear la serpiente inicial (3 celdas)
        self.snake = []
        start_x = random.randint(3, self.board_size - 3)
        start_y = random.randint(3, self.board_size - 3)
        
        # Posiciones de la serpiente (cabeza al final)
        self.snake = [(start_x - 2, start_y), (start_x - 1, start_y), (start_x, start_y)]
        
        # Colocar la serpiente en el tablero
        for x, y in self.snake[:-1]:
            self.board[x][y] = 'S'  # Cuerpo
        self.board[self.snake[-1][0]][self.snake[-1][1]] = 'H'  # Cabeza
        
        # Colocar manzanas
        self._place_apples()
        
        # Variables de estado
        self.game_over = False
        self.score = 0
        self.steps = 0
        
        # Obtener estado inicial
        return self._get_snake_vision()
    
    def _place_apples(self):
        """
        Coloca las manzanas (2 verdes, 1 roja) en posiciones aleatorias vacías.
        """
        # Manzanas verdes (2)
        green_apples_placed = 0
        while green_apples_placed < 2:
            x, y = random.randint(0, self.board_size - 1), random.randint(0, self.board_size - 1)
            if self.board[x][y] == '0':
                self.board[x][y] = 'G'
                green_apples_placed += 1
        
        # Manzana roja (1)
        red_apple_placed = False
        while not red_apple_placed:
            x, y = random.randint(0, self.board_size - 1), random.randint(0, self.board_size - 1)
            if self.board[x][y] == '0':
                self.board[x][y] = 'R'
                red_apple_placed = True
    
    def _get_snake_vision(self):
        """
        Obtiene lo que la serpiente ve en las 4 direcciones.
        
        Returns:
        str: Representación de lo que ve la serpiente
        """
        # Posición de la cabeza
        head_x, head_y = self.snake[-1]
        
        # Visión en las 4 direcciones
        vision = [[] for _ in range(4)]  # UP, DOWN, LEFT, RIGHT
        
        # Mirar hacia arriba
        for i in range(head_x - 1, -1, -1):
            cell = self.board[i][head_y]
            if cell == 'H':  # Si vemos otra parte de la cabeza, es realmente el cuerpo
                cell = 'S'
            vision[0].append(cell)
        
        # Mirar hacia abajo
        for i in range(head_x + 1, self.board_size):
            cell = self.board[i][head_y]
            if cell == 'H':
                cell = 'S'
            vision[1].append(cell)
        
        # Mirar hacia la izquierda
        for j in range(head_y - 1, -1, -1):
            cell = self.board[head_x][j]
            if cell == 'H':
                cell = 'S'
            vision[2].append(cell)
        
        # Mirar hacia la derecha
        for j in range(head_y + 1, self.board_size):
            cell = self.board[head_x][j]
            if cell == 'H':
                cell = 'S'
            vision[3].append(cell)
        
        # Añadir paredes virtuales al final de cada dirección si no llegamos al borde
        for i in range(4):
            if len(vision[i]) < self.board_size:
                vision[i].extend(['W'] * (self.board_size - len(vision[i])))
        
        # Convertir la visión a formato de string
        vision_str = []
        for direction in vision:
            vision_str.append(''.join(direction))
        
        return ','.join(vision_str)
    
    def step(self, action):
        """
        Ejecuta un paso en el entorno basado en la acción seleccionada.
        
        Parameters:
        action (str): Una de las acciones "UP", "DOWN", "LEFT", "RIGHT"
        
        Returns:
        tuple: (next_state, reward, game_over, info)
        """
        self.steps += 1
        
        # Obtener la cabeza actual
        head_x, head_y = self.snake[-1]
        
        # Calcular nueva posición basada en la acción
        if action == "UP":
            new_head = (head_x - 1, head_y)
        elif action == "DOWN":
            new_head = (head_x + 1, head_y)
        elif action == "LEFT":
            new_head = (head_x, head_y - 1)
        elif action == "RIGHT":
            new_head = (head_x, head_y + 1)
        
        # Verificar si la nueva posición está fuera del tablero
        new_x, new_y = new_head
        if new_x < 0 or new_x >= self.board_size or new_y < 0 or new_y >= self.board_size:
            # Choque con pared
            self.game_over = True
            return self._get_snake_vision(), -20, True, {"message": "Choque con pared", "length": len(self.snake)}
        
        # Verificar colisión con el cuerpo
        if new_head in self.snake:
            # Colisión con el cuerpo
            self.game_over = True
            return self._get_snake_vision(), -20, True, {"message": "Colisión con el cuerpo", "length": len(self.snake)}
        
        # Contenido de la nueva posición
        cell_content = self.board[new_x][new_y]
        
        # Actualizar la posición de la serpiente
        self.snake.append(new_head)
        
        # Actualizar el tablero marcando la cabeza
        self.board[new_x][new_y] = 'H'
        
        # La antigua cabeza ahora es parte del cuerpo
        self.board[head_x][head_y] = 'S'
        
        reward = 0
        
        # Procesar efectos según contenido de la celda
        if cell_content == 'G':  # Manzana verde
            # Aumentar la longitud (no eliminar la cola)
            reward = 10
            # Colocar una nueva manzana verde
            self._place_green_apple()
        elif cell_content == 'R':  # Manzana roja
            # Disminuir la longitud (eliminar la cola y una celda más)
            if len(self.snake) > 1:
                tail = self.snake.pop(0)
                self.board[tail[0]][tail[1]] = '0'
            
            # Si la longitud es 0, fin del juego
            if len(self.snake) == 0:
                self.game_over = True
                return self._get_snake_vision(), -30, True, {"message": "Longitud 0", "length": 0}
            
            reward = -5
            # Colocar una nueva manzana roja
            self._place_red_apple()
        else:
            # Movimiento normal, eliminar la cola
            tail = self.snake.pop(0)
            self.board[tail[0]][tail[1]] = '0'
            reward = -0.1  # Pequeña penalización para fomentar eficiencia
        
        # Renderizar si el display está habilitado
        if self.display_enabled:
            self._render()
        
        # Verificar condición de victoria (longitud 10+)
        if len(self.snake) >= 10:
            reward += 20  # Bonificación por alcanzar el objetivo
        
        return self._get_snake_vision(), reward, self.game_over, {"length": len(self.snake)}
    
    def _place_green_apple(self):
        """
        Coloca una nueva manzana verde en una posición aleatoria vacía.
        """
        while True:
            x, y = random.randint(0, self.board_size - 1), random.randint(0, self.board_size - 1)
            if self.board[x][y] == '0':
                self.board[x][y] = 'G'
                break
    
    def _place_red_apple(self):
        """
        Coloca una nueva manzana roja en una posición aleatoria vacía.
        """
        while True:
            x, y = random.randint(0, self.board_size - 1), random.randint(0, self.board_size - 1)
            if self.board[x][y] == '0':
                self.board[x][y] = 'R'
                break
    
    def _render(self):
        """
        Renderiza el estado actual del juego en la ventana de Pygame.
        """
        self.display.fill((0, 0, 0))  # Fondo negro
        
        # Dibujar el tablero
        for i in range(self.board_size):
            for j in range(self.board_size):
                color = self.colors['EMPTY']
                
                if self.board[i][j] == 'S':
                    color = self.colors['SNAKE']
                elif self.board[i][j] == 'H':
                    color = self.colors['HEAD']
                elif self.board[i][j] == 'G':
                    color = self.colors['GREEN_APPLE']
                elif self.board[i][j] == 'R':
                    color = self.colors['RED_APPLE']
                
                pygame.draw.rect(
                    self.display,
                    color,
                    (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                )
                pygame.draw.rect(
                    self.display,
                    (0, 0, 0),
                    (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size),
                    1
                )
        
        # Mostrar información
        score_text = self.font.render(f"Longitud: {len(self.snake)}", True, (255, 255, 255))
        self.display.blit(score_text, (10, 10))
        
        pygame.display.update()
    
    def close(self):
        """
        Cierra el entorno y libera recursos.
        """
        if self.display_enabled:
            pygame.quit()