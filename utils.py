def get_snake_vision_display(game, BLOCK_SIZE):
    """
    Muestra solo las líneas de visión desde la cabeza de la serpiente,
    el resto del tablero serán espacios en blanco.
    """
    # Calcular dimensiones del tablero
    board_size_x = game.w // BLOCK_SIZE
    board_size_y = game.h // BLOCK_SIZE
    
    # Crear tablero vacío con espacios
    board = []
    for y in range(board_size_y):
        row = []
        for x in range(board_size_x):
            row.append(" ")  # Todo inicialmente son espacios
        board.append(row)
    
    # Posición de la cabeza
    head_x = int(game.head.x // BLOCK_SIZE)
    head_y = int(game.head.y // BLOCK_SIZE)
    
    # Asegurar que la cabeza está dentro de los límites
    if 0 <= head_x < board_size_x and 0 <= head_y < board_size_y:
        board[head_y][head_x] = "H"
    
    # 1. Línea vertical hacia ARRIBA
    for y in range(head_y-1, -1, -1):
        x = head_x
        if not (0 <= x < board_size_x) or not (0 <= y < board_size_y):
            continue
            
        # Borde del tablero
        if y == 0:
            board[y][x] = "W"
            continue
            
        # Verificar si hay cuerpo de serpiente
        for segment in game.snake:
            seg_x = int(segment.x // BLOCK_SIZE)
            seg_y = int(segment.y // BLOCK_SIZE)
            if seg_x == x and seg_y == y:
                board[y][x] = "S"
                break
                
        # Verificar comidas (solo si no se encontró segmento)
        if board[y][x] == " ":  # Si sigue siendo espacio
            # Comida buena 1
            if game.food_good1:
                food_x = int(game.food_good1.x // BLOCK_SIZE)
                food_y = int(game.food_good1.y // BLOCK_SIZE)
                if food_x == x and food_y == y:
                    board[y][x] = "G"
            
            # Comida buena 2
            if game.food_good2 and board[y][x] == " ":
                food_x = int(game.food_good2.x // BLOCK_SIZE)
                food_y = int(game.food_good2.y // BLOCK_SIZE)
                if food_x == x and food_y == y:
                    board[y][x] = "G"
            
            # Comida mala
            if game.food_bad and board[y][x] == " ":
                food_x = int(game.food_bad.x // BLOCK_SIZE)
                food_y = int(game.food_bad.y // BLOCK_SIZE)
                if food_x == x and food_y == y:
                    board[y][x] = "R"
            
            # Si no hay nada, es espacio vacío visible
            if board[y][x] == " ":
                board[y][x] = "0"
    
    # 2. Línea vertical hacia ABAJO
    for y in range(head_y+1, board_size_y):
        x = head_x
        if not (0 <= x < board_size_x) or not (0 <= y < board_size_y):
            continue
            
        # Borde del tablero
        if y == board_size_y-1:
            board[y][x] = "W"
            continue
            
        # Verificar si hay cuerpo de serpiente
        for segment in game.snake:
            seg_x = int(segment.x // BLOCK_SIZE)
            seg_y = int(segment.y // BLOCK_SIZE)
            if seg_x == x and seg_y == y:
                board[y][x] = "S"
                break
                
        # Verificar comidas (solo si no se encontró segmento)
        if board[y][x] == " ":
            # Comida buena 1
            if game.food_good1:
                food_x = int(game.food_good1.x // BLOCK_SIZE)
                food_y = int(game.food_good1.y // BLOCK_SIZE)
                if food_x == x and food_y == y:
                    board[y][x] = "G"
            
            # Comida buena 2
            if game.food_good2 and board[y][x] == " ":
                food_x = int(game.food_good2.x // BLOCK_SIZE)
                food_y = int(game.food_good2.y // BLOCK_SIZE)
                if food_x == x and food_y == y:
                    board[y][x] = "G"
            
            # Comida mala
            if game.food_bad and board[y][x] == " ":
                food_x = int(game.food_bad.x // BLOCK_SIZE)
                food_y = int(game.food_bad.y // BLOCK_SIZE)
                if food_x == x and food_y == y:
                    board[y][x] = "R"
            
            # Si no hay nada, es espacio vacío visible
            if board[y][x] == " ":
                board[y][x] = "0"
    
    # 3. Línea horizontal hacia IZQUIERDA
    for x in range(head_x-1, -1, -1):
        y = head_y
        if not (0 <= x < board_size_x) or not (0 <= y < board_size_y):
            continue
            
        # Borde del tablero
        if x == 0:
            board[y][x] = "W"
            continue
            
        # Verificar si hay cuerpo de serpiente
        for segment in game.snake:
            seg_x = int(segment.x // BLOCK_SIZE)
            seg_y = int(segment.y // BLOCK_SIZE)
            if seg_x == x and seg_y == y:
                board[y][x] = "S"
                break
                
        # Verificar comidas (solo si no se encontró segmento)
        if board[y][x] == " ":
            # Comida buena 1
            if game.food_good1:
                food_x = int(game.food_good1.x // BLOCK_SIZE)
                food_y = int(game.food_good1.y // BLOCK_SIZE)
                if food_x == x and food_y == y:
                    board[y][x] = "G"
            
            # Comida buena 2
            if game.food_good2 and board[y][x] == " ":
                food_x = int(game.food_good2.x // BLOCK_SIZE)
                food_y = int(game.food_good2.y // BLOCK_SIZE)
                if food_x == x and food_y == y:
                    board[y][x] = "G"
            
            # Comida mala
            if game.food_bad and board[y][x] == " ":
                food_x = int(game.food_bad.x // BLOCK_SIZE)
                food_y = int(game.food_bad.y // BLOCK_SIZE)
                if food_x == x and food_y == y:
                    board[y][x] = "R"
            
            # Si no hay nada, es espacio vacío visible
            if board[y][x] == " ":
                board[y][x] = "0"
    
    # 4. Línea horizontal hacia DERECHA
    for x in range(head_x+1, board_size_x):
        y = head_y
        if not (0 <= x < board_size_x) or not (0 <= y < board_size_y):
            continue
            
        # Borde del tablero
        if x == board_size_x-1:
            board[y][x] = "W"
            continue
            
        # Verificar si hay cuerpo de serpiente
        for segment in game.snake:
            seg_x = int(segment.x // BLOCK_SIZE)
            seg_y = int(segment.y // BLOCK_SIZE)
            if seg_x == x and seg_y == y:
                board[y][x] = "S"
                break
                
        # Verificar comidas (solo si no se encontró segmento)
        if board[y][x] == " ":
            # Comida buena 1
            if game.food_good1:
                food_x = int(game.food_good1.x // BLOCK_SIZE)
                food_y = int(game.food_good1.y // BLOCK_SIZE)
                if food_x == x and food_y == y:
                    board[y][x] = "G"
            
            # Comida buena 2
            if game.food_good2 and board[y][x] == " ":
                food_x = int(game.food_good2.x // BLOCK_SIZE)
                food_y = int(game.food_good2.y // BLOCK_SIZE)
                if food_x == x and food_y == y:
                    board[y][x] = "G"
            
            # Comida mala
            if game.food_bad and board[y][x] == " ":
                food_x = int(game.food_bad.x // BLOCK_SIZE)
                food_y = int(game.food_bad.y // BLOCK_SIZE)
                if food_x == x and food_y == y:
                    board[y][x] = "R"
            
            # Si no hay nada, es espacio vacío visible
            if board[y][x] == " ":
                board[y][x] = "0"
    
    # Convertir la matriz a cadena
    lines = []
    for row in board:
        lines.append(''.join(row))
    
    return '\n'.join(lines)