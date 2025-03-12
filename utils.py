def get_snake_vision_display(game, BLOCK_SIZE):

    board_size_x = game.w // BLOCK_SIZE
    board_size_y = game.h // BLOCK_SIZE

    board = []
    for y in range(board_size_y):
        row = []
        for x in range(board_size_x):
            row.append(" ")
        board.append(row)

    head_x = int(game.head.x // BLOCK_SIZE)
    head_y = int(game.head.y // BLOCK_SIZE)

    if 0 <= head_x < board_size_x and 0 <= head_y < board_size_y:
        board[head_y][head_x] = "H"

    for y in range(head_y-1, -1, -1):
        x = head_x
        if not (0 <= x < board_size_x) or not (0 <= y < board_size_y):
            continue

        if y == 0:
            board[y][x] = "W"
            continue

        for segment in game.snake:
            seg_x = int(segment.x // BLOCK_SIZE)
            seg_y = int(segment.y // BLOCK_SIZE)
            if seg_x == x and seg_y == y:
                board[y][x] = "S"
                break

        if board[y][x] == " ":
            if game.food_good1:
                food_x = int(game.food_good1.x // BLOCK_SIZE)
                food_y = int(game.food_good1.y // BLOCK_SIZE)
                if food_x == x and food_y == y:
                    board[y][x] = "G"

            if game.food_good2 and board[y][x] == " ":
                food_x = int(game.food_good2.x // BLOCK_SIZE)
                food_y = int(game.food_good2.y // BLOCK_SIZE)
                if food_x == x and food_y == y:
                    board[y][x] = "G"

            if game.food_bad and board[y][x] == " ":
                food_x = int(game.food_bad.x // BLOCK_SIZE)
                food_y = int(game.food_bad.y // BLOCK_SIZE)
                if food_x == x and food_y == y:
                    board[y][x] = "R"

            if board[y][x] == " ":
                board[y][x] = "0"

    for y in range(head_y+1, board_size_y):
        x = head_x
        if not (0 <= x < board_size_x) or not (0 <= y < board_size_y):
            continue

        if y == board_size_y-1:
            board[y][x] = "W"
            continue

        for segment in game.snake:
            seg_x = int(segment.x // BLOCK_SIZE)
            seg_y = int(segment.y // BLOCK_SIZE)
            if seg_x == x and seg_y == y:
                board[y][x] = "S"
                break

        if board[y][x] == " ":
            if game.food_good1:
                food_x = int(game.food_good1.x // BLOCK_SIZE)
                food_y = int(game.food_good1.y // BLOCK_SIZE)
                if food_x == x and food_y == y:
                    board[y][x] = "G"

            if game.food_good2 and board[y][x] == " ":
                food_x = int(game.food_good2.x // BLOCK_SIZE)
                food_y = int(game.food_good2.y // BLOCK_SIZE)
                if food_x == x and food_y == y:
                    board[y][x] = "G"

            if game.food_bad and board[y][x] == " ":
                food_x = int(game.food_bad.x // BLOCK_SIZE)
                food_y = int(game.food_bad.y // BLOCK_SIZE)
                if food_x == x and food_y == y:
                    board[y][x] = "R"

            if board[y][x] == " ":
                board[y][x] = "0"

    for x in range(head_x-1, -1, -1):
        y = head_y
        if not (0 <= x < board_size_x) or not (0 <= y < board_size_y):
            continue

        if x == 0:
            board[y][x] = "W"
            continue

        for segment in game.snake:
            seg_x = int(segment.x // BLOCK_SIZE)
            seg_y = int(segment.y // BLOCK_SIZE)
            if seg_x == x and seg_y == y:
                board[y][x] = "S"
                break

        if board[y][x] == " ":
            if game.food_good1:
                food_x = int(game.food_good1.x // BLOCK_SIZE)
                food_y = int(game.food_good1.y // BLOCK_SIZE)
                if food_x == x and food_y == y:
                    board[y][x] = "G"

            if game.food_good2 and board[y][x] == " ":
                food_x = int(game.food_good2.x // BLOCK_SIZE)
                food_y = int(game.food_good2.y // BLOCK_SIZE)
                if food_x == x and food_y == y:
                    board[y][x] = "G"

            if game.food_bad and board[y][x] == " ":
                food_x = int(game.food_bad.x // BLOCK_SIZE)
                food_y = int(game.food_bad.y // BLOCK_SIZE)
                if food_x == x and food_y == y:
                    board[y][x] = "R"

            if board[y][x] == " ":
                board[y][x] = "0"

    for x in range(head_x+1, board_size_x):
        y = head_y
        if not (0 <= x < board_size_x) or not (0 <= y < board_size_y):
            continue

        if x == board_size_x-1:
            board[y][x] = "W"
            continue

        for segment in game.snake:
            seg_x = int(segment.x // BLOCK_SIZE)
            seg_y = int(segment.y // BLOCK_SIZE)
            if seg_x == x and seg_y == y:
                board[y][x] = "S"
                break

        if board[y][x] == " ":
            if game.food_good1:
                food_x = int(game.food_good1.x // BLOCK_SIZE)
                food_y = int(game.food_good1.y // BLOCK_SIZE)
                if food_x == x and food_y == y:
                    board[y][x] = "G"

            if game.food_good2 and board[y][x] == " ":
                food_x = int(game.food_good2.x // BLOCK_SIZE)
                food_y = int(game.food_good2.y // BLOCK_SIZE)
                if food_x == x and food_y == y:
                    board[y][x] = "G"

            if game.food_bad and board[y][x] == " ":
                food_x = int(game.food_bad.x // BLOCK_SIZE)
                food_y = int(game.food_bad.y // BLOCK_SIZE)
                if food_x == x and food_y == y:
                    board[y][x] = "R"

            if board[y][x] == " ":
                board[y][x] = "0"

    lines = []
    for row in board:
        lines.append(''.join(row))

    return '\n'.join(lines)
