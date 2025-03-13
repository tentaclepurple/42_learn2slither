import argparse
import matplotlib.pyplot as plt
from IPython import display
import pygame
from environment import SnakeGameAI, SPEED
from agent import Agent
from utils import get_snake_vision_display


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
    parser = argparse.ArgumentParser(description='Train Snake AI')
    parser.add_argument('-save', type=str, help='Path to save model')
    parser.add_argument('-load', type=str, help='Path to load model')
    parser.add_argument('-speed', type=float, default=SPEED,
                        help='Game speed (0 for max)')
    parser.add_argument('-sessions', type=int, default=0,
                        help='Number of sessions (0 for infinite)')
    parser.add_argument('-visual', choices=['on', 'off'], default='on',
                        help='Enable visualization')
    parser.add_argument('-dontlearn', action='store_true',
                        help='Disable learning')
    parser.add_argument('-step', action='store_true', help='Step by step mode')
    parser.add_argument('-size', type=int, default=20,
                        help='Size of the board (number of cells)')
    parser.add_argument('-state', choices=['on', 'off'], default='off',
                        help='Show snake vision state in terminal')
    args = parser.parse_args()

    is_visual = args.visual == 'on'
    agent = Agent(load_model=args.load, dont_learn=args.dontlearn)
    game = SnakeGameAI(size=args.size, is_visual=is_visual)

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    sessions = 0

    print("Starting with configuration:")
    print(f" - Save model: {args.save or 'None'}")
    print(f" - Load model: {args.load or 'None'}")
    print(f" - Speed: {args.speed}")
    print(f" - Visual: {args.visual}")
    print(f" - Board size: {args.size} x {args.size}")
    print(f" - Learning: {'Disabled' if args.dontlearn else 'Enabled'}")
    print(f" - Step mode: {'Enabled' if args.step else 'Disabled'}")

    try:
        while True:
            if args.sessions > 0 and sessions >= args.sessions:
                break

            state_old = agent.get_state(game)

            final_move = agent.get_action(state_old)

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
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            agent.train_short_memory(state_old, final_move, reward,
                                     state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

            if args.state == 'on':
                print("\033[H\033[J", end="")
                print(get_snake_vision_display(game, 20))
                print(f"Score: {game.score} | "
                      f"Games: {agent.n_games} | "
                      f"Record: {record}")

            if args.speed > 0 and is_visual:
                game.clock.tick(args.speed)

            if done:
                # Game over - reset game
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                # save record
                if score > record:
                    record = score
                    if args.save:
                        print(f"New record! Saving model to {args.save}")
                        agent.model.save(args.save)

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                sessions += 1

                if args.speed == 0 and not is_visual:
                    if agent.n_games % 20 == 0:  # show every 20 games
                        print(f"Game {agent.n_games} | Score {score}, "
                              f"Record: {record} | Avg: {mean_score:.1f}")
                else:
                    print(f"Game {agent.n_games} | "
                          f"Score {score}, Record: {record}")

                if is_visual:
                    try:
                        plot(plot_scores, plot_mean_scores)
                    except Exception as e:
                        print(e)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if args.save:
            print(f"Saving final model to {args.save}")
            agent.model.save(args.save)

    print(f"Training completed: {agent.n_games} games, Record: {record}")


if __name__ == '__main__':
    train()
