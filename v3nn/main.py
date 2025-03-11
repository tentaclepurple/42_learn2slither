# main_pytorch.py
import argparse
import time
import sys
import pygame
import matplotlib.pyplot as plt
import numpy as np
from environment import SnakeEnvironment
from agent import SnakeAgent

def parse_arguments():
    """Procesa argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description="Learn2Slither - PyTorch RL")
    
    parser.add_argument("-sessions", type=int, default=1, 
                        help="Número de sesiones de entrenamiento")
    parser.add_argument("-save", type=str, default=None,
                        help="Guardar modelo en archivo")
    parser.add_argument("-load", type=str, default=None,
                        help="Cargar modelo desde archivo")
    parser.add_argument("-visual", type=str, choices=["on", "off"], default="on",
                        help="Activar/desactivar visualización")
    parser.add_argument("-dontlearn", action="store_true", 
                        help="Desactivar aprendizaje")
    parser.add_argument("-step-by-step", action="store_true", 
                        help="Modo paso a paso")
    parser.add_argument("-speed", type=float, default=0.1,
                        help="Velocidad de visualización")
    parser.add_argument("-plot", action="store_true",
                        help="Mostrar gráfico de progreso")
    
    return parser.parse_args()

def print_state(state):
    """Muestra visualmente el estado de la serpiente"""
    directions = state.split(',')
    
    print("\nVisión de la serpiente:")
    
    # UP
    print(" " * 10 + "↑")
    for char in directions[0][:10]:
        print(" " * 10 + char)
    
    # LEFT, HEAD, RIGHT
    left_str = directions[2][:10]
    right_str = directions[3][:10]
    max_len = max(len(left_str), len(right_str))
    left_str = left_str.ljust(max_len)
    right_str = right_str.ljust(max_len)
    
    print("← " + left_str + "H" + right_str + " →")
    
    # DOWN
    for char in directions[1][:10]:
        print(" " * 10 + char)
    print(" " * 10 + "↓")

def plot_progress(scores, mean_scores):
    """Dibuja gráfico de progreso del entrenamiento"""
    plt.figure(1)
    plt.clf()
    plt.title('Entrenamiento Learn2Slither')
    plt.xlabel('Sesiones')
    plt.ylabel('Puntuación')
    plt.plot(scores, label='Puntuación')
    plt.plot(mean_scores, label='Media móvil')
    plt.legend()
    plt.ylim(ymin=0)
    
    if len(scores) > 0:
        plt.text(len(scores)-1, scores[-1], f"{scores[-1]}")
    if len(mean_scores) > 0:
        plt.text(len(mean_scores)-1, mean_scores[-1], f"{mean_scores[-1]:.1f}")
    
    plt.pause(0.001)

def main():
    """Función principal del programa"""
    args = parse_arguments()
    
    # Configuración
    display_enabled = args.visual == "on"
    plot_enabled = args.plot
    
    # Inicializar entorno y agente
    env = SnakeEnvironment(board_size=10, display_enabled=display_enabled)
    agent = SnakeAgent(state_size=40, hidden_size=256, verbose=True)
    
    # Cargar modelo
    if args.load:
        agent.load_model(args.load)
    
    # Modo sin aprendizaje
    if args.dontlearn:
        agent.dontlearn()
        print("Modo de explotación activado (sin aprendizaje)")
    
    # Estadísticas
    scores = []
    mean_scores = []
    total_score = 0
    max_length = 0
    max_duration = 0
    
    # Inicializar gráfico
    if plot_enabled:
        plt.ion()
    
    # Bucle principal
    for session in range(args.sessions):
        print(f"\nSesión {session + 1}/{args.sessions}")
        
        # Inicializar sesión
        state = env.reset()
        agent.reset_session()
        session_reward = 0
        steps = 0
        running = True
        
        # Bucle de sesión
        while running:
            # Mostrar estado
            if display_enabled or args.step_by_step:
                print_state(state)
            
            # Elegir acción
            action = agent.choose_action(state, training=not args.dontlearn)
            
            # Ejecutar acción
            next_state, reward, done, info = env.step(action)
            
            # Mostrar información
            if display_enabled or args.step_by_step:
                print(f"Acción: {action}, Recompensa: {reward}, Longitud: {info['length']}")
            
            # Aprender
            agent.learn(state, action, reward, next_state)
            
            # Actualizar longitud
            agent.update_snake_length(info["length"])
            
            # Actualizar estado
            state = next_state
            session_reward += reward
            steps += 1
            
            # Verificar fin de sesión
            if done:
                print(f"Sesión {session + 1} terminada: {info.get('message', 'Desconocido')}")
                print(f"Recompensa: {session_reward:.2f}, Pasos: {steps}")
                print(f"Longitud máxima: {agent.max_snake_length}")
                running = False
            
            # Actualizar estadísticas
            max_length = max(max_length, agent.max_snake_length)
            max_duration = max(max_duration, steps)
            
            # Control de visualización
            if args.step_by_step and not done:
                input("Presiona Enter para continuar...")
            elif display_enabled and not done:
                time.sleep(args.speed)
            
            # Eventos Pygame
            if display_enabled:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        running = False
                        if plot_enabled:
                            plt.close()
                        return
        
        # Actualizar estadísticas al final de la sesión
        scores.append(agent.snake_length)
        total_score += agent.snake_length
        mean_score = total_score / (session + 1)
        mean_scores.append(mean_score)
        
        # Actualizar gráfico
        if plot_enabled:
            plot_progress(scores, mean_scores)
        
        # Reducir exploración
        agent.decay_exploration()
    
    # Estadísticas finales
    print("\n===== Resultados =====")
    print(f"Sesiones: {args.sessions}")
    print(f"Recompensa total: {total_score:.2f}")
    print(f"Recompensa promedio: {total_score / args.sessions:.2f}")
    print(f"Longitud máxima: {max_length}")
    print(f"Duración máxima: {max_duration}")
    
    # Guardar modelo
    if args.save:
        agent.save_model(args.save)
        print(f"Modelo guardado en {args.save}")
    
    # Cerrar
    env.close()
    if plot_enabled:
        plt.ioff()
        plt.show()
    print("Programa finalizado")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrograma interrumpido")
        plt.close('all')
        sys.exit(0)