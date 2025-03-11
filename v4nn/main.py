# main_optimized.py
import argparse
import time
import sys
import pygame
import matplotlib.pyplot as plt
from environment import SnakeEnvironment
from agent import OptimizedAgent

def parse_arguments():
    """Procesa argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description="Learn2Slither - Optimizado")
    
    parser.add_argument("-sessions", type=int, default=100, 
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
    parser.add_argument("-speed", type=float, default=0.05,
                        help="Velocidad de visualización")
    
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

def plot_training(scores, mean_scores, record):
    """Dibuja gráfico de entrenamiento en tiempo real"""
    plt.clf()
    plt.title('Entrenamiento Learn2Slither')
    plt.xlabel('Sesiones')
    plt.ylabel('Puntuación')
    plt.plot(scores, label='Puntuación')
    plt.plot(mean_scores, label='Media')
    plt.axhline(y=record, color='r', linestyle='-', label=f'Récord: {record}')
    plt.legend(loc='upper left')
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
    dontlearn = args.dontlearn
    
    # Inicializar entorno y agente
    env = SnakeEnvironment(board_size=10, display_enabled=display_enabled)
    agent = OptimizedAgent(verbose=True)
    
    # Cargar modelo
    if args.load:
        agent.load_model(args.load)
    
    # Estadísticas
    scores = []
    mean_scores = []
    total_score = 0
    record = agent.max_snake_length
    
    # Configurar gráfico
    plt.figure(figsize=(10, 5))
    plt.ion()
    
    # Bucle principal
    try:
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
                action = agent.choose_action(state, training=not dontlearn)
                
                # Ejecutar acción
                next_state, reward, done, info = env.step(action)
                
                # Mostrar información
                if display_enabled or args.step_by_step:
                    print(f"Acción: {action}, Recompensa: {reward}, Longitud: {info['length']}")
                
                # Aprender (si está habilitado)
                if not dontlearn:
                    agent.learn(state, action, reward, next_state)
                
                # Actualizar longitud
                agent.update_snake_length(info["length"])
                
                # Actualizar estado
                state = next_state
                session_reward += reward
                steps += 1
                
                # Verificar fin de sesión
                if done:
                    if not dontlearn:
                        agent.train_long_memory()
                        agent.decay_exploration()
                    
                    print(f"Sesión {session + 1} terminada: {info.get('message', 'Fin')}")
                    print(f"Recompensa: {session_reward:.2f}, Pasos: {steps}")
                    print(f"Longitud: {agent.snake_length}, Máxima: {agent.max_snake_length}")
                    
                    scores.append(agent.snake_length)
                    total_score += agent.snake_length
                    mean_score = total_score / (session + 1)
                    mean_scores.append(mean_score)
                    
                    if agent.snake_length > record:
                        record = agent.snake_length
                        # Guardar mejor modelo automáticamente
                        if not dontlearn and args.save:
                            best_model_path = f"{args.save}_best"
                            agent.save_model(best_model_path)
                            print(f"¡Nuevo récord! Modelo guardado en {best_model_path}")
                    
                    # Actualizar gráfico
                    plot_training(scores, mean_scores, record)
                    
                    running = False
                
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
                            if args.save:
                                agent.save_model(args.save)
                            plt.close()
                            return
        
        # Estadísticas finales
        print("\n===== Resultados =====")
        print(f"Sesiones: {args.sessions}")
        print(f"Longitud máxima: {record}")
        print(f"Longitud promedio: {mean_scores[-1] if mean_scores else 0:.2f}")
        
        # Guardar modelo final
        if args.save and not dontlearn:
            agent.save_model(args.save)
            print(f"Modelo guardado en {args.save}")
        
        # Cerrar
        env.close()
        
        # Mostrar gráfico final
        plt.ioff()
        plt.show()
        
    except KeyboardInterrupt:
        print("\nPrograma interrumpido")
        if args.save and not dontlearn:
            agent.save_model(args.save)
            print(f"Modelo guardado en {args.save}")
        plt.close()
        sys.exit(0)

if __name__ == "__main__":
    main()