# main.py
import argparse
import time
import sys
import pygame
from agent import QLearningAgent
from environment import SnakeEnvironment


def parse_arguments():
    """
    Analiza los argumentos de línea de comandos.
    
    Returns:
    argparse.Namespace: Objeto con los argumentos parseados
    """
    parser = argparse.ArgumentParser(description="Learn2Slither - Reinforcement Learning Snake")
    
    parser.add_argument("-sessions", type=int, default=1, 
                        help="Número de sesiones de entrenamiento")
    parser.add_argument("-save", type=str, default=None,
                        help="Guardar el modelo en el archivo especificado")
    parser.add_argument("-load", type=str, default=None,
                        help="Cargar el modelo desde el archivo especificado")
    parser.add_argument("-visual", type=str, choices=["on", "off"], default="on",
                        help="Habilitar/deshabilitar la visualización gráfica")
    parser.add_argument("-dontlearn", action="store_true", 
                        help="Desactivar el aprendizaje (solo usar el modelo cargado)")
    parser.add_argument("-step-by-step", action="store_true", 
                        help="Modo paso a paso (requiere presionar tecla para cada acción)")
    parser.add_argument("-speed", type=float, default=0.1,
                        help="Velocidad de visualización (tiempo entre pasos en segundos)")
    
    return parser.parse_args()


def print_state(state):
    """
    Imprime el estado (visión de la serpiente) en un formato legible.
    
    Parameters:
    state (str): Estado de la serpiente en formato de visión en 4 direcciones
    """
    directions = state.split(',')
    
    # Construir la representación visual
    print("\nVisión de la serpiente:")
    
    # Columna central para UP
    print(" " * 10 + "↑")
    for char in directions[0]:
        print(" " * 10 + char)
    
    # Fila central para LEFT, HEAD, RIGHT
    left_str = directions[2]
    right_str = directions[3]
    
    # Asegurar que ambas cadenas tienen la misma longitud
    max_len = max(len(left_str), len(right_str))
    left_str = left_str.ljust(max_len)
    right_str = right_str.ljust(max_len)
    
    print("← " + left_str + "H" + right_str + " →")
    
    # Columna central para DOWN
    for char in directions[1]:
        print(" " * 10 + char)
    print(" " * 10 + "↓")


def main():
    """
    Función principal del programa.
    """
    args = parse_arguments()
    
    # Configurar visualización
    display_enabled = args.visual == "on"
    
    # Inicializar entorno
    env = SnakeEnvironment(board_size=10, display_enabled=display_enabled)
    
    # Inicializar agente
    agent = QLearningAgent(verbose=True)
    
    # Cargar modelo si se especificó
    if args.load:
        try:
            agent.load_model(args.load)
            print(f"Modelo cargado desde {args.load}")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            return
    
    # Configurar modo dontlearn si se especificó
    if args.dontlearn:
        agent.dontlearn()
        print("Modo de aprendizaje desactivado (solo explotación)")
    
    # Estadísticas globales
    total_rewards = 0
    max_length = 0
    max_duration = 0
    
    # Bucle principal para todas las sesiones
    for session in range(args.sessions):
        print(f"\nIniciando sesión {session + 1}/{args.sessions}")
        
        # Reiniciar entorno y variables
        state = env.reset()
        agent.reset_session()
        session_reward = 0
        steps = 0
        running = True
        
        # Bucle de la sesión actual
        while running:
            # Imprimir estado actual (visión de la serpiente)
            if display_enabled or args.step_by_step:
                print_state(state)
            
            # Seleccionar acción según el estado actual
            action = agent.choose_action(state, training=not args.dontlearn)
            
            # Realizar la acción en el entorno
            next_state, reward, done, info = env.step(action)
            
            # Imprimir información de la acción
            if display_enabled or args.step_by_step:
                print(f"Acción: {action}, Recompensa: {reward}, Longitud: {info['length']}")
            
            # Actualizar agente con la experiencia
            agent.learn(state, action, reward, next_state)
            
            # Actualizar la longitud de la serpiente
            agent.update_snake_length(info["length"])
            
            # Actualizar estado actual
            state = next_state
            session_reward += reward
            steps += 1
            
            # Verificar fin de sesión
            if done:
                print(f"Sesión {session + 1} terminada. Motivo: {info.get('message', 'Desconocido')}")
                print(f"Recompensa total: {session_reward:.2f}")
                print(f"Pasos: {steps}")
                print(f"Longitud máxima: {agent.max_snake_length}")
                running = False
            
            # Actualizar estadísticas globales
            max_length = max(max_length, agent.max_snake_length)
            max_duration = max(max_duration, steps)
            
            # Modo paso a paso
            if args.step_by_step and not done:
                input("Presiona Enter para el siguiente paso...")
            elif display_enabled and not done:
                time.sleep(args.speed)
            
            # Manejar eventos de Pygame
            if display_enabled:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        running = False
                        return
        
        # Decrementar la exploración después de cada sesión
        agent.decay_exploration()
        
        # Actualizar recompensa total
        total_rewards += session_reward
    
    # Imprimir estadísticas finales
    print("\n===== Estadísticas finales =====")
    print(f"Sesiones completadas: {args.sessions}")
    print(f"Recompensa total: {total_rewards:.2f}")
    print(f"Recompensa promedio por sesión: {total_rewards / args.sessions:.2f}")
    print(f"Longitud máxima alcanzada: {max_length}")
    print(f"Duración máxima (pasos): {max_duration}")
    
    # Guardar modelo si se especificó
    if args.save:
        agent.save_model(args.save)
        print(f"Modelo guardado en {args.save}")
    
    # Cerrar entorno
    env.close()
    print("Programa finalizado")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrograma interrumpido por el usuario")
        sys.exit(0)