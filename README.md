I had a great time building this!

🐍 Reinforcement Learning in Action: Meet Learn2Slither AKA The self learning snake.

The game consists in a self-learning agent navigating a board, eating green apples to grow while avoiding walls, its own body, and red apples (which make it shrink). The twist? The snake has no predefined rules — it learns optimal strategies entirely through interaction and feedback from the environment.

🧠 Key AI Concepts Used:
Q-Learning algorithm for experience-based learning through trial and error
Environment mapping to build a mental model of the board
Positional memory to avoid repetitive loops and dead ends
Free space analysis to assess future mobility
Heatmaps to discourage overvisited paths and promote exploration
Contextual reward systems that dynamically adapt to changing conditions

🔧 Tech Stack:
Python 🐍 as the core language, leveraging scientific libraries
PyTorch 🧠 as a Deep Learning module
Pygame 🎮 for real-time visual simulation
Modular architecture 🧩

🔥 What we see in the video (don't miss the beginning and the end):
Starts with random movements and no strategy.
Learns from rewards (e.g. green apples) and penalties (e.g. crashes, red apples).
Q-Learning updates its decisions over time.
Gradually masters efficient movement and survival.

💡 Why does this matter?
 Beyond games, these same techniques are powering next-gen industrial automation, robotic navigation, supply chain optimization, and smart resource allocation. Systems that adapt and learn in dynamic environments open powerful possibilities for business and industry.
