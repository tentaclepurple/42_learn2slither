import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import threading


class SnakeAILauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Snake AI Launcher")
        self.root.geometry("600x650")
        self.root.resizable(False, False)

        # Variables
        self.mode_var = tk.StringVar(value="train")
        self.size_var = tk.IntVar(value=20)
        self.speed_var = tk.DoubleVar(value=20)
        self.sessions_var = tk.IntVar(value=0)
        self.visual_var = tk.BooleanVar(value=True)
        self.dontlearn_var = tk.BooleanVar(value=False)
        self.load_model_path = tk.StringVar(value="")
        self.save_model_path = tk.StringVar(value="")

        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="Learn2Slither AI Launcher",
                                font=("Arial", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # Mode selection
        mode_frame = ttk.LabelFrame(main_frame, text="Mode", padding=10)
        mode_frame.grid(row=1, column=0, columnspan=3, sticky="ew",
                        pady=(0, 10))

        train_radio = ttk.Radiobutton(mode_frame, text="Train",
                                      variable=self.mode_var,
                                      value="train",
                                      command=self.update_ui)
        train_radio.grid(row=0, column=0, padx=(0, 20))

        play_radio = ttk.Radiobutton(mode_frame, text="Play (IA)",
                                     variable=self.mode_var,
                                     value="play",
                                     command=self.update_ui)
        play_radio.grid(row=0, column=1, padx=(0, 20))

        manual_radio = ttk.Radiobutton(mode_frame, text="Play (Human)",
                                       variable=self.mode_var,
                                       value="manual",
                                       command=self.update_ui)
        manual_radio.grid(row=0, column=2)

        # Board size
        size_frame = ttk.LabelFrame(main_frame, text="Board size",
                                    padding=10)
        size_frame.grid(row=2, column=0, columnspan=3,
                        sticky="ew",
                        pady=(0, 10))

        size_scale = ttk.Scale(size_frame, from_=10, to=50,
                               variable=self.size_var,
                               orient=tk.HORIZONTAL)
        size_scale.grid(row=0, column=0, sticky="ew")

        size_label = ttk.Label(size_frame, textvariable=self.size_var)
        size_label.grid(row=0, column=1, padx=(10, 0))

        # Speed
        speed_frame = ttk.LabelFrame(main_frame, text="Speed", padding=10)
        speed_frame.grid(row=3, column=0, columnspan=3, sticky="ew",
                         pady=(0, 10))

        speed_scale = ttk.Scale(speed_frame, from_=0, to=100,
                                variable=self.speed_var,
                                orient=tk.HORIZONTAL)
        speed_scale.grid(row=0, column=0, sticky="ew")

        speed_label = ttk.Label(speed_frame, textvariable=self.speed_var)
        speed_label.grid(row=0, column=1, padx=(10, 0))

        ttk.Label(speed_frame, text="(0 = Max speed)").grid(row=1,
                                                            column=0,
                                                            sticky="w")

        # Sessions
        self.sessions_frame = ttk.LabelFrame(main_frame, text="Sessions",
                                             padding=10)
        self.sessions_frame.grid(row=4, column=0, columnspan=3, sticky="ew",
                                 pady=(0, 10))

        sessions_entry = ttk.Entry(self.sessions_frame,
                                   textvariable=self.sessions_var,
                                   width=10)
        sessions_entry.grid(row=0, column=0, sticky="w")

        ttk.Label(self.sessions_frame, text="(0 = infnite)").grid(row=0,
                                                                  column=1,
                                                                  sticky="w",
                                                                  padx=(10, 0))

        # Options
        self.options_frame = ttk.LabelFrame(main_frame, text="Options",
                                            padding=10)
        self.options_frame.grid(row=5, column=0, columnspan=3, sticky="ew",
                                pady=(0, 10))

        visual_check = ttk.Checkbutton(self.options_frame,
                                       text="Visualization",
                                       variable=self.visual_var)
        visual_check.grid(row=0, column=0, sticky="w", padx=(0, 20))

        dontlearn_check = ttk.Checkbutton(self.options_frame,
                                          text="Don't learn",
                                          variable=self.dontlearn_var)
        dontlearn_check.grid(row=0, column=1, sticky="w")

        # Load model
        self.load_frame = ttk.LabelFrame(main_frame, text="Load Model",
                                         padding=10)
        self.load_frame.grid(row=6, column=0, columnspan=3, sticky="ew",
                             pady=(0, 10))

        load_entry = ttk.Entry(self.load_frame,
                               textvariable=self.load_model_path, width=40)
        load_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        load_button = ttk.Button(self.load_frame, text="Search",
                                 command=self.browse_load_model)
        load_button.grid(row=0, column=1)

        # Save model
        self.save_frame = ttk.LabelFrame(main_frame, text="Save Model",
                                         padding=10)
        self.save_frame.grid(row=7, column=0, columnspan=3, sticky="ew",
                             pady=(0, 10))

        save_entry = ttk.Entry(self.save_frame,
                               textvariable=self.save_model_path, width=40)
        save_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        save_button = ttk.Button(self.save_frame, text="Examinar",
                                 command=self.browse_save_model)
        save_button.grid(row=0, column=1)

        # Manual mode info
        self.manual_frame = ttk.LabelFrame(main_frame,
                                           text="Human Mode Instructions",
                                           padding=10)
        self.manual_frame.grid(row=8, column=0, columnspan=3, sticky="ew",
                               pady=(0, 10))

        manual_text = "Controls: Arrow keys\n" \
            "Objective: Eat the green food\nAvoid: Red food and walls\n" \

        ttk.Label(self.manual_frame, text=manual_text).grid(row=0, column=0,
                                                            sticky="w")

        # Launch button
        launch_button = ttk.Button(main_frame, text="¡Iniciar!",
                                   command=self.launch_snake_ai,
                                   style="Accent.TButton")
        launch_button.grid(row=9, column=0, columnspan=3, pady=(10, 0))

        # Status label
        self.status_var = tk.StringVar(value="Listo para iniciar")
        status_label = ttk.Label(main_frame, textvariable=self.status_var,
                                 font=("Arial", 10, "italic"))
        status_label.grid(row=10, column=0, columnspan=3, pady=(10, 0))

        # Configure grid
        main_frame.columnconfigure(0, weight=1)
        size_frame.columnconfigure(0, weight=1)
        speed_frame.columnconfigure(0, weight=1)
        self.load_frame.columnconfigure(0, weight=1)
        self.save_frame.columnconfigure(0, weight=1)

        # Apply styles
        self.apply_styles()

        # Initial UI update
        self.update_ui()

    def apply_styles(self):
        # Create a modern style
        style = ttk.Style()

        # Configure frame padding
        style.configure("TLabelframe", padding=10)

        # Configure button style
        style.configure("TButton", padding=5)
        style.configure("Accent.TButton", padding=10, font=("Arial",
                                                            12,
                                                            "bold"))

    def update_ui(self):
        mode = self.mode_var.get()

        if mode == "play":
            self.dontlearn_var.set(True)
            self.sessions_var.set(1)
            self.sessions_frame.grid()
            self.options_frame.grid()
            self.load_frame.grid()
            self.save_frame.grid()
            self.manual_frame.grid_remove()

        elif mode == "train":
            self.sessions_frame.grid()
            self.options_frame.grid()
            self.load_frame.grid()
            self.save_frame.grid()
            self.manual_frame.grid_remove()

        elif mode == "manual":
            self.sessions_frame.grid_remove()
            self.options_frame.grid_remove()
            self.load_frame.grid_remove()
            self.save_frame.grid_remove()
            self.manual_frame.grid()

    def browse_load_model(self):
        filename = filedialog.askopenfilename(
            title="Seleccionar modelo para cargar",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
        )
        if filename:
            self.load_model_path.set(filename)

    def browse_save_model(self):
        filename = filedialog.asksaveasfilename(
            title="Seleccionar dónde guardar el modelo",
            defaultextension=".pth",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
        )
        if filename:
            self.save_model_path.set(filename)

    def launch_snake_ai(self):
        mode = self.mode_var.get()

        if mode == "play" and not self.load_model_path.get():
            messagebox.showerror("Error", "You need to load a model.")
            return

        if mode == "manual":
            cmd = ["python3", "learn2slither_manual.py"]
            cmd.extend(["-size", str(self.size_var.get())])
            cmd.extend(["-speed", str(int(self.speed_var.get()))])
        else:
            cmd = ["python3", "learn2slither.py"]

            cmd.extend(["-size", str(self.size_var.get())])
            cmd.extend(["-speed", str(self.speed_var.get())])
            cmd.extend(["-sessions", str(self.sessions_var.get())])
            cmd.extend(["-visual", "on" if self.visual_var.get() else "off"])

            if self.dontlearn_var.get():
                cmd.append("-dontlearn")

            if self.load_model_path.get():
                cmd.extend(["-load", self.load_model_path.get()])

            if self.save_model_path.get():
                cmd.extend(["-save", self.save_model_path.get()])

        self.status_var.set("Iniciando Snake...")

        threading.Thread(target=self.run_process, args=(cmd,),
                         daemon=True).start()

    def run_process(self, cmd):
        try:
            print("Running:", " ".join(cmd))

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )

            self.root.after(0, lambda: self.status_var.set(
                "Learn2Slither running..."
                ))

            stdout, stderr = process.communicate()

            if process.returncode == 0:
                self.root.after(0, lambda: self.status_var.set(
                    "Completed"
                    ))
            else:
                self.root.after(0, lambda: self.status_var.set(
                    f"Error: {stderr}"
                    ))
                print("Error:", stderr)

        except Exception as e:
            self.root.after(0, lambda: self.status_var.set("Error"))
            print("Excepción:", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = SnakeAILauncher(root)
    root.mainloop()
