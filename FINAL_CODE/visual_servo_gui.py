import threading
from typing import Optional

import tkinter as tk
from tkinter import messagebox

try:
    import swift  # type: ignore
except Exception:
    swift = None


def build_gui(demo):
    """Construct and launch the Tkinter GUI for the visual servo demo."""
    run_thread: Optional[threading.Thread] = None

    def start_demo():
        nonlocal run_thread
        if demo.thread_running or (run_thread and run_thread.is_alive()):
            messagebox.showinfo("Info", "Demo already running.")
            return
        if swift is None:
            messagebox.showerror("Swift Missing", "Swift simulator not available. Cannot start demo.")
            return
        run_thread = threading.Thread(target=demo.run, daemon=True)
        run_thread.start()
        print("[GUI] Started visual servo demo.")

    def stop_demo():
        demo.stop()
        messagebox.showinfo("Stopped", "Simulation stopped.")

    def emergency_stop():
        demo.emergency_stop()
        messagebox.showwarning("E-STOP", "Emergency Stop Activated!")

    def resume_operations():
        demo.resume()
        messagebox.showinfo("Resume", "Emergency Stop Released.")

    def open_teach_pendant():
        teach_window = tk.Toplevel()
        teach_window.title("Robot Teach Pendant")
        teach_window.geometry("600x650")

        tk.Label(teach_window, text="Robot Teach Pendant", font=("Arial", 16, "bold")).pack(pady=10)

        robot_frame = tk.Frame(teach_window)
        robot_frame.pack(pady=5)
        tk.Label(robot_frame, text="Select Robot:", font=("Arial", 11)).pack(side=tk.LEFT, padx=5)

        robot_var = tk.IntVar(value=0)
        tk.Radiobutton(robot_frame, text="PAROL6", variable=robot_var, value=0).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(robot_frame, text="LRMate200iC", variable=robot_var, value=1).pack(side=tk.LEFT, padx=5)

        joint_frame = tk.LabelFrame(teach_window, text="Joint Control (±0.1 rad)", font=("Arial", 11, "bold"))
        joint_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        def move_joint_gui(joint, delta):
            robot_idx = robot_var.get()
            if demo.move_joint(robot_idx, joint, delta):
                update_status()
            else:
                messagebox.showwarning("Error", "Unable to move joint. Is simulation running?")

        def create_joint_controls():
            for widget in joint_frame.winfo_children():
                widget.destroy()
            robot_idx = robot_var.get()
            joint_names = [
                "Joint 1", "Joint 2", "Joint 3", "Joint 4", "Joint 5", "Joint 6"
            ]
            if robot_idx == 0:
                joint_names.append("Rail")
            for i, name in enumerate(joint_names):
                frame = tk.Frame(joint_frame)
                frame.pack(pady=5, fill=tk.X, padx=10)
                tk.Label(frame, text=name, width=18, anchor='w').pack(side=tk.LEFT)
                tk.Button(frame, text="◀◀", command=lambda j=i: move_joint_gui(j, -0.1),
                          width=4, bg="lightblue").pack(side=tk.LEFT, padx=2)
                tk.Button(frame, text="◀", command=lambda j=i: move_joint_gui(j, -0.05),
                          width=4, bg="lightblue").pack(side=tk.LEFT, padx=2)
                tk.Button(frame, text="▶", command=lambda j=i: move_joint_gui(j, 0.05),
                          width=4, bg="lightblue").pack(side=tk.LEFT, padx=2)
                tk.Button(frame, text="▶▶", command=lambda j=i: move_joint_gui(j, 0.1),
                          width=4, bg="lightblue").pack(side=tk.LEFT, padx=2)

        def on_robot_change():
            create_joint_controls()
            update_status()

        for rb in robot_frame.winfo_children():
            if isinstance(rb, tk.Radiobutton):
                rb.config(command=on_robot_change)

        create_joint_controls()

        status_frame = tk.LabelFrame(teach_window, text="Robot Status", font=("Arial", 11, "bold"))
        status_frame.pack(pady=10, padx=20, fill=tk.BOTH)
        status_text = tk.Text(status_frame, height=8, width=60, font=("Courier", 9))
        status_text.pack(pady=5, padx=5)

        def update_status():
            robot_idx = robot_var.get()
            status = demo.get_robot_status(robot_idx)
            status_text.delete(1.0, tk.END)
            if status:
                robot_name = "PAROL6" if robot_idx == 0 else "LRMate200iC"
                status_text.insert(tk.END, f"{robot_name} Status\n")
                status_text.insert(tk.END, "="*50 + "\n")
                status_text.insert(tk.END, "Joint Configuration (rad):\n")
                for i, q in enumerate(status['joint_config']):
                    status_text.insert(tk.END, f"  q{i+1}: {q:.3f}\n")
                status_text.insert(tk.END, f"\nRecorded Positions: {len(demo.teach_positions)}\n")
            else:
                status_text.insert(tk.END, "Simulation not running or robot unavailable.")

        control_frame = tk.Frame(teach_window)
        control_frame.pack(pady=10)

        def record_pos():
            if demo.record_position(robot_var.get()):
                update_status()
                messagebox.showinfo("Success", "Position recorded!")
            else:
                messagebox.showwarning("Error", "Cannot record position. Is simulation running?")

        def clear_pos():
            if messagebox.askyesno("Confirm", "Clear all recorded positions?"):
                demo.clear_positions()
                update_status()

        def playback():
            if not demo.teach_positions:
                messagebox.showinfo("Info", "No positions recorded.")
                return
            if messagebox.askyesno("Playback", "Play back recorded positions?"):
                threading.Thread(target=demo.playback_positions, daemon=True).start()

        tk.Button(control_frame, text="📍 Record Position", command=record_pos,
                  bg="green", fg="white", font=("Arial", 11), width=18).pack(pady=3)
        tk.Button(control_frame, text="🔄 Refresh Status", command=update_status,
                  bg="blue", fg="white", font=("Arial", 11), width=18).pack(pady=3)
        tk.Button(control_frame, text="▶ Playback All", command=playback,
                  bg="purple", fg="white", font=("Arial", 11), width=18).pack(pady=3)
        tk.Button(control_frame, text="🗑️ Clear Positions", command=clear_pos,
                  bg="orange", fg="white", font=("Arial", 11), width=18).pack(pady=3)

        teach_window.after(500, update_status)

    root = tk.Tk()
    root.title("Visual Servo Control System")
    root.geometry("400x480")

    tk.Label(root, text="Visual Servo Control", font=("Arial", 14, "bold")).pack(pady=10)
    tk.Label(root, text="PAROL6 & LRMate200iC", font=("Arial", 10)).pack(pady=5)

    estop_frame = tk.Frame(root, bg="lightgray", relief=tk.RIDGE, bd=2)
    estop_frame.pack(pady=5, padx=20, fill=tk.X)
    tk.Label(estop_frame, text="⚠️ E-Stop: Press 'E' to stop | 'R' to resume",
             font=("Arial", 9, "bold"), bg="lightgray", fg="red").pack(pady=3)

    tk.Button(root, text="▶ Start Simulation", command=start_demo,
              bg="green", fg="white", font=("Arial", 12), width=20).pack(pady=10)
    tk.Button(root, text="🎮 Teach Pendant", command=open_teach_pendant,
              bg="blue", fg="white", font=("Arial", 12), width=20).pack(pady=5)
    tk.Button(root, text="🛑 EMERGENCY STOP", command=emergency_stop,
              bg="red", fg="white", font=("Arial", 12, "bold"), width=20).pack(pady=5)
    tk.Button(root, text="▶ Resume Operations", command=resume_operations,
              bg="orange", fg="white", font=("Arial", 12), width=20).pack(pady=5)
    tk.Button(root, text="⏹ Stop Simulation", command=stop_demo,
              bg="darkred", fg="white", font=("Arial", 12), width=20).pack(pady=5)

    tk.Label(root, text="Assignment 2 - Industrial Robotics", font=("Arial", 10, "italic")).pack(pady=10)

    root.mainloop()
