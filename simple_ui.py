"""
Simple Tkinter UI for Wake Word Detection.
Runs: realtime (live mic) and file-test (single WAV file).
"""
import sys
import subprocess
import threading
from pathlib import Path

# Ensure project root is in path
root = Path(__file__).parent
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, scrolledtext, messagebox
except ImportError:
    print("Tkinter not found. On Linux, install: sudo apt-get install python3-tk")
    sys.exit(1)


def run_realtime_subprocess(threshold, smoothing, output_queue, stop_event):
    """Run realtime in subprocess, pipe stdout to queue."""
    cmd = [
        sys.executable,
        str(root / "run_wakeword.py"),
        "realtime",
        "-t", str(threshold),
        "-s", str(smoothing),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_queue.put(("process", proc))
    try:
        for line in iter(proc.stdout.readline, ""):
            if stop_event.is_set():
                break
            if line:
                output_queue.put(("line", line.rstrip()))
    except Exception:
        pass
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
        output_queue.put(("done", None))


def poll_queue(root, output_queue, text_widget, process_holder):
    """Poll output queue and update text widget. Start/stop process handling."""
    try:
        while True:
            item = output_queue.get_nowait()
            if item[0] == "process":
                process_holder["proc"] = item[1]
            elif item[0] == "line":
                text_widget.insert(tk.END, item[1] + "\n")
                text_widget.see(tk.END)
            elif item[0] == "done":
                process_holder["proc"] = None
                break
    except Exception:
        pass
    root.after(100, lambda: poll_queue(root, output_queue, text_widget, process_holder))


class WakeWordUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Wake Word Detection — Hey Pakize")
        self.root.minsize(500, 450)
        self.root.geometry("580x520")

        self.realtime_proc = None
        self.realtime_stop = threading.Event()
        self.realtime_process_holder = {"proc": None}

        self._build_ui()

    def _build_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # ----- Realtime tab -----
        rt_frame = ttk.Frame(notebook, padding=8)
        notebook.add(rt_frame, text="Live Mic (Realtime)")

        rt_ctrl = ttk.Frame(rt_frame)
        rt_ctrl.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(rt_ctrl, text="Threshold:").pack(side=tk.LEFT, padx=(0, 4))
        self.threshold_var = tk.DoubleVar(value=0.70)
        self.threshold_spin = ttk.Spinbox(
            rt_ctrl, from_=0.1, to=0.99, increment=0.05,
            textvariable=self.threshold_var, width=6
        )
        self.threshold_spin.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(rt_ctrl, text="Smoothing (windows):").pack(side=tk.LEFT, padx=(0, 4))
        self.smoothing_var = tk.IntVar(value=2)
        self.smoothing_spin = ttk.Spinbox(
            rt_ctrl, from_=1, to=5, increment=1,
            textvariable=self.smoothing_var, width=4
        )
        self.smoothing_spin.pack(side=tk.LEFT, padx=(0, 12))

        self.rt_start_btn = ttk.Button(rt_ctrl, text="Start Listening", command=self._start_realtime)
        self.rt_start_btn.pack(side=tk.LEFT, padx=4)
        self.rt_stop_btn = ttk.Button(rt_ctrl, text="Stop", command=self._stop_realtime, state=tk.DISABLED)
        self.rt_stop_btn.pack(side=tk.LEFT)

        self.rt_output = scrolledtext.ScrolledText(rt_frame, height=14, state=tk.NORMAL, wrap=tk.WORD)
        self.rt_output.pack(fill=tk.BOTH, expand=True)
        self.rt_output.insert(tk.END, "Click 'Start Listening' to begin. Output appears here.\n")

        # ----- File Test tab -----
        ft_frame = ttk.Frame(notebook, padding=8)
        notebook.add(ft_frame, text="File Test")

        ft_ctrl = ttk.Frame(ft_frame)
        ft_ctrl.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(ft_ctrl, text="Threshold:").pack(side=tk.LEFT, padx=(0, 4))
        self.ft_threshold_var = tk.DoubleVar(value=0.70)
        ttk.Spinbox(
            ft_ctrl, from_=0.1, to=0.99, increment=0.05,
            textvariable=self.ft_threshold_var, width=6
        ).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(ft_ctrl, text="Smoothing:").pack(side=tk.LEFT, padx=(0, 4))
        self.ft_smoothing_var = tk.IntVar(value=2)
        ttk.Spinbox(
            ft_ctrl, from_=1, to=5, increment=1,
            textvariable=self.ft_smoothing_var, width=4
        ).pack(side=tk.LEFT, padx=(0, 12))

        self.ft_file_var = tk.StringVar()
        ttk.Entry(ft_ctrl, textvariable=self.ft_file_var, width=40).pack(side=tk.LEFT, padx=4)
        ttk.Button(ft_ctrl, text="Browse...", command=self._browse_file).pack(side=tk.LEFT)
        ttk.Button(ft_ctrl, text="Run Test", command=self._run_file_test).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(ft_ctrl, text="Test all (test_samples/)", command=self._run_folder_test).pack(side=tk.LEFT, padx=(8, 0))

        self.ft_output = scrolledtext.ScrolledText(ft_frame, height=14, state=tk.NORMAL, wrap=tk.WORD)
        self.ft_output.pack(fill=tk.BOTH, expand=True)
        self.ft_output.insert(tk.END, "Select a WAV file and click 'Run Test'.\n")

    def _get_threshold(self, tab="realtime"):
        try:
            return float(self.threshold_var.get() if tab == "realtime" else self.ft_threshold_var.get())
        except tk.TclError:
            return 0.70

    def _get_smoothing(self, tab="realtime"):
        try:
            return int(self.smoothing_var.get() if tab == "realtime" else self.ft_smoothing_var.get())
        except tk.TclError:
            return 2

    def _start_realtime(self):
        self.realtime_stop.clear()
        self.rt_output.delete(1.0, tk.END)
        self.rt_output.insert(tk.END, "Model is loading....\n")
        self.rt_output.see(tk.END)
        self.rt_start_btn.config(state=tk.DISABLED)
        self.rt_stop_btn.config(state=tk.NORMAL)
        self.root.update_idletasks()

        threshold = self._get_threshold("realtime")
        smoothing = self._get_smoothing("realtime")
        from queue import Queue
        self.rt_queue = Queue()
        self.rt_output_queue = self.rt_queue

        t = threading.Thread(
            target=run_realtime_subprocess,
            args=(threshold, smoothing, self.rt_queue, self.realtime_stop),
            daemon=True,
        )
        t.start()
        poll_queue(self.root, self.rt_queue, self.rt_output, self.realtime_process_holder)

    def _stop_realtime(self):
        self.realtime_stop.set()
        proc = self.realtime_process_holder.get("proc")
        if proc and proc.poll() is None:
            proc.terminate()
        self.rt_start_btn.config(state=tk.NORMAL)
        self.rt_stop_btn.config(state=tk.DISABLED)
        self.rt_output.insert(tk.END, "\nStopped.\n")

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            initialdir=str(root / "test_samples"),
        )
        if path:
            self.ft_file_var.set(path)

    def _run_file_test(self):
        path = self.ft_file_var.get().strip()
        if not path:
            messagebox.showwarning("File Test", "Please select a WAV file.")
            return

        self.ft_output.delete(1.0, tk.END)
        self.ft_output.insert(tk.END, f"Testing: {Path(path).name}\n")
        self.ft_output.update()

        def run():
            from wakeword.file_test import test_single_file
            threshold = self._get_threshold("file")
            smoothing = self._get_smoothing("file")
            result = test_single_file(path, threshold_override=threshold, smoothing_windows=smoothing)
            self.root.after(0, lambda: self._show_file_result(result, path))

        threading.Thread(target=run, daemon=True).start()

    def _show_file_result(self, result, path):
        if result.get("error"):
            self.ft_output.insert(tk.END, f"Error: {result['error']}\n")
            return
        conf = result.get("confidence_pct", 0)
        pred = result.get("prediction", "?")
        triggered = result.get("triggered", False)

        self.ft_output.insert(tk.END, f"  Confidence: {conf:.1f}%\n")
        self.ft_output.insert(tk.END, f"  Prediction: {pred}\n")
        if triggered:
            self.ft_output.insert(tk.END, "\n  *** Hey Pakize detected! ***\n")
        self.ft_output.see(tk.END)

    def _on_closing(self):
        self.realtime_stop.set()
        proc = self.realtime_process_holder.get("proc")
        if proc and proc.poll() is None:
            proc.terminate()
        self.root.destroy()

    def _run_folder_test(self):
        """Run file-test on entire test_samples folder with scorecard."""
        self.ft_output.delete(1.0, tk.END)
        self.ft_output.insert(tk.END, "Running on test_samples/ ...\n\n")
        self.ft_output.update()

        def run():
            from wakeword.file_test import run_file_test_with_scorecard
            threshold = self._get_threshold("file")
            smoothing = self._get_smoothing("file")
            results = run_file_test_with_scorecard(
                threshold_override=threshold,
                smoothing_windows=smoothing,
            )
            self.root.after(0, lambda: self._show_folder_results(results))

        threading.Thread(target=run, daemon=True).start()

    def _show_folder_results(self, results):
        if not results:
            self.ft_output.insert(tk.END, "No WAV files found in test_samples/ or folder missing.\n")
            self.ft_output.see(tk.END)
            return

        # Per-file output (old format)
        for r in results:
            self.ft_output.insert(tk.END, "-" * 50 + "\n")
            self.ft_output.insert(tk.END, f"File: {r['fname']}\n")
            self.ft_output.insert(tk.END, f"  Wake Confidence (max): {r['confidence_pct']:.1f}%\n")
            self.ft_output.insert(tk.END, f"  Prediction: {r['prediction']}\n")
            if r["prediction"] == "wake":
                self.ft_output.insert(tk.END, "\n  *** TRIGGERED: Hey Pakize detected! ***\n\n")
            self.ft_output.insert(tk.END, "\n")

        # Scorecard and wrong detections at the bottom
        self.ft_output.insert(tk.END, "\n" + "=" * 60 + "\n")
        self.ft_output.insert(tk.END, 'The "Hey Pakize" Scorecard\n')
        self.ft_output.insert(tk.END, "=" * 60 + "\n\n")

        wake_results = [r for r in results if r["ground_truth"] == "wake"]
        nonwake_results = [r for r in results if r["ground_truth"] == "nonwake"]

        wake_correct = sum(1 for r in wake_results if r["correct"])
        wake_fail = len(wake_results) - wake_correct
        nonwake_correct = sum(1 for r in nonwake_results if r["correct"])
        nonwake_fail = len(nonwake_results) - nonwake_correct
        total = len(results)
        total_correct = wake_correct + nonwake_correct
        total_fail = total - total_correct

        self.ft_output.insert(tk.END, "Category                    Total  Success  Fail\n")
        self.ft_output.insert(tk.END, "-" * 50 + "\n")
        self.ft_output.insert(tk.END, f"Wake (\"Hey Pakize\")          {len(wake_results):>5}  {wake_correct:>7}  {wake_fail} (Missed)\n")
        self.ft_output.insert(tk.END, f"Nonwake (Noise/Names)      {len(nonwake_results):>5}  {nonwake_correct:>7}  {nonwake_fail} (False Alarms)\n")
        self.ft_output.insert(tk.END, f"OVERALL                     {total:>5}  {total_correct:>7}  {total_fail}\n\n")

        wrong = [r for r in results if not r["correct"]]
        if wrong:
            self.ft_output.insert(tk.END, "Wrong detections:\n")
            self.ft_output.insert(tk.END, "-" * 50 + "\n")
            for r in wrong:
                if r["ground_truth"] == "wake":
                    self.ft_output.insert(tk.END, f"  MISSED: {r['fname']} (expected wake, got {r['prediction']})\n")
                else:
                    self.ft_output.insert(tk.END, f"  FALSE ALARM: {r['fname']} (expected nonwake, got {r['prediction']})\n")
        else:
            self.ft_output.insert(tk.END, "Wrong detections: None — all correct!\n")

        self.ft_output.see(tk.END)

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.mainloop()


if __name__ == "__main__":
    app = WakeWordUI()
    app.run()
