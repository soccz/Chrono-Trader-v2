import subprocess
import threading
import queue
import time
from datetime import datetime
from typing import Dict, Optional

class TaskRunner:
    """Manages background execution of main.py commands"""
    
    def __init__(self):
        self.current_process = None
        self.current_task_name = None
        self.current_task_key = None
        self.current_command = None
        self.log_queue = queue.Queue()
        self.status = "idle"  # idle, running, completed, failed
        self.start_time = None
        self.end_time = None
        self.last_returncode = None
        self._lock = threading.Lock()
    
    def start_task(
        self,
        command: list,
        task_name: str = "task",
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        task_key: Optional[str] = None,
    ):
        """
        Start a background task
        Args:
            command: List of command arguments (e.g., ['python', 'main.py', '--mode', 'train'])
            task_name: Human-readable task name for logging
        """
        with self._lock:
            if self.current_process and self.current_process.poll() is None:
                return {
                    "success": False,
                    "message": "A task is already running",
                    "task_name": self.current_task_name,
                    "task_key": self.current_task_key,
                }
            
            try:
                self.status = "running"
                self.start_time = datetime.now()
                self.end_time = None
                self.current_task_name = str(task_name)
                self.current_task_key = str(task_key or task_name).strip().lower().replace(" ", "_")
                self.current_command = list(command)
                self.last_returncode = None
                
                # Clear old logs
                while not self.log_queue.empty():
                    self.log_queue.get()
                
                # Start subprocess
                self.current_process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    cwd=cwd,
                    env=env,
                )
                
                # Start log capture thread
                log_thread = threading.Thread(target=self._capture_logs, daemon=True)
                log_thread.start()
                
                # Start monitor thread
                monitor_thread = threading.Thread(target=self._monitor_process, daemon=True)
                monitor_thread.start()
                
                return {
                    "success": True,
                    "message": f"{task_name} started",
                    "task_name": self.current_task_name,
                    "task_key": self.current_task_key,
                }
            
            except Exception as e:
                self.status = "failed"
                self.current_task_name = str(task_name)
                self.current_task_key = str(task_key or task_name).strip().lower().replace(" ", "_")
                self.current_command = list(command)
                self.last_returncode = None
                return {
                    "success": False,
                    "message": str(e),
                    "task_name": self.current_task_name,
                    "task_key": self.current_task_key,
                }
    
    def _capture_logs(self):
        """Capture stdout/stderr from subprocess"""
        try:
            for line in iter(self.current_process.stdout.readline, ''):
                if line:
                    self.log_queue.put(line.rstrip())
        except Exception as e:
            self.log_queue.put(f"Error capturing logs: {str(e)}")
    
    def _monitor_process(self):
        """Monitor process completion"""
        try:
            returncode = self.current_process.wait()
            self.end_time = datetime.now()
            self.last_returncode = int(returncode)
            
            if returncode == 0:
                self.status = "completed"
                self.log_queue.put("[SYSTEM] Task completed successfully")
            else:
                self.status = "failed"
                self.log_queue.put(f"[SYSTEM] Task failed with exit code {returncode}")
        except Exception as e:
            self.status = "failed"
            self.log_queue.put(f"[SYSTEM] Error monitoring process: {str(e)}")
    
    def get_status(self):
        """Get current task status"""
        with self._lock:
            return {
                "status": self.status,
                "task_name": self.current_task_name,
                "task_key": self.current_task_key,
                "command": list(self.current_command) if self.current_command else None,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "returncode": self.last_returncode,
                "running": self.current_process and self.current_process.poll() is None
            }
    
    def get_logs(self, max_lines: int = 100):
        """Get recent logs from queue"""
        logs = []
        temp_logs = []
        
        # Get all available logs
        while not self.log_queue.empty() and len(temp_logs) < max_lines:
            try:
                log = self.log_queue.get_nowait()
                temp_logs.append(log)
            except queue.Empty:
                break
        
        # Put them back for SSE streaming
        for log in temp_logs:
            self.log_queue.put(log)
        
        return temp_logs
    
    def stream_logs(self):
        """Generator for SSE log streaming"""
        while True:
            try:
                log = self.log_queue.get(timeout=1)
                yield f"data: {log}\n\n"
            except queue.Empty:
                # Check if process ended
                if self.status in ["completed", "failed"] and self.log_queue.empty():
                    yield f"data: [STREAM_END]\n\n"
                    break
                else:
                    yield f"data: [HEARTBEAT]\n\n"
