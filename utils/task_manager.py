import sqlite3
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TaskManager:
    def __init__(self, db_path="data/tasks.db"):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT,
                    status TEXT DEFAULT 'planned', -- planned, in_progress, completed
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to init Task DB: {e}")

    def add_task(self, title, description="", status="planned"):
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO tasks (title, description, status)
                VALUES (?, ?, ?)
            """, (title, description, status))
            conn.commit()
            task_id = cursor.lastrowid
            conn.close()
            return task_id
        except Exception as e:
            logger.error(f"Failed to add task: {e}")
            return None

    def get_tasks(self):
        try:
            conn = self._get_conn()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tasks ORDER BY created_at DESC")
            rows = cursor.fetchall()
            conn.close()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get tasks: {e}")
            return []

    def update_status(self, task_id, status):
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE tasks
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (status, task_id))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to update task: {e}")
            return False

# Global Instance
task_manager = TaskManager()
