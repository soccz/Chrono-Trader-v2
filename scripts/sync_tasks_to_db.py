
import sys
import os
import sqlite3

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.task_manager import task_manager

def sync_tasks():
    print("Syncing tasks to dashboard DB...")
    
    # 1. Clear existing tasks (optional, strictly for demo/sync purposes to avoid dups)
    # conn = task_manager._get_conn()
    # conn.execute("DELETE FROM tasks")
    # conn.commit()
    # conn.close()
    
    tasks_to_add = [
        ("Implement Dual Notification Strategy", "Daily Deep Dive (8 AM) + 4H Continuous Scalp Report", "in_progress"),
        ("Refactor Telegram Bot", "Created structured HTML report with Trending/Pattern/Pump sections", "completed"),
        ("Fix Performance Data Loading", "Debugged empty performance page and verified with dummy data", "completed"),
        ("Site Layout Audit", "Fixed sidebar overlap issues in all templates", "completed"),
        ("Roadmap: Model Architecture Upgrade", "TCN, Market Filter, Performance Tracker", "completed")
    ]

    for title, desc, status in tasks_to_add:
        # Check if exists to avoid duplicates
        conn = task_manager._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM tasks WHERE title = ?", (title,))
        exists = cursor.fetchone()
        conn.close()
        
        if not exists:
            print(f"Adding: {title}")
            task_manager.add_task(title, desc, status)
        else:
            print(f"Skipping (exists): {title}")
            # Update status if needed
            task_manager.update_status(exists[0], status)

if __name__ == "__main__":
    sync_tasks()
