import os
import shutil
import glob
from datetime import datetime, timedelta

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECS_DIR = os.path.join(BASE_DIR, 'recommendations')
ARCHIVE_DIR = os.path.join(RECS_DIR, 'archive')

def get_file_date(filename):
    """
    Extracts date from filename.
    Formats expected: 
    - recs_daily_YYYYMMDD_HHMMSS.csv
    - recs_short_YYYYMMDD_HHMMSS.csv
    - pump_preds_YYYYMMDD_HHMMSS.csv
    """
    try:
        parts = filename.replace('.csv', '').split('_')
        for part in parts:
            if len(part) == 8 and part.isdigit():
                return datetime.strptime(part, "%Y%m%d")
    except:
        pass
    return None

def get_week_folder(date_obj):
    """
    Returns folder name strictly based on Monday start.
    ISO calendar uses Monday as start.
    Format: YYYY_Www (e.g., 2025_W01)
    """
    year, week, _ = date_obj.isocalendar()
    return f"{year}_W{week:02d}"

def run_archiving():
    print("Starting Weekly Data Archiving...")
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    
    # Get current week to exclude from archiving
    today = datetime.now()
    current_week_folder = get_week_folder(today)
    print(f"Current Week: {current_week_folder} (Files from this week will stay)")

    # Scan files
    files = glob.glob(os.path.join(RECS_DIR, "*.csv"))
    moved_count = 0
    
    for file_path in files:
        filename = os.path.basename(file_path)
        file_date = get_file_date(filename)
        
        if not file_date:
            continue
            
        file_week_folder = get_week_folder(file_date)
        
        # Archive if NOT current week
        if file_week_folder != current_week_folder:
            target_dir = os.path.join(ARCHIVE_DIR, file_week_folder)
            os.makedirs(target_dir, exist_ok=True)
            
            shutil.move(file_path, os.path.join(target_dir, filename))
            print(f"Moved {filename} -> {file_week_folder}")
            moved_count += 1
            
    print(f"Archiving Complete. Moved {moved_count} files.")

if __name__ == "__main__":
    run_archiving()
