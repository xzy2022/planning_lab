import datetime
import os
import sys

def append_log(tag, module, message):
    log_dir = "logs"
    log_file = os.path.join(log_dir, "dev_journal.md")
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    now = datetime.datetime.now().strftime("%H:%M")
    log_entry = f"[{now}] {tag} [{module}] {message}"
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

if __name__ == "__main__":
    if len(sys.argv) >= 4:
        append_log(sys.argv[1], sys.argv[2], sys.argv[3])
