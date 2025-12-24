import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    script_path = os.path.join(BASE_DIR, "modelling_tuning.py")
    cmd = [sys.executable, script_path]
    print(f"[INFO] Menjalankan script tuning: {cmd}")
    result = subprocess.run(cmd, check=True)
    print(f"[INFO] Selesai dengan return code {result.returncode}")

if __name__ == "__main__":
    main()
