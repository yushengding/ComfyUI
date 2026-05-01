"""
ComfyUI lock mechanism — prevent multiple scripts from conflicting.

Usage in generation scripts:
    from comfyui_lock import acquire_lock, release_lock
    acquire_lock("building_gaps")   # blocks until lock is available
    ... generate images ...
    release_lock()

Or as context manager:
    with ComfyUILock("building_gaps"):
        ... generate images ...
"""
import os, time, json, sys

LOCK_FILE = os.path.join(os.path.dirname(__file__), ".comfyui_lock")

def acquire_lock(task_name, timeout=600, poll_interval=5):
    """Acquire lock. Blocks until available or timeout."""
    start = time.time()
    while True:
        if not os.path.exists(LOCK_FILE):
            # Create lock
            with open(LOCK_FILE, "w") as f:
                json.dump({"task": task_name, "pid": os.getpid(), "time": time.time()}, f)
            print("[LOCK] Acquired by: %s (pid %d)" % (task_name, os.getpid()))
            return True

        # Lock exists — check if stale (>30 min old)
        try:
            with open(LOCK_FILE, "r") as f:
                lock_info = json.load(f)
            age = time.time() - lock_info.get("time", 0)
            if age > 1800:  # 30 min stale
                print("[LOCK] Stale lock from '%s' (%.0f min old), stealing" % (lock_info.get("task"), age/60))
                os.remove(LOCK_FILE)
                continue

            if time.time() - start > timeout:
                print("[LOCK] Timeout waiting for lock held by '%s'" % lock_info.get("task"))
                return False

            print("[LOCK] Waiting... held by '%s' (pid %d, %.0fs ago)" % (
                lock_info.get("task", "?"), lock_info.get("pid", 0), age))
        except:
            # Corrupt lock file, remove it
            os.remove(LOCK_FILE)
            continue

        time.sleep(poll_interval)


def release_lock():
    """Release lock."""
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)
        print("[LOCK] Released")


class ComfyUILock:
    """Context manager for ComfyUI lock."""
    def __init__(self, task_name):
        self.task_name = task_name
    def __enter__(self):
        acquire_lock(self.task_name)
        return self
    def __exit__(self, *args):
        release_lock()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        if os.path.exists(LOCK_FILE):
            with open(LOCK_FILE) as f:
                info = json.load(f)
            age = time.time() - info.get("time", 0)
            print("LOCKED by '%s' (pid %d, %.0fs ago)" % (info.get("task"), info.get("pid"), age))
        else:
            print("UNLOCKED")
    elif len(sys.argv) > 1 and sys.argv[1] == "release":
        release_lock()
        print("Force released")
    else:
        print("Usage: python comfyui_lock.py [status|release]")
