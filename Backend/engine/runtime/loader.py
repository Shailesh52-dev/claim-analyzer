import threading
from engine.runtime.core import build_runtime

_runtime = None
_lock = threading.Lock()

def get_runtime():
    global _runtime
    if _runtime:
        return _runtime

    with _lock:
        if _runtime:
            return _runtime

        print("⏳ Loading Fact Analyzer runtime (one-time)")

        from engine.runtime.core import build_runtime
        _runtime = build_runtime()

        print("✅ Runtime ready")

    return _runtime