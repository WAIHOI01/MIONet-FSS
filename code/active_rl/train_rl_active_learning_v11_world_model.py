"""Compatibility entry point: the v11 trainer now lives in the active_rl package."""
import os, sys
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
from active_rl.cli import main
if __name__ == "__main__":
    main()
