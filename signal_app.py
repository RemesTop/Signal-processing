import sys
from pathlib import Path

# Add project root and src directory to sys.path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir))

from src.signal_app import SignalApp, main

if __name__ == "__main__":
    main()
