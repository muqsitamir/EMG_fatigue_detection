import sys
from pathlib import Path

# Add src to pythonpath so local dev works without installation if needed, 
# though editable install is preferred.
# sys.path.insert(0, str(Path(__file__).parent / "src"))

from emg_fatigue_detection.cli import main

if __name__ == "__main__":
    main()
