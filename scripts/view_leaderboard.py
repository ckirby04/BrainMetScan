"""
View the model leaderboard.

Usage:
    python scripts/view_leaderboard.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from segmentation.leaderboard import Leaderboard

def main():
    project_dir = Path(__file__).parent.parent
    leaderboard = Leaderboard(str(project_dir / "model" / "leaderboard.json"))
    leaderboard.print_summary()

if __name__ == '__main__':
    main()
