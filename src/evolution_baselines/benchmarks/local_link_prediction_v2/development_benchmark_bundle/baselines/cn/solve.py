import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from target_runtime import solve
from solution import score_candidates

if __name__ == '__main__':
    solve(score_candidates)
