import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    evaluator = Path(__file__).with_name("evaluation_runner.py")
    subprocess.run(
        [sys.executable, str(evaluator), "--split", "validation", *sys.argv[1:]],
        check=True,
    )
