#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "framework"))
from core.posttrainbench_inspect import main

if __name__ == "__main__":
    main("humaneval")
