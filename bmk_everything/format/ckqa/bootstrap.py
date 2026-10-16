#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "framework"))
from automation.bootstrap_dataprep_eval import materialize
if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    print(materialize("ckqa", root / "data/cache/dataprep-bench", root / "data/datasets"))
