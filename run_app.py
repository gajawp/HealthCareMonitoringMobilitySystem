from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(project_root / "app" / "main.py"),
        ],
        cwd=project_root,
        check=True,
    )


if __name__ == "__main__":
    main()
