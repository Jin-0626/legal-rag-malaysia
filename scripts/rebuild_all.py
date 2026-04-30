"""Rebuild generated corpus artifacts from raw PDFs through evaluation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    python_executable = sys.executable
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{project_root / 'src'}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(project_root / "src")
    )

    commands = [
        [python_executable, str(project_root / "scripts" / "export_law_chunks.py")],
        [python_executable, str(project_root / "scripts" / "index_law_chunks.py")],
        [python_executable, str(project_root / "scripts" / "evaluate_final_gold_set_v2.py")],
    ]

    for command in commands:
        print(f"Running: {' '.join(command)}")
        subprocess.run(command, cwd=project_root, env=env, check=True)


if __name__ == "__main__":
    main()
