"""Collect hard negatives from hard_negatives.txt into hard_negatives/ folder."""
import os
import shutil
from pathlib import Path

from .config import load_config, get_project_root


def collect_hard_negatives():
    """Copy files listed in hard_negatives.txt to hard_negatives/ directory."""
    cfg = load_config()
    root = get_project_root()
    hn_dir = root / cfg["paths"]["hard_negatives"]
    hn_txt = root / cfg["paths"]["hard_negatives_txt"]

    if not hn_txt.is_file():
        print(f"{hn_txt.name} not found. Run train first to generate it.")
        return
    hn_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(hn_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            src = line.replace("/", os.sep)
            if not os.path.isfile(src):
                print(f"Skip (not found): {src}")
                continue
            fname = os.path.basename(src)
            dst = hn_dir / fname
            try:
                if Path(src).resolve() == Path(dst).resolve():
                    continue
            except OSError:
                pass
            try:
                shutil.copy2(src, dst)
            except shutil.SameFileError:
                continue
            count += 1
    print(f"Copied {count} files to {hn_dir}/")
