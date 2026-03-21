"""
Copy FP files from hard_negatives.txt into hard_negatives/ folder.
Run after train_model.py writes hard_negatives.txt, then re-run create_dataset.py and train_model.py.
"""
import os
import shutil

HARD_NEG_DIR = "hard_negatives"
HARD_NEG_TXT = "hard_negatives.txt"


def main():
    if not os.path.isfile(HARD_NEG_TXT):
        print(f"{HARD_NEG_TXT} not found. Run train_model.py first to generate it.")
        return
    os.makedirs(HARD_NEG_DIR, exist_ok=True)
    count = 0
    with open(HARD_NEG_TXT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            src = line.replace("/", os.sep)
            if not os.path.isfile(src):
                print(f"Skip (not found): {src}")
                continue
            fname = os.path.basename(src)
            dst = os.path.join(HARD_NEG_DIR, fname)
            if os.path.normpath(src) == os.path.normpath(dst):
                continue  # already in hard_negatives
            shutil.copy2(src, dst)
            count += 1
    print(f"Copied {count} files to {HARD_NEG_DIR}/")


if __name__ == "__main__":
    main()
