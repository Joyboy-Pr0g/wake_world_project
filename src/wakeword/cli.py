import sys
import argparse
import os
from pathlib import Path

if "wakeword" not in str(Path(__file__).resolve()):
    _src = Path(__file__).resolve().parent.parent
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))


def main():
    from wakeword.config import load_config, get_project_root
    from wakeword.dataset import build_dataset
    from wakeword.train import train_model
    from wakeword.collect import collect_hard_negatives
    from wakeword.realtime import run_realtime
    from wakeword.file_test import run_file_test
    from wakeword.evaluate import run_evaluate_command, run_scorecard_metrics
    from wakeword.mine_hard_negatives import run_mine_hard_negatives

    def cmd_dataset(args):
        load_config(args.config)
        build_dataset()

    def cmd_train(args):
        load_config(args.config)
        train_model()

    def cmd_realtime(args):
        load_config(args.config)
        run_realtime(
            threshold_override=getattr(args, "threshold", None),
            smoothing_windows=getattr(args, "smoothing", None),
        )

    def cmd_file_test(args):
        load_config(args.config)
        run_file_test(
            threshold_override=getattr(args, "threshold", None),
            smoothing_windows=getattr(args, "smoothing", 2),
        )

    def cmd_ui(args):
        root = get_project_root()
        simple_ui = root / "simple_ui.py"
        if not simple_ui.exists():
            raise FileNotFoundError(f"{simple_ui} not found")
        import subprocess
        subprocess.run([sys.executable, str(simple_ui)], cwd=str(root))

    def cmd_collect(args):
        load_config(args.config)
        collect_hard_negatives()

    def cmd_evaluate(args):
        load_config(args.config)
        run_evaluate_command()

    def cmd_scorecard(args):
        load_config(args.config)
        run_scorecard_metrics(
            threshold_override=getattr(args, "threshold", None),
            smoothing_windows=getattr(args, "smoothing", 2),
        )

    def cmd_mine(args):
        load_config(args.config)
        run_mine_hard_negatives(
            input_dir=getattr(args, "dir", None),
            confidence_threshold=getattr(args, "threshold", 0.5),
            dry_run=getattr(args, "dry_run", False),
        )

    parser = argparse.ArgumentParser(
        description="Wake word detection - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  dataset       Build dataset.csv from dataset/wake and dataset/nonwake
  train         Train ensemble model, save model.pkl, scaler.pkl, etc.
  evaluate      Generate ROC/PR curves, metrics report (requires trained model)
  scorecard     Metric dashboard from test_samples (Precision, Recall, F1, Confusion Matrix)
  realtime      Live microphone detection
  file-test     Test .wav files in test_samples/
  ui            Simple Tkinter GUI (realtime + file-test)
  collect       Copy hard_negatives.txt entries to hard_negatives/
  mine          Copy nonwake files with Wake Confidence > 50% to hard_negatives/
        """,
    )
    parser.add_argument("-c", "--config", default=None, help="Path to config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("dataset", help="Build dataset").set_defaults(func=cmd_dataset)
    subparsers.add_parser("train", help="Train model").set_defaults(func=cmd_train)
    p_realtime = subparsers.add_parser("realtime", help="Live mic detection")
    p_realtime.add_argument("-t", "--threshold", type=float, default=None, help="Threshold (recommended: 0.70)")
    p_realtime.add_argument("-s", "--smoothing", type=int, default=None, help="Consecutive windows required (default: 2)")
    p_realtime.set_defaults(func=cmd_realtime)
    p_ft = subparsers.add_parser("file-test", help="Test files in test_samples/")
    p_ft.add_argument("-t", "--threshold", type=float, default=None, help="Threshold (recommended: 0.70)")
    p_ft.add_argument("-s", "--smoothing", type=int, default=2, help="Consecutive windows required (default: 2)")
    p_ft.set_defaults(func=cmd_file_test)
    subparsers.add_parser("ui", help="Simple GUI for realtime + file-test").set_defaults(func=cmd_ui)
    subparsers.add_parser("collect", help="Copy hard_negatives.txt to hard_negatives/").set_defaults(func=cmd_collect)
    subparsers.add_parser("evaluate", help="Generate evaluation report").set_defaults(func=cmd_evaluate)
    p_scorecard = subparsers.add_parser("scorecard", help="Metric dashboard from test_samples")
    p_scorecard.add_argument("-t", "--threshold", type=float, default=None, help="Threshold")
    p_scorecard.add_argument("-s", "--smoothing", type=int, default=2, help="Consecutive windows")
    p_scorecard.set_defaults(func=cmd_scorecard)
    p_mine = subparsers.add_parser("mine", help="Copy nonwake FPs from test_samples to hard_negatives/")
    p_mine.add_argument("-d", "--dir", default=None, help="Input folder (default: test_samples)")
    p_mine.add_argument("-t", "--threshold", type=float, default=0.5, help="Min confidence to copy (default: 0.5)")
    p_mine.add_argument("--dry-run", action="store_true", help="Only print what would be copied")
    p_mine.set_defaults(func=cmd_mine)

    args = parser.parse_args()
    os.chdir(get_project_root())
    args.func(args)
