import argparse
import os

import pandas as pd

from utils import create_prediction_tables


def main():
    parser = argparse.ArgumentParser(
        description="Recompute out-of-fold overall metrics from a Preds/<run> directory."
    )
    parser.add_argument(
        "preds_dir",
        help="Directory containing predictions_fold1.csv and predictions_fold2.csv.",
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    preds_dir = os.path.abspath(args.preds_dir)
    os.chdir(repo_root)

    create_prediction_tables(preds_dir)
    overall = pd.read_csv(os.path.join(preds_dir, "overall_metrics.csv"))
    print("Out-of-fold overall metrics written.")
    print(f"Run dir: {preds_dir}")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
