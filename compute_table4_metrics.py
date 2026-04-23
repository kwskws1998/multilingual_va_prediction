import argparse
import os
from pathlib import Path

import pandas as pd

from utils import create_prediction_tables


DISPLAY_NAME_BY_DATASET = {
    "Emobank": "EmoBank",
    "IEMOCAP sentences": "IEMOCAP",
    "fb": "Facebook Posts",
    "EmoTales sentences": "EmoTales",
    "ANET sentences": "ANET",
    "PANIG sentences": "PANIG",
    "COMETA sentences": "COMETA sent.",
    "COMETA stories": "COMETA stories",
    "CVAT": "CVAT",
    "CVAI": "CVAI",
    "Polish sentences": "ANPST",
    "MAS": "MAS",
    "Cantonese Nouns": "Yee",
    "CroatianNorms": "Coso et al.",
    "DutchAdj": "Moors et al.",
    "word ratings NL": "Verheyen et al.",
    "nrc-vad": "NRC-VAD",
    "word ratings ENG": "Warriner et al.",
    "GlasgowNorms": "Scott et al.",
    "FinnishNorms": "Soderholm et al.",
    "FinnishNouns": "Eilola et al.",
    "FAN - french words": "FAN",
    "FEEL": "FEEL",
    "BAWL_R": "BAWL-R",
    "ANGST": "ANGST",
    "German words": "LANG",
    "Italian words": "Italian ANEW",
    "Chinese words": "Xu et al.",
    "ChineseW11k": "CVAW",
    "ANPW_R": "ANPW_R",
    "NAWL": "NAWL",
    "ANEW to EP": "Portuguese ANEW",
    "word ratings ES": "S.-Gonzalez et al.",
    "TurkishNorms": "Kapucu et al.",
}

LANGUAGE_BY_DATASET = {
    "ANGST": "German",
    "BAWL_R": "German",
    "German words": "German",
    "COMETA sentences": "German",
    "COMETA stories": "German",
    "PANIG sentences": "German",
    "ANPW_R": "Polish",
    "NAWL": "Polish",
    "Polish sentences": "Polish",
    "Chinese words": "Mandarin",
    "ChineseW11k": "Mandarin",
    "CVAI": "Mandarin",
    "CVAT": "Mandarin",
    "FAN - french words": "French",
    "FEEL": "French",
    "Italian words": "Italian",
    "CroatianNorms": "Croatian",
    "FinnishNorms": "Finnish",
    "FinnishNouns": "Finnish",
    "TurkishNorms": "Turkish",
    "word ratings NL": "Dutch",
    "DutchAdj": "Dutch",
    "GlasgowNorms": "English",
    "nrc-vad": "English",
    "word ratings ENG": "English",
    "ANET sentences": "English",
    "Emobank": "English",
    "EmoTales sentences": "English",
    "fb": "English",
    "IEMOCAP sentences": "English",
    "word ratings ES": "Spanish",
    "Cantonese Nouns": "Cantonese",
    "ANEW to EP": "Portuguese",
    "MAS": "Portuguese",
}

WORDS_DATASETS = [
    "ANEW to EP",
    "ANGST",
    "ANPW_R",
    "BAWL_R",
    "Cantonese Nouns",
    "Chinese words",
    "ChineseW11k",
    "CroatianNorms",
    "DutchAdj",
    "FAN - french words",
    "FEEL",
    "FinnishNorms",
    "FinnishNouns",
    "German words",
    "GlasgowNorms",
    "Italian words",
    "NAWL",
    "nrc-vad",
    "TurkishNorms",
    "word ratings NL",
    "word ratings ES",
    "word ratings ENG",
]

SHORT_TEXT_DATASETS = [
    "ANET sentences",
    "CVAI",
    "CVAT",
    "COMETA sentences",
    "COMETA stories",
    "Emobank",
    "EmoTales sentences",
    "fb",
    "IEMOCAP sentences",
    "MAS",
    "PANIG sentences",
    "Polish sentences",
]

SHORT_TEXT_ORDER = [
    "Emobank",
    "IEMOCAP sentences",
    "fb",
    "EmoTales sentences",
    "ANET sentences",
    "PANIG sentences",
    "COMETA sentences",
    "COMETA stories",
    "CVAT",
    "CVAI",
    "Polish sentences",
    "MAS",
]

WORD_ORDER = [
    "Cantonese Nouns",
    "CroatianNorms",
    "DutchAdj",
    "word ratings NL",
    "nrc-vad",
    "word ratings ENG",
    "GlasgowNorms",
    "FinnishNorms",
    "FinnishNouns",
    "FAN - french words",
    "FEEL",
    "BAWL_R",
    "ANGST",
    "German words",
    "Italian words",
    "Chinese words",
    "ChineseW11k",
    "ANPW_R",
    "NAWL",
    "ANEW to EP",
    "word ratings ES",
    "TurkishNorms",
]


def _paper_name(dataset_name):
    return DISPLAY_NAME_BY_DATASET.get(dataset_name, dataset_name)


def _dataset_type(dataset_name):
    if dataset_name in SHORT_TEXT_DATASETS:
        return "short_text"
    if dataset_name in WORDS_DATASETS:
        return "word"
    return "unknown"


def _order_key(dataset_name):
    if dataset_name in SHORT_TEXT_ORDER:
        return (0, SHORT_TEXT_ORDER.index(dataset_name))
    if dataset_name in WORD_ORDER:
        return (1, WORD_ORDER.index(dataset_name))
    if dataset_name in SHORT_TEXT_DATASETS:
        return (0, len(SHORT_TEXT_ORDER))
    if dataset_name in WORDS_DATASETS:
        return (1, len(WORD_ORDER))
    return (2, dataset_name)


def _to_markdown(df):
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def _write_markdown(path, short_df, word_df):
    sections = ["# Table 4-style dataset metrics", ""]
    if not short_df.empty:
        sections.extend(["## Short texts", "", _to_markdown(short_df), ""])
    if not word_df.empty:
        sections.extend(["## Words", "", _to_markdown(word_df), ""])
    path.write_text("\n".join(sections), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create paper-style dataset-level metrics (similar to Table 4) "
            "from a Preds/<run> directory."
        )
    )
    parser.add_argument(
        "preds_dir",
        help="Directory containing predictions_fold1.csv and predictions_fold2.csv.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Recompute all_predictions.csv and dataset_metrics.csv before formatting.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    preds_dir = Path(args.preds_dir).expanduser().resolve()
    os.chdir(repo_root)

    dataset_metrics_path = preds_dir / "dataset_metrics.csv"
    if args.refresh or not dataset_metrics_path.exists():
        create_prediction_tables(str(preds_dir))

    df = pd.read_csv(dataset_metrics_path)
    df["DatasetType"] = df["dataset_of_origin"].apply(_dataset_type)
    df["Dataset"] = df["dataset_of_origin"].apply(_paper_name)
    df["Language"] = df["dataset_of_origin"].map(LANGUAGE_BY_DATASET).fillna("Unknown")
    df["sort_key"] = df["dataset_of_origin"].apply(_order_key)
    df = df.sort_values("sort_key").drop(columns=["sort_key"])

    output_cols = [
        "Dataset",
        "Language",
        "pearson_corr_valence",
        "pearson_corr_arousal",
        "rmse_valence",
        "rmse_arousal",
        "mae_valence",
        "mae_arousal",
    ]
    rename_cols = {
        "pearson_corr_valence": "rho_V",
        "pearson_corr_arousal": "rho_A",
        "rmse_valence": "RMSE_V",
        "rmse_arousal": "RMSE_A",
        "mae_valence": "MAE_V",
        "mae_arousal": "MAE_A",
    }

    formatted = df[["DatasetType", *output_cols]].rename(columns=rename_cols)
    metric_cols = ["rho_V", "rho_A", "RMSE_V", "RMSE_A", "MAE_V", "MAE_A"]
    formatted[metric_cols] = formatted[metric_cols].round(6)

    short_df = (
        formatted[formatted["DatasetType"] == "short_text"]
        .drop(columns=["DatasetType"])
        .reset_index(drop=True)
    )
    word_df = (
        formatted[formatted["DatasetType"] == "word"]
        .drop(columns=["DatasetType"])
        .reset_index(drop=True)
    )
    full_df = pd.concat([short_df, word_df], axis=0, ignore_index=True)

    full_path = preds_dir / "table4_metrics.csv"
    short_path = preds_dir / "table4_short_text.csv"
    word_path = preds_dir / "table4_words.csv"
    md_path = preds_dir / "table4_metrics.md"

    full_df.to_csv(full_path, index=False)
    short_df.to_csv(short_path, index=False)
    word_df.to_csv(word_path, index=False)
    _write_markdown(md_path, short_df, word_df)

    print("Table 4-style dataset metrics written.")
    print(f"Run dir: {preds_dir}")
    print(f"- {full_path.name}")
    print(f"- {short_path.name}")
    print(f"- {word_path.name}")
    print(f"- {md_path.name}")
    if not short_df.empty:
        print("\n[Short texts]")
        print(short_df.to_string(index=False))
    if not word_df.empty:
        print("\n[Words]")
        print(word_df.to_string(index=False))


if __name__ == "__main__":
    main()
