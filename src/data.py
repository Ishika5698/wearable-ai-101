"""Load the feature tables from the UCI HAR smartphone activity dataset."""

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "UCI HAR Dataset"


def load_split(data_dir: str | Path, split: str) -> dict[str, object]:
    """Load one official train or test split and check its rows are aligned."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    root = Path(data_dir).expanduser()
    folder = root / split
    paths = {
        "X": folder / f"X_{split}.txt",
        "y": folder / f"y_{split}.txt",
        "subject": folder / f"subject_{split}.txt",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "UCI HAR files are missing. Download the official archive and extract "
            f"'UCI HAR Dataset' to {root}. Missing: {', '.join(missing)}"
        )

    X = pd.read_csv(paths["X"], sep=r"\s+", header=None).to_numpy(dtype=np.float32)
    y = pd.read_csv(paths["y"], sep=r"\s+", header=None).iloc[:, 0].to_numpy(dtype=int)
    subjects = pd.read_csv(paths["subject"], sep=r"\s+", header=None).iloc[:, 0].to_numpy(dtype=int)
    if X.shape[0] != y.size or y.size != subjects.size:
        raise ValueError(
            f"{split} rows are misaligned: X={X.shape[0]}, y={y.size}, subjects={subjects.size}."
        )
    if not np.isfinite(X).all():
        raise ValueError(f"{split} feature table contains missing or non-finite values.")
    if not np.isin(y, np.arange(1, 7)).all():
        raise ValueError(f"{split} labels should be integers from 1 to 6.")
    return {"X": X, "y": y, "subjects": subjects}


def load_dataset(data_dir: str | Path = DEFAULT_DATA_DIR) -> dict[str, object]:
    """Load the original train/test split, feature names, and activity labels."""
    root = Path(data_dir).expanduser()
    features_file = root / "features.txt"
    activities_file = root / "activity_labels.txt"
    missing = [str(p) for p in (features_file, activities_file) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Dataset metadata is missing under {root}. Extract the UCI HAR Dataset "
            f"folder there. Missing: {', '.join(missing)}"
        )
    feature_names = pd.read_csv(features_file, sep=r"\s+", header=None, usecols=[1])[1].astype(str).to_list()
    activity_rows = pd.read_csv(activities_file, sep=r"\s+", header=None, names=["label", "activity"])
    activity_names = dict(zip(activity_rows.label.astype(int), activity_rows.activity.astype(str)))
    train = load_split(root, "train")
    test = load_split(root, "test")
    if train["X"].shape[1] != len(feature_names) or test["X"].shape[1] != len(feature_names):
        raise ValueError("Feature table width does not match features.txt.")
    return {"train": train, "test": test, "feature_names": feature_names, "activity_names": activity_names}
