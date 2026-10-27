# Wearable AI 101

This is the first project in a small wearable-sensing portfolio. It uses the UCI Human Activity Recognition Using Smartphones data to classify six everyday activities from the researchers' precomputed sensor features. The later projects build on the same dataset: one explores signal processing, and one introduces a different physiological dataset, WESAD.

## What this project does

The notebook reads the official `train` and `test` feature tables, checks their dimensions and labels, and compares a majority-class baseline with a random forest. It keeps UCI's participant-separated train/test split. The model uses the provided 561 engineered features; it does not start from raw, continuous sensor recordings.

The six target activities are walking, walking upstairs, walking downstairs, sitting, standing, and laying. This is a learning exercise on a controlled smartphone study, not a general activity or health monitor.

## Set up

Use Python 3.10 or newer. From this folder in Terminal:

```zsh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m jupyter notebook
```

Open `notebooks/01_data_exploration.ipynb` and run it from top to bottom. The notebook uses a path relative to this project. If you run it in VS Code, select `.venv/bin/python` as the kernel.

## Add the dataset

Get the archive from the [UCI dataset page](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones), then extract the `UCI HAR Dataset` folder to:

```text
data/raw/UCI HAR Dataset/
```

The local ZIP currently in `human+activity+recognition+using+smartphones/` is ignored by Git along with extracted and derived data. Only the code and documentation belong in the public repository. UCI lists the dataset under CC BY 4.0; retain its citation when sharing work based on it.

## Project files

- `notebooks/01_data_exploration.ipynb` — checks the feature tables and participant split, explores activity labels, fits models, and reports test-set metrics.
- `src/data.py` — loads and validates the supplied train/test tables.
- `src/models.py` — evaluates a majority baseline and a random forest.
- `reports/research_notes.md` — records the analysis status and dataset limitations.

The notebook's score is specific to UCI's provided split and feature set. It should not be compared directly with scores from a different split, preprocessing pipeline, or sensor setup.
