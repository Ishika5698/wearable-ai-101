# UCI HAR files

The notebook expects the extracted UCI directory at `data/raw/UCI HAR Dataset/`. It reads these files:

```text
train/X_train.txt
train/y_train.txt
train/subject_train.txt
test/X_test.txt
test/y_test.txt
test/subject_test.txt
features.txt
activity_labels.txt
```

Download the archive from the [UCI dataset page](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) and extract it under `data/raw/`. The data is not part of the GitHub repository. The local ZIP archive and files under `data/raw/` and `data/processed/` are excluded by `.gitignore`.

This notebook uses UCI's engineered feature matrices. For the accelerometer and gyroscope signal-window work, see `wearable-signal-lab`.
