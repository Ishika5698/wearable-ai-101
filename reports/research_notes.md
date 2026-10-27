# Activity-classification notes

## Question

How well does a random forest classify the six UCI HAR activities from the dataset's 561 supplied features, when evaluated on UCI's held-out participants?

## Run recorded here

The notebook used the extracted UCI HAR files in `data/raw/UCI HAR Dataset/`. The training split has 7,352 windows from 21 participants; the test split has 2,947 windows from nine different participants. Each example has 561 features. The model was a random forest with 300 trees, `min_samples_leaf=2`, balanced subsampling of classes, and random seed 42. The train/test split was kept as supplied.

| Model | Test accuracy | Test macro F1 |
|---|---:|---:|
| Most-frequent-class baseline | 0.1822 | 0.0514 |
| Random forest | 0.9294 | 0.9274 |

Random-forest test results by activity:

| Activity | Precision | Recall | F1 | Test windows |
|---|---:|---:|---:|---:|
| Walking | 0.8976 | 0.9718 | 0.9332 | 496 |
| Walking upstairs | 0.8942 | 0.9151 | 0.9045 | 471 |
| Walking downstairs | 0.9647 | 0.8452 | 0.9010 | 420 |
| Sitting | 0.9205 | 0.8961 | 0.9082 | 491 |
| Standing | 0.9064 | 0.9286 | 0.9174 | 532 |
| Laying | 1.0000 | 1.0000 | 1.0000 | 537 |

## Reading the result

The model did well on this test split, though walking downstairs was its weakest recall. Most of the errors among the walking classes were between upstairs and downstairs. Sitting and standing were also occasionally confused. These scores describe a single benchmark split and this particular model setup; they are not an estimate of performance on every phone, placement, or person.

## Dataset and limits

The UCI study recorded 30 volunteers wearing a waist-mounted Samsung Galaxy S II while performing six scripted activities. The feature matrix was calculated by the dataset authors from sensor windows, so this project is not training directly on raw sensor samples. The original train/test split keeps participants separate, which is helpful when evaluating new people in the same study. The experiment remains small and controlled, and it does not cover everyday activity, fatigue, stress, or medical use.

## Source

Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., and Parra, X. (2013). [Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones). UCI Machine Learning Repository. DOI: 10.24432/C54S4K. Dataset license: CC BY 4.0; cite the source when reusing it.
