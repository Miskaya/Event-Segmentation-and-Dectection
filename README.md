# Event-Segmentation-and-Detection
SCAI Master Thesis - Event Segmentation and Detection in Time-Series for Monitoring Activities of Daily Living in SCI Individuals
==========

Overview
==========
Spinal cord injuries (SCI) pose significant challenges, impacting millions globally and often leading to reduced mobility and difficulties in daily activities. Monitoring activities of daily living (ADLs) is crucial for assessing overall health. Wearable sensors like IMU sensors, pressure mats, and cameras show promise, but intelligently segmenting daily activities remains a hurdle. This thesis focuses on researching event segmentation and detection techniques on time-series data from diverse signal modalities. Precise segmentation aims to reveal patterns, anomalies, and critical events, serving as key indicators for individuals' well-being. Accurate event detection holds potential for personalized healthcare, enabling timely interventions and tailored care plans.

Run Program
========

### Initialize the Working Environment
To be able to run this program, the user must first set up a `conda` environment. 
The user must have [Anaconda](https://www.anaconda.com/) installed on their device. 
To create a new environment, run the following command on a terminal: 
```
conda env create --name bp_estimation
```

Activate the created environment by running the following command: 

```
conda activate ADL_FEN_FLN
```

The user must install the correct packages to run the Jupyter notebooks with the command: 
```
conda install <PACKAGE>
```

The user can also directly create an environment by running the command:
```
conda env create -f requirements.yml
```

The user will have to start a Jupyter notebook server to be able to run the `.ipynb` files. 
This is done either by openning the application of choice (the application used for coding was [Visual Studio Code](https://code.visualstudio.com)) or the Jupyter Notebook browser of Anaconda with: 

```
jupyter notebook
```
### Data Preprocessing
Data_Generated.ipynb: Performs data preprocessing for all 11 public HAR datasets. Each signal is represented as one column with the last two columns being subjects and labels. All datasets are resampled to 20Hz and standardized. Each dataset is saved as a pickle file.

Data_preprocessing_OutSense.py: Handles the preprocessing of the OutSense dataset, also resampled to 20Hz.

### Model Training
GeAR-FEN_training.py: Conducts the training process for all 11 public datasets to obtain a pretrained Generalized Activity Recognition Feature Extraction Network (GeAR-FEN).

### Metadata Handling
Metadata_of_videos.py: Used for data labeling of the OutSense dataset, obtaining metadata for each video.

### Transfer Learning
TL_to_SCAI_HAR.py: Applies the pretrained GeAR-FEN model to the SCAI-HAR dataset.

TL_to_OutSense.py: Applies the pretrained GeAR-FEN model to the OutSense dataset.

### Data Splits
data_splits.pkl and data_splits_independent_dataset.pkl: Contain detailed information on the subjects included in each of the 5 folds for test subjects. "independent dataset" refers to the SCAI-HAR dataset.

### Model
model.py: Defines the Generalized Activity Recognition Feature Extraction Network (GeAR-FEN) and Feature Learning Network (FLN).

### Subject Analysis
subject_wise_analysis.py: Uses Clasp for unsupervised segmentation of each subject.

Directory Layout 
=========

    Experiment
    │
    ├───Codes
    │   ├───Data_Generated.ipynb
    │   ├───Data_preprocessing_OutSense.py
    │   ├───GeAR-FEN_training.py
    │   ├───Metadata_of_videos.py
    │   ├───TL_to_SCAI_HAR.py
    │   ├───TL_to_OutSense.py
    │   ├───model.py
    │   ├───subject_wise_analysis.py
    │   ├───cp_detector.py
    │   ├───dataset.py
    │   └───multivariate_claspy.py
    │
    ├───Data
    │   ├───data_splits.pkl
    │   └───data_splits_independent_dataset.pkl
    │
    ├───requirements.yml
    │
    └───ReadMe.md

Description of files 
============

Non Python files: 
---------

Filename        | Description 
----------------|--------------------------
ReadMe.md       | Markdown text file, describes the project and its components.
requirements.yml       | Lists all the dependencies required to run the Python scripts in the project.
data_splits.pkl       | Contains subject details for 5-fold cross-validation on public 11 datasets.
data_splits_independent_dataset.pkl       | Contains subject details for 5-fold cross-validation on the SCAI-HAR dataset.

Python files: 
---------

Filename        | Description 
----------------|--------------------------
Data_Generated.ipynb         | For data pre-processing the public datasets and SCAI-HAR dataset
Data_preprocessing_OutSense.py        | For preprocessing the OutSense dataset
GeAR-FEN_training.py      | Training process for the GeAR-FEN model on 11 public datasets
Metadata_of_videos.py      | Generates metadata for labeling the OutSense dataset videos
TL_to_SCAI_HAR.py   | Applies pretrained GeAR-FEN to the SCAI-HAR dataset
TL_to_OutSense.py         | Applies pretrained GeAR-FEN to the OutSense dataset
model.py        | Defines the Generalized Activity Recognition Feature Extraction Network (GeAR-FEN) and Feature Learning Network (FLN)
subject_wise_analysis.py      | Performs unsupervised segmentation analysis on each subject using Clasp
cp_detector.py      | 
dataset.py   |
multivariate_claspy.py   |
