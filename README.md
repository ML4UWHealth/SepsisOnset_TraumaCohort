# Rare Event Dataset: Early PostTraumatic Sepsis Detection

This repository provides a **standardized dataset for early sepsis onset detection in critically ill trauma patients**, extracted from **MIMIC-III v1.4**. The dataset supports research on **rare event detection** and **machine learning applications in clinical settings** by offering a **well-defined trauma cohort, structured post-trumma sepsis onset labels, and a deployable detection setup**.  
  
The implementation is **primarily in Python**, with some SQL queries, and follows the methodology described in the paper *"Rare Event Early Detection: Sepsis Onset for Critically Ill Trauma Patients."* This project leverages **Google BigQuery** for efficient access and management of the MIMIC-III database.  

**Note:** This repository provides **only the code** for extracting the dataset from MIMIC-III and does **not include raw MIMIC data**. To use this project, you must have **access to MIMIC-III v1.4 in BigQuery**. (For details on obtaining access, refer to [`notebooks/S0_MIMIC-III_Data_Access_Instructions.ipynb`](https://github.com/ML4UWHealth/SepsisOnset_TraumaCohort/blob/main/notebooks/S0_MIMIC-III%20Data%20Access%20Instructions.ipynb)  

## Usage  

To support future studies using this dataset, the project is structured into **three main modules**: **cohort extraction, label assignment, and dataset construction**. Each module includes dedicated scripts for seamless execution, along with notebooks that provide detailed explanations of the implementation, methodology, and relevant references.  



### **Section 0: Verify MIMIC-III Dataset Access**  

If you **do not have access**, please refer to [`notebooks/S0_MIMIC-III_Data_Access_Instructions.ipynb`](https://github.com/ML4UWHealth/SepsisOnset_TraumaCohort/blob/main/notebooks/S0_MIMIC-III%20Data%20Access%20Instructions.ipynb) for detailed setup instructions.  

If you **believe you have access** to **MIMIC-III v1.4** via **Google BigQuery**, you can quickly verify it using the following code:  

```python
from src.data import data_utils

print("This is your PROJECT_ID:", PROJECT_ID)
data_utils.test_mimiciii_bigquery_access(PROJECT_ID)
```  
**Expected Output:**  
If access is successful, you should see output similar to:  

```markdown
This is your PROJECT_ID: sepsis-mimic3
Successfully accessed MIMIC-III via BigQuery
True
```
**Troubleshooting Access Issues**  

If the above code encounters any errors, check the following:  

1. **Verify Access on BigQuery Console:**  
   - Test your access using the **Google BigQuery console** (see Section 1 in `notebooks/S0_MIMIC-III_Data_Access_Instructions.ipynb`).  

2. **Ensure You Are Logged in with the Correct Google Account:**  
   - Use the **Google account** that has been granted access to MIMIC-III via PhysioNet (the same account used for the BigQuery console).  

3. **Check Your Project ID:**  
   - Ensure you have set the correct **Project ID**.  
   - If you are unsure of your Project ID, refer to Section 1 in `notebooks/S0_MIMIC-III_Data_Access_Instructions.ipynb`.  


### **Section 1: Cohort Extraction – Critically Ill Trauma Patients**  

This step extracts a well-defined **trauma cohort** from **MIMIC-III v1.4**, following the inclusion criteria outlined in **Section 3.1, "Cohort Extraction: Critically Ill Trauma Patients,"** of our paper. The final cohort consists of **1,570 admissions**, optimized for **early sepsis onset detection**.  

**Cohort Criteria:**  
- **Instance ID:** Use qualified `HADM_ID` as the instance identifier.  
- **ICD-9 E-codes Selection:** Identify trauma patients based on ICD-9 E-codes.  
- **Age Criteria:** Include patients aged **18–89 years**.  
- **Hospital Stay Duration:** Include admissions with a hospital stay of **at least 48 hours**.  
- **Ventilation Days:** Include patients with **at least 3 days of mechanical ventilation**.  

For a detailed explanation of the cohort extraction process, refer to [`S1_notebooks/Cohort_Extraction.ipynb`](https://github.com/ML4UWHealth/SepsisOnset_TraumaCohort/blob/main/notebooks/S1_Cohort_Extraction.ipynb).  

**Running the Extraction Script**  
To execute the extraction, use the following code:  
> **Note:** The first run of the following block may take approximately **4 minutes**. 

```python
from scripts.cohort_extraction import extract_trauma_cohort_ids

# Extract cohort IDs and generate a statistics report
trum_ids = extract_trauma_cohort_ids(project_path_obj, PROJECT_ID, is_report=True, is_saved=True)
```
**Expected Output:**
```markdown
             TOTAL    TRUM Basic Filter    TRUM Vent Filter  
subject_id   46428    4826                 1551  
hadm_id      57328    4977                 1570  
icustay_id   61051    5410                 1828  

MIMIC III includes: 57328 (qualified hospital admissions)
After Trauma Selection (ICD-9): 6148
After Age Filter: 5651
After Hospital Length of Stay >= 48h Filter: 4977
    Hospital Length of Stay < 48h: 674 = 200 (Died) + 474 (Discharged Alive)
Mechanical Ventilation Day Filter: 
    2271 (Not Intubated) + 1136 (Intubated < 3 days)
Final Cohort Size: 1570
Save to:  /content/drive/MyDrive/REED/SepsisOnset_TraumaCohort/data/processed/trauma_cohort_info.csv
```


### Section2: Assign Labels: Post-Trauma Sepsis Labels  
This section calls the **core function** for **sepsis label assignment**, integrating essential criteria to accurately assign sepsis labels and timestamp **sepsis onset** in the critically ill trauma cohort from **MIMIC-III v1.4**. The methodology is detailed in **Section 3.2, "Post-Trauma Sepsis Definition,"** of our paper. For implementation details, refer to [`notebooks/S2_Sepsis_Onset_Label_Assignment.ipynb`](https://github.com/ML4UWHealth/SepsisOnset_TraumaCohort/blob/main/notebooks/S2_Sepsis_Onset_Label_Assignment.ipynb).  

### Step 1: Preprocess Pertinent Features
We preprocess data from four primary tables to establish the sepsis definition for the trauma cohort within the MIMIC-III dataset:
- **Cohort Admit Info**: Contains trauma cohort patient IDs and corresponding admission information.
- **Blood Culture**: Includes blood culture events taken at or after 72 hospital hours.
- **Antibiotics:** Includes qualifying IV or specific oral antibiotics (e.g., vancomycin, linezolid) that are newly administered (not given the previous day), exclude first-day and prophylactic use, and must be given for at least four consecutive days or until death/discharge.
- **Modified SOFA Score**: Includes the Sequential Organ Failure Assessment (SOFA) score for every hour of the trauma cohort's ICU stay, with the Glasgow Coma Scale (GCS) and Urine Output (UO) components excluded from the standard SOFA score.

### Step 2: Define Post-Trauma Sepsis
Post-Trauma Sepsis is defined based on preprocessed data following Sepsis-3 consensus guidelines. It is identified as a clinically suspected infection associated with acute worsening of organ dysfunction. The sepsis onset time is defined as the chart time of the first sepsis onset candidate.

**Criteria for Sepsis Onset Candidate Timestamp**:
- **Qualifying Culture Timestamp**: Must be for a qualifying blood culture's chart time.
- **Suspected Infection**: Identified by blood culture records (specifically chart time) within a 5-day window of qualifying antibiotic initiation.
- **Organ Dysfunction**: Identified by a **2-point increase** in the SOFA score within a 7-day window (-3 days, day of, +3 days) relative to the culture chart time.

### Step 3: Post-Process Sepsis Candidates
Finally, we assign sepsis labels and onset timestamps as follows:
- **0 as Non-Sepsis**: If a patient (HADM_ID) has no qualified sepsis candidates, they are labeled as non-sepsis, with the onset time set to NaN.
- **1 as Sepsis**: If a patient (HADM_ID) has more than one sepsis candidate, we retain only the earliest culture timestamp as the onset time.

### Summary
Among the 1,570 trauma admissions analyzed, 729 admissions had suspected infections, and 535 had confirmed sepsis. As shown in the following section's graph, the peak of sepsis onset occurs on the 5th day after hospital admission.

**Running the Extraction Script**  
To execute the extraction, use the following code:  
> **Note**: The first time running the following block may take about **31 minutes**.

```python
from scripts.sepsis_onset_label_assignment import assign_sepsis_labels

# Assign sepsis labels and onset times for each patient in the cohort
sepsis_label_df = assign_sepsis_labels(project_path_obj, PROJECT_ID)
```
**Expected Output:**
```markdown
╔══════════════════════════════════╗
║        Preprocessing Data        ║
╚══════════════════════════════════╝
--------------Trauma Cohort Information--------------
Loading trauma cohort information...
Loaded 1570 trauma patients.

--------------Blood Culture Events-------------------
Extracting blood culture events...
Saved trauma blood culture events to /content/drive/MyDrive/REED/SepsisOnset_TraumaCohort/data/processed/trauma_blood_cx_events.csv
TOTAL 8821 trauma blood culture events for 1037 trauma patients
Extraction completed in 14.61 seconds.
After processing (drop duplicates), 3826 unique records remain.

--------------Antibiotic Events----------------------
Extracting antibiotic events...
Included 154834 qualified IV antibiotic samples
and
Included 3380 qualified oral antibiotic samples
TOTAL 9999 antibiotic samples for 1239 trauma patients
#of qualifying antibiotic order entries:  9859
Drop 63 noise abx records s.t. startdate>enddate
#of qualifying antibiotic event: (4886, 7)
After dropped 1st day antibiotic events: (4206, 8)
After filtering the duration criteria: (2780, 9)
Saved clean, well-organized, and qualified antibiotic events to /content/drive/MyDrive/REED/SepsisOnset_TraumaCohort/data/processed/trauma_abx_event.csv
Extraction completed in 97.80 seconds.
After processing (drop duplicates), 2039 unique records remain.

--------------SOFA Scores----------------------------
Calculating SOFA scores...
Total 433825 SOFA samples for 1570 trauma patients.
Saved SOFA score for trauma patients to /content/drive/MyDrive/REED/SepsisOnset_TraumaCohort/data/processed/trauma_sofa_score.csv.
Calculation completed in 1730.32 seconds.
After processing, 433825 unique records remain.


╔══════════════════════════════════╗
║      Assigning Sepsis Labels     ║
╚══════════════════════════════════╝
Number of trauma patients: 1570
Number of infections: 729.0
Number of sepsis cases: 535.0

Saving sepsis label information at /content/drive/MyDrive/REED/SepsisOnset_TraumaCohort/data/processed/sepsis_label.csv
```

```
Display descriptive statistics for the onset day
```
![Sepsis Onset Day Distribution](https://github.com/ML4UWHealth/SepsisOnset_TraumaCohort/blob/main/supplementary/TimingofSepsis_cx.png)


### **Section 3: Generate Dataset**  

This section loads the **Post-Traumatic Sepsis dataset** (without missing values), derived from **MIMIC-III v1.4**. For a detailed explanation of dataset construction, refer to [`notebooks/S3_Early_Sepsis_Onset_Detection_Setup.ipynb`](https://github.com/ML4UWHealth/SepsisOnset_TraumaCohort/blob/main/notebooks/S3_Early_Sepsis_Onset_Detection_Setup.ipynb).  

The dataset consists of the following columns:  

- **Temporal Features**: Multivariate time-series input data with dimensions **(# of timestamps, # of features)**.  
- **Label**: Binary value (**0 or 1**) indicating sepsis onset.  
- **Fold**: Specifies the fold assignment for cross-validation.  

Each row represents a **nighttime instance** and includes patient identifiers (`subject_id`, `hadm_id`) along with a timestamp (`Date`, `Night`).  

**Running the Extraction Script**  
To execute the extraction, use the following code:  
> **Note**: The first time running the following block may take about ** 23 minutes**.
```python
from scripts.early_sepsis_onset_detection_setup import dataset_construction

# Generate dataset versions with and without missing values
data_with_nan_df, data_wo_nan_df = dataset_construction(project_path_obj, PROJECT_ID, is_report=True)

```
**Expected Output:**
```markdown
Dataset: N dataset | Shape: (10565, 7) | Unique Patients (hadm_id): 1536  

Fold    Total Instances    Positive Instances    Negative Instances    Imbalance Ratio  
0       2183              90                    2093                  0.041228  
1       2053              89                    1964                  0.043351  
2       2025              90                    1935                  0.044444  
3       1990              91                    1899                  0.045729  
4       2129              90                    2039                  0.042273  
Total   10380             450                   9930                  0.043353  

Dataset: S dataset | Shape: (6340, 7) | Unique Patients (hadm_id): 1165  

Fold    Total Instances    Positive Instances    Negative Instances    Imbalance Ratio  
0       1252              69                    1183                  0.055112  
1       1262              71                    1191                  0.056260  
2       1313              76                    1237                  0.057883  
3       1132              66                    1066                  0.058304  
4       1272              72                    1200                  0.056604  
Total   6231              354                   5877                  0.056813  
```

# Project Organization

    ├── data/              <- Data saved in this directory.
    │   ├── raw/           <- Contains raw data extracted from the MIMIC dataset.
    │   ├── processed/     <- Contains processed data organized as reusable modules for final dataset generation and other future tasks.
    │
    ├── dataset/           <- Contains the final dataset ready for model training.
    │   ├── Fold_IDs.csv   <- Patient IDs and their assigned 5-fold cross-validation folds.
    │   ├── PostTraumaticSepsis_dataset_w_nan.pkl  <- N dataset: Dataset including missing values (Not included in GitHub; this is the location where the file will be saved).
    │   ├── PostTraumaticSepsis_dataset_wo_nan.pkl <- S dataset: Dataset with missing values handled (Not included in GitHub; this is the location where the file will be saved).
    │
    ├── LICENSE   
    │
    ├── notebooks/         <- Jupyter notebooks matching with scripts
    │   ├── S0_MIMIC-III Data Access Instructions.ipynb      <- Instructions on how to access the MIMIC-III dataset.
    │   ├── S1_cohort_extraction.ipynb                       <- Cohort extraction for critically ill trauma patients.
    │   ├── S2_Sepsis_Onset_Label_Assignment.ipynb           <- Assign Post-trauma Sepsis Onset Label according to the definition.
    │   ├── S3_Early_Sepsis_Onset_Detection_Setup.ipynb      <- Generate dataset according to the Early Sepsis Onset Detection Setup.
    │
    ├── README.md  
    │
    ├── scripts/           <- Task-oriented scripts utilizing functions and classes defined in the src directory.
    │   ├── cohort_extraction.py                       <- Cohort extraction for critically ill trauma patients.
    │   ├── Sepsis_Onset_Label_Assignment.py           <- Assign Post-trauma Sepsis Onset Label according to the definition.
    │   ├── Early_Sepsis_Onset_Detection_Setup.py      <- Generate dataset according to the Early Sepsis Onset Prediction Setup.
    │
    ├── src/               <- Source code for use in this project
    │   ├── path_manager.py      <- Manages file paths and directory structures for the project.
    │   │
    │   ├── data/
    │       ├── data_fetcher.py  <- Functions for querying and retrieving MIMIC-III data.
    │       ├── data_utils.py    <- Utility functions for preprocessing and dataset handling.
    │       ├── sql2df.py        <- Functions to convert SQL query results into pandas DataFrames.
    │
    └── supplementary/
        ├── qualified_traumatic_ICD9_Ecodes.xlsx <- Qualifying ICD-9 E codes for the trauma cohort.
        ├── TimingofSepsis_cx.png                <- Graph illustrating the timing of sepsis onset in the trauma cohort.

