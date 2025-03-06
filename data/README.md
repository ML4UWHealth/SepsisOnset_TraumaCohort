# Data Folder README

This folder contains **raw and processed data** used for extracting and constructing the **Post-Traumatic Sepsis dataset**.

This repository **is initialized as empty** and will store **intermediate files** while the code runs.

**Note:** This repository **does not** include raw MIMIC-III data. Users must obtain access to MIMIC-III via **[PhysioNet](https://physionet.org/content/mimiciii/1.4/)** and comply with its **data use agreement**.



## Folder Structure

### `raw/`
- **Contains**: Raw extracted data from MIMIC-III.
- **Usage**: Typically includes **all patients** before any cohort filtering.

### `processed/`
- **Contains**: Processed data specifically for the **trauma cohort**.
- **Usage**: May include **cleaned, aggregated, or transformed** data relevant to the **post-traumatic sepsis** task.


## File Overview


### **File Overview**
| **Folder**    | **File Name**              | **Description** | **Source Code** |
|--------------|---------------------------|----------------|----------------|
| `raw/`      | `demographics.csv`         | Basic patient demographics | `demog_sql2df`[^1] |
| `processed/` | `MVday.csv`               | Number of days the patient (HADM_ID) was on mechanical ventilation | `ventilation_day_processed`[^1] |
| `processed/` | `trauma_cohort_info.csv`  | Trauma cohort and their corresponding hospital admission information | `extract_trauma_cohort_ids`[^2] |
| `processed/` | `trauma_blood_cx_events.csv` | Blood culture events for trauma patients | `extract_blood_cx_events`[^3] |
| `processed/` | `trauma_abx_order.csv`    | Antibiotic prescription orders for trauma patients | `select_relevant_abx_data`[^3] |
| `processed/` | `trauma_abx_event.csv`    | Qualified antibiotic events used for sepsis assignment | `preprocess_abx_data`[^3] |
| `processed/` | `sofa_score.csv`         | Modified SOFA (Sequential Organ Failure Assessment) score | `SOFA_calculate`[^1] |
| `processed/` | `trauma_sofa_score.csv`   | SOFA scores for the trauma cohort | `calculate_sofa_score`[^3] |
| `processed/` | `sepsis_label.csv`        | Sepsis onset labels for the trauma cohort | `assign_sepsis_labels`[^3] |

File Location: 
[^1]: **../src/data/sql2df.py** – SQL-to-DataFrame conversion scripts.  
[^2]: **../scripts/cohort_extraction** – Scripts for extracting trauma cohort information.  
[^3]: **../scripts/epsis_onset_label_assignment** – Scripts for assigning sepsis labels and related processing.  
