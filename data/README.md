# Data Folder README

This folder contains **raw and processed data** used for extracting and constructing the **Post-Traumatic Sepsis dataset**.

This repository **is initialized as empty** and will store **intermediate files** while the code runs.

🚨 **Note:** This repository **does not** include raw MIMIC-III data. Users must obtain access to MIMIC-III via **[PhysioNet](https://physionet.org/content/mimiciii/1.4/)** and comply with its **data use agreement**.

---

## Folder Structure

### `raw/`
- **Contains**: Raw extracted data from MIMIC-III.
- **Usage**: Typically includes **all patients** before any cohort filtering.

### `processed/`
- **Contains**: Processed data specifically for the **trauma cohort**.
- **Usage**: May include **cleaned, aggregated, or transformed** data relevant to the **post-traumatic sepsis** task.


Description: 
    raw/        Contains raw extracted data from MIMIC-III. Normally, files saved in this foder will contans information from all patients. 
     processed/ Stores processed versions to cresate the dataset. Normally, files saved in this foder only contans trauma cohort, and may apply cleaning and aggragete propecess that specified to our task(post-truma sepsis). 

## File Overview

| **Folder**    | **File Name**              | **Description** | **Source Code** |
|--------------|---------------------------|----------------|----------------|
| `raw/`      | `demographics.csv`         | Basic patient demographics | `demog_sql2df`¹ |
| `processed/` | `MVday.csv`               | Number of days the patient (HADM_ID) was on mechanical ventilation | `ventilation_day_processed`¹ |
| `processed/` | `trauma_cohort_info.csv`  | Trauma cohort and their corresponding hospital admission information | `extract_trauma_cohort_ids`² |
| `processed/` | `trauma_blood_cx_events.csv` | Blood culture events for trauma patients | `extract_blood_cx_events`³ |
| `processed/` | `trauma_abx_order.csv`    | Antibiotic prescription orders for trauma patients | `select_relevant_abx_data`³ |
| `processed/` | `trauma_abx_event.csv`    | Qualified antibiotic events used for sepsis assignment | `preprocess_abx_data`³ |
| `processed/` | `sofa_score.csv`          | Modified SOFA (Sequential Organ Failure Assessment) score | `SOFA_calculate`¹ |
| `processed/` | `trauma_sofa_score.csv`   | SOFA scores for the trauma cohort | `calculate_sofa_score`³ |
| `processed/` | `sepsis_label.csv`        | Sepsis onset labels for the trauma cohort | `assign_sepsis_labels`³ |
|-------------------------------------------------------------------------|
**Footnotes (Source Code Locations)**
¹ **sql2df** – SQL-to-DataFrame conversion scripts.  
² **cohort_extraction** – Scripts for extracting trauma cohort information.  
³ **sepsis_onset_label_assignment** – Scripts for assigning sepsis labels and related processing.  
----
