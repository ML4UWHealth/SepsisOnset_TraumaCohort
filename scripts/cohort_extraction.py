"""
This script extracts a cohort of critically ill trauma patients from the MIMIC-III v1.4 dataset, 
  applying stringent criteria including age range, admission duration, and mechanical ventilation days, 
  as detailed in Section 3.1 "Cohort Extraction: Critically Ill Trauma Patients" of our paper. 
  The refined cohort, tailored for studying early sepsis onset prediction, comprises 1,570 admissions.

This script is a cleaned-up version of the original notebook located at
    `notebooks/cohort_extraction.ipynb`
While this file focuses on the implementation of functions, the notebook contains more detailed information, including how the functions return and display their results. 
For a deeper understanding and additional context, please refer to the original notebook.

**Notes**:
This script assumes that you have access to MIMIC-III on Google BigQuery. 
If you are unsure about your access or need to apply for it, please visit the `notebooks/MIMIC-III access page` for instructions.
"""

# Set Up
# Import libraries
import os
import re   # the regular expressions module

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Commented out IPython magic to ensure Python compatibility.
from datetime import date
from src.data.data_fetcher import get_demographics_data, get_ventilation_data
from src.data import data_utils#, sql2df, data_fetcher

"""
# Qulified ICD9 E-code
"""
def select_ICDcode_df(project_path_obj, #Saved File Paths
                      project_id,       #Source File
                      ):
	'''
    Select trauma patients according to E-codes of ICD9_CODE.

    Parameters:
    - project_path_obj: An object containing paths to project-related files.
    - project_id: The Google BigQuery project ID for querying the MIMIC-III dataset.

    Returns:
    - DataFrame: A DataFrame containing patients' ICD9 diagnoses filtered by trauma E-codes.

    Notes:
    - This function reads a list of qualified traumatic injury E-codes from an Excel file located
      at 'supplementary/qualified_traumatic_ICD9_Ecodes.xlsx'.
    - The E-codes are reformatted to match the format used in the MIMIC-III dataset.
    - Queries the MIMIC-III clinical database to select all ICD diagnoses where ICD9_CODE is not NULL.
    - Filters the diagnoses to include only those starting with 'E' and matching the qualified trauma E-codes.
	'''
	# HMC's list of traumatic injury E-codes.
  # Detail saved at "supplementary/qualified_traumatic_ICD9_Ecodes.xlsx"
	ICD9Code_file_path = os.path.join(project_path_obj.get_supplementary_file("qualified_traumatic_ICD9_Ecodes.xlsx"))
  # print(ICD9Code_file_path)
	df_hmc_e = pd.read_excel(ICD9Code_file_path, sheet_name="Ecodes ICD 9") 
  # Reformat the codes to be consistent with MIMIC's format.
	df_hmc_e["Ecode"] = df_hmc_e["Ecode"].apply(lambda x: "E" + re.sub(r'\W+', '', str(x)))
	Ecodes = df_hmc_e["Ecode"].unique()

	# select all ICD diagnoses where ICD9_CODE is not NULL
	query =  """
	SELECT *
	FROM `physionet-data.mimiciii_clinical.diagnoses_icd`
	WHERE ICD9_CODE IS NOT NULL;
	"""
	ICDDIAGNOSES_df = data_utils.run_query(query, project_id)

	# select "ICD9_CODE" starting with E
	trum_df = ICDDIAGNOSES_df[ICDDIAGNOSES_df['ICD9_CODE'].str.startswith('E')]
	trum_df.loc[:, "ICD9_CODE"] = trum_df.loc[:, "ICD9_CODE"].apply(lambda e: str(e)+"0" if len(str(e)) < 5 else e)

	# Select all E-code diagnoses.
	trum_df = trum_df[trum_df['ICD9_CODE'].isin(Ecodes)]

	return trum_df
# usage
# df = df = select_ICDcode_df(project_path_obj, PROJECT_ID)


"""
Cohort Extraction: Critically Ill Trauma Patients
"""
def extract_trauma_cohort_ids(project_path_obj, #Saved File Paths
               project_id,       #Source File
               vent_threshold=3, is_report=False, is_saved=False):
  """
  Cohort Selection Criteria:
    1. Get qualified HADM_ID (with corresponding CHARTEVENTS data and at least 1 ICUStay_ID.)    
    2. Select trauma patients according to E-codes of ICD9_CODE
    3. Age between [18, 89]
    4. Hospital stay duration between [48 hours, None)
    5. Ventilation days >= 3
       Ventilation days
       vent_threshold:
          -> if None, then MV filter is not included
          -> or an int, the lower bound for MV filter.
          -> this boundary is experimental
          -> in Stern's paper, they use 3 as the threshold
             (Stern K, Qiu Q, Weykamp M, O’Keefe G, Brakenridge SC.
              Defining Posttraumatic Sepsis for Population-Level Research.
              JAMA Netw Open. 2023;6(1):e2251445. doi:10.1001/jamanetworkopen.2022.51445 )
  """
  # Get qualified patients' demographics: (with corresponding CHARTEVENTS data and at least 1 ICUStay_ID) 
  demog_df = get_demographics_data(project_path_obj, project_id)
  demog_df = demog_df[['subject_id', 'hadm_id', 'icustay_id',
                       'admission_age', 'admittime', 'dischtime',
                       'los_hospital_hours', 'los_hospital_days', 'hospital_expire_flag']]
  count_df = demog_df[['subject_id', 'hadm_id', 'icustay_id']].nunique().to_frame(name='TOTAL')

  # Selected according to E-codes
  # group by IDs and aggregate ICD9_CODE info because we want unique IDs
  TRUM_df = select_ICDcode_df(project_path_obj, project_id).groupby(['SUBJECT_ID', 'HADM_ID'])['ICD9_CODE'].agg(set).reset_index()
  TRUM_df = demog_df[demog_df['hadm_id'].isin(TRUM_df.HADM_ID)]

  # filter according to the age in range [18, 89]
  TRUM_df_age = TRUM_df[(TRUM_df.admission_age >= 18) & (TRUM_df.admission_age <= 89)]

  # filter according to the hospital stay duration in the range [48 hours, no upper bound)
  TRUM_df_los = TRUM_df_age[TRUM_df_age.los_hospital_hours >= 48]
  count_df['TRUM basic filter'] = TRUM_df_los[['subject_id', 'hadm_id', 'icustay_id']].nunique()

  # Select according to Ventilation Days
  if vent_threshold == None:
    trum_df = TRUM_df_los
  else:
    # get ventilation day table
    vent_day_count = get_ventilation_data(project_path_obj, project_id)
    # select according to Ventilation days >= vent_threshold (default is 3) for each patient
    TRUM_df_vent = TRUM_df_los.merge(vent_day_count, on='hadm_id', how='left')
    trum_vent_day_count = TRUM_df_vent[['hadm_id','date_count']].drop_duplicates()
    TRUM_df_vent = TRUM_df_vent[TRUM_df_vent['date_count'] >= vent_threshold]
    count_df['TRUM Vent filter'] = TRUM_df_vent[['subject_id', 'hadm_id', 'icustay_id']].nunique()
    trum_df = TRUM_df_vent

  # Statistics ###
  if is_report:
    display(count_df)
    print("MIMIC III includes: %d (qualified hospital admissions)" % demog_df.hadm_id.nunique())
    print("After Trauma Selection (ICD-9): %d" % TRUM_df.hadm_id.nunique())
    print("After Age Filter: %d" % TRUM_df_age.hadm_id.nunique())

    TRUM_df_firstfewday = TRUM_df_age.loc[TRUM_df_age.los_hospital_hours < 48, ['hadm_id', 'hospital_expire_flag']].drop_duplicates()
    TRUM_df_firstfewday_mortalitycount = TRUM_df_firstfewday.hospital_expire_flag.value_counts()

    print("After Hospital Length of Stay >= 48h Filter: %d" % TRUM_df_los.hadm_id.nunique())
    print("\tHospital Length of Stay < 48h: %d = %d (Died) + %d (Discharged Alive)" % 
          (TRUM_df_firstfewday.shape[0], TRUM_df_firstfewday_mortalitycount[1], TRUM_df_firstfewday_mortalitycount[0]))

    if vent_threshold is not None:
        num_not_intubated = trum_vent_day_count.date_count.isna().sum()
        intubated_less_thr = trum_vent_day_count[trum_vent_day_count.date_count < vent_threshold].shape[0]
        print('Mechanical Ventilation Day Filter: \n\t%d (Not Intubated) + %d (Intubated < %d days)' % 
              (num_not_intubated, intubated_less_thr, vent_threshold))
    print("Final Cohort Size: %d" % trum_df.hadm_id.nunique())
  if is_saved:
    print("Save to: ", project_path_obj.trauma_cohort_info_path)
    trum_df.to_csv(project_path_obj.trauma_cohort_info_path)

  return trum_df
