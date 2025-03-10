"""
### Early Sepsis Onset Detection Setup

This script details the methodology for establishing a dataset aimed at early sepsis onset detection, as described in Section 3.3, "Early Sepsis Onset Prediction Setup," of our paper. The setup process is organized into three primary stages: feature extraction, instance construction, and data splitting to mitigate data leakage.

The script contains three main processing steps:
1. **Feature Extraction:** Extraction and preprocessing of input feature data.
2. **Instance Construction:** Assignment of sepsis labels for each instance.
3. **Data Split:** Division of data at the patient level to prevent data leakage.

In alignment with the approach proposed by [Stewart et al. 2023]((https://www.computer.org/csdl/proceedings-article/bigdata/2023/10386180/1TUPtOpspXy)), we implement a nightly detection setup tailored to the specific needs of Intensive Care Unit (ICU) environments. This setup utilizes data recorded during nighttime hours, from 10 p.m. to 6 a.m. the following day. Positive labels are assigned exclusively to instances where sepsis is predicted to develop within 24 hours following the night in question.

This script is a cleaned-up version of the original notebook located at
    Notebooks/Early_Sepsis_Onset_Detection_Setup.ipynb`
"""

## Importing libraries.
import os
import numpy as np
import pandas as pd
import time
from datetime import datetime, time, date, timedelta
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


from src.data import data_utils, sql2df
from scripts.cohort_extraction import extract_trauma_cohort_ids
from scripts.sepsis_onset_label_assignment import assign_sepsis_labels

"""
# 1. Feature Extraction

The model leverages nighttime vital signs data to detect early sepsis onset within the next 24 hours. The focus is on data collected from 22:00 to 06:00 the following day, emphasizing nine key features: heart rate, systolic blood pressure, diastolic blood pressure, mean blood pressure, respiratory rate, temperature, SpO2, glucose, and FiO2. These features are essential for assessing physiological status and are commonly used for early sepsis detection.
"""

### 1.1 Extract Vital Sign Records
#This section details the extraction of nine vital sign features from the CHARTEVENTS table of the MIMIC-III dataset for trauma patients.
def extract_trauma_vitalsign(project_path_obj, project_id,
                              trauma_ids,
                              is_report=True):
    """
    Extracts and merges vital signs and FiO2 data for trauma patients from the MIMIC-III dataset.
    The extracted features include: 'HeartRate', 'SysBP', 'DiasBP', 'MeanBP', 'RespRate', 'TempC', 'SpO2', 'Glucose', and 'FiO2'.

    Parameters:
        project_path_obj (object): Provides paths to processed data files.
        project_id (str): Project identifier for BigQuery database access.
        trauma_ids (DataFrame): DataFrame containing IDs and their corresponding hospital admission information of trauma patients.
        is_report (bool): Flag to enable printing of summary statistics for the extracted data.

    Returns:
        DataFrame: A DataFrame containing vital signs and FiO2 data for the specified trauma patients,
                  sorted by 'icustay_id' and 'charttime'.

    Source files:
        - pivoted_vital.sql: Extracts general vital signs [View Script](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/pivot/pivoted_vital.sql)
        - pivoted_fio2.sql: Specifically extracts FiO2 levels [View Script](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/pivot/pivoted_fio2.sql)
    """
    # Load vital signs data
    path = project_path_obj.get_processed_data_file("pivoted_vital.csv")
    if os.path.exists(path):
        vital_df = pd.read_csv(path, index_col=0)
    else:
        query = """
        SELECT *
        FROM `physionet-data.mimiciii_derived.pivoted_vital`
        ORDER BY icustay_id, charttime;
        """
        vital_df = data_utils.run_query(query, project_id)
        vital_df.to_csv(path)

    # Load FiO2 data
    path = project_path_obj.get_processed_data_file("pivoted_fio2.csv")
    if os.path.exists(path):
        fio2_df = pd.read_csv(path, index_col=0)
    else:
        query = """
        SELECT *
        FROM `physionet-data.mimiciii_derived.pivoted_fio2`
        """
        fio2_df = data_utils.run_query(query, project_id)
        fio2_df = fio2_df[~(fio2_df.fio2.isna())]
        fio2_df.to_csv(path)

    # Merge trauma patients' IDs with FiO2 and vital signs data
    trauma_fio2 = trauma_ids.merge(fio2_df, on='icustay_id', how='inner')
    trauma_vital_df = trauma_ids.merge(vital_df, on='icustay_id', how='inner')
    raw_df = trauma_vital_df.merge(trauma_fio2, on=['subject_id', 'hadm_id', 'icustay_id', 'admittime', 'charttime'], how='outer')
    raw_df.rename(columns={'fio2': 'FiO2'}, inplace=True)

    if is_report:
        print(f"Extracted {trauma_fio2.shape[0]} FiO2 samples for {trauma_fio2['hadm_id'].nunique()} trauma patients.")
        print(f"Extracted {trauma_vital_df.shape[0]} vital sign samples for {trauma_vital_df['hadm_id'].nunique()} trauma patients.")
        print(f"Total samples after merging 2 tables: {raw_df.shape[0]} for {raw_df['hadm_id'].nunique()} trauma patients.")


    # Prepare datetime and time variables
    raw_df['admittime'] = pd.to_datetime(raw_df['admittime'])
    raw_df['charttime'] = pd.to_datetime(raw_df['charttime'])
    raw_df['Date'] = raw_df['charttime'].dt.date
    raw_df['Day'] = (raw_df['charttime'].dt.date - raw_df['admittime'].dt.date).apply(lambda x: x.days) + 1
    raw_df.loc[:,['Hour']] = raw_df.charttime.dt.hour

    return raw_df.sort_values(by=['icustay_id', 'charttime'])[
        ['subject_id', 'hadm_id', #'icustay_id',
         'Date', 'Day', 'Hour', #'admittime', 'charttime',
         'HeartRate', 'SysBP', 'DiasBP', 'MeanBP', 'RespRate', 'TempC', 'SpO2', 'Glucose', 'FiO2'
          ]]
## Example usage
#raw_vs = extract_trauma_vitalsign(project_path_obj, PROJECT_ID, trum_cohort_info_df, is_report=True)
#raw_vs.head()


### 1.2 Extract and Process Nighttime Data
#This section describes the process of aggregating and preparing nighttime data for analysis.
#The function performs the following tasks:
#1. **Nighttime Data Extraction**: Isolates data recorded between 22:00 and 06:00 for analysis.
#2. **Fill Missing Timestamps**: Ensures continuous time coverage by filling in any missing hourly timestamps.
#3. **Fill Missing Values**: Optionally fills missing data values based on the specified method.
#4. **Aggregation**: Combines multiple values recorded within the same hour into a single value for each feature.
#5. **Drop Invalid Data**: Removes rows with remaining NaN values, ensuring each row represents one patient's record at a specific timestamp.
def extract_night_data(df, filling_method=None, ffill_window_size=15):
  """
  Extracts and processes night-time data from the given DataFrame with raw data from the MIMIC-III dataset.

  This function optionally filters missing records and aggregates hourly values.
  At the end, it retains only data recorded during nighttime hours (22:00 to 06:00).

  Parameters:
  -----------
  df : pandas.DataFrame
      The input DataFrame containing raw input data with at least 'hadm_id', 'Day', 'Hour', and feature columns.

  filling_method : str, optional
      The method to use for filling missing values. Supported values are:
      - 'f_and_b': Forward fill with a specified window size followed by backward fill within the night-time period itself (up to 06:00).
      - 'forward': Forward fill with a specified window size.
      If None (default), no filling is applied, and the returned DataFrame may contain null values.

  ffill_window_size : int, optional (default=15)
      The size of the window (in hours) before the beginning of the nighttime period (22:00).
      This parameter is used only for forward filling; default is 15 hours, meaning data from 07:00 to 06:00 the next day will be used for forward filling.

  Returns:
  --------
  pandas.DataFrame
      A DataFrame containing the processed night-time data, with missing values filled (if specified)
      and aggregated into 2D arrays representing hourly data for each patient.
      If `filling_method` is None, the returned DataFrame may contain null values.

  Notes:
  ------
  - The function assumes that the DataFrame includes a 'Day' column representing the hospital day since admission and an 'Hour' column representing the hour of the day.
  - If `filling_method` is not None, the function will fill missing values.
   """
  # Filtering for nighttime hours
  if filling_method==None:
    # Extract nighttime data without filling
    night_df = df[(df['Hour'] >= 22) | (df['Hour'] <= 6)].sort_values(['hadm_id', 'Day', 'Hour'])
    print(f"Extracted nighttime data without filling: {night_df.shape[0]} samples for {night_df.hadm_id.nunique()} trauma patients")


    # Assign Night number and adjust dates for overnight periods
    night_df.loc[night_df['Hour']<=6, 'Day'] = (night_df.Day - 1)
    night_df.rename(columns={'Day': 'Night'}, inplace=True)
    night_df.loc[night_df['Hour']<=6, 'Date'] = (night_df.Date - timedelta(days=1))
  else:
    # Extend the time window based on the filling method
    # (i.e. if ffill_window_size=15, then ffill_window is 7am- next day 6am)
    window_s = 22-ffill_window_size
    window_e = 6 # backward fill uses data within the nighttime period (before 06:00).
    night_df = df[(df['Hour'] >= window_s) | (df['Hour'] <= window_e)].sort_values(['hadm_id', 'Day', 'Hour']) # with filling window
    # night_df_only_night = df[(df['Hour'] >= 22) | (df['Hour'] <= 6)]#.sort_values(['hadm_id', 'Day', 'Hour'])
    print(f"Extracted nighttime data with filling window: {night_df.shape[0]} samples for {night_df.hadm_id.nunique()} trauma patients")

    # Unifying data group for overnight dates with filling windows
    night_df.loc[night_df['Hour']<= window_e, 'Day'] = (night_df.Day - 1)
    night_df.rename(columns={'Day': 'Night'}, inplace=True)
    night_df.loc[night_df['Hour']<= window_e, 'Date'] = (night_df.Date - timedelta(days=1))

  # Fill missing timestamps in the nighttime range
  day_ids = ['subject_id', 'hadm_id','Date', 'Night']
  hour_ids = day_ids + ['Hour']
  night_time_list = [22, 23] + [i for i in range(7)]
  night_hour = night_df.groupby(day_ids).apply(
      lambda x: pd.DataFrame(night_time_list, columns=['Hour'])
      ).reset_index(names= day_ids +['TimeIndex'])
  full_night = night_df.merge(
      night_hour, on=hour_ids,how='outer'
      ).sort_values(['hadm_id', 'Night', 'TimeIndex'])
  print(f"After filling in missing timestamps: {full_night.shape[0]} samples for {full_night.hadm_id.nunique()} trauma patients")

  # Apply the filling method if specified
  if filling_method!=None:
    if (filling_method=='f_and_b'):
      # Forward fill followed by backward fill
      full_night = full_night.groupby(day_ids).apply(lambda group: group.ffill()).reset_index(drop=True)
      full_night = full_night.groupby(day_ids).apply(lambda group: group.bfill()).reset_index(drop=True)
      print(f"After forward and backward filling: {full_night.shape[0]} samples for {full_night.hadm_id.nunique()} trauma patients")

    if (filling_method=='forward'):
      # Forward fill only
      full_night = full_night.groupby(day_ids).apply(lambda group: group.ffill()).reset_index(drop=True)
      print(f"After forward filling: {full_night.shape[0]} samples for {full_night.hadm_id.nunique()} trauma patients")

  # Aggregate values in the same hour into one value per feature
  # day_ids = ['subject_id', 'hadm_id','Date', 'Night']
  # hour_ids = day_ids + ['Hour']
  night_AggInHour_df = full_night.groupby(hour_ids).mean().reset_index()
  print(f"After aggregating one hour into one value: {night_AggInHour_df.shape[0]} samples for {night_AggInHour_df.hadm_id.nunique()} trauma patients")

  if filling_method!=None:
    # Drop rows with remaining NaN values
    night_AggInHour_df.dropna(subset=night_AggInHour_df.columns, axis=0, how='any', inplace=True)
    print(f"After dropping NaN values: {night_AggInHour_df.shape[0]} samples for {night_AggInHour_df.hadm_id.nunique()} trauma patients")
    # Filter for rows between 22:00 and 06:00
    night_AggInHour_df = night_AggInHour_df[(night_AggInHour_df['Hour'] >= 22) | (night_AggInHour_df['Hour'] <= 6)]
    print(f"After removing filling window: {night_AggInHour_df.shape[0]} samples for {night_AggInHour_df.hadm_id.nunique()} trauma patients")

    # Keep only nights that have all 9 timestamps
    night_timestamp_count = night_AggInHour_df.groupby(day_ids).size().reset_index().rename({0:'num'}, axis=1)
    full_night_timestamp = night_timestamp_count.loc[night_timestamp_count.num==9, day_ids]
    night_AggInHour_df = night_AggInHour_df.merge(full_night_timestamp, on=day_ids)
    print(f"After retaining complete nights: {night_AggInHour_df.shape[0]} samples for {night_AggInHour_df.hadm_id.nunique()} trauma patients")

  return night_AggInHour_df.sort_values(['hadm_id', 'Night', 'TimeIndex'])
# # Example usage
## Extract night-time data with missing values retained
#data_w_null = extract_night_data(raw_vs, filling_method=None)
## Extract night-time data with missing values filled using forward and backward filling
#data_wo_null = extract_night_data(raw_vs, filling_method='f_and_b')

### 1.3 Convert to 2D Time-Series Data
#The final step converts the records into a 2D time-series format by grouping the data by night and aggregating 1D chart records. It then filters the nights to include only those from days 2 to 14, focusing on the critical period for early sepsis detection.
def gen_2Dnight_ti(df):
  """
  Groups by patient and night, then aggregates the values into 2D arrays.
  Each row represents one patient on one night.
  Filters the nights to include only those from days 2 to 14
  """
  index_columns = ['subject_id', 'hadm_id', #'icustay_id',
                   'Date', 'Night', 'Hour', 'TimeIndex']
  df = df.sort_values(index_columns)

  # Group by patient and night, then aggregate values into 2D arrays
  ti = df.groupby(['subject_id', 'hadm_id','Date','Night']).apply(
      lambda x: x.drop(columns=index_columns).values
      ).reset_index()
  ti.columns = ['subject_id', 'hadm_id', 'Date','Night', 'Temporal Features']
  print(f"After aggregating one night into 2D time-series, {ti.shape[0]} samples for {ti['hadm_id'].nunique()} trauma patients.")

  # Filter the nights to exclude the first 1 days
  ti_after2D = ti[(ti.Night>=2)]
  print(f"After filtering out the first night, {ti_after2D.shape[0]} samples for {ti_after2D['hadm_id'].nunique()} trauma patients.")
  # Filter out nights after day 14
  ti = ti_after2D[ti_after2D.Night<=14]
  print(f"After filtering out nights beyond day 14, {ti.shape[0]} samples for {ti['hadm_id'].nunique()} trauma patients.")

  return ti
# night_ti = gen_2Dnight_ti(night_data)
# night_ti.head()

"""
# 2. Instance Construction
This section involves labeling nighttime instances based on the sepsis onset data of each patient (HADM_ID). A nighttime instance is labeled 1 if **sepsis occurs within 24 hours after the nighttime instance**; otherwise, it is labeled 0. That means all nighttime instances of non-sepsis patients are assigned a negative label (0). For sepsis patients, only one nighttime instance receives a positive label (1), while the rest before the onset are labeled negative and the ones after onset are not of interest of early sepsis detection.
"""
## 2.1 Load Post-Trauma Sepsis Onset Timestamps
#Post-Trauma Sepsis is defined based on [Stern et al. (2023)](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2800552) and adheres to Sepsis-3 consensus guidelines. Each row records the sepsis label and the corresponding onset timestamp for a patient (HADM_ID).
#More detailed explanations and applications can be found in `notebooks/Sepsis_Onset_Label_Assignment.ipynb`.


## 2.2 Assign Instance Labels
#Assign labels to each nighttime instance based on the sepsis status of the patient. The label is set as follows:
#- **1**: If a patient develops sepsis within 24 hours after the nighttime instance. (excluding the instance at hour 0 and including up to 24 hours).
#- **0**: Otherwise.
#**Note**: Instances after sepsis onset are dropped, as they reflect a physiological status affected by sepsis treatment.
#This means that:
#- All instances for non-sepsis patients will be labeled as negative (0).
#- For sepsis patients, only one nighttime instance will be labeled as positive (1), while all other nighttime instances will be labeled as negative (0).
def assign_label2instance(ti_df, label_df):
    """
    Assigns labels (0/1) to nighttime instances based on sepsis onset timestamps.
    Specifically, assigns a positive label if sepsis onset occurs within 24 hours after the night.
    """
    # Identify sepsis and non-sepsis patient identifiers based on labels
    nonsepsis_ids = label_df.is_sepsis == 0
    sepsis_ids = label_df.is_sepsis == 1
    # print(f"Trauma Cohort: sepsis patients ({sum(sepsis_ids)}) + non-sepsis patients ({sum(nonsepsis_ids)}) = {label_df.shape[0]}")

    # Extract data for non-sepsis patients & assign negative label; these data are ready
    nonsepsis_patient_ti_df = ti_df[ti_df['hadm_id'].isin(label_df[nonsepsis_ids]['hadm_id'])]
    nonsepsis_patient_ti_df = nonsepsis_patient_ti_df.assign(Label=0)
    print(f"{nonsepsis_patient_ti_df.shape[0]} Negative instances for {sum(nonsepsis_ids)} non-sepsis patients")

    # Extract data for sepsis patients
    sepsis_patient_ti_df = ti_df[ti_df['hadm_id'].isin(label_df[sepsis_ids]['hadm_id'])]
    print(f"{sepsis_patient_ti_df.shape[0]} instances for {sum(sepsis_ids)} sepsis patients")

    sepsis_patient_df = sepsis_patient_ti_df.merge(label_df[['hadm_id', 'onset_datetime', 'onset_day']], on='hadm_id')

    # Classify the relationship between recorded time and onset time
    night_end_time = pd.to_datetime(sepsis_patient_df.Date) + pd.to_timedelta(1, unit='d') + pd.to_timedelta(6, unit='h')
    time_diff = (pd.to_datetime(sepsis_patient_df['onset_datetime']) - night_end_time)
    is_positive = (time_diff > pd.to_timedelta(0, unit='d')) & (time_diff <= pd.to_timedelta(1, unit='d'))
    sepsis_patient_df['Label'] = np.where(is_positive, 1, 0)
    # Drop instances after the onset time
    after_onset = (time_diff > pd.to_timedelta(1, unit='d'))
    sepsis_patient_df = sepsis_patient_df[~after_onset]
    print(f"Dropped {after_onset.sum()} instances after sepsis onset")
    print(f"\t {sepsis_patient_df.Label.value_counts()[1]} (1s) + {sepsis_patient_df.Label.value_counts()[0]} (0s)")

    # Combine data from sepsis and non-sepsis patients
    mimic_data_df = pd.concat([nonsepsis_patient_ti_df, sepsis_patient_df[nonsepsis_patient_ti_df.columns]])
    print(f"Final Dataset: {mimic_data_df['Label'].value_counts()[1]}(1s) + {mimic_data_df['Label'].value_counts()[0]}(0s) = {mimic_data_df.shape[0]} (Patients={mimic_data_df['hadm_id'].nunique()})")

    return mimic_data_df
# mimic_data_df = assign_label2instance(data_w_null, sepsis_label_df)
# mimic_data_df.head()

"""
# Integration and Execution Instance Construction
"""
def instance_construction(project_path_obj, project_id, trum_cohort_info_df, is_fill=True, is_report=True):
    """
    Extracts and processes night-time data from the trauma cohort based on specified parameters.

    Parameters:
    -----------
    project_path_obj : object
        The object that provides access to project paths.
    PROJECT_ID : str
        The ID of the project.
    trum_cohort_info_df : pandas.DataFrame
        DataFrame containing trauma cohort information.
    is_fill : bool, optional
        If True, fills missing values in night-time data using forward and backward filling. Default is True.
    is_report : bool, optional
        If True, generates a report. Default is True.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing processed night-time data, with missing values filled or retained as specified.
    """
    # Extract raw vital sign data
    raw_vs = extract_trauma_vitalsign(project_path_obj, project_id, trum_cohort_info_df, is_report=is_report)

    # Extract night-time data with or without filling missing values based on is_fill
    if is_fill:
        # Extract night-time data with missing values filled using forward and backward filling
        night_data = extract_night_data(raw_vs, filling_method='f_and_b')
    else:
        # Extract night-time data with missing values retained
        night_data = extract_night_data(raw_vs, filling_method=None)

    # Generate 2D night-time instances
    night_ti = gen_2Dnight_ti(night_data)

    # Load sepsis patient labels and corresponding onset timestamps
    # More detailed explanations and applications can be found in `notebooks/Sepsis_Onset_Label_Assignment.ipynb`.
    sepsis_label_path = project_path_obj.sepsis_label_path  # Define the path to sepsis labels
    if os.path.exists(sepsis_label_path):
        # If the file exists, load it from the specified path
        sepsis_label_df = pd.read_csv(sepsis_label_path, index_col=0)
    else:
        # If the file does not exist, generate the sepsis labels by querying the raw data
        sepsis_label_df = assign_sepsis_labels(project_path_obj,  # Pass object containing file paths
                                               project_id         # Provide the project ID for database access
        )

    # Assigns labels (0/1) to nighttime instances based on sepsis onset timestamps.
    mimic_data_df = assign_label2instance(night_ti, sepsis_label_df)
    return mimic_data_df

"""
# 3. Data Split

The requirements for data splitting to ensure fair test evaluation and prevent data leakage:
1. **Exclusive Subject IDs**: A subject ID will not appear in both the training and test datasets.
2. **Stratified Split**: The split is stratified to ensure that both training and validation datasets have similar positive instance ratios.
3. **Consistent Instances**: Datasets with and without missing values should share the same instances (i.e., from the same patient at the same time). In other words, they should have the same test dataset size.
"""
def split_and_report_data(data_with_nan, data_wo_nan, project_path_obj, is_saved=True, is_report=True):
    """
    Splits data into training and test sets while maintaining a balanced representation of sepsis and non-sepsis patients.
    Reports statistics on the split datasets if requested.

    Parameters:
    -----------
    data_with_nan : pandas.DataFrame
        DataFrame containing data with NaN values.
    data_wo_nan : pandas.DataFrame
        DataFrame containing data with NaN values filled.
    project_path_obj:
        Object containing paths to save the dataset.
    is_saved : bool, optional
        If True, saves the split datasets. Default is True.
    isReport : bool, optional
        If True, generates and displays a report on the split datasets. Default is True.

    Returns:
    --------
    tuple of pandas.DataFrame
        Two DataFrames: data_with_nan and data_wo_nan with updated dataset labels.
    """

    # Count records per patient in datasets with and without NaN values
    count_df_wo_nan = data_wo_nan.groupby(['subject_id', 'hadm_id']).agg({"Night": 'count', 'Label': 'max'}).reset_index()
    count_df_with_nan = data_with_nan.groupby(['subject_id', 'hadm_id']).agg({"Night": 'count', 'Label': 'max'}).reset_index()

    # Merge counts to find patients with the same records in both datasets
    count_df = count_df_with_nan.merge(count_df_wo_nan, on=['subject_id', 'hadm_id'], how='outer', suffixes=('_with_nan', '_wo_nan'))
    test_candidates = count_df[count_df.Night_with_nan == count_df.Night_wo_nan]

    print(f"Got {test_candidates.shape[0]} test candidates (same records should be in both tables).")

    # Split test candidates into training and test sets
    test_candidates = test_candidates.groupby(['subject_id']).agg({'Label_wo_nan': 'max'}).reset_index()
    test_candidates.rename(columns={'Label_wo_nan': 'label'}, inplace=True)
    test_size_wo_nan = int(0.2 * test_candidates.shape[0])

    id_train, id_test, label_train, label_test = train_test_split(
        test_candidates.subject_id, test_candidates.label,
        random_state=8, test_size=test_size_wo_nan, shuffle=True,
        stratify=test_candidates.label
    )

    # Assign dataset labels
    data_wo_nan.loc[:, 'Dataset'] = 'train'
    data_wo_nan.loc[data_wo_nan.subject_id.isin(id_test), 'Dataset'] = 'test'
    # data_wo_nan.Dataset.fillna('train')
    data_with_nan.loc[:, 'Dataset'] = 'train'
    data_with_nan.loc[data_with_nan.subject_id.isin(id_test), 'Dataset'] = 'test'
    # data_with_nan.Dataset.fillna('train')

    # Report statistics
    if is_report:
        for name, df in {"with NaN": data_with_nan, "w/o NaN": data_wo_nan}.items():
            print(f"For data table {name} with shape {df.shape} for {df.hadm_id.nunique()} patients(hadm_id).")
            report_df = pd.DataFrame(columns=['NumInstance', 'NumPosInstance', 'RatioPosInstance', 'NumPatient(subject_id)', 'NumSepPatient(subject_id)', 'RatioSepPatient(subject_id)'],
                                     index=['test', 'train'])

            # Instance-level counts
            test_0, test_1, train_0, train_1 = df.groupby(['Dataset', 'Label']).size()
            report_df.loc[:, ['NumInstance', 'NumPosInstance', 'RatioPosInstance']] = [
                [test_0 + test_1, test_1, round(test_1 / (test_0 + test_1), 3)],
                [train_0 + train_1, train_1, round(train_1 / (train_0 + train_1), 3)]
            ]

            # Patient-level counts
            test_0, test_1, train_0, train_1 = df.groupby(['Dataset', 'subject_id']).Label.max().reset_index().groupby(['Dataset', 'Label']).size()
            report_df.loc[:, ['NumPatient(subject_id)', 'NumSepPatient(subject_id)', 'RatioSepPatient(subject_id)']] = [
                [test_0 + test_1, test_1, round(test_1 / (test_0 + test_1), 3)],
                [train_0 + train_1, train_1, round(train_1 / (train_0 + train_1), 3)]
            ]
            # print(f"Patient-level (subject_id) counts:")
            display(report_df)

    # Save datasets
    if is_saved:
      print(f"Saving datasets to {project_path_obj.dataset_with_nan_path}")
      data_with_nan.to_pickle(project_path_obj.dataset_with_nan_path)
      print(f"Saving datasets to {project_path_obj.dataset_wo_nan_path}")
      data_wo_nan.to_pickle(project_path_obj.dataset_wo_nan_path)
    return data_with_nan, data_wo_nan


"""
# Integration and Execution Dataset Construction
"""
def dataset_construction(project_path_obj, project_id, is_report=True):
    """
    Generates and saves 2 datasets: one with NaN values filled and one with NaN values.
    Each dataset contains the following columns:
    - Temporal Features: Multivariate time-series input data with shape (# of timestamps, # of features).
    - Label: 0/1, representing the output label.
    - Dataset: Indicates whether this instance is part of the training or test dataset.
    
    Each row represents one nighttime instance, which is associated with patient IDs ('subject_id', 'hadm_id') and timestamp ID (Night).

    Parameters:
    -----------
    project_path_obj : object
        Provides paths to processed data files.
    project_id : str
        Project identifier for BigQuery database access.
    trauma_cohort_info_df : DataFrame
        DataFrame containing trauma cohort ids and their corresponding hospital admission information of trauma patients.
    is_report : bool, optional (default=True)
        If True, enables printing of summary statistics for the datasets.

    Returns:
    --------
    tuple of DataFrames
        - DataFrame with NaN values.
        - DataFrame without NaN values.
    """
    # Load Trauma Cohort
    # Detailed explanations of the cohort extraction process can be found in the `notebooks/cohort_extraction.ipynb` file.
    if os.path.exists(project_path_obj.trauma_cohort_info_path):
        # Load the existing file
        trum_ids = pd.read_csv(project_path_obj.trauma_cohort_info_path, index_col=0)
    else:
        # File does not exist, extract cohort IDs and generate statistics report
        trum_ids = extract_trauma_cohort_ids(project_path_obj,    # Saved file paths
                                             project_id,          # To query raw data
                                             is_report=False,     # Print statistics report
                                             is_saved=True        # Save the cohort IDs
                                            )
    # This table should contain the hospital admission IDs (hadm_id) of trauma patients and the corresponding admission information.
    # 1 row per patient
    trauma_cohort_info_df = trum_ids[['subject_id', 'hadm_id', 'icustay_id', 'admittime']]
    
    # Generate dataset without NaN values
    print("Generating Dataset without NaN values...")
    data_wo_nan = instance_construction(project_path_obj, project_id, trauma_cohort_info_df, is_fill=True, is_report=is_report)
    
    # Generate dataset with NaN values
    print("\nGenerating Dataset with NaN values...")
    data_with_nan = instance_construction(project_path_obj, project_id, trauma_cohort_info_df, is_fill=False, is_report=is_report)
    
    # Split, save, and report data
    data_with_nan_df, data_wo_nan_df = split_and_report_data(data_with_nan.copy(), data_wo_nan.copy(), project_path_obj, is_saved=False, is_report=is_report)
    
    return data_with_nan_df, data_wo_nan_df
# Example usage
# data_with_nan_df, data_wo_nan_df = dataset_construction(project_path_obj, PROJECT_ID, is_report=True)
