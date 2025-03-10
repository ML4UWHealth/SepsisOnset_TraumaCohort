import os
import numpy as np
import pandas as pd

from .sql2df import demog_sql2df, ventilation_day_processed

def get_demographics_data(project_path_obj, project_id):
    """
    Load the demographics data from a CSV file if it exists, otherwise query it using BigQuery.
    More detailed information about the demographics table can be found in the src.sql2df.demog_sql2df function.

    Args:
    - project_path_obj: An object that provides the path to the raw data file.
    - project_id: The Google Cloud project ID used for accessing MIMIC data via BigQuery..

    Returns:
    - demog_df: A DataFrame containing the demographics data.
    """
    # Check if the file exists
    demog_path = project_path_obj.get_raw_data_file('demographics.csv')
    if os.path.exists(demog_path):
        # Load the CSV file into a DataFrame
        demog_df = pd.read_csv(demog_path, index_col=0)
    else:
        # Query demographics information by using BigQuery
        demog_df = demog_sql2df(project_id, saved_path=demog_path)
    
    return demog_df

def get_ventilation_data(project_path_obj, project_id):
    """
    Load the ventilation day data from a CSV file if it exists, otherwise query it using BigQuery.

    Args:
    - project_path_obj: An object that provides the path to the raw data file.
    - project_id: The Google Cloud project ID used for accessing MIMIC data via BigQuery.

    Returns:
    - vent_df:  A DataFrame containing the ventilation day data. ()
                This data represents the number of days the patient (HADM_ID) was receiving ventilation events, 
                regardless of how many hours in that day the patient received ventilation.
    """
    # Get the path to the ventilation day CSV file
    vent_path = project_path_obj.get_processed_data_file('MVday.csv')
    
    if os.path.exists(vent_path):
        # Load the CSV file into a DataFrame
        vent_df = pd.read_csv(vent_path, index_col=0)
    else:
        # Query ventilation day data using BigQuery
        vent_df = ventilation_day_processed(project_id, vent_type=['MechVent'], saved_path=vent_path)
    
    return vent_df
