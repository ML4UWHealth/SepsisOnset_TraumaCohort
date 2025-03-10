import os

import numpy as np
import pandas as pd


# Read data from BigQuery(sql) into pandas dataframes.
def run_query(query, project_id):
  """
  Executes a SQL query on Google BigQuery and returns the result as a DataFrame.

  Args:
  - query (str): The SQL query to execute.
  - project_id (str): The Google Cloud project ID used for accessing BigQuery.

  Returns:
  - DataFrame: The result of the query as a pandas DataFrame.
  """
  os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
  return pd.io.gbq.read_gbq(
      query,
      project_id=project_id,
      dialect='standard')

def test_mimiciii_bigquery_access(project_id):
    """
    Test if Google Colab can successfully access the MIMIC III v1.4 data through BigQuery.

    Parameters:
    project_id (str): The Google Cloud Project ID for accessing MIMIC III data.

    Returns:
    bool: True if access is successful, False otherwise.
    """
    query = """
            -- total number of hospital admission IDs in MIMIC III
            SELECT COUNT(DISTINCT HADM_ID)
            FROM `physionet-data.mimiciii_clinical.admissions`
            """
    try:
        df = run_query(query, project_id)
        if df.values[0][0] == 58976:
            print("Successfully accessed the MIMIC III Via BigQuery")
            return True
        else:
            print("Access to MIMIC III failed. Ensure you are using MIMIC III v1.4.")
            return False
    except Exception as e:
        print(f"Error accessing MIMIC III via BigQuery: {str(e)}")
        return False

# test_mimiciii_bigquery_access('sepsis-mimic3')



