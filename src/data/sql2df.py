#import gzip #local(Drive) mimic dataset( file type .csv.gz file)
import os
import re
import numpy as np
import pandas as pd
from datetime import datetime

from .data_utils import run_query

###################################
# Raw DATA
###################################


####################### Demographics ##################################################################
# Description: This query provides a useful set of information regarding patient
#              ICU stays. The information is combined from the admissions, patients, and icustays tables. It includes age, length of stay, sequence, and expiry flags.
# MIMIC version: MIMIC-III v1.4
# A modified version of: https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/demographics/icustay_detail.sql
# Modifications include:
#   1. Duration result is expressed in hours instead of in days
# ------------------------------------------------------------------
# This table includes:
# *   Patient Info:
#  >  'subject_id', 'gender', 'dod', 'ethnicity', 'ethnicity_grouped'
# *   Hospital Admission Info:
#  >  'hadm_id', 'admittime', 'dischtime', 'los_hospital_hours', 'admission_age', 'hospital_expire_flag' (0/1), 'hospstay_seq', 'first_hosp_stay' (T/F)
# * ICUstay Info:
#  > 'icustay_id', 'intime', 'outtime', 'los_icu_hours', 'icustay_seq', 'first_icu_stay' (T/F)
########################################################################################################

def demog_sql2df(project_id, saved_path=None):
  demog_query = """
  SELECT ie.subject_id, ie.hadm_id, ie.icustay_id
  -- patient level factors
  , pat.gender, pat.dod
  -- hospital level factors
  , adm.admittime, adm.dischtime
  , DATETIME_DIFF(adm.dischtime, adm.admittime, DAY) as los_hospital_days
  , DATETIME_DIFF(adm.dischtime, adm.admittime, HOUR) as los_hospital_hours 
  , DATETIME_DIFF(ie.intime, pat.dob, YEAR) as admission_age
  , adm.ethnicity
  , case when ethnicity in
    (
        'WHITE' --  40996
      , 'WHITE - RUSSIAN' --    164
      , 'WHITE - OTHER EUROPEAN' --     81
      , 'WHITE - BRAZILIAN' --     59
      , 'WHITE - EASTERN EUROPEAN' --     25
    ) then 'white'
    when ethnicity in
    (
        'BLACK/AFRICAN AMERICAN' --   5440
      , 'BLACK/CAPE VERDEAN' --    200
      , 'BLACK/HAITIAN' --    101
      , 'BLACK/AFRICAN' --     44
      , 'CARIBBEAN ISLAND' --      9
    ) then 'black'
    when ethnicity in
      (
        'HISPANIC OR LATINO' --   1696
      , 'HISPANIC/LATINO - PUERTO RICAN' --    232
      , 'HISPANIC/LATINO - DOMINICAN' --     78
      , 'HISPANIC/LATINO - GUATEMALAN' --     40
      , 'HISPANIC/LATINO - CUBAN' --     24
      , 'HISPANIC/LATINO - SALVADORAN' --     19
      , 'HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)' --     13
      , 'HISPANIC/LATINO - MEXICAN' --     13
      , 'HISPANIC/LATINO - COLOMBIAN' --      9
      , 'HISPANIC/LATINO - HONDURAN' --      4
    ) then 'hispanic'
    when ethnicity in
    (
        'ASIAN' --   1509
      , 'ASIAN - CHINESE' --    277
      , 'ASIAN - ASIAN INDIAN' --     85
      , 'ASIAN - VIETNAMESE' --     53
      , 'ASIAN - FILIPINO' --     25
      , 'ASIAN - CAMBODIAN' --     17
      , 'ASIAN - OTHER' --     17
      , 'ASIAN - KOREAN' --     13
      , 'ASIAN - JAPANESE' --      7
      , 'ASIAN - THAI' --      4
    ) then 'asian'
    when ethnicity in
    (
        'AMERICAN INDIAN/ALASKA NATIVE' --     51
      , 'AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE' --      3
    ) then 'native'
    when ethnicity in
    (
        'UNKNOWN/NOT SPECIFIED' --   4523
      , 'UNABLE TO OBTAIN' --    814
      , 'PATIENT DECLINED TO ANSWER' --    559
    ) then 'unknown'
    else 'other' end as ethnicity_grouped
    -- , 'OTHER' --   1512
    -- , 'MULTI RACE ETHNICITY' --    130
    -- , 'PORTUGUESE' --     61
    -- , 'MIDDLE EASTERN' --     43
    -- , 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER' --     18
    -- , 'SOUTH AMERICAN' --      8
  , adm.hospital_expire_flag
  , DENSE_RANK() OVER (PARTITION BY adm.subject_id ORDER BY adm.admittime) AS hospstay_seq
  , CASE
      WHEN DENSE_RANK() OVER (PARTITION BY adm.subject_id ORDER BY adm.admittime) = 1 THEN True
      ELSE False END AS first_hosp_stay
  -- icu level factors
  , ie.intime, ie.outtime
  , DATETIME_DIFF(ie.outtime, ie.intime, DAY) as los_icu_days
  , DATETIME_DIFF(ie.outtime, ie.intime, HOUR) as los_icu_hours
  , DENSE_RANK() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime) AS icustay_seq
  -- first ICU stay *for the current hospitalization*
  , CASE
      WHEN DENSE_RANK() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime) = 1 THEN True
      ELSE False END AS first_icu_stay
  FROM `physionet-data.mimiciii_clinical.icustays` ie
  INNER JOIN `physionet-data.mimiciii_clinical.admissions` adm
      ON ie.hadm_id = adm.hadm_id
  INNER JOIN `physionet-data.mimiciii_clinical.patients` pat
      ON ie.subject_id = pat.subject_id
  WHERE adm.has_chartevents_data = 1
  ORDER BY ie.subject_id, adm.admittime, ie.intime;
  """
  demog_df = run_query(demog_query, project_id)
  if saved_path != None:
    print("File saved at:", saved_path)
    demog_df.to_csv(saved_path)
  return demog_df

####################### Antibiotic ##################################################################
# Description: This query extracts antibiotic events related to Post-trauma Sepsis from Prescriptions table. 
#              It focuses on selecting relevant antibiotic administration records that are pertinent to the study of sepsis in trauma patients. 
# MIMIC Version: MIMIC-III v1.4
# A modified version of: https://github.com/MIT-LCP/mimic-code/blob/b9ed7a3d22a85dd95a50797e15bd24d566bce337/mimic-iv/concepts/medication/antibiotic.sql#L4

########################################################################################################
def abx_sql2df(project_id):
  abx_df = run_query(
    """
    WITH abx AS (
        SELECT DISTINCT
            gsn
            , drug
            , route
            , CASE
                WHEN LOWER(drug) LIKE '%amikacin%' THEN 1
                WHEN LOWER(drug) LIKE '%amphotericin%' THEN 1
                WHEN LOWER(drug) LIKE '%ampicillin%' THEN 1
                WHEN LOWER(drug) LIKE '%azithromycin%' THEN 1
                WHEN LOWER(drug) LIKE '%aztreonam%' THEN 1
                WHEN LOWER(drug) LIKE '%cefazolin%' THEN 1
                WHEN LOWER(drug) LIKE '%ceftazidime%' THEN 1
                WHEN LOWER(drug) LIKE '%cefepime%' THEN 1
                WHEN LOWER(drug) LIKE '%cefotetan%' THEN 1
                WHEN LOWER(drug) LIKE '%cefotaxime%' THEN 1
                WHEN LOWER(drug) LIKE '%ceftriaxone%' THEN 1
                WHEN LOWER(drug) LIKE '%cefuroxime%' THEN 1
                WHEN LOWER(drug) LIKE '%cipro%' THEN 1
                WHEN LOWER(drug) LIKE '%ciprofloxacin%' THEN 1
                WHEN LOWER(drug) LIKE '%clindamycin%' THEN 1
                WHEN LOWER(drug) LIKE '%doxycy%' THEN 1
                WHEN LOWER(drug) LIKE '%erythromycin%' THEN 1
                WHEN LOWER(drug) LIKE '%gentamicin%' THEN 1
                WHEN LOWER(drug) LIKE '%levofloxacin%' THEN 1
                WHEN LOWER(drug) LIKE '%linezolid%' THEN 1
                WHEN LOWER(drug) LIKE '%metronidazole%' THEN 1
                WHEN LOWER(drug) LIKE '%meropenem%' THEN 1
                WHEN LOWER(drug) LIKE '%metronidazole%' THEN 1
                WHEN LOWER(drug) LIKE '%meropenem%' THEN 1
                WHEN LOWER(drug) LIKE '%minocycline%' THEN 1
                WHEN LOWER(drug) LIKE '%moxifloxacin%' THEN 1
                WHEN LOWER(drug) LIKE '%nafcillin%' THEN 1
                WHEN LOWER(drug) LIKE '%penicillin%' THEN 1
                WHEN LOWER(drug) LIKE '%piperacillin%' THEN 1
                WHEN LOWER(drug) LIKE '%rifampin%' THEN 1
                WHEN LOWER(drug) LIKE '%sulfamethoxazole%' THEN 1
                WHEN LOWER(drug) LIKE '%trimethoprim%' THEN 1
                WHEN LOWER(drug) LIKE '%vancomycin%' THEN 1
                -- Additional abx
                WHEN LOWER(drug) LIKE '%ertapenem%' THEN 1
                WHEN LOWER(drug) LIKE '%imipenem-cilastatin%' THEN 1
                ELSE 0
            END AS antibiotic
        FROM `physionet-data.mimiciii_clinical.prescriptions`
        WHERE route IN (
          'IV', 'IV DRIP', 'IVPCA', 'IV BOLUS', 'EX-VIVO', 'PO/IV', 'IVT', 'IVS' -- iv
          ,'PO/NG','PO', 'NG', 'ORAL' -- oral
        )
    )
    SELECT
        pr.subject_id, pr.hadm_id, pr.icustay_id
        , pr.gsn
        , pr.drug --AS antibiotic
        , pr.drug_name_generic
        , pr.route
        , pr.startdate
        , pr.enddate
    FROM `physionet-data.mimiciii_clinical.prescriptions` pr
    -- inner join to subselect to only antibiotic prescriptions
    INNER JOIN abx
        ON pr.drug = abx.drug
            AND pr.route = abx.route
    WHERE abx.antibiotic = 1
    ;
    """, project_id).drop_duplicates()
  return abx_df

####################### SOFA score ##################################################################
# Description: This function calculates a modified version of the SOFA (Sequential Organ Failure Assessment) score,
#              which omits the Glasgow Coma Scale (GCS) and Urine Output (UO) components.
#              The score is calculated for every hour of the adult patients' ICU stay.
# 
# MIMIC Version: MIMIC-III v1.4
# A modified version of the SOFA calculation from the MIT-LCP repository: 
#   https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/pivot/pivoted_sofa.sql

########################################################################################################
def SOFA_calculate(project_id, saved_path=None):
  sofa_query = """
    -- ------------------------------------------------------------------
    -- Title: A modified version of the SOFA(Sequential Organ Failure Assessment) score
    -- This query extracts the sequential organ failure assessment (formally: sepsis-related organ failure assessment).
    -- This score is a measure of organ failure for patients in the ICU.
    -- The score is calculated for every hour of the patient's ICU stay.
    -- However, as the calculation window is 24 hours, care should be taken when
    -- using the score before the end of the first day.
    -- ------------------------------------------------------------------

    -- Variables used in SOFA:
    --  MAP, FiO2, Ventilation status (sourced FROM `physionet-data.mimiciii_clinical.chartevents`)
    --  Creatinine, Bilirubin, FiO2, PaO2, Platelets (sourced FROM `physionet-data.mimiciii_clinical.labevents`)
    --  Dopamine, Dobutamine, Epinephrine, Norepinephrine (sourced FROM `physionet-data.mimiciii_clinical.inputevents_mv` and INPUTEVENTS_CV)

    -- The following views required to run this query:
    --  1) pivoted_bg_art - generated by pivoted-bg.sql
    --  2) (Excluded) pivoted_uo - generated by pivoted-uo.sql
    --  3) pivoted_lab - generated by pivoted-lab.sql
    --  4) (Excluded) pivoted_gcs - generated by pivoted-gcs.sql
    --  5) ventilation_durations - generated by ../durations/ventilation_durations.sql
    --  6) norepinephrine_dose - generated by ../durations/norepinephrine-dose.sql
    --  7) epinephrine_dose - generated by ../durations/epinephrine-dose.sql
    --  8) dopamine_dose - generated by ../durations/dopamine-dose.sql
    --  9) dobutamine_dose - generated by ../durations/dobutamine-dose.sql

    -- Note:
    -- The score is calculated only for adult ICU patients,
    -- generating a row for every hour the patient was in the ICU.

    WITH co AS
    (
      select ie.hadm_id, ih.icustay_id
      , hr
      -- start/endtime can be used to filter to values within this hour
      , DATETIME_SUB(ih.endtime, INTERVAL '1' HOUR) AS starttime
      , ih.endtime
      from `physionet-data.mimiciii_derived.icustay_hours` ih
      INNER JOIN `physionet-data.mimiciii_clinical.icustays` ie
        ON ih.icustay_id = ie.icustay_id
    )
    -- get minimum blood pressure FROM `physionet-data.mimiciii_clinical.chartevents`
    , bp as
    (
      select ce.icustay_id
        , ce.charttime
        , min(valuenum) as meanbp_min
      FROM `physionet-data.mimiciii_clinical.chartevents` ce
      -- exclude rows marked as error
      where (ce.error IS NULL OR ce.error != 1)
      and ce.itemid in
      (
      -- MEAN ARTERIAL PRESSURE
      456, --"NBP Mean"
      52, --"Arterial BP Mean"
      6702, --	Arterial BP Mean #2
      443, --	Manual BP Mean(calc)
      220052, --"Arterial Blood Pressure mean"
      220181, --"Non Invasive Blood Pressure mean"
      225312  --"ART BP mean"
      )
      and valuenum > 0 and valuenum < 300
      group by ce.icustay_id, ce.charttime
    )
    , mini_agg as
    (
      select co.icustay_id, co.hr
      -- vitals
      , min(bp.meanbp_min) as meanbp_min
      -- labs
      , max(labs.bilirubin) as bilirubin_max
      , max(labs.creatinine) as creatinine_max
      , min(labs.platelet) as platelet_min
      -- because pafi has an interaction between vent/PaO2:FiO2, we need two columns for the score
      -- it can happen that the lowest unventilated PaO2/FiO2 is 68, but the lowest ventilated PaO2/FiO2 is 120
      -- in this case, the SOFA score is 3, *not* 4.
      , min(case when vd.icustay_id is null then pao2fio2ratio else null end) AS pao2fio2ratio_novent
      , min(case when vd.icustay_id is not null then pao2fio2ratio else null end) AS pao2fio2ratio_vent
      from co
      left join bp
        on co.icustay_id = bp.icustay_id
        and co.starttime < bp.charttime
        and co.endtime >= bp.charttime
      left join `physionet-data.mimiciii_derived.pivoted_lab` labs
        on co.hadm_id = labs.hadm_id
        and co.starttime < labs.charttime
        and co.endtime >= labs.charttime
      -- bring in blood gases that occurred during this hour
      left join `physionet-data.mimiciii_derived.pivoted_bg_art` bg
        on co.icustay_id = bg.icustay_id
        and co.starttime < bg.charttime
        and co.endtime >= bg.charttime
      -- at the time of the blood gas, determine if patient was ventilated
      left join `physionet-data.mimiciii_derived.ventilation_durations` vd
        on co.icustay_id = vd.icustay_id
        and bg.charttime >= vd.starttime
        and bg.charttime <= vd.endtime
      group by co.icustay_id, co.hr
    )
    , scorecomp as
    (
      select
          co.hadm_id
        , co.icustay_id
        , co.hr
        , co.starttime, co.endtime
        , ma.pao2fio2ratio_novent
        , ma.pao2fio2ratio_vent
        , epi.vaso_rate as rate_epinephrine
        , nor.vaso_rate as rate_norepinephrine
        , dop.vaso_rate as rate_dopamine
        , dob.vaso_rate as rate_dobutamine
        , ma.meanbp_min
        -- labs
        , ma.bilirubin_max
        , ma.creatinine_max
        , ma.platelet_min
      from co
      left join mini_agg ma
        on co.icustay_id = ma.icustay_id
        and co.hr = ma.hr
      left join `physionet-data.mimiciii_derived.epinephrine_dose` epi
        on co.icustay_id = epi.icustay_id
        and co.endtime > epi.starttime
        and co.endtime <= epi.endtime
      left join `physionet-data.mimiciii_derived.norepinephrine_dose` nor
        on co.icustay_id = nor.icustay_id
        and co.endtime > nor.starttime
        and co.endtime <= nor.endtime
      left join `physionet-data.mimiciii_derived.dopamine_dose` dop
        on co.icustay_id = dop.icustay_id
        and co.endtime > dop.starttime
        and co.endtime <= dop.endtime
      left join `physionet-data.mimiciii_derived.dobutamine_dose` dob
        on co.icustay_id = dob.icustay_id
        and co.endtime > dob.starttime
        and co.endtime <= dob.endtime
    )
    , scorecalc as
    (
      -- Calculate the final score
      -- note that if the underlying data is missing, the component is null
      -- eventually these are treated as 0 (normal), but knowing when data is missing is useful for debugging
      select scorecomp.*
      -- Respiration
      , cast(case
          when pao2fio2ratio_vent   < 100 then 4
          when pao2fio2ratio_vent   < 200 then 3
          when pao2fio2ratio_novent < 300 then 2
          when pao2fio2ratio_novent < 400 then 1
          when coalesce(pao2fio2ratio_vent, pao2fio2ratio_novent) is null then null
          else 0
        end as SMALLINT) as respiration

      -- Coagulation
      , cast(case
          when platelet_min < 20  then 4
          when platelet_min < 50  then 3
          when platelet_min < 100 then 2
          when platelet_min < 150 then 1
          when platelet_min is null then null
          else 0
        end as SMALLINT) as coagulation

      -- Liver
      , cast(case
          -- Bilirubin checks in mg/dL
            when Bilirubin_Max >= 12.0 then 4
            when Bilirubin_Max >= 6.0  then 3
            when Bilirubin_Max >= 2.0  then 2
            when Bilirubin_Max >= 1.2  then 1
            when Bilirubin_Max is null then null
            else 0
          end as SMALLINT) as liver

      -- Cardiovascular
      , cast(case
          when rate_dopamine > 15 or rate_epinephrine >  0.1 or rate_norepinephrine >  0.1 then 4
          when rate_dopamine >  5 or rate_epinephrine <= 0.1 or rate_norepinephrine <= 0.1 then 3
          when rate_dopamine >  0 or rate_dobutamine > 0 then 2
          when meanbp_min < 70 then 1
          when coalesce(meanbp_min, rate_dopamine, rate_dobutamine, rate_epinephrine, rate_norepinephrine) is null then null
          else 0
        end as SMALLINT) as cardiovascular

      -- Renal failure - high creatinine or low urine output
      , cast(case
        when (Creatinine_Max >= 5.0) then 4
        when (Creatinine_Max >= 3.5 and Creatinine_Max < 5.0) then 3
        when (Creatinine_Max >= 2.0 and Creatinine_Max < 3.5) then 2
        when (Creatinine_Max >= 1.2 and Creatinine_Max < 2.0) then 1
        when Creatinine_Max is null then null
        else 0 end as SMALLINT)
        as renal

      from scorecomp
      WINDOW W as
      (
        PARTITION BY icustay_id
        ORDER BY hr
        ROWS BETWEEN 23 PRECEDING AND 0 FOLLOWING
      )
    )
    , score_final as
    (
      select s.*
        -- Combine all the scores to get SOFA
        -- Impute 0 if the score is missing
      -- the window function takes the max over the last 24 hours
        , cast(coalesce(
            MAX(respiration) OVER (PARTITION BY icustay_id ORDER BY HR
            ROWS BETWEEN 24 PRECEDING AND 0 FOLLOWING)
          ,0) as SMALLINT) as respiration_24hours
        , cast(coalesce(
            MAX(coagulation) OVER (PARTITION BY icustay_id ORDER BY HR
            ROWS BETWEEN 24 PRECEDING AND 0 FOLLOWING)
            ,0) as SMALLINT) as coagulation_24hours
        , cast(coalesce(
            MAX(liver) OVER (PARTITION BY icustay_id ORDER BY HR
            ROWS BETWEEN 24 PRECEDING AND 0 FOLLOWING)
          ,0) as SMALLINT) as liver_24hours
        , cast(coalesce(
            MAX(cardiovascular) OVER (PARTITION BY icustay_id ORDER BY HR
            ROWS BETWEEN 24 PRECEDING AND 0 FOLLOWING)
          ,0) as SMALLINT) as cardiovascular_24hours
        , cast(coalesce(
            MAX(renal) OVER (PARTITION BY icustay_id ORDER BY HR
            ROWS BETWEEN 24 PRECEDING AND 0 FOLLOWING)
          ,0) as SMALLINT) as renal_24hours

        -- sum together data for final SOFA
        , coalesce(
            MAX(respiration) OVER (PARTITION BY icustay_id ORDER BY HR
            ROWS BETWEEN 24 PRECEDING AND 0 FOLLOWING)
          ,0)
        + coalesce(
            MAX(coagulation) OVER (PARTITION BY icustay_id ORDER BY HR
            ROWS BETWEEN 24 PRECEDING AND 0 FOLLOWING)
          ,0)
        + coalesce(
            MAX(liver) OVER (PARTITION BY icustay_id ORDER BY HR
            ROWS BETWEEN 24 PRECEDING AND 0 FOLLOWING)
          ,0)
        + coalesce(
            MAX(cardiovascular) OVER (PARTITION BY icustay_id ORDER BY HR
            ROWS BETWEEN 24 PRECEDING AND 0 FOLLOWING)
          ,0)
        + cast(coalesce(
            MAX(renal) OVER (PARTITION BY icustay_id ORDER BY HR
            ROWS BETWEEN 24 PRECEDING AND 0 FOLLOWING)
          ,0) as SMALLINT)
        as sofa_24hours
      from scorecalc s
      WINDOW W as
      (
        PARTITION BY icustay_id
        ORDER BY hr
        ROWS BETWEEN 23 PRECEDING AND 0 FOLLOWING
      )
    )
    select * from score_final
    where hr >= 0
    order by icustay_id, hr;
    """
  sofa_df = run_query(sofa_query, project_id).sort_values(['hadm_id', 'icustay_id', 'hr']).reset_index(drop=True)
  if saved_path is not None:
      print("Saved SOFA score at", saved_path)
      sofa_df.to_csv(saved_path)
  return sofa_df


###################################
# Processed DATA
###################################

def ventilation_day_processed(project_id, vent_type=['MechVent'], saved_path=None):
  '''
  Identify the presence of mechanical ventilation
  - Based on source file: [ventilation_classification.sql](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/durations/ventilation_classification.sql)

  Compute the number of days the patient (HADM_ID) was receiving ventilation events
  - Regardless of how many hours in that day the patient received ventilation

  Parameters:
    project_id: BigQuery MIMIC-III Project ID
    vent_type: A subset of 4 ventilation types ['MechVent', 'OxygenTherapy', 'Extubated', 'SelfExtubated']. 
              This function only counts the ventilation types within this subset.
              By default, only 'MechVent' will be considered a qualifying ventilation event.
    saved_path: (Optional) path to save the resulting CSV file.
  '''
  # Identify the presence of a mechanical ventilation using settings
  vent_df = run_query(
      """
      SELECT i.hadm_id, v.*
      FROM `physionet-data.mimiciii_derived.ventilation_classification` v
      JOIN `physionet-data.mimiciii_clinical.icustays` i
      ON v.ICUSTAY_ID = i.ICUSTAY_ID;
      """, project_id)

  # Select qualified ventilation event according to vent_type
  vent_df['sum'] = vent_df[vent_type].sum(axis=1)
  qualified_vent_df = vent_df[vent_df['sum']>0]
  numevent = qualified_vent_df.shape[0]
  # Get date
  qualified_vent_df['date_count'] = pd.to_datetime(qualified_vent_df['charttime']).dt.date
  vent_day_df = qualified_vent_df[['hadm_id', 'date_count']].drop_duplicates()
  # Count ventilation days: if a patient get ventilation (regaless of specific hours), then +1
  vent_day_count = vent_day_df.groupby('hadm_id').date_count.count().reset_index()
  if saved_path is not None:
    print("Saved mechanical ventilation day at",  saved_path)
    vent_day_count.to_csv(saved_path)
  return vent_day_count

##########
# Input Features:
# vital_signs & FiO2
#########

def vital_signs_sql2df(project_id, saved_path=None):
  """
  A modified version of pivoted_vital.sql: (https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/pivot/pivoted_vital.sql)
  * add hadm_id to the return table 

  """
  vs_query = """
  -- This query pivots the vital signs a patient's stay (hadm_id)
  -- Vital signs include heart rate, blood pressure, respiration rate, temperature, spo2 and glucose

  with ce as
  (
    select ce.hadm_id, ce.icustay_id
      , ce.charttime
      , (case when itemid in (211,220045) and valuenum > 0 and valuenum < 300 then valuenum else null end) as heartrate
      , (case when itemid in (51,442,455,6701,220179,220050) and valuenum > 0 and valuenum < 400 then valuenum else null end) as sysbp
      , (case when itemid in (8368,8440,8441,8555,220180,220051) and valuenum > 0 and valuenum < 300 then valuenum else null end) as diasbp
      , (case when itemid in (456,52,6702,443,220052,220181,225312) and valuenum > 0 and valuenum < 300 then valuenum else null end) as meanbp
      , (case when itemid in (615,618,220210,224690) and valuenum > 0 and valuenum < 70 then valuenum else null end) as resprate
      , (case when itemid in (223761,678) and valuenum > 70 and valuenum < 120 then (valuenum-32)/1.8 -- converted to degC in valuenum call
                when itemid in (223762,676) and valuenum > 10 and valuenum < 50  then valuenum else null end) as tempc
      , (case when itemid in (646,220277) and valuenum > 0 and valuenum <= 100 then valuenum else null end) as spo2
      , (case when itemid in (807,811,1529,3745,3744,225664,220621,226537) and valuenum > 0 then valuenum else null end) as glucose
    FROM `physionet-data.mimiciii_clinical.chartevents` ce
    -- exclude rows marked as error
    where (ce.error IS NULL OR ce.error != 1)
    and ce.icustay_id IS NOT NULL
    and ce.itemid in
    (
    -- HEART RATE
    211, --"Heart Rate"
    220045, --"Heart Rate"

    -- Systolic/diastolic

    51, --	Arterial BP [Systolic]
    442, --	Manual BP [Systolic]
    455, --	NBP [Systolic]
    6701, --	Arterial BP #2 [Systolic]
    220179, --	Non Invasive Blood Pressure systolic
    220050, --	Arterial Blood Pressure systolic

    8368, --	Arterial BP [Diastolic]
    8440, --	Manual BP [Diastolic]
    8441, --	NBP [Diastolic]
    8555, --	Arterial BP #2 [Diastolic]
    220180, --	Non Invasive Blood Pressure diastolic
    220051, --	Arterial Blood Pressure diastolic


    -- MEAN ARTERIAL PRESSURE
    456, --"NBP Mean"
    52, --"Arterial BP Mean"
    6702, --	Arterial BP Mean #2
    443, --	Manual BP Mean(calc)
    220052, --"Arterial Blood Pressure mean"
    220181, --"Non Invasive Blood Pressure mean"
    225312, --"ART BP mean"

    -- RESPIRATORY RATE
    618,--	Respiratory Rate
    615,--	Resp Rate (Total)
    220210,--	Respiratory Rate
    224690, --	Respiratory Rate (Total)


    -- spo2, peripheral
    646, 220277,

    -- glucose, both lab and fingerstick
    807,--	Fingerstick glucose
    811,--	glucose (70-105)
    1529,--	glucose
    3745,--	Bloodglucose
    3744,--	Blood glucose
    225664,--	glucose finger stick
    220621,--	glucose (serum)
    226537,--	glucose (whole blood)

    -- TEMPERATURE
    223762, -- "Temperature Celsius"
    676,	-- "Temperature C"
    223761, -- "Temperature Fahrenheit"
    678 --	"Temperature F"

    )
  )
  select 
    ce.hadm_id,  ce.icustay_id
    , ce.charttime
    , avg(heartrate) as heartrate
    , avg(sysbp) as sysbp
    , avg(diasbp) as diasbp
    , avg(meanbp) as meanbp
    , avg(resprate) as resprate
    , avg(tempc) as tempc
    , avg(spo2) as spo2
    , avg(glucose) as glucose
  from ce
  group by ce.hadm_id, ce.icustay_id, ce.charttime
  order by ce.hadm_id, ce.icustay_id, ce.charttime
  ;
  """
  vs_df = run_query(vs_query, project_id)
  if saved_path != None:
    vs_df.to_csv(os.path.join(saved_path, "pivot_vital.csv"))
  return vs_df


def fio2_sql2df(project_id, saved_path=None):
  """
  A modified version of pivoted_fio2.sql(https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/pivot/pivoted_fio2.sql)
  * add hadm_id to the return table 
  """
  query = """
  with pvt as
  ( -- begin query that extracts the data
    select le.hadm_id
    , le.charttime
    -- here we assign labels to ITEMIDs
    -- this also fuses together multiple ITEMIDs containing the same data
      -- add in some sanity checks on the values
      , ROUND(MAX(case
          when valuenum <= 0 then null
          -- ensure FiO2 is a valid number between 21-100
          -- mistakes are rare (<100 obs out of ~100,000)
          -- there are 862 obs of valuenum == 20 - some people round down!
          -- rather than risk imputing garbage data for FiO2, we simply NULL invalid values
          when itemid = 50816 and valuenum < 20 then null
          when itemid = 50816 and valuenum > 100 then null
      ELSE valuenum END), 2) AS valuenum
      FROM `physionet-data.mimiciii_clinical.labevents` le
      where le.ITEMID = 50816
      GROUP BY le.hadm_id, le.charttime
  )
  , stg_fio2 as
  (
    select hadm_id, charttime
      -- pre-process the FiO2s to ensure they are between 21-100%
      , ROUND(MAX(
          case
            when itemid = 223835
              then case
                when valuenum > 0 and valuenum <= 1
                  then valuenum * 100
                -- improperly input data - looks like O2 flow in litres
                when valuenum > 1 and valuenum < 21
                  then null
                when valuenum >= 21 and valuenum <= 100
                  then valuenum
                else null end -- unphysiological
          when itemid in (3420, 3422)
          -- all these values are well formatted
              then valuenum
          when itemid = 190 and valuenum > 0.20 and valuenum < 1
          -- well formatted but not in %
              then valuenum * 100
        else null end
      ), 2) as fio2_chartevents
    FROM `physionet-data.mimiciii_clinical.chartevents`
    where ITEMID in
    (
      3420 -- FiO2
    , 190 -- FiO2 set
    , 223835 -- Inspired O2 Fraction (FiO2)
    , 3422 -- FiO2 [measured]
    )
    and valuenum > 0 and valuenum < 100
    -- exclude rows marked as error
    AND (error IS NULL OR error != 1)
    group by hadm_id, charttime
  )
  select *
  from
  (
    SELECT hadm_id, charttime, valuenum AS fio2
    FROM pvt
    UNION ALL
    SELECT hadm_id, charttime, fio2_chartevents AS fio2
    FROM stg_fio2
  )
  ORDER BY hadm_id, charttime;
  """
  fio2_df = run_query(query, project_id)
  if saved_path is not None:
    fio2_df.to_csv(saved_path)
  return fio2_df


