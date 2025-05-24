import os

import numpy as np
import pandas as pd

import Utils.Data_Cleaning_utils as DataCleaning

# ---------------------------------------------------------------------------
# List of the most commonly used tables from ADNI in:
# https://adni.loni.usc.edu/wp-content/uploads/2008/07/inst_commonly_used_table.pdf

# The tables used in this code:
# 1. Diagnostic Summary [ADNI1,GO,2,3]                   : DXSUM_PDXCONV_ADNIALL
# 2. Clinical Dementia Rating Scale (CDR) [ADNI1,GO,2,3] : CDR
# 3. Mini-Mental State Examination (MMSE) [ADNI1,GO,2,3] : MMSE
# 4. Neuropsychological Battery [ADNI1,GO,2,3]           : NEUROBAT
# 5. UW - Neuropsych Summary Scores [ADNI1,GO,2,3]       : UWNPSYCHSUM
# 6. UCSD - Derived Volumes [ADNI1]                      : UCSDVOL
# 7. UPENN - Plasma Biomarker Data [ADNI1]               : UPENNPLASMA
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# --- Directories of the datasets ---
ROOT_DIR = '../'
CSV_DIR = os.path.join(ROOT_DIR, 'csv')

# --- File names of the csv files to be merged ---

# CURRENT DATASET -> ADNI-1 COMPLETE 1 YEAR - 1.5T MRIs
current_data_csv = os.path.join(ROOT_DIR, 'ADNI_Standarized_Datasets_csv', 'ADNI1_Complete_1Yr_1.5T.csv')

# DIAGNOSTIC DATA
# The original dataset contains the groups in which the subjects were initially included after
# their screening visit, but some subjects change diagnosis between visits
diagnosis_csv = os.path.join(CSV_DIR, 'DXSUM_PDXCONV_ADNIALL_06Nov2023.csv')

# DEMOGRAPHIC DATA
demographics_csv = os.path.join(CSV_DIR, 'PTDEMOG_09Feb2024.csv')

# MINI-MENTAL STATE EXAM SCORES
mmse_csv = os.path.join(CSV_DIR, 'MMSE_09Feb2024.csv')

# CLINICAL DEMENTIA RATINGS
cdr_csv = os.path.join(CSV_DIR, 'CDR_09Feb2024.csv')

# LOGICAL MEMORY II SCORES
logical_memory_csv = os.path.join(CSV_DIR, 'NEUROBAT_10May2024.csv')

# CRANE LAB (UW) - NEUROPSYCHOLOGICAL SUMMARY SCORES
neuropsych_csv = os.path.join(CSV_DIR, 'UWNPSYCHSUM_21May2024.csv')

# ANDERS DALE LAB (UCSD) - DERIVED VOLUMES
mri_measures_csv = os.path.join(CSV_DIR, 'UCSDVOL_24May2024.csv')

# UPENN PLASMA BIOMARKER DATA
plasma_biomarkers_csv = os.path.join(CSV_DIR, 'UPENNPLASMA_25May2024.csv')
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# --- Defining the columns to keep from each of the datasets to be merged ---

# The column's definition are defined in https://adni.loni.usc.edu/data-dictionary-search/?q=COLUMN

# In each of the datasets there are 4 columns that are required to be kept, when they are available:
# 1. 'Phase'    : ADNI Cohort ID (ADNI1, ADNIGO, ADNI2, ADNI3)
# 2. 'RID'      : Participant roster ID
# 3. 'PTID'     : Participant ID
# 4. 'VISCODE2' : Translated visit code

# For each dataset the additional columns to be kept are defined below, along with their encoding

columns_to_keep = {
    'diagnosis': ['Phase', 'RID', 'PTID', 'VISCODE2',
                  'DXCURREN', 'DXCHANGE', 'DIAGNOSIS',
                  'DXCONV', 'DXCONTYP', 'DXCONFID'],

    'demographics': ['Phase', 'RID', 'PTID', 'VISCODE2',
                     'PTHAND', 'PTEDUCAT', 'PTRACCAT'],

    'mmse': ['Phase', 'RID', 'PTID', 'VISCODE2', 'MMSCORE'],

    'cdr': ['Phase', 'RID', 'PTID', 'VISCODE2', 'CDGLOBAL'],

    'logical_memory': ['Phase', 'PTID', 'RID', 'VISCODE2', 'LDELTOTAL'],

    'neuropsychological': ['Phase', 'RID', 'VISCODE2',
                           'ADNI_MEM', 'ADNI_EF', 'ADNI_LAN', 'ADNI_VS', 'ADNI_EF2'],

    'mri_measures': [
        'RID', 'VISCODE2',
        'BRAIN', 'EICV', 'VENTRICLES', 'LHIPPOC', 'RHIPPOC',
        'LINFLATVEN', 'RINFLATVEN', 'LMIDTEMP', 'RMIDTEMP', 'LINFTEMP',
        'RINFTEMP', 'LFUSIFORM', 'RFUSIFORM', 'LENTORHIN', 'RENTORHIN'],

    'plasma_biomarkers': ['RID', 'VISCODE', 'AB40', 'AB42'],

}
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # ------------------------------------------------------------------------
    # LOADING AND PREPARING DATASETS
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # --- Load current dataset ---
    current_data = pd.read_csv(current_data_csv)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # --- Preparing diagnostic data ---

    # Load the dataset
    diagnosis_df_og = pd.read_csv(diagnosis_csv)

    # Removing the unused columns
    diagnosis_df = diagnosis_df_og.loc[:, columns_to_keep['diagnosis']]

    # Replacing the null values with NaNs
    diagnosis_df.replace(-4, np.nan, inplace=True)

    # Add the labels from the diagnosis (originally encoded to integers)
    diagnosis_df = DataCleaning.code2label(diagnosis_df)

    # Filling NaN at baseline visit
    # Some of the exams were contacted at the screening visits and then usually within 10 days of
    # the baseline visit. So the data from the screening visit will be used also in the baseline visit.

    # Save the baselines visits
    baseline_visits = diagnosis_df.VISCODE2 == 'bl'

    # Temporarily change the baseline visit code to screening
    diagnosis_df.loc[baseline_visits, 'VISCODE2'] = 'sc'
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # --- Preparing demographic data ---

    # Load the dataset
    demographics_df = pd.read_csv(demographics_csv)

    # Removing the unused columns
    demographics_df = demographics_df.loc[:, columns_to_keep['demographics']]

    # Replacing the null values with NaNs
    demographics_df.replace(-4, np.nan, inplace=True)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # --- Preparing MMSE data ---

    # Load the dataset
    mmse_df = pd.read_csv(mmse_csv)

    # Removing the unused columns
    mmse_df = mmse_df.loc[:, columns_to_keep['mmse']]

    # Replacing the null values with NaNs
    mmse_df.replace(-1, np.nan, inplace=True)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # --- Preparing CDR data ---

    # Load the dataset
    cdr_df = pd.read_csv(cdr_csv)

    # Removing the unused columns
    cdr_df = cdr_df.loc[:, columns_to_keep['cdr']]

    # Replacing the null values with NaNs
    cdr_df.replace(-1, np.nan, inplace=True)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # --- Preparing Logical Memory II - Delayed Recall data ---

    # Load the dataset
    logical_memory_df = pd.read_csv(logical_memory_csv)

    # Removing the unused columns
    logical_memory_df = logical_memory_df.loc[:, columns_to_keep['logical_memory']]

    # Replacing the null values with NaNs
    logical_memory_df.replace(-1, np.nan, inplace=True)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # --- Preparing neuropsychological summary scores ---

    # Load the dataset
    neuropsych_df = pd.read_csv(neuropsych_csv)

    # Change the name of the visit's code column
    neuropsych_df = neuropsych_df.rename({'PHASE': 'Phase'}, axis=1)

    # Removing the unused columns
    neuropsych_df = neuropsych_df.loc[:, columns_to_keep['neuropsychological']]

    # Save the baselines visits
    baseline_visits_neuropsych = neuropsych_df.VISCODE2 == 'bl'

    # Temporarily change the baseline visit code to screening
    neuropsych_df.loc[baseline_visits_neuropsych, 'VISCODE2'] = 'sc'
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # --- Preparing MRI measures data ---

    # Load the dataset
    mri_measures_df = pd.read_csv(mri_measures_csv)

    # Change the name of the visit's code column
    mri_measures_df = mri_measures_df.rename({'VISCODE': 'VISCODE2'}, axis=1)

    # Removing the unused columns
    mri_measures_df = mri_measures_df.loc[:, columns_to_keep['mri_measures']]

    # Calculating z-scores for the variables
    for col in columns_to_keep['mri_measures'][2:]:
        current_column = mri_measures_df.loc[:, col]
        mri_measures_df.loc[:, col] = (current_column - current_column.mean()) / current_column.std()

    # Calculate the mean measures between the MRIs corresponding to the same patient in the same visit
    mri_measures_df = mri_measures_df.groupby(['RID', 'VISCODE2']).mean().reset_index()
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # --- Preparing plasma biomarkers' data ---

    # Load the dataset
    plasma_biomarkers_df = pd.read_csv(plasma_biomarkers_csv)

    # Removing the unused columns
    plasma_biomarkers_df = plasma_biomarkers_df.loc[:, columns_to_keep['plasma_biomarkers']]

    # Change the name of the visit's code column
    plasma_biomarkers_df = plasma_biomarkers_df.rename({'VISCODE': 'VISCODE2'}, axis=1)

    # Save the baselines visits
    baseline_visits_plasma_biomarkers = plasma_biomarkers_df.VISCODE2 == 'bl'

    # Temporarily change the baseline visit code to screening
    plasma_biomarkers_df.loc[baseline_visits_plasma_biomarkers, 'VISCODE2'] = 'sc'
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # MERGING THE DATASETS
    # ------------------------------------------------------------------------

    # Keys used for merging
    keys = ['Phase', 'RID', 'PTID', 'VISCODE2']

    # Merging the subjects' diagnostic data with their respective demographic data
    MERGED = diagnosis_df.merge(demographics_df, on=keys, how='left')

    # Continue merging
    MERGED = MERGED.merge(cdr_df, on=keys, how='left')
    MERGED = MERGED.merge(mmse_df, on=keys, how='left')
    MERGED = MERGED.merge(logical_memory_df, on=keys, how='left')

    # Changing the keys for merging with neuropsychological data, since they do not contain 'PTID' and merging them
    keys = ['RID', 'VISCODE2', 'Phase']
    MERGED = MERGED.merge(neuropsych_df, on=keys, how='left')

    # Changing the keys for merging with biomarkers' data, since they do not contain 'PTID' and 'Phase'
    keys = ['RID', 'VISCODE2']

    # For merging the MRI measures the previous keys are used, since they do not contain 'PTID' and 'Phase'
    MERGED = MERGED.merge(mri_measures_df, on=keys, how='left')

    # Continue merging
    MERGED = MERGED.merge(plasma_biomarkers_df, on=keys, how='left')
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # MERGING DATA TO THE CURRENT DATASET
    # ------------------------------------------------------------------------
    initial_data_additional = DataCleaning.add_info_to_current_data(current_data, MERGED)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # HANDLING MISSING VALUES
    # ------------------------------------------------------------------------
    # --- Handling missing demographic values ---

    # Columns of interest: 'PTHAND', 'PTEDUCAT', 'PTRACCAT'

    # COMMENT: Since demographic data were gathered during the screening visit, the values of the other visits'
    #          are filled using the available data, except age that changes between visits and is also available in
    #          the original dataset

    # Different demographic values between visits: NO

    # Filling the values
    for col in ['PTHAND', 'PTEDUCAT', 'PTRACCAT']:

        # Fetching the only non-missing value for every patient
        new_col = initial_data_additional.groupby('PTID')[col].transform(lambda x: x.dropna().values[0])

        # Replacing the original data with non-missing values
        initial_data_additional.loc[:, col] = new_col

    # Printing the number of missing values after filling
    # DataCleaning.print_missing_values_counts(df=initial_data_additional,
    #                                cols=['PTHAND', 'PTEDUCAT', 'PTRACCAT'])

    # Number of missing values after filling
    # PTHAND   : 1
    # PTEDUCAT : 1
    # PTRACCAT : 1
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # --- Handling missing neuropsychological values ---

    # Columns of interest: 'CDGLOBAL', 'MMSCORE', 'LDELTOTAL'

    # COMMENT: No imputation for these scores since not many values are missing, except LDELTOTAL that will
    #          not be used in the model

    # Printing the number of missing values after filling
    # DataCleaning.print_missing_values_counts(df=initial_data_additional,
    #                                cols=['CDGLOBAL', 'MMSCORE', 'LDELTOTAL'])

    # Number of missing values
    # CDGLOBAL  : 8
    # MMSCORE   : 3
    # LDELTOTAL : 760
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # --- Handling missing memory and executive function composite scores ---

    # Columns of interest: 'ADNI_MEM', 'ADNI_EF', 'ADNI_LAN', 'ADNI_VS', 'ADNI_EF2'

    # ADNI_EF2 is more precise according to:
    # https://ida.loni.usc.edu/download/files/study/42de06c6-7254-4462-9c69-7a46156bee55/file/
    # adni/ADNI_Methods_UWNPSYCHSUM_20231018.pdf

    # Printing number of missing values
    # DataCleaning.print_missing_values_counts(initial_data_additional,
    #                                          ['ADNI_MEM', 'ADNI_EF', 'ADNI_LAN',
    #                                           'ADNI_VS', 'ADNI_EF2'])

    # Number of missing values
    # ADNI_MEM : 3
    # ADNI_EF  : 2
    # ADNI_LAN : 2
    # ADNI_VS  : 2
    # ADNI_EF2 : 2

    # Calculating the skewness of the distributions by age (50s, 60s, ..., 90s) and group (CN, MCI, AD)
    #          the distributions are mostly skewed, so the value used for the imputation
    #          of the missing values is their respective median
    # initial_data_additional.groupby('label')[['ADNI_MEM', 'ADNI_EF',
    #                                           'ADNI_LAN', 'ADNI_VS',
    #                                           'ADNI_EF2']].agg(lambda x: x.skew())

    # COMMENT: The distributions are mostly skewed, so the value used for the imputation is their respective median
    initial_data_additional = DataCleaning.impute_median_of_groups(df=initial_data_additional,
                                                                   cols=['ADNI_MEM', 'ADNI_EF', 'ADNI_LAN',
                                                                                      'ADNI_VS', 'ADNI_EF2'])
    # Printing the number of missing values after filling
    # DataCleaning.print_missing_values_counts(initial_data_additional,
    #                                          ['ADNI_MEM', 'ADNI_EF', 'ADNI_LAN',
    #                                           'ADNI_VS', 'ADNI_EF2'])

    # Number of missing values after imputation
    # ADNI_MEM : 1
    # ADNI_EF  : 1
    # ADNI_LAN : 1
    # ADNI_VS  : 1
    # ADNI_EF2 : 1
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # --- Handling missing MRI measures ---

    # Columns of interest: 'VENTRICLES',
    #                      'LHIPPOC', 'RHIPPOC',
    #                      'LINFLATVEN', 'RINFLATVEN',
    #                      'LMIDTEMP', 'RMIDTEMP',
    #                      'LINFTEMP', 'RINFTEMP',
    #                      'LFUSIFORM', 'RFUSIFORM',
    #                      'LENTORHIN', 'RENTORHIN'

    # Printing number of missing values
    # DataCleaning.print_missing_values_counts(df=initial_data_additional,
    #                                          cols=columns_to_keep['mri_measures'][2:])

    # Number of missing values
    # BRAIN                  : 130
    # EICV                   : 130
    # VENTRICLES             : 130
    # LHIPPOC, RHIPPOC       : 130
    # LINFLATVEN, RINFLATVEN : 130
    # LMIDTEMP, RMIDTEMP     : 130
    # LINFTEMP, RINFTEMP     : 130
    # LFUSIFORM, RFUSIFORM   : 130
    # LENTORHIN, RENTORHIN   : 130

    # Calculating the skewness of the distributions by group (CN, MCI, AD)
    # The distributions are mostly skewed, so the value used for the imputation of the missing values
    # is their respective median
    # initial_data_additional.groupby('label')[['VENTRICLES',
    #                                           'LHIPPOC', 'RHIPPOC',
    #                                           'LINFLATVEN', 'RINFLATVEN',
    #                                           'LMIDTEMP', 'RMIDTEMP',
    #                                           'LINFTEMP', 'RINFTEMP',
    #                                           'LFUSIFORM', 'RFUSIFORM',
    #                                           'LENTORHIN', 'RENTORHIN']].agg(lambda x: x.skew())

    # COMMENT: The distributions are mostly skewed, so the value used for the imputation is their respective median
    initial_data_additional = DataCleaning.impute_median_of_groups(df=initial_data_additional,
                                                                   cols=['VENTRICLES',
                                                                         'LHIPPOC', 'RHIPPOC',
                                                                         'LINFLATVEN', 'RINFLATVEN',
                                                                         'LMIDTEMP', 'RMIDTEMP',
                                                                         'LINFTEMP', 'RINFTEMP',
                                                                         'LFUSIFORM', 'RFUSIFORM',
                                                                         'LENTORHIN', 'RENTORHIN'])

    # Printing number of missing values after filling
    # DataCleaning.print_missing_values_counts(df=initial_data_additional,
    #                                          cols=columns_to_keep['mri_measures'][2:])

    # Number of missing values after imputation
    # BRAIN                  : 2
    # EICV                   : 2
    # VENTRICLES             : 2
    # LHIPPOC, RHIPPOC       : 2
    # LINFLATVEN, RINFLATVEN : 2
    # LMIDTEMP, RMIDTEMP     : 2
    # LINFTEMP, RINFTEMP     : 2
    # LFUSIFORM, RFUSIFORM   : 2
    # LENTORHIN, RENTORHIN   : 2
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # --- Handling biomarkers measures ---

    # Columns of interest: 'AB40', 'AB42'

    # Printing number of missing values
    # DataCleaning.print_missing_values_counts(df=initial_data_additional, cols=['AB40', 'AB42'])

    # Number of missing values
    # AB40 : 807
    # AB42 : 803

    # COMMENT: Missing values are due to exams not being contacted in 6 month visits and also for some subjects
    #          not having bio samples taken

    # Imputing plasma m06 missing values using the mean between sc and m12 visits
    medians_by_patient = initial_data_additional.groupby('Subject')[['AB40', 'AB42']].mean()

    # Saving the subjects
    subjects = medians_by_patient.index

    # Imputing m06 visits
    for subject in subjects:
        subject_medians = medians_by_patient.loc[subject]
        mask = (initial_data_additional.Subject == subject) & (initial_data_additional.VISCODE2 == 'm06')

        for col in ['AB40', 'AB42']:
            initial_data_additional.loc[mask, col] = subject_medians[col]

    # Calculating the AB42/AB40 ratio
    initial_data_additional['AB4240_RATIO'] = initial_data_additional.AB42 / initial_data_additional.AB40

    # Printing number of missing values after filling
    # DataCleaning.print_missing_values_counts(df=initial_data_additional, cols=['AB40', 'AB42', 'AB4240_RATIO'])

    # Number of missing values after imputation
    # AB40         : 68
    # AB42         : 65
    # AB4240_RATIO : 72
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # CLEANING DATASET AND SAVING IT
    # ------------------------------------------------------------------------

    # Defining columns to be removed before saving
    columns_to_drop = ['Modality', 'Type',
                       'Acq Date', 'Format', 'Downloaded', 'Phase',
                       # 'RID',
                       'PTID', 'VISCODE2',
                       'DXCURREN', 'DXCHANGE', 'DIAGNOSIS',
                       'DXCONV',
                       'DXCONTYP',
                       ]

    # Remove diagnosis with Uncertain or Mildly Confident
    initial_data_additional = initial_data_additional[~initial_data_additional.DXCONFID.isin([0, 1])]

    # Dropping the columns
    initial_data_additional.drop(columns_to_drop, axis=1, inplace=True)

    # Saving the data to CSV
    initial_data_additional.to_csv(os.path.join(CSV_DIR, 'ORIGINAL_SUBSET_WITH_ADDITIONAL_INFO_TEST.csv'), index=False)
    # ------------------------------------------------------------------------
