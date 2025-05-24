import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer

idx = pd.IndexSlice  # MultiIndex slicer


# ---------------------------------------------------------------------------
def print_missing_values_counts(df, cols):
    """
    Function for printing the missing values from the chosen columns of a dataset
    :param df: pandas.Dataframe, the current dataset
    :param cols: list, the list of the columns to print their respective missing values
    """
    for col in cols:
        n_nan = len(df[df.loc[:, col].isna()])
        print(f'Column {col} has {n_nan} missing values!')
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def code2label(df):
    """
    Function for decoding diagnostic labels according to the phase of the data.

    :param df: The dataset with the encoded diagnostic data
    :return: The dataset with the decoded diagnostic data
    """

    df['label'] = np.nan

    for index, row in df.iterrows():

        # Current phase of sample
        phase = row['Phase']

        # Column names with the encoded diagnosis by every phase of ADNI
        column_codes = {'ADNI1': 'DXCURREN',
                        'ADNIGO': 'DXCHANGE',
                        'ADNI2': 'DXCHANGE',
                        'ADNI3': 'DIAGNOSIS'}

        # The current encoded diagnosis
        code = row[column_codes[phase]]

        # Check if diagnosis is nan (the current sample has a diagnosis for a different phase -> different column)
        if np.isnan(code): continue

        # Mappings by phase
        if phase == 'ADNI1' or phase == 'ADNI3':
            mapping = {1: 'CN', 2: 'MCI', 3: 'AD'}

        elif phase == 'ADNIGO' or phase == 'ADNI2':
            mapping = {1: 'CN', 2: 'MCI', 3: 'AD',
                       4: 'MCI', 5: 'AD', 6: 'AD',
                       7: 'CN', 8: 'MCI', 9: 'CN'}

        # Decode the diagnosis
        # df.loc[index, column_codes[phase]] = mapping[code]
        df.loc[index, 'label'] = mapping[code]

    return df
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def add_info_to_current_data(df, diagnostic):
    """
    Function for adding the additional metadata to the MRI dataset used

    :param df: pandas.Dataframe, the MRI dataset used
    :param diagnostic: pandas.Dataframe, the dataset with additional metadata
    :return: pandas.Dataframe, the merged datasets
    """

    # Defining the keys used for merging between the datasets
    df_keys = ['Subject', 'Visit']
    diagnostic_keys = ['PTID', 'VISCODE2']

    # Merging the datasets
    result = df.merge(diagnostic, left_on=df_keys, right_on=diagnostic_keys, how='left')

    return result
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def impute_median_of_groups(df, cols):
    """
    Function for imputing missing values using the median of the diagnostic groups grouped by the sex, age
    and the diagnosis of the subjects

    :param df: pandas.Dataframe, the current dataset used for imputing
    :param cols: list, the columns for which imputing must be done
    :return: pandas.Dataframe, the dataset after imputing
    """

    # Defining the age groups for the imputing, according to the decade of the age
    age_labels = {'50s': 1, '60s': 2, '70s': 3, '80s': 4, '90s': 5}
    bins = [50, 60, 70, 80, 90, 100]

    # Creating the age groups
    df['Age_bin'] = pd.cut(df['Age'],
                           bins=bins,
                           labels=age_labels.values())

    # Calculating the median by sex, age group and diagnosis
    medians_by_age_in_class = df.groupby(['Sex', 'Age_bin', 'label'])[cols].median()

    # Imputing the values for each diagnosis, variable, sex and age group
    for label in ['CN', 'AD', 'MCI']:
        for col in cols:
            for sex in ['M', 'F']:
                for age_bin in age_labels.values():

                    # Fetching current median value used for imputing
                    current_median = medians_by_age_in_class.loc[idx[sex, age_bin, label], col]

                    # Saving mask for rows corresponding to current sex, age group and diagnosis values
                    mask = (df.Sex == sex) & (df.Age_bin == age_bin) & (df.label == label)

                    # Checking if the above combination returned ant rows (some combinations are not possible)
                    if df.loc[mask, col].any():

                        # Defining the imputer using a constant value -> the current median
                        imputer = SimpleImputer(strategy='constant', fill_value=current_median)

                        # Applying the imputer
                        df.loc[mask, col] = imputer.fit_transform(df.loc[mask, col].to_frame())

    return df
# ---------------------------------------------------------------------------
