import os
import glob

import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import KFold


# ---------------------------------------------------------------------------
def MakeImagesDataframe(labeled_data_dir, labels):
    """
    Function for producing the labeled images' dataset (columns: image_path, label).

    :param labeled_data_dir: str, the directory of the subfolders of the labeled dataset
    :param labels: iterable, the labels of the images

    :return: pandas.Dataframe, the dataframe
    """
    # Dataframe with image paths and labels
    images = pd.DataFrame(columns=["image_path", "label"])

    for label in labels:
        # Directory of the images with the current label
        directory = os.path.join(labeled_data_dir, label)

        # List with the paths of all images in the current directory
        images_path = glob.glob(os.path.join(directory, '*.png'))

        # Dataframe with only the paths of the images
        section = pd.DataFrame(data=images_path, columns=["image_path"])

        # Adding the label of the images
        section['label'] = label

        # Add the dataframe with the images of the current label to the final dataframe
        images = pd.concat([images, section], axis=0)

    # Resetting the index of the dataframe
    images.reset_index(inplace=True, drop=True)

    return images
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def PatientSplit(df, test_size=0.2, shuffle_sets=True, split_index=None):
    """
    Function for splitting the dataset by grouping patients to either test or train subset of the data.

    :param df: The original dataset
    :param test_size: The size of the test subset
    :param shuffle_sets: Whether to shuffle the resulting subsets
    :param split_index: Where to split the dataset (only if test_size is not given)

    :return: tuple with the train and test subset of the dataset
    """

    # Calculate the split index for the test set, if it is not given
    if not split_index:
        split_index = int(np.ceil(test_size * len(df)))

    # Add patient_id column if it's not available
    if 'patient_id' not in df.columns:
        df['patient_id'] = df.image_path.apply(get_Subject_ID)

    # Find the number of images belonging to each patient
    patient_freq = df.groupby('patient_id').label.count().to_frame('patient_counts')

    # Shuffle the patients
    patient_freq = shuffle(patient_freq)

    # Calculate the cumulative sum of the frequencies of the images by each patient
    patient_freq['cumsum'] = patient_freq.cumsum()

    # Calculate the distance of the initial split index from the possible indices that the dataset can be split by
    patient_freq['dist_from_split_index'] = (patient_freq['cumsum'] - split_index).abs()

    # Find the patients by whom the dataset will be split by
    patient_to_split_by = patient_freq['dist_from_split_index'].idxmin()

    # Find the index of the patient by whom the dataset will be split by
    stop = list(patient_freq.index).index(patient_to_split_by) + 1

    # Get the patients in each set
    test_patients = list(patient_freq.index)[:stop]
    train_patients = list(patient_freq.index)[stop:]

    # Split the dataset
    test_df = df[df.patient_id.isin(test_patients)].loc[:, ['image_path', 'label']]
    train_df = df[df.patient_id.isin(train_patients)].loc[:, ['image_path', 'label']]

    # Shuffle the two sets
    if shuffle_sets:
        test_df = shuffle(test_df)
        train_df = shuffle(train_df)

    return train_df, test_df
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def PatientSplitCV(df, test_size=0.2, n_splits=5, shuffle_sets=True):
    """
    Function for k-fold cross validation splitting the dataset

    :param df: pandas.Daraframe, the labeled images dataset
    :param test_size: float, the size of the test set
    :param n_splits: int, the number of folds
    :param shuffle_sets: bool, whether to shuffle the two sets

    :return: tuple of two lists, the first list contains the trainings subsets and the other their respective test sets
    """

    # --- Adding the patient_id column, if it not already available ---
    if 'patient_id' not in df.columns:
        df['patient_id'] = df.image_path.apply(get_Subject_ID)

    # --- Defining the two subsets lists as empty ---
    train_dfs = []
    test_dfs = []

    # --- Defining list with the patients, already used in a testing subset, as empty at first ---
    test_patients = []

    # --- Calculate the split index for the test set ---
    split_index = int(np.ceil(test_size * len(df)))

    # --- Copying the original dataset ---
    dataset = df.copy()

    # --- Splitting in n_splits test sets the dataset ---
    for i in range(n_splits):

        # In case it is not the last split use the function PatientSplit by passing the split_index
        # and not the test_size.
        if i != n_splits - 1:
            _, test_df = PatientSplit(dataset, split_index=split_index, shuffle_sets=shuffle_sets)

        # In case it is the last split, get the remaining patients' data for the last test set
        else:
            test_df = df[~df.patient_id.isin(test_patients)]

        # Adding the patient_id column, if it not already available (Warning otherwise)
        if 'patient_id' not in test_df.columns:
            test_df['patient_id'] = test_df.image_path.apply(get_Subject_ID)

        # Finding the patients in the current test set
        current_test_patients = test_df['patient_id'].unique()

        # Append these patients to the list containing the used patients' ids
        test_patients.extend(current_test_patients)

        # Remove from the dataset the patients already used in a test set
        dataset = df[~df.patient_id.isin(test_patients)]

        # Fetching the current training set by removing the current test set's patients
        train_df = df[~df.patient_id.isin(current_test_patients)]

        # Removing the patient_id columns
        train_df.drop('patient_id', inplace=True, axis=1)
        test_df.drop('patient_id', inplace=True, axis=1)

        # Appending the sets and shuffling them, if needed
        if shuffle_sets:
            train_dfs.append(shuffle(train_df))
            test_dfs.append(shuffle(test_df))
        else:
            train_dfs.append(train_df)
            test_dfs.append(test_df)

    return train_dfs, test_dfs
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def DataLeakageCV(df, n_splits=5, shuffle_sets=True):
    """
    Function for randomly train-test splitting the dataset, not considering the possibility of data leakage

    :param df: pd.Dataframe, the data frame used for train-test splitting
    :param n_splits: int, the number of folds for CV
    :param shuffle_sets: bool, whether to shuffle the sets when using KFold from scikit
    :return: (list, list), containing the training and test subsets for each fold
    """
    # Set cross validation parameters
    kf = KFold(n_splits=n_splits, shuffle=shuffle_sets)

    # Define the lists for the training and test subsets
    train_dfs, test_dfs = [], []

    for train_index, test_index in kf.split(df):

        # Fetching and test images subsets and their respective metadata
        images_test = df.iloc[test_index, :]
        images_train = df.iloc[train_index, :]

        # Appending the subsets to their respective list
        train_dfs.append(images_train)
        test_dfs.append(images_test)

    return train_dfs, test_dfs
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
"""
Functions for getting the ids of the subject, image but also the index of the slice from the initial MRI
"""
get_Subject_ID = lambda row: os.path.basename(row).split('__')[0]
get_Image_ID = lambda row: os.path.basename(row).split('__')[1]
get_slice_index = lambda row: os.path.basename(row).split('__')[-1].replace('.png', '')
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def PrepareMetadata(metadata_df, class_indices):
    """
    Function for preparing the metadata data frame (mainly encoding categorical features).

    :param metadata_df: pandas.Dataframe, the data frame
    :return: pandas.Dataframe, the prepared dataframe
    """

    # Defining bins for years of education
    educ_labels = {'Primary': 1, 'Secondary': 2, 'Tertiary': 3}
    bins = [0, 7, 12, 20]  # Primary (0-8), Secondary (9-12), Tertiary (13+)

    # Encoding years of education
    metadata_df['PTEDUCAT'] = pd.cut(metadata_df['PTEDUCAT'], bins=bins, labels=educ_labels.values())

    # Encoding sex of subjects
    metadata_df['Sex'] = metadata_df['Sex'].map(lambda row: 1 if row == 'M' else 0)

    # Encoding labels as needed
    metadata_df['label'] = metadata_df.label.map(class_indices)

    return metadata_df
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def make_metrics_row(labels, max_epoch, patience, history_df, report):
    """
    Function for making a row for the training metrics

    :param labels: list, the current labels of the model
    :param max_epoch: int, the maximum epoch for the training of the model
    :param patience: int, the patience used to avoid over fitting
    :param history_df: pd.Dataframe, the data set containing the history of the training
    :param report: dict, the dictionary with the classification results
    :return: numpy.array, containing the current model's classification metrics
    """
    # Saving the epoch where the training was stopped
    if len(history_df) < max_epoch:
        stopped_epoch = len(history_df) - patience
    else:
        stopped_epoch = max_epoch - 1

    row = [
        stopped_epoch + 1,
        history_df.loc[stopped_epoch, 'accuracy'],
        history_df.loc[stopped_epoch, 'val_accuracy']
    ] + [report[label]['precision'] for label in labels] + [report['accuracy']]

    return np.array(row)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def print_cv_results(training_metrics_df):
    """
    Function for printing the final results of the cross validation

    :param training_metrics_df: pd.Dataframe, the dataframe containing the classification metrics from the CV models
    :return: pd.Series, containing the CV mean metrics
    """
    # Calculating the mean of every metric
    mean_metrics = training_metrics_df.mean()

    # Printing the results
    print(100 * '-')
    print("Cross Validation results:")
    print(f"Mean maximum epoch:         {int(mean_metrics.loc['Max_epoch'])}")
    print(f"Mean training accuracy:     {mean_metrics.loc['Max_train_accuracy'] * 100:.2f} %")
    print(f"Mean validation accuracy:   {mean_metrics.loc['Max_val_accuracy'] * 100:.2f} %")

    for col in mean_metrics.index:
        if 'Acc_' in col:
            print(f"Mean {col.replace('Acc_', '')} precision:          {mean_metrics.loc[col] * 100:.2f} %")

    print(f"Mean accuracy:              {mean_metrics.loc['Acc'] * 100:.2f} %")

    return mean_metrics
# ---------------------------------------------------------------------------
