import os
import random

import numpy as np  # numpy-1.19.5 for CUDA
import pandas as pd

from time import time
import json

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import matplotlib.pyplot as plt

import Utils.Dataset_utils as DataUtils
import Utils.Model_utils as ModelUtils
import Utils.Generator_utils as GeneratorUtils
import Utils.Plotting_utils as PlotUtils
import Model_config as MC

# ---------------------------------------------------------------------------
# --- Setting the seed for reproducible results (https://stackoverflow.com/a/56606207) ---
SEED = 0

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
tf.compat.v1.disable_control_flow_v2()

# Saving the starting time
start_time = time()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The current labels used
labels = MC.CLASS_INDICES.keys()

# Early stopping to avoid over-training
early_stopping = EarlyStopping(monitor='val_loss',  # using val_loss due to imbalanced dataset
                               mode='min',
                               patience=MC.PATIENCE,
                               restore_best_weights=True)

# Setting thw images' subsets' generator
train_image_generator = ImageDataGenerator(**MC.TRAINING_CONFIG)
val_image_generator = ImageDataGenerator(**MC.VAL_CONFIG)
test_image_generator = ImageDataGenerator(**MC.TEST_CONFIG)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Directory for saving the results
result_dir = os.path.join('./', 'Results')

# Subdirectories of the results directory
fold_dir = os.path.join(result_dir, 'Fold')
training_metrics_dir = os.path.join(result_dir, 'Training Metrics')
training_progress_dir = os.path.join(result_dir, 'Training Progress')
confusion_matrix_dir = os.path.join(result_dir, 'Confusion Matrix')
classification_report_dir = os.path.join(result_dir, 'Classification Report')

dirs = [
    fold_dir,
    training_metrics_dir,
    training_progress_dir,
    confusion_matrix_dir,
    classification_report_dir
]

# Making the directories in case they do not exist
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

# Dataframe for saving the metrics in every model
training_metrics_columns = [
    'Max_epoch',
    'Max_train_accuracy',
    'Max_val_accuracy'
] + [f'Acc_{label}' for label in labels] + ['Acc']

training_metrics = pd.DataFrame(columns=training_metrics_columns)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Prepare results' figures
training_figure, training_axis = PlotUtils.prepare_training_figure()
CM_figure, CM_axis = PlotUtils.prepare_ConfMatrices_figure()
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # --- Setting the directories and the class encodings of the dataset ---
    labeled_data_dir = MC.PATHS['labeled_image_dir']
    metadata_dir = MC.PATHS['metadata_dir']

    # --- Import datasets ---

    # Import labeled images dataset
    images = DataUtils.MakeImagesDataframe(labeled_data_dir, labels)
    images['label'] = images['label'].map(MC.CLASS_INDICES)
    images['image_id'] = images.image_path.apply(DataUtils.get_Image_ID)

    # Import labeled images metadata dataset
    metadata_csv = os.path.join(metadata_dir, 'ORIGINAL_SUBSET_WITH_ADDITIONAL_INFO_TEST.csv')
    metadata_df = pd.read_csv(metadata_csv)

    # Remove data with different labels from the current ones
    metadata_df = metadata_df[metadata_df.label.isin(labels)]

    # Prepare the dataset
    metadata_df = DataUtils.PrepareMetadata(metadata_df, MC.CLASS_INDICES)

    # --- Reindexing the metadata df according to the images df ---
    # The rows of the two datasets will have to be synced at the MRI level, else their generators will not
    # provide the metadata for the respective MRI

    # Remove MRIs from images not in metadata_df
    current_MRI_ids = metadata_df['Image Data ID'].unique()
    images = images[images['image_id'].isin(current_MRI_ids)]

    # Setting the image_id (MRI ID) as the index of the datasets
    images.set_index('image_id', inplace=True)
    metadata_df.set_index('Image Data ID', inplace=True)

    # Reordering the rows of metadata data according to the index of the images and keeping relevant columns
    features = ['Age',

                'ADNI_MEM',
                'ADNI_LAN',
                'ADNI_EF2',

                'AB4240_RATIO',

                'VENTRICLES',
                'LHIPPOC', 'RHIPPOC',
                'LINFLATVEN', 'RINFLATVEN',
                'LMIDTEMP', 'RMIDTEMP',
                'LINFTEMP', 'RINFTEMP',
                'LFUSIFORM', 'RFUSIFORM',
                'LENTORHIN', 'RENTORHIN',

                'MMSCORE', 'CDGLOBAL',

                'label']

    metadata_df = metadata_df.reindex(images.index).loc[:, features]

    # -- Removing rows (MRIs) with any nan metadata (only one MRI will be removed) ---

    # Mask with the rows where any of the column has nan values
    any_nan_mask = metadata_df.isna().any(axis=1)

    # Saving image_ids with any nan metadata by keeping the respective index
    mri_scan_with_nan = metadata_df[any_nan_mask].index

    # Removing the MRIs with any nan metadata from the metadata df
    metadata_df = metadata_df[~any_nan_mask]

    # Removing the MRIs with any nan metadata from the images df
    for patient_id in mri_scan_with_nan:
        images.drop(patient_id, inplace=True)

    # Resetting the index
    images.reset_index(inplace=True)
    metadata_df.reset_index(inplace=True)
    metadata_df.to_csv('test.csv', index=False)

    # --- Splitting the datasets at the patient level ---

    # Split the datasets
    if MC.DATA_LEAKAGE:
        train_dfs, test_dfs = DataUtils.DataLeakageCV(images)
    else:
        train_dfs, test_dfs = DataUtils.PatientSplitCV(images, test_size=0.2)  # 0.2 == 1/5 for 5-fold CV

    # Begin cross validation loop
    for fold, (train_df_og, test_df) in enumerate(zip(train_dfs, test_dfs)):

        if MC.DATA_LEAKAGE:
            train_df, val_df = train_test_split(train_df_og, test_size=0.1)
        else:
            train_df, val_df = DataUtils.PatientSplit(train_df_og, test_size=0.1)

        # Adding the images' id for finding the images in each dataset
        train_df['image_id'] = train_df.image_path.apply(DataUtils.get_Image_ID)
        val_df['image_id'] = val_df.image_path.apply(DataUtils.get_Image_ID)
        test_df['image_id'] = test_df.image_path.apply(DataUtils.get_Image_ID)

        # Splitting the metadata df using the images' id as found above
        metadata_train_df = metadata_df.iloc[train_df.index, :]
        metadata_val_df = metadata_df.iloc[val_df.index, :]
        metadata_test_df = metadata_df.iloc[test_df.index, :]

        # --- Defining the generators for the train, validation and test dataset ---
        combined_gen_train = GeneratorUtils.FlowFromDataframes(
            train_image_generator,
            train_df, metadata_train_df,
            MC.BATCH_SIZE,
            MC.INPUT_IMAGE_WIDTH,
            MC.INPUT_IMAGE_HEIGHT,
            mode='train'
        )

        combined_gen_val = GeneratorUtils.FlowFromDataframes(
            val_image_generator,
            val_df, metadata_val_df,
            MC.BATCH_SIZE,
            MC.INPUT_IMAGE_WIDTH,
            MC.INPUT_IMAGE_HEIGHT,
            mode='val'
        )

        combined_gen_test = GeneratorUtils.FlowFromDataframes(
            test_image_generator,
            test_df, metadata_test_df,
            1,
            MC.INPUT_IMAGE_WIDTH,
            MC.INPUT_IMAGE_HEIGHT,
            mode='test'
        )

        # --- Configuring the multi input model ---

        # Setting the optimizer
        optimizer = MC.OPTIMIZER

        # In case a learning scheduler is used, resetting the learning rate to the original one
        if fold > 0:
            # Get configuration of the optimizer
            optimizer_config = optimizer.get_config()

            # Resetting the learning rate to the original one
            optimizer_config['learning_rate'] = MC.LEARNING_RATE

            # Apply the change to the optimizer
            optimizer = optimizer.from_config(optimizer_config)

        # Compiling the model
        model = ModelUtils.Multi_Input_Model(
            optimizer=optimizer,
            width=MC.INPUT_IMAGE_WIDTH,
            height=MC.INPUT_IMAGE_HEIGHT,
            pretrained_model=MC.BASE_MODEL,
            training_basemodel=MC.TRAINING_BASEMODEL,
            n_features=len(features) - 1,  # label is part of the features
            num_classes=MC.NUM_CLASSES,
        )

        # Create the LearningRateScheduler callback
        lr_scheduler = LearningRateScheduler(ModelUtils.learning_rate_schedule)

        # Defining the steps per epoch for training and validation
        steps_per_epoch = int(np.ceil(len(train_df) / MC.BATCH_SIZE))
        val_steps = int(np.ceil(len(val_df) / MC.BATCH_SIZE))

        # --- Fitting the model ---
        with tf.device('/GPU:0'):
            history = model.fit(
                x=combined_gen_train,
                validation_data=combined_gen_val,
                epochs=MC.EPOCHS,
                steps_per_epoch=steps_per_epoch,
                validation_steps=val_steps,
                callbacks=[early_stopping, lr_scheduler],
            )

        # Saving the training history
        history_df = pd.DataFrame(history.history)

        # Save the training progress
        training_progress_file_name = f'{MC.MODEL}__{MC.BASE_MODEL}__fold_{fold+1}.csv'
        history_df.to_csv(os.path.join(fold_dir, training_progress_file_name))

        # --- Presenting training progress ---

        # Loss and accuracy plots
        history_df.loss.plot(
            ax=training_axis[fold][0],
            color=PlotUtils.colors_dict['loss']
        )
        history_df.accuracy.plot(
            ax=training_axis[fold][1],
            color=PlotUtils.colors_dict['acc']
        )
        history_df.val_loss.plot(
            ax=training_axis[fold][0],
            color=PlotUtils.colors_dict['val_loss'],
            linestyle='dashed'
        )
        history_df.val_accuracy.plot(
            ax=training_axis[fold][1],
            color=PlotUtils.colors_dict['val_acc'],
            linestyle='dashed'
        )

        # Adding a grid to the plots
        for ax in training_axis[fold]:
            ax.grid(linestyle='--')

        # --- Presenting predictive results ---

        # Defining the steps for prediction
        test_steps = len(test_df)

        # Predictions using the test images
        predict = model.predict(
            combined_gen_test,
            steps=test_steps,
            verbose=1
        )

        # Finding the predicted class according to the number of classes used
        if MC.NUM_CLASSES == 3:
            predictions = np.argmax(predict, axis=1)
        else:
            predictions = list(map(lambda possibility: 1 if possibility > 0.5 else 0, predict))

        # Confusion matrix from predictions
        conf_matrix = confusion_matrix(metadata_test_df.label, predictions)

        # Add the confusion matrix to the current axis
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                      display_labels=MC.CLASS_INDICES)
        disp.plot(ax=CM_axis[fold])

        # Report of the classification
        report = classification_report(metadata_test_df.label, predictions,
                                       target_names=MC.CLASS_INDICES,
                                       output_dict=True,
                                       digits=4,
                                       zero_division=0)

        print(100*'-')
        print(f"Fold {fold + 1} accuracy: {report['accuracy'] * 100:.4f}")
        print(100*'-')

        # Saving the report as a JSON  file
        with open(os.path.join(classification_report_dir, f'{MC.MODEL}__{MC.BASE_MODEL}__fold_{fold+1}__report.json'), 'w') as file:
            json.dump(report, file, indent=4)

        # Save the training metrics to the training_metrics dataset
        training_metrics.loc[fold, :] = DataUtils.make_metrics_row(
            labels=labels,
            max_epoch=MC.EPOCHS,
            patience=MC.PATIENCE,
            history_df=history_df,
            report=report
        )

    # Saving the figures presenting the training process and its results
    training_figure.savefig(os.path.join(training_progress_dir, f'{MC.MODEL}__{MC.BASE_MODEL}__training.png'),
                            facecolor='white', bbox_inches='tight', dpi=200)
    CM_figure.savefig(os.path.join(confusion_matrix_dir, f'{MC.MODEL}__{MC.BASE_MODEL}__ConfMat.png'),
                      facecolor='white', bbox_inches='tight', dpi=200)

    # Print and save the final CV results
    CV_results = DataUtils.print_cv_results(training_metrics)
    training_metrics.to_csv(os.path.join(training_metrics_dir, f'{MC.MODEL}__{MC.BASE_MODEL}__training_metrics.csv'))

    # Print elapsed time
    print(100*'-')
    print(f"Elapsed time: {(time() - start_time)/60:.2f} minutes")
    print(100*'-')
