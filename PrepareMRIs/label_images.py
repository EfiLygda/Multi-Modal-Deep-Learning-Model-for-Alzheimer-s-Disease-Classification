import os
import shutil
import pandas as pd
from tqdm import tqdm

# Set the directory containing all the MRIs as nii.gz
data_dir = r'./'

# Set the directory for the outputed images
# Two new subdirectories will be created if they do not exist
out_dir = r'./ADNI_labeled'

# Set the current labels
labels = ['CN', 'MCI', 'AD']

# Read csv file with original labels
ADNI_df = pd.read_csv('../csv/ORIGINAL_SUBSET_WITH_ADDITIONAL_INFO_TEST.csv')

# List of planes of view for MRIs
planes = ['sagittal', 'axial', 'coronal']

# Dirs with pngs for every plane of the MRIs
png_plane_dirs = [os.path.join(data_dir, f'Processed_{plane}') for plane in planes]

# New dirs after labeling the planes
new_dataset_dirs = [os.path.join(out_dir, f'ADNI_labeled_{plane}') for plane in planes]

# Making directory for every plane
for current_plane, png_dir, new_dir in zip(planes, png_plane_dirs, new_dataset_dirs):

    # Making directory for every label
    for label in labels:

        # Dir for the current plane and label
        current_label_images_path = os.path.join(new_dir, label)

        # Making the dir if it does not exist
        if not os.path.exists(current_label_images_path):
            os.makedirs(current_label_images_path)

    # Png file names for the current plane
    png_files = [file for file in os.listdir(png_dir) if file.endswith('png')]

    # Labeling the files
    for file_name in tqdm(png_files, desc=f'Labeling MRI slices for {current_plane} plane'):

        # Find the current patient id and image id
        if file_name.count('__') == 1:
            current_patient_id, current_image_id = file_name.split("__")
            current_image_id = current_image_id.replace('.png', '')

        elif file_name.count('__') == 2:
            current_patient_id, current_image_id = file_name.split("__")[:-1]

        # Continue if image id not in the available data
        if current_image_id not in ADNI_df['Image Data ID'].values:
            continue

        # Find the current label
        mask = (ADNI_df['Subject'] == current_patient_id) & (ADNI_df['Image Data ID'] == current_image_id)

        # Only one value is returned so the first among the uniques is the label
        current_label = ADNI_df.label[mask].unique()[0]

        if pd.isna(current_label):
            current_label = ADNI_df.Group[mask].unique()[0]

        # Save the labeled image
        # Define the source and destination paths
        src_path_file = os.path.join(png_dir, file_name)
        dst_path_file = os.path.join(new_dir, current_label, file_name)

        # Copy the image file from source to destination
        shutil.copy(src_path_file, dst_path_file)