import os
from Utils.MRI_utils import MRImage
from Utils.Preprocess_MRI_utils import PreprocessMRI

# Set the directory containing all the MRIs as nii.gz
mri_dir = '../Nifti_preprocessed'

# Set the directory for the outputed images
# Two new subdirectories will be created if they do not exist
out_dir = './'

# Get all the MRIs names
img_names = MRImage.get_all_nifti_files(mri_dir)

# Add their respective root directory to their fila path names
img_paths = [os.path.join(mri_dir, img) for img in img_names]

# Loop through the images for processing them
for i, img_path in enumerate(img_paths):

    # Print a progress message
    progress_msg = f'Processing image {i + 1}/{len(img_paths)}'
    print(progress_msg, end='\r')

    # Use the PreprocessMRI class for setting the image's path and its respective output directory
    pMRI = PreprocessMRI(img_path, out_dir)

    # Apply the processing
    pMRI.PreprocessFromBashFile()
