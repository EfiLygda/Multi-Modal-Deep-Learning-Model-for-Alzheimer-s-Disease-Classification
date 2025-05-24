import os
from Utils.MRI_utils import MRImage
from Utils.Process_Slice_utils import Process2DSlice

# Set the directory containing all the MRIs as nii.gz
processed_mri_dir = './Registered'

# Set the directory for the outputed images
# Two new subdirectories will be created if they do not exist
out_dir = './'

# Get all the MRIs names
img_names = MRImage.get_all_nifti_files(processed_mri_dir)

# Filter the images
filtering = lambda path: True if '_skull_stripped_linear_BrainExtractionBrain.nii' in path else False
img_names = list(filter(filtering, img_names))

# Add their respective root directory to their fila path names
img_paths = [os.path.join(processed_mri_dir, img) for img in img_names]

# Configure the processing of the 2D slices
config = {
    'plane': 'all',
    'rotate': 90,
    'n_slices': 0
}

# Loop through the images for processing them
for i, img_path in enumerate(img_paths):
    
    # Print a progress message
    progress_msg = f'Extracting from image {i + 1}/{len(img_paths)}'
    print(progress_msg, end='\r')

    # Get the current image's data
    image_data = MRImage.get_img_data(img_path)

    # Save the base name of the image
    img_name = os.path.basename(img_path).replace(".nii.gz", '')

    # Fetch the orthogonal slice
    index = Process2DSlice.GetOrthoSlice(image_data, config['plane'])

    # Set the output dir via Process2DSlice
    mri = Process2DSlice(out_dir)

    # Export the slice
    mri.ExportSlicesByPlane(
        img_array=image_data,
        output_name=img_name.replace("_skull_stripped_linear_BrainExtractionBrain", ""),
        plane=config['plane'],
        rotate=config['rotate'],
        index=index,
        n_slices=config['n_slices'],
    )
