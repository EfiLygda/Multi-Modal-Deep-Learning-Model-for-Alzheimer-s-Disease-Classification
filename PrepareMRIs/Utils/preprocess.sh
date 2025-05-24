#!/bin/bash

# ---------------------------------------------------------------------------
# --- Fetching the inputs to the bash file ---

# File path of the MRI
input_file=$1

# The output directory of the processed MRIs
out_dir=$2

# The output directory of the processed MRIs' previews
out_processed_dir=$3
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# --- Setting constants ---

# Setting the basename of the MRI (basically the Subject's ID and the MRI's ID)
basename_with_ext=$(basename "$input_file")
basename="${basename_with_ext%.nii.gz}"

# Setting the paths of the templates used - MNI152
MNI152="$FSLDIR/data/standard/MNI152_T1_1mm.nii.gz"
MNI152_BRAIN="$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz"
MNI152_BRAIN_MASK="$FSLDIR/data/standard/MNI152_T1_1mm_brain_mask_dil.nii.gz"
MNI152_BRAIN_MASK_DETAILED="$FSLDIR/data/standard/MNI152_T1_1mm_brain_mask.nii.gz"

# Setting the output shape of the preview
output_dim=256
width=$((output_dim*3))
height=$output_dim

# Basic line break used in printing results
line_break="------------------------------------------------------------"

# Setting the start time for measuring the time elapsed
SECONDS=0
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Function to time a command and print the time
time_command() {
    local start=$(date +%s.%N)
	local cmd_name=$3
	echo $line_break
	echo $2
	echo $line_break
    $1
	echo
    local end=$(date +%s.%N)
    local delta=$(echo "$end - $start" | bc)
    echo "Command ${cmd_name} took ${delta} seconds"
	echo $line_break
	echo
}
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# --- Start preprocessing ---

# Info of the original image
# fslinfo $input_file

# Resample to 1mm x 1mm x 1mm 
time_command "ResampleImageBySpacing 3 $input_file ${out_dir}/${basename}_resampled.nii.gz 1 1 1" \
			 "Resampling image to 1mm x 1mm x 1mm ..."\
			 "ResampleImageBySpacing"

# Reorient the MRI 
time_command "fslreorient2std ${out_dir}/${basename}_resampled.nii.gz ${out_dir}/${basename}_reorient.nii.gz" \
			 "Reorienting MRI image to standard..."\
			 "fslreorient2std"

# Cropping Lower Neck
time_command "robustfov -i ${out_dir}/${basename}_reorient.nii.gz -r ${out_dir}/${basename}_robust.nii.gz" \
			 "Cropping to remove neck and lower head..."\
			 "robustfov"

# N4 Bias Field Correction
time_command "N4BiasFieldCorrection -i ${out_dir}/${basename}_robust.nii.gz -o ${out_dir}/${basename}_n4.nii.gz" \
			 "N4 Bias Field Correcting MRI image..."\
			 "N4BiasFieldCorrection"

# Apply affine registration to MNI152
time_command "$ANTSPATH/antsRegistrationSyNQuick.sh -d 3 -n 7 \
									  -t a -r 256 \
									  -f $MNI152 \
									  -m ${out_dir}/${basename}_n4.nii.gz \
									  -o ${out_dir}/${basename}_registered_linear_" \
			 "Affine registration of MRI image to MNI152..."\
			 "antsRegistrationSyNQuick"
									  
# Extraction brain from MRI
time_command "antsBrainExtraction.sh -d 3 \
                       -a ${out_dir}/${basename}_registered_linear_Warped.nii.gz \
					   -e $MNI152 \
					   -m $MNI152_BRAIN_MASK_DETAILED \
					   -o ${out_dir}/${basename}_skull_stripped_linear_ \
					   -f $MNI152_BRAIN_MASK" \
			 "Skull stripping of MRI image..."\
			 "antsBrainExtraction"				   
					   
# Extract preview
time_command "fsleyes render --outfile ${out_processed_dir}/${basename}.png \
			   --size $width $height \
			   --scene ortho \
			   --hideLabels \
			   --hideCursor \
			   ${out_dir}/${basename}_skull_stripped_linear_BrainExtractionBrain.nii.gz" \
			 "Extracting orthogonal planes of MRI image..."\
			 "fsleyes render"

echo "Elapsed Time: $SECONDS seconds"
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# --- Removing temporary files ---

echo $line_break
echo "Removed temporary files..."
echo $line_break

# List of suffixes to remove
suffixes=(
  "_registered_linear_InverseWarped.nii.gz"
	"_registered_linear_1Warp.nii.gz"
	"_registered_linear_1InverseWarp.nii.gz"
  "_reorient.nii.gz"
  "_resampled.nii.gz"
	"_robust.nii.gz"
	"_n4.nii.gz"
	"_registered_linear_Warped.nii.gz"
	"_skull_stripped_linear_BrainExtractionMask.nii.gz"
	"_skull_stripped_linear_BrainExtractionPrior0GenericAffine.mat"
)

# Find and remove files with the current suffix
for suffix in "${suffixes[@]}"; do 
    rm "${out_dir}/${basename}${suffix}"
done
# ---------------------------------------------------------------------------
