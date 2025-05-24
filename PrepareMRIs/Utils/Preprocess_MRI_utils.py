import os
import subprocess


class PreprocessMRI:

    def __init__(self, img_path, out_dir):
        """
        Class used for preprocessing MRIs

        :param img_path: str, the path of the image to preprocess
        :param out_dir: str, the directory that will contain the processed MRIs and their previews
        """
        self.img_path = img_path
        self.preprocess_bash_file = './Utils/preprocess.sh'

        self.out_dir = os.path.join(out_dir, 'Registered')
        self.processed_dir = os.path.join(out_dir, 'Processed_preview')

        # In case the two directories do not exist, then make them in the out_dir directory
        for dir in [self.out_dir, self.processed_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def Windows2WSL_path(self, path):
        """
        Function for converting Windows path to Linux path
        :param path: str, the path to be converted
        :return: str, the converted path
        """
        return os.path.abspath(path).replace('C:', '/mnt/c').replace('\\', '/')

    def WSL2Windows_path(self, path):
        """
        Function for converting Linux path to  Windows path
        :param path: str, the path to be converted
        :return: str, the converted path
        """
        return os.path.abspath(path).decode().replace(r'\mnt\c', '').replace('/', '\\')

    def PreprocessFromBashFile(self):
        """
        Function for preprocessing the MRI given in the original path using the predefined bash file.

        The preprocessing steps are:
        1. Resample the MRI to 1mm x 1mm x 1mm spacing
        2. Reorient it to MNI space
        3. Crop lower neck slices to create a robust field of view
        4. N4 bias field correct it
        5. Spatial normalization to MNI152 using two step registration (Rigid + Affine)
        6. Skull-stripping it
        7. Extract a preview along all fields of view for checking the result's quality

        The function also outputs a log file with the results outputed from the above preprocessing steps
        """

        # Save the current image's path
        input_filepath = self.img_path

        # Convert the input image, the output directory, processed previews' folder and bash file's path to Linux paths
        input_filepath_WSL = self.Windows2WSL_path(os.path.abspath(input_filepath))
        out_dir_WSL = self.Windows2WSL_path(os.path.abspath(self.out_dir))
        processed_dir_WSL = self.Windows2WSL_path(os.path.abspath(self.processed_dir))
        bash_file = self.Windows2WSL_path(self.preprocess_bash_file)

        # Open cmd in WSL
        PIPE = subprocess.PIPE
        proc = subprocess.Popen('wsl ~', stdin=PIPE, stdout=PIPE, stderr=PIPE)

        # Run the bash file by passing:
        # 1. input_filepath_WSL : the converted original image's path
        # 2. out_dir_WSL        : the converted output directory's path
        # 3. processed_dir_WSL  : the converted processed previews' path
        cmd = f'bash {bash_file} {input_filepath_WSL} {out_dir_WSL} {processed_dir_WSL}'

        # Save the command prompts output and errors
        output, err = proc.communicate(cmd.encode())

        # Save the file name of the original name, which is basically the Subject's ID and the original MRI's ID
        basename = os.path.basename(input_filepath)

        # Set the log file's basename
        log_file = basename.split('.')[0] + '_log' + '.txt'

        # Set the log file's path
        log_file_path = os.path.join(self.out_dir, log_file)

        # Save the log file
        with open(log_file_path, 'w') as log:
            log.write(output.decode())
