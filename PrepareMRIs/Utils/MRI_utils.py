import os
import nibabel as nib
import re


class MRImage:
    """
    Class for fetching the basic data for the image.
    """

    @staticmethod
    def get_img_data(img_path):
        """
        Method for getting the image data as np.array
        """
        return nib.load(img_path).get_fdata()

    @staticmethod
    def get_subject_id_img_id(img_path):
        """
        Method for getting the patient id and image id form the path of the image file
        """
        # Pattern for patient id: ddd_S_dddd
        patient_id_pattern = r"\d{3}_S_\d{4}"
        patient_id = re.findall(patient_id_pattern, img_path)[0]

        # Pattern for image id: Idddddd
        image_id_pattern = r"I\d{2,}"
        image_id = re.findall(image_id_pattern, img_path)[0]

        return patient_id, image_id

    @staticmethod
    def get_all_nifti_files(dir=''):
        """
        Function for getting all the mgz files from a given folder
        :param dir: str, the folder to find all the mgz files
        :return: list with all the files
        """
        return [file for file in os.listdir(dir) if file.endswith(".nii.gz")]