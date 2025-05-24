import os
import numpy as np
import matplotlib.pyplot as plt

import Utils.Basic_Process_2D as PR2D


class Process2DSlice:

    def __init__(self, out_dir):
        """
        Class for processing the 2D slices of an MRI

        :param out_dir: str, the output directory of the processed 2D slice
        """

        # Set the output directories of the 3 planes of view
        self.out_dirs = {'sagittal': os.path.join(out_dir, 'Processed_sagittal'),
                         'coronal': os.path.join(out_dir, 'Processed_coronal'),
                         'axial': os.path.join(out_dir, 'Processed_axial')}

        # In case the above directories do not exist then they are created
        for dir in self.out_dirs.values():
            if not os.path.exists(dir):
                os.makedirs(dir)

        # Save the names of the planes
        self.planes = self.out_dirs.keys()

    @staticmethod
    def ProcessSlice(
        slice,
        output_size=None,
        rotate=None,
        denoise=False
    ):
        """
        Function for applying basic processing methods to the 2D slice

        :param slice: np.array, the slice to be processed
        :param output_size: int, the width and height of the outputed slice
        :param rotate: float, 90 or 180, whether the slice to be rotated by 90 or 180 degrees
        :param denoise: bool, whether to denoise the image using non-local means
        :return: np.array, the processed 2D slice
        """

        # Perform histogram normalization, to correct brightness
        processed_slice = PR2D.HistogramEqualization(slice)

        # Resize the image as needed
        if output_size:
            processed_slice = PR2D.Resize(processed_slice, output_size)
        else:
            processed_slice = PR2D.Resize(processed_slice, 224)

        # Rotate the image as needed
        if rotate:
            if rotate == 180:
                processed_slice = PR2D.Rotate180(processed_slice)
            elif rotate == 90:
                processed_slice = PR2D.Rotate90(processed_slice)

        # Denoise the image if needed
        if denoise:
            processed_slice = PR2D.DenoiseSlice(processed_slice)

        return processed_slice

    @staticmethod
    def GetSliceByPlane(img_array, plane, index):
        """
        Function for fetching a slice from an MRI, by its plane and index in that plane

        :param img_array: np.array, the MRI image's data
        :param plane: str, the plane from which the slice to be extracted. Possible values 'axia', 'coronal',
         'sagittal', 'all'. In case 'all' is used then the index must be an iterable of 3 integers.
        :param index: int or iterable, the index of the slice or slices to be extracted
        :return: np.array (tuple of 3 np.arrays) with the slice(s)
        """
        if plane == 'axial':
            return img_array[:, :, index]
        elif plane == 'coronal':
            return img_array[:, index, :]
        elif plane == 'sagittal':
            return img_array[index, :, :]
        elif plane == 'all':
            x, y, z = index
            return img_array[x, :, :], img_array[:, y, :], img_array[:, :, z]

    @staticmethod
    def GetNotAllBlackSlicesIndices(img_array, plane='axial'):
        """
        Function for fetching the indices of the slices in the chosen plane that are not all black.

        :param img_array: np.array, the MRI image
        :param plane: str, the plane to fetch the indices from. Possible values are 'axial', 'sagittal', 'coronal'
        :return: tuple, with the indices
        """
        axis = {'axial': (0, 1), 'coronal': (0, 2), 'sagittal': (1, 2)}
        all_black_mask = np.all(img_array == 0, axis=axis[plane])
        slices_not_all_black = np.where(~all_black_mask)[0]

        return slices_not_all_black

    @staticmethod
    def GetOrthoSlice(img_array, plane='axial'):
        """
        Function for calculating the orthogonal slice in a chosen plane after removing all black slices

        :param img_array: np.array, the MRI data
        :param plane: str, the plane used for calculating the index. Possible values are 'axial', 'sagittal',
        'coronal', 'all'
        :return: int (list of 3 int), the true orthogonal index (indices)
        """

        # Possible planes of view
        planes = ['axial', 'coronal', 'sagittal']

        # Check if the orthogonal indices for all the planes must be calculated, or for only one
        if plane == 'all':

            # Setting the list with final orthogonal indices
            orthogonal_index = []

            for pl in planes:

                # For every plane fetch the not all-black indices
                slices_not_all_black = Process2DSlice.GetNotAllBlackSlicesIndices(img_array, pl)

                # Find the middle index between the above indices
                orthogonal_idx = slices_not_all_black[len(slices_not_all_black) // 2]

                # Append the current orthogonal index to the list consisting of all of them
                orthogonal_index.append(orthogonal_idx)

        elif plane in planes:

            # For the chosen plane fetch the not all-black indices
            slices_not_all_black = Process2DSlice.GetNotAllBlackSlicesIndices(img_array, plane)

            # Find the middle index between the above indices
            orthogonal_index = slices_not_all_black[len(slices_not_all_black)//2]

        return orthogonal_index

    @staticmethod
    def ExportSlice(img_array, output_name, output_dir,
                    index=90, n_slices=0, plane='axial',
                    **process_args):
        """
        Function for exporting a 2D slice from an MRI

        :param img_array: np.array, the MRI from which the slice must be extracted and saved
        :param output_name: str, the file name of the outputed image
        :param output_dir: str, the directory path of the outputed image
        :param index: int (or iterable), the index (indices) for fetching the outputed image
        :param n_slices: int, number of slices above and below the current index to also be outputed
        :param plane: str, the plane used for exracting slices. Possible values are 'axial', 'sagittal', 'coronal'
        :param process_args: dict, containing the processing configuration as defined in Process2DSlice.ProcessSlice
        """

        # Set the indices used for fetching the 2D slices using the original index and the n_slices above and below
        slice_indices = list(range(index - n_slices, index + n_slices + 1))

        # For each of these indices save the image
        for x in slice_indices:

            # Set the extracted image's file name and output directory
            file_name = f'{output_name}__{x}.png'
            file_path = os.path.join(output_dir, file_name)

            # Fetch the slice according to the chosen plane and current index
            slice = Process2DSlice.GetSliceByPlane(img_array, plane, x)

            # Process the slice as given
            slice = Process2DSlice.ProcessSlice(
                slice,
                output_size=process_args['output_size'],
                rotate=process_args['rotate'],
                denoise=process_args['denoise']
            )

            # Save the image as greyscale
            plt.imsave(file_path, slice/slice.max(), cmap='gray')

    def ExportSlicesByPlane(self, img_array, output_name, plane='all',
                            index=(90, 108, 90), n_slices=0,
                            output_size=None, rotate=None,
                            denoise=True
                            ):
        """
        Wrapper function for Process2DSlice.ExportSlice supporting extraction of slices from all planes

        :param img_array: np.array, the MRI from which the slice must be extracted and saved
        :param output_name: str, the file name of the outputed image
        :param plane: str, the plane used for exracting slices. Possible values are 'axial', 'sagittal', 'coronal', 'all'
        :param index: int (or iterable), the index (indices) for fetching the outputed image
        :param n_slices: int, number of slices above and below the current index to also be outputed
        :param output_size: int, the width and height of the outputed slices
        :param rotate: float (90 or 180), the degree of rotation of the image
        :param denoise: bool, whether to denoise the slice using Non-local means
        """

        if plane == 'all':

            # In case extraction for all planes is chosen then iterate over the plane output dirs parallel to the
            # given indices
            for idx, (plane, output_dir) in zip(index, self.out_dirs.items()):

                # Export using Process2DSlice.ExportSlice
                Process2DSlice.ExportSlice(
                    img_array=img_array,
                    output_name=output_name,
                    output_dir=output_dir,
                    index=idx,
                    n_slices=n_slices,
                    plane=plane,
                    output_size=output_size,
                    rotate=rotate,
                    denoise=denoise
                )

        else:
            # In case extraction for a specific plane is chosen then set the plane output directory
            output_dir = self.out_dirs[plane]

            # Export using Process2DSlice.ExportSlice
            Process2DSlice.ExportSlice(
                img_array=img_array,
                output_name=output_name,
                output_dir=output_dir,
                index=index,
                n_slices=n_slices,
                plane=plane,
                output_size=output_size,
                rotate=rotate,
                denoise=denoise
            )
