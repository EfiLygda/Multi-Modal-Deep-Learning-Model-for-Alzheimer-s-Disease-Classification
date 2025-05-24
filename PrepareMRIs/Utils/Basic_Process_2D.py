import cv2
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma


def Resize(img_array, output_size=224):
    """
    Function for resizing an image to a chosen output size

    :param img_array: numpy.array, the image
    :param output_size: int, the width and height of the image
    :return: numpy.array, the resized image
    """
    # Calculate the aspect ratio of the image
    aspect_ratio = img_array.shape[1] / img_array.shape[0]

    # Calculate the new dimensions while maintaining the aspect ratio
    # The biggest dimension is always output_size
    if img_array.shape[0] > img_array.shape[1]:
        new_height = output_size
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = output_size
        new_height = int(new_width / aspect_ratio)

    # Resize the image
    resized_img = cv2.resize(img_array, (new_width, new_height))

    # All black image with output_size x output_size shape
    result = np.zeros([output_size, output_size])

    # Calculate the position where the smaller image should be placed
    # This is the center of the larger image minus half the size of the smaller image
    pos_x = (result.shape[1] - resized_img.shape[1]) // 2
    pos_y = (result.shape[0] - resized_img.shape[0]) // 2

    # Place the smaller image at the calculated position in the larger image
    result[pos_y:pos_y + resized_img.shape[0], pos_x:pos_x + resized_img.shape[1]] = resized_img

    return result


def HistogramEqualization(img_array):
    """
    Function for performing histogram equalization to greyscale images

    :param img_array: numpy.array, the original image
    :return: numpy.array, the histogram equalized image
    """

    # Normalize the image to the range [0, 255] and convert it to uint8
    gray_img = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(gray_img)

    return equalized_image


def Rotate180(img_array):
    """
    Function for rotating an image by 180 degrees
    :param img_array: numpy.array, the original image
    :return: numpy.array, the rotated image
    """
    return cv2.rotate(img_array, cv2.ROTATE_180)


def Rotate90(img_array):
    """
    Function for rotating an image by 90 degrees
    :param img_array: numpy.array, the original image
    :return: numpy.array, the rotated image
    """
    return cv2.rotate(img_array, cv2.ROTATE_90_COUNTERCLOCKWISE)


def Gray2RGB(img_array):
    """
    Function for converting a greyscale image to RGB
    :param img_array: numpy.array, the original image
    :return: numpy.array, the converted image
    """
    return np.repeat(np.expand_dims(img_array, axis=-1), 3, axis=-1)


def DenoiseSlice(img_array):
    """
    Function for denoising an image via Non-local Means
    :param img_array: numpy.array, the original image
    :return: numpy.array, the denoised image
    """

    # Convert grayscale image to RGB image
    img_array = Gray2RGB(img_array)

    # Estimate the noise standard deviation from the noisy image
    sigma_est = np.mean(estimate_sigma(img_array, channel_axis=-1))

    patch_kw = dict(patch_size=5,
                    patch_distance=6,
                    channel_axis=-1)

    # Fast algorithm
    denoised_img = denoise_nl_means(img_array, h=5 * sigma_est, fast_mode=True,
                                    **patch_kw)

    return denoised_img
